from __future__ import annotations

import json
import math
import os
import re
import statistics
import traceback
from difflib import SequenceMatcher
from datetime import datetime, timezone, timedelta
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from model_with_sae import ModelWithSAEModule
from neuronpedia import fetch_feature_json, parse_activation_corpus, parse_logits


def _normalize_token(token: str) -> str:
    tok = (token or "").strip()
    if not tok:
        return ""
    if tok.startswith("\u2581") or tok.startswith("\u0120"):
        tok = tok[1:]
    if tok.startswith("##"):
        tok = tok[2:]
    return tok.strip().lower()


def _dedupe_keep_order(items: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Directly change into json or extract json from text then parse it.
    """
    text = text.strip()
    if not text:
        return None
    try:
        loaded = json.loads(text)
        if isinstance(loaded, dict):
            return loaded
        return None
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        loaded = json.loads(match.group(0))
        if isinstance(loaded, dict):
            return loaded
    except Exception:
        return None
    return None


def _safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _safe_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(statistics.pstdev(values))


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _normalize_prompt_key(prompt: str) -> str:
    text = re.sub(r"\s+", " ", str(prompt or "").strip().lower())
    return text


def _description_similarity(a: str, b: str) -> float:
    a_norm = re.sub(r"\s+", " ", str(a or "").strip().lower())
    b_norm = re.sub(r"\s+", " ", str(b or "").strip().lower())
    if not a_norm or not b_norm:
        return 0.0
    return float(SequenceMatcher(None, a_norm, b_norm).ratio())


@dataclass
class NeuronpediaObservation:
    model_id: str
    source: str
    feature_id: int
    positive_logits: List[Tuple[str, float]]
    negative_logits: List[Tuple[str, float]]
    top_activating_examples: List[Dict[str, Any]]
    seed_tokens: List[str]


@dataclass
class Hypothesis:
    hypothesis_id: str
    description: str
    expected_logit_increase: List[str]
    expected_logit_decrease: List[str]
    confidence_prior: float = 0.5


@dataclass
class ExperimentPlan:
    activation_positive_prompts: List[str]
    activation_negative_prompts: List[str]
    boundary_prompts: List[str]
    causal_prompts: List[str]


@dataclass
class ActivationTokenEvidence:
    token: str
    activation: float
    token_index: int


@dataclass
class PromptActivationEvidence:
    prompt: str
    score: float
    nonzero_tokens: List[ActivationTokenEvidence]
    prompt_tokens: List[str] = field(default_factory=list)


@dataclass
class FailureEvent:
    type: str  # boundary_violation | negative_fp | positive_fn
    prompt: str
    score: float
    threshold: float
    margin: float


@dataclass
class InputEvidence:
    positive_scores: List[PromptActivationEvidence]
    negative_scores: List[PromptActivationEvidence]
    boundary_scores: List[PromptActivationEvidence]
    activation_threshold: float
    activation_threshold_ratio: float
    positive_hit_rate: float
    negative_reject_rate: float
    boundary_violation_rate: float
    separation: float
    failure_events: List[FailureEvent]
    counterexamples: List[str]
    score: float


@dataclass
class OutputEvidence:
    amplify_top_increase: List[Tuple[str, float, int]]
    amplify_top_decrease: List[Tuple[str, float, int]]
    suppress_top_increase: List[Tuple[str, float, int]]
    suppress_top_decrease: List[Tuple[str, float, int]]
    expected_increase_coverage: float
    expected_decrease_coverage: float
    sign_consistency: float
    missing_expected_tokens: List[str]
    score: float


@dataclass
class HypothesisRoundResult:
    hypothesis: Hypothesis
    plan: ExperimentPlan
    input_evidence: InputEvidence
    output_evidence: OutputEvidence
    total_score: float


@dataclass
class FinalExplanation:
    feature_id: int
    rounds: int
    best_hypothesis: Hypothesis
    input_side_explanation: str
    output_side_explanation: str
    combined_explanation: str
    boundaries: List[str]
    testable_predictions: List[str]
    round_history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ReasoningMemoryStore:
    def __init__(self, memory_file: str):
        self.memory_file = Path(memory_file)
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)

    def clear(self) -> None:
        if self.memory_file.exists():
            self.memory_file.unlink()

    def append(self, item: Dict[str, Any]) -> None:
        record = dict(item)
        china_tz = timezone(timedelta(hours=8))
        record.setdefault("timestamp", datetime.now(china_tz).isoformat() + "Z")
        with self.memory_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _load_recent(self, feature_id: int, limit: int = 12) -> List[Dict[str, Any]]:
        # aborted
        if not self.memory_file.exists():
            return []
        rows: List[Dict[str, Any]] = []
        with self.memory_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    continue
                if int(data.get("feature_id", -1)) != int(feature_id):
                    continue
                rows.append(data)
        return rows[-limit:]


class OpenAICompatibleReasoner:
    RETRYABLE_ERRORS = (
        "APIConnectionError",
        "Timeout",
        "RateLimitError",
        "ServiceUnavailableError",
    )

    def __init__(
        self,
        *,
        model_name: str = "zai-org/glm-4.7",
        enabled: bool = True,
        llm_call_log: str = "outputs/llm_calls.jsonl",
        prompt_log_dir: Optional[str] = None,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
    ):
        self.model_name = model_name
        self.enabled = bool(enabled)
        self._init_error: Optional[str] = None
        self.llm_call_log = Path(llm_call_log)
        self.llm_call_log.parent.mkdir(parents=True, exist_ok=True)
        self.prompt_log_dir = Path(prompt_log_dir) if prompt_log_dir else None
        if self.prompt_log_dir is not None:
            self.prompt_log_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._client = None
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

        if not self.enabled:
            return
        init_errors: List[str] = []
        try:
            from myagents.runtime import infer_provider
            from openai import OpenAI

            provider = infer_provider(self.model_name)
            api_key = os.getenv(provider.compat.api_key_env)
            if not api_key:
                raise RuntimeError(f"Missing env var: {provider.compat.api_key_env}")
            base_url = os.getenv(provider.compat.base_url_env) or provider.compat.default_base_url
            self._client = OpenAI(api_key=api_key, base_url=base_url)
        except Exception as exc:
            self._client = None
            init_errors.append(f"chat_client_init_failed({type(exc).__name__}: {exc})")

        try:
            from myagents.runtime import build_agents_sdk_model

            self._model = build_agents_sdk_model(model_name=self.model_name)
        except Exception as exc:
            self._model = None
            init_errors.append(f"runner_model_init_failed({type(exc).__name__}: {exc})")

        if self._client is None and self._model is None:
            self._init_error = "; ".join(init_errors) if init_errors else "unknown_init_failure"
            print(f"Failed to initialize model {self.model_name}: {self._init_error}")

    @property
    def available(self) -> bool:
        return self.enabled and (self._client is not None or self._model is not None)

    @staticmethod
    def _coerce_content_text(content: Any) -> str:
        """
        模型返回可能被chunk化，可能有多模态，只提取文字的
        不用看这个函数
        """
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: List[str] = []
            for item in content:
                if isinstance(item, str):
                    chunks.append(item)
                    continue
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
                        continue
                    output_text = item.get("output_text")
                    if isinstance(output_text, str):
                        chunks.append(output_text)
            return "".join(chunks)
        return str(content)

    def _should_retry(self, error_type: str) -> bool:
        return error_type in self.RETRYABLE_ERRORS

    def _calculate_delay(self, attempt: int) -> float:
        import random
        delay = self.base_delay * (2 ** (attempt - 1))
        jitter = delay * 0.1 * random.random()
        return min(delay + jitter, self.max_delay)

    def _append_call_log(self, payload: Dict[str, Any]) -> None:
        record = dict(payload)
        china_tz = timezone(timedelta(hours=8))
        record.setdefault("timestamp", datetime.now(china_tz).isoformat() + "Z")
        with self.llm_call_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, indent=2) + "\n")

    def _append_prompt_log(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        agent_name: str,
        call_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.prompt_log_dir is None:
            return

        md = dict(metadata or {})
        feature_id = md.get("feature_id")
        round_tag = md.get("round")
        if round_tag is None:
            round_tag = "unknown"

        feature_part = "unknown" if feature_id is None else str(int(feature_id))
        round_part = str(round_tag)
        prompt_file = self.prompt_log_dir / f"feature_{feature_part}_round_{round_part}.json"

        payload = {
            "feature_id": feature_id,
            "round": round_tag,
            "prompts": [],
        }
        # if prompt_file.exists():
        try:
            with prompt_file.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                payload = loaded
            if not isinstance(payload.get("prompts"), list):
                payload["prompts"] = []
        except Exception:
            payload = {
                "feature_id": feature_id,
                "round": round_tag,
                "prompts": [],
            }

        china_tz = timezone(timedelta(hours=8))
        payload["feature_id"] = feature_id
        payload["round"] = round_tag
        payload["prompts"].append(
            {
                "timestamp": datetime.now(china_tz).isoformat() + "Z",
                "agent_name": agent_name,
                "call_type": call_type,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "metadata": md,
            }
        )

        prompt_file.parent.mkdir(parents=True, exist_ok=True)
        with prompt_file.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def chat_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        agent_name: str = "reasoning_agent",
        call_type: str = "generic",
        metadata: Optional[Dict[str, Any]] = None,
        temperature: float = 0.0,
        max_tokens: int = 10000,
        max_retries: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        就当chat用，其实就是处理了异常和日志，json本身还是依赖prompt的要求
        """
        if not self.available:
            self._append_call_log(
                {
                    "status": "skipped",
                    "reason": self._init_error or "reasoner_disabled_or_unavailable",
                    "call_type": call_type,
                    "agent_name": agent_name,
                    "metadata": metadata or {},
                }
            )
            return None

        self._append_prompt_log(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            agent_name=agent_name,
            call_type=call_type,
            metadata=metadata,
        )

        retries = max_retries if max_retries is not None else self.max_retries
        last_error: Optional[Exception] = None

        for attempt in range(1, retries + 1):
            try:
                return self._execute_llm_call(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    agent_name=agent_name,
                    call_type=call_type,  # 
                    metadata=metadata,  # 
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                last_error = exc
                error_type = type(exc).__name__
                self._append_call_log(
                    {
                        "status": "error",
                        "call_type": call_type,
                        "agent_name": agent_name,
                        "metadata": metadata or {},
                        "error_type": error_type,
                        "error": str(exc),
                        "attempt": attempt,
                        "max_retries": retries,
                        "traceback": traceback.format_exc(),
                    }
                )
                if attempt < retries and self._should_retry(error_type):
                    delay = self._calculate_delay(attempt)
                    print(f"[Retry] {agent_name} failed (attempt {attempt}/{retries}): {error_type}. Retrying in {delay:.1f}s...")
                    import time
                    time.sleep(delay)
                else:
                    break

        print(f"[Error] {agent_name} failed after {retries} attempts: {last_error}")
        return None

    def _execute_llm_call(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        agent_name: str,
        call_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        temperature: float,
        max_tokens: int,
    ) -> Optional[Dict[str, Any]]:
        try:
            output_source = "direct_chat_completions"
            finish_reason: Optional[str] = None
            direct_max_tokens_used = int(max_tokens)
            final_output: Any = ""

            if self._client is not None:
                model_name_lc = (self.model_name or "").lower()
                extra_body = {"thinking": {"type": "disabled"}} if model_name_lc.startswith(("glm", "zai-org/glm")) else None
                # Note 开思考
                # 双重预算策略：先尝试原始预算，若因长度被截断则翻倍
                for budget in (int(max_tokens), min(int(max_tokens) * 2, 10000)):
                    direct_max_tokens_used = int(budget)
                    kwargs: Dict[str, Any] = {
                        "model": self.model_name,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": float(temperature),
                        "max_tokens": int(budget),
                    }
                    if extra_body is not None:
                        kwargs["extra_body"] = extra_body

                    response = self._client.chat.completions.create(**kwargs)
                    message = None
                    if getattr(response, "choices", None):
                        finish_reason = str(response.choices[0].finish_reason or "")
                        message = response.choices[0].message
                    if message is not None:
                        content_text = self._coerce_content_text(getattr(message, "content", None)).strip()
                        if content_text:
                            final_output = content_text
                            break
                    if finish_reason != "length":
                        break
                    else:
                        print(f"[Warning] {agent_name} truncated output with budget {budget}. Retrying with double budget...")

            if (not final_output) and self._model is not None:
                # 再尝试openai agents sdk
                from agents import Agent, ModelSettings, Runner

                output_source = "agents_runner"
                agent = Agent(
                    name=agent_name,
                    instructions=system_prompt,
                    model=self._model,
                    model_settings=ModelSettings(
                        temperature=float(temperature),
                        max_tokens=int(max_tokens),
                    ),
                )

                conversation_id = None
                if metadata and metadata.get("feature_id") is not None:
                    conversation_id = f"feature-{metadata['feature_id']}-{call_type}"

                result = Runner.run_sync(
                    starting_agent=agent,
                    input=user_prompt,
                    conversation_id=conversation_id,
                )
                final_output = result.final_output

            if isinstance(final_output, dict):
                self._append_call_log(
                    {
                        "status": "success",
                        "call_type": call_type,
                        "agent_name": agent_name,
                        "metadata": metadata or {},
                        "raw_output": final_output,
                        "parsed": True,
                        "output_source": output_source,
                        "finish_reason": finish_reason,
                        "max_tokens_used": direct_max_tokens_used if output_source == "direct_chat_completions" else int(max_tokens),
                    }
                )
                return final_output

            raw_text = str(final_output)
            if not raw_text.strip():
                self._append_call_log(
                    {
                        "status": "parse_failed",
                        "call_type": call_type,
                        "agent_name": agent_name,
                        "metadata": metadata or {},
                        "raw_output": raw_text,
                        "reason": "empty_model_output",
                        "output_source": output_source,
                        "finish_reason": finish_reason,
                        "max_tokens_used": direct_max_tokens_used if output_source == "direct_chat_completions" else int(max_tokens),
                    }
                )
                return None
            parsed = _extract_json(raw_text)
            if parsed is None:
                self._append_call_log(
                    {
                        "status": "parse_failed",
                        "call_type": call_type,
                        "agent_name": agent_name,
                        "metadata": metadata or {},
                        "raw_output": raw_text,
                        "reason": "json_parse_failed",
                        "output_source": output_source,
                        "finish_reason": finish_reason,
                        "max_tokens_used": direct_max_tokens_used if output_source == "direct_chat_completions" else int(max_tokens),
                    }
                )
                return None
            self._append_call_log(
                {
                    "status": "success",
                    "call_type": call_type,
                    "agent_name": agent_name,
                    "metadata": metadata or {},
                    "raw_output": raw_text,
                    "parsed": True,
                    "output_source": output_source,
                    "finish_reason": finish_reason,
                    "max_tokens_used": direct_max_tokens_used if output_source == "direct_chat_completions" else int(max_tokens),
                }
            )
            return parsed
        except Exception as exc:
            raise exc


class AgentBrain:
    """
    Multi-agent reasoning roles over one shared LLM backend:
    - Hypothesis agent
    - Planner agent
    - Critic agent
    - Synthesizer agent
    """

    def __init__(self, reasoner: OpenAICompatibleReasoner):
        self.reasoner = reasoner

    @staticmethod
    def _format_memory_context(memory_context: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        if not memory_context:
            return []
        compact: List[Dict[str, Any]] = []
        for row in memory_context[-8:]:
            raw_failure_events = row.get("failure_events", [])
            failure_events: List[Dict[str, Any]] = []
            if isinstance(raw_failure_events, list):
                for event in raw_failure_events:
                    if not isinstance(event, dict):
                        continue
                    prompt = str(event.get("prompt", "")).strip()
                    if not prompt:
                        continue
                    try:
                        margin = float(event.get("margin", 0.0))
                    except Exception:
                        margin = 0.0
                    try:
                        score = float(event.get("score", 0.0))
                    except Exception:
                        score = 0.0
                    try:
                        threshold = float(event.get("threshold", 0.0))
                    except Exception:
                        threshold = 0.0
                    failure_events.append(
                        {
                            "type": str(event.get("type", "")),
                            "prompt": prompt,
                            "score": score,
                            "threshold": threshold,
                            "margin": margin,
                        }
                    )
            failure_events.sort(key=lambda x: float(x.get("margin", 0.0)), reverse=True)
            compact.append(
                {
                    # "timestamp": row.get("timestamp"),
                    "round": row.get("round"),
                    "hypothesis_id": row.get("hypothesis_id"),
                    "hypothesis_summary": row.get("hypothesis_summary") or row.get("notes"),
                    "counterexamples": row.get("counterexamples", []),
                    "failure_events": failure_events[:6],
                    "repeated_boundary_failures": row.get("repeated_boundary_failures", []),
                    "missing_expected_tokens": row.get("missing_expected_tokens", [])[:4],
                    "score_breakdown": row.get("score_breakdown", {}),
                }
            )
        return compact

    @staticmethod
    def _has_repeated_boundary_failures(memory_context: Optional[List[Dict[str, Any]]]) -> bool:
        if not memory_context:
            return False
        for row in memory_context[-8:]:
            repeated = row.get("repeated_boundary_failures", [])
            if isinstance(repeated, list) and repeated:
                return True
        return False

    @staticmethod
    def _needs_forced_critic_retry(
        parsed_hypotheses: Sequence[Hypothesis],
        previous_descriptions: Sequence[str],
        memory_context: Optional[List[Dict[str, Any]]],
    ) -> bool:
        if not parsed_hypotheses or not previous_descriptions:
            return False
        if not AgentBrain._has_repeated_boundary_failures(memory_context):
            return False
        top_new = parsed_hypotheses[0].description
        if not top_new:
            return False
        max_similarity = max(
            _description_similarity(top_new, old_desc) for old_desc in previous_descriptions if old_desc
        )
        return max_similarity >= 0.92

    @staticmethod
    def _format_observation_context(observation: NeuronpediaObservation) -> Dict[str, Any]:
        pos = [{"token": t, "logit": v} for t, v in observation.positive_logits[:20]]
        neg = [{"token": t, "logit": v} for t, v in observation.negative_logits[:20]]
        acts: List[Dict[str, Any]] = []
        for ex in observation.top_activating_examples[:8]:
            acts.append(
                {
                    "text": ex.get("text", ""),
                    # "marked_text": ex.get("marked_text", ""),
                    "top_tokens": ex.get("top_tokens", [])[:6],
                }
            )
        return {
            "model_id": observation.model_id,
            "source": observation.source,
            "feature_id": observation.feature_id,
            "positive_logits_top": pos,
            "negative_logits_top": neg,
            "top_activating_examples": acts,
            "seed_tokens": observation.seed_tokens[:40],
        }

    @staticmethod
    def _extract_hypothesis_rows(llm_json: Optional[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        if not isinstance(llm_json, dict):
            return None
        hypotheses = llm_json.get("hypotheses")
        if isinstance(hypotheses, list):
            return hypotheses
        schema_obj = llm_json.get("schema")
        if isinstance(schema_obj, dict) and isinstance(schema_obj.get("hypotheses"), list):
            return schema_obj["hypotheses"]
        return None

    def propose_hypotheses(
        self,
        observation: NeuronpediaObservation,
        num_hypotheses: int,
    ) -> List[Hypothesis]:
        print(f"In proposing hypotheses, observation: {observation}")
        system_prompt = (
            "You are the Hypothesis Agent for SAE mechanistic interpretability.\n"
            "Goal: generate precise, falsifiable input-side hypotheses and output-side hypotheses.\n"
            "Rules:\n"
            "1) Use only provided evidence; do not invent external facts.\n"
            "2) Input-side hypothesis must describe explicit activation patterns.\n"
            "3) Output_side hypothesis must predict both logit increases and decreases under amplification.\n"
            "4) Both sides hypothesis must be precise and clear.\n"
            "5) Prefer specific patterns over vague themes.\n"
            "Return strict JSON only."
        )
        observation_context = json.dumps(
            self._format_observation_context(observation),
            ensure_ascii=False,
            indent=2,
        )
        schema_prompt = json.dumps(
            {
                "hypotheses": [
                    {
                        "hypothesis_id": "h1",
                        "input-side feature description": "a description of the input-side feature",
                        "output-side feature description": "a description of the output-side feature",
                        "expected_logit_increase": ["token"],
                        "expected_logit_decrease": ["token"],
                        "confidence_prior": "a score",
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        )
        user_prompt = (
            "Task: Propose candidate feature hypotheses with explicit testability.\n"
            "Constraints:\n"
            "- Input-side hypothesis must clearly describe what contexts activate this feature and where it should fail.\n"
            "- Output-side hypothesis must predict expected logit increase/decrease tokens under feature amplification.\n"
            "- Description should be precise, not generic. No more than 40 words.\n"
            "- A good explanation needs to enable people to distinguish what kind of input can activate the feature, and what kind of token's logits will increase or decrease when intervening on the feature.\n"
            "Evidence quality guidance:\n"
            "- Activation side: Use top activating examples and seed contexts to infer triggering patterns.\n"
            "- Causal side: Use positive/negative logits as weak priors for expected intervention effects.\n"
            f"Number of hypotheses: {int(num_hypotheses)}\n\n"
            "Observation:\n"
            f"{observation_context}\n\n"
            "Output schema (JSON):\n"
            f"{schema_prompt}\n\n"
            "Field descriptions:\n"
            "- hypothesis_id: identifier (e.g., h1, h2)\n"
            "- input-side/output-side feature description: observation-based description, each no more than 40 words each.\n"
            "- expected_logit_increase: tokens expected to increase under amplification\n"
            "- expected_logit_decrease: tokens expected to decrease under amplification\n"
            "- confidence_prior: confidence score between 0 and 1"
        )

        llm_json = self.reasoner.chat_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            agent_name="hypothesis_agent",
            call_type="hypothesis_generation",
            metadata={"feature_id": observation.feature_id, "round": 1},
        )
        rows = self._extract_hypothesis_rows(llm_json)
        if rows:
            parsed = self._parse_hypotheses(rows)
            if parsed:
                return parsed[:num_hypotheses]

        raise RuntimeError(
            f"Hypothesis generation failed for feature {observation.feature_id}: missing or invalid LLM JSON."
        )

    def _parse_hypotheses(self, rows: Sequence[Dict[str, Any]]) -> List[Hypothesis]:
        def _as_clean_list(value: Any) -> List[str]:
            if isinstance(value, str):
                raw_items = re.split(r"[,;\n|]", value)
                return [x.strip() for x in raw_items if x.strip()]
            if isinstance(value, (list, tuple)):
                return [str(x).strip() for x in value if str(x).strip()]
            return []

        out: List[Hypothesis] = []
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            hypothesis_id = str(row.get("hypothesis_id") or f"h{idx+1}")
            input_desc = str(
                row.get("input-side feature description")
                or row.get("input_side_feature_description")
                or row.get("input_side_description")
                or ""
            ).strip()
            output_desc = str(
                row.get("output-side feature description")
                or row.get("output_side_feature_description")
                or row.get("output_side_description")
                or ""
            ).strip()

            description = str(
                row.get("description")
                or row.get("feature_description")
                or row.get("combined_description")
                or ""
            ).strip()
            if not description:
                if input_desc and output_desc:
                    description = f"Input-side: {input_desc} Output-side: {output_desc}"
                elif input_desc:
                    description = f"Input-side: {input_desc}"
                elif output_desc:
                    description = f"Output-side: {output_desc}"

            up = _as_clean_list(
                row.get("expected_logit_increase", row.get("expected_logit_increases", []))
            )
            down = _as_clean_list(
                row.get("expected_logit_decrease", row.get("expected_logit_decreases", []))
            )
            try:
                prior = float(row.get("confidence_prior", 0.5))
            except Exception:
                prior = 0.5
            if not description:
                description = f"The feature encodes a specific pattern (hypothesis {idx+1})."
            out.append(
                Hypothesis(
                    hypothesis_id=hypothesis_id,
                    description=description,
                    expected_logit_increase=up[:10],
                    expected_logit_decrease=down[:10],
                    confidence_prior=_clip01(prior),
                )
            )
        return out

    def plan_experiments(
        self,
        hypothesis: Hypothesis,
        observation: NeuronpediaObservation,
        prompts_per_split: int,
        memory_context: Optional[List[Dict[str, Any]]] = None,
        round_idx: int = 1,
    ) -> ExperimentPlan:
        system_prompt = (
            "You are the Experiment Planner Agent.\n"
            "Design experiments that can falsify test hypotheses.\n"
            "You must create four prompt sets:\n"
            "- activation_positive_prompts: should trigger the feature. "
            "Construct examples based on the input-side hypothesis description.\n"
            # Note 这个随机性有点强，需要更多的性质约束
            "- activation_negative_prompts: semantically nearby but should not trigger. "
            "Construct examples based on the input-side hypothesis description.\n"
            "- boundary_prompts: hard-negative adversarial/counterfactual prompts expected NOT to trigger. "
            "Construct examples based on the input-side hypothesis description.\n"
            "- causal_prompts: contexts suitable for measuring logit shifts after feature intervention. "
            "Construct examples based on the output-side hypothesis description.\n"
            "You must use memory failure patterns to avoid repeating uninformative boundary prompts.\n"
            "Return strict JSON only."
        )
        hypothesis_context = json.dumps(asdict(hypothesis), ensure_ascii=False, indent=2)
        memory_context_json = json.dumps(
            self._format_memory_context(memory_context),
            ensure_ascii=False,
            indent=2,
        )
        schema_prompt = json.dumps(
            {
                "activation_positive_prompts": ["..."],
                "activation_negative_prompts": ["..."],
                "boundary_prompts": ["..."],
                "causal_prompts": ["..."],
            },
            ensure_ascii=False,
            indent=2,
        )
        user_prompt = (
            "Task: Plan prompt-level experiments for this hypothesis.\n"
            f"Round: {int(round_idx)}\n"
            f"Prompts per split: {int(prompts_per_split)}\n\n"
            "Hypothesis:\n"
            f"{hypothesis_context}\n\n"
            "Memory from past runs:\n"
            f"{memory_context_json}\n\n"
            "Design requirements:\n"
            "- activation_positive: Should include hypothesized trigger semantics in diverse phrasing.\n"
            "- activation_negative: Should be hard negatives (close topic but missing key trigger).\n"
            "- boundary: Must be hard negatives expected not to activate.\n"
            "- boundary: Must include at least one near-counterexample and one confounder.\n"
            "- boundary: If memory shows repeated boundary failures, create new prompts that target those failure patterns without copying old strings.\n"
            "- boundary: Do not reuse exact prompts listed in repeated_boundary_failures.\n"
            "- causal: Should place a predictive next-token position where style/semantics can shift.\n\n"
            "Output schema (JSON):\n"
            f"{schema_prompt}"
        )
        llm_json = self.reasoner.chat_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            agent_name="planner_agent",
            call_type="experiment_planning",
            metadata={"feature_id": observation.feature_id, "round": round_idx, "hypothesis_id": hypothesis.hypothesis_id},
        )
        if llm_json:
            plan = self._parse_plan(llm_json, prompts_per_split)
            if plan:
                return plan
        raise RuntimeError(
            f"Experiment planning failed for hypothesis {hypothesis.hypothesis_id}: missing or invalid LLM JSON."
        )

    def _parse_plan(self, row: Dict[str, Any], prompts_per_split: int) -> Optional[ExperimentPlan]:
        payload = row
        schema_obj = row.get("schema")
        if isinstance(schema_obj, dict):
            payload = schema_obj

        keys = [
            "activation_positive_prompts",
            "activation_negative_prompts",
            "boundary_prompts",
            "causal_prompts",
        ]
        cols: Dict[str, List[str]] = {}
        for key in keys:
            vals = payload.get(key, [])
            if not isinstance(vals, list):
                vals = []
            cleaned = [str(v).strip() for v in vals if str(v).strip()]
            cols[key] = _dedupe_keep_order(cleaned)[:prompts_per_split]  # 最多有这么多个prompts
        if not cols["activation_positive_prompts"] or not cols["causal_prompts"]:
            return None
        if not cols["activation_negative_prompts"]:
            cols["activation_negative_prompts"] = ["The sentence discusses random weather statistics and nothing else."]
        if not cols["boundary_prompts"]:
            cols["boundary_prompts"] = cols["activation_negative_prompts"][:]

        return ExperimentPlan(
            activation_positive_prompts=cols["activation_positive_prompts"],
            activation_negative_prompts=cols["activation_negative_prompts"],
            boundary_prompts=cols["boundary_prompts"],
            causal_prompts=cols["causal_prompts"],
        )

    def critic_refine(
        self,
        round_results: List[HypothesisRoundResult],
        keep_top_k: int,
        memory_context: Optional[List[Dict[str, Any]]] = None,
        feature_id: Optional[int] = None,
        round_idx: int = 1,
    ) -> List[Hypothesis]:
        ordered = sorted(round_results, key=lambda r: r.total_score, reverse=True)
        if not ordered:
            return []

        base_system_prompt = (
            "You are the Critic Agent for iterative hypothesis refinement.\n"
            "Use measured evidence to reject weak assumptions and produce stronger, more falsifiable hypotheses.\n"
            "Prioritize:\n"
            "1) fixing boundary failures\n"
            "2) fixing incorrect output-token direction predictions\n"
            "3) preserving only evidence-supported trigger patterns\n"
            "Hard constraints:\n"
            "- You must first identify the top 2 failure points from evidence.\n"
            "- You must map each top failure to an explicit hypothesis change.\n"
            "- If repeated boundary failures exist, do not return near-identical descriptions.\n"
            "Return strict JSON only."
        )

        evidence_payload = []
        for row in ordered:
            failure_events = sorted(
                [asdict(evt) for evt in row.input_evidence.failure_events],
                key=lambda x: float(x.get("margin", 0.0)),
                reverse=True,
            )
            evidence_payload.append(
                {
                    "hypothesis": asdict(row.hypothesis),
                    "score": row.total_score,
                    "input_summary": {
                        "score": row.input_evidence.score,
                        "separation": row.input_evidence.separation,
                        "failure_events": failure_events[:6],
                        "counterexamples": row.input_evidence.counterexamples,
                        "boundary_violation_rate": row.input_evidence.boundary_violation_rate,
                    },
                    "output_summary": {
                        "score": row.output_evidence.score,
                        "missing_expected_tokens": row.output_evidence.missing_expected_tokens,
                        "sign_consistency": row.output_evidence.sign_consistency,
                        "amplify_top_increase": row.output_evidence.amplify_top_increase[:6],
                        "amplify_top_decrease": row.output_evidence.amplify_top_decrease[:6],
                    },
                }
            )

        evidence_context = json.dumps(evidence_payload, ensure_ascii=False, indent=2)
        memory_context_json = json.dumps(
            self._format_memory_context(memory_context),
            ensure_ascii=False,
            indent=2,
        )
        schema_prompt = json.dumps(
            {
                "hypotheses": [
                    {
                        "hypothesis_id": "h1r",
                        "description": "refined description",
                        "expected_logit_increase": ["..."],
                        "expected_logit_decrease": ["..."],
                        "confidence_prior": 0.5,
                        "revision_notes": "failure->change mapping used in this refinement",
                        "resolved_failures": ["..."],
                        "unresolved_failures": ["..."],
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        )
        previous_descriptions = [row.hypothesis.description for row in ordered[:keep_top_k]]

        def _build_prompts(force_rewrite: bool) -> Tuple[str, str]:
            rewrite_clause = ""
            if force_rewrite:
                rewrite_clause = (
                    "STRICT REWRITE MODE:\n"
                    "- Repeated boundary failures persist.\n"
                    "- The previous refinement was too similar to old descriptions.\n"
                    "- You must materially rewrite the input-side trigger statement.\n"
                    "- Avoid lexical reuse of previous core phrasing.\n\n"
                )
            system_prompt = base_system_prompt
            if force_rewrite:
                system_prompt += "\nDo not keep prior wording if repeated boundary failures remain."
            user_prompt = (
                "Task: Refine hypotheses and keep only the strongest candidates.\n"
                f"Round: {int(round_idx)}\n"
                f"keep_top_k: {int(keep_top_k)}\n\n"
                "Description of evidence fields:\n"
                "- score: The score of the hypothesis, based on the input and output evidence.\n"
                "- separation: The separation between activation and non-activation examples.\n"
                "- failure_events: structured failures with type/prompt/score/threshold/margin.\n"
                "- counterexamples: textual summaries of main failures.\n"
                "- boundary_violation_rate: boundary failure rate in the input evidence.\n"
                "- missing_expected_tokens: expected but missing output tokens.\n"
                "- sign_consistency: consistency of logit direction.\n"
                "- amplify_top_increase / amplify_top_decrease: observed intervention effects.\n\n"
                f"{rewrite_clause}"
                "Mandatory reasoning steps:\n"
                "1) List top 2 failures from evidence (or fewer if fewer exist).\n"
                "2) Map each failure to a concrete hypothesis edit.\n"
                "3) Produce refined hypotheses consistent with that mapping.\n\n"
                "Evidence:\n"
                f"{evidence_context}\n\n"
                "Memory from past runs:\n"
                f"{memory_context_json}\n\n"
                "Refinement policy:\n"
                "- drop_if: high boundary violation and low output coverage; repeated counterexample failures across rounds.\n"
                "- keep_if: good activation separation and stable sign-consistent causal effects.\n\n"
                "Output schema (JSON):\n"
                f"{schema_prompt}\n\n"
                "Field descriptions:\n"
                "- hypothesis_id: identifier (e.g., h1r as h1 refined)\n"
                "- description: refined combined description for input/output behavior.\n"
                "- expected_logit_increase: tokens expected to increase under amplification\n"
                "- expected_logit_decrease: tokens expected to decrease under amplification\n"
                "- confidence_prior: confidence score between 0 and 1\n"
                "- revision_notes: concise failure->change mapping used for this refinement (required)\n"
                "- resolved_failures: failures addressed by this revision (optional)\n"
                "- unresolved_failures: failures intentionally left unresolved (optional)"
            )
            return system_prompt, user_prompt

        def _run_critic(force_rewrite: bool) -> Optional[List[Hypothesis]]:
            system_prompt, user_prompt = _build_prompts(force_rewrite=force_rewrite)
            llm_json = self.reasoner.chat_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                agent_name="critic_agent",
                call_type="hypothesis_refinement",
                metadata={
                    "feature_id": feature_id,
                    "round": round_idx,
                    "forced_rewrite": bool(force_rewrite),
                },
            )
            rows = self._extract_hypothesis_rows(llm_json)
            if not rows:
                return None
            parsed = self._parse_hypotheses(rows)
            if not parsed:
                return None
            return parsed[:keep_top_k]

        parsed = _run_critic(force_rewrite=False)
        if parsed and self._needs_forced_critic_retry(parsed, previous_descriptions, memory_context):
            retry_parsed = _run_critic(force_rewrite=True)
            if retry_parsed:
                return retry_parsed
        if parsed:
            return parsed

        fallback: List[Hypothesis] = []
        for idx, row in enumerate(ordered[:keep_top_k]):
            hyp = row.hypothesis
            top_inc = [tok for tok, _, _ in row.output_evidence.amplify_top_increase[:6]]
            top_dec = [tok for tok, _, _ in row.output_evidence.amplify_top_decrease[:6]]
            refined = Hypothesis(
                hypothesis_id=f"{hyp.hypothesis_id}_r{idx+1}",
                description=hyp.description,
                expected_logit_increase=_dedupe_keep_order(
                    hyp.expected_logit_increase[:4] + top_inc[:4]
                )[:8],
                expected_logit_decrease=_dedupe_keep_order(
                    hyp.expected_logit_decrease[:4] + top_dec[:4]
                )[:8],
                confidence_prior=_clip01(0.5 * hyp.confidence_prior + 0.5 * row.total_score),
            )
            fallback.append(refined)
        return fallback
    def synthesize_final(
        self,
        observation: NeuronpediaObservation,
        best_row: HypothesisRoundResult,
        memory_context: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        system_prompt = (
            "You are the Synthesizer Agent for mechanistic interpretability.\n"
            "Produce precise and testable feature explanations grounded in measured evidence.\n"
            "Requirements:\n"
            "- Input-side explanation: when feature activates.\n"
            "- Output-side explanation: which token logits increase/decrease under amplification.\n"
            "- Combined explanation: only if causally coherent.\n"
            "Return strict JSON only."
        )
        observation_context = json.dumps(
            self._format_observation_context(observation),
            ensure_ascii=False,
            indent=2,
        )
        best_result_context = json.dumps(
            {
                "hypothesis": asdict(best_row.hypothesis),
                "input_evidence": asdict(best_row.input_evidence),
                "output_evidence": asdict(best_row.output_evidence),
            },
            ensure_ascii=False,
            indent=2,
        )
        memory_context_json = json.dumps(
            self._format_memory_context(memory_context),
            ensure_ascii=False,
            indent=2,
        )
        schema_prompt = json.dumps(
            {
                "input_side_explanation": "string",
                "output_side_explanation": "string",
                "combined_explanation": "string",
                "boundaries": ["string"],
                "testable_predictions": ["string"],
            },
            ensure_ascii=False,
            indent=2,
        )
        user_prompt = (
            "Task: Produce input-side explanation, output-side explanation, and fused explanation.\n\n"
            "Observation:\n"
            f"{observation_context}\n\n"
            "Best result:\n"
            f"{best_result_context}\n\n"
            "Memory from past runs:\n"
            f"{memory_context_json}\n\n"
            "Output schema (JSON):\n"
            f"{schema_prompt}"
        )
        llm_json = self.reasoner.chat_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            agent_name="synthesizer_agent",
            call_type="final_synthesis",
            metadata={"feature_id": observation.feature_id, "round": "final"},
        )
        if llm_json:
            required = [
                "input_side_explanation",
                "output_side_explanation",
                "combined_explanation",
                "boundaries",
                "testable_predictions",
            ]
            if all(k in llm_json for k in required):
                return llm_json

        hyp = best_row.hypothesis
        amp_up = [tok for tok, _, _ in best_row.output_evidence.amplify_top_increase[:5]]
        amp_down = [tok for tok, _, _ in best_row.output_evidence.amplify_top_decrease[:5]]
        input_exp = (
            f"Input-side evidence supports this hypothesis: {hyp.description}"
        )
        output_exp = (
            f"Amplifying the feature raises logits such as {', '.join(amp_up[:4])} "
            f"and lowers logits such as {', '.join(amp_down[:4])}."
        )
        combined_exp = (
            f"{hyp.description} Causally, higher activation shifts next-token preference toward "
            f"{', '.join(amp_up[:3])} and away from {', '.join(amp_down[:3])}."
        )
        return {
            "input_side_explanation": input_exp,
            "output_side_explanation": output_exp,
            "combined_explanation": combined_exp,
            "boundaries": best_row.input_evidence.counterexamples[:4],
            "testable_predictions": [
                "Prompts matching the hypothesis description should exceed activation threshold.",
                "Feature amplification should increase probability of listed positive tokens.",
                "Feature suppression should partially reverse the same token shifts.",
            ],
        }


class FeatureExperimentRunner:
    def __init__(
        self,
        model_with_sae: ModelWithSAEModule,
        *,
        feature_index: int,
        intervention_value: float = 8.0,
        top_k_output_tokens: int = 12,
        activation_threshold_ratio: float = 0.5,
    ):
        self.module = model_with_sae
        self.feature_index = int(feature_index)
        self.intervention_value = float(intervention_value)
        self.top_k_output_tokens = int(top_k_output_tokens)
        self.activation_threshold_ratio = float(activation_threshold_ratio)
        if not (0.0 <= self.activation_threshold_ratio <= 1.0):
            raise ValueError("activation_threshold_ratio must be in [0.0, 1.0].")

    @staticmethod
    def _extract_nonzero_tokens(trace: Dict[str, Any]) -> List[ActivationTokenEvidence]:
        tokens = trace.get("tokens", [])
        per_token = trace.get("per_token_activation", [])
        out: List[ActivationTokenEvidence] = []
        limit = min(len(tokens), len(per_token))
        for idx in range(limit):
            act = float(per_token[idx])
            if act == 0.0:
                continue
            out.append(
                ActivationTokenEvidence(
                    token=str(tokens[idx]),
                    activation=act,
                    token_index=int(idx),
                )
            )
        return out

    def _activation_scores(
        self,
        prompts: Sequence[str],
        *,
        include_prompt_tokens: bool = True,
    ) -> List[PromptActivationEvidence]:
        out: List[PromptActivationEvidence] = []
        for prompt in prompts:
            trace = self.module.get_activation_trace(prompt)
            score = float(trace.get("summary_activation", 0.0))
            tokens = [str(tok) for tok in trace.get("tokens", [])]
            out.append(
                PromptActivationEvidence(
                    prompt=prompt,
                    score=score,
                    nonzero_tokens=self._extract_nonzero_tokens(trace),
                    prompt_tokens=tokens if include_prompt_tokens else [],
                )
            )
        return out

    def evaluate_input_side(
        self,
        plan: ExperimentPlan,
    ) -> InputEvidence:
        pos_scores = self._activation_scores(
            plan.activation_positive_prompts,
            include_prompt_tokens=True,
        )
        neg_scores = self._activation_scores(plan.activation_negative_prompts)
        boundary_scores = self._activation_scores(plan.boundary_prompts)

        pos_vals = [row.score for row in pos_scores]
        neg_vals = [row.score for row in neg_scores]
        boundary_vals = [row.score for row in boundary_scores]

        all_vals = pos_vals + neg_vals + boundary_vals
        max_activation = max(all_vals) if all_vals else 0.0
        threshold = max_activation * self.activation_threshold_ratio
        pos_hit_rate = _safe_mean([1.0 if v > threshold else 0.0 for v in pos_vals])
        neg_reject_rate = _safe_mean([1.0 if v <= threshold else 0.0 for v in neg_vals])
        boundary_violation_rate = _safe_mean([1.0 if v > threshold else 0.0 for v in boundary_vals])
        separation = _safe_mean(pos_vals) - _safe_mean(neg_vals)

        failure_events: List[FailureEvent] = []
        for row in sorted(neg_scores, key=lambda x: x.score, reverse=True):
            if row.score > threshold:
                failure_events.append(
                    FailureEvent(
                        type="negative_fp",
                        prompt=row.prompt,
                        score=float(row.score),
                        threshold=float(threshold),
                        margin=float(row.score - threshold),
                    )
                )
        for row in sorted(pos_scores, key=lambda x: x.score):
            if row.score <= threshold:
                failure_events.append(
                    FailureEvent(
                        type="positive_fn",
                        prompt=row.prompt,
                        score=float(row.score),
                        threshold=float(threshold),
                        margin=float(threshold - row.score),
                    )
                )
        for row in sorted(boundary_scores, key=lambda x: x.score, reverse=True):
            if row.score > threshold:
                failure_events.append(
                    FailureEvent(
                        type="boundary_violation",
                        prompt=row.prompt,
                        score=float(row.score),
                        threshold=float(threshold),
                        margin=float(row.score - threshold),
                    )
                )

        failure_events = sorted(failure_events, key=lambda x: x.margin, reverse=True)
        counterexamples: List[str] = []
        for event in failure_events[:6]:
            if event.type == "boundary_violation":
                counterexamples.append(
                    f"Boundary violation ({event.score:.4f} > {event.threshold:.4f}): {event.prompt}"
                )
            elif event.type == "negative_fp":
                counterexamples.append(
                    f"Negative prompt activated strongly ({event.score:.4f} > {event.threshold:.4f}): {event.prompt}"
                )
            elif event.type == "positive_fn":
                counterexamples.append(
                    f"Positive prompt under-activated ({event.score:.4f} <= {event.threshold:.4f}): {event.prompt}"
                )

        separation_term = 1.0 / (1.0 + math.exp(-2.0 * separation))
        score = (
            0.30 * separation_term
            + 0.25 * pos_hit_rate
            + 0.25 * neg_reject_rate
            + 0.20 * (1.0 - boundary_violation_rate)
        )

        return InputEvidence(
            positive_scores=pos_scores,
            negative_scores=neg_scores,
            boundary_scores=boundary_scores,
            activation_threshold=float(threshold),
            activation_threshold_ratio=float(self.activation_threshold_ratio),
            positive_hit_rate=float(pos_hit_rate),
            negative_reject_rate=float(neg_reject_rate),
            boundary_violation_rate=float(boundary_violation_rate),
            separation=float(separation),
            failure_events=failure_events[:8],
            counterexamples=counterexamples[:6],
            score=_clip01(score),
        )

    def evaluate_output_side(
        self,
        plan: ExperimentPlan,
        hypothesis: Hypothesis,
    ) -> OutputEvidence:
        if self.module.tokenizer is None:
            raise RuntimeError("Tokenizer is required for output-side intervention.")

        encoded = self.module.tokenizer(
            plan.causal_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        input_ids = encoded["input_ids"].to(self.module.device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.module.device)

        try:
            token_change = self.module.token_change_from_tokens(
                input_ids=input_ids,
                feature_index=self.feature_index,
                intervention_value=self.intervention_value,
                top_k=self.top_k_output_tokens,
                attention_mask=attention_mask,
                skip_special_tokens=True,
            )
        except Exception:
            missing = _dedupe_keep_order(
                [_normalize_token(t) for t in hypothesis.expected_logit_increase]
                + [_normalize_token(t) for t in hypothesis.expected_logit_decrease]
            )
            return OutputEvidence(
                amplify_top_increase=[],
                amplify_top_decrease=[],
                suppress_top_increase=[],
                suppress_top_decrease=[],
                expected_increase_coverage=0.0,
                expected_decrease_coverage=0.0,
                sign_consistency=0.0,
                missing_expected_tokens=[x for x in missing if x][:8],
                score=0.0,
            )

        amp_inc = token_change["amplify_top_increase"]
        amp_dec = token_change["amplify_top_decrease"]
        sup_inc = token_change["suppress_top_increase"]
        sup_dec = token_change["suppress_top_decrease"]

        observed_inc_set = {_normalize_token(tok) for tok, _, _ in amp_inc}
        observed_dec_set = {_normalize_token(tok) for tok, _, _ in amp_dec}
        sup_inc_set = {_normalize_token(tok) for tok, _, _ in sup_inc}
        sup_dec_set = {_normalize_token(tok) for tok, _, _ in sup_dec}

        expected_inc = {_normalize_token(t) for t in hypothesis.expected_logit_increase if _normalize_token(t)}
        expected_dec = {_normalize_token(t) for t in hypothesis.expected_logit_decrease if _normalize_token(t)}

        inc_cov = 0.0
        if expected_inc:
            inc_cov = len(expected_inc.intersection(observed_inc_set)) / len(expected_inc)
        dec_cov = 0.0
        if expected_dec:
            dec_cov = len(expected_dec.intersection(observed_dec_set)) / len(expected_dec)

        pos_consistency = len(observed_inc_set.intersection(sup_dec_set))
        neg_consistency = len(observed_dec_set.intersection(sup_inc_set))
        denom = max(1, len(observed_inc_set) + len(observed_dec_set))
        sign_consistency = (pos_consistency + neg_consistency) / denom

        missing_expected = sorted(list(expected_inc.difference(observed_inc_set)))[:4] + sorted(
            list(expected_dec.difference(observed_dec_set))
        )[:4]
        missing_expected = _dedupe_keep_order(missing_expected)

        effect_strength = _safe_mean([abs(v) for _, v, _ in (amp_inc[:5] + amp_dec[:5])])
        effect_term = _clip01(effect_strength / 2.5)
        score = 0.35 * inc_cov + 0.35 * dec_cov + 0.2 * sign_consistency + 0.1 * effect_term

        return OutputEvidence(
            amplify_top_increase=amp_inc,
            amplify_top_decrease=amp_dec,
            suppress_top_increase=sup_inc,
            suppress_top_decrease=sup_dec,
            expected_increase_coverage=float(inc_cov),
            expected_decrease_coverage=float(dec_cov),
            sign_consistency=float(sign_consistency),
            missing_expected_tokens=missing_expected,
            score=_clip01(score),
        )


class AgenticFeatureExplainer:
    def __init__(
        self,
        model_with_sae: ModelWithSAEModule,
        *,
        neuronpedia_model_id: str = "gemma-2-2b",
        neuronpedia_source: str = "0-gemmascope-res-16k",
        reasoner: Optional[OpenAICompatibleReasoner] = None,
        max_rounds: int = 3,
        initial_hypotheses: int = 3,
        prompts_per_split: int = 4,
        top_k_output_tokens: int = 12,
        intervention_value: float = 8.0,
        activation_threshold_ratio: float = 0.5,
        history_dir: str = "outputs/iteration_history",
        memory_file: str = "outputs/reasoning_memory.jsonl",
    ):
        self.model_with_sae = model_with_sae
        self.neuronpedia_model_id = neuronpedia_model_id
        self.neuronpedia_source = neuronpedia_source
        self.max_rounds = int(max_rounds)
        self.initial_hypotheses = int(initial_hypotheses)
        self.prompts_per_split = int(prompts_per_split)
        self.top_k_output_tokens = int(top_k_output_tokens)
        self.intervention_value = float(intervention_value)
        self.activation_threshold_ratio = float(activation_threshold_ratio)
        if not (0.0 <= self.activation_threshold_ratio <= 1.0):
            raise ValueError("activation_threshold_ratio must be in [0.0, 1.0].")
        self.reasoner = reasoner or OpenAICompatibleReasoner()
        self.agent_brain = AgentBrain(self.reasoner)
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.memory_store = ReasoningMemoryStore(memory_file=memory_file)
        self.memory_store.clear()

    @staticmethod
    def _write_json(path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _record_round_files(
        self,
        *,
        feature_id: int,
        round_idx: int,
        round_results: List[HypothesisRoundResult],
    ) -> Dict[str, Any]:
        snapshot = {
            "feature_id": int(feature_id),
            "round": int(round_idx),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "results": [
                {
                    "hypothesis": asdict(row.hypothesis),
                    "plan": asdict(row.plan),
                    "input_evidence": asdict(row.input_evidence),
                    "output_evidence": asdict(row.output_evidence),
                    "total_score": row.total_score,
                }
                for row in round_results
            ],
        }
        round_file = self.history_dir / f"feature_{feature_id}_round_{round_idx}.json"
        self._write_json(round_file, snapshot)
        return snapshot

    def collect_initial_observation(self, feature_id: int, topk_logits: int = 12) -> NeuronpediaObservation:
        try:
            raw = fetch_feature_json(
                model_id=self.neuronpedia_model_id,
                source=self.neuronpedia_source,
                index=str(feature_id),
            )
            raw_file = self.history_dir.parent / f"feature_{feature_id}_raw.json"
            self._write_json(raw_file, raw)
            raw.pop("explanations", None)
            logits = parse_logits(raw, topk=topk_logits)
            print(f"Logits for feature {feature_id}: {logits}")
            acts = parse_activation_corpus(raw, topn_examples=6, topk_tokens=6)
            print(f"Activation corpus for feature {feature_id}: {acts}")
        except Exception as exc:
            print(f"Warning: failed to fetch neuronpedia feature ({exc}). Falling back to empty observation.")
            logits = {"positive": [], "negative": []}
            acts = []

        seed_tokens: List[str] = []
        # vocabproj
        for tok, _ in logits.get("positive", []):  # token, value
            seed_tokens.append(str(tok))
        for tok, _ in logits.get("negative", []):
            seed_tokens.append(str(tok))
        # activate
        # Note: Here mix the steer and activate tokens
        for row in acts:  # one sentence in activations
            for item in row.get("top_tokens", []):
                if isinstance(item, dict) and "token" in item:
                    # item includes token, id and value
                    seed_tokens.append(str(item["token"]))

        seed_tokens = _dedupe_keep_order([t for t in seed_tokens if t])[:60]
        print(f"Seed tokens for feature {feature_id}: {seed_tokens}")

        return NeuronpediaObservation(
            model_id=self.neuronpedia_model_id,
            source=self.neuronpedia_source,
            feature_id=int(feature_id),
            positive_logits=[(str(t), float(v)) for t, v in logits.get("positive", [])],
            negative_logits=[(str(t), float(v)) for t, v in logits.get("negative", [])],
            top_activating_examples=acts,
            seed_tokens=seed_tokens,
        )

    def explain(self, feature_id: int) -> FinalExplanation:
        observation = self.collect_initial_observation(feature_id=feature_id)
        with open(self.history_dir.parent / f"feature_{feature_id}_observation.md", "w", encoding="utf-8") as f:
            f.write(json.dumps(asdict(observation), ensure_ascii=False, indent=2))

        memory_context = []
        hypotheses = self.agent_brain.propose_hypotheses(
            observation=observation,
            num_hypotheses=self.initial_hypotheses,
        )
        if not hypotheses:
            raise RuntimeError("No hypotheses generated.")

        runner = FeatureExperimentRunner(
            self.model_with_sae,
            feature_index=int(feature_id),
            intervention_value=self.intervention_value,
            top_k_output_tokens=self.top_k_output_tokens,
            activation_threshold_ratio=self.activation_threshold_ratio,
        )

        round_history: List[Dict[str, Any]] = []
        best_row: Optional[HypothesisRoundResult] = None
        best_score = -1.0
        stale_rounds = 0
        boundary_failure_counts: Dict[str, int] = {}
        consecutive_repeated_boundary_rounds = 0

        for round_idx in range(1, self.max_rounds + 1):
            round_results: List[HypothesisRoundResult] = []
            for hyp in hypotheses:
                plan = self.agent_brain.plan_experiments(
                    hypothesis=hyp,
                    observation=observation,
                    prompts_per_split=self.prompts_per_split,
                    memory_context=memory_context,
                    round_idx=round_idx,
                )
                # do experiments
                input_evidence = runner.evaluate_input_side(plan=plan)
                output_evidence = runner.evaluate_output_side(plan=plan, hypothesis=hyp)

                total_score = (
                    0.55 * input_evidence.score
                    + 0.45 * output_evidence.score
                    + 0.05 * hyp.confidence_prior
                    - 0.15 * input_evidence.boundary_violation_rate
                )
                total_score = _clip01(total_score)
                row = HypothesisRoundResult(
                    hypothesis=hyp,
                    plan=plan,
                    input_evidence=input_evidence,
                    output_evidence=output_evidence,
                    total_score=total_score,
                )
                round_results.append(row)

            # Record round results
            round_results.sort(key=lambda x: x.total_score, reverse=True)
            self._record_round_files(
                feature_id=feature_id,
                round_idx=round_idx,
                round_results=round_results,
            )
            round_best = round_results[0]
            if round_best.total_score > best_score:
                if best_score >= 0 and (round_best.total_score - best_score) < 0.01:
                    stale_rounds += 1
                else:
                    stale_rounds = 0
                best_row = round_best
                best_score = round_best.total_score
            else:
                stale_rounds += 1

            round_history.append(
                {
                    "round": round_idx,
                    "ranked_hypotheses": [
                        {
                            "hypothesis_id": row.hypothesis.hypothesis_id,
                            "score": row.total_score,
                            "input_score": row.input_evidence.score,
                            "output_score": row.output_evidence.score,
                            "counterexamples": row.input_evidence.counterexamples,
                            "missing_expected_tokens": row.output_evidence.missing_expected_tokens,
                        }
                        for row in round_results
                    ],
                    "round_file": str(
                        self.history_dir / f"feature_{feature_id}_round_{round_idx}.json"
                    ),
                }
            )

            china_tz = timezone(timedelta(hours=8))
            round_has_repeated_boundary_failures = False
            for row in round_results:
                row_failure_events = [asdict(evt) for evt in row.input_evidence.failure_events]
                repeated_boundary_failures: List[Dict[str, Any]] = []
                for event in row_failure_events:
                    if str(event.get("type")) != "boundary_violation":
                        continue
                    prompt = str(event.get("prompt", "")).strip()
                    prompt_key = _normalize_prompt_key(prompt)
                    if not prompt_key:
                        continue
                    boundary_failure_counts[prompt_key] = boundary_failure_counts.get(prompt_key, 0) + 1
                    repeat_count = boundary_failure_counts[prompt_key]
                    if repeat_count >= 2:
                        repeated_boundary_failures.append(
                            {
                                "prompt": prompt,
                                "count": int(repeat_count),
                                "latest_margin": float(event.get("margin", 0.0)),
                            }
                        )
                repeated_boundary_failures = sorted(
                    repeated_boundary_failures,
                    key=lambda x: (int(x.get("count", 0)), float(x.get("latest_margin", 0.0))),
                    reverse=True,
                )
                if repeated_boundary_failures:
                    round_has_repeated_boundary_failures = True
                memory_entry = {
                    "timestamp": datetime.now(china_tz).isoformat() + "Z",
                    "feature_id": int(feature_id),
                    "round": int(round_idx),
                    "hypothesis_id": row.hypothesis.hypothesis_id,
                    "hypothesis_summary": row.hypothesis.description,
                    "notes": row.hypothesis.description,
                    "score": row.total_score,
                    "score_breakdown": {
                        "total_score": row.total_score,
                        "input_score": row.input_evidence.score,
                        "output_score": row.output_evidence.score,
                        "boundary_violation_rate": row.input_evidence.boundary_violation_rate,
                        "confidence_prior": row.hypothesis.confidence_prior,
                    },
                    "counterexamples": row.input_evidence.counterexamples,
                    "failure_events": row_failure_events[:8],
                    "repeated_boundary_failures": repeated_boundary_failures[:6],
                    "missing_expected_tokens": row.output_evidence.missing_expected_tokens[:6],
                }
                self.memory_store.append(memory_entry)
                memory_context.append(memory_entry)
            if round_has_repeated_boundary_failures:
                consecutive_repeated_boundary_rounds += 1
            else:
                consecutive_repeated_boundary_rounds = 0

            # early stop
            if (
                best_score >= 0.82
                and stale_rounds >= 1
                and round_best.input_evidence.boundary_violation_rate <= 0.25
                and consecutive_repeated_boundary_rounds < 2
            ):
                break

            # critic refine
            if round_idx < self.max_rounds:
                hypotheses = self.agent_brain.critic_refine(
                    round_results=round_results,
                    keep_top_k=max(2, min(self.initial_hypotheses, len(round_results))),
                    memory_context=memory_context,
                    feature_id=feature_id,
                    round_idx=round_idx,
                )
                if not hypotheses:
                    hypotheses = [round_best.hypothesis]

        if best_row is None:
            raise RuntimeError("No successful hypothesis evaluation.")

        synthesized = self.agent_brain.synthesize_final(
            observation=observation,
            best_row=best_row,
            memory_context=memory_context,
        )
        boundaries = [str(x) for x in synthesized.get("boundaries", []) if str(x).strip()]
        testable_predictions = [
            str(x) for x in synthesized.get("testable_predictions", []) if str(x).strip()
        ]
        final_report = FinalExplanation(
            feature_id=int(feature_id),
            rounds=len(round_history),
            best_hypothesis=best_row.hypothesis,
            input_side_explanation=str(synthesized.get("input_side_explanation", "")).strip(),
            output_side_explanation=str(synthesized.get("output_side_explanation", "")).strip(),
            combined_explanation=str(synthesized.get("combined_explanation", "")).strip(),
            boundaries=boundaries,
            testable_predictions=testable_predictions,
            round_history=round_history,
        )
        run_tag = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        final_file = self.history_dir / f"feature_{feature_id}_final_{run_tag}.json"
        self._write_json(
            final_file,
            {
                "feature_id": int(feature_id),
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "final_report": final_report.to_dict(),
            },
        )
        self.memory_store.append(
            {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "feature_id": int(feature_id),
                "round": "final",
                "hypothesis_id": best_row.hypothesis.hypothesis_id,
                "score": best_row.total_score,
                "hypothesis_summary": final_report.combined_explanation,
                "notes": final_report.combined_explanation,
                "score_breakdown": {
                    "total_score": best_row.total_score,
                    "input_score": best_row.input_evidence.score,
                    "output_score": best_row.output_evidence.score,
                    "boundary_violation_rate": best_row.input_evidence.boundary_violation_rate,
                    "confidence_prior": best_row.hypothesis.confidence_prior,
                },
                "counterexamples": final_report.boundaries,
                "failure_events": [asdict(evt) for evt in best_row.input_evidence.failure_events][:8],
                "repeated_boundary_failures": [],
                "missing_expected_tokens": best_row.output_evidence.missing_expected_tokens[:6],
            }
        )
        return final_report


def format_explanation_markdown(report: FinalExplanation) -> str:
    lines: List[str] = []
    lines.append(f"# Feature {report.feature_id} Interpretation")
    lines.append("")
    lines.append("## Best Hypothesis")
    lines.append(f"- ID: {report.best_hypothesis.hypothesis_id}")
    lines.append(f"- Description: {report.best_hypothesis.description}")
    lines.append("")
    lines.append("## Input-Side Explanation")
    lines.append(report.input_side_explanation)
    lines.append("")
    lines.append("## Output-Side Explanation")
    lines.append(report.output_side_explanation)
    lines.append("")
    lines.append("## Combined Explanation")
    lines.append(report.combined_explanation)
    lines.append("")
    if report.boundaries:
        lines.append("## Boundaries / Counterexamples")
        for item in report.boundaries:
            lines.append(f"- {item}")
        lines.append("")
    if report.testable_predictions:
        lines.append("## Testable Predictions")
        for item in report.testable_predictions:
            lines.append(f"- {item}")
    return "\n".join(lines).strip()


__all__ = [
    "AgenticFeatureExplainer",
    "FinalExplanation",
    "ReasoningMemoryStore",
    "OpenAICompatibleReasoner",
    "format_explanation_markdown",
]


if __name__ == "__main__":
    from agents import set_tracing_disabled
    set_tracing_disabled(True)
    
    os.environ["ZHIPU_API_KEY"] = "sk_c5Y5ITLRO436DN6Y93BRQ733ukLavbpo_qViGETeZAA"
    os.environ["ZHIPU_BASE_URL"] = "https://api.ppio.com/openai"

    # agent = OpenAICompatibleReasoner()
    # response = agent.chat_json(
    #     system_prompt="You are a helpful assistant.",
    #     user_prompt="你好，请介绍一下你自己",
    # )

    # print(response)


    import argparse
    def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Iterative multi-agent SAE feature explainer.")
        parser.add_argument("--feature_id", type=int, required=True, help="Target SAE feature index.")

        parser.add_argument(
            "--model_name",
            type=str,
            default="models/models--google--gemma-2-2b",
            help="Model path or HF name.",
        )
        parser.add_argument(
            "--sae_path",
            type=str,
            default="models/models--google--gemma-scope-2b-pt-res/layer_0/width_16k/average_l0_105/params.npz",
            help="SAE path (.npz local, .pt local, or sae-lens:// URI).",
        )
        parser.add_argument("--sae_layer", type=int, default=None, help="SAE layer index. Optional if inferrable.")

        parser.add_argument("--neuronpedia_model_id", type=str, default="gemma-2-2b")
        parser.add_argument("--neuronpedia_source", type=str, default="0-gemmascope-res-16k")

        parser.add_argument("--rounds", type=int, default=3)
        parser.add_argument("--initial_hypotheses", type=int, default=3)
        parser.add_argument("--prompts_per_split", type=int, default=4)
        parser.add_argument("--top_k_output_tokens", type=int, default=12)
        parser.add_argument("--intervention_value", type=float, default=8.0)
        parser.add_argument(
            "--activation_threshold_ratio",
            type=float,
            default=0.5,
            help="Input-side activation threshold = max_activation * ratio (range [0,1]).",
        )

        parser.add_argument("--reasoner_model", type=str, default="zai-org/glm-4.7")
        parser.add_argument("--disable_llm", action="store_true")
        parser.add_argument("--llm_call_log", type=str, default="outputs/llm_calls.jsonl")

        parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
        parser.add_argument("--debug", action="store_true")
        parser.add_argument("--history_dir", type=str, default="outputs/iteration_history")
        parser.add_argument("--memory_file", type=str, default="outputs/reasoning_memory.jsonl")

        parser.add_argument("--output_json", type=str, default="")
        parser.add_argument("--output_md", type=str, default="")
        parser.add_argument("--quiet", action="store_true", help="Do not print markdown report to stdout.")
        return parser.parse_args()

    def resolve_device(device_arg: str) -> str:
        if device_arg != "auto":
            return device_arg
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    def build_run_output_dir(*, layer_id: int | None, feature_id: int) -> Path:
        layer_part = str(layer_id) if layer_id is not None else "unknown"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path("outputs") / Path(f"{layer_part}_{int(feature_id)}") / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def resolve_output_path_in_run_dir(*, run_dir: Path, user_value: str, default_name: str) -> Path:
        file_name = Path(user_value).name if user_value else default_name
        return run_dir / file_name
    
    args = parse_args()
    device = resolve_device(args.device)

    model_with_sae = ModelWithSAEModule(
        llm_name=args.model_name,
        sae_path=args.sae_path,
        sae_layer=args.sae_layer,
        feature_index=args.feature_id,
        device=device,
        debug=args.debug,
    )

    run_dir = build_run_output_dir(layer_id=model_with_sae.layer, feature_id=args.feature_id)
    llm_call_log_path = resolve_output_path_in_run_dir(
        run_dir=run_dir,
        user_value=args.llm_call_log,
        default_name="llm_calls.jsonl",
    )
    input_prompt_dir_path = run_dir / "input_prompt"
    history_dir_path = run_dir / Path(args.history_dir).name
    memory_file_path = resolve_output_path_in_run_dir(
        run_dir=run_dir,
        user_value=args.memory_file,
        default_name="reasoning_memory.jsonl",
    )
    output_json_path = resolve_output_path_in_run_dir(
        run_dir=run_dir,
        user_value=args.output_json,
        default_name="report.json",
    )
    output_md_path = resolve_output_path_in_run_dir(
        run_dir=run_dir,
        user_value=args.output_md,
        default_name="report.md",
    )

    if llm_call_log_path.exists():
        llm_call_log_path.unlink()

    print(f"Run output directory: {run_dir}")

    reasoner = OpenAICompatibleReasoner(
        model_name=args.reasoner_model,
        enabled=not args.disable_llm,
        llm_call_log=str(llm_call_log_path),
        prompt_log_dir=str(input_prompt_dir_path),
    )

    explainer = AgenticFeatureExplainer(
        model_with_sae=model_with_sae,
        neuronpedia_model_id=args.neuronpedia_model_id,
        neuronpedia_source=args.neuronpedia_source,
        reasoner=reasoner,
        max_rounds=args.rounds,
        initial_hypotheses=args.initial_hypotheses,
        prompts_per_split=args.prompts_per_split,
        top_k_output_tokens=args.top_k_output_tokens,
        intervention_value=args.intervention_value,
        activation_threshold_ratio=args.activation_threshold_ratio,
        history_dir=str(history_dir_path),
        memory_file=str(memory_file_path),
    )

    explainer.collect_initial_observation(feature_id=args.feature_id)


