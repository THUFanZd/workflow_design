from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path

os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "disabled")

import torch

from agentic_feature_explainer import (
    AgenticFeatureExplainer,
    OpenAICompatibleReasoner,
    format_explanation_markdown,
)
from download_checkpoints import resolve_canonical_sae_id
from model_with_sae import ModelWithSAEModule

try:
    # Loads environment variables for ZHIPU API from local helper.
    import myagents.api_key  # noqa: F401
except Exception:
    pass


DEFAULT_SAE_RELEASE = "gemma-scope-2b-pt-res"
DEFAULT_SAE_REPO_ID = "google/gemma-scope-2b-pt-res"
DEFAULT_SAE_WIDTH = "16k"
DEFAULT_LAYER = 6


def infer_layer_from_sae_path(sae_path: str) -> int | None:
    if not sae_path:
        return None
    candidates = [sae_path]
    if sae_path.startswith("sae-lens://"):
        spec = sae_path[len("sae-lens://") :]
        kv: dict[str, str] = {}
        for part in [p.strip() for p in spec.split(";") if p.strip()]:
            if "=" in part:
                key, value = part.split("=", 1)
                kv[key.strip()] = value.strip()
        sae_id = kv.get("sae_id") or kv.get("path")
        if sae_id:
            candidates.insert(0, sae_id)

    for text in candidates:
        match = re.search(r"layer[_-]?(\d+)", text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def infer_layer_from_neuronpedia_source(source: str) -> int | None:
    if not source:
        return None
    match = re.match(r"^\s*(\d+)-", source)
    return int(match.group(1)) if match else None


def resolve_conflict_free_sae_settings(args: argparse.Namespace) -> None:
    canonical_map_path = Path(args.canonical_map)
    if not canonical_map_path.is_absolute():
        canonical_map_path = Path(__file__).with_name(args.canonical_map)

    if not args.sae_path:
        resolved = resolve_canonical_sae_id(
            layer=args.layer,
            width=args.width,
            release=args.sae_release,
            repo_id=args.sae_repo_id,
            canonical_map_path=canonical_map_path,
        )
        args.sae_path = f"sae-lens://release={args.sae_release};sae_id={resolved.sae_id}"

    if not args.neuronpedia_source:
        args.neuronpedia_source = f"{args.layer}-gemmascope-res-{args.width}"

    if args.sae_layer is None:
        args.sae_layer = args.layer

    sae_layer = infer_layer_from_sae_path(args.sae_path)
    if sae_layer is not None and sae_layer != args.layer:
        raise ValueError(
            f"Parameter conflict: --layer={args.layer} but sae_path implies layer={sae_layer}. "
            "Please keep them consistent."
        )

    np_layer = infer_layer_from_neuronpedia_source(args.neuronpedia_source)
    if np_layer is not None and np_layer != args.layer:
        raise ValueError(
            f"Parameter conflict: --layer={args.layer} but neuronpedia_source implies layer={np_layer}. "
            "Please keep them consistent."
        )

    if args.sae_layer != args.layer:
        raise ValueError(
            f"Parameter conflict: --sae_layer={args.sae_layer} but --layer={args.layer}. "
            "Please keep them consistent."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Iterative multi-agent SAE feature explainer.")
    parser.add_argument("--feature_id", type=int, required=True, help="Target SAE feature index.")

    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-2-2b",  # "models/models--google--gemma-2-2b",
        help="Model path or HF name.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=DEFAULT_LAYER,
        help="Shared SAE layer index; used to build default sae_path and neuronpedia_source.",
    )
    parser.add_argument("--width", type=str, default=DEFAULT_SAE_WIDTH, help="SAE width, e.g. 16k/65k/1m.")
    parser.add_argument("--sae_release", type=str, default=DEFAULT_SAE_RELEASE, help="SAE-Lens release name.")
    parser.add_argument(
        "--sae_repo_id",
        type=str,
        default=DEFAULT_SAE_REPO_ID,
        help="HF repo id used by canonical fallback listing.",
    )
    parser.add_argument(
        "--canonical_map",
        type=str,
        default="canonical_map.txt",
        help="Path to canonical map file for resolving average_l0.",
    )
    parser.add_argument(
        "--sae_path",
        type=str,
        default="",
        help="SAE path (.npz local, .pt local, or sae-lens:// URI). If empty, auto-resolved from layer/width.",
    )
    parser.add_argument("--sae_layer", type=int, default=None, help="SAE layer index. Optional if inferrable.")

    parser.add_argument("--neuronpedia_model_id", type=str, default="gemma-2-2b")
    parser.add_argument(
        "--neuronpedia_source",
        type=str,
        default="",
        help="Neuronpedia source. If empty, auto-set as '<layer>-gemmascope-res-<width>'.",
    )

    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--initial_hypotheses", type=int, default=1)
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
    args = parser.parse_args()
    resolve_conflict_free_sae_settings(args)
    return args


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


def main() -> None:
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

    report = explainer.explain(feature_id=args.feature_id)
    markdown = format_explanation_markdown(report)
    if not args.quiet:
        try:
            print(markdown)
        except UnicodeEncodeError:
            print(markdown.encode("ascii", errors="replace").decode("ascii"))

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved JSON report to: {output_json_path}")

    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    output_md_path.write_text(markdown, encoding="utf-8")
    print(f"\nSaved markdown report to: {output_md_path}")


if __name__ == "__main__":
    os.environ["ZHIPU_API_KEY"] = "sk_c5Y5ITLRO436DN6Y93BRQ733ukLavbpo_qViGETeZAA"
    os.environ["ZHIPU_BASE_URL"] = "https://api.ppio.com/openai"
    from agents import set_tracing_disabled
    set_tracing_disabled(True)
    main()
