import os
import json
import requests
from tqdm import tqdm
from typing import Any, Dict, List, Tuple, Optional

BASE = "https://www.neuronpedia.org"

def fetch_feature_json(
    model_id: str,
    source: str,
    index: str,
    api_key: Optional[str] = None,
    timeout: int = 30,
) -> Dict[str, Any]:
    url = f"{BASE}/api/feature/{model_id}/{source}/{index}"
    headers = {}
    api_key = api_key or os.getenv("NEURONPEDIA_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()

def parse_logits(data: Dict[str, Any], topk: int = 10) -> Dict[str, List[Tuple[str, float]]]:
    # vocabproj的logits结果
    pos_str = data.get("pos_str") or []
    pos_val = data.get("pos_values") or []
    neg_str = data.get("neg_str") or []
    neg_val = data.get("neg_values") or []
    return {
        "positive": list(zip(pos_str, pos_val))[:topk],
        "negative": list(zip(neg_str, neg_val))[:topk],
    }

def parse_activation_corpus(
    data: Dict[str, Any],
    topn_examples: int = 5,
    topk_tokens: int = 5,   # 每条语料显示前 K 个 token
    include_pre_underline: bool = True,  # 是否忽略 token 前的下划线
) -> List[Dict[str, Any]]:
    '''
    np返回的activations字段是从top activate开始，包含了每句话的token和value
    函数对每句话抽取了前k个激活token
    只取前n个句子
    '''
    acts = data.get("activations") or []
    results = []
    seen_keys = set()

    for act in acts:  # one sentence in activations
        tokens = act.get("tokens") or []  # sentence tokens
        values = act.get("values") or []

        # 以防万一，进行对齐
        m = min(len(tokens), len(values))
        tokens, values = tokens[:m], values[:m]
        if m == 0:
            continue

        # 去重：top activations 中可能出现完全重复的样本
        dedup_key = (tuple(tokens), tuple(round(float(v), 6) for v in values))
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)

        k = min(topk_tokens, m)
        # 取激活值最高的前 k 个 token 的下标（按 value 降序）
        top_idx = sorted(range(m), key=lambda i: values[i], reverse=True)[:k]

        top_tokens = [
            {"index": int(i), "token": tokens[i], "value": float(values[i])}
            for i in top_idx
        ]

        # 标注：把 top-k token 用 [] 包起来（可能会标多个）
        # Note：Necessary? in case the feature activates on []?
        marked_tokens = tokens.copy()
        for i in top_idx:
            marked_tokens[i] = f"[{marked_tokens[i]}]"

        results.append(
            {
                "text": "".join(tokens) if include_pre_underline else " ".join([t[1:] if t and t[0] == "▁" else t for t in tokens]),
                "marked_text": "".join(marked_tokens) if include_pre_underline else " ".join([t[1:] if t and t[0] == "▁" else t for t in marked_tokens]),
                "top_tokens": top_tokens,
            }
        )
        if len(results) >= topn_examples:
            break

    return results

def parse_explanations(data: Dict[str, Any], topn: int = 3) -> List[Dict[str, Any]]:
    """解析explanations数组，返回前topn个解释"""
    explanations = data.get("explanations", [])
    return explanations[:topn]

def pretty_print_feature_summary(
    model_id: str,
    source: str,
    index: str,
    topk_logits: int = 10,
    topn_acts: int = 5,
    topk_tokens_each_act: int = 5,
) -> str:
    data = fetch_feature_json(model_id, source, index)
    # 保存到json
    with open(f"{model_id}_{source}_{index}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    lines = []
    lines.append(f"Feature: {model_id}/{source}/{index}")
    lines.append("")

    logits = parse_logits(data, topk=topk_logits)
    lines.append("POSITIVE LOGITS (top):")
    for tok, v in logits["positive"]:
        lines.append(f"  {tok!r}\t{v:.4f}")
    lines.append("")

    lines.append("NEGATIVE LOGITS (top):")
    for tok, v in logits["negative"]:
        lines.append(f"  {tok!r}\t{v:.4f}")
    lines.append("")

    acts = parse_activation_corpus(
        data,
        topn_examples=topn_acts,
        topk_tokens=topk_tokens_each_act,
    )
    if not acts:
        lines.append("No activation corpus found in JSON（可能该 feature 没有已保存的 top activations）。")
        return "\n".join(lines).rstrip()

    lines.append("TOP ACTIVATING EXAMPLES:")
    for i, ex in enumerate(acts, 1):
        lines.append(f"{i}. Top-{topk_tokens_each_act} tokens (by activation value):")
        for rank, item in enumerate(ex["top_tokens"], 1):
            lines.append(
                f"   {rank:>2}. idx={item['index']:>4}  tok={item['token']!r}  val={item['value']:.4f}"
            )
        lines.append("   Marked text:")
        lines.append(f"   {ex['marked_text']}")
        lines.append("")
    lines.append("")

    lines_with_explanations = lines.copy()
    explanations = parse_explanations(data, topn=3)
    if explanations:
        lines_with_explanations.append("EXPLANATIONS:")
        for i, exp in enumerate(explanations, 1):
            model_name = exp.get("explanationModelName", "Unknown")
            description = exp.get("description", "No description")
            lines_with_explanations.append(f"  {i}. Model: {model_name}")
            lines_with_explanations.append(f"     Description: {description.strip()}")
        lines_with_explanations.append("")

    return "\n".join(lines).rstrip(), "\n".join(lines_with_explanations).rstrip()


if __name__ == "__main__":
    sub_commands = [
        # ("22-gemmascope-transcoder-16k", "9388"),  # dis
        # ("18-gemmascope-res-65k", "43929"),        # Microsoft
        # ("22-gemmascope-transcoder-16k", "10450"), # such
        # ("7-gemmascope-transcoder-16k", "6394"),   # punctuation
        # ("22-gemmascope-res-65k", "5303"),         # latex
        # ("0-clt-hp", "18965"),                     # jury
        ("11-clt-hp", "84423"),                    # weight
        # ("5-clt-hp", "14420"),                     # sad
    ]
    for source, index in sub_commands:
        summary, summary_with_explanations = pretty_print_feature_summary(
            model_id="gemma-2-2b",
            source=source,
            index=index,
            topk_logits=10,
            topn_acts=5,
        )
        print(summary_with_explanations)
