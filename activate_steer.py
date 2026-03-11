# inspect_feature_activation_and_logits.py
from __future__ import annotations

import argparse
from typing import Iterable, List

import torch

from model_with_sae import ModelWithSAEModule


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def tok_repr(tok: str) -> str:
    # Safer display for weird tokens/newlines
    return repr(tok)


def print_activation_trace(module: ModelWithSAEModule, prompt: str, topk: int = 10) -> None:
    trace = module.get_activation_trace(prompt)
    tokens = trace.get("tokens", [])
    token_ids = trace.get("token_ids", [])
    acts = trace.get("per_token_activation", [])

    print("\n" + "=" * 100)
    print(f"PROMPT: {prompt}")
    print(f"summary_activation(max): {trace.get('summary_activation')}")
    print(f"summary_activation_mean: {trace.get('summary_activation_mean')}")
    print(f"summary_activation_sum : {trace.get('summary_activation_sum')}")
    print(f"max_token_index       : {trace.get('max_token_index')}")

    if tokens and acts:
        i0_tok = tokens[0]
        i0_id = token_ids[0]
        i0_act = acts[0]
        print(f"token[0]              : {tok_repr(i0_tok)} (id={i0_id}), activation={i0_act}")

    rows = list(enumerate(zip(tokens, token_ids, acts)))
    rows_sorted = sorted(rows, key=lambda x: x[1][2], reverse=True)[:topk]

    print(f"\nTop-{topk} token activations:")
    print(f"{'rank':>4} {'idx':>4} {'token_id':>8} {'activation':>12}  token")
    for rank, (idx, (tok, tok_id, act)) in enumerate(rows_sorted, start=1):
        print(f"{rank:>4} {idx:>4} {tok_id:>8} {act:>12.4f}  {tok_repr(tok)}")


def topk_logit_changes(
    *,
    clean_mean: torch.Tensor,
    steered_mean: torch.Tensor,
    tokenizer,
    top_k: int = 10,
    skip_special_tokens: bool = True,
) -> None:
    delta = steered_mean - clean_mean

    sel_inc = delta.clone()
    sel_dec = delta.clone()
    if skip_special_tokens and hasattr(tokenizer, "all_special_ids"):
        for sid in tokenizer.all_special_ids:
            if 0 <= sid < delta.shape[0]:
                sel_inc[sid] = float("-inf")
                sel_dec[sid] = float("inf")

    k = min(top_k, delta.shape[0])
    inc_vals, inc_ids = torch.topk(sel_inc, k=k, largest=True)
    dec_vals, dec_ids = torch.topk(sel_dec, k=k, largest=False)

    def print_table(title: str, ids: Iterable[int], vals: Iterable[float]) -> None:
        print("\n" + title)
        print(f"{'rank':>4} {'token_id':>8} {'clean':>12} {'steered':>12} {'delta':>12}  token")
        for rank, (tid, dval) in enumerate(zip(ids, vals), start=1):
            tid = int(tid)
            c = float(clean_mean[tid].item())
            s = float(steered_mean[tid].item())
            tok = tokenizer.convert_ids_to_tokens([tid])[0]
            print(f"{rank:>4} {tid:>8} {c:>12.4f} {s:>12.4f} {float(dval):>12.4f}  {tok_repr(tok)}")

    print_table(f"Top-{k} LOGIT INCREASE", inc_ids.tolist(), inc_vals.tolist())
    print_table(f"Top-{k} LOGIT DECREASE", dec_ids.tolist(), dec_vals.tolist())


def inspect_logits(
    module: ModelWithSAEModule,
    prompt: str,
    feature_index: int,
    intervention_value: float,
    top_k: int,
) -> None:
    tok = module.tokenizer
    if tok is None:
        raise RuntimeError("Tokenizer is required.")

    encoded = tok(
        [prompt],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )
    input_ids = encoded["input_ids"].to(module.device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(module.device)

    clean_logits = module.run_logits(input_ids=input_ids, attention_mask=attention_mask)
    amp_logits = module.run_logits_with_feature_intervention(
        input_ids=input_ids,
        feature_index=feature_index,
        value=float(intervention_value),
        mode="add",
        attention_mask=attention_mask,
    )
    sup_logits = module.run_logits_with_feature_intervention(
        input_ids=input_ids,
        feature_index=feature_index,
        value=-float(intervention_value),
        mode="add",
        attention_mask=attention_mask,
    )

    clean_mean = clean_logits.mean(dim=(0, 1)).detach().float().cpu()
    amp_mean = amp_logits.mean(dim=(0, 1)).detach().float().cpu()
    sup_mean = sup_logits.mean(dim=(0, 1)).detach().float().cpu()

    print("\n" + "=" * 100)
    print(f"LOGIT ANALYSIS PROMPT: {prompt}")
    print(f"feature_index={feature_index}, intervention_value=+{intervention_value}")
    topk_logit_changes(
        clean_mean=clean_mean,
        steered_mean=amp_mean,
        tokenizer=tok,
        top_k=top_k,
        skip_special_tokens=True,
    )

    print(f"\nfeature_index={feature_index}, intervention_value=-{intervention_value}")
    topk_logit_changes(
        clean_mean=clean_mean,
        steered_mean=sup_mean,
        tokenizer=tok,
        top_k=top_k,
        skip_special_tokens=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser("Inspect activation and logit intervention for one SAE feature.")
    parser.add_argument("--feature_id", type=int, required=True)
    parser.add_argument("--sae_layer", type=int, default=None)
    parser.add_argument(
        "--model_name",
        type=str,
        default="models/models--google--gemma-2-2b",
    )
    parser.add_argument(
        "--sae_path",
        type=str,
        default="models/models--google--gemma-scope-2b-pt-res/layer_0/width_16k/average_l0_105/params.npz",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--intervention_value", type=float, default=8.0)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument(
        "--prompt",
        action="append",
        help="Can be used multiple times. If omitted, uses built-in test prompts.",
    )
    parser.add_argument("--logit_prompt", type=str, default="")
    args = parser.parse_args()

    prompts: List[str] = args.prompt or [
        r"The proof is complete. See \label{thm1con",
        "The concept of continuity is",
        "hello world",
    ]
    logit_prompt = args.logit_prompt or prompts[0]

    module = ModelWithSAEModule(
        llm_name=args.model_name,
        sae_path=args.sae_path,
        sae_layer=args.sae_layer,
        feature_index=args.feature_id,
        device=resolve_device(args.device),
        debug=False,
    )

    for p in prompts:
        print_activation_trace(module, p, topk=args.top_k)

    inspect_logits(
        module=module,
        prompt=logit_prompt,
        feature_index=args.feature_id,
        intervention_value=args.intervention_value,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
