from __future__ import annotations

if __name__ == "tokenize":
    import importlib.util
    import os
    import sysconfig

    _stdlib_tokenize = os.path.join(sysconfig.get_path("stdlib"), "tokenize.py")
    _spec = importlib.util.spec_from_file_location("tokenize", _stdlib_tokenize)
    if _spec is None or _spec.loader is None:
        raise RuntimeError("Failed to load stdlib tokenize module.")
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    globals().update(_module.__dict__)
else:
    import argparse
    import os

    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("WANDB_MODE", "disabled")

    import torch

    from model_with_sae import load_model, load_tokenizer


    def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Tokenize a prompt with the same LLM as run_explainer.py.")
        parser.add_argument(
            "--model_name",
            type=str,
            default="models/models--google--gemma-2-2b",
            help="Model path or HF name (same default as run_explainer.py).",
        )
        parser.add_argument(
            "--device",
            type=str,
            default="auto",
            choices=["auto", "cpu", "cuda"],
            help="Device for loading the model.",
        )
        parser.add_argument(
            "--prompt",
            type=str,
            default="Hello world.",
        )
        return parser.parse_args()


    def resolve_device(device_arg: str) -> str:
        if device_arg != "auto":
            return device_arg
        return "cuda" if torch.cuda.is_available() else "cpu"


    def main() -> None:
        args = parse_args()
        device = resolve_device(args.device)

        model = load_model(args.model_name, device, use_hooked_transformer=False)
        if model is None:
            raise RuntimeError(f"Failed to load model: {args.model_name}")

        tokenizer = load_tokenizer(args.model_name)
        if tokenizer is None:
            raise RuntimeError(f"Failed to load tokenizer: {args.model_name}")

        prompt = args.prompt
        # 如果 ./tokenize_text.txt 存在，从该文件读取 prompt
        if os.path.exists("./tokenize_text.txt"):
            with open("./tokenize_text.txt", "r", encoding="utf-8") as f:
                prompt = f.read().strip()
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = encoded["input_ids"][0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        print(f"Model: {args.model_name}")
        print(f"Device: {device}")
        print(f"Prompt: {prompt}")
        print(f"Token count: {len(input_ids)}")
        print("Token IDs:")
        print(input_ids)
        print("Tokens:")
        for idx, (token_id, token) in enumerate(zip(input_ids, tokens)):
            print(f"{idx:>4}: id={token_id:<8} token={token}")


    if __name__ == "__main__":
        main()
