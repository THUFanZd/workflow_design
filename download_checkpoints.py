import argparse
import inspect
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from list_dir import list_repo_dir


DEFAULT_RELEASE = "gemma-scope-2b-pt-res"
DEFAULT_REPO_ID = "google/gemma-scope-2b-pt-res"
DEFAULT_LAYERS = (18, 24)  # 配置
DEFAULT_WIDTH = "16k"
DEFAULT_TARGET_L0 = 100.0


@dataclass(frozen=True)
class CanonicalResolution:
    layer: int
    width: str
    average_l0: str
    sae_id: str
    source: str


def extract_average_l0_from_canonical_map(
    canonical_map_path: str | Path,
    layer: int,
    width: str = DEFAULT_WIDTH,
) -> Optional[str]:
    """
    Extract canonical average_l0 for a given layer/width from canonical_map.txt.
    Returns the numeric part as a string, e.g. "82", or None if not found.
    """
    target_id = f"layer_{layer}/width_{width}/canonical"
    in_target_block = False
    path_pattern = re.compile(rf"layer_{layer}/width_{re.escape(width)}/average_l0_([0-9]+(?:\.[0-9]+)?)")

    with Path(canonical_map_path).open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            if line.startswith("- id:"):
                current_id = line.split(":", 1)[1].strip()
                in_target_block = current_id == target_id
                continue

            if in_target_block and line.startswith("path:"):
                match = path_pattern.search(line.split(":", 1)[1].strip())
                if match:
                    return match.group(1)
                return None

    return None


def _extract_average_l0_token(path_like: str) -> Optional[tuple[str, float]]:
    normalized = path_like.replace("\\", "/")
    tail = normalized.split("/")[-1]
    if not tail.startswith("average_l0_"):
        return None
    number_str = tail[len("average_l0_") :]
    try:
        return tail, float(number_str)
    except ValueError:
        return None


def select_canonical_leaf_from_repo_listing(
    repo_listing: Iterable[str],
    target_l0: float = DEFAULT_TARGET_L0,
) -> Optional[str]:
    """
    Fallback selector: pick the average_l0_* leaf whose numeric value is closest to target_l0.
    """
    best_leaf: Optional[str] = None
    best_distance = float("inf")
    best_value = float("inf")

    for item in repo_listing:
        parsed = _extract_average_l0_token(item)
        if not parsed:
            continue
        leaf, value = parsed
        distance = abs(value - target_l0)
        if distance < best_distance or (distance == best_distance and value < best_value):
            best_leaf = leaf
            best_distance = distance
            best_value = value

    return best_leaf


def resolve_canonical_sae_id(
    layer: int,
    width: str = DEFAULT_WIDTH,
    release: str = DEFAULT_RELEASE,
    repo_id: str = DEFAULT_REPO_ID,
    canonical_map_path: str | Path = "canonical_map.txt",
    target_l0: float = DEFAULT_TARGET_L0,
) -> CanonicalResolution:
    """
    Resolve canonical SAE id like:
      layer_12/width_16k/average_l0_82
    First try canonical_map.txt, then fallback to list_repo_dir(...) if parsing fails.
    """
    base_path = f"layer_{layer}/width_{width}"
    avg_l0 = extract_average_l0_from_canonical_map(canonical_map_path, layer=layer, width=width)
    if avg_l0 is not None:
        return CanonicalResolution(
            layer=layer,
            width=width,
            average_l0=avg_l0,
            sae_id=f"{base_path}/average_l0_{avg_l0}",
            source="canonical_map",
        )

    listing = list_repo_dir(path_in_repo=base_path, repo_id=repo_id, repo_type="model")
    leaf = select_canonical_leaf_from_repo_listing(listing, target_l0=target_l0)
    if not leaf:
        raise ValueError(
            f"Could not resolve canonical SAE for layer={layer}, width={width}. "
            f"canonical_map and list_repo_dir fallback both failed."
        )

    avg_l0_from_leaf = leaf[len("average_l0_") :]
    return CanonicalResolution(
        layer=layer,
        width=width,
        average_l0=avg_l0_from_leaf,
        sae_id=f"{base_path}/{leaf}",
        source="repo_listing_fallback",
    )


def download_sae_checkpoint_via_sae_lens(
    release: str,
    sae_id: str,
    device: str = "cpu",
    force_download: bool = False,
) -> str:
    """
    Download checkpoint to local HF cache via sae-lens.
    Prefer SAE's explicit download method (if available), otherwise use SAE.from_pretrained(...).
    """
    try:
        from sae_lens import SAE  # local import so map parsing can run without importing heavy deps
    except ImportError as exc:
        raise RuntimeError(
            "Failed to import sae_lens. In this environment, transformers requires "
            "huggingface-hub>=0.34.0,<1.0, but the installed version is incompatible. "
            "Please fix env versions first, then rerun this script."
        ) from exc

    for name in dir(SAE):
        if "download" not in name.lower():
            continue
        fn = getattr(SAE, name)
        if not callable(fn):
            continue
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            continue
        params = sig.parameters
        if "release" not in params or "sae_id" not in params:
            continue
        kwargs = {"release": release, "sae_id": sae_id}
        if "force_download" in params:
            kwargs["force_download"] = force_download
        fn(**kwargs)
        return f"SAE.{name}"

    from_pretrained_sig = inspect.signature(SAE.from_pretrained)
    kwargs = {"release": release, "sae_id": sae_id}
    if "device" in from_pretrained_sig.parameters:
        kwargs["device"] = device
    if "force_download" in from_pretrained_sig.parameters:
        kwargs["force_download"] = force_download
    SAE.from_pretrained(**kwargs)
    return "SAE.from_pretrained"


def download_requested_checkpoints(
    layers: Iterable[int] = DEFAULT_LAYERS,
    width: str = DEFAULT_WIDTH,
    release: str = DEFAULT_RELEASE,
    repo_id: str = DEFAULT_REPO_ID,
    canonical_map_path: str | Path = "canonical_map.txt",
    device: str = "cpu",
    target_l0: float = DEFAULT_TARGET_L0,
    dry_run: bool = False,
    force_download: bool = False,
) -> list[CanonicalResolution]:
    resolved: list[CanonicalResolution] = []
    for layer in layers:
        item = resolve_canonical_sae_id(
            layer=layer,
            width=width,
            release=release,
            repo_id=repo_id,
            canonical_map_path=canonical_map_path,
            target_l0=target_l0,
        )
        print(
            f"[resolve] layer={item.layer} width={item.width} "
            f"sae_id={item.sae_id} source={item.source}"
        )

        if not dry_run:
            method = download_sae_checkpoint_via_sae_lens(
                release=release,
                sae_id=item.sae_id,
                device=device,
                force_download=force_download,
            )
            print(f"[download] release={release} sae_id={item.sae_id} via={method}")

        resolved.append(item)
    return resolved


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resolve and download Gemma Scope canonical SAE checkpoints."
    )
    parser.add_argument("--release", default=DEFAULT_RELEASE)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--layers", nargs="+", type=int, default=list(DEFAULT_LAYERS))
    parser.add_argument("--width", default=DEFAULT_WIDTH)
    parser.add_argument("--canonical-map", default="canonical_map.txt")
    parser.add_argument("--target-l0", type=float, default=DEFAULT_TARGET_L0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    return parser


def main() -> None:
    import os
    # 设置 Hugging Face 国内镜像加速
    # os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    # 如果你需要同时设置代理（两者可以配合使用，也可以只用镜像）
    # os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
    # os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
    parser = build_arg_parser()
    args = parser.parse_args()
    download_requested_checkpoints(
        layers=args.layers,
        width=args.width,
        release=args.release,
        repo_id=args.repo_id,
        canonical_map_path=args.canonical_map,
        device=args.device,
        target_l0=args.target_l0,
        dry_run=args.dry_run,
        force_download=args.force_download,
    )


if __name__ == "__main__":
    main()
