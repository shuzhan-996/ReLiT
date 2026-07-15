import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model_repo",
        type=str,
        default="zai-org/chatglm3-6b",
        help="HuggingFace model repo id.",
    )
    p.add_argument(
        "--target_dir",
        type=str,
        default="/data/huggingface_model/THUDM/chatglm3-6b-base/",
        help=(
            "Where to place the downloaded model files. This should match run.py --pretrain_LM. "
            "Default matches MSE-ChatGLM3-6B/run.py parse_args() default."
        ),
    )
    p.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Optional HF cache dir for snapshot_download.",
    )
    p.add_argument(
        "--weights_format",
        type=str,
        default="safetensors",
        choices=["safetensors", "bin"],
        help=(
            "Which weight format to download. Use 'safetensors' (recommended) or 'bin'. "
            "This script will ONLY download the selected format to save bandwidth/disk."
        ),
    )
    p.add_argument(
        "--hf_token",
        type=str,
        default=os.environ.get("HF_TOKEN", None),
        help="HuggingFace token. Can also be provided via env HF_TOKEN.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite target_dir if it already exists (will delete it first).",
    )
    return p.parse_args()


def main():
    args = _parse_args()

    target_dir = Path(args.target_dir).expanduser().resolve()

    if target_dir.exists() and args.force:
        shutil.rmtree(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)

    base_patterns = [
        "config.json",
        "generation_config.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "configuration_chatglm.py",
        "modeling_chatglm.py",
        "quantization.py",
        "*.txt",
        "*.md",
    ]

    if args.weights_format == "safetensors":
        weight_patterns = ["model.safetensors.index.json", "model-*.safetensors"]
    else:
        weight_patterns = ["pytorch_model.bin.index.json", "pytorch_model-*.bin"]

    allow_patterns = base_patterns + weight_patterns

    snapshot_download(
        repo_id=args.model_repo,
        repo_type="model",
        allow_patterns=allow_patterns,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        cache_dir=args.cache_dir,
        token=args.hf_token,
    )

    print("Downloaded OK")
    print(f"- Repo: {args.model_repo}")
    print(f"- Target: {target_dir}")
    print("\nVerify files:")
    for name in ["config.json", "tokenizer.model"]:
        p = target_dir / name
        print(f"- {name}: {'OK' if p.exists() else 'MISSING'} ({p})")

    if args.weights_format == "safetensors":
        index_file = target_dir / "model.safetensors.index.json"
    else:
        index_file = target_dir / "pytorch_model.bin.index.json"
    print(f"- weights_index: {'OK' if index_file.exists() else 'MISSING'} ({index_file})")


if __name__ == "__main__":
    main()
