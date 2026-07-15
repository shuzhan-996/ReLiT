import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset_repo",
        type=str,
        default="AZYoung/MOSEI_processed",
        help="HuggingFace dataset repo id.",
    )
    p.add_argument(
        "--root_dataset_dir",
        type=str,
        default=None,
        help=(
            "Where to create the project-expected dataset layout. "
            "This should be the same value you pass to run.py --root_dataset_dir. "
            "If omitted, defaults to <repo_root>/datasets (repo_root is parent of MSE-ChatGLM3-6B)."
        ),
    )
    p.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Optional HF cache dir for snapshot_download.",
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
        help="Overwrite the target file if it already exists.",
    )
    return p.parse_args()


def main():
    args = _parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    root_dataset_dir = (
        Path(args.root_dataset_dir).expanduser().resolve()
        if args.root_dataset_dir
        else (repo_root / "datasets")
    )

    target_dir = root_dataset_dir / "MOSEI" / "Processed"
    target_dir.mkdir(parents=True, exist_ok=True)

    target_file = target_dir / "unaligned_50.pkl"
    if target_file.exists() and not args.force:
        raise FileExistsError(
            f"Target already exists: {target_file}. Use --force to overwrite."
        )

    api = HfApi()
    try:
        repo_files = api.list_repo_files(
            repo_id=args.dataset_repo,
            repo_type="dataset",
            token=args.hf_token,
        )
    except Exception:
        repo_files = None

    wanted_path = None
    if repo_files:
        # Prefer the exact expected filename; fall back to any file containing 'unaligned_50'.
        exact = [p for p in repo_files if p.endswith("unaligned_50.pkl")]
        if exact:
            wanted_path = exact[0]
        else:
            fuzzy = [p for p in repo_files if ("unaligned_50" in p and p.endswith(".pkl"))]
            if fuzzy:
                wanted_path = fuzzy[0]

    snapshot_path = Path(
        snapshot_download(
            repo_id=args.dataset_repo,
            repo_type="dataset",
            # NOTE: In huggingface_hub, patterns like '**/foo' may not match root-level 'foo'.
            allow_patterns=(
                [wanted_path] if wanted_path else ["unaligned_50.pkl", "**/unaligned_50.pkl"]
            ),
            cache_dir=args.cache_dir,
            token=args.hf_token,
        )
    )

    candidates = list(snapshot_path.rglob("unaligned_50.pkl"))
    if not candidates:
        hint = ""
        if repo_files:
            pkl_files = [p for p in repo_files if p.endswith(".pkl")]
            hint = (
                "\nRepo .pkl files (first 50):\n- "
                + "\n- ".join(pkl_files[:50])
            )
        raise FileNotFoundError(
            f"unaligned_50.pkl not found in snapshot: {snapshot_path}{hint}"
        )

    src = candidates[0]
    shutil.copy2(src, target_file)

    print("Downloaded OK")
    print(f"- Source: {src}")
    print(f"- Target: {target_file}")


if __name__ == "__main__":
    main()
