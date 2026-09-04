"""Create a small offline source bundle for a Kaggle dataset/notebook."""
from __future__ import annotations

import json
from pathlib import Path
import shutil


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    output = Path(__import__("os").environ.get("E1_BUNDLE_DIR", str(root / "artifacts" / "arc3-e1-bundle"))).resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing bundle: {output}; choose a new E1_BUNDLE_DIR.")
    output.mkdir(parents=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(root / "arc3", output / "arc3", ignore=ignore)
    shutil.copytree(root / "experiments" / "v2", output / "experiments" / "v2", ignore=ignore)
    shutil.copytree(root / "kaggle" / "e1_qwen", output / "kaggle" / "e1_qwen", ignore=ignore)
    (output / "artifacts").mkdir(parents=True, exist_ok=True)
    shutil.copy(root / "artifacts" / "public_games.json", output / "artifacts" / "public_games.json")
    manifest = {
        "bundle": "arc3-e1-bundle",
        "source": str(root),
        "included": ["arc3", "experiments/v2", "kaggle/e1_qwen", "artifacts/public_games.json"],
        "excluded": ["environment_files", "traces", "recordings", "results", "__pycache__"],
        "target_model_dataset": "foysalemonshanto/qwen3-8-27b-fp8-repacked-v1",
        "target_model_path": "/kaggle/input/qwen3-8-27b-fp8-repacked-v1/pytorch/hf-fp8/1",
        "notebook": "kaggle/e1_qwen/e1_qwen.ipynb",
        "wheelhouse_dataset": "ARC3 vLLM H100 Wheelhouse V3",
        "wheelhouse_input_slug": "arc3-vllm-h100-wheelhouse-v3",
        "note": "Attach the target model, this bundle, and the existing ARC3 vLLM H100 Wheelhouse V3; run the notebook with Run All. No new arc3-e1-wheels dataset is required.",
    }
    (output / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
    (output / "BUNDLE_README.md").write_text(
        "# ARC3 E1 bundle\n\n"
        "Run `kaggle/e1_qwen/e1_qwen.ipynb` with **Run All**.\n\n"
        "Required model mount: `/kaggle/input/qwen3-8-27b-fp8-repacked-v1/pytorch/hf-fp8/1`.\n"
        "Attach the existing `ARC3 vLLM H100 Wheelhouse V3` dataset (normally mounted at\n"
        "`/kaggle/input/arc3-vllm-h100-wheelhouse-v3`); the notebook discovers it\n"
        "automatically and uses `--no-index`. `arc3-e1-wheels` is not required.\n"
        "The first notebook cell resolves this bundle by `bundle_manifest.json`, so\n"
        "both a direct mount and an extra outer `arc3-e1-bundle/` directory work.\n"
        "It then checks all required mounts and runtime packages.\n",
        encoding="utf8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
