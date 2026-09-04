# ARC3 E1 bundle

Run `kaggle/e1_qwen/e1_qwen.ipynb` with **Run All**.

Required model mount: `/kaggle/input/qwen3-8-27b-fp8-repacked-v1/pytorch/hf-fp8/1`.
Attach the existing `ARC3 vLLM H100 Wheelhouse V3` dataset (normally mounted at
`/kaggle/input/arc3-vllm-h100-wheelhouse-v3`); the notebook discovers it
automatically and uses `--no-index`. `arc3-e1-wheels` is not required.
The first notebook cell resolves this bundle by `bundle_manifest.json`, so
both a direct mount and an extra outer `arc3-e1-bundle/` directory work.
It then checks all required mounts and runtime packages.
