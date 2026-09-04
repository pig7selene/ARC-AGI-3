# ARC-AGI-3 agent V1

Inspectable, game-agnostic baseline for interactive ARC-AGI-3 research. It runs without third-party dependencies and is designed to accept the official toolkit or a local OpenAI-compatible model later.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[official]'
```

```bash
python3 -m unittest discover -s tests -v
python3 -m arc3.eval.runner --agent plumbing --max-actions 20
python3 -m arc3.eval.runner --agent reasoning --max-actions 40 --trace-dir traces
```

The mock game is a regression fixture, not a public-game score. For official games, instantiate the installed toolkit environment and pass it through `arc3.env.protocol.adapt_environment`; use its scorer/replay writer. See [docs/research.md](docs/research.md) and [docs/architecture.md](docs/architecture.md).

With the official toolkit installed and locally downloaded games:

```bash
.venv/bin/python -m arc3.eval.runner --toolkit-mode offline --agent plumbing --max-actions 20 --trace-dir traces/plumbing_real
.venv/bin/python -m arc3.eval.runner --toolkit-mode offline --agent reasoning --max-actions 30 --trace-dir traces/reasoning_real
.venv/bin/python -m arc3.eval.analyze_trace traces/reasoning_real/ls20-9607627b.jsonl
.venv/bin/python -m arc3.eval.replay traces/reasoning_real/ls20-9607627b.jsonl -o traces/reasoning_real/ls20.html
```

The first V2 research experiment is reproducible with:

```bash
.venv/bin/python -m experiments.v2.run_fixed_subset
```

It writes the baseline/intervention comparison to `results/v2/retrodict_plan.json` and detailed traces to `traces/v2/`.

For the semantic-policy experiment, provide an explicitly authorized OpenAI-compatible endpoint without committing credentials:

```bash
ARC3_MODEL_BASE_URL=... ARC3_MODEL=... ARC3_MODEL_API_KEY=... \
  .venv/bin/python -m experiments.v2.run_model_policy
```

## E1 — local Qwen/vLLM Kaggle run

E1 keeps the existing `ReasoningAgent` and replaces only its policy backend with a local Qwen3.8-27B-FP8 checkpoint served by vLLM. Build the offline source bundle before uploading it as a private Kaggle dataset:

```bash
E1_BUNDLE_DIR=/tmp/arc3-e1-bundle \
  .venv/bin/python kaggle/e1_qwen/build_bundle.py
```

In Kaggle, attach the bundle, the model dataset
`foysalemonshanto/qwen3-8-27b-fp8-repacked-v1`, and offline wheel datasets;
disable Internet; select an RTX Pro 6000; then run
`kaggle/e1_qwen/e1_qwen.ipynb`.  The default `E1_MODEL_ID` is the confirmed
mount `/kaggle/input/qwen3-8-27b-fp8-repacked-v1/pytorch/hf-fp8/1`; override it
only by explicitly setting `E1_MODEL_ID`.  Optionally set
`E1_SERVED_MODEL_NAME`, `E1_BUNDLE_DIR`, and `E1_WHEEL_DIR`.  The notebook's
default run is the fixed three-game subset, seed 0, 30 actions/game. It writes
`e1_results.json`, `e1_model_metadata.json`, `submission.parquet` (trace
metrics), and `e1_traces/`; the official toolkit scorecard remains
authoritative.

The launcher accepts an OpenAI-compatible base URL (`E1_BASE_URL`, default `http://127.0.0.1:1234/v1`) and optional `E1_GAME_IDS=a,b,c` or `E1_GAME_SET=full` for an explicitly requested benchmark expansion. See [docs/experiments.md](docs/experiments.md) for the E0/E1 controls, deferred Duck features, and reporting schema.
