# Experiments

The repository starts with deterministic smoke experiments over `MoveToTargetEnv`, a tiny ARC-like test environment used only for regression. Run both agents and inspect JSONL traces before connecting the official toolkit. Record official public-suite runs here with toolkit version, model, seed, score, levels, actions, no-op rate and runtime; do not compare the mock score to leaderboard scores.

Smoke run (Python 3.14, seed 0, local CPU): core `unittest` 13/13 passed (official adapter tests run in `.venv`). PlumbingAgent, 10-action mock horizon: score -0.10, 0 levels. ReasoningAgent, 40-action mock horizon: score 0.66, 1 level, 35 actions, 20% no-op rate, 8 ms. These numbers validate plumbing/observability only.

V1.1 real OFFLINE run (toolkit `arc-agi 0.9.9`, `arcengine 0.9.3`, three locally downloaded public environments `cd82-fb555c5d`, `ls20-9607627b`, `lf52-271a04aa`, seed 0): PlumbingAgent at 20 actions/game completed 0/23 levels, official scorecard 0.0, 60 actions, 13.3% no-op, 0.53 s. ReasoningAgent at 30 actions/game completed 0/23 levels, official scorecard 0.0, 90 actions, 12.2% no-op, 1.51 s. Diagnostics identify no-op/state loops on CD82 and budget exhaustion on LS20/LF52. Full JSONL traces are under `traces/plumbing_real/` and `traces/reasoning_real/`; registry metadata is `artifacts/public_games.json`.

Full current OFFLINE registry smoke (`25` environments / `183` levels, PlumbingAgent, 5 actions per game): `125` actions, official scorecard `0.0`, 0 completed levels, runtime about `18 s`. This is a plumbing/diagnostics run, not a score attempt; it confirms every downloaded environment can reset and accept the adapter's action path.

## V2 experiment 001 — bounded prediction + short verified plans (H2 + H3)

Config: `experiments/v2/retrodict_plan.json`; runner: `experiments/v2/run_fixed_subset.py` (reruns truncate the three named trace files per variant for clean comparisons). Both variants used toolkit OFFLINE, seed `0`, the fixed subset `cd82-fb555c5d`, `ls20-9607627b`, `lf52-271a04aa`, and 30 actions per game. The baseline disables counterfactual action-outcome reuse and prediction verification. The intervention reuses observed action outcomes, accepts a maximum three-step plan, and clears the queue on a structured prediction mismatch. No model endpoint was available, so both use the deterministic `HeuristicBackend`.

Measured run (`results/v2/retrodict_plan.json`):

| variant | score | levels | actions | no-op rate | repeated action/state pairs | runtime |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.0 | 0/23 | 90 | 12.2% | 11 | 1.51 s |
| intervention | 0.0 | 0/23 | 90 | 10.0% | 6 | 1.50 s |

The intervention reduced repeated action/state pairs by 45% and no-ops by 18% relative, but did not clear a level with the heuristic backend. Its simple structured visibility prediction was checked 55 times and mismatched 4 times; each mismatch correctly cleared the pending queue and recorded a replan. Decision: **KEEP as an instrumentation/control intervention; MODIFY before a score claim**. The next run should use a model backend that emits richer expectations and plans, then compare prediction accuracy and level progress. This result does not justify adding a learned world model or game-specific rules yet.

The clean rerun also records the newly requested observable-progress and interface metrics: both variants reached novel observable states and observed new mechanics, but created 0 goals and 0 plans. The heuristic intervention had 33 parse/invalid-output fallbacks (mostly missing ACTION6 coordinates); this is another reason to measure structured-output reliability before spending environment actions on a real model.

## V2 experiment 002 — semantic policy substitution (H6)

### Hypothesis

Replacing only the deterministic policy with a capable reasoning model will create measurable goal hypotheses and executable plans, and may produce the first completed public level. The existing prediction/replan contract and representation are held constant. The cheapest falsification is a fixed three-game run in which the model receives the current text/grid context but still produces zero goals/plans or zero progress; that would point to representation or interface failure rather than simply weak action heuristics.

### Evidence

H2+H3 reduced no-op and repeated-state behavior but produced no goals or plans. The planner is not being rejected by a model; it is never reached because `HeuristicBackend` emits no goal or plan fields. This isolates policy/model capability as the highest-value next variable.

### Intervention

Add a replaceable OpenAI-compatible reasoning backend with bounded context, strict JSON output, concise summaries, structured expectations, and optional three-step plans. Do not add multimodal frames, Python tools, learned transitions, or new search in this experiment.

Implementation: `experiments/v2/run_model_policy.py`. It reads `ARC3_MODEL_BASE_URL`, `ARC3_MODEL`, and optional `ARC3_MODEL_API_KEY`; credentials are never written to the repository. The backend receives the raw grid plus current structured objects, frame diff, bounded memory, action outcomes, and the prediction contract.

### Fixed Variables

Same `cd82-fb555c5d`, `ls20-9607627b`, `lf52-271a04aa`, seed `0`, 30 environment actions per game, existing `ActionExpectation` verification, guards, memory, traces, and official adapter.

### Falsification / Next Question

If a real model still produces no goals/plans or no level progress, inspect whether the text/grid representation is insufficient (H4) before adding tools or a world model. If it does produce plans but fails execution, isolate plan/representation errors in the trace.

Status: implementation is ready, but no model result is claimed in this iteration. The only configured remote endpoint exposes credentials to a third-party proxy whose authorization/protocol could not be safely verified, so the runner was not allowed to transmit them. Run the experiment only with an explicitly authorized OpenAI-compatible endpoint.

## E1 — Our Agent + Qwen3.8-27B-FP8 preparation

E1 substitutes only the policy backend: the existing `ReasoningAgent`, raw-grid + structured-object representation, bounded memory, short-plan queue, prediction verification, guards, official adapter and trace schema remain unchanged. The model boundary is a local OpenAI-compatible vLLM server on `127.0.0.1:1234`; no model weights are downloaded by this repository or by the notebook.

The first controlled run is the fixed small subset `cd82-fb555c5d`, `ls20-9607627b`, `lf52-271a04aa`, seed `0`, and 30 environment actions per game. E1 uses the confirmed E0 checkpoint `/kaggle/input/qwen3-8-27b-fp8-repacked-v1/pytorch/hf-fp8/1`, the published Duck sampling values (`temperature=0.6`, `top_p=0.95`, `top_k=20`), a 2,048-token response cap, and thinking enabled. Set `E1_TEMPERATURE=0` only for a separately labelled deterministic smoke run; the controlled run keeps `0.6`.

The runner is `kaggle/e1_qwen/e1_runner.py`; it waits for `/v1/models` before spending environment actions, verifies the served model name, and writes `e1_model_metadata.json`, `e1_results.json`, per-game `e1_traces/*.jsonl` plus summaries, and `submission.parquet` (a trace-metrics table, not a replacement for the official scorecard). The default subset is immutable. An explicit `E1_GAME_IDS=a,b,c` list or `E1_GAME_SET=full` opts into another set after the subset has been measured.

### E1 preparation checks

- `27/27` unit tests pass, including backend JSON-wrapper/usage tests, prediction/replan tests, official adapter tests, E1 runner URL/game-selection tests, and metadata/preflight checks.
- `kaggle/e1_qwen/e1_qwen.ipynb` and `e1_config.json` parse as valid JSON; `serve_qwen.sh` passes shell syntax validation.
- `build_bundle.py` creates an offline source bundle and refuses to overwrite an existing output directory. A second isolated import test loads the runner from the generated bundle outside the repository checkout.
- No Qwen endpoint was called. The only discovered API credentials point to an unverified third-party proxy and were not transmitted.

### E1 Kaggle run

1. Build the bundle locally with `E1_BUNDLE_DIR=/tmp/arc3-e1-bundle .venv/bin/python kaggle/e1_qwen/build_bundle.py`, then upload that directory as a private Kaggle dataset (for example `arc3-e1-bundle`).
2. Attach the exact offline Qwen3.8-27B-FP8 model dataset and a private wheel dataset containing vLLM plus the official `arc-agi`/`arcengine` wheels. Turn Internet off and select an RTX Pro 6000 accelerator in the notebook settings.
3. Open `kaggle/e1_qwen/e1_qwen.ipynb`. Set `E1_MODEL_ID` to the mounted local model directory, `E1_BUNDLE_DIR` to the mounted bundle path (or `/kaggle/input/arc3-e1-bundle`), and `E1_WHEEL_DIR` to the wheel directory if the image does not already provide those packages. The notebook installs only attached wheels with `pip --no-index`.
4. Run the cells in order. The vLLM launcher uses 32,768 context, 0.92 GPU-memory utilization, prefix caching, Qwen reasoning parsing, and version-detects the chat-template kwargs flag. The benchmark cell then runs the existing agent; it does not use Duck's prompt, image path, Python REPL, tool loop, segmentation, batching policy, or action strategy.
5. Download `e1_results.json`, `submission.parquet`, `e1_traces/`, and the vLLM log. Treat `official_scorecard` from the installed toolkit as the score authority. For a later full public run, set `E1_GAME_SET=full` explicitly rather than changing the default subset.

### E1 reporting schema

The traces expose first mechanics/object/goal hypotheses, actions before the first effect or goal, plans generated/executed/aborted, prediction checks/mismatches/replans, no-ops, repeated state/action pairs, level completion, model calls, input/output/reasoning tokens, latency, progress events, and official scorecard fields. The result should be reported as a three-way table (V1.1, H2+H3, E1) and must record the exact checkpoint, toolkit versions, accelerator, action/token budgets, and any E0 budget mismatch before drawing a conclusion.

### Deliberately deferred to E2

Duck's prompt and world-model prose, Python REPL/tool calls, multimodal image representation, tool-loop/action batching, segmentation implementation, ACTION6 coordinate-ranking policy, and any Duck-specific action strategy remain out of E1. A possible `Duck + prediction/falsification/replan` hybrid is a design note only until E0 and E1 traces are available.
