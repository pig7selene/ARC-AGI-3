# E1 Preparation Report — Our Agent + Qwen3.8-27B-FP8

Status: the E1 infrastructure and fixed benchmark are ready. No Qwen weights were downloaded and no real model request was sent in this session.

## E0 Duck architecture

The public Duck harness is a local OpenAI-compatible client around a vLLM model server. Its analyzer maintains a bounded conversation and a compact world-model summary, presents frame/history/transition information, and can call an ephemeral Python tool. The tool exposes ASCII/segmentation views and can submit one or more `action(...)` calls; the harness stops a batch at terminal or level boundaries. A current-grid image can be attached in configurations that enable multimodal input. Public artifacts track usage, timing, transitions, and replay data.

The current E0 checkpoint is confirmed as Kaggle model
`foysalemonshanto/qwen3-8-27b-fp8-repacked-v1`, mounted at
`/kaggle/input/qwen3-8-27b-fp8-repacked-v1/pytorch/hf-fp8/1` and corresponding
to `Qwen/Qwen3.8-27B-FP8`.  The older public Qwen3.6 config is historical only
and is not an E1 fallback.  E1 startup now records the remaining checkpoint
and serving metadata without guessing.

The following are facts about the public harness/config. Explanations for its score, repeated actions, and history entries are hypotheses until the E0 trace is available.

## Important differences

| Dimension | Duck E0 | Our E1 |
|---|---|---|
| Representation | Frame/history plus optional image; Python view is ASCII/segmentation | Raw grid, dimensions/hash, connected components, color/symmetry, frame diff, bounded transitions, mechanics/action outcomes, valid actions |
| Model boundary | Local vLLM/OpenAI-compatible or OpenRouter | Same local vLLM/OpenAI-compatible boundary; only `HeuristicBackend` is replaced |
| Reasoning loop | Persistent analyzer conversation and tool-call turns | One structured decision per environment action, or an optional bounded three-step plan |
| Memory | World/goal/action summary with history trimming and cross-level notes | Per-game `GameMemory`, bounded recent transitions, hypotheses, goals, effects, visited states, and replans |
| Tools | Ephemeral Python REPL and `action(actions)` | No model-facing tool in E1; existing package utilities stay internal |
| Planning | Tool-driven action batches and committed trajectories | Existing short plan queue, maximum three steps, prediction checked after every step |
| Action queue | May execute multiple actions after one inference | Queue is only populated by our parsed bounded plan; it is cleared on mismatch |
| Prediction | Harness/tool loop provides observations; no E1 assumption about a typed contract | Existing `ActionExpectation` contract verifies visibility, changed-cell bounds, state hash, or diff text |
| Replanning | New analyzer/tool turn at harness-defined boundaries | Structured mismatch records a replan and forces a fresh policy decision |
| Context | Persistent bounded conversation and world-model summary | `build_context()` sends raw + structured state and compact memory; hidden chain-of-thought is not stored |
| ACTION6 | Duck-specific coordinate selection inside its tool/action policy | Existing `Action6(x,y)` validation and guards; no new ranking or coordinate strategy |

Duck-specific prompt, Python workspace, image representation, segmentation implementation, action batching, and action strategy are deliberately not imported into E1.

## Qwen integration

The path is:

```text
Qwen3.8-27B-FP8 checkpoint
        ↓
local vLLM server (127.0.0.1:1234)
        ↓
OpenAI-compatible /v1/chat/completions
        ↓
OpenAICompatibleBackend
        ↓
existing ReasoningAgent + prediction/replan contract
```

`OpenAICompatibleBackend` accepts either a `/v1` base URL or a full completion URL, forwards `temperature`, `top_p`, `top_k`, `max_tokens`, and Qwen `chat_template_kwargs`, and records latency/usage without retaining chain-of-thought. Its parser handles plain JSON, markdown-fenced JSON, `<think>...</think>` wrappers, content lists, and `reasoning_content` fallbacks.

`kaggle/e1_qwen/serve_qwen.sh` starts vLLM with the model path supplied by `E1_MODEL_ID`, 32,768 context, tensor parallel size 1, automatic dtype, 0.92 GPU-memory utilization, prefix caching, and `qwen3` reasoning parsing. The script detects the installed vLLM spelling for the optional default chat-template kwargs flag; the request itself always carries the kwargs.

## Files changed for E1

- `arc3/llm/backend.py` — local OpenAI-compatible transport and Qwen-compatible response/usage handling.
- `arc3/agents/reasoning_agent.py`, `arc3/llm/parser.py`, `arc3/llm/prompt.py`, `arc3/eval/diagnostics.py`, `arc3/eval/metrics.py` — existing semantic/prediction/replan instrumentation used unchanged by E1.
- `kaggle/e1_qwen/e1_runner.py` — fixed-subset runner, localhost health gate, trace metrics parquet, official scorecard capture, and opt-in full-game selection.
- `kaggle/e1_qwen/serve_qwen.sh` — offline vLLM launcher.
- `kaggle/e1_qwen/e1_qwen.ipynb` — bundle discovery, optional offline wheel install, server start, benchmark, and artifact inspection.
- `kaggle/e1_qwen/e1_config.json` — locked Qwen3.8 checkpoint/config record.
- `kaggle/e1_qwen/build_bundle.py` — source-bundle builder that refuses accidental overwrite.
- `experiments/v2/validate_model_schema.py` — two isolated schema calls before burning environment actions.
- `tests/test_llm_backend.py`, `tests/test_e1_runner.py` — transport/parser and runner-selection regression coverage.
- `docs/e0_duck_analysis.md`, `docs/experiments.md` — E0 facts/hypotheses, controls, and E1 protocol.

## Tests and checks

The current local checks are:

- `27/27` unit tests pass (`.venv/bin/python -m unittest discover -s tests -v`).
- Notebook and config both pass `python -m json.tool` validation.
- `serve_qwen.sh` passes `bash -n` syntax validation.
- `build_bundle.py` creates an isolated bundle; the runner imports successfully from that bundle outside the repository checkout.
- The model schema validator is ready but was not run because no explicitly authorized local endpoint is available.

## E1 benchmark design

### Small fixed subset

The default is exactly:

```text
cd82-fb555c5d
ls20-9607627b
lf52-271a04aa
```

Each game uses toolkit `OFFLINE`, seed `0`, and 30 environment actions. Representation, agent architecture, action validation, prediction/replan contract, and trace schema are held constant. E1 uses the published Duck sampler (`0.6/0.95/20`) and a 2,048-token response cap; the exact checkpoint, toolkit versions, accelerator, startup time, model calls, tokens, and latency must be recorded.

### Full public run

After the small subset, set `E1_GAME_SET=full` (or provide an explicit comma-separated `E1_GAME_IDS`) to use the toolkit registry. The runner does not silently expand the default. Duck's actual pass/action/token budget should be recorded as observed; do not change Duck to make the numbers look equal.

Every E1 trace records first mechanics/object/goal hypotheses, actions before the first effect/goal, plans generated/executed/aborted, prediction checks/mismatches/replans, no-ops, repeated action/state pairs, level completion, model calls, input/output/reasoning tokens, latency, progress events, and official scorecard fields.

## Expected research value

E1 isolates whether Qwen semantic policy capacity can connect the existing interface:

```text
transition → semantics → goal → plan → execution
```

If goals/plans/progress appear, the next question is whether our verification contract improves action efficiency relative to Duck's loop. If the model still produces no goals/plans, the evidence points first to a representation/interface limitation rather than a missing Duck mechanism. If plans are produced but fail, traces separate model-output, representation, prediction, and execution failures.

## Main risks

- The E0 model path is resolved; the launcher/runner records the actual path,
  served model name, config architecture/model type, quantization, tokenizer,
  vLLM version, and server response at startup.
- vLLM startup can fail from a wheel/driver mismatch, unsupported reasoning parser, or chat-template flag spelling; the launcher has a version check, and the notebook captures `e1_vllm.log`.
- A 32K context may still be exceeded by dense raw grids and object lists; compact memory is bounded, but token counts must be inspected in traces.
- Qwen may emit valid-looking prose, fenced JSON, or reasoning wrappers; the parser handles known wrappers, while parse failures remain explicit metrics.
- Model latency can dominate a 30-action budget; health-checking prevents spending environment actions before the server is ready, but runtime and per-call latency must be reported.
- `submission.parquet` is a trace-metrics artifact. It is not claimed to be the official ARC competition submission format; the installed toolkit scorecard is authoritative.

## Kaggle instructions

1. Build the bundle locally: `E1_BUNDLE_DIR=/tmp/arc3-e1-bundle .venv/bin/python kaggle/e1_qwen/build_bundle.py`, then upload the directory as a private Kaggle dataset.
2. Attach the exact offline Qwen model dataset and offline wheels for vLLM plus `arc-agi`/`arcengine`. Disable Internet and choose the RTX Pro 6000 accelerator.
3. Open `kaggle/e1_qwen/e1_qwen.ipynb`. Set `E1_MODEL_ID` to the mounted model directory. Set `E1_BUNDLE_DIR` to the mounted bundle path (the notebook also checks `/kaggle/input/arc3-e1-bundle`) and set `E1_WHEEL_DIR` if packages are not already present.
4. Run cells in order. The wheel cell uses `pip install --no-index`; the server cell starts local vLLM; the benchmark cell imports the normal package runner. No model download or external API call is performed.
5. Download `e1_results.json`, `submission.parquet`, `e1_traces/`, and `e1_vllm.log`. Compare V1.1, H2+H3, and E1 in one table, including first-goal step, goal quality, plan/prediction metrics, completion, model calls/tokens/latency, and progress events.

For a non-Kaggle authorized local server, validate the response contract first:

```bash
ARC3_MODEL_BASE_URL=http://127.0.0.1:1234/v1 \
ARC3_MODEL=<exact-qwen-checkpoint> \
  .venv/bin/python -m experiments.v2.validate_model_schema
```

Only after schema validation should a real E1 run be started. Never place API keys in source, notebooks, traces, or result files.

## Deliberately deferred to E2

The following remain design notes only: Duck prompt/world-model prose, Python REPL/tool calls, multimodal image input, tool-loop/action batching, Duck segmentation, new ACTION6 coordinate ranking, learned transition models, RL, and any game-specific action strategy. A `Duck + prediction/falsification/replan` hybrid is not implemented until E0 and E1 traces are available.
