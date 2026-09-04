# E0 / E1 fairness audit

Research question: keep model infrastructure as constant as the evidence
allows, while treating agent architecture as the experimental variable.

## Final parameter table

| Parameter | E0 (confirmed/reconstructed) | E1 (locked target) | Fairness status |
|---|---|---|---|
| Checkpoint | `/kaggle/input/qwen3-8-27b-fp8-repacked-v1/pytorch/hf-fp8/1` (`Qwen/Qwen3.8-27B-FP8`) | Same exact path by default | IDENTICAL checkpoint; E0 revision hash remains UNKNOWN |
| vLLM config | Local vLLM; TP=1, dtype auto, GPU utilization 0.92, prefix cache, trust remote code; wheel version UNKNOWN | Launcher sets the same flags; runtime version/config saved in `e1_model_metadata.json` | Matched flags; versions are captured, not guessed |
| Context length | 32768 | 32768 | IDENTICAL |
| Temperature | 0.6 | 0.6 | IDENTICAL |
| top_p | 0.95 | 0.95 | IDENTICAL |
| top_k | 20 | 20 | IDENTICAL |
| max_tokens | Analyzer `max_output=0` (server default UNKNOWN) | 2048 request cap | UNKNOWN / explicit cap difference; do not claim exact parity |
| Thinking | Enabled | `E1_ENABLE_THINKING=true` | IDENTICAL intent |
| preserve_thinking | `true` in public config | `E1_PRESERVE_THINKING=true` and request kwargs | IDENTICAL intent |
| Reasoning parser | `qwen3` | `E1_REASONING_PARSER=qwen3` | IDENTICAL |

The E1 metadata file also records served model name, config
`architectures`/`model_type`, quantization fields, tokenizer path, and vLLM
version. Missing fields are written as `UNKNOWN`.

## IDENTICAL (or intentionally matched)

| Dimension | Evidence |
|---|---|
| Local OpenAI-compatible boundary | E0 public config and E1 launcher use localhost vLLM `/v1/chat/completions`. |
| Context length | `32768` in public Duck config and E1 launcher. |
| Temperature / top-p / top-k | `0.6 / 0.95 / 20` in E0 public config and E1 defaults. |
| Thinking | Enabled in both configs; E1 sends Qwen `chat_template_kwargs`. |
| Reasoning parser | `qwen3`. |
| Tensor parallelism | `1`. |
| GPU memory utilization | `0.92`. |
| Dtype | `auto`. |
| Prefix caching | Enabled. |
| Trust remote code | `true` in the serving target. |

These are matches to the **publicly observable target configuration**. They do
not prove that the hidden E0 process used the same wheel, checkpoint, or
runtime image.

## INTENTIONALLY DIFFERENT (the architecture variable)

- Duck analyzer/tool loop versus our `ReasoningAgent` decision loop.
- Duck prompt and ASCII/segmentation context versus our raw-grid plus
  structured-object/transition context.
- Duck Python REPL versus no model-facing Python tool in E1.
- Duck optional image/multimodal representation versus E1 text/grid only.
- Duck model-driven action batching versus E1's bounded parsed plan queue.
- Duck world-model prose/history policy versus our `GameMemory` and bounded
  transition records.
- Duck implicit planning and harness boundaries versus E1's explicit
  goal/plan/prediction/replan contract.
- Duck-specific action strategy versus the unchanged E1 action validation and
  ACTION6 handling.

These differences are not fairness defects; importing them would change the
research question.

## ACCIDENTALLY DIFFERENT / UNKNOWN

| Dimension | Current status | Required action |
|---|---|---|
| Exact checkpoint identity | Current E0 and E1 target: `/kaggle/input/qwen3-8-27b-fp8-repacked-v1/pytorch/hf-fp8/1` (`Qwen/Qwen3.8-27B-FP8`) | **Resolved from user-confirmed Kaggle model input.** |
| Served model name | E1 explicit; E0 not recorded | E1 captures `/v1/models`; E0 remains UNKNOWN. |
| Model revision | Not recorded | Capture revision/hash. |
| Tokenizer | Not recorded | Capture tokenizer path/revision/config. |
| Quantization metadata | `FP8` only appears in names | Inspect model config/weights. |
| vLLM wheel/version and runtime image | Not recorded in E0 artifact | Record exact package versions and accelerator image. |
| Analyzer max output | E0 `max_output=0` delegates to server; E1 sends `2048` | Record E0 server default or label E1 as a cap difference. |
| Tool-step behavior | E0 public config `0` overrides source default `12` | Keep documented; do not silently set E1 tool steps. |
| Pass/action/token budgets | E0 benchmark `n_passes=1`; per-game caps absent | Record observed E0 protocol; do not equalize after the fact. |
| Wallclock | E0 per-game values cluster near ~7,920 s | Treat as censored; E1 runtime must be reported separately. |
| Concurrency | Not present in benchmark JSON | Capture worker count if E0 metadata becomes available. |

## Current verdict

The checkpoint identity is now locked to the user-confirmed current E0 model,
and E1 has no Qwen3.6 fallback.  E1 startup captures the remaining runtime
metadata (config architecture, tokenizer, quantization, vLLM version, and
served-model response); absent fields remain `UNKNOWN`.  These metadata gaps
are reporting caveats, not reasons to substitute a different checkpoint.

## Scope guard

No Duck prompt, Python tool, multimodal input, ACTION6 heuristic, learned model,
RL, or planner rewrite is added to E1 as part of this audit.
