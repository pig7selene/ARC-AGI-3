# E0 inference configuration reconstruction

This is a source/config audit, not a claim that every value was captured from
the hidden Kaggle process.  Values are marked as **public config**, **source
default**, **benchmark observation**, or **UNKNOWN**.

| Setting | Reconstructed value | Evidence / caveat |
|---|---|---|
| Model | `/kaggle/input/qwen3-8-27b-fp8-repacked-v1/pytorch/hf-fp8/1` (`Qwen/Qwen3.8-27B-FP8`) | User-confirmed current E0 checkpoint; the older `vrfai/Qwen3.6-27B-FP8` public config is historical only. |
| Endpoint | `http://127.0.0.1:1234/v1` | Public config; local OpenAI-compatible boundary. |
| Context window / max model length | `32768` | Public config. |
| Temperature | `0.6` | Public analyzer config. |
| Top-p | `0.95` | Public analyzer config. |
| Top-k | `20` | Public analyzer config. |
| Analyzer max output | `0` = server default | Public config. The resulting server default token cap was not captured. |
| Chat-probing max tokens | `2048` | Separate public chat-probing section; do not substitute it for analyzer output. |
| Thinking / reasoning | Enabled | Public analyzer and chat-probing config. |
| Chat template | Qwen template with `preserve_thinking=true` kwargs | Exact template file/name is UNKNOWN; public config records the kwargs. |
| Reasoning parser | `qwen3` | Public vLLM config. |
| Tool-call parser | `qwen3_coder` | Public vLLM config. |
| Tensor parallelism | `1` | Public server config. |
| GPU memory utilization | `0.92` | Public server config. |
| Dtype | `auto` | Public server config. |
| Prefix caching | Enabled | Public server config. |
| Trust remote code | `true` | Public server config. |
| Analyzer timeout | `120 s` | Public analyzer config. |
| Python-tool timeout | `30 s` | Public analyzer config. |
| Tool output token budget | `1024` | Public analyzer config. |
| Tool steps | `0` in public config; source default `12` | Source normalizes `tool_steps <= 0` to `None`, so the analyzer is not stopped by a fixed tool-step count. |
| Passes | `n_passes=1` in local benchmark | Benchmark observation. Public example environment config uses `20`; this is a real protocol difference. |
| Per-game action budget | **UNKNOWN** | `base_actions_per_level` is game metadata, not an observed E0 cap. History length is an outcome. |
| Per-game token budget | **UNKNOWN** | Benchmark totals and `solver_note` totals exist, but no configured per-game cap is recorded. |
| Per-game wallclock budget | **UNKNOWN; likely censored near ~7,920 s** | Many `final_wallclock_seconds` values cluster near the same high value. Treat as a benchmark cap/censoring signal, not independent game runtime. |
| Concurrent game execution | **UNKNOWN** | The benchmark has aggregate start/end times but no worker/concurrency metadata. |
| Request logs | Disabled by public default | No request IDs or transcript boundary data are present in the local benchmark. |
| Intermediate states | `record_intermediate_states=true` | Field is present, but the supplied JSON does not contain the corresponding frame transcript. |

## Configuration versus benchmark discrepancies

1. **Historical config:** the older public Duck config says Qwen3.6, but the
   current E0 checkpoint is user-confirmed as Qwen3.8 Repacked.  E1 is locked
   to the latter and has no Qwen3.6 fallback.
2. **Tool steps:** public config overrides the source default (`0` versus
   `12`), changing how long the analyzer may continue its tool loop.
3. **Passes:** the local benchmark records `n_passes=1`, while the public
   example environment configuration advertises `20`.
4. **Output cap:** analyzer `max_output=0` delegates to the server; E1's
   explicit `2048` cap is therefore not proven identical.
5. **Timing:** repeated near-7,920-second per-game values suggest a global or
   per-run wallclock censoring point.  The value must not be used as a direct
   E0-vs-E1 runtime comparison.

## E1 parity target

E1 intentionally copies the confirmed checkpoint and the public sampler/
serving values that are observable (`0.6/0.95/20`, thinking enabled, 32K
context, TP=1, dtype auto, 0.92 GPU utilization, prefix caching, Qwen3
parser).  Remaining unavailable fields (revision, tokenizer details, vLLM
version, and analyzer output-cap behavior) are captured as `UNKNOWN` rather
than guessed.

## Sources

- [Duck inference config](https://github.com/Tufalabs/duck-harness/blob/main/ARC3-Inference/configs/inference.json)
- [Duck ToolAgent source](https://github.com/Tufalabs/duck-harness/blob/main/ARC3-Inference/inference/agent/tool_agent.py)
- [Duck solver source](https://github.com/Tufalabs/duck-harness/blob/main/ARC3-Inference/inference/framework/solver.py)
- Local artifact: `benchmark.json` supplied outside this repository
