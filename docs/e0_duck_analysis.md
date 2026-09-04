# E0 Duck analysis

Research date: 2026-09-01. This document separates observations from the public Duck harness from hypotheses about why its reported E0 run is effective. It is an infrastructure study only; Duck agent mechanisms are not copied into E1.

## FACT — public implementation

- The public repository describes The Duck as a self-contained ARC-AGI-3 harness built around TAAF, with a local OpenAI-compatible vLLM server or OpenRouter, a Python package, structured run artifacts, and a viewer.
- The current E0 high-score route is confirmed as Kaggle model `foysalemonshanto/qwen3-8-27b-fp8-repacked-v1`, mounted at `/kaggle/input/qwen3-8-27b-fp8-repacked-v1/pytorch/hf-fp8/1`, corresponding to `Qwen/Qwen3.8-27B-FP8`. The older public `vrfai/Qwen3.6-27B-FP8` entry is historical config only. The public Duck inference values are a 32,768-token context, localhost `http://127.0.0.1:1234/v1`, `temperature=0.6`, `top_p=0.95`, `top_k=20`, and thinking enabled.
- Its vLLM settings include `gpu_memory_utilization=0.92`, tensor parallel size 1, prefix caching, `reasoning_parser=qwen3`, `tool_call_parser=qwen3_coder`, and `default_chat_template_kwargs.preserve_thinking=true`. The public Makefile/config supports a two-GPU Slurm deployment and a Kaggle launcher, but model serving remains a local HTTP endpoint.
- Duck's analyzer is a direct OpenAI-compatible tool-calling client. The model receives current frame state, valid actions, history, transitions, and the last action result, then calls one ephemeral `python` tool. The tool can call `action(actions)` with one or many actions; the harness stops the batch on terminal state, level completion, game over, or run completion.
- The Python tool exposes ASCII and segmentation views plus history/transition metadata. The documented preferred view is segmentation; the raw numeric grid is not available inside the Python sandbox. The harness can optionally attach a current-grid image through its multimodal context configuration.
- The analyzer keeps a persistent bounded conversation, summarizes a compact working world model (world/goal/action model, recent findings, open questions, plan, cross-level notes), trims history to a context budget, and resets/summarizes knowledge at terminal transitions.
- The public config sets `tool_steps=0` for its shown run, while the source default is 12. In the source, `tool_steps <= 0` is normalized to `None`, so this means **no fixed tool-step limit**, not that the Python tool is disabled.
- Usage and timing are tracked from the OpenAI-compatible response; generated-token accounting accepts completion/output/generated token fields. Request logs are optional and disabled in the public default config.
- The public repository and README describe a 25-game × 20-pass example run and provide a Kaggle notebook that packages the source, installs the runtime, starts local inference, runs games, and writes viewer/scoring artifacts.

## FACT — model-infrastructure implications for E1

- The reusable boundary is the local `/v1/chat/completions` server, not Duck's Python tool or prompt. E1 can use the same vLLM endpoint and checkpoint while sending our own `ReasoningAgent` context.
- Qwen/vLLM compatibility requires accepting a base URL ending in `/v1` as well as a full `/chat/completions` URL, forwarding `top_k` and Qwen thinking settings when supported, and tolerating `reasoning_content`/`<think>` wrappers without storing chain-of-thought.
- A Kaggle runtime must keep internet disabled during evaluation and rely on pre-provisioned wheels/model datasets. The local server should be health-checked before any environment action is spent.

## HYPOTHESIS — why Duck can outperform our heuristic

- H1: The primary difference is model semantic capacity plus a persistent world-model summary, not merely lower-level frame parsing.
- H2: Batching several actions inside one model turn reduces model-call overhead and allows the model to test a short trajectory, but can waste environment actions when predictions are wrong. Our E1 intentionally does not enable Duck batching so this factor stays out of the first comparison.
- H3: The Python workspace is likely most valuable for geometry/path calculations and compact evidence queries after a goal is already plausible; it is not evidence that Python alone creates goal inference.
- H4: The multimodal current-grid option may explain performance on dense visual scenes, but adding it in E1 would confound model substitution with representation modality. It is reserved for a later H4 ablation.
- H5: The Duck's repeated model calls are driven by tool-call/terminal boundaries and context summarization, whereas our E1 model call is one decision per environment action unless it emits our bounded plan. This is an intentional architectural difference to measure, not an accidental claim of parity.
- H6: Prediction/falsification/replan in our agent may reduce Duck-like repeated-action failures, but only a trace with a real model can show whether our stricter contract helps or over-constrains useful exploration.

## What E1 reuses

- Qwen checkpoint/server model infrastructure, local OpenAI-compatible transport, Qwen-compatible generation knobs, and health-check/packaging ideas.

## What E1 deliberately does not reuse

- Duck prompt, Python REPL, tool-call loop, multimodal image representation, segmentation implementation, action batching policy, world-model prose format, or action strategy.

## Sources

- [Tufalabs/duck-harness](https://github.com/Tufalabs/duck-harness)
- [Duck ARC3-Inference README](https://github.com/Tufalabs/duck-harness/blob/main/ARC3-Inference/README.md)
- [Duck inference config](https://github.com/Tufalabs/duck-harness/blob/main/ARC3-Inference/configs/inference.json)
- [Duck OpenAI compatibility helpers](https://github.com/Tufalabs/duck-harness/blob/main/ARC3-Inference/inference/utils/openai_compat.py)
- [Duck tool agent](https://github.com/Tufalabs/duck-harness/blob/main/ARC3-Inference/inference/agent/tool_agent.py)
- [Duck Kaggle notebook](https://github.com/Tufalabs/duck-harness/blob/main/taaf-duck-harness-kaggle-share.ipynb)
