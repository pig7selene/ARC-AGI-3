# ARC-AGI-3 V1 research notes

Research was checked on 2026-09-01 against the live [ARC-AGI-3 page](https://arcprize.org/arc-agi/3) and [official docs](https://docs.arcprize.org/). The docs currently specify `pip install arc-agi`, `import arc_agi`, `from arcengine import GameAction`, `arc = arc_agi.Arcade()`, and `arc.make("ls20")`; the toolkit can run headless at roughly +2K FPS and exposes `arc.get_scorecard()`. This workspace now has toolkit `arc-agi 0.9.9` / `arcengine 0.9.3` installed in `.venv`; the adapter still probes runtime fields instead of assuming a historical API.

## Benchmark mechanics

ARC-AGI-3 is an interactive benchmark: an agent receives a rendered grid/frame, a game/level state and the currently legal action subset, then acts and observes the resulting transition. Actions are named rather than semantically fixed; a game may expose RESET, ACTION1–ACTION5, ACTION6(x,y), and ACTION7 in different subsets. Mechanics, controllable entities and win conditions are hidden.

The principal resources are [ARC Prize ARC-AGI-3](https://arcprize.org/arc-agi/3), [official ARC-AGI-3 repository](https://github.com/arcprize/ARC-AGI-3), [ARC-AGI-3 Agents](https://github.com/arcprize/ARC-AGI-3-Agents), the [arc-agi toolkit](https://github.com/arcprize/arc-agi), the [toolkit overview](https://docs.arcprize.org/toolkit/overview), and the [technical report](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf). Official scoring and replay/evaluation code should be imported from the installed toolkit for a real submission; this project never re-implements leaderboard scoring.

The current toolkit changelog records important scoring details: an individual level score is squared; game-level averages are weighted by 1-indexed level; and toolkit 0.9.7 raised the per-level cap to 115% while capping a game by the weighted maximum of completed levels. These details are why `arc3.eval.metrics` is explicitly a diagnostic summary, not a replacement scorer.

## What strong baselines teach us

The Duck harness (Tufa Labs), Just Explore, Stochastic Goose and Milestone-1 solutions converge on a few useful patterns: an LLM is most effective as a general policy when given a compact structured observation; a Python/REPL surface enables ad-hoc BFS and geometry; connected components and frame diffs turn pixels into evidence; a bounded recent trace plus durable game memory prevents context blow-up. Search-based agents add learned frame-change prediction and information-directed exploration, but add model/training complexity.

The official `ARC-AGI-3-Agents` repository is a useful plumbing reference (MIT licensed): its current README runs `uv run main.py --agent=random --game=ls20`, and its changelog notes breaking `FrameData` field renames. The current toolkit quickstart says an API key is optional (anonymous key fallback), but registering one unlocks public games at release. These observations reinforce runtime API probing and keeping our own trace schema explicit.

These are architectural lessons, not game solutions. V1 intentionally contains no public-game rules or fixed ACTION-direction mapping.

## Constraints and evaluation split

Kaggle evaluation is offline: code, model weights and dependencies must be packaged in the notebook/runtime, with finite runtime and accelerator memory. The current toolkit's Competition Mode is required for the unverified leaderboard and (per the live docs) uses API environments, scores all available environments, permits only level resets (game resets become level resets), allows one `make` per environment and one scorecard; `get_scorecard` is unavailable in-flight. Public games are useful for debugging only; private games are unseen, so hardcoding layouts or action semantics is invalid. The `arc3.env.protocol.adapt_environment` shim accepts common reset/step return shapes and keeps the core independent from toolkit releases.

## V1 decisions

V1 uses pure-Python dataclasses and a small pipeline: canonical Observation → connected-component Perception → FrameDiff → bounded GameMemory/Hypotheses → information-efficient Exploration/guards → optional generic BFS planner → strict Decision parser → action and JSONL trace. `HeuristicBackend` makes the whole stack executable without a model. `OpenAICompatibleBackend` targets a local vLLM/OpenAI-compatible server for offline inference; no remote provider is required by the architecture.

## Model integration note

The environment here has no GPU or Kaggle runtime, so I did not make an unverified Qwen-27B/Gemma checkpoint claim. The backend accepts any local OpenAI-compatible server and keeps model choice/configuration outside core code; a Kaggle experiment should measure quantized Qwen/Gemma VRAM, startup latency, context length and per-game action efficiency on the current accelerator before selecting weights.

## Deliberately postponed

V2/V3 candidates are learned transition prediction (Stochastic Goose style), uncertainty-calibrated Bayesian exploration, richer object tracking/containment graphs, beam/MCTS over compact world models, visual-language model fine-tuning, parallel rollouts and GPU-optimized inference. PPO/RL is postponed until instrumentation demonstrates a stable state/action abstraction.
