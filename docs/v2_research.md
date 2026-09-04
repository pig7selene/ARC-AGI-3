# ARC-AGI-3 V2 research and diagnosis

Research date: 2026-09-01. Sources were read from the live ARC Prize docs, the official agents/benchmarking repositories, Tufa Labs' Duck harness, and the public Tycho and Retrodict repositories. Claims below are labelled so that public score claims are not mixed with our own observations.

## Current facts (FACT)

- The current toolkit is `arc-agi 0.9.9` / `arcengine 0.9.3`. The official benchmark harness accepts structured actions, trims serialized reasoning/output to 16 KB, and provides repeatable scorecards/replays.
- The official LLM templates include a function-calling agent with a 10-message conversation limit, a fast no-observation variant, a reasoning model template, and a guided template explicitly marked as non-generalizing. This is a useful API reference, not evidence that a particular provider is best for our agent.
- The official ARC Prize 2026 starter targets Python 3.12, local development with the same `arc-agi` package, a generated Kaggle notebook, and T4/P100/RTX6000 accelerator options. Competition runs are code submissions executed twice; accelerated sessions have internet disabled.
- The Duck repository describes a Milestone-1 winning tool-using harness: Python analysis, structured logs/diffs/components, short information-seeking probes followed by committed plans, local vLLM/OpenRouter support, and a replay viewer. Its example run is 25 games × 20 passes.
- Retrodict reports 99.86% mean RHAE across all 25 public games (all 183 levels) using GPT-5.6-sol at max reasoning. Its public implementation uses a text log as the ground truth, free counterfactual retrodiction over that log, per-action expected-cell verification, a plan queue that halts on prediction mismatch, playbook memory across context resets, and one-time visual priming.
- Tycho reports 100.00% public RHAE with an actor-controlled executable world-model builder and verifier, and 79.07% for a no-world-model policy in its own ablations. Its README exposes a multimodal observe → executable State/transition/render/outcome model → plan → verify loop and a network-disabled Python containment sandbox.
- The live official docs' benchmarking harness is a useful reproducibility reference: same game/config, model/provider config, scorecard and replay artifacts. It does not imply that all community leaderboard claims use identical cost or rerun rules.
- Reki, forge and Stochastic Goose were searched as named community methods, but their current source/notebook artifacts were not available through the accessible public pages during this pass. They are therefore treated as research leads, not asserted facts.

## Our observations (OBSERVATION)

The following are measured from `traces/reasoning_real/` and `traces/plumbing_real/`, not inferred from the game source:

1. On `cd82-fb555c5d`, ACTION1/2 effects become visible immediately. The reasoning agent still spends 30 steps cycling probes; 10 actions are no-ops and 10 frame hashes repeat. No plan is created.
2. On `ls20-9607627b`, ACTION1 causes large repeated multi-object/frame changes (including animation-like appeared/disappeared components). The agent repeats ACTION1 for six steps before trying ACTION2, but never records a goal hypothesis or planner attempt.
3. On `lf52-271a04aa`, ACTION1 changes the frame on every step while the agent repeats it for all 30 steps. The first frame contains 59 segmented components, including large background/HUD-like regions; the component abstraction is too noisy to identify the task by itself.
4. `GameMemory.action_effects` is populated for movement-like diffs, but `HeuristicBackend` only selects the first action not marked no-op. It does not synthesize goal hypotheses, use effect vectors to choose a target, or call the planner.
5. At the 20/30 action budgets, all three real games score 0.0 and clear no levels. No-op rate is only about 12%, so “random invalid actions” is not the dominant failure. The dominant signal is semantic stagnation after an effect is already observable.
6. Current traces contain compact objects and diffs but no initial visual rendering/description, no predicted post-action frame, no explicit hypothesis being tested per action, and no plan verification event.

## Diagnosis

The infrastructure is sufficient for iteration. The first research bottleneck is a missing semantic control loop: observe an effect → state what it means → infer a falsifiable objective → predict a consequence → commit a short plan → verify and re-plan. Our current abstraction is not necessarily wrong, but it is incomplete and can be misleading when animation/HUD regions dominate. The first intervention should therefore be a minimum Retrodict-style contract (explicit prediction + post-step verification + counterfactual reuse of the existing trace) while preserving raw grid access and existing instrumentation.

## Candidate hypotheses

| ID | Hypothesis | Evidence | Expected benefit | Cost | Risk | Falsification |
|---|---|---|---|---|---|---|
| H1 | Semantic visual reasoning, not exploration count, is the main bottleneck. | Effects are often found quickly; score remains 0 and goals stay empty. | Better control/goal identification and fewer purposeless repeats. | Medium: multimodal/visual priming or stronger backend. | Model cost/latency; visual input may not help. | On a fixed backend and budget, adding visual priming does not improve goal/plan metrics or solved levels. |
| H2 | Counterfactual retrodiction over recorded frames can replace many live probes. | Retrodict and Duck both validate hypotheses against logs before acting; our memory does not. | Lower action cost and fewer repeated semantic probes. | Low–medium: replayable transition predicates and prediction fields. | A weak predicate can create false confidence. | Prediction-verified policy has no lower no-op/repetition rate or is less accurate than single-step probing. |
| H3 | Commit-and-verify short plan queues are more valuable than reactive one-action calls. | Retrodict reports expected-cell plans and halts on first mismatch; our planner is never invoked. | More progress per action and explicit re-planning. | Medium. | Wrong plan can waste actions unless horizon is short. | Queue execution does not raise effective actions/level or increases recovery cost. |
| H4 | Dense component summaries can obscure task semantics; dual raw+structured context is required. | LF52 starts with 59 components and animated regions; current policy sees no rendered visual priming. | Better object/goal selection without deleting useful geometry tools. | Low: add saliency/thumbnail/raw-frame channel. | More context tokens. | Raw/visual channel fails to improve goal hypothesis quality on dense-game subset. |
| H5 | A typed action-effect table is more useful than generic confidence hypotheses. | Effects are captured but not mapped to subsequent behavior. | Stable control identification and action selection. | Low. | Overfitting effect labels to movement games. | Effect table does not change action diversity, planning attempts or completion. |

## First experiment selection

Highest expected value is H2 + H3 as one narrowly scoped intervention: add an explicit per-action predicted diff/expected state, reuse the existing trace to check whether a candidate mechanic is already falsified, and execute at most a short verified queue before re-observing. This directly targets the measurable failure (semantic stagnation) without committing to a model, RL, or game-specific rule. H1/H4 remain the next experiment if a real model endpoint is available; H5 is an enabling ablation rather than the main score hypothesis.

## What is deliberately not being adopted yet

No PPO/GRPO/SFT, learned transition network, game-specific solver, public solution sequence, giant beam search, or provider lock-in. The current evidence supports better hypothesis testing and execution discipline first.

## Sources

- https://docs.arcprize.org/arc-prize-2026
- https://docs.arcprize.org/llm_agents
- https://docs.arcprize.org/benchmarking-agent
- https://github.com/arcprize/ARC-AGI-3-Agents
- https://github.com/Tufalabs/duck-harness
- https://github.com/NIMI-research/Tycho
- https://github.com/ryanbbrown/Retrodict
- https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter
