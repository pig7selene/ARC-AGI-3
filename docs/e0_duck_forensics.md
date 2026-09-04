# E0 Duck forensics

This reconstruction follows the public `ToolAgent` and `Solver` source.  It
does not infer hidden chain-of-thought from the benchmark.  The benchmark has
action/usage history but no model request IDs, tool transcripts, or serialized
intermediate frames, so case-level causal claims are intentionally limited.

## Control flow

```text
environment observation
        ↓
runtime state + bounded persistent history
        ↓
ToolAgent.analyze()
        ↓
OpenAI-compatible model inference
        ├─ no tool call → save response → follow-up inference
        └─ python tool call
                ↓
        ephemeral Python inspection
                ↓
        action(actions)
                ↓
        Solver._normalize_actions()
                ↓
        one or more environment steps
                ↓
        HistoryEntry + token accounting
                ↓
        terminal / level boundary or next analyzer turn
```

## Observation

- The analyzer builds a runtime state containing the current frame, previous
  frame/transition information, valid actions, history, and the last action
  result.
- The Python workspace exposes `current_frame`, `previous_frame`, `history`,
  `transitions`, `last_transition`, `valid_actions`, and `last_action_result`.
- The preferred textual views are ASCII and segmentation.  Raw numeric grid
  data is not exposed to the Python sandbox.  A current-grid image is an
  optional multimodal input in configurations that enable it; the supplied
  benchmark does not prove that the image path was enabled.

## Memory

- Conversation history is persistent across analyzer turns but is bounded by
  a context-budget calculation.  The source keeps roughly the most recent
  thirty assistant turns in the normal configuration, subject to the actual
  token estimate and reply reserve.
- A compact world-model summary carries world, goal, action, recent-finding,
  open-question, plan, and cross-level notes.  This is persistent memory, not
  an unbounded transcript.
- On level transition and terminal/game-over boundaries, the solver saves a
  summary and clears/reinitializes portions of the active world state before
  the next analyzer turn.  The exact retained prose is not recoverable from
  the supplied benchmark JSON.

## Tools

- There is one model-facing tool: `python`.
- Each Python invocation is ephemeral; variables do not form a persistent REPL
  state between tool calls.  Persistence comes from the analyzer conversation
  and summary, not the Python process.
- `action(actions)` accepts one action or a list.  Its callback enters the
  solver execution path, so Python is both an inspection surface and the way
  a model-generated batch reaches the environment.

## Planning

There is no separate symbolic planner in the public control path.  Planning is
implicit in model reasoning, Python calculations, and the optional list passed
to `action(actions)`.  A list is a committed short trajectory, not evidence of
a learned planner or a reusable game policy.

## Execution and level transitions

1. `analyze()` constructs the prompt and persistent history, estimates tokens,
   trims context, and calls `/chat/completions`.
2. A response without a tool call is saved and followed by another inference;
   the analyzer continues until it produces a tool call, a terminal/yield
   condition, or an execution boundary.
3. A Python tool call executes in the ephemeral sandbox.  If it invokes
   `action(...)`, the current analyzer turn ends after the callback has
   submitted the action or batch.
4. `_normalize_actions()` converts a scalar action or iterable to a canonical
   ordered list.  `step_env()` executes that list one environment step at a
   time through `_execute_action()`.
5. A batch is stopped when the environment reports terminal, game-over,
   level-completed, run-complete, or `done`.  The next analyzer turn receives
   the new observation and a transition summary.

## Why `generated_tokens=0` actions occur

The source accounts usage at the **model-inference boundary**, while the
history records **environment actions**.  When one Python call submits a list
of actions, the model generated one piece of Python/tool output, but the
solver may then execute several queued actions without another model call.
The token usage is attributed to the model-bearing action/turn; subsequent
queued environment entries naturally carry `generated_tokens=0`.  An
auto-reset or other deterministic solver-side continuation can likewise be an
environment step with no fresh generation.

Therefore the zero-token count is source-backed evidence that batching or
solver-side continuation is possible, not proof that every zero-token entry in
`benchmark.json` was queued.  The JSON lacks request IDs, batch IDs, and tool
boundaries, so it cannot distinguish queued actions from every other
zero-generation branch entry.

## What the supplied benchmark can and cannot show

It can show action counts, token totals, action IDs, adjacent repetition, and
the aggregate `levels_completed`/score fields.  It cannot show:

- the first action that changed the frame or caused a level completion;
- exact model-call count (non-zero-token entries are only a proxy);
- whether Python was used on a particular turn;
- the model's goal statement, plan, prediction, or rationale;
- the exact action-to-level boundary beyond the aggregate
  `actions_per_level` list;
- the true per-game wallclock when values are censored near the benchmark cap.

## Reusable versus risky mechanisms

The reusable mechanism visible in source is the observation → bounded memory →
tool-assisted action loop with a fresh observation after boundaries.  Batching
can reduce inference overhead and support short trajectories, but it also
creates committed actions whose token attribution is zero and whose outcome
cannot be evaluated until the batch stops.  Repeated-action suppression,
prediction, and falsification are therefore clean E2 candidates, not E1
features.

## Sources

- [Duck ToolAgent](https://github.com/Tufalabs/duck-harness/blob/main/ARC3-Inference/inference/agent/tool_agent.py)
- [Duck Solver](https://github.com/Tufalabs/duck-harness/blob/main/ARC3-Inference/inference/framework/solver.py)
- [Duck inference config](https://github.com/Tufalabs/duck-harness/blob/main/ARC3-Inference/configs/inference.json)
