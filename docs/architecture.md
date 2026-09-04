# Architecture

`Observation` is the only environment-facing frame type. `segment()` produces stable, explainable object descriptors (color, cells, bbox, centroid, shape and borders). `frame_diff()` reports changed cells plus appeared/disappeared/moved/resized/recolored objects and no-op status.

`GameMemory` is created per game, bounded to recent transitions, and retains mechanics hypotheses across levels while layout observations are refreshed. `ReasoningAgent` compresses this memory into a JSON context, asks a replaceable backend for a strict decision, validates it against the environment's action subset, applies RESET/no-op guards, then records the full transition. `PlumbingAgent` exercises the same loop deterministically.

The planner is deliberately generic (`bfs`, `grid_shortest_path`) and can be called by a reasoning backend or integrated learned component later. V2 adds a small declarative `ActionExpectation` contract: structured predictions (visibility, changed-cell bounds, state hash, or diff text) are checked after each step, and a bounded plan queue is cleared on mismatch. Explanatory legacy `expected_effect` prose remains non-binding. Evaluation and trace logging are separate from the agent so official scoring can be plugged in without changing policy code.
