# E2 candidate hypotheses (design only)

Only the top three candidates are retained.  None is implemented in this
phase; selection waits for E1 traces.

## 1. Duck + prediction / falsification / replan

- **Evidence:** Duck's source exposes a strong observation/tool/action loop,
  while the E0 benchmark contains long zero-token and repeated-action runs.
  Our framework already has declarative expectations and mismatch-triggered
  replans, but the heuristic backend never exercised semantic goals/plans.
- **Expected benefit:** preserve Duck's effective observation/tool execution
  while interrupting stale trajectories when a predicted effect is falsified.
- **Risk:** a hybrid could confound several mechanisms at once and overreact to
  harmless no-ops, increasing inference cost.
- **Clean ablation:** Duck loop with only prediction checks/replan toggled;
  keep prompt, representation, model, budgets, and tool behavior fixed.
- **Falsification condition:** no reduction in repeated state/action pairs or
  no progress improvement at equal model-call/action budgets.
- **Research value / cost:** highest expected value, medium-to-high cost;
  requires trace-level integration and a clean control.

## 2. Our Agent + rendered visual input

- **Evidence:** Duck's public harness can attach a current-grid image, while
  our E1 uses raw grid and structured text only. Dense spatial scenes may be a
  representation bottleneck even when the model is capable.
- **Expected benefit:** improve object/geometry recognition and goal grounding
  without importing Duck's prompt or tool loop.
- **Risk:** visual preprocessing, token/image cost, and modality changes could
  hide whether gains come from better perception or better planning.
- **Clean ablation:** same `ReasoningAgent`, checkpoint, sampler, action
  budget, and subset; toggle only a deterministic rendered-grid image.
- **Falsification condition:** no improvement in goal quality, plan validity,
  or level progress, or gains disappear when image resolution is controlled.
- **Research value / cost:** high value, medium cost; relatively clean after
  E1 establishes a text/grid baseline.

## 3. Our Agent + Python workspace

- **Evidence:** Duck's ephemeral Python tool exposes segmentation/history and
  can compute geometry before issuing actions. Our E1 deliberately has no
  model-facing computation tool.
- **Expected benefit:** compact path/geometry queries and explicit evidence
  tests could improve execution after a goal is plausible.
- **Risk:** arbitrary code/tool latency, sandbox complexity, and accidental
  transfer of Duck's action strategy; Python may not solve goal inference.
- **Clean ablation:** add a read-only workspace first; hold prompt contract,
  representation, model, and action submission fixed, then separately test
  whether tool-produced actions are allowed.
- **Falsification condition:** tool calls do not improve prediction accuracy,
  plan success, or level progress at equal compute/action budgets, or gains
  require importing Duck-specific prompting.
- **Research value / cost:** medium-to-high value, high cost; defer until E1
  shows semantic goals but execution/geometry remains limiting.
