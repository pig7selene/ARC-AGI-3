# E1 success criteria (pre-registered)

These criteria are fixed before the real Qwen run.  They evaluate the
architecture on the fixed subset `cd82-fb555c5d`, `ls20-9607627b`, and
`lf52-271a04aa`, seed `0`, with the configured action budget.  They must not be
rewritten after seeing scores.

## Strong success

All of the following should be visible in the E1 traces:

1. At least one real level completion on the fixed subset.
2. A semantic chain connecting `goal hypothesis → plan → execution` rather
   than only recover/explore fallbacks.
3. At least one interpretable progress event (novel state, new mechanic,
   reward increase, level advance/completion, or a verified replan) tied to the
   trace.

## Partial success

No level is completed, but the run shows a reliable semantic chain: non-empty
goal hypotheses, executable bounded plans, and/or prediction/replan events that
are materially different from the deterministic `HeuristicBackend` baseline.
The trace must make clear whether the limiting factor is representation,
model output, prediction, or environment execution.

## Failure

Classify as failure when the real-model run remains dominated by one or more of:

- zero useful goals and zero useful plans;
- repeated exploration or repeated state/action pairs without progress;
- parse/backend failures that prevent semantic decisions;
- action-loop or horizon exhaustion with no interpretable progress.

A failure is still useful evidence: it falsifies the claim that replacing only
the policy backend is sufficient under the current representation and contract.

## Reporting requirements

Report per game and aggregate:

- levels, score, actions, output/reasoning tokens, latency/runtime;
- first goal/mechanics evidence and plan/prediction/replan metrics;
- parse/backend errors and repeated-state/no-op diagnostics;
- exact checkpoint/config identity and any remaining fairness caveat.
