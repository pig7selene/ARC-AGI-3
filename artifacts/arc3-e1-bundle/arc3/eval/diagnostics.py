"""Deterministic failure taxonomy and measurable trace diagnostics."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Any, Iterable

FAILURES=("PERCEPTION_FAILURE","ACTION_SEMANTICS_FAILURE","CONTROL_IDENTIFICATION_FAILURE","GOAL_INFERENCE_FAILURE",
"EXPLORATION_FAILURE","NO_OP_LOOP","STATE_LOOP","PLANNING_FAILURE","PLAN_EXECUTION_FAILURE","CLICK_CANDIDATE_FAILURE",
"MEMORY_FAILURE","MODEL_FAILURE","TOOL_FAILURE","BUDGET_EXHAUSTED","UNKNOWN")

@dataclass
class TraceDiagnostics:
    primary_failure: str
    secondary_failures: list[str]
    evidence: list[str]
    metrics: dict[str,Any]
    def as_dict(self): return asdict(self)

def analyze_trace(trace: list[dict[str,Any]], completed: bool | None=None, max_actions: int | None=None) -> TraceDiagnostics:
    n=len(trace); hashes=[e.get("frame_hash") for e in trace if e.get("frame_hash")]; actions=[e.get("chosen_action",e.get("action",{})).get("type") for e in trace]
    diffs=[e.get("actual_frame_diff","") for e in trace]; noops=sum(d=="no visible effect" for d in diffs)
    repeated_states=n-len(set(hashes)); pairs=[(h,a) for h,a in zip(hashes,actions)]; repeated_pairs=len(pairs)-len(set(pairs))
    phase_counts=Counter(e.get("phase","UNKNOWN") for e in trace); click=[e for e in trace if e.get("chosen_action",e.get("action",{})).get("type")=="ACTION6"]
    effect_steps=[i for i,d in enumerate(diffs) if d and d != "no visible effect"]
    metrics={"actions":n,"no_op_rate":noops/n if n else 0.0,"unique_state_ratio":len(set(hashes))/n if n else 0.0,
      "repeated_state_rate":repeated_states/n if n else 0.0,"repeated_action_state_pairs":repeated_pairs,
      "effective_action_rate":len(effect_steps)/n if n else 0.0,"actions_before_first_effect":effect_steps[0] if effect_steps else n,
      "actions_by_phase":dict(phase_counts),"click_candidate_count":len(click),"click_no_op_rate":sum(e.get("actual_frame_diff")=="no visible effect" for e in click)/len(click) if click else 0.0,
      "planning_attempts":sum(e.get("mode")=="execute" or e.get("phase")=="PLAN" for e in trace),"llm_calls":sum(1 for e in trace if e.get("llm_call",True)),
      "hypothesis_flip_count":sum(1 for e in trace for h in e.get("current_hypotheses",{}).values() if h.get("confidence",.5) < .4),
      "goal_hypotheses_created":sum(e.get("goal_hypotheses_created",0) for e in trace),
      "goal_hypothesis_updates":sum(e.get("goal_hypothesis_updates",0) for e in trace),
      "goal_hypothesis_steps":sum(1 for e in trace if e.get("goals")),
      "goal_confidence_max":max((e.get("goal_confidence_max") for e in trace if e.get("goal_confidence_max") is not None), default=None),
      "plans_generated":sum(1 for e in trace if e.get("plan_generated")),
      "plan_steps_executed":sum(1 for e in trace if e.get("plan_step_executed")),
      "plan_aborts":sum(1 for e in trace if e.get("plan_aborted")),
      "prediction_checks":sum(1 for e in trace if e.get("expectation") is not None),
      "prediction_mismatches":sum(1 for e in trace if e.get("prediction_mismatch")),
      "replans":sum(1 for e in trace if e.get("prediction_mismatch")),
      "successful_replans":sum(1 for e in trace if e.get("successful_replan")),
      "parse_failures":sum(1 for e in trace if e.get("parse_failure")),
      "invalid_outputs":sum(1 for e in trace if e.get("parse_failure")),
      "backend_errors":sum(1 for e in trace if e.get("backend_error")),
      "model_calls":sum(1 for e in trace if e.get("llm_call")),
      "input_tokens":sum(e.get("input_tokens") or 0 for e in trace),
      "output_tokens":sum(e.get("output_tokens") or 0 for e in trace),
      "reasoning_tokens":sum(e.get("reasoning_tokens") or 0 for e in trace),
      "latency_ms_total":sum(e.get("model_latency_ms") or 0 for e in trace),
      "progress_events":sum(len(e.get("progress_events",[])) for e in trace),
      "progress_event_types":dict(Counter(p for e in trace for p in e.get("progress_events",[]))),
      "plan_steps":sum(1 for e in trace if e.get("plan_remaining",0) > 0 or e.get("mode")=="execute")}
    candidates: list[tuple[int,str,str]]=[]
    if noops >= 2: candidates.append((100,"NO_OP_LOOP",f"{noops}/{n} actions produced no visible effect"))
    if repeated_states >= 2: candidates.append((95,"STATE_LOOP",f"{repeated_states} repeated frame hashes"))
    if n and noops/n >= .35: candidates.append((80,"EXPLORATION_FAILURE",f"no-op rate {noops/n:.1%}"))
    unique_actions=len(set(a for a in actions if a))
    if n >= 5 and not effect_steps: candidates.append((85,"ACTION_SEMANTICS_FAILURE",f"no measurable effect after {n} actions"))
    if click and all(e.get("actual_frame_diff")=="no visible effect" for e in click): candidates.append((75,"CLICK_CANDIDATE_FAILURE","all ACTION6 candidates were no-op"))
    if completed is False or (trace and not trace[-1].get("game_completion") and max_actions and n>=max_actions): candidates.append((70,"BUDGET_EXHAUSTED",f"run ended at {n} actions without completion"))
    if trace and any(not e.get("objects") and e.get("grid") for e in trace): candidates.append((60,"PERCEPTION_FAILURE","non-empty frame had no detected objects"))
    candidates.sort(reverse=True); primary=candidates[0][1] if candidates else ("UNKNOWN" if completed is False else "UNKNOWN")
    return TraceDiagnostics(primary,[x[1] for x in candidates[1:]], [x[2] for x in candidates],metrics)
