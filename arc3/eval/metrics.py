from __future__ import annotations
from dataclasses import dataclass,asdict
from typing import Iterable
from .diagnostics import analyze_trace
@dataclass
class EvaluationSummary:
    games:int; overall_score:float; levels_cleared:int; actions:int; no_op_rate:float; runtime_seconds:float; failures:list[str]
    per_game:list[dict]
    diagnostics:dict = None
    def as_dict(self): return asdict(self)
def summarize(results:Iterable, runtime_seconds:float=0.0)->EvaluationSummary:
    rs=list(results); total=sum(len(r.trace) for r in rs); noops=sum(1 for r in rs for e in r.trace if e.get("actual_frame_diff")=="no visible effect")
    per=[{"score":r.score,"levels_cleared":r.levels_cleared,"actions":r.actions,"failure":r.failure_reason,"diagnostics":analyze_trace(r.trace,r.failure_reason is None,r.actions).as_dict()} for r in rs]
    metric_keys=("goal_hypotheses_created","goal_hypothesis_updates","plans_generated","plan_steps_executed","plan_aborts","prediction_checks","prediction_mismatches","replans","successful_replans","parse_failures","invalid_outputs","backend_errors","model_calls","input_tokens","output_tokens","reasoning_tokens","progress_events")
    aggregate={"unique_state_ratio":len({e.get("frame_hash") for r in rs for e in r.trace})/total if total else 0.0,
      "repeated_action_state_pairs":sum(x["diagnostics"]["metrics"]["repeated_action_state_pairs"] for x in per),
      "actions_by_phase":{k:sum(x["diagnostics"]["metrics"]["actions_by_phase"].get(k,0) for x in per) for k in {p for x in per for p in x["diagnostics"]["metrics"]["actions_by_phase"]}}}
    for key in metric_keys: aggregate[key]=sum(x["diagnostics"]["metrics"].get(key,0) or 0 for x in per)
    aggregate["latency_ms_total"]=sum(x["diagnostics"]["metrics"].get("latency_ms_total",0) or 0 for x in per)
    aggregate["progress_event_types"]={}
    for x in per:
        for name,count in x["diagnostics"]["metrics"].get("progress_event_types",{}).items(): aggregate["progress_event_types"][name]=aggregate["progress_event_types"].get(name,0)+count
    return EvaluationSummary(len(rs),sum(r.score for r in rs),sum(r.levels_cleared for r in rs),total,noops/total if total else 0.0,runtime_seconds,[r.failure_reason for r in rs if r.failure_reason],per,aggregate)
