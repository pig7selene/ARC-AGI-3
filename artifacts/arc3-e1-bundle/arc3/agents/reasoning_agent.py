from __future__ import annotations
import time
from .base import Agent,AgentResult
from arc3.core.actions import Action,ActionType,normalize_valid_actions,validate_action
from arc3.core.memory import GameMemory
from arc3.core.perception import segment
from arc3.core.transition import frame_diff
from arc3.core.exploration import rank_exploration
from arc3.llm.backend import BaseLLMBackend,HeuristicBackend
from arc3.llm.parser import parse_decision, parse_plan
from arc3.llm.prompt import build_context
from arc3.core.prediction import ActionExpectation, verify_expectation

class ReasoningAgent(Agent):
    def __init__(self,backend:BaseLLMBackend|None=None,max_actions=100,no_op_limit=3,max_plan_steps=3,enable_prediction=True):
        super().__init__(max_actions); self.backend=backend or HeuristicBackend(); self.no_op_limit=no_op_limit; self.max_plan_steps=max_plan_steps; self.enable_prediction=enable_prediction
    def run(self,env,seed=None,logger=None):
        obs=env.reset(seed=seed); mem=GameMemory(); mem.observe(obs); trace=[]; score=0.; levels=0; consecutive_noop=0; previous=None; seen_pairs=set(); pending_plan=[]; expectation=None; awaiting_replan=False
        for step in range(self.max_actions):
            valid=normalize_valid_actions(obs.valid_actions or env.valid_actions()); abstraction=segment(obs.grid); diff=frame_diff(previous,obs.grid) if previous is not None else None
            mem.current_plan = [item["action"] for item in pending_plan]
            context=build_context(obs,abstraction,diff,mem,valid)
            raw: dict = {}; plan_error = None; hypothesis_test = ""; goals_created = 0; goals_updated = 0; plan_generated = False; plan_length = 0; parse_failure = False; parse_error = None; backend_error = None; model_latency_ms = None; usage = {}; plan_step_executed = False; plan_aborted = False; successful_replan = False
            had_pending_plan = bool(pending_plan)
            if pending_plan:
                plan_step_executed = True
                item = pending_plan.pop(0); action=item["action"]; expectation=item.get("expectation"); mode="execute"; summary="execute verified plan step"; model_called=False
            else:
                model_called=True; started=time.perf_counter()
                try:
                    raw=self.backend.decide(context)
                except Exception as exc:
                    raw={}; backend_error=f"{type(exc).__name__}: {exc}"
                model_latency_ms = getattr(self.backend, "last_latency_ms", None)
                if model_latency_ms is None:
                    model_latency_ms = round((time.perf_counter()-started)*1000, 3)
                usage = dict(getattr(self.backend, "last_usage", {}) or {})
                decision,error=parse_decision(raw,valid,obs.width,obs.height)
                if backend_error:
                    decision, error = None, f"backend error: {backend_error}"
                if decision is None:
                    parse_failure=True; parse_error=error
                successful_replan = bool(awaiting_replan and decision is not None)
                awaiting_replan=False
                if decision is not None:
                    candidates = []
                    for key in ("goal_hypothesis", "goal"):
                        if decision.get(key): candidates.append(decision[key])
                    if isinstance(decision.get("goals"), (list, tuple)): candidates.extend(decision["goals"])
                    for candidate in candidates:
                        if isinstance(candidate, dict):
                            statement = candidate.get("statement", candidate.get("description", "")); confidence = candidate.get("confidence", 0.5)
                        else:
                            statement, confidence = str(candidate), decision.get("confidence", 0.5)
                        before = len(mem.goals); existing = any(g.statement == str(statement).strip() for g in mem.goals)
                        try: mem.add_goal(statement, confidence, decision.get("analysis_summary", ""))
                        except (TypeError, ValueError): continue
                        goals_created += int(len(mem.goals) > before); goals_updated += int(existing)
                parsed_plan, plan_error = parse_plan(raw,valid,obs.width,obs.height,self.max_plan_steps) if self.enable_prediction else ([], None)
                plan_generated = bool(parsed_plan); plan_length = len(parsed_plan)
                if decision is None:
                    ranked=rank_exploration(valid,mem); action=ranked[0].action if ranked else Action(ActionType.ACTION1); mode="recover"; summary=f"repaired invalid model output: {error}"
                    expectation=None
                    if action.type is ActionType.ACTION6 and (action.x is None or action.y is None): action=Action(ActionType.ACTION6,obs.width//2,obs.height//2)
                elif parsed_plan:
                    # A plan owns the first action as well as its remaining queue.
                    first=parsed_plan[0]; action=first["action"]; expectation=first.get("expectation"); pending_plan=parsed_plan[1:]; mem.current_plan=[item["action"] for item in pending_plan]; plan_step_executed=True
                    mode="execute"; summary=decision.get("analysis_summary","") or "commit short plan and verify each step"
                else:
                    action=decision["action"]; mode=decision.get("mode","explore"); summary=decision.get("analysis_summary",""); expectation=ActionExpectation.from_value(decision.get("expectation",decision.get("prediction",decision.get("expected_effect")))) if self.enable_prediction else None
                if mode == "explore":
                    hypothesis_test = f"test whether {action.type.value} changes the current state"
                    if mem.action_already_tested(action.type.value):
                        summary = (summary + "; prior outcome available in memory, retesting only if needed").strip("; ")
            if action.type is ActionType.RESET and not (obs.done or step==0):
                ranked=rank_exploration(valid,mem); action=ranked[0].action if ranked else Action(ActionType.ACTION1); mode="recover"; summary="RESET guard replaced unnecessary reset"
                expectation=None
                if action.type is ActionType.ACTION6 and (action.x is None or action.y is None): action=Action(ActionType.ACTION6,obs.width//2,obs.height//2)
            pair=(obs.state_hash,action.as_dict().get("type"),action.x,action.y)
            if pair in seen_pairs:
                ranked=rank_exploration(valid,mem); alternatives=[c.action for c in ranked if (obs.state_hash,c.action.type.value,c.action.x,c.action.y) not in seen_pairs]
                if alternatives:
                    action=alternatives[0]
                    if action.type is ActionType.ACTION6 and (action.x is None or action.y is None): action=Action(ActionType.ACTION6,obs.width//2,obs.height//2)
                    mode="recover"; summary="repeated state/action guard selected new probe"; expectation=None
            seen_pairs.add((obs.state_hash,action.type.value,action.x,action.y))
            before_abs=abstraction; known_states=set(mem.visited_state_hashes); known_effects=set(mem.action_effects)
            after=env.step(action); actual=self._record(mem,step,obs,action,after); score+=after.reward; consecutive_noop=consecutive_noop+1 if actual.no_visible_effect else 0
            prediction_ok, prediction_reason = verify_expectation(expectation, actual, after) if self.enable_prediction else (True, "verification disabled")
            if not prediction_ok:
                mem.prediction_mismatches += 1; mem.replans += 1; pending_plan.clear(); awaiting_replan=True; plan_aborted = bool(had_pending_plan or plan_generated)
                summary = (summary + "; prediction mismatch -> replan").strip()
            if actual.moved_objects:
                after_abs=segment(after.grid); old_by={o.id:o for o in before_abs.objects}; new_by={o.id:o for o in after_abs.objects}
                for old_id,new_id in actual.moved_objects:
                    if old_id in old_by and new_id in new_by:
                        oy,ox=old_by[old_id].centroid; ny,nx=new_by[new_id].centroid
                        mem.record_effect(action,{"kind":"movement","dy":round(ny-oy,2),"dx":round(nx-ox,2)})
                mem.add_mechanic(action.type.value, f"{action.type.value} changes the scene (likely movement/interact)", True, actual.summary())
            elif actual.no_visible_effect:
                mem.add_mechanic(action.type.value, f"{action.type.value} has no visible effect in this state", False, actual.summary())
            if actual.no_visible_effect and consecutive_noop >= self.no_op_limit:
                mem.failed_plans.append("no-op loop"); consecutive_noop=0
            phase=("RECOVER" if mode=="recover" else "DISCOVER_ACTIONS" if len(mem.action_effects)==0 and step < len(valid) else "DISCOVER_MECHANICS" if len(mem.action_effects)==0 else "INFER_GOAL" if not mem.goals else "EXECUTE")
            progress_events=[]
            if after.state_hash not in known_states: progress_events.append("novel_observable_state")
            if after.level > obs.level: progress_events.append("level_advanced")
            if after.reward > obs.reward: progress_events.append("reward_increased")
            if actual.moved_objects and action.type.value not in known_effects: progress_events.append("new_mechanic_observed")
            if goals_created: progress_events.append("goal_hypothesis_created")
            if after.won: progress_events.append("level_completed")
            goals=mem.compact()["goals"]
            confidences=[g["confidence"] for g in goals]
            event={"step":step,"level":after.level,"frame_hash":after.state_hash,"grid":after.grid,"objects":[o.__dict__ for o in abstraction.objects],"chosen_action":action.as_dict(),"valid_actions":[a.value for a in valid],"reasoning_summary":summary,"mode":mode,"phase":phase,"llm_call":model_called,"current_hypotheses":mem.compact()["mechanics"],"goals":goals,"goal_hypotheses_created":goals_created,"goal_hypothesis_updates":goals_updated,"goal_confidence_max":max(confidences) if confidences else None,"goal_confidence_mean":sum(confidences)/len(confidences) if confidences else None,"action_effects":mem.action_effects,"expected_effect":raw.get("expected_effect","") if isinstance(raw,dict) else "","expectation":expectation.__dict__ if expectation else None,"prediction_verified":prediction_ok,"prediction_result":prediction_reason,"prediction_mismatch":not prediction_ok,"plan_generated":plan_generated,"plan_length":plan_length,"plan_step_executed":plan_step_executed,"plan_remaining":len(pending_plan),"plan_aborted":plan_aborted,"successful_replan":successful_replan,"plan_error":plan_error,"hypothesis_test":hypothesis_test,"parse_failure":parse_failure,"parse_error":parse_error,"backend_error":backend_error,"model_latency_ms":model_latency_ms,"input_tokens":usage.get("prompt_tokens",usage.get("input_tokens")),"output_tokens":usage.get("completion_tokens",usage.get("output_tokens")),"reasoning_tokens":usage.get("reasoning_tokens"),"progress_events":progress_events,"actual_frame_diff":actual.summary(),"reward":after.reward,"level_completion":after.won,"game_completion":after.done and after.won}
            trace.append(event); logger and logger(event)
            previous=obs.grid; obs=after
            if after.won: levels+=1
            if after.done: break
        return AgentResult(score,levels,len(trace),trace,None if obs.won else "horizon or game not solved")
