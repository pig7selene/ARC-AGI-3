from __future__ import annotations
from .base import Agent,AgentResult
from arc3.core.actions import Action,ActionType,normalize_valid_actions,validate_action
from arc3.core.memory import GameMemory

class PlumbingAgent(Agent):
    def run(self,env,seed=None,logger=None):
        obs=env.reset(seed=seed); mem=GameMemory(); mem.observe(obs); trace=[]; score=0.0; levels=0
        for step in range(self.max_actions):
            valid=normalize_valid_actions(obs.valid_actions or env.valid_actions()); choices=[a for a in valid if a is not ActionType.RESET and a.value not in mem.no_op_actions]
            typ=(choices or [a for a in valid if a is not ActionType.RESET] or list(valid))[0] if valid else ActionType.ACTION1
            action=Action(typ)
            if typ is ActionType.ACTION6: action=Action(typ,obs.width//2,obs.height//2)
            ok,_=validate_action(action,valid,obs.width,obs.height)
            if not ok: break
            after=env.step(action); diff=self._record(mem,step,obs,action,after); score+=after.reward
            event={"step":step,"level":after.level,"frame_hash":after.state_hash,"grid":after.grid,"action":action.as_dict(),"valid_actions":[a.value for a in valid],"mode":"explore","phase":"DISCOVER_ACTIONS","llm_call":False,"reasoning_summary":"deterministic plumbing probe","expected_effect":"observe environment response","actual_frame_diff":diff.summary(),"reward":after.reward,"level_completion":after.won,"game_completion":after.done and after.won}
            trace.append(event); logger and logger(event); obs=after
            if after.won: levels+=1
            if after.done: break
        return AgentResult(score,levels,len(trace),trace,None if obs.won else "horizon or game not solved")
