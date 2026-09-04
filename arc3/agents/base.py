from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from arc3.core.actions import Action
from arc3.core.memory import GameMemory, TransitionRecord
from arc3.core.observation import Observation
from arc3.core.perception import segment
from arc3.core.transition import frame_diff

@dataclass
class AgentResult:
    score: float; levels_cleared: int; actions: int; trace: list[dict[str,Any]]; failure_reason: str | None = None

class Agent:
    def __init__(self,max_actions=100): self.max_actions=max_actions
    def run(self,env,seed=None,logger=None)->AgentResult: raise NotImplementedError
    def _record(self,memory,step,before,action,after):
        diff=frame_diff(before.grid if before else None,after.grid)
        memory.record_transition(TransitionRecord(step,after.level,action,before.state_hash if before else "",after.state_hash,diff.summary(),diff.no_visible_effect,after.reward,dict(after.info)))
        return diff
