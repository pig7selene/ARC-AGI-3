from __future__ import annotations
from dataclasses import dataclass
from .actions import Action, ActionType
from .memory import GameMemory

@dataclass(frozen=True)
class CandidateScore:
    action: Action; information_gain: float; progress: float; risk: float; reversibility: float; duplicate_penalty: float
    @property
    def total(self) -> float: return self.information_gain + self.progress + self.reversibility - self.risk - self.duplicate_penalty

def rank_exploration(valid_actions, memory: GameMemory, allow_reset: bool=False) -> list[CandidateScore]:
    out=[]
    for typ in valid_actions:
        typ = typ if isinstance(typ,ActionType) else ActionType(str(typ).upper())
        if typ is ActionType.RESET and not allow_reset: continue
        noop = typ.value in memory.no_op_actions
        out.append(CandidateScore(Action(typ), 0.9 if not noop else 0.05, 0.2, 0.8 if typ is ActionType.RESET else 0.1,
                                  0.7 if typ not in (ActionType.RESET,ActionType.ACTION7) else 0.2, 1.0 if noop else 0.0))
    return sorted(out,key=lambda c:c.total,reverse=True)
