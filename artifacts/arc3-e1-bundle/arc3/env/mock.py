from __future__ import annotations
from dataclasses import dataclass
from arc3.core.actions import Action,ActionType
from arc3.core.observation import Observation,freeze_grid

@dataclass
class MoveToTargetEnv:
    """Tiny deterministic ARC-like game for regression tests, not a benchmark rule."""
    width:int=7; height:int=7; max_steps:int=40
    def __post_init__(self): self.reset()
    def reset(self,seed=None): self.y,self.x=1,1; self.steps=0; self.done=False; return self._obs()
    def valid_actions(self): return (ActionType.ACTION1,ActionType.ACTION2,ActionType.ACTION3,ActionType.ACTION4,ActionType.RESET,ActionType.ACTION6)
    def _obs(self):
        g=[[0]*self.width for _ in range(self.height)]; g[self.height-2][self.width-2]=2; g[self.y][self.x]=1
        won=(self.y,self.x)==(self.height-2,self.width-2); return Observation(freeze_grid(g),self.valid_actions(),0,1.0 if won else -0.01,self.done or won,won,{},self.steps)
    def step(self,a):
        a=Action.from_value(a); self.steps+=1
        if a.type is ActionType.RESET: return self.reset()
        if a.type is ActionType.ACTION1: self.y=max(0,self.y-1)
        elif a.type is ActionType.ACTION2: self.y=min(self.height-1,self.y+1)
        elif a.type is ActionType.ACTION3: self.x=max(0,self.x-1)
        elif a.type is ActionType.ACTION4: self.x=min(self.width-1,self.x+1)
        self.done=self.steps>=self.max_steps; return self._obs()
