"""Small generic planners; intentionally independent of game rules."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Callable, Hashable, Iterable, TypeVar
from .actions import Action

S=TypeVar('S', bound=Hashable)

@dataclass(frozen=True)
class PlanResult:
    actions: tuple[Action, ...]
    explored: int
    found: bool

def bfs(start: S, goal: Callable[[S], bool], successors: Callable[[S], Iterable[tuple[Action,S]]], max_nodes: int = 2000) -> PlanResult:
    q=deque([(start,())]); seen={start}; explored=0
    while q and explored < max_nodes:
        state,path=q.popleft(); explored += 1
        if goal(state): return PlanResult(path,explored,True)
        for action,nxt in successors(state):
            if nxt not in seen:
                seen.add(nxt); q.append((nxt,path+(action,)))
    return PlanResult((),explored,False)

def grid_shortest_path(grid: tuple[tuple[int,...],...], start: tuple[int,int], goal: tuple[int,int], blocked: set[int] | None = None, max_nodes: int=4096) -> PlanResult:
    blocked=blocked or set(); h,w=len(grid),len(grid[0]) if grid else 0
    dirs=(("ACTION1",-1,0),("ACTION2",1,0),("ACTION3",0,-1),("ACTION4",0,1))
    def succ(p):
        for name,dy,dx in dirs:
            y,x=p; ny,nx=y+dy,x+dx
            if 0<=ny<h and 0<=nx<w and grid[ny][nx] not in blocked: yield Action.from_value(name),(ny,nx)
    return bfs(start,lambda p:p==goal,succ,max_nodes)
