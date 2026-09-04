from __future__ import annotations
from typing import Protocol, Iterable, Any
from arc3.core.actions import Action
from arc3.core.observation import Observation

class InteractiveEnvironment(Protocol):
    def reset(self, seed: int | None = None) -> Observation: ...
    def step(self, action: Action) -> Observation: ...
    def valid_actions(self) -> Iterable[Any]: ...

def adapt_environment(env: Any) -> Any:
    """Adapt common gym/official harness styles to reset/step/Observation."""
    class Adapter:
        def __init__(self,e): self.e=e; self.last=None
        def _observation(self, value):
            if isinstance(value, Observation): return value
            # arcengine.FrameDataRaw stores a frame *sequence* in a private
            # runtime field and exposes it via `.frame`; use the latest frame
            # while retaining sequence metadata in `info`.
            if hasattr(value, "available_actions") and hasattr(value, "state") and hasattr(value, "frame"):
                frames = getattr(value, "frame", [])
                if frames:
                    grid = frames[-1].tolist() if hasattr(frames[-1], "tolist") else frames[-1]
                    state = getattr(getattr(value, "state", None), "value", str(getattr(value, "state", "")))
                    return Observation.from_value({"grid":grid,
                        "valid_actions":[f"ACTION{int(a)}" for a in getattr(value,"available_actions",[])],
                        "level":getattr(value,"levels_completed",0),
                        "reward":float(getattr(value,"levels_completed",0)),
                        "done":state in {"WIN","GAME_OVER"}, "won":state == "WIN",
                        "info":{"game_id":getattr(value,"game_id",None),"state":state,"guid":getattr(value,"guid",None),
                                "levels_completed":getattr(value,"levels_completed",0),"win_levels":getattr(value,"win_levels",0),
                                "frame_count":len(frames)}})
            if hasattr(value, "frame") or hasattr(value, "grid"):
                grid=getattr(value,"frame",getattr(value,"grid",None));
                if grid is not None:
                    return Observation.from_value({"grid":grid,"valid_actions":getattr(value,"valid_actions",()),"level":getattr(value,"level",0),"reward":getattr(value,"reward",0.0),"done":getattr(value,"done",False),"won":getattr(value,"won",False)})
            return Observation.from_value(value)
        def reset(self,seed=None):
            try: value=self.e.reset(seed=seed) if seed is not None else self.e.reset()
            except TypeError: value=self.e.reset()
            if isinstance(value,tuple) and value and not isinstance(value[0],(int,float)):
                value=value[0]
            self.last=self._observation(value); return self.last
        def valid_actions(self):
            vals=getattr(self.e,"valid_actions",None)
            if vals is None: vals=getattr(self.e,"action_space",None)
            if vals is not None and hasattr(vals,"actions"): vals=vals.actions
            return vals() if callable(vals) else (vals or self.last.valid_actions if self.last else ())
        def step(self,action):
            native=action
            try:
                from arcengine import GameAction
                native=getattr(GameAction, action.type.value)
                if action.type.value == "ACTION6" and callable(native): native=native(action.x,action.y)
            except (ImportError,AttributeError,TypeError):
                native=action.as_dict()
            if action.type.value == "ACTION6":
                # Current arcengine 0.9.x represents clicks as GameAction.ACTION6
                # plus a separate data payload (x, y), in grid column/row order.
                value=self.e.step(native, data={"x": int(action.x), "y": int(action.y)})
            else:
                value=self.e.step(native)
            if isinstance(value,tuple):
                # Gymnasium style: observation, reward, terminated, truncated, info.
                if len(value) >= 4:
                    raw,reward,terminated,truncated,info=(list(value)+[{}]*5)[:5]
                    if isinstance(raw,dict):
                        raw=dict(raw); raw.setdefault("reward",reward); raw.setdefault("done",bool(terminated or truncated)); raw.setdefault("info",info)
                    else:
                        raw={"grid":raw,"reward":reward,"done":bool(terminated or truncated),"info":info}
                    value=raw
                elif value: value=value[0]
            if value is None:
                raise RuntimeError("environment returned no FrameDataRaw (the action may have been rejected)")
            self.last=self._observation(value); return self.last
    return Adapter(env)
