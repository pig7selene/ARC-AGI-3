"""Best-effort bridge to whichever official toolkit release is installed."""
from __future__ import annotations
import importlib, os
import json
from pathlib import Path
from typing import Any

_LAST_ARCADE: Any = None

def discover_public_games(mode: str | None = None, artifact_path: str | None = None) -> dict[str, Any]:
    """Return toolkit-provided public game factories without hardcoding game IDs.

    Toolkit releases expose different registries; this probes common registry names and
    otherwise returns an empty mapping with an actionable error from `require_toolkit`.
    """
    # Current 0.9.x toolkit exposes discovery through Arcade.get_environments.
    try:
        import arc_agi
        selected = getattr(arc_agi.OperationMode, (mode or os.getenv("ARC3_TOOLKIT_MODE", "OFFLINE")).upper())
        arcade = arc_agi.Arcade(operation_mode=selected)
        global _LAST_ARCADE
        _LAST_ARCADE = arcade
        infos = arcade.get_environments()
        if infos:
            metadata=[]; games={}
            for info in infos:
                gid=info.game_id
                metadata.append({k:(getattr(info,k).isoformat() if hasattr(getattr(info,k),"isoformat") else getattr(info,k)) for k in info.__class__.model_fields})
                games[gid] = (lambda game_id=gid, a=arcade: a.make(game_id, include_frame_data=True, save_recording=True))
            if artifact_path:
                p=Path(artifact_path); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(metadata,indent=2,default=str),encoding="utf8")
            return games
    except Exception:
        pass
    modules=("arc_agi_3", "arcade", "arc_agi_3.environment")
    for name in modules:
        try: mod=importlib.import_module(name)
        except ImportError: continue
        for attr in ("PUBLIC_GAMES","public_games","GAME_REGISTRY","games"):
            registry=getattr(mod,attr,None)
            if isinstance(registry,dict): return dict(registry)
    return {}

def get_last_arcade() -> Any:
    return _LAST_ARCADE

def require_toolkit() -> None:
    if not discover_public_games():
        raise RuntimeError("Official ARC-AGI-3 toolkit/public game registry is not installed. Install the current arc-agi package in the target environment, then pass its factories through adapt_environment().")
