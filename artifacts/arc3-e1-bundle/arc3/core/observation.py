"""Canonical observation representation and deterministic state hashing."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Mapping

Grid = tuple[tuple[int, ...], ...]


def freeze_grid(grid: Any) -> Grid:
    rows = tuple(tuple(int(c) for c in row) for row in grid)
    if not rows:
        return rows
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("grid must be rectangular")
    if any(c < 0 or c > 15 for row in rows for c in row):
        raise ValueError("ARC grid colors must be integers in [0, 15]")
    return rows


@dataclass(frozen=True)
class Observation:
    grid: Grid
    valid_actions: tuple[Any, ...] = ()
    level: int = 0
    reward: float = 0.0
    done: bool = False
    won: bool = False
    info: Mapping[str, Any] = field(default_factory=dict)
    frame_id: int = 0

    @classmethod
    def from_value(cls, value: Any) -> "Observation":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            grid = value.get("grid", value.get("frame"))
            if grid is None:
                raise ValueError("observation has no grid/frame")
            return cls(freeze_grid(grid), tuple(value.get("valid_actions", value.get("actions", ()))),
                       int(value.get("level", 0)), float(value.get("reward", 0.0)), bool(value.get("done", False)),
                       bool(value.get("won", value.get("success", False))), value.get("info", {}), int(value.get("frame_id", 0)))
        if isinstance(value, (list, tuple)):
            return cls(freeze_grid(value))
        raise ValueError(f"Cannot parse observation: {value!r}")

    @property
    def height(self) -> int:
        return len(self.grid)

    @property
    def width(self) -> int:
        return len(self.grid[0]) if self.grid else 0

    @property
    def state_hash(self) -> str:
        payload = {"grid": self.grid, "level": self.level, "done": self.done, "won": self.won}
        return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()[:16]
