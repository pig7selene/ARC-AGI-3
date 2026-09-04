"""Action types and validation for ARC-AGI-3 style environments."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping


class ActionType(str, Enum):
    RESET = "RESET"
    ACTION1 = "ACTION1"
    ACTION2 = "ACTION2"
    ACTION3 = "ACTION3"
    ACTION4 = "ACTION4"
    ACTION5 = "ACTION5"
    ACTION6 = "ACTION6"
    ACTION7 = "ACTION7"


@dataclass(frozen=True)
class Action:
    type: ActionType
    x: int | None = None
    y: int | None = None

    @classmethod
    def from_value(cls, value: Any) -> "Action":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(ActionType(value.upper()))
        if isinstance(value, Mapping):
            typ = value.get("type", value.get("action"))
            x,y=value.get("x"),value.get("y")
            return cls(ActionType(str(typ).upper()), int(x) if x is not None else None, int(y) if y is not None else None)
        if isinstance(value, (tuple, list)) and value:
            return cls(ActionType(str(value[0]).upper()), *(list(value[1:3]) + [None, None])[:2])
        raise ValueError(f"Cannot parse action: {value!r}")

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": self.type.value}
        if self.type is ActionType.ACTION6:
            out.update(x=self.x, y=self.y)
        return out


def normalize_valid_actions(values: Iterable[Any]) -> tuple[ActionType, ...]:
    """Normalize the several representations used by official/local harnesses."""
    result: list[ActionType] = []
    for value in values:
        if isinstance(value, ActionType):
            action = value
            if action not in result:
                result.append(action)
            continue
        if isinstance(value, Mapping):
            value = value.get("type", value.get("action"))
        if isinstance(value, (tuple, list)):
            value = value[0]
        try:
            action = ActionType(str(value).upper())
        except ValueError:
            continue
        if action not in result:
            result.append(action)
    return tuple(result)


def validate_action(action: Action, valid_actions: Iterable[ActionType], width: int, height: int) -> tuple[bool, str]:
    allowed = set(normalize_valid_actions(valid_actions))
    if action.type not in allowed:
        return False, f"{action.type.value} is not in valid_actions"
    if action.type is ActionType.ACTION6:
        if action.x is None or action.y is None:
            return False, "ACTION6 requires x and y"
        if not (0 <= int(action.x) < width and 0 <= int(action.y) < height):
            return False, "ACTION6 coordinates outside frame"
    elif action.x is not None or action.y is not None:
        return False, "coordinates are only valid for ACTION6"
    return True, ""
