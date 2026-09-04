from __future__ import annotations
from typing import Any
from .backend import BaseLLMBackend
from arc3.core.actions import Action, ActionType, validate_action
from arc3.core.prediction import ActionExpectation

def parse_decision(raw: Any, valid_actions, width: int, height: int) -> tuple[dict[str,Any] | None, str | None]:
    if not isinstance(raw,dict): return None,"decision must be a JSON object"
    try: mode=str(raw.get("mode","explore")); action=Action.from_value(raw.get("action"))
    except Exception as exc: return None,f"invalid decision: {exc}"
    if mode not in {"explore","execute","recover"}: return None,"invalid mode"
    ok,reason=validate_action(action,valid_actions,width,height)
    if not ok: return None,reason
    raw=dict(raw); raw["mode"]=mode; raw["action"]=action; return raw,None


def parse_plan(raw: Any, valid_actions, width: int, height: int, max_steps: int = 3) -> tuple[list[dict[str, Any]], str | None]:
    """Parse an optional bounded plan while preserving per-step predictions.

    Invalid plans are rejected as a whole; the caller can safely fall back to
    the already parsed primary action.  This keeps malformed model output from
    consuming environment actions.
    """
    if not isinstance(raw, dict) or raw.get("plan") in (None, [], ()):
        return [], None
    value = raw.get("plan")
    if not isinstance(value, (list, tuple)):
        return [], "plan must be a list"
    if len(value) > max_steps:
        return [], f"plan exceeds {max_steps} steps"
    result: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if isinstance(item, dict) and "action" in item:
            action_value = item.get("action")
            expectation = item.get("expectation", item.get("prediction", item.get("expected_effect")))
        else:
            action_value, expectation = item, None
        try:
            action = Action.from_value(action_value)
        except Exception as exc:
            return [], f"invalid plan action {index}: {exc}"
        ok, reason = validate_action(action, valid_actions, width, height)
        if not ok:
            return [], f"invalid plan action {index}: {reason}"
        result.append({"action": action, "expectation": ActionExpectation.from_value(expectation)})
    return result, None
