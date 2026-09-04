"""Make isolated, no-environment-action model calls before benchmark burn."""
from __future__ import annotations

import json
import os
from pathlib import Path

from arc3.core.actions import ActionType, normalize_valid_actions
from arc3.llm.backend import OpenAICompatibleBackend
from arc3.llm.parser import parse_decision, parse_plan


def contexts() -> list[dict]:
    recorded = next(Path("traces/v2/intervention").glob("*.jsonl"), None)
    sample = json.loads(recorded.read_text(encoding="utf8").splitlines()[0]) if recorded else {}
    grid = sample.get("grid") or [[0, 1, 0], [0, 0, 2], [0, 0, 0]]
    return [
        {"observation": {"width": len(grid[0]), "height": len(grid), "level": 0, "state_hash": "synthetic", "grid": grid},
         "objects": [], "last_transition": None, "memory": {"mechanics": {}, "goals": [], "recent_transitions": [], "action_outcomes": {}},
         "valid_actions": ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION6"], "no_op_actions": [], "current_plan": [],
         "synthetic_grid": grid},
        {"observation": {"width": len(grid[0]), "height": len(grid), "level": sample.get("level", 0), "state_hash": sample.get("frame_hash", "recorded"), "grid": grid},
         "objects": sample.get("objects", []), "last_transition": sample.get("actual_frame_diff"), "memory": {"mechanics": sample.get("current_hypotheses", {}), "goals": sample.get("goals", []), "recent_transitions": [], "action_outcomes": {}},
         "valid_actions": sample.get("valid_actions", ["ACTION1", "ACTION2"]), "no_op_actions": [], "current_plan": [], "synthetic_grid": grid},
    ]


def main() -> int:
    base_url, model = os.getenv("ARC3_MODEL_BASE_URL"), os.getenv("ARC3_MODEL")
    if not base_url or not model:
        raise SystemExit("Set ARC3_MODEL_BASE_URL and ARC3_MODEL before isolated validation.")
    backend = OpenAICompatibleBackend(base_url=base_url, model=model, api_key=os.getenv("ARC3_MODEL_API_KEY"))
    successes = 0; plans = 0; goals = 0; failures = []
    for index, context in enumerate(contexts()):
        try:
            raw = backend.decide(context)
            valid = normalize_valid_actions(context["valid_actions"])
            decision, error = parse_decision(raw, valid, context["observation"]["width"], context["observation"]["height"])
            plan, plan_error = parse_plan(raw, valid, context["observation"]["width"], context["observation"]["height"], 3)
            if decision is None: failures.append(f"call {index}: {error}"); continue
            successes += 1; plans += int(bool(plan)); goals += int(bool(raw.get("goal_hypothesis") or raw.get("goal") or raw.get("goals")))
            if plan_error: failures.append(f"call {index}: {plan_error}")
        except Exception as exc:
            failures.append(f"call {index}: {type(exc).__name__}: {exc}")
    print(json.dumps({"model": model, "calls": len(contexts()), "parse_successes": successes, "parse_success_rate": successes / len(contexts()), "goal_outputs": goals, "plan_outputs": plans, "failures": failures}, indent=2))
    return 0 if successes == len(contexts()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
