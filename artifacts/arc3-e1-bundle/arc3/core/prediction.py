"""Small, model-agnostic contracts for action prediction and verification.

The V2 experiment deliberately keeps predictions declarative.  A backend may
provide only ``visible_effect`` or add bounds on changed cells, a state hash,
or a substring expected in the diff summary.  Unknown fields are ignored so
older backends remain compatible.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .observation import Observation
from .transition import FrameDiff


@dataclass(frozen=True)
class ActionExpectation:
    visible_effect: bool | None = None
    changed_cells_min: int | None = None
    changed_cells_max: int | None = None
    state_hash: str | None = None
    diff_contains: str | None = None
    label: str = ""

    @classmethod
    def from_value(cls, value: Any) -> "ActionExpectation | None":
        if value is None or value == "":
            return None
        if isinstance(value, str):
            # Legacy ``expected_effect`` is explanatory prose, not a
            # falsifiable assertion.  Structured mappings are required for a
            # prediction check.
            return None
        if not isinstance(value, Mapping):
            return None
        visible = value.get("visible_effect", value.get("changed"))
        if visible is not None:
            visible = bool(visible)
        bounds = value.get("changed_cells")
        low = value.get("changed_cells_min")
        high = value.get("changed_cells_max")
        if isinstance(bounds, int):
            low = high = bounds
        elif isinstance(bounds, (list, tuple)):
            if len(bounds) > 0:
                low = bounds[0]
            if len(bounds) > 1:
                high = bounds[1]
        return cls(visible, int(low) if low is not None else None,
                   int(high) if high is not None else None,
                   str(value["state_hash"]) if value.get("state_hash") else None,
                   str(value["diff_contains"]) if value.get("diff_contains") else None,
                   str(value.get("label", value.get("description", ""))))

    def is_empty(self) -> bool:
        return self.visible_effect is None and self.changed_cells_min is None and self.changed_cells_max is None and self.state_hash is None and self.diff_contains is None


def verify_expectation(expectation: ActionExpectation | None, diff: FrameDiff, after: Observation) -> tuple[bool, str]:
    """Return whether an observed transition satisfies an expectation."""
    if expectation is not None and not isinstance(expectation, ActionExpectation):
        expectation = ActionExpectation.from_value(expectation)
    if expectation is None or expectation.is_empty():
        return True, "no hard prediction"
    changed = len(diff.changed_cells)
    if expectation.visible_effect is not None and expectation.visible_effect != (not diff.no_visible_effect):
        return False, f"visible_effect expected {expectation.visible_effect}, observed {not diff.no_visible_effect}"
    if expectation.changed_cells_min is not None and changed < expectation.changed_cells_min:
        return False, f"changed_cells {changed} < minimum {expectation.changed_cells_min}"
    if expectation.changed_cells_max is not None and changed > expectation.changed_cells_max:
        return False, f"changed_cells {changed} > maximum {expectation.changed_cells_max}"
    if expectation.state_hash is not None and after.state_hash != expectation.state_hash:
        return False, f"state_hash expected {expectation.state_hash}, observed {after.state_hash}"
    if expectation.diff_contains is not None and expectation.diff_contains.lower() not in diff.summary().lower():
        return False, f"diff summary does not contain {expectation.diff_contains!r}"
    return True, "prediction matched"
