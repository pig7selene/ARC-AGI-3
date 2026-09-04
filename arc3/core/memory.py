"""Bounded per-game world model and interaction memory."""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
from typing import Any
from .actions import Action
from .observation import Observation
from .perception import FrameAbstraction, segment
from .transition import FrameDiff

@dataclass
class Hypothesis:
    statement: str
    evidence: list[str] = field(default_factory=list)
    counter_evidence: list[str] = field(default_factory=list)
    confidence: float = 0.5
    def update(self, supports: bool, evidence: str, amount: float = .12) -> None:
        (self.evidence if supports else self.counter_evidence).append(evidence)
        self.confidence = max(0.0, min(1.0, self.confidence + (amount if supports else -amount)))

@dataclass
class TransitionRecord:
    step: int; level: int; action: Action; before_hash: str; after_hash: str
    diff_summary: str; no_op: bool; reward: float; info: dict[str, Any] = field(default_factory=dict)

@dataclass
class GameMemory:
    max_recent: int = 12
    mechanics: dict[str, Hypothesis] = field(default_factory=dict)
    goals: list[Hypothesis] = field(default_factory=list)
    object_roles: dict[str, Hypothesis] = field(default_factory=dict)
    known_hazards: list[str] = field(default_factory=list)
    known_targets: list[str] = field(default_factory=list)
    transitions: deque[TransitionRecord] = field(default_factory=lambda: deque(maxlen=12))
    attempted_actions: list[dict[str, Any]] = field(default_factory=list)
    no_op_actions: set[str] = field(default_factory=set)
    failed_plans: list[str] = field(default_factory=list)
    visited_state_hashes: set[str] = field(default_factory=set)
    current_objects: FrameAbstraction | None = None
    current_plan: list[Action] = field(default_factory=list)
    level: int = 0
    resets: int = 0
    action_effects: dict[str, dict[str, Any]] = field(default_factory=dict)
    prediction_mismatches: int = 0
    replans: int = 0

    def __post_init__(self):
        self.transitions = deque(self.transitions, maxlen=self.max_recent)

    def observe(self, obs: Observation) -> None:
        self.level = obs.level; self.current_objects = segment(obs.grid); self.visited_state_hashes.add(obs.state_hash)
    def record_transition(self, rec: TransitionRecord) -> None:
        self.transitions.append(rec); self.attempted_actions.append(rec.action.as_dict())
        if rec.no_op: self.no_op_actions.add(rec.action.type.value)
        self.visited_state_hashes.add(rec.after_hash)
    def record_effect(self, action: Action, effect: dict[str, Any]) -> None:
        self.action_effects[action.type.value] = dict(effect)

    def action_outcomes(self, action_type: str) -> dict[str, int]:
        """Count observed effect/no-op outcomes for counterfactual reuse."""
        outcomes = {"effect": 0, "no_op": 0}
        for rec in self.transitions:
            if rec.action.type.value == action_type:
                outcomes["no_op" if rec.no_op else "effect"] += 1
        return outcomes

    def action_already_tested(self, action_type: str, *, outcome: str | None = None) -> bool:
        outcomes = self.action_outcomes(action_type)
        return bool(outcomes.get(outcome, 0)) if outcome else sum(outcomes.values()) > 0

    def falsifies_visibility(self, action_type: str, expected_visible: bool) -> bool:
        """Whether recorded transitions contradict a visibility prediction."""
        outcomes = self.action_outcomes(action_type)
        return outcomes["no_op" if expected_visible else "effect"] > 0
    def add_mechanic(self, key: str, statement: str, supports: bool, evidence: str) -> Hypothesis:
        h=self.mechanics.setdefault(key,Hypothesis(statement)); h.update(supports,evidence); return h

    def add_goal(self, statement: str, confidence: float = 0.5, evidence: str = "") -> Hypothesis:
        """Register a concise, falsifiable goal hypothesis from a backend."""
        statement = str(statement).strip()
        if not statement:
            raise ValueError("goal statement must not be empty")
        for goal in self.goals:
            if goal.statement == statement:
                goal.confidence = max(0.0, min(1.0, float(confidence)))
                if evidence:
                    goal.evidence.append(str(evidence))
                return goal
        goal = Hypothesis(statement, evidence=[str(evidence)] if evidence else [], confidence=max(0.0, min(1.0, float(confidence))))
        self.goals.append(goal)
        return goal
    def compact(self) -> dict[str, Any]:
        return {"level":self.level,"mechanics":{k:{"statement":h.statement,"confidence":round(h.confidence,2)} for k,h in self.mechanics.items()},
          "goals":[{"statement":h.statement,"confidence":round(h.confidence,2)} for h in self.goals[-5:]],
          "recent_transitions":[{"action":r.action.as_dict(),"diff":r.diff_summary,"no_op":r.no_op,"reward":r.reward} for r in self.transitions],
          "visited_states":len(self.visited_state_hashes),"no_op_actions":sorted(self.no_op_actions),"action_effects":self.action_effects,
          "action_outcomes":{a:self.action_outcomes(a) for a in sorted({r.action.type.value for r in self.transitions})},
          "prediction_mismatches":self.prediction_mismatches,"replans":self.replans}
