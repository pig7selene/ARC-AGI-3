import unittest

from arc3.agents.reasoning_agent import ReasoningAgent
from arc3.core.actions import ActionType
from arc3.core.prediction import ActionExpectation, verify_expectation
from arc3.core.transition import frame_diff
from arc3.core.observation import freeze_grid, Observation
from arc3.core.memory import GameMemory, TransitionRecord
from arc3.core.actions import Action
from arc3.env.mock import MoveToTargetEnv
from arc3.llm.backend import BaseLLMBackend
from arc3.llm.parser import parse_plan
from arc3.llm.backend import HeuristicBackend


class PlanBackend(BaseLLMBackend):
    def __init__(self, plan):
        self.plan = plan
        self.calls = 0

    def decide(self, context):
        self.calls += 1
        return {"mode": "execute", "analysis_summary": "test plan", "action": self.plan[0]["action"], "plan": self.plan}


class GoalBackend(BaseLLMBackend):
    def decide(self, context):
        return {"mode": "explore", "action": {"type": "ACTION1"}, "goal_hypothesis": {"statement": "reach the marked object", "confidence": 0.8}}


class PredictionTests(unittest.TestCase):
    def test_prediction_matches_changed_cells(self):
        old = freeze_grid([[1, 0], [0, 0]])
        new = freeze_grid([[0, 1], [0, 0]])
        diff = frame_diff(old, new)
        ok, reason = verify_expectation(ActionExpectation(visible_effect=True, changed_cells_min=2, changed_cells_max=2), diff, Observation(new))
        self.assertTrue(ok, reason)

    def test_plan_is_bounded_and_validated(self):
        plan, error = parse_plan({"plan": [{"action": {"type": "ACTION1"}}] * 4}, (ActionType.ACTION1,), 3, 3, max_steps=3)
        self.assertEqual(plan, [])
        self.assertIn("exceeds", error)

    def test_mismatch_halts_queue_and_records_trace(self):
        backend = PlanBackend([
            {"action": {"type": "ACTION1"}, "expectation": {"visible_effect": False}},
            {"action": {"type": "ACTION2"}, "expectation": {"visible_effect": True}},
        ])
        result = ReasoningAgent(backend, max_actions=2).run(MoveToTargetEnv())
        self.assertTrue(result.trace[0]["prediction_mismatch"])
        self.assertEqual(result.trace[0]["plan_remaining"], 0)
        self.assertEqual(result.trace[0]["prediction_verified"], False)
        self.assertGreaterEqual(backend.calls, 2)

    def test_prior_trace_can_falsify_visibility_hypothesis(self):
        memory = GameMemory()
        action = Action(ActionType.ACTION1)
        memory.record_transition(TransitionRecord(0, 0, action, "a", "a", "no visible effect", True, 0.0))
        self.assertTrue(memory.falsifies_visibility("ACTION1", True))
        self.assertFalse(memory.falsifies_visibility("ACTION1", False))

    def test_heuristic_reuses_trace_to_choose_untested_action(self):
        backend = HeuristicBackend()
        context = {"valid_actions": ["ACTION1", "ACTION2"], "no_op_actions": [],
                   "memory": {"action_outcomes": {"ACTION1": {"effect": 1, "no_op": 0}}}}
        self.assertEqual(backend.decide(context)["action"]["type"], "ACTION2")

    def test_goal_hypothesis_is_recorded(self):
        result = ReasoningAgent(GoalBackend(), max_actions=1).run(MoveToTargetEnv())
        self.assertEqual(result.trace[0]["goal_hypotheses_created"], 1)
        self.assertEqual(result.trace[0]["goals"][0]["statement"], "reach the marked object")


if __name__ == "__main__":
    unittest.main()
