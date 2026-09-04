import unittest

from analysis.e0_benchmark_analysis import analyze_benchmark
from analysis.compare_e0_e1 import compare


class AnalysisScriptTests(unittest.TestCase):
    def test_e0_metrics_keep_type_and_exact_streaks_distinct(self):
        payload = {"game_runs": [{
            "game_id": "g",
            "levels_completed": 1,
            "final_score": 2,
            "history": [
                {"action": {"id": "ACTION6", "data": {"x": 1, "y": 1}}, "generated_tokens": 3},
                {"action": {"id": "ACTION6", "data": {"x": 2, "y": 1}}, "generated_tokens": 0},
                {"action": {"id": "ACTION1", "data": {}}, "generated_tokens": 0},
            ],
        }]}
        row = analyze_benchmark(payload)["per_game"][0]
        self.assertEqual(row["environment_actions"], 3)
        self.assertEqual(row["model_generating_actions"], 1)
        self.assertEqual(row["zero_token_actions"], 2)
        self.assertEqual(row["longest_identical_action_streak"], 2)
        self.assertEqual(row["longest_exact_action_streak"], 1)

    def test_compare_uses_e1_top_level_order_when_ids_absent(self):
        e0 = {"game_runs": [
            {"game_id": "cd82-fb555c5d", "levels_completed": 0, "final_score": 0, "history": []},
            {"game_id": "ls20-9607627b", "levels_completed": 1, "final_score": 1, "history": []},
        ]}
        e1 = {"games": ["cd82-fb555c5d", "ls20-9607627b"], "per_game": [
            {"levels_cleared": 1, "score": 2, "actions": 4},
            {"levels_cleared": 0, "score": 0, "actions": 4},
        ]}
        report = compare(e0, e1)
        self.assertIn("cd82-fb555c5d", report["categories"]["Our-Agent-only wins"])
        self.assertIn("ls20-9607627b", report["categories"]["Duck-only wins"])
        self.assertIn("top-level games order", report["mapping_assumption"])

    def test_compare_missing_e1_is_pending(self):
        e0 = {"game_runs": [{"game_id": "g", "levels_completed": 0, "history": []}]}
        report = compare(e0, None)
        self.assertEqual(report["status"], "pending_e1")
        self.assertEqual(report["per_game"][0]["classification"], "pending_e1")


if __name__ == "__main__":
    unittest.main()
