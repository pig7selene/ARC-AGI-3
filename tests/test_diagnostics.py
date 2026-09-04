import unittest
from arc3.eval.diagnostics import analyze_trace

class DiagnosticTests(unittest.TestCase):
    def test_noop_and_state_loop_taxonomy(self):
        trace=[{"step":i,"frame_hash":"same" if i else "a","chosen_action":{"type":"ACTION1"},"actual_frame_diff":"no visible effect","game_completion":False,"grid":[[1]]} for i in range(4)]
        d=analyze_trace(trace,completed=False,max_actions=4)
        self.assertEqual(d.primary_failure,"NO_OP_LOOP"); self.assertIn("STATE_LOOP",d.secondary_failures); self.assertEqual(d.metrics["repeated_action_state_pairs"],2)

if __name__=="__main__": unittest.main()
