import unittest
from arc3.agents.plumbing_agent import PlumbingAgent
from arc3.agents.reasoning_agent import ReasoningAgent
from arc3.env.mock import MoveToTargetEnv
from arc3.llm.backend import BaseLLMBackend

class FixedBackend(BaseLLMBackend):
    def decide(self,ctx):
        # ACTION2 then ACTION4 moves toward the fixture target; semantics are fixture-only.
        n=sum(1 for t in ctx["memory"]["recent_transitions"] if t["action"]["type"] in ("ACTION2","ACTION4"))
        return {"mode":"execute","analysis_summary":"fixed test policy","action":{"type":"ACTION2" if n<4 else "ACTION4"},"confidence":1}

class AgentTests(unittest.TestCase):
    def test_plumbing(self):
        r=PlumbingAgent(max_actions=5).run(MoveToTargetEnv()); self.assertEqual(r.actions,5); self.assertTrue(r.trace[0]["frame_hash"])
    def test_reasoning_trace(self):
        r=ReasoningAgent(FixedBackend(),max_actions=12).run(MoveToTargetEnv()); self.assertGreater(r.actions,0); self.assertIn("actual_frame_diff",r.trace[0]); self.assertTrue(all("chosen_action" in e for e in r.trace))

if __name__=="__main__": unittest.main()
