import unittest
from arc3.core.actions import *
from arc3.core.observation import Observation,freeze_grid
from arc3.core.perception import segment
from arc3.core.transition import frame_diff
from arc3.core.memory import GameMemory
from arc3.core.planner import bfs,grid_shortest_path
from arc3.core.exploration import rank_exploration
from arc3.llm.parser import parse_decision
from arc3.tools.python_tool import PythonAnalysisTool

class CoreTests(unittest.TestCase):
    def test_segmentation(self):
        a=segment(freeze_grid([[0,1,0],[0,1,2],[2,0,0]]))
        self.assertEqual(a.color_frequencies,{0:5,1:2,2:2}); self.assertEqual(len(a.objects),3)
        self.assertEqual(a.objects[0].area,2)
    def test_diff_and_noop(self):
        old=freeze_grid([[0,1],[0,0]]); new=freeze_grid([[0,0],[0,1]])
        d=frame_diff(old,new); self.assertFalse(d.no_visible_effect); self.assertEqual(len(d.changed_cells),2)
        self.assertTrue(frame_diff(old,old).no_visible_effect)
    def test_memory_hypothesis(self):
        m=GameMemory(); h=m.add_mechanic("a1","moves target",True,"step 1"); self.assertGreater(h.confidence,.5); m.add_mechanic("a1","moves target",False,"step 2"); self.assertAlmostEqual(h.confidence,.5)
    def test_guards_and_parser(self):
        ok,_=validate_action(Action(ActionType.ACTION6,2,2),(ActionType.ACTION6,),3,3); self.assertTrue(ok)
        self.assertIsNone(parse_decision({"action":{"type":"ACTION6","x":3,"y":0}},(ActionType.ACTION6,),3,3)[0])
    def test_bfs(self):
        r=grid_shortest_path(((0,0,0),(0,9,0),(0,0,0)),(0,0),(2,2),{9}); self.assertTrue(r.found); self.assertEqual(len(r.actions),4)
    def test_exploration_avoids_noop(self):
        m=GameMemory(); m.no_op_actions.add("ACTION1"); ranked=rank_exploration((ActionType.ACTION1,ActionType.ACTION2),m); self.assertEqual(ranked[0].action.type,ActionType.ACTION2)
    def test_ephemeral_python_tool(self):
        t=PythonAnalysisTool({"cells":[(1,2),(2,2)]}); self.assertEqual(t.run("len(cells)"),2)
        with self.assertRaises(ValueError): t.run("__import__('os')")
    def test_context_is_bounded(self):
        m=GameMemory(max_recent=2)
        self.assertEqual(m.transitions.maxlen,2)

if __name__=="__main__": unittest.main()
