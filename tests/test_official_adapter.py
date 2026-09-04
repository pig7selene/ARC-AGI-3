import unittest
from arc3.env.protocol import adapt_environment

try:
    import numpy as np
    from arcengine import FrameDataRaw, GameAction, GameState
except ImportError:
    np = None

@unittest.skipUnless(np is not None, "official toolkit not installed")
class OfficialAdapterTests(unittest.TestCase):
    def test_frame_data_raw_sequence_and_actions(self):
        raw=FrameDataRaw(game_id="x", state=GameState.NOT_FINISHED, available_actions=[1,2,6,7])
        raw.frame=[np.zeros((3,4),dtype=int), np.ones((3,4),dtype=int)]
        obs=adapt_environment(type("E",(),{})())
        got=obs._observation(raw)
        self.assertEqual((got.height,got.width),(3,4)); self.assertEqual(got.grid[0][0],1)
        self.assertEqual(list(got.valid_actions),["ACTION1","ACTION2","ACTION6","ACTION7"])
        self.assertEqual(got.info["frame_count"],2)

    def test_action6_payload_is_x_y(self):
        class E:
            def reset(self):
                raw=FrameDataRaw(state=GameState.NOT_FINISHED,available_actions=[6]); raw.frame=[np.zeros((4,5),dtype=int)]; return raw
            def step(self, action, data=None):
                self.payload=(action,data); raw=FrameDataRaw(state=GameState.NOT_FINISHED,available_actions=[6]); raw.frame=[np.zeros((4,5),dtype=int)]; return raw
        e=E(); a=adapt_environment(e); a.reset(); a.step(__import__('arc3').core.actions.Action.from_value({"type":"ACTION6","x":3,"y":2}))
        self.assertEqual(e.payload[1],{"x":3,"y":2})

if __name__ == "__main__": unittest.main()
