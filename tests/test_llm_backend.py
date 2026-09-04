import json
import unittest
from unittest.mock import patch

from arc3.llm.backend import OpenAICompatibleBackend


class _Response:
    def __init__(self, payload): self.payload = payload
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def read(self): return json.dumps(self.payload).encode()


class BackendTests(unittest.TestCase):
    def test_vllm_base_url_and_qwen_wrappers(self):
        payload = {"choices": [{"message": {"content": '''<think>internal</think>
```json
{"mode":"explore","action":{"type":"ACTION1"}}
```'''}}],
                   "usage": {"prompt_tokens": 12, "completion_tokens": 8, "completion_tokens_details": {"reasoning_tokens": 3}}}
        seen = {}
        def fake_open(req, timeout):
            seen["url"] = req.full_url; seen["body"] = json.loads(req.data); seen["timeout"] = timeout
            return _Response(payload)
        backend = OpenAICompatibleBackend("http://127.0.0.1:1234/v1", "qwen", timeout=7)
        with patch("urllib.request.urlopen", fake_open):
            decision = backend.decide({"observation": {"grid": [[0]], "width": 1, "height": 1}})
        self.assertEqual(seen["url"], "http://127.0.0.1:1234/v1/chat/completions")
        self.assertEqual(seen["body"]["top_k"], 20)
        self.assertEqual(seen["body"]["chat_template_kwargs"], {"enable_thinking": True, "preserve_thinking": True})
        self.assertEqual(decision["action"]["type"], "ACTION1")
        self.assertEqual(backend.last_usage["reasoning_tokens"], 3)
        self.assertIsNotNone(backend.last_latency_ms)


if __name__ == "__main__": unittest.main()
