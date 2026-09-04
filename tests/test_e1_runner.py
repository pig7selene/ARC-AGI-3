import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kaggle.e1_qwen.e1_runner import DEFAULT_MODEL_ID, _health_url, _requested_game_ids
from kaggle.e1_qwen.model_metadata import collect_local_metadata


class E1RunnerTests(unittest.TestCase):
    def test_confirmed_qwen38_checkpoint_is_the_default(self):
        self.assertEqual(DEFAULT_MODEL_ID, "/kaggle/input/qwen3-8-27b-fp8-repacked-v1/pytorch/hf-fp8/1")

    def test_health_url_accepts_base_and_full_endpoint(self):
        self.assertEqual(_health_url("http://127.0.0.1:1234/v1"), "http://127.0.0.1:1234/v1/models")
        self.assertEqual(_health_url("http://127.0.0.1:1234"), "http://127.0.0.1:1234/v1/models")
        self.assertEqual(_health_url("http://127.0.0.1:1234/v1/chat/completions"), "http://127.0.0.1:1234/v1/models")

    def test_default_subset_is_fixed_and_full_set_is_opt_in(self):
        registry = {"a": object(), "b": object()}
        with patch.dict(os.environ, {"E1_GAME_SET": "small", "E1_GAME_IDS": ""}, clear=False):
            self.assertEqual(_requested_game_ids(registry), ("cd82-fb555c5d", "ls20-9607627b", "lf52-271a04aa"))
        with patch.dict(os.environ, {"E1_GAME_SET": "full", "E1_GAME_IDS": ""}, clear=False):
            self.assertEqual(_requested_game_ids(registry), ("a", "b"))

    def test_explicit_game_ids_are_deduplicated_in_order(self):
        with patch.dict(os.environ, {"E1_GAME_SET": "small", "E1_GAME_IDS": "b, a, b"}, clear=False):
            self.assertEqual(_requested_game_ids({}), ("b", "a"))

    def test_metadata_reads_checkpoint_config_without_guessing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text(json.dumps({
                "architectures": ["Qwen3ForCausalLM"],
                "model_type": "qwen3",
                "quantization_config": {"quant_method": "fp8"},
            }))
            (path / "tokenizer.json").write_text("{}")
            metadata = collect_local_metadata(
                path, served_model_name="served", max_model_len=32768,
                temperature=0.6, top_p=0.95, top_k=20, max_tokens=2048,
                enable_thinking=True, preserve_thinking=True,
                reasoning_parser="qwen3", tool_call_parser="qwen3_coder",
            )
            self.assertEqual(metadata["model_config"]["model_type"], "qwen3")
            self.assertEqual(metadata["model_config"]["quantization_config"]["quant_method"], "fp8")
            self.assertEqual(metadata["tokenizer_path"], str(path.resolve()))
            self.assertEqual(metadata["checkpoint"]["revision"], "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
