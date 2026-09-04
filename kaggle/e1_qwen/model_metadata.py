"""Capture E1 checkpoint and serving metadata without network access.

The E1 notebook is expected to run with an attached local HF-compatible model
directory.  Local config/tokenizer inspection is best effort and records
``UNKNOWN`` rather than guessing.  The optional ``/v1/models`` probe is only
against the local vLLM server.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any
import urllib.request


DEFAULT_MODEL_ID = "/kaggle/input/qwen3-8-27b-fp8-repacked-v1/pytorch/hf-fp8/1"
UNKNOWN = "UNKNOWN"


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _vllm_version() -> str:
    try:
        return importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        try:
            import vllm  # type: ignore
            return str(getattr(vllm, "__version__", UNKNOWN))
        except Exception:
            return UNKNOWN


def _tokenizer_path(model_path: Path) -> str:
    tokenizer_markers = (
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "tokenizer.model", "spiece.model", "vocab.json", "merges.txt",
    )
    if any((model_path / marker).exists() for marker in tokenizer_markers):
        return str(model_path)
    return UNKNOWN


def collect_local_metadata(
    model_path: str | os.PathLike[str],
    *,
    served_model_name: str,
    max_model_len: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    enable_thinking: bool,
    preserve_thinking: bool,
    reasoning_parser: str,
    tool_call_parser: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    path = Path(model_path).expanduser()
    try:
        actual_path = path.resolve()
    except OSError:
        actual_path = path
    config_path = path / "config.json"
    config = _read_json(config_path) if path.is_dir() else None
    quantization = (config or {}).get("quantization_config", UNKNOWN)
    if quantization in (None, ""):
        quantization = (config or {}).get("quantization", UNKNOWN)
    return {
        "checkpoint": {
            "model_id": str(model_path),
            "filesystem_path": str(actual_path),
            "exists": actual_path.is_dir(),
            "model_card": "Qwen/Qwen3.8-27B-FP8",
            "revision": UNKNOWN,
        },
        "served_model_name": served_model_name,
        "model_config": {
            "config_path": str(config_path.resolve()) if config_path.exists() else UNKNOWN,
            "architectures": (config or {}).get("architectures", UNKNOWN),
            "model_type": (config or {}).get("model_type", UNKNOWN),
            "torch_dtype": (config or {}).get("torch_dtype", UNKNOWN),
            "quantization_config": quantization,
        },
        "tokenizer_path": _tokenizer_path(actual_path) if actual_path.is_dir() else UNKNOWN,
        "vllm_version": _vllm_version(),
        "server_config": {
            "base_url": base_url or UNKNOWN,
            "max_model_len": max_model_len,
            "reasoning_parser": reasoning_parser,
            "tool_call_parser": tool_call_parser or UNKNOWN,
            "dtype": "auto",
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.92,
            "enable_prefix_caching": True,
            "trust_remote_code": True,
        },
        "sampling": {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": max_tokens,
        },
        "thinking": {
            "enable_thinking": enable_thinking,
            "preserve_thinking": preserve_thinking,
        },
        "evidence": {
            "local_config_inspection": str(config_path.resolve()) if config is not None else UNKNOWN,
            "unknowns_are_not_inferred": True,
        },
    }


def _models_url(base_url: str) -> str:
    value = str(base_url).rstrip("/")
    if value.endswith("/chat/completions"):
        return value[:-len("/chat/completions")] + "/models"
    return value + ("/models" if value.endswith("/v1") else "/v1/models")


def fetch_server_models(base_url: str, *, api_key: str | None = None, timeout: float = 5.0) -> dict[str, Any]:
    """Read the local vLLM model registry; return an explicit error object."""
    url = _models_url(base_url)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            return {"url": url, "error": "models response is not a JSON object"}
        return {"url": url, "response": payload}
    except Exception as exc:
        return {"url": url, "error": f"{type(exc).__name__}: {exc}"}


def write_metadata(metadata: dict[str, Any], path: str | os.PathLike[str]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=os.environ.get("E1_MODEL_ID", DEFAULT_MODEL_ID))
    parser.add_argument("--served-model-name", default=os.environ.get("E1_SERVED_MODEL_NAME"))
    parser.add_argument("--output", default=os.environ.get("E1_METADATA_PATH", "e1_model_metadata.json"))
    parser.add_argument("--base-url", default=os.environ.get("E1_BASE_URL", "http://127.0.0.1:1234/v1"))
    parser.add_argument("--max-model-len", type=int, default=int(os.environ.get("E1_MAX_MODEL_LEN", "32768")))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("E1_TEMPERATURE", "0.6")))
    parser.add_argument("--top-p", type=float, default=float(os.environ.get("E1_TOP_P", "0.95")))
    parser.add_argument("--top-k", type=int, default=int(os.environ.get("E1_TOP_K", "20")))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("E1_MAX_TOKENS", "2048")))
    args = parser.parse_args(argv)
    served = args.served_model_name or args.model_path
    metadata = collect_local_metadata(
        args.model_path, served_model_name=served, max_model_len=args.max_model_len,
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
        max_tokens=args.max_tokens,
        enable_thinking=os.environ.get("E1_ENABLE_THINKING", "true").lower() in {"1", "true", "yes", "on"},
        preserve_thinking=os.environ.get("E1_PRESERVE_THINKING", "true").lower() in {"1", "true", "yes", "on"},
        reasoning_parser=os.environ.get("E1_REASONING_PARSER", "qwen3"),
        tool_call_parser=os.environ.get("E1_TOOL_CALL_PARSER", "qwen3_coder"),
        base_url=args.base_url,
    )
    write_metadata(metadata, args.output)
    print(json.dumps(metadata, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
