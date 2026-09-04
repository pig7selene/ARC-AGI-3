"""Kaggle/local E1 runner: our ReasoningAgent against a local vLLM server."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
import urllib.request

from arc3.agents.reasoning_agent import ReasoningAgent
from arc3.env.official import discover_public_games, get_last_arcade
from arc3.env.protocol import adapt_environment
from arc3.eval.logger import TraceLogger
from arc3.eval.metrics import summarize
from arc3.llm.backend import OpenAICompatibleBackend
try:
    from .model_metadata import DEFAULT_MODEL_ID, collect_local_metadata, fetch_server_models, write_metadata
except ImportError:  # direct file loading from the Kaggle notebook bundle
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from model_metadata import DEFAULT_MODEL_ID, collect_local_metadata, fetch_server_models, write_metadata

DEFAULT_GAMES = ("cd82-fb555c5d", "ls20-9607627b", "lf52-271a04aa")


def _health_url(base_url: str) -> str:
    """Map either a vLLM base URL or a full completion URL to ``/models``."""
    value = str(base_url).rstrip("/")
    suffix = "/chat/completions"
    if value.endswith(suffix):
        return value[: -len(suffix)] + "/models"
    return value + ("/models" if value.endswith("/v1") else "/v1/models")


def _requested_game_ids(registry: dict) -> tuple[str, ...]:
    """Resolve the fixed small subset, an explicit list, or the full registry."""
    explicit = os.getenv("E1_GAME_IDS", "").strip()
    if explicit:
        values = [item.strip() for item in explicit.split(",") if item.strip()]
        return tuple(dict.fromkeys(values))
    game_set = os.getenv("E1_GAME_SET", "small").strip().lower()
    if game_set in {"full", "all"}:
        return tuple(registry)
    return DEFAULT_GAMES


def wait_for_server(base_url: str, timeout_seconds: int = 900, api_key: str | None = None) -> None:
    health_url = _health_url(base_url)
    deadline = time.time() + timeout_seconds
    last_error = ""
    while time.time() < deadline:
        try:
            request = urllib.request.Request(health_url, headers={"Authorization": f"Bearer {api_key}"} if api_key else {})
            with urllib.request.urlopen(request, timeout=5) as response:
                if response.status < 400:
                    return
        except Exception as exc:
            last_error = str(exc)
        time.sleep(2)
    raise TimeoutError(f"vLLM server did not become healthy at {health_url}: {last_error}")


def _write_parquet(rows: list[dict], path: Path) -> None:
    try:
        import pandas as pd
        try:
            pd.DataFrame(rows).to_parquet(path, index=False)
            return
        except (ImportError, ModuleNotFoundError, ValueError):
            pass
    except ImportError:
        pass
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        pq.write_table(pa.Table.from_pylist(rows), path)
        return
    except ImportError as exc:
        raise RuntimeError("Kaggle image needs pandas or pyarrow to write submission.parquet") from exc


def main() -> int:
    base_url = os.getenv("E1_BASE_URL", "http://127.0.0.1:1234/v1")
    # The confirmed E0 Kaggle input is the default. Any alternate model must
    # be an explicit E1_MODEL_ID override; there is no historical Qwen3.6
    # fallback path.
    model = os.getenv("E1_MODEL_ID", DEFAULT_MODEL_ID).strip() or DEFAULT_MODEL_ID
    served_model_name = os.getenv("E1_SERVED_MODEL_NAME", model).strip() or model
    toolkit_mode = os.getenv("E1_TOOLKIT_MODE", "OFFLINE")
    action_budget = int(os.getenv("E1_MAX_ACTIONS", "30"))
    max_model_len = int(os.getenv("E1_MAX_MODEL_LEN", "32768"))
    temperature = float(os.getenv("E1_TEMPERATURE", "0.6"))
    top_p = float(os.getenv("E1_TOP_P", "0.95"))
    top_k = int(os.getenv("E1_TOP_K", "20"))
    max_tokens = int(os.getenv("E1_MAX_TOKENS", "2048"))
    enable_thinking = os.getenv("E1_ENABLE_THINKING", "true").lower() in {"1", "true", "yes", "on"}
    preserve_thinking = os.getenv("E1_PRESERVE_THINKING", "true").lower() in {"1", "true", "yes", "on"}
    reasoning_parser = os.getenv("E1_REASONING_PARSER", "qwen3")
    tool_call_parser = os.getenv("E1_TOOL_CALL_PARSER", "qwen3_coder")
    model_path = Path(model).expanduser()
    if not model_path.is_dir():
        raise SystemExit(f"E1_MODEL_ID must be an existing HF checkpoint directory: {model}")
    wait_for_server(base_url, int(os.getenv("E1_SERVER_WAIT_SECONDS", "900")), os.getenv("E1_MODEL_API_KEY"))
    metadata = collect_local_metadata(
        model, served_model_name=served_model_name, max_model_len=max_model_len,
        temperature=temperature, top_p=top_p, top_k=top_k, max_tokens=max_tokens,
        enable_thinking=enable_thinking, preserve_thinking=preserve_thinking,
        reasoning_parser=reasoning_parser, tool_call_parser=tool_call_parser,
        base_url=base_url,
    )
    metadata["served_models"] = fetch_server_models(base_url, api_key=os.getenv("E1_MODEL_API_KEY"))
    models_response = metadata["served_models"].get("response", {}) if isinstance(metadata.get("served_models"), dict) else {}
    served_ids = [item.get("id") for item in models_response.get("data", []) if isinstance(item, dict) and item.get("id")]
    metadata["served_model_name_observed"] = served_ids or "UNKNOWN"
    metadata["served_model_name_match"] = (served_model_name in served_ids) if served_ids else "UNKNOWN"
    metadata_path = Path(os.getenv("E1_METADATA_PATH", "e1_model_metadata.json"))
    write_metadata(metadata, metadata_path)
    print("E1 model metadata:")
    print(json.dumps(metadata, indent=2, ensure_ascii=False, default=str))
    if served_ids and served_model_name not in served_ids:
        raise RuntimeError(f"vLLM served model mismatch: requested {served_model_name!r}, available {served_ids!r}")
    registry = discover_public_games(toolkit_mode, artifact_path="artifacts/public_games.json")
    requested = _requested_game_ids(registry)
    selected = {name: registry[name] for name in requested if name in registry}
    if len(selected) != len(requested):
        missing = sorted(set(requested) - set(selected))
        raise RuntimeError(f"Required E1 games are missing from the toolkit registry: {missing}")
    backend = OpenAICompatibleBackend(base_url=base_url, model=served_model_name,
                                     api_key=os.getenv("E1_MODEL_API_KEY"),
                                     # Match Duck's published sampling setting for the
                                     # first controlled model-infrastructure comparison.
                                     temperature=temperature, top_p=top_p, top_k=top_k,
                                     max_tokens=max_tokens, enable_thinking=enable_thinking,
                                     preserve_thinking=preserve_thinking)
    results = []
    all_rows = []
    started = time.perf_counter()
    for name, factory in selected.items():
        logger = TraceLogger(Path("e1_traces") / f"{name}.jsonl", append=False)
        result = ReasoningAgent(backend, max_actions=action_budget, max_plan_steps=3, enable_prediction=True).run(adapt_environment(factory()), seed=0, logger=logger)
        logger.write_summary(result)
        results.append(result)
        all_rows.extend({"game_id": name, "step": e.get("step"), "level": e.get("level"), "action": (e.get("chosen_action") or {}).get("type"), "frame_hash": e.get("frame_hash"), "mode": e.get("mode"), "goal_hypotheses_created": e.get("goal_hypotheses_created", 0), "plan_generated": bool(e.get("plan_generated")), "prediction_mismatch": bool(e.get("prediction_mismatch")), "progress_events": ",".join(e.get("progress_events", []))} for e in result.trace)
    payload = summarize(results, time.perf_counter() - started).as_dict()
    payload.update({"experiment": "E1_our_agent_qwen38", "model": model, "model_path": model,
                    "served_model_name": served_model_name, "model_metadata_path": str(metadata_path),
                    "model_metadata": metadata,
                    "base_url": base_url,
                    "toolkit_mode": toolkit_mode, "seed": 0, "max_actions": action_budget,
                    "game_set": os.getenv("E1_GAME_SET", "small"), "games": list(requested),
                    "sampling": {"temperature": temperature, "top_p": top_p, "top_k": top_k, "max_tokens": max_tokens},
                    "thinking": {"enable_thinking": enable_thinking, "preserve_thinking": preserve_thinking},
                    "reasoning_parser": reasoning_parser, "tool_call_parser": tool_call_parser,
                    "max_model_len": max_model_len})
    arcade = get_last_arcade()
    if arcade is not None:
        try:
            card = arcade.get_scorecard()
            payload["official_scorecard"] = card.model_dump() if hasattr(card, "model_dump") else str(card)
        except Exception as exc:
            payload["official_scorecard_error"] = str(exc)
    Path("e1_results.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf8")
    _write_parquet(all_rows, Path("submission.parquet"))
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
