"""Run experiment 002 with a real OpenAI-compatible reasoning backend.

The endpoint is deliberately supplied through environment variables so model
credentials never enter the repository or trace files.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

from arc3.agents.reasoning_agent import ReasoningAgent
from arc3.eval.logger import TraceLogger
from arc3.eval.metrics import summarize
from arc3.env.official import discover_public_games, get_last_arcade
from arc3.env.protocol import adapt_environment
from arc3.llm.backend import OpenAICompatibleBackend

GAMES = ("cd82-fb555c5d", "ls20-9607627b", "lf52-271a04aa")


def main() -> int:
    base_url = os.getenv("ARC3_MODEL_BASE_URL")
    model = os.getenv("ARC3_MODEL")
    if not base_url or not model:
        raise SystemExit("Set ARC3_MODEL_BASE_URL and ARC3_MODEL (and optionally ARC3_MODEL_API_KEY) before running.")
    registry = discover_public_games("OFFLINE", artifact_path="artifacts/public_games.json")
    selected = {name: registry[name] for name in GAMES if name in registry}
    backend = OpenAICompatibleBackend(base_url=base_url, model=model, api_key=os.getenv("ARC3_MODEL_API_KEY"))
    results = []
    started = time.perf_counter()
    for name, factory in selected.items():
        logger = TraceLogger(Path("traces/v2/model") / f"{name}.jsonl", append=False)
        result = ReasoningAgent(backend, max_actions=30, max_plan_steps=3, enable_prediction=True).run(adapt_environment(factory()), seed=0, logger=logger)
        logger.write_summary(result)
        results.append(result)
    payload = summarize(results, time.perf_counter() - started).as_dict()
    payload.update({"experiment": "v2_semantic_policy_h6", "variant": "model", "model": model, "base_url": base_url})
    arcade = get_last_arcade()
    if arcade is not None:
        try:
            card = arcade.get_scorecard()
            payload["official_scorecard"] = card.model_dump() if hasattr(card, "model_dump") else str(card)
        except Exception as exc:
            payload["official_scorecard_error"] = str(exc)
    Path("results/v2").mkdir(parents=True, exist_ok=True)
    Path("results/v2/model_policy.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf8")
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

