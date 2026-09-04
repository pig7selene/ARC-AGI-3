"""Reproducible baseline/intervention run for the first V2 experiment."""
from __future__ import annotations

import json
from pathlib import Path
import time

from arc3.agents.reasoning_agent import ReasoningAgent
from arc3.eval.logger import TraceLogger
from arc3.eval.metrics import summarize
from arc3.env.official import discover_public_games, get_last_arcade
from arc3.env.protocol import adapt_environment
from arc3.llm.backend import HeuristicBackend

GAMES = ("cd82-fb555c5d", "ls20-9607627b", "lf52-271a04aa")


def run(label: str, intervention: bool, out_dir: Path):
    registry = discover_public_games("OFFLINE", artifact_path="artifacts/public_games.json")
    selected = {name: registry[name] for name in GAMES if name in registry}
    backend = HeuristicBackend(reuse_action_outcomes=intervention)
    agent = ReasoningAgent(backend, max_actions=30, max_plan_steps=3, enable_prediction=intervention)
    results = []
    started = time.perf_counter()
    for name, factory in selected.items():
        logger = TraceLogger(out_dir / label / f"{name}.jsonl", append=False)
        result = agent.run(adapt_environment(factory()), seed=0, logger=logger)
        logger.write_summary(result)
        results.append(result)
    summary = summarize(results, time.perf_counter() - started).as_dict()
    arcade = get_last_arcade()
    if arcade is not None:
        try:
            card = arcade.get_scorecard()
            summary["official_scorecard"] = card.model_dump() if hasattr(card, "model_dump") else str(card)
        except Exception as exc:
            summary["official_scorecard_error"] = str(exc)
    summary["experiment"] = "v2_retrodict_plan_h2_h3"
    summary["variant"] = label
    return summary


if __name__ == "__main__":
    root = Path("results/v2")
    payload = {"baseline": run("baseline", False, Path("traces/v2")),
               "intervention": run("intervention", True, Path("traces/v2"))}
    root.mkdir(parents=True, exist_ok=True)
    (root / "retrodict_plan.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf8")
    print(json.dumps(payload, indent=2, default=str))
