#!/usr/bin/env python3
"""Offline forensics for a Duck ARC-AGI-3 benchmark JSON.

The benchmark format contains environment action history and aggregate game
fields, but not model-request IDs or transcript boundaries.  This module keeps
those limitations explicit: a zero-token entry is counted and classified as
*compatible with* a queued/batched continuation, never as proof that a
particular entry came from a queue.

Examples
--------
    python analysis/e0_benchmark_analysis.py --input /path/benchmark.json
    python analysis/e0_benchmark_analysis.py --output /tmp/e0-analysis.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path(os.environ.get(
    "E0_BENCHMARK_PATH", "/Users/infiniteejl/Downloads/benchmark.json"
))


class BenchmarkInputError(ValueError):
    """Raised for a missing or structurally invalid benchmark file."""


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _integer(value: Any, default: int = 0) -> int:
    return int(_number(value, default))


def _action_id(entry: dict[str, Any]) -> str:
    action = entry.get("action")
    if isinstance(action, dict):
        value = action.get("id", action.get("type"))
        if value is not None:
            return str(value)
    if action is None:
        return "<missing>"
    return str(action)


def _action_key(entry: dict[str, Any]) -> str:
    """Canonical action identity, including ACTION6 coordinates."""
    action = entry.get("action")
    try:
        return json.dumps(action, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return repr(action)


def _action_runs(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = [_action_key(entry) for entry in entries]
    if not keys:
        return []
    result: list[dict[str, Any]] = []
    start = 0
    for index in range(1, len(keys) + 1):
        if index == len(keys) or keys[index] != keys[start]:
            result.append({
                "action": _action_id(entries[start]),
                "action_key": keys[start],
                "start": start + 1,
                "end": index,
                "length": index - start,
            })
            start = index
    return result


def _runs_for_values(values: list[str]) -> list[int]:
    """Return run lengths for a sequence (used for type and exact identity)."""
    if not values:
        return []
    lengths: list[int] = []
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] != values[start]:
            lengths.append(index - start)
            start = index
    return lengths


def load_benchmark(path: str | os.PathLike[str]) -> dict[str, Any]:
    candidate = Path(path).expanduser()
    if not candidate.exists():
        raise BenchmarkInputError(f"benchmark file does not exist: {candidate}")
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except OSError as exc:
        raise BenchmarkInputError(f"cannot read benchmark file {candidate}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise BenchmarkInputError(f"malformed JSON in {candidate}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("game_runs"), list):
        raise BenchmarkInputError("benchmark JSON must be an object containing a game_runs list")
    return payload


def _levels(game: dict[str, Any]) -> tuple[int, str]:
    if game.get("levels_completed") is not None:
        return _integer(game.get("levels_completed")), "levels_completed"
    actions_per_level = game.get("actions_per_level")
    if isinstance(actions_per_level, list):
        # Fallback only.  A non-zero entry usually denotes an attempted level,
        # not necessarily a completed level, so label this inference clearly.
        return sum(1 for item in actions_per_level if _integer(item) > 0), "inferred_from_actions_per_level"
    return 0, "missing"


def analyze_game(game: dict[str, Any], *, benchmark_max_wallclock: float | None = None) -> dict[str, Any]:
    history = game.get("history")
    if not isinstance(history, list):
        history = []
    entries = [item for item in history if isinstance(item, dict)]
    tokens = [_integer(item.get("generated_tokens")) for item in entries]
    actions = [_action_id(item) for item in entries]
    runs = _action_runs(entries)
    exact_keys = [_action_key(item) for item in entries]
    levels, levels_source = _levels(game)
    score = _number(game.get("final_score", game.get("score")))
    observed_wallclock = _number(game.get("final_wallclock_seconds"), default=float("nan"))
    history_wallclock = _number(entries[-1].get("wallclock_seconds"), default=float("nan")) if entries else float("nan")
    if not math.isfinite(observed_wallclock):
        observed_wallclock = history_wallclock if math.isfinite(history_wallclock) else None
    if benchmark_max_wallclock and observed_wallclock is not None:
        censored = observed_wallclock >= 0.95 * benchmark_max_wallclock
    else:
        censored = False
    action_counts: dict[str, int] = {}
    for action in actions:
        action_counts[action] = action_counts.get(action, 0) + 1
    zero_count = sum(token == 0 for token in tokens)
    nonzero_count = sum(token > 0 for token in tokens)
    adjacent_repeats = sum(1 for left, right in zip(entries, entries[1:])
                           if _action_key(left) == _action_key(right))
    type_repeated_steps = sum(1 for left, right in zip(actions, actions[1:]) if left == right)
    # The historical E0 tables used action IDs for the streak statistic. Keep
    # that metric, and expose exact payload streaks separately for ACTION6
    # coordinates.
    longest = max(_runs_for_values(actions), default=0)
    longest_exact = max(_runs_for_values(exact_keys), default=0)
    return {
        "game_id": str(game.get("game_id", "<missing>")),
        "state": game.get("state"),
        "number_of_levels": game.get("number_of_levels"),
        "levels_completed": levels,
        "levels": levels,
        "levels_source": levels_source,
        "score": score,
        "environment_actions": len(entries),
        "actions": len(entries),
        "generated_tokens": sum(tokens),
        "tokens": sum(tokens),
        "tokens_per_action": (sum(tokens) / len(entries)) if entries else None,
        "actions_per_completed_level": (len(entries) / levels) if levels else None,
        "zero_token_actions": zero_count,
        "model_generating_actions": nonzero_count,
        "zero_token_fraction": zero_count / len(entries) if entries else None,
        "repeated_identical_actions": adjacent_repeats,
        "repeated_action_type_steps": type_repeated_steps,
        "longest_identical_action_streak": longest,
        "longest_exact_action_streak": longest_exact,
        "action6_count": action_counts.get("ACTION6", 0),
        "reset_count": action_counts.get("RESET", 0),
        "ACTION6_count": action_counts.get("ACTION6", 0),
        "RESET_count": action_counts.get("RESET", 0),
        "action_counts": action_counts,
        "action_runs": runs,
        "wallclock_seconds": observed_wallclock,
        "history_last_wallclock_seconds": history_wallclock if math.isfinite(history_wallclock) else None,
        "wallclock_is_likely_censored": censored,
        "zero_token_action_classification": (
            "source-backed-compatible-with-queued-or-batched-continuation; "
            "individual-entry-origin-not-identifiable-from-benchmark"
        ),
    }


def analyze_benchmark(payload: dict[str, Any]) -> dict[str, Any]:
    games = payload.get("game_runs", [])
    valid_games = [game for game in games if isinstance(game, dict)]
    final_wallclocks = [_number(game.get("final_wallclock_seconds"), default=float("nan"))
                        for game in valid_games]
    finite_wallclocks = [value for value in final_wallclocks if math.isfinite(value)]
    max_wallclock = max(finite_wallclocks, default=None)
    per_game = [analyze_game(game, benchmark_max_wallclock=max_wallclock) for game in valid_games]
    successes = [game for game in per_game if game["levels_completed"] > 0]
    failures = [game for game in per_game if game["levels_completed"] <= 0]
    total_actions = sum(game["environment_actions"] for game in per_game)
    total_tokens = sum(game["generated_tokens"] for game in per_game)
    return {
        "status": "ok",
        "source": {
            "label": payload.get("label"),
            "solver_label": payload.get("solver_label"),
            "n_passes": payload.get("n_passes"),
            "start_time": payload.get("start_time"),
            "end_time": payload.get("end_time"),
            "record_intermediate_states": any(bool(game.get("record_intermediate_states")) for game in valid_games),
            "wallclock_note": "Per-game final wallclock values near the benchmark maximum are flagged as likely censored/cap values; they are not treated as independent elapsed durations.",
        },
        "overall": {
            "games": len(per_game),
            "games_with_at_least_one_completed_level": len(successes),
            "games_with_zero_completed_levels": len(failures),
            "total_completed_levels": sum(game["levels_completed"] for game in per_game),
            "total_environment_actions": total_actions,
            "total_generated_tokens": total_tokens,
            "tokens_per_action": total_tokens / total_actions if total_actions else None,
            "success_game_ids": [game["game_id"] for game in successes],
            "failure_game_ids": [game["game_id"] for game in failures],
            "mean_actions_success_games": (sum(game["environment_actions"] for game in successes) / len(successes)) if successes else None,
            "mean_actions_failure_games": (sum(game["environment_actions"] for game in failures) / len(failures)) if failures else None,
            "mean_zero_token_fraction_success_games": (sum(game["zero_token_fraction"] for game in successes if game["zero_token_fraction"] is not None) / len([game for game in successes if game["zero_token_fraction"] is not None])) if any(game["zero_token_fraction"] is not None for game in successes) else None,
            "mean_zero_token_fraction_failure_games": (sum(game["zero_token_fraction"] for game in failures if game["zero_token_fraction"] is not None) / len([game for game in failures if game["zero_token_fraction"] is not None])) if any(game["zero_token_fraction"] is not None for game in failures) else None,
        },
        "per_game": per_game,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_table(report: dict[str, Any]) -> str:
    rows = [[
        game["game_id"], game["levels_completed"], _fmt(game["score"]),
        game["environment_actions"], game["generated_tokens"],
        game["zero_token_actions"], game["model_generating_actions"],
        game["repeated_identical_actions"], game["longest_identical_action_streak"],
        game["longest_exact_action_streak"],
        game["action6_count"], game["reset_count"],
    ] for game in report.get("per_game", [])]
    headers = ["game", "levels", "score", "actions", "tokens", "zero", "model>0", "repeat-pairs", "max-type-streak", "max-exact-streak", "A6", "RESET"]
    widths = [max(len(str(value)) for value in [header] + [row[index] for row in rows]) for index, header in enumerate(headers)]
    line = " | ".join(str(header).ljust(widths[index]) for index, header in enumerate(headers))
    divider = "-+-".join("-" * width for width in widths)
    body = [" | ".join(str(value).ljust(widths[index]) for index, value in enumerate(row)) for row in rows]
    return "\n".join([line, divider] + body)


def _error_report(status: str, message: str, path: Path) -> dict[str, Any]:
    return {"status": status, "input": str(path), "message": message}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="benchmark JSON path (or E0_BENCHMARK_PATH)")
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    args = parser.parse_args(argv)
    try:
        report = analyze_benchmark(load_benchmark(args.input))
    except BenchmarkInputError as exc:
        report = _error_report("missing" if not args.input.exists() else "malformed", str(exc), args.input)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report.get("overall", report), ensure_ascii=False, indent=2))
    if report.get("status") == "ok":
        print("\nPer-game summary:\n" + render_table(report))
    else:
        print(f"\nE0 benchmark analysis status: {report['status']}: {report['message']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
