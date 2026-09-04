#!/usr/bin/env python3
"""Compare an E0 Duck benchmark with an E1 result JSON.

E1's current ``summarize()`` payload stores ``per_game`` rows without a
``game_id``.  When that happens this tool maps rows to the top-level ``games``
list in order (or to the documented fixed-subset order as a fallback) and
records the assumption in the report.  It never silently invents a join key.

The E1 result is optional while the GPU run is pending:

    python analysis/compare_e0_e1.py --e0 /path/benchmark.json --e1 e1_results.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any


DEFAULT_E0 = Path(os.environ.get(
    "E0_BENCHMARK_PATH", "/Users/infiniteejl/Downloads/benchmark.json"
))
DEFAULT_E1 = Path(os.environ.get("E1_RESULT_PATH", "e1_results.json"))
DEFAULT_E1_GAMES = ("cd82-fb555c5d", "ls20-9607627b", "lf52-271a04aa")


def _number(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, f"file does not exist: {path}"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return None, f"cannot read {path}: {exc}"
    except json.JSONDecodeError as exc:
        return None, f"malformed JSON in {path}: {exc}"
    if not isinstance(value, dict):
        return None, f"JSON root must be an object: {path}"
    return value, None


def _e0_rows(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for game in payload.get("game_runs", []):
        if not isinstance(game, dict):
            continue
        game_id = str(game.get("game_id", "<missing>"))
        history = game.get("history") if isinstance(game.get("history"), list) else []
        rows[game_id] = {
            "game_id": game_id,
            "levels": int(_number(game.get("levels_completed"), 0) or 0),
            "score": _number(game.get("final_score"), 0.0),
            "actions": len([entry for entry in history if isinstance(entry, dict)]),
            "tokens": sum(int(_number(entry.get("generated_tokens"), 0) or 0)
                         for entry in history if isinstance(entry, dict)),
            "runtime": _number(game.get("final_wallclock_seconds")),
            "runtime_note": "observed final_wallclock_seconds; near benchmark cap may be censored",
            "source": "E0 benchmark game_runs",
        }
    return rows


def _e1_metric(row: dict[str, Any], key: str, default: Any = None) -> Any:
    if key in row:
        return row[key]
    diagnostics = row.get("diagnostics")
    if isinstance(diagnostics, dict):
        metrics = diagnostics.get("metrics")
        if isinstance(metrics, dict) and key in metrics:
            return metrics[key]
    return default


def _e1_rows(payload: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], str]:
    raw_rows = payload.get("per_game")
    if not isinstance(raw_rows, list):
        return {}, "missing E1 per_game list"
    explicit_ids = [str(row.get("game_id")) for row in raw_rows
                    if isinstance(row, dict) and row.get("game_id")]
    top_games = payload.get("games")
    if len(explicit_ids) == len(raw_rows) and len(set(explicit_ids)) == len(explicit_ids):
        ids = explicit_ids
        assumption = "per_game.game_id"
    elif isinstance(top_games, list) and len(top_games) >= len(raw_rows):
        ids = [str(item) for item in top_games[:len(raw_rows)]]
        assumption = "E1 top-level games order paired with per_game order; per_game has no game_id"
    elif len(raw_rows) <= len(DEFAULT_E1_GAMES):
        ids = list(DEFAULT_E1_GAMES[:len(raw_rows)])
        assumption = "documented fixed E1 subset order paired with per_game order; verify against run metadata"
    else:
        ids = [f"e1_row_{index + 1}" for index in range(len(raw_rows))]
        assumption = "synthetic row IDs because no E1 game mapping was present"
    result: dict[str, dict[str, Any]] = {}
    for game_id, row in zip(ids, raw_rows):
        if not isinstance(row, dict):
            row = {}
        diagnostics = row.get("diagnostics") if isinstance(row.get("diagnostics"), dict) else {}
        metrics = diagnostics.get("metrics") if isinstance(diagnostics.get("metrics"), dict) else {}
        runtime = row.get("runtime_seconds", row.get("runtime"))
        result[game_id] = {
            "game_id": game_id,
            "levels": int(_number(row.get("levels_cleared", row.get("levels")), 0) or 0),
            "score": _number(row.get("score"), 0.0),
            "actions": int(_number(row.get("actions"), 0) or 0),
            "tokens": int(_number(_e1_metric(row, "output_tokens", metrics.get("output_tokens")), 0) or 0),
            "runtime": _number(runtime),
            "runtime_note": "E1 per-game runtime not emitted by current summarize(); null unless supplied by a future result schema",
            "model_calls": int(_number(_e1_metric(row, "model_calls", metrics.get("model_calls")), 0) or 0),
            "source": "E1 per_game row",
        }
    return result, assumption


def compare(e0: dict[str, Any], e1: dict[str, Any] | None) -> dict[str, Any]:
    e0_rows = _e0_rows(e0)
    if e1 is None:
        rows = [{"game_id": game_id, "e0": row, "e1": None, "classification": "pending_e1"}
                for game_id, row in e0_rows.items()]
        return {
            "status": "pending_e1",
            "mapping_assumption": "E1 result unavailable; no cross-run classification performed",
            "per_game": rows,
            "categories": {"Duck-only wins": [], "Our-Agent-only wins": [], "both fail": [], "both progress": [], "pending": list(e0_rows)},
        }
    e1_rows, assumption = _e1_rows(e1)
    rows: list[dict[str, Any]] = []
    categories = {"Duck-only wins": [], "Our-Agent-only wins": [], "both fail": [], "both progress": [], "unmatched": []}
    for game_id in sorted(set(e0_rows) | set(e1_rows)):
        e0_row = e0_rows.get(game_id)
        e1_row = e1_rows.get(game_id)
        if e0_row is None or e1_row is None:
            classification = "unmatched"
            categories[classification].append(game_id)
        else:
            e0_progress = e0_row["levels"] > 0
            e1_progress = e1_row["levels"] > 0
            if e0_progress and not e1_progress:
                classification = "Duck-only wins"
            elif e1_progress and not e0_progress:
                classification = "Our-Agent-only wins"
            elif not e0_progress and not e1_progress:
                classification = "both fail"
            else:
                classification = "both progress"
            categories[classification].append(game_id)
        rows.append({"game_id": game_id, "e0": e0_row, "e1": e1_row, "classification": classification})
    return {"status": "ok", "mapping_assumption": assumption, "per_game": rows, "categories": categories}


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_table(report: dict[str, Any]) -> str:
    rows = []
    for item in report.get("per_game", []):
        e0 = item.get("e0") or {}
        e1 = item.get("e1") or {}
        rows.append([item["game_id"], e0.get("levels", "-"), e1.get("levels", "-"),
                     _fmt(e0.get("score")), _fmt(e1.get("score")),
                     e0.get("actions", "-"), e1.get("actions", "-"),
                     e0.get("tokens", "-"), e1.get("tokens", "-"),
                     _fmt(e0.get("runtime")), _fmt(e1.get("runtime")), item.get("classification", "-")])
    headers = ["game", "E0 lev", "E1 lev", "E0 score", "E1 score", "E0 act", "E1 act", "E0 tok", "E1 tok", "E0 sec", "E1 sec", "classification"]
    if not rows:
        return "(no comparable rows)"
    widths = [max(len(str(value)) for value in [header] + [row[index] for row in rows]) for index, header in enumerate(headers)]
    top = " | ".join(str(header).ljust(widths[index]) for index, header in enumerate(headers))
    divider = "-+-".join("-" * width for width in widths)
    body = [" | ".join(str(value).ljust(widths[index]) for index, value in enumerate(row)) for row in rows]
    return "\n".join([top, divider] + body)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e0", type=Path, default=DEFAULT_E0, help="E0 benchmark JSON path")
    parser.add_argument("--e1", type=Path, default=DEFAULT_E1, help="E1 result JSON path; missing is reported as pending")
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    args = parser.parse_args(argv)
    e0, e0_error = _read_json(args.e0)
    if e0_error or e0 is None or not isinstance(e0.get("game_runs"), list):
        report = {"status": "invalid_e0", "message": e0_error or "E0 JSON lacks game_runs list", "e0": str(args.e0)}
    else:
        e1, e1_error = _read_json(args.e1)
        if e1_error:
            report = compare(e0, None)
            report["e1_message"] = e1_error
        else:
            report = compare(e0, e1)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "per_game"}, ensure_ascii=False, indent=2))
    print("\nPer-game comparison:\n" + render_table(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
