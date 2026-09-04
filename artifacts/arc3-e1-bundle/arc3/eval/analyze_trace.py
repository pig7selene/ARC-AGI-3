from __future__ import annotations
import argparse,json
from pathlib import Path
from .diagnostics import analyze_trace

def main():
    p=argparse.ArgumentParser(description="Compact deterministic ARC-AGI-3 trace diagnostics"); p.add_argument("trace",type=Path); a=p.parse_args()
    rows=[json.loads(line) for line in a.trace.read_text(encoding="utf8").splitlines() if line.strip()]
    d=analyze_trace(rows,completed=bool(rows and rows[-1].get("game_completion")),max_actions=len(rows))
    print("GAME",rows[0].get("game_id",a.trace.stem) if rows else a.trace.stem)
    print("RESULT", "complete" if rows and rows[-1].get("game_completion") else "incomplete")
    print("ACTIONS",d.metrics["actions"],"LEVELS",sum(bool(r.get("level_completion")) for r in rows))
    print("NO-OP RATE",f"{d.metrics['no_op_rate']:.1%}","REPEATED STATE RATE",f"{d.metrics['repeated_state_rate']:.1%}")
    print("PHASES",json.dumps(d.metrics["actions_by_phase"],sort_keys=True))
    print("FAILURE",d.primary_failure)
    if d.secondary_failures: print("SECONDARY",", ".join(d.secondary_failures))
    if d.evidence: print("EVIDENCE", "; ".join(d.evidence[:5]))
    print("METRICS",json.dumps(d.metrics,sort_keys=True))
if __name__=="__main__": main()
