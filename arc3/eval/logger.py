from __future__ import annotations
import json
from pathlib import Path
class TraceLogger:
    def __init__(self,path,append=True): self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True); self.append=append
    def __call__(self,event):
        with self.path.open("a" if self.append else "w",encoding="utf8") as f: f.write(json.dumps(event,ensure_ascii=False,default=str)+"\n")
        self.append=True
    def write_summary(self,result):
        summary=self.summary(result)
        self.path.with_suffix(".summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf8")
        lines=[f"score: {summary['score']}",f"levels cleared: {summary['levels_cleared']}",f"actions: {summary['total_actions']}",f"no-op rate: {summary['no_op_rate']:.1%}",f"unique states: {summary['unique_states_visited']}",f"resets: {summary['resets']}",f"failure: {summary['failure_reason'] or 'none'}"]
        self.path.with_suffix(".summary.txt").write_text("\n".join(lines)+"\n",encoding="utf8")
    @staticmethod
    def summary(result):
        return {"levels_cleared":result.levels_cleared,"total_actions":result.actions,"score":result.score,"failure_reason":result.failure_reason,
          "no_op_rate":sum(e.get("actual_frame_diff")=="no visible effect" for e in result.trace)/result.actions if result.actions else 0.0,
          "unique_states_visited":len({e.get("frame_hash") for e in result.trace}),"resets":sum(e.get("chosen_action",{}).get("type")=="RESET" for e in result.trace)}
