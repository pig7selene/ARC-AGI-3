from __future__ import annotations
import argparse,time,json
from pathlib import Path
from .metrics import summarize
from .logger import TraceLogger
from arc3.agents.plumbing_agent import PlumbingAgent
from arc3.agents.reasoning_agent import ReasoningAgent
from arc3.env.mock import MoveToTargetEnv
from arc3.env.official import discover_public_games,get_last_arcade
from arc3.env.protocol import adapt_environment

def run_games(game_factories,agent_name="reasoning",max_actions=100,seed=0,trace_dir=None):
    agent=(PlumbingAgent if agent_name=="plumbing" else ReasoningAgent)(max_actions=max_actions); results=[]; started=time.perf_counter()
    for name,factory in game_factories.items():
        logger=TraceLogger(f"{trace_dir}/{name}.jsonl") if trace_dir else None
        environment=factory() if callable(factory) else factory
        result=agent.run(adapt_environment(environment),seed=seed,logger=logger)
        if logger: logger.write_summary(result)
        results.append(result)
    return summarize(results,time.perf_counter()-started)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--game"); p.add_argument("--all-games",action="store_true"); p.add_argument("--max-actions",type=int,default=100); p.add_argument("--seed",type=int,default=0); p.add_argument("--agent",choices=["plumbing","reasoning"],default="reasoning"); p.add_argument("--trace-dir",default="traces"); p.add_argument("--toolkit-mode",choices=["offline","normal","online","competition"],default=None); a=p.parse_args()
    games=discover_public_games(a.toolkit_mode,artifact_path="artifacts/public_games.json")
    if not games: games={"move_to_target":MoveToTargetEnv}
    if a.game: games={a.game:games[a.game]} if a.game in games else {}
    summary=run_games(games,a.agent,a.max_actions,a.seed,a.trace_dir)
    payload=summary.as_dict()
    arcade=get_last_arcade()
    if arcade is not None:
        try:
            card=arcade.get_scorecard(); payload["official_scorecard"]=card.model_dump() if hasattr(card,"model_dump") else str(card)
        except Exception as exc: payload["official_scorecard_error"]=str(exc)
    out=Path("results")/(f"{a.agent}_real.json" if games and "move_to_target" not in games else f"{a.agent}_mock.json"); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,indent=2,default=str),encoding="utf8")
    print(payload)
if __name__=="__main__": main()
