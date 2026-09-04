"""Minimal offline submission entry point.

The competition notebook can call this file after packaging the current `arc3`
directory and a local OpenAI-compatible server/model. Official scorecard and
submission serialization remain owned by the installed toolkit.
"""
from __future__ import annotations
import os
from arc3.agents.reasoning_agent import ReasoningAgent
from arc3.env.official import discover_public_games
from arc3.env.protocol import adapt_environment
from arc3.eval.runner import run_games
from arc3.llm.backend import HeuristicBackend, OpenAICompatibleBackend

def main():
    games=discover_public_games()
    if not games: raise RuntimeError("No official public games found; install arc-agi in the Kaggle image")
    url=os.getenv("ARC_LLM_URL","http://127.0.0.1:8000/v1/chat/completions")
    model=os.getenv("ARC_LLM_MODEL","local-model")
    backend=OpenAICompatibleBackend(url,model) if os.getenv("ARC_LLM_URL") else HeuristicBackend()
    # Keep this entry point explicit: toolkit competition mode should supply the
    # official game factory and scorer in the notebook around this call.
    print(run_games(games,"reasoning",max_actions=int(os.getenv("ARC_MAX_ACTIONS","100")),trace_dir="submission_traces"))

if __name__ == "__main__": main()
