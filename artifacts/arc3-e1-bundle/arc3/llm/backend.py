"""Replaceable reasoning backends. Remote calls are optional; offline heuristic is default."""
from __future__ import annotations
from abc import ABC, abstractmethod
import json, time, urllib.request
from dataclasses import dataclass
from typing import Any


def _chat_completions_url(base_url: str) -> str:
    """Accept either a vLLM `/v1` base URL or a full endpoint URL."""
    value = str(base_url).rstrip("/")
    return value if value.endswith("/chat/completions") else value + "/chat/completions"


def _decode_json_content(content: Any) -> dict[str, Any]:
    """Decode common Qwen/vLLM JSON wrappers without retaining hidden thought."""
    if isinstance(content, list):
        content = "".join(str(item.get("text", "")) for item in content if isinstance(item, dict) and item.get("type") == "text")
    if isinstance(content, dict):
        return content
    text = str(content or "").strip()
    if "</think>" in text.lower():
        text = text[text.lower().rfind("</think>") + len("</think>"):].strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines and lines[-1].strip().startswith("```") else lines[1:]).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("model response JSON must be an object")
    return value

class BaseLLMBackend(ABC):
    @abstractmethod
    def decide(self, context: dict[str, Any]) -> dict[str, Any]: ...

class HeuristicBackend(BaseLLMBackend):
    """Deterministic fallback useful for plumbing and offline tests."""
    def __init__(self, reuse_action_outcomes: bool = True):
        self.reuse_action_outcomes = reuse_action_outcomes

    def decide(self, context):
        valid=context.get("valid_actions",[]); memory=context.get("memory",{})
        outcomes=memory.get("action_outcomes",{})
        # Counterfactual reuse: do not repeat an action type once its outcome
        # is already known while another valid type remains untested.
        untested=[a for a in valid if not self.reuse_action_outcomes or (not outcomes.get(a,{}).get("effect",0) and not outcomes.get(a,{}).get("no_op",0))]
        unknown=[a for a in untested if a not in context.get("no_op_actions",[])]
        action=(unknown or untested or [a for a in valid if a not in context.get("no_op_actions",[])] or valid or ["ACTION1"])[0]
        known=outcomes.get(action,{})
        expected_visible = False if known.get("no_op",0) and not known.get("effect",0) else True
        return {"mode":"explore","analysis_summary":"select highest-information untested action",
                "updated_hypotheses":[],"plan":[],"action":{"type":action},"confidence":0.25,
                "expected_effect":"discover whether action changes the frame",
                "expectation":{"visible_effect":expected_visible}}

@dataclass
class OpenAICompatibleBackend(BaseLLMBackend):
    base_url: str = "http://localhost:8000/v1/chat/completions"
    model: str = "local-model"
    api_key: str | None = None
    timeout: float = 30.0
    temperature: float = 0.0
    top_p: float = 0.95
    top_k: int = 20
    max_tokens: int = 2048
    enable_thinking: bool = True
    preserve_thinking: bool = True
    system_prompt: str = ("You are a careful ARC-AGI-3 interactive reasoning agent. Return JSON only. "
                          "Use action:{type,x,y} and mode explore|execute|recover. "
                          "State a concise falsifiable goal_hypothesis when one is plausible, and distinguish explore from execute. "
                          "When you can make a falsifiable prediction, include expectation "
                          "with visible_effect and/or changed_cells_min/changed_cells_max. "
                          "Optionally include a plan of at most 3 action items, each with action and expectation. "
                          "Do not invent coordinates outside the observed frame.")
    def __post_init__(self):
        self.last_usage: dict[str, Any] = {}
        self.last_latency_ms: float | None = None

    def decide(self, context):
        started = time.perf_counter()
        self.last_usage = {}
        body={"model":self.model,"temperature":self.temperature,"top_p":self.top_p,"messages":[{"role":"system","content":self.system_prompt},
          {"role":"user","content":json.dumps(context,separators=(",",":"))}],"response_format":{"type":"json_object"},"max_tokens":self.max_tokens}
        if self.top_k > 0: body["top_k"] = self.top_k
        # vLLM's Qwen chat template accepts this extra body; providers that do
        # not understand it can disable it at construction time.
        body["chat_template_kwargs"] = {
            "enable_thinking": bool(self.enable_thinking),
            "preserve_thinking": bool(self.preserve_thinking),
        }
        req=urllib.request.Request(_chat_completions_url(self.base_url),data=json.dumps(body).encode(),headers={"Content-Type":"application/json",**(({"Authorization":f"Bearer {self.api_key}"} if self.api_key else {}))})
        try:
            with urllib.request.urlopen(req,timeout=self.timeout) as response:
                payload=json.loads(response.read().decode())
            message=payload["choices"][0]["message"]
            content=message.get("content")
            if content in (None, ""):
                content=message.get("reasoning_content", "")
            self.last_usage = dict(payload.get("usage") or {})
            usage_details=self.last_usage.get("completion_tokens_details") or self.last_usage.get("output_tokens_details") or {}
            if isinstance(usage_details, dict) and usage_details.get("reasoning_tokens") is not None:
                self.last_usage["reasoning_tokens"] = usage_details.get("reasoning_tokens")
            return _decode_json_content(content)
        finally:
            self.last_latency_ms = round((time.perf_counter() - started) * 1000, 3)
