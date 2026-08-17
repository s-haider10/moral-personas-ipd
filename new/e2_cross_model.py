"""Compatibility helpers for Phase 1 scripts in this directory.

The original experiment primitives live in ``experiments/e2_cross_model.py``.
The newer E10/E12/E13 scripts import ``e2_cross_model`` from ``new/`` and use a
slightly friendlier client API, so this module adapts the old helpers without
changing the locked experiment scripts in ``experiments/``.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_E2_PATH = _ROOT / "experiments" / "e2_cross_model.py"
_SPEC = importlib.util.spec_from_file_location("_legacy_e2_cross_model", _E2_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"could not load {_E2_PATH}")
_legacy = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_legacy)

PAYOFFS = _legacy.PAYOFFS
PERSONA_PROMPT = _legacy.PERSONA_PROMPT
PROMPT_TEMPLATE = _legacy.PROMPT_TEMPLATE
opponent_action = _legacy.opponent_action
normative_action = _legacy.normative_action

MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
]


class ClientAdapter:
    def __init__(self, client):
        self._client = client
        self.provider = client.provider
        self.model = client.model

    def generate(self, prompt, temperature=1.0, seed=None):
        return self._client.generate(prompt, temperature=temperature)

    def __call__(self, prompt, temperature=1.0, seed=None):
        return self.generate(prompt, temperature=temperature, seed=seed)


def infer_provider(model):
    if model.startswith("gemini"):
        return "gemini"
    if model.startswith("gpt"):
        return "openai"
    if model.startswith("claude"):
        return "anthropic"
    raise ValueError(f"cannot infer provider from model name: {model}")


def make_client(provider_or_model, model=None):
    if model is None:
        model = provider_or_model
        provider = infer_provider(model)
    else:
        provider = provider_or_model
    return ClientAdapter(_legacy.make_client(provider, model))


def format_history(history):
    if not history:
        return _legacy.format_history(history)
    first = history[0]
    if not isinstance(first, dict):
        return _legacy.format_history(history)

    lines = ["History of prior rounds:"]
    for rec in history:
        mine = rec.get("agent_action")
        theirs = rec.get("opp_action", rec.get("opponent_action"))
        my_pay = rec.get("payoff_agent", rec.get("agent_payoff"))
        their_pay = rec.get("payoff_opp", rec.get("opp_payoff"))
        lines.append(
            f"  Round {rec['round']}: you played {mine}, opponent played {theirs}. "
            f"You earned {my_pay}, opponent earned {their_pay}."
        )
    return "\n".join(lines)


def parse_response(text):
    if text is None:
        return None, ""
    match = re.search(r"ACTION:\s*([CD])", text, re.IGNORECASE)
    action = match.group(1).upper() if match else None
    justification = (
        text.split("ACTION:")[0].strip() if "ACTION:" in text else text.strip()
    )
    return action, justification
