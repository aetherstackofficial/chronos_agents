"""LLM strategy copilot (PROTOCOL_V2.md §4): natural language → SDK strategy.

Primary path is Gemini (``gemini-2.5-flash`` via ``google-genai``'s async
client, 12 s budget — same client pattern as ``oracle.py``). The returned code
is VALIDATED (fences stripped, ``compile()``, ``class UserStrategy`` and
``def on_tick`` present). On any failure — missing key, timeout, API error,
invalid code — a REAL deterministic template is built from hints parsed out of
the description (RSI/VWAP thresholds, side, qty). The result is always
runnable ``class UserStrategy(Strategy)`` code, never a stub. The copilot never
auto-runs code; the user reviews and saves/runs it via the script endpoints.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("chronos.features")

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_TIMEOUT = 12.0

# Template defaults (VWAP mean-reversion with RSI gates).
DEFAULT_RSI_BUY = 35.0
DEFAULT_RSI_SELL = 65.0
DEFAULT_QTY = 50
MAX_QTY = 1000
POS_CAP_MULTIPLIER = 10  # MAX_POS = qty * this

_PROMPT_TEMPLATE = (
    "Write the body of a Python class UserStrategy(Strategy) for the Chronos "
    "trading SDK. It must subclass Strategy (imported `from runner.sdk import "
    "Strategy`), implement on_tick(self, state)->list where state has keys "
    "last_price, vwap, rsi, ofi, mid_price, best_bid, best_ask, regime, and "
    "return a list of order dicts like "
    "{{'action':'BUY'|'SELL','type':'MARKET'|'LIMIT','qty':int,'price':float?}}. "
    "Track position via on_fill(self, ack) reading ack['pos']. "
    "Description: {description}. Return ONLY python code."
)

_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_+-]*[ \t]*\n?|```[ \t]*$", re.MULTILINE)


def _strip_fences(text: str) -> str:
    """Remove markdown code fences (```python ... ```) from an LLM reply."""
    return _FENCE_RE.sub("", text or "").strip()


def _validate_code(code: str) -> Optional[str]:
    """Return normalized runnable code, or ``None`` if it fails validation."""
    if not code:
        return None
    if "class UserStrategy" not in code or "def on_tick" not in code:
        return None
    if "runner.sdk" not in code:
        # Defensive: guarantee the SDK import so the sandbox can exec it.
        code = "from runner.sdk import Strategy\n\n" + code
    try:
        compile(code, "<copilot>", "exec")
    except SyntaxError as exc:
        log.warning("copilot: generated code failed to compile (%s)", exc)
        return None
    return code


# --------------------------------------------------------------------------- #
# Deterministic template fallback                                              #
# --------------------------------------------------------------------------- #
_BUY_WORDS = ("below", "under", "less", "<", "dips", "drops", "oversold")
_SELL_WORDS = ("above", "over", "greater", ">", "exceeds", "overbought")


def _numbers_near(desc: str, keyword: str, span: int = 24) -> List[Tuple[float, str]]:
    """Find numbers within ``span`` chars after ``keyword``.

    Returns ``(value, segment)`` pairs where ``segment`` runs from just before
    the keyword up to the number itself — so direction words BETWEEN keyword
    and number ("rsi above 70") classify that number, not a later clause.
    """
    out: List[Tuple[float, str]] = []
    for kw in re.finditer(keyword, desc, re.IGNORECASE):
        window = desc[kw.start():kw.end() + span]
        m = re.search(r"(\d+(?:\.\d+)?)", window)
        if m:
            segment = desc[max(0, kw.start() - 16):kw.start() + m.end()].lower()
            out.append((float(m.group(1)), segment))
    return out


def _classify_direction(segment: str) -> Optional[str]:
    """'buy' / 'sell' / None by the direction word nearest the number."""
    buy_pos = max((segment.rfind(w) for w in _BUY_WORDS), default=-1)
    sell_pos = max((segment.rfind(w) for w in _SELL_WORDS), default=-1)
    if buy_pos < 0 and sell_pos < 0:
        return None
    return "buy" if buy_pos > sell_pos else "sell"


def parse_hints(description: str) -> Dict[str, Any]:
    """Deterministically parse RSI/qty hints from a strategy description.

    Pure function: regex for numbers near ``rsi``/``qty``/``buy``/``sell``,
    direction words classify each RSI number as a buy or sell threshold; two
    unlabeled numbers split low→buy / high→sell. Defaults: VWAP mean-reversion,
    rsi<35 buy / rsi>65 sell, qty 50.
    """
    desc = description or ""
    rsi_buy: Optional[float] = None
    rsi_sell: Optional[float] = None
    unlabeled: List[float] = []

    for value, segment in _numbers_near(desc, r"rsi"):
        if not (1.0 <= value <= 99.0):
            continue
        direction = _classify_direction(segment)
        if direction == "buy":
            rsi_buy = value if rsi_buy is None else min(rsi_buy, value)
        elif direction == "sell":
            rsi_sell = value if rsi_sell is None else max(rsi_sell, value)
        else:
            unlabeled.append(value)

    if unlabeled:
        if rsi_buy is None and rsi_sell is None and len(unlabeled) >= 2:
            rsi_buy, rsi_sell = min(unlabeled), max(unlabeled)
        elif rsi_buy is None and rsi_sell is None:
            # A single bare RSI number: 'buy' in the text → buy gate, else sell.
            if "sell" in desc.lower() and "buy" not in desc.lower():
                rsi_sell = unlabeled[0]
            else:
                rsi_buy = unlabeled[0]
        elif rsi_buy is None:
            rsi_buy = min(unlabeled)
        elif rsi_sell is None:
            rsi_sell = max(unlabeled)

    if rsi_buy is None:
        rsi_buy = DEFAULT_RSI_BUY
    if rsi_sell is None:
        rsi_sell = DEFAULT_RSI_SELL
    if rsi_buy >= rsi_sell:  # nonsensical crossing gates → restore defaults
        rsi_buy, rsi_sell = DEFAULT_RSI_BUY, DEFAULT_RSI_SELL

    qty = DEFAULT_QTY
    qty_match = re.search(r"qty\W{0,6}(\d+)", desc, re.IGNORECASE) or \
        re.search(r"(\d+)\s*(?:shares?|lots?|units?)", desc, re.IGNORECASE)
    if qty_match:
        qty = max(1, min(MAX_QTY, int(qty_match.group(1))))

    return {
        "rsi_buy": float(rsi_buy),
        "rsi_sell": float(rsi_sell),
        "qty": int(qty),
        "use_vwap": True,  # template is a VWAP mean-reversion by design
    }


def build_template(description: str) -> Dict[str, str]:
    """Build the deterministic fallback strategy from parsed hints.

    :returns: ``{"code": ..., "explanation": ...}`` — always compiles.
    """
    hints = parse_hints(description)
    max_pos = hints["qty"] * POS_CAP_MULTIPLIER
    summary = (str(description or "").strip() or "VWAP mean-reversion")\
        .replace('"""', "'").replace("\n", " ")[:160]

    code = f'''from runner.sdk import Strategy


class UserStrategy(Strategy):
    """Deterministic Chronos copilot template: {summary}

    VWAP mean-reversion gated by RSI: BUY when RSI < {hints["rsi_buy"]:.0f} and price
    is at/below VWAP; SELL when RSI > {hints["rsi_sell"]:.0f} and price is at/above
    VWAP. Position is tracked exclusively from engine acks (ack['pos']).
    """

    RSI_BUY = {hints["rsi_buy"]}
    RSI_SELL = {hints["rsi_sell"]}
    QTY = {hints["qty"]}
    MAX_POS = {max_pos}

    def __init__(self) -> None:
        self.position = 0

    def on_tick(self, state: dict) -> list:
        rsi = state.get("rsi")
        vwap = float(state.get("vwap") or 0.0)
        last = float(state.get("last_price") or 0.0)
        if rsi is None or last <= 0.0:
            return []
        rsi = float(rsi)
        below_vwap = vwap <= 0.0 or last <= vwap
        above_vwap = vwap <= 0.0 or last >= vwap
        if rsi < self.RSI_BUY and below_vwap and self.position < self.MAX_POS:
            return [{{"action": "BUY", "type": "MARKET", "qty": self.QTY}}]
        if rsi > self.RSI_SELL and above_vwap and self.position > -self.MAX_POS:
            return [{{"action": "SELL", "type": "MARKET", "qty": self.QTY}}]
        return []

    def on_fill(self, ack: dict) -> None:
        pos = ack.get("pos")
        if pos is not None:
            self.position = int(pos)
'''
    compile(code, "<copilot-template>", "exec")  # invariant: always runnable
    explanation = (
        f"Deterministic template built from your description: buys {hints['qty']} "
        f"shares when RSI drops below {hints['rsi_buy']:.0f} with price at/below VWAP, "
        f"sells {hints['qty']} when RSI rises above {hints['rsi_sell']:.0f} with price "
        f"at/above VWAP, capped at ±{max_pos} shares. Position comes from engine acks."
    )
    return {"code": code, "explanation": explanation}


# --------------------------------------------------------------------------- #
# Gemini path                                                                  #
# --------------------------------------------------------------------------- #
async def _generate_gemini(description: str, api_key: str) -> Optional[str]:
    """Ask Gemini for the strategy code; return validated code or ``None``."""
    try:
        from google import genai  # local import: optional dependency
        client = genai.Client(api_key=api_key)
    except Exception as exc:
        log.warning("copilot: Gemini client unavailable (%s)", exc)
        return None

    prompt = _PROMPT_TEMPLATE.format(description=str(description).strip())
    try:
        response = await asyncio.wait_for(
            client.aio.models.generate_content(model=GEMINI_MODEL, contents=prompt),
            timeout=GEMINI_TIMEOUT,
        )
        return _validate_code(_strip_fences(response.text or ""))
    except asyncio.TimeoutError:
        log.warning("copilot: Gemini timed out after %.0fs", GEMINI_TIMEOUT)
    except Exception as exc:  # API/transport/parse errors
        log.warning("copilot: Gemini error (%s)", exc)
    return None


async def generate_strategy(description: str, api_key: Optional[str] = None) -> dict:
    """Turn a natural-language description into runnable UserStrategy code.

    :param description: what the strategy should do, in plain language.
    :param api_key: Gemini API key; ``None``/empty → template path directly.
    :returns: ``{"code": str, "explanation": str, "source": "gemini"|"template"}``
        — the code ALWAYS compiles and defines ``class UserStrategy`` with
        ``on_tick`` (validated for the Gemini path, guaranteed by construction
        for the template path).
    """
    description = str(description or "").strip()

    if api_key:
        code = await _generate_gemini(description, api_key)
        if code is not None:
            log.info("copilot: Gemini produced a valid strategy (%d chars)", len(code))
            return {
                "code": code,
                "explanation": (
                    "Gemini-generated strategy from your description. Review the "
                    "code before saving or running it — the copilot never "
                    "auto-runs code."
                ),
                "source": "gemini",
            }

    template = build_template(description)
    log.info("copilot: using deterministic template (Gemini %s)",
             "failed/invalid" if api_key else "key not configured")
    return {**template, "source": "template"}
