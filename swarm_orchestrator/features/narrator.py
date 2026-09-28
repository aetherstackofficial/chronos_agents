"""Explainable market narration (PROTOCOL_V2.md §6).

Answers "WHY is the market doing this?" from the latest §5.1 tick and recent
§5.2 trades: regime, order-flow imbalance, spread, price versus VWAP, the top
of the leaderboard, and the buy/sell aggressor balance of recent prints.

Primary path is Gemini (``gemini-2.5-flash``, async, 8 s budget — same client
pattern as ``oracle.py``). The fallback is a REAL deterministic heuristic that
composes one or two sentences directly from the numbers — never a stub.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("chronos.features")

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_TIMEOUT = 8.0

IDLE_NARRATION = "Market is idle — start a simulation."

# Heuristic thresholds (all deterministic policy constants).
OFI_HEAVY = 0.15          # |ofi| above this reads as one-sided flow
TIGHT_SPREAD = 0.15       # ₹; at/below this the book is "tight"
VWAP_STRETCH_PCT = 0.15   # % from VWAP that reads as "stretched"


def _fmt_inr(value: float) -> str:
    """Compact ₹ formatting for narration text."""
    return f"₹{value:,.2f}"


def _aggressor_balance(trades: List[dict]) -> Tuple[int, int]:
    """Return (buy_qty, sell_qty) aggressor volume from recent trade prints."""
    buy_qty = 0
    sell_qty = 0
    for trade in trades or []:
        try:
            qty = int(trade.get("qty", 0))
        except (TypeError, ValueError):
            continue
        if trade.get("aggressor") == "BUY":
            buy_qty += qty
        elif trade.get("aggressor") == "SELL":
            sell_qty += qty
    return buy_qty, sell_qty


def _tick_facts(tick: Dict[str, Any], trades: List[dict]) -> Dict[str, Any]:
    """Extract the narration-relevant numbers defensively from a tick."""
    leaderboard = tick.get("leaderboard") or []
    buy_qty, sell_qty = _aggressor_balance(trades)
    total_qty = buy_qty + sell_qty
    return {
        "symbol": str(tick.get("symbol") or "The market"),
        "regime": str(tick.get("regime") or "RANGING"),
        "ofi": float(tick.get("ofi") or 0.0),
        "spread": float(tick.get("spread") or 0.0),
        "last_price": float(tick.get("last_price") or 0.0),
        "vwap": float(tick.get("vwap") or 0.0),
        "top3": [
            {"id": str(row.get("id", "?")), "type": str(row.get("type", "?")),
             "pnl": float(row.get("pnl") or 0.0)}
            for row in leaderboard[:3]
        ],
        "buy_qty": buy_qty,
        "sell_qty": sell_qty,
        "buy_share": (buy_qty / total_qty) if total_qty > 0 else None,
    }


def heuristic_narration(facts: Dict[str, Any]) -> str:
    """Compose 1-2 deterministic sentences straight from the numbers."""
    parts: List[str] = []

    # Sentence 1: regime + flow + spread.
    ofi = facts["ofi"]
    if ofi > OFI_HEAVY:
        flow = f"order flow is buy-heavy (OFI {ofi:+.2f})"
    elif ofi < -OFI_HEAVY:
        flow = f"order flow is sell-heavy (OFI {ofi:+.2f})"
    else:
        flow = f"order flow is balanced (OFI {ofi:+.2f})"
    spread = facts["spread"]
    spread_desc = "tight" if spread <= TIGHT_SPREAD else "wide"
    parts.append(
        f"{facts['symbol']} is in a {facts['regime']} regime; {flow} and the "
        f"spread is {spread_desc} at {_fmt_inr(spread)}."
    )

    # Sentence 2: price vs VWAP, leaderboard, aggressor balance.
    clauses: List[str] = []
    last, vwap = facts["last_price"], facts["vwap"]
    if last > 0.0 and vwap > 0.0:
        stretch = (last - vwap) / vwap * 100.0
        if abs(stretch) >= VWAP_STRETCH_PCT:
            side = "above" if stretch > 0 else "below"
            clauses.append(
                f"price {_fmt_inr(last)} is {abs(stretch):.2f}% {side} "
                f"VWAP {_fmt_inr(vwap)}"
            )
        else:
            clauses.append(f"price is pinned to VWAP near {_fmt_inr(vwap)}")
    if facts["top3"]:
        leader = facts["top3"][0]
        clauses.append(
            f"{leader['id']} ({leader['type']}) leads the tape with "
            f"{_fmt_inr(leader['pnl'])} P&L"
        )
    if facts["buy_share"] is not None:
        pct = facts["buy_share"] * 100.0
        if pct >= 55.0:
            clauses.append(f"recent prints are {pct:.0f}% buyer-initiated")
        elif pct <= 45.0:
            clauses.append(f"recent prints are {100 - pct:.0f}% seller-initiated")
        else:
            clauses.append("recent prints are evenly matched between buyers and sellers")

    if clauses:
        sentence = " — ".join(clauses) + "."
        parts.append(sentence[0].upper() + sentence[1:])
    return " ".join(parts)


async def _narrate_gemini(facts: Dict[str, Any], api_key: str) -> Optional[str]:
    """LLM narration path; returns ``None`` on any failure (caller falls back)."""
    try:
        from google import genai  # local import: optional dependency
        client = genai.Client(api_key=api_key)
    except Exception as exc:
        log.warning("narrator: Gemini client unavailable (%s)", exc)
        return None

    top3 = "; ".join(
        f"{row['id']} ({row['type']}) pnl {row['pnl']:+.0f}" for row in facts["top3"]
    ) or "no ranked agents yet"
    if facts["buy_share"] is None:
        balance = "no recent prints"
    else:
        balance = (f"{facts['buy_qty']} shares bought vs {facts['sell_qty']} sold "
                   f"aggressively ({facts['buy_share'] * 100:.0f}% buyer-initiated)")
    prompt = (
        "You are the narrator of a live market simulation on an NSE-style venue "
        "(currency ₹). In 1-2 plain sentences explain WHY the market is doing "
        "what it is doing right now, citing the numbers. No preamble, no "
        "markdown, no advice.\n"
        f"Symbol: {facts['symbol']}\n"
        f"Regime: {facts['regime']}\n"
        f"Order-flow imbalance (OFI, -1..1): {facts['ofi']:+.2f}\n"
        f"Spread: ₹{facts['spread']:.2f}\n"
        f"Last price: ₹{facts['last_price']:.2f} vs VWAP ₹{facts['vwap']:.2f}\n"
        f"Leaderboard top 3: {top3}\n"
        f"Recent aggressor balance: {balance}\n"
    )
    try:
        response = await asyncio.wait_for(
            client.aio.models.generate_content(model=GEMINI_MODEL, contents=prompt),
            timeout=GEMINI_TIMEOUT,
        )
        text = (response.text or "").strip().replace("```", "").strip()
        return text or None
    except asyncio.TimeoutError:
        log.warning("narrator: Gemini timed out after %.0fs", GEMINI_TIMEOUT)
    except Exception as exc:  # API/transport errors
        log.warning("narrator: Gemini error (%s)", exc)
    return None


async def narrate(tick: Optional[dict], recent_trades: Optional[List[dict]],
                  api_key: Optional[str] = None) -> dict:
    """Explain the current market state.

    :param tick: latest §5.1 tick payload (``None``/empty → idle message).
    :param recent_trades: recent §5.2 trade payloads (may be empty).
    :param api_key: Gemini key; missing/failed → deterministic heuristic.
    :returns: ``{"narration": str, "source": "gemini"|"heuristic"}``.
    """
    if not tick:
        return {"narration": IDLE_NARRATION, "source": "heuristic"}

    facts = _tick_facts(tick, recent_trades or [])

    if api_key:
        text = await _narrate_gemini(facts, api_key)
        if text:
            return {"narration": text, "source": "gemini"}

    return {"narration": heuristic_narration(facts), "source": "heuristic"}
