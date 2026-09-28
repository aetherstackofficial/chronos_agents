"""Chronos news oracle (Phase 2).

Scores breaking-news headlines against the currently simulated instrument and,
when the impact is material, expresses that view as **real market orders**
through the engine's order-entry plane. The engine's emergent impact model does
the rest -- no reverse-engineered "gap math", no magic 50,001-share thresholds.

This replaces the Phase-1 oracle and fixes three flaws:

1. **Blocking Gemini call** -- the Phase-1 REP loop stalled on every LLM
   round-trip, so a second headline queued behind a slow one. Here the ROUTER
   acks ``{"status": "accepted"}`` immediately and processing happens in a
   spawned :func:`asyncio.Task`.
2. **Magic-number coupling** -- Phase-1 sized orders by inverting engine
   internals (``gap = qty/100000 * $2``). Here sizing is derived from the
   *observed* book: ``total_qty = max(500, int(|score| * 1.5 * visible_depth))``
   where ``visible_depth`` is real resting quantity on the top 10 levels of
   each side, split into 3 TWAP-style child MARKET orders ~1 s apart.
3. **Fake fallback** -- Phase-1 "fell back" to ``return 0.0``. Here
   :func:`fallback_score` is a genuine deterministic lexicon/sector matrix.

Transport map (PROTOCOL.md sections 1, 2, 4)::

    ROUTER bind :ORACLE_PORT      <- bridge DEALER "BRIDGE_ORACLE"  {"msg":"NEWS",...}
    DEALER "GEMINI_ORACLE"        -> engine ROUTER :ZMQ_ORDER_PORT  FETCH_STATE / ORDER

The FIRST child order carries ``meta.news`` (headline/score/sector/reasoning/
source); per PROTOCOL section 4.1 the engine publishes an ``event kind=news``
before executing it -- that is how the scored shock reaches the dashboard.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from collections import OrderedDict
from typing import Any, Optional

import zmq
import zmq.asyncio
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - [ORACLE] - %(message)s"
)
log = logging.getLogger("chronos.oracle")

# --------------------------------------------------------------------------- #
# Configuration (PROTOCOL section 2)                                          #
# --------------------------------------------------------------------------- #
ZMQ_HOST = os.getenv("ZMQ_HOST", "127.0.0.1")
ZMQ_ORDER_PORT = os.getenv("ZMQ_ORDER_PORT", "5555")
ORACLE_PORT = os.getenv("ORACLE_PORT", "5557")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_TIMEOUT = 8.0        # LLM scoring budget before falling back
ENGINE_TIMEOUT = 5.0        # engine FETCH_STATE / ORDER ack budget
MIN_IMPACT = 0.1            # |score| below this -> no intervention
MIN_TOTAL_QTY = 500         # floor so a thin book still prints a visible shock
DEPTH_FACTOR = 1.5          # aggression: fraction of visible depth consumed at |score|=1
CHILD_ORDERS = 3            # TWAP-style split
CHILD_SPACING = 1.0         # seconds between child orders
CACHE_SIZE = 256            # LRU entries for (headline, symbol) -> score result

# --------------------------------------------------------------------------- #
# Deterministic fallback scoring matrix                                        #
# --------------------------------------------------------------------------- #
# Keyword polarity lexicon. Multi-word phrases are matched as consecutive
# tokens and take priority over their constituent words (longest match wins;
# matched tokens are consumed so "record profit" does not also count "profit").
_STRONG_NEGATIVE: dict[str, float] = {
    "bankruptcy": -0.9,
    "fraud": -0.9,
    "default": -0.8,
    "crash": -0.8,
    "hack": -0.7,
    "breach": -0.7,
    "war": -0.7,
    "scandal": -0.7,
    "plunge": -0.7,
    "recall": -0.6,
    "lawsuit": -0.6,
    "ban": -0.6,
    "sanction": -0.6,
    "sanctions": -0.6,
    "downgrade": -0.6,
    "probe": -0.5,
    "resign": -0.5,
    "resigns": -0.5,
    "strike": -0.5,
    "layoff": -0.5,
    "layoffs": -0.5,
    "misses": -0.5,
    "loss": -0.5,
}
_STRONG_POSITIVE: dict[str, float] = {
    "record profit": 0.8,
    "record profits": 0.8,
    "breakthrough": 0.7,
    "beats": 0.6,
    "upgrade": 0.6,
    "approval": 0.6,
    "surge": 0.6,
    "acquisition": 0.5,
    "partnership": 0.5,
    "buyback": 0.5,
    "wins": 0.5,
    "expansion": 0.4,
    "contract": 0.4,
    "dividend": 0.4,
    "profit": 0.4,
}
_POLARITY: dict[str, float] = {**_STRONG_NEGATIVE, **_STRONG_POSITIVE}

# Negation within 3 tokens before a keyword flips its polarity
# ("denies fraud", "avoids bankruptcy", "not resigning").
_NEGATORS: frozenset[str] = frozenset(
    {"not", "no", "never", "denies", "denied", "deny", "avoids", "avoided",
     "avoid", "averts", "averted", "dismisses", "dismissed", "without"}
)

# Sector-relevance table. If the headline clearly talks about OTHER sectors
# only (and never names the target symbol or its sector), the impact on the
# target is dampened x0.4 -- an OPEC headline should barely move a TECH stock.
_SECTOR_KEYWORDS: dict[str, frozenset[str]] = {
    "TECH": frozenset({"tech", "software", "chip", "chips", "semiconductor",
                       "cloud", "ai", "cyber", "app", "data", "internet",
                       "startup", "hardware"}),
    "PHARMA": frozenset({"pharma", "drug", "drugs", "vaccine", "fda", "trial",
                         "clinical", "medicine", "hospital", "biotech",
                         "patent"}),
    "OIL": frozenset({"oil", "crude", "opec", "refinery", "gas", "barrel",
                      "petrol", "energy", "pipeline", "diesel"}),
    "DEFENSE": frozenset({"defense", "defence", "missile", "military", "army",
                          "weapons", "aerospace", "navy", "artillery"}),
    "BANK": frozenset({"bank", "banks", "banking", "loan", "loans", "rbi",
                       "npa", "credit", "deposit", "deposits", "lending",
                       "rates"}),
}
_SECTOR_DAMPEN = 0.4


def _find_phrase(tokens: list[str], words: list[str], consumed: set[int]) -> Optional[int]:
    """Return the start index of ``words`` as consecutive unconsumed tokens."""
    n = len(words)
    for i in range(len(tokens) - n + 1):
        if any(j in consumed for j in range(i, i + n)):
            continue
        if tokens[i:i + n] == words:
            return i
    return None


def fallback_score(headline: str, symbol: str, sector: str) -> float:
    """Deterministic local impact score in ``[-1, 1]`` -- the REAL fallback.

    Method (pure function of its arguments, fully unit-testable):

    1. Tokenize the headline (lowercase word tokens).
    2. Sum keyword weights from the polarity lexicon; multi-word phrases match
       first and consume their tokens so constituents are not double-counted.
    3. A negator ("not", "denies", "avoids", ...) within the 3 tokens before a
       keyword flips that keyword's polarity.
    4. If the headline mentions keywords of OTHER sectors only -- and neither
       the target ``symbol`` nor the target ``sector``'s own keywords -- the
       total is dampened by ``x0.4`` (irrelevant-sector shock).
    5. Clamp to ``[-1.0, 1.0]``.

    :param headline: raw news headline.
    :param symbol: target instrument symbol (e.g. ``"TCS"``).
    :param sector: target instrument sector (e.g. ``"TECH"``).
    :returns: impact score, negative = bearish, ``0.0`` = irrelevant.
    """
    tokens: list[str] = re.findall(r"[a-z0-9']+", headline.lower())
    if not tokens:
        return 0.0

    consumed: set[int] = set()
    score = 0.0
    # Longest phrases first so "record profit" wins over "profit".
    for phrase in sorted(_POLARITY, key=lambda p: -len(p.split())):
        words = phrase.split()
        idx = _find_phrase(tokens, words, consumed)
        if idx is None:
            continue
        consumed.update(range(idx, idx + len(words)))
        weight = _POLARITY[phrase]
        window = tokens[max(0, idx - 3):idx]
        if any(t in _NEGATORS for t in window):
            weight = -weight
        score += weight

    if score == 0.0:
        return 0.0

    token_set = set(tokens)
    mentioned = {s for s, kws in _SECTOR_KEYWORDS.items() if token_set & kws}
    target = sector.upper()
    if mentioned and target not in mentioned and symbol.lower() not in token_set:
        score *= _SECTOR_DAMPEN

    return max(-1.0, min(1.0, score))


# --------------------------------------------------------------------------- #
# Oracle service                                                               #
# --------------------------------------------------------------------------- #
class Oracle:
    """Async news oracle: ROUTER front-end + DEALER order-entry back-end."""

    def __init__(self) -> None:
        self._ctx: Optional[zmq.asyncio.Context] = None
        self._router: Optional[zmq.asyncio.Socket] = None
        self._engine: Optional[zmq.asyncio.Socket] = None
        self._engine_lock = asyncio.Lock()
        self._tasks: set[asyncio.Task] = set()
        self._cache: OrderedDict[tuple[str, str], dict] = OrderedDict()
        self._news_counter = 0
        self._gemini: Optional[Any] = None
        if GEMINI_API_KEY:
            try:
                from google import genai  # local import: optional dependency

                self._gemini = genai.Client()
                log.info("Gemini client ready (model %s)", GEMINI_MODEL)
            except Exception as exc:  # pragma: no cover - bad key / missing pkg
                log.warning("Gemini client unavailable (%s); fallback-only mode", exc)
        else:
            log.warning(
                "GEMINI_API_KEY not set -- running in fallback-only mode "
                "(deterministic lexicon scoring; fully functional, no LLM)."
            )

    # ---------------------------------------------------------------- engine #
    async def _engine_request(self, msg: dict, timeout: float = ENGINE_TIMEOUT) -> Optional[dict]:
        """Send one request on the engine DEALER and await one JSON reply.

        Serialized by a lock so concurrent scoring tasks never interleave
        frames; stale replies from previously timed-out requests are drained
        first. Returns ``None`` on timeout or an undecodable reply.
        """
        assert self._engine is not None
        async with self._engine_lock:
            while await self._engine.poll(0, zmq.POLLIN):  # drop stale replies
                await self._engine.recv_multipart()
            await self._engine.send_json(msg)
            if await self._engine.poll(int(timeout * 1000), zmq.POLLIN):
                frames = await self._engine.recv_multipart()
                try:
                    return json.loads(frames[-1])
                except (ValueError, IndexError):
                    return None
            return None

    # --------------------------------------------------------------- scoring #
    async def score_news(
        self, headline: str, symbol: str, price: float, sector: str
    ) -> dict:
        """Score ``headline`` against ``symbol``; never raises.

        Primary path: Gemini (async, 8 s budget). On timeout, API error or a
        malformed response the deterministic :func:`fallback_score` matrix is
        used instead. Results are LRU-cached by ``(headline, symbol)`` so a
        repeated injection does not re-bill the API.

        :returns: ``{"score", "sector", "reasoning", "source"}`` with
            ``score`` clamped to ``[-1, 1]`` and ``source`` one of
            ``"gemini" | "fallback"``.
        """
        key = (headline.lower().strip(), symbol)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            log.info("cache hit for %r -> %.2f (%s)", headline, cached["score"], cached["source"])
            return cached

        result: Optional[dict] = None
        if self._gemini is not None:
            result = await self._score_gemini(headline, symbol, price)
        if result is None:
            score = fallback_score(headline, symbol, sector)
            result = {
                "score": score,
                "sector": sector,
                "reasoning": (
                    "Deterministic lexicon/sector fallback matrix "
                    f"(score {score:+.2f}; Gemini unavailable or failed)."
                ),
                "source": "fallback",
            }

        self._cache[key] = result
        if len(self._cache) > CACHE_SIZE:
            self._cache.popitem(last=False)
        return result

    async def _score_gemini(self, headline: str, symbol: str, price: float) -> Optional[dict]:
        """LLM scoring path. Returns ``None`` on any failure (caller falls back)."""
        prompt = f"""
You are an expert quantitative analyst and risk manager.
Evaluate the impact of the following breaking news on a specific publicly traded company.

News Headline: "{headline}"
Target Stock: "{symbol}"
Current Stock Price: {price}

Instructions:
1. Identify the primary industry sector of "{symbol}".
2. Determine if the news affects this specific company (either directly or via a macro sector shock).
3. Assign an impact score between -1.0 and 1.0.
   * 1.0 represents an extreme euphoric event causing a maximum +20% surge in stock price.
   * -1.0 represents a catastrophic event causing a maximum -20% crash in stock price.
   * 0.0 means the news is completely irrelevant to the stock.

Respond ONLY with a valid JSON object. Do not include any markdown formatting.
Format exactly like this:
{{
    "inferred_sector": "<string>",
    "impact_score": <float between -1.0 and 1.0>,
    "relevance": <boolean>,
    "reasoning": "<1-2 sentences explaining why>"
}}
"""
        assert self._gemini is not None
        try:
            response = await asyncio.wait_for(
                self._gemini.aio.models.generate_content(
                    model=GEMINI_MODEL, contents=prompt
                ),
                timeout=GEMINI_TIMEOUT,
            )
            raw = (response.text or "").strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            data = json.loads(raw)
            score = float(data.get("impact_score", 0.0))
            if not data.get("relevance", False):
                score = 0.0
            return {
                "score": max(-1.0, min(1.0, score)),
                "sector": str(data.get("inferred_sector", "UNKNOWN")),
                "reasoning": str(data.get("reasoning", "No reasoning provided.")),
                "source": "gemini",
            }
        except asyncio.TimeoutError:
            log.warning("Gemini timed out after %.1fs; using fallback matrix", GEMINI_TIMEOUT)
        except (ValueError, TypeError, KeyError) as exc:
            log.warning("Gemini response unparsable (%s); using fallback matrix", exc)
        except Exception as exc:  # API/transport errors
            log.warning("Gemini API error (%s); using fallback matrix", exc)
        return None

    # ------------------------------------------------------------- execution #
    async def _process_news(self, headline: str) -> None:
        """Full pipeline for one accepted headline (runs as its own task)."""
        log.info("processing headline: %r", headline)
        state = await self._engine_request({"msg": "FETCH_STATE"})
        if state is None:
            log.warning("engine did not answer FETCH_STATE; dropping headline %r", headline)
            return

        symbol: str = state.get("symbol", "TCS")
        sector: str = state.get("sector", "TECH")
        price: float = float(state.get("last_price") or 0.0)
        lob_bids: list = state.get("lob_bids") or []
        lob_asks: list = state.get("lob_asks") or []

        result = await self.score_news(headline, symbol, price, sector)
        score: float = result["score"]
        log.info(
            "scored %r vs %s @ %.2f -> %.2f [%s] (%s)",
            headline, symbol, price, score, result["source"], result["reasoning"],
        )

        if abs(score) < MIN_IMPACT:
            log.info("impact %.2f below threshold %.2f; no intervention", score, MIN_IMPACT)
            return

        # Size from the OBSERVED book, not engine internals: consume a
        # score-scaled fraction of the visible top-10 depth on both sides.
        visible_depth = sum(
            int(level[1]) for level in (lob_bids[:10] + lob_asks[:10]) if len(level) >= 2
        )
        total_qty = max(MIN_TOTAL_QTY, int(abs(score) * DEPTH_FACTOR * visible_depth))
        side = "BUY" if score > 0 else "SELL"
        base, rem = divmod(total_qty, CHILD_ORDERS)
        children = [base + (1 if i < rem else 0) for i in range(CHILD_ORDERS)]
        children = [q for q in children if q > 0]

        self._news_counter += 1
        news_id = self._news_counter
        log.info(
            "ORACLE SHOCK: %s %d shares (visible depth %d) in %d children",
            side, total_qty, visible_depth, len(children),
        )

        for i, qty in enumerate(children):
            order: dict = {
                "msg": "ORDER",
                "agent_id": "GEMINI_ORACLE",
                "action": side,
                "type": "MARKET",
                "qty": qty,
                "client_order_id": f"news-{news_id}-{i + 1}",
            }
            if i == 0:
                # PROTOCOL section 4.1: the engine publishes `event kind=news`
                # before executing an order that carries meta.news.
                order["meta"] = {
                    "news": {
                        "headline": headline,
                        "score": score,
                        "sector": result["sector"],
                        "reasoning": result["reasoning"],
                        "source": result["source"],
                    }
                }
            ack = await self._engine_request(order)
            if ack is None:
                log.warning("child %d/%d (%s %d) got no ack; aborting slice",
                            i + 1, len(children), side, qty)
                return
            log.info(
                "child %d/%d ack: status=%s executed=%s avg_price=%s",
                i + 1, len(children), ack.get("status"),
                ack.get("executed_qty"), ack.get("average_price"),
            )
            if i < len(children) - 1:
                await asyncio.sleep(CHILD_SPACING)

    # ------------------------------------------------------------ main loop #
    async def run(self) -> None:
        """Bind sockets and serve news requests until cancelled."""
        self._ctx = zmq.asyncio.Context()

        self._router = self._ctx.socket(zmq.ROUTER)
        self._router.bind(f"tcp://0.0.0.0:{ORACLE_PORT}")

        self._engine = self._ctx.socket(zmq.DEALER)
        self._engine.setsockopt(zmq.IDENTITY, b"GEMINI_ORACLE")
        self._engine.setsockopt(zmq.LINGER, 0)
        self._engine.connect(f"tcp://{ZMQ_HOST}:{ZMQ_ORDER_PORT}")

        log.info(
            "oracle online: ROUTER :%s, engine tcp://%s:%s",
            ORACLE_PORT, ZMQ_HOST, ZMQ_ORDER_PORT,
        )
        try:
            while True:
                frames = await self._router.recv_multipart()
                if len(frames) < 2:
                    continue
                identity = frames[0]
                try:
                    request = json.loads(frames[-1])
                except ValueError:
                    await self._reply(identity, {"status": "rejected", "reason": "bad json"})
                    continue

                if request.get("msg") == "NEWS" and request.get("headline"):
                    # Ack IMMEDIATELY, then process concurrently -- the next
                    # headline is never blocked behind an LLM round-trip.
                    await self._reply(identity, {"status": "accepted"})
                    task = asyncio.create_task(self._process_news(str(request["headline"])))
                    self._tasks.add(task)
                    task.add_done_callback(self._tasks.discard)
                else:
                    await self._reply(
                        identity,
                        {"status": "rejected", "reason": "expected {'msg':'NEWS','headline':...}"},
                    )
        finally:
            await self._shutdown()

    async def _reply(self, identity: bytes, payload: dict) -> None:
        """Send one JSON frame back to the ROUTER peer ``identity``."""
        assert self._router is not None
        await self._router.send_multipart(
            [identity, json.dumps(payload).encode("utf-8")]
        )

    async def _shutdown(self) -> None:
        """Cancel in-flight scoring tasks and close all sockets."""
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        for sock in (self._router, self._engine):
            if sock is not None:
                sock.close(0)
        if self._ctx is not None:
            self._ctx.term()
        self._router = self._engine = self._ctx = None
        log.info("oracle shut down")


def main() -> None:
    """Entrypoint: ``python oracle.py``."""
    if sys.platform == "win32":
        # zmq.asyncio requires a selector loop on Windows.
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(Oracle().run())
    except KeyboardInterrupt:
        log.info("interrupted -- goodbye")


if __name__ == "__main__":
    main()
