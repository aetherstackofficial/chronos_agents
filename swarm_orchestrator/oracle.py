from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from collections import OrderedDict
from typing import Any, Optional

import httpx
import zmq
import zmq.asyncio
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s[ORACLE] - %(message)s")
log = logging.getLogger("chronos.oracle")

ZMQ_HOST = os.getenv("ZMQ_HOST", "127.0.0.1")
ZMQ_ORDER_PORT = os.getenv("ZMQ_ORDER_PORT", "5555")
ORACLE_PORT = os.getenv("ORACLE_PORT", "5557")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

GEMINI_MODEL = "gemini-3.5-flash"
GEMINI_TIMEOUT = 20.0

GROQ_MODEL = os.getenv("GROQ_MODEL") or "llama-3.3-70b-versatile"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_TIMEOUT = 20.0

ENGINE_TIMEOUT = 5.0
MIN_IMPACT = 0.1
MIN_TOTAL_QTY = 500
DEPTH_FACTOR = 1.5
CHILD_ORDERS = 3
CHILD_SPACING = 1.0
CACHE_SIZE = 256

_STRONG_NEGATIVE = {"bankruptcy": -0.9, "fraud": -0.9, "default": -0.8, "crash": -0.8, "hack": -0.7, "breach": -0.7, "war": -0.7, "scandal": -0.7, "plunge": -0.7, "recall": -0.6, "lawsuit": -0.6, "ban": -0.6, "sanction": -0.6, "sanctions": -0.6, "downgrade": -0.6, "probe": -0.5, "resign": -0.5, "resigns": -0.5, "strike": -0.5, "layoff": -0.5, "layoffs": -0.5, "misses": -0.5, "loss": -0.5}
_STRONG_POSITIVE = {"record profit": 0.8, "record profits": 0.8, "breakthrough": 0.7, "beats": 0.6, "upgrade": 0.6, "approval": 0.6, "surge": 0.6, "acquisition": 0.5, "partnership": 0.5, "buyback": 0.5, "wins": 0.5, "expansion": 0.4, "contract": 0.4, "dividend": 0.4, "profit": 0.4}
_POLARITY = {**_STRONG_NEGATIVE, **_STRONG_POSITIVE}
_POLARITY_COMPILED = [(phrase.split(), weight) for phrase, weight in sorted(_POLARITY.items(), key=lambda p: -len(p[0].split()))]

_NEGATORS = frozenset({"not", "no", "never", "denies", "denied", "deny", "avoids", "avoided", "avoid", "averts", "averted", "dismisses", "dismissed", "without"})

_SECTOR_KEYWORDS = {
    "TECH": frozenset({"tech", "software", "chip", "chips", "semiconductor", "cloud", "ai", "cyber", "app", "data", "internet", "startup", "hardware"}),
    "PHARMA": frozenset({"pharma", "drug", "drugs", "vaccine", "fda", "trial", "clinical", "medicine", "hospital", "biotech", "patent"}),
    "OIL": frozenset({"oil", "crude", "opec", "refinery", "gas", "barrel", "petrol", "energy", "pipeline", "diesel"}),
    "DEFENSE": frozenset({"defense", "defence", "missile", "military", "army", "weapons", "aerospace", "navy", "artillery"}),
    "BANK": frozenset({"bank", "banks", "banking", "loan", "loans", "rbi", "npa", "credit", "deposit", "deposits", "lending", "rates"}),
}
_SECTOR_DAMPEN = 0.4
_TOKENIZER = re.compile(r"[a-z0-9']+")

def _find_phrase(tokens: list[str], words: list[str], consumed: set[int]) -> Optional[int]:
    n = len(words)
    for i in range(len(tokens) - n + 1):
        if not consumed.intersection(range(i, i + n)) and tokens[i:i + n] == words:
            return i
    return None

def fallback_score(headline: str, symbol: str, sector: str) -> float:
    tokens = _TOKENIZER.findall(headline.lower())
    if not tokens:
        return 0.0

    consumed, score = set(), 0.0

    for words, weight in _POLARITY_COMPILED:
        if (idx := _find_phrase(tokens, words, consumed)) is not None:
            consumed.update(range(idx, idx + len(words)))
            if any(t in _NEGATORS for t in tokens[max(0, idx - 3):idx]):
                weight = -weight
            score += weight

    if score == 0.0:
        return 0.0

    token_set = set(tokens)
    mentioned = {s for s, kws in _SECTOR_KEYWORDS.items() if token_set & kws}
    if mentioned and sector.upper() not in mentioned and symbol.lower() not in token_set:
        score *= _SECTOR_DAMPEN

    return max(-1.0, min(1.0, score))

def _build_prompt(headline: str, symbol: str, price: float, sector: str) -> str:
    return (
        f'You are a senior equity analyst on a trading desk. Estimate the REALISTIC\n'
        f'price reaction of ONE stock to a news headline within a single session.\n\n'
        f'News Headline: "{headline}"\nTarget Stock: "{symbol}"\n'
        f'Target Sector: "{sector}"\nCurrent Stock Price: {price}\n\n'
        f'Step 1 - Classify the scope of the news:\n'
        f'  A. COMPANY-SPECIFIC: names "{symbol}" or is directly about its business.\n'
        f'  B. SECTOR-WIDE: affects the whole "{sector}" sector, company not named.\n'
        f'  C. MACRO / OTHER SECTOR: general economy, or a different sector only.\n'
        f'  D. IRRELEVANT: no plausible link to this stock.\n\n'
        f'Step 2 - Estimate the expected move in percent. Typical ranges:\n'
        f'  D: about 0%\n  C: 0% to 1.5%\n  B: 1% to 4%\n'
        f'  A, routine (earnings beat/miss, order win, upgrade/downgrade, guidance): 3% to 7%\n'
        f'  A, extreme and rare (fraud, bankruptcy, regulator ban, takeover bid, major disaster): 8% to 20%\n'
        f'Most headlines are ordinary. Do NOT choose an extreme unless the headline\n'
        f'clearly describes a rare, company-defining event. Reduce the magnitude by\n'
        f'about half for hedged or uncertain wording ("reportedly", "may", "plans to",\n'
        f'"rumour") or for news that is old or already widely expected.\n\n'
        f'Step 3 - Convert to a score:\n'
        f'  impact_score = expected_move_percent / 20, negative if bearish.\n'
        f'  Examples: +4% -> 0.20, -7% -> -0.35, +1.5% -> 0.075, +/-20% -> +/-1.0.\n'
        f'Use the full range of values; do not default to round numbers like 0.5 or 0.9.\n\n'
        f'Respond ONLY with a valid JSON object. Do not include any markdown formatting.\n'
        f'Format exactly like this:\n'
        f'{{\n  "inferred_sector": "<string>",\n  "impact_score": <float between -1.0 and 1.0>,\n'
        f'  "relevance": <boolean>,\n  "reasoning": "<1-2 sentences: scope (A/B/C/D), expected move percent, why>"\n}}'
    )

def _parse_llm_json(raw: str, source: str) -> dict:
    data = json.loads(raw.strip().replace("```json", "").replace("```", "").strip())
    return {
        "score": max(-1.0, min(1.0, float(data.get("impact_score", 0.0)) if data.get("relevance", False) else 0.0)),
        "sector": str(data.get("inferred_sector", "UNKNOWN")),
        "reasoning": str(data.get("reasoning", "No reasoning provided.")),
        "source": source,
    }

class Oracle:
    def __init__(self) -> None:
        self._ctx: Optional[zmq.asyncio.Context] = None
        self._router: Optional[zmq.asyncio.Socket] = None
        self._engine: Optional[zmq.asyncio.Socket] = None
        self._engine_lock = asyncio.Lock()
        self._tasks: set[asyncio.Task] = set()
        self._cache: OrderedDict[tuple[str, str], dict] = OrderedDict()
        self._news_counter = 0
        self._gemini = None
        
        if GEMINI_API_KEY:
            try:
                from google import genai
                self._gemini = genai.Client()
                log.info("Tier 1 Gemini ready (model %s)", GEMINI_MODEL)
            except Exception as exc:
                log.warning("Tier 1 Gemini unavailable (%s)", exc)
        else:
            log.warning("Tier 1 Gemini disabled (GEMINI_API_KEY not set)")

        if GROQ_API_KEY:
            log.info("Tier 2 Groq ready (model %s)", GROQ_MODEL)
        else:
            log.warning("Tier 2 Groq disabled (GROQ_API_KEY not set)")

        if not self._gemini and not GROQ_API_KEY:
            log.warning("No LLM tier available -- lexicon-only mode. Scoring still works, but any wording outside the 41-keyword lexicon scores 0.00 and produces no market intervention.")

    async def _engine_request(self, msg: dict, timeout: float = ENGINE_TIMEOUT) -> Optional[dict]:
        async with self._engine_lock:
            while await self._engine.poll(0, zmq.POLLIN):
                await self._engine.recv_multipart()
            await self._engine.send_json(msg)
            if await self._engine.poll(int(timeout * 1000), zmq.POLLIN):
                frames = await self._engine.recv_multipart()
                try:
                    return json.loads(frames[-1])
                except (ValueError, IndexError):
                    return None
            return None

    async def score_news(self, headline: str, symbol: str, price: float, sector: str) -> dict:
        key = (headline.lower().strip(), symbol)
        if (cached := self._cache.get(key)) is not None:
            self._cache.move_to_end(key)
            log.info("cache hit for %r -> %.2f (%s)", headline, cached["score"], cached["source"])
            return cached

        result = None
        if self._gemini:
            result = await self._score_gemini(headline, symbol, price, sector)
        if not result and GROQ_API_KEY:
            result = await self._score_groq(headline, symbol, price, sector)
            
        if not result:
            score = fallback_score(headline, symbol, sector)
            result = {
                "score": score,
                "sector": sector,
                "reasoning": f"Deterministic lexicon/sector fallback matrix (score {score:+.2f}; every LLM tier unavailable or failed).",
                "source": "fallback",
            }

        if result["source"] != "fallback":
            self._cache[key] = result
            if len(self._cache) > CACHE_SIZE:
                self._cache.popitem(last=False)
                
        return result

    async def _score_gemini(self, headline: str, symbol: str, price: float, sector: str) -> Optional[dict]:
        prompt = _build_prompt(headline, symbol, price, sector)
        try:
            response = await asyncio.wait_for(
                self._gemini.aio.models.generate_content(model=GEMINI_MODEL, contents=prompt),
                timeout=GEMINI_TIMEOUT
            )
            return _parse_llm_json(response.text or "", "gemini")
        except asyncio.TimeoutError:
            log.warning("Gemini timed out after %.1fs; trying next tier", GEMINI_TIMEOUT)
        except (ValueError, TypeError, KeyError) as exc:
            log.warning("Gemini response unparsable (%s); trying next tier", exc)
        except Exception as exc:
            log.warning("Gemini API error (%s); trying next tier", exc)
        return None

    async def _score_groq(self, headline: str, symbol: str, price: float, sector: str) -> Optional[dict]:
        prompt = _build_prompt(headline, symbol, price, sector)
        payload = {
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }
        try:
            async with httpx.AsyncClient(timeout=GROQ_TIMEOUT) as client:
                r = await client.post(
                    GROQ_URL,
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                    json=payload,
                )
            if r.status_code != 200:
                log.warning("Groq HTTP %d (%s); using fallback matrix", r.status_code, r.text[:200])
                return None
            return _parse_llm_json(r.json()["choices"][0]["message"]["content"], "groq")
        except httpx.TimeoutException:
            log.warning("Groq timed out after %.1fs; using fallback matrix", GROQ_TIMEOUT)
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            log.warning("Groq response unparsable (%s); using fallback matrix", exc)
        except Exception as exc:
            log.warning("Groq API error (%s); using fallback matrix", exc)
        return None

    async def _process_news(self, headline: str) -> None:
        log.info("processing headline: %r", headline)
        if not (state := await self._engine_request({"msg": "FETCH_STATE"})):
            log.warning("engine did not answer FETCH_STATE; dropping headline %r", headline)
            return

        symbol = state.get("symbol", "TCS")
        sector = state.get("sector", "TECH")
        price = float(state.get("last_price") or 0.0)
        lob_bids = state.get("lob_bids") or []
        lob_asks = state.get("lob_asks") or []

        result = await self.score_news(headline, symbol, price, sector)
        score = result["score"]
        log.info("scored %r vs %s @ %.2f -> %.2f [%s] (%s)", headline, symbol, price, score, result["source"], result["reasoning"])

        if abs(score) < MIN_IMPACT:
            log.info("impact %.2f below threshold %.2f; no intervention", score, MIN_IMPACT)
            return

        visible_depth = sum(int(level[1]) for level in (lob_bids[:10] + lob_asks[:10]) if len(level) >= 2)
        total_qty = max(MIN_TOTAL_QTY, int(abs(score) * DEPTH_FACTOR * visible_depth))
        side = "BUY" if score > 0 else "SELL"
        
        base, rem = divmod(total_qty, CHILD_ORDERS)
        children = [q for q in (base + (1 if i < rem else 0) for i in range(CHILD_ORDERS)) if q > 0]

        self._news_counter += 1
        news_id = self._news_counter
        log.info("ORACLE SHOCK: %s %d shares (visible depth %d) in %d children", side, total_qty, visible_depth, len(children))

        for i, qty in enumerate(children):
            order = {
                "msg": "ORDER",
                "agent_id": "GEMINI_ORACLE",
                "action": side,
                "type": "MARKET",
                "qty": qty,
                "client_order_id": f"news-{news_id}-{i + 1}",
            }
            if i == 0:
                order["meta"] = {
                    "news": {
                        "headline": headline,
                        "score": score,
                        "sector": result["sector"],
                        "reasoning": result["reasoning"],
                        "source": result["source"],
                    }
                }
            
            if (ack := await self._engine_request(order)) is None:
                log.warning("child %d/%d (%s %d) got no ack; aborting slice", i + 1, len(children), side, qty)
                return
                
            log.info("child %d/%d ack: status=%s executed=%s avg_price=%s", i + 1, len(children), ack.get("status"), ack.get("executed_qty"), ack.get("average_price"))
            if i < len(children) - 1:
                await asyncio.sleep(CHILD_SPACING)

    async def run(self) -> None:
        self._ctx = zmq.asyncio.Context()
        self._router = self._ctx.socket(zmq.ROUTER)
        self._router.bind(f"tcp://0.0.0.0:{ORACLE_PORT}")

        self._engine = self._ctx.socket(zmq.DEALER)
        self._engine.setsockopt(zmq.IDENTITY, b"GEMINI_ORACLE")
        self._engine.setsockopt(zmq.LINGER, 0)
        self._engine.connect(f"tcp://{ZMQ_HOST}:{ZMQ_ORDER_PORT}")

        log.info("oracle online: ROUTER :%s, engine tcp://%s:%s", ORACLE_PORT, ZMQ_HOST, ZMQ_ORDER_PORT)
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
                    await self._reply(identity, {"status": "accepted"})
                    task = asyncio.create_task(self._process_news(str(request["headline"])))
                    self._tasks.add(task)
                    task.add_done_callback(self._tasks.discard)
                else:
                    await self._reply(identity, {"status": "rejected", "reason": "expected {'msg':'NEWS','headline':...}"})
        finally:
            await self._shutdown()

    async def _reply(self, identity: bytes, payload: dict) -> None:
        await self._router.send_multipart([identity, json.dumps(payload).encode("utf-8")])

    async def _shutdown(self) -> None:
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
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(Oracle().run())
    except KeyboardInterrupt:
        log.info("interrupted -- goodbye")

if __name__ == "__main__":
    main()
    