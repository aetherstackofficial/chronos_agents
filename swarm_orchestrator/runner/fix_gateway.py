"""Chronos FIX 4.4 gateway — the institutions' front door (PROTOCOL section 12.2).

A minimal-but-real FIX acceptor: an asyncio TCP server (``FIX_PORT``, default
9878) that speaks enough of FIX 4.4 for an institutional client to log on and
trade against the Chronos engine.

Supported message types (tag 35):

* ``A``  Logon            — acknowledged with a Logon; opens the session.
* ``0``  Heartbeat        — accepted; the gateway also emits its own heartbeats.
* ``1``  TestRequest      — answered with a Heartbeat echoing TestReqID (112).
* ``D``  NewOrderSingle   — mapped to an engine ``ORDER`` (PROTOCOL section 4.1)
  over a ZeroMQ DEALER whose identity is ``FIX_<SenderCompID>``; the engine ack
  (section 4.2) is mapped back to an ExecutionReport (35=8):
  FILLED -> 39=2, PARTIAL -> 39=1, RESTING -> 39=0 (new), else 39=8 (rejected),
  with 31 = average price, 32 = cumulative executed qty, 11 echoed.
* ``F``  OrderCancelRequest — SIMPLIFICATION: mapped to engine ``CANCEL_ALL``
  for that CompID (the engine keys resting orders by agent, and per-order FIX
  OrigClOrdID bookkeeping is out of scope for this venue). Documented honestly;
  acknowledged with an ExecutionReport 39=4.
* ``5``  Logout           — acknowledged with a Logout; closes the session.

Sequence numbers: outgoing MsgSeqNum (34) is tracked per session; incoming
sequence gaps are logged (no resend/gap-fill machinery — this is an acceptor
for a simulation venue, not a certified FIX engine).

Run with: ``python -m runner.fix_gateway``  (Windows-compatible — installs
``WindowsSelectorEventLoopPolicy`` before any zmq.asyncio use).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import uuid
from typing import Optional

import simplefix
import zmq
import zmq.asyncio

try:  # optional convenience — .env support when python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

log = logging.getLogger("runner.fix_gateway")
if not log.handlers:  # dedicated handler so the [FIX] tag survives co-imports
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s - [FIX] - %(message)s"))
    log.addHandler(_handler)
    log.setLevel(logging.INFO)
    log.propagate = False

FIX_VERSION = "FIX.4.4"
GATEWAY_COMP_ID = "CHRONOS"
DEFAULT_HEARTBEAT_S = 30
ACK_TIMEOUT_S = 5.0

FIX_PORT = int(os.environ.get("FIX_PORT", "9878"))
ZMQ_HOST = os.environ.get("ZMQ_HOST", "127.0.0.1")
ZMQ_ORDER_PORT = int(os.environ.get("ZMQ_ORDER_PORT", "5555"))

# Engine ack status -> FIX OrdStatus (39) / ExecType (150)
_STATUS_TO_ORDSTATUS = {
    "FILLED": "2",   # Filled
    "PARTIAL": "1",  # Partially filled
    "RESTING": "0",  # New (resting on the book)
}


class FixSession:
    """One authenticated FIX session bound to one TCP connection."""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
                 zmq_ctx: zmq.asyncio.Context) -> None:
        self.reader = reader
        self.writer = writer
        self.zmq_ctx = zmq_ctx
        self.parser = simplefix.FixParser()
        self.peer = writer.get_extra_info("peername")
        self.comp_id: Optional[str] = None  # client SenderCompID (49)
        self.agent_id: Optional[str] = None  # "FIX_<comp_id>"
        self.dealer: Optional[zmq.asyncio.Socket] = None
        self.seq_out = 0
        self.seq_in_expected = 1
        self.heartbeat_s = DEFAULT_HEARTBEAT_S
        self.logged_on = False
        self._closing = False
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._send_lock = asyncio.Lock()

    # ------------------------------------------------------------------ #
    # FIX message construction / transmission                             #
    # ------------------------------------------------------------------ #
    def _base_message(self, msg_type: str) -> simplefix.FixMessage:
        """New outgoing message with the standard header populated."""
        self.seq_out += 1
        msg = simplefix.FixMessage()
        msg.append_pair(8, FIX_VERSION)
        msg.append_pair(35, msg_type)
        msg.append_pair(49, GATEWAY_COMP_ID)
        msg.append_pair(56, self.comp_id or "UNKNOWN")
        msg.append_pair(34, self.seq_out)
        msg.append_utc_timestamp(52)
        return msg

    async def _send(self, msg: simplefix.FixMessage) -> None:
        """Encode and write one FIX message to the peer."""
        async with self._send_lock:
            self.writer.write(msg.encode())
            await self.writer.drain()

    # ------------------------------------------------------------------ #
    # Engine bridge                                                       #
    # ------------------------------------------------------------------ #
    def _open_dealer(self) -> None:
        """Connect the per-session DEALER (identity ``FIX_<SenderCompID>``)."""
        self.dealer = self.zmq_ctx.socket(zmq.DEALER)
        self.dealer.setsockopt(zmq.IDENTITY, self.agent_id.encode())
        self.dealer.setsockopt(zmq.LINGER, 500)
        self.dealer.connect(f"tcp://{ZMQ_HOST}:{ZMQ_ORDER_PORT}")

    async def _engine_request(self, payload: dict) -> Optional[dict]:
        """Send one order-plane message and await its ack (<= 5s)."""
        if self.dealer is None:
            return None
        await self.dealer.send_json(payload)
        try:
            frames = await asyncio.wait_for(
                self.dealer.recv_multipart(), timeout=ACK_TIMEOUT_S)
            return json.loads(frames[-1])
        except asyncio.TimeoutError:
            log.warning("%s: engine ack timeout (%.0fs)", self.agent_id, ACK_TIMEOUT_S)
            return None
        except (ValueError, zmq.ZMQError) as exc:
            log.warning("%s: malformed engine reply: %s", self.agent_id, exc)
            return None

    # ------------------------------------------------------------------ #
    # Inbound dispatch                                                    #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _get(msg: simplefix.FixMessage, tag: int) -> Optional[str]:
        """Field value as str, or None."""
        raw = msg.get(tag)
        return raw.decode("ascii", "replace") if raw is not None else None

    def _check_seq(self, msg: simplefix.FixMessage) -> None:
        """Track inbound MsgSeqNum; log gaps (no resend machinery)."""
        seq = self._get(msg, 34)
        if seq is None:
            return
        try:
            seq_num = int(seq)
        except ValueError:
            return
        if seq_num != self.seq_in_expected:
            log.warning("%s: MsgSeqNum gap — expected %d, got %d (continuing)",
                        self.comp_id, self.seq_in_expected, seq_num)
        self.seq_in_expected = seq_num + 1

    async def dispatch(self, msg: simplefix.FixMessage) -> bool:
        """Handle one inbound message. Returns False when the session ends."""
        msg_type = self._get(msg, 35)
        self._check_seq(msg)

        if msg_type == "A":
            await self._on_logon(msg)
        elif not self.logged_on:
            log.warning("Message 35=%s from %s before Logon — dropped",
                        msg_type, self.peer)
        elif msg_type == "0":  # Heartbeat — nothing to do
            pass
        elif msg_type == "1":
            await self._on_test_request(msg)
        elif msg_type == "D":
            await self._on_new_order(msg)
        elif msg_type == "F":
            await self._on_cancel(msg)
        elif msg_type == "5":
            await self._on_logout()
            return False
        else:
            log.info("%s: unsupported 35=%s ignored", self.comp_id, msg_type)
        return True

    # ------------------------------------------------------------------ #
    # Handlers                                                            #
    # ------------------------------------------------------------------ #
    async def _on_logon(self, msg: simplefix.FixMessage) -> None:
        """Logon (35=A): register the CompID, open the engine DEALER, reply."""
        self.comp_id = self._get(msg, 49) or f"ANON{uuid.uuid4().hex[:4]}"
        self.agent_id = f"FIX_{self.comp_id}"
        hb = self._get(msg, 108)
        if hb and hb.isdigit() and int(hb) > 0:
            self.heartbeat_s = int(hb)
        self._open_dealer()
        self.logged_on = True

        reply = self._base_message("A")
        reply.append_pair(98, "0")  # EncryptMethod = none
        reply.append_pair(108, self.heartbeat_s)
        await self._send(reply)
        self._heartbeat_task = asyncio.ensure_future(self._heartbeat_loop())
        log.info("Logon: %s from %s (agent_id=%s, HeartBtInt=%ds)",
                 self.comp_id, self.peer, self.agent_id, self.heartbeat_s)

    async def _on_test_request(self, msg: simplefix.FixMessage) -> None:
        """TestRequest (35=1) -> Heartbeat (35=0) echoing TestReqID (112)."""
        reply = self._base_message("0")
        test_req_id = self._get(msg, 112)
        if test_req_id:
            reply.append_pair(112, test_req_id)
        await self._send(reply)

    async def _on_new_order(self, msg: simplefix.FixMessage) -> None:
        """NewOrderSingle (35=D) -> engine ORDER -> ExecutionReport (35=8)."""
        cl_ord_id = self._get(msg, 11) or uuid.uuid4().hex[:12]
        side = self._get(msg, 54)
        qty_raw = self._get(msg, 38)
        ord_type = self._get(msg, 40)
        price_raw = self._get(msg, 44)
        symbol = self._get(msg, 55) or ""

        reject_reason: Optional[str] = None
        action = {"1": "BUY", "2": "SELL"}.get(side or "")
        if action is None:
            reject_reason = f"unsupported Side (54={side})"
        order_type = {"1": "MARKET", "2": "LIMIT"}.get(ord_type or "")
        if reject_reason is None and order_type is None:
            reject_reason = f"unsupported OrdType (40={ord_type})"
        qty = 0
        if reject_reason is None:
            try:
                qty = int(float(qty_raw))  # institutions love "100.0"
                if qty <= 0:
                    raise ValueError
            except (TypeError, ValueError):
                reject_reason = f"invalid OrderQty (38={qty_raw})"
        price: Optional[float] = None
        if reject_reason is None and order_type == "LIMIT":
            try:
                price = float(price_raw)
                if price <= 0:
                    raise ValueError
            except (TypeError, ValueError):
                reject_reason = f"invalid Price (44={price_raw})"

        if reject_reason is not None:
            log.warning("%s: rejecting order %s — %s", self.agent_id, cl_ord_id,
                        reject_reason)
            await self._send_exec_report(
                cl_ord_id=cl_ord_id, symbol=symbol, side=side or "1",
                ord_status="8", avg_px=0.0, cum_qty=0, order_id="0",
                text=reject_reason)
            return

        order = {
            "msg": "ORDER",
            "agent_id": self.agent_id,
            "action": action,
            "type": order_type,
            "qty": qty,
            "client_order_id": cl_ord_id,
        }
        if price is not None:
            order["price"] = price

        ack = await self._engine_request(order)
        if ack is None:
            await self._send_exec_report(
                cl_ord_id=cl_ord_id, symbol=symbol, side=side, ord_status="8",
                avg_px=0.0, cum_qty=0, order_id="0",
                text="engine unavailable (no ack)")
            return

        status = ack.get("status", "REJECTED")
        ord_status = _STATUS_TO_ORDSTATUS.get(status, "8")
        await self._send_exec_report(
            cl_ord_id=cl_ord_id,
            symbol=symbol,
            side=side,
            ord_status=ord_status,
            avg_px=float(ack.get("average_price") or 0.0),
            cum_qty=int(ack.get("executed_qty") or 0),
            order_id=str(ack.get("order_id", 0)),
            text=ack.get("reason") if ord_status == "8" else None,
        )
        log.info("%s: order %s %s x%d -> %s", self.agent_id, cl_ord_id,
                 action, qty, status)

    async def _on_cancel(self, msg: simplefix.FixMessage) -> None:
        """OrderCancelRequest (35=F) -> engine CANCEL_ALL for this CompID.

        SIMPLIFICATION (documented): Chronos cancels ALL resting orders for the
        session's agent_id rather than one order by OrigClOrdID.
        """
        cl_ord_id = self._get(msg, 11) or self._get(msg, 41) or ""
        ack = await self._engine_request(
            {"msg": "CANCEL_ALL", "agent_id": self.agent_id})
        ok = ack is not None and ack.get("status") != "REJECTED"
        await self._send_exec_report(
            cl_ord_id=cl_ord_id, symbol=self._get(msg, 55) or "",
            side=self._get(msg, 54) or "1",
            ord_status="4" if ok else "8",  # 4 = Canceled
            avg_px=0.0, cum_qty=0,
            order_id=str(ack.get("order_id", 0)) if ack else "0",
            text="CANCEL_ALL executed (Chronos maps 35=F to cancel-all "
                 "for this CompID)" if ok else "cancel failed: engine unavailable")
        log.info("%s: OrderCancelRequest -> CANCEL_ALL (%s)",
                 self.agent_id, "ok" if ok else "failed")

    async def _on_logout(self) -> None:
        """Logout (35=5): acknowledge and let the session close."""
        await self._send(self._base_message("5"))
        log.info("Logout: %s", self.comp_id)

    async def _send_exec_report(self, cl_ord_id: str, symbol: str,
                                side: Optional[str], ord_status: str,
                                avg_px: float, cum_qty: int, order_id: str,
                                text: Optional[str] = None) -> None:
        """Emit an ExecutionReport (35=8) for one order outcome."""
        report = self._base_message("8")
        report.append_pair(37, order_id or "0")          # OrderID (engine)
        report.append_pair(11, cl_ord_id)                # ClOrdID echoed
        report.append_pair(17, uuid.uuid4().hex[:16])    # ExecID
        report.append_pair(150, ord_status)              # ExecType mirrors 39
        report.append_pair(39, ord_status)               # OrdStatus
        if symbol:
            report.append_pair(55, symbol)
        if side in ("1", "2"):
            report.append_pair(54, side)
        report.append_pair(31, f"{avg_px:.2f}")          # AvgPx / last px proxy
        report.append_pair(32, cum_qty)                  # executed (cum) qty
        report.append_pair(14, cum_qty)                  # CumQty (standard tag)
        if text:
            report.append_pair(58, text)
        await self._send(report)

    # ------------------------------------------------------------------ #
    # Session lifecycle                                                   #
    # ------------------------------------------------------------------ #
    async def _heartbeat_loop(self) -> None:
        """Emit a Heartbeat (35=0) every HeartBtInt seconds while connected."""
        try:
            while not self._closing:
                await asyncio.sleep(self.heartbeat_s)
                if self._closing:
                    break
                await self._send(self._base_message("0"))
        except (asyncio.CancelledError, ConnectionError, OSError):
            pass

    async def run(self) -> None:
        """Read/parse/dispatch loop for this connection."""
        log.info("Connection from %s", self.peer)
        try:
            while True:
                data = await self.reader.read(4096)
                if not data:
                    break
                self.parser.append_buffer(data)
                while True:
                    msg = self.parser.get_message()
                    if msg is None:
                        break
                    if not await self.dispatch(msg):
                        return
        except (ConnectionError, asyncio.IncompleteReadError, OSError) as exc:
            log.info("Connection %s dropped: %s", self.peer, exc)
        finally:
            await self.close()

    async def close(self) -> None:
        """Tear down the session: heartbeats, DEALER, TCP."""
        if self._closing:
            return
        self._closing = True
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
        if self.dealer is not None:
            # Session gone: pull this CompID's resting quotes off the book.
            try:
                await self.dealer.send_json(
                    {"msg": "CANCEL_ALL", "agent_id": self.agent_id})
            except zmq.ZMQError:
                pass
            self.dealer.close()
            self.dealer = None
        try:
            self.writer.close()
            await self.writer.wait_closed()
        except (ConnectionError, OSError):
            pass
        log.info("Session %s closed", self.comp_id or self.peer)


async def serve(host: str = "0.0.0.0", port: int = FIX_PORT) -> None:
    """Run the FIX acceptor until cancelled."""
    zmq_ctx = zmq.asyncio.Context.instance()

    async def _handle(reader: asyncio.StreamReader,
                      writer: asyncio.StreamWriter) -> None:
        await FixSession(reader, writer, zmq_ctx).run()

    server = await asyncio.start_server(_handle, host, port)
    log.info("FIX 4.4 gateway listening on %s:%d (engine tcp://%s:%d)",
             host, port, ZMQ_HOST, ZMQ_ORDER_PORT)
    async with server:
        await server.serve_forever()


def main() -> None:
    """Entry point: ``python -m runner.fix_gateway``."""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        log.info("FIX gateway stopped")


if __name__ == "__main__":
    main()
