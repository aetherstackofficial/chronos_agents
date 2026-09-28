"""Report rendering for strategy results: RFC-4180 CSVs and a reportlab PDF.

Pure functions over the dict shapes produced by ``analytics.results``:

* :func:`trades_csv`    — full trade log, one row per fill.
* :func:`outcomes_csv`  — per-day outcome summary.
* :func:`strategy_pdf`  — printable report (title block, summary table,
  per-day outcomes table, trade log capped at 500 rows).

All functions return raw ``bytes`` ready to stream as a download. Nothing here
reads the clock: the PDF's "generated" stamp comes from ``detail['generated_at']``
(injected by the caller) and defaults to ``'n/a'``.
"""

from __future__ import annotations

import csv
import io
import logging
from typing import Iterable
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

logger = logging.getLogger("chronos.analytics")

TRADE_COLUMNS: list[str] = [
    "ts",
    "unix_time",
    "day_count",
    "market_minute",
    "run_id",
    "side",
    "price",
    "qty",
    "cash_delta",
    "counterparty",
]

OUTCOME_COLUMNS: list[str] = [
    "run_id",
    "day_count",
    "symbol",
    "n_trades",
    "start_equity",
    "end_equity",
    "gross_pnl",
    "max_drawdown",
    "win_rate",
    "realized_pnl",
]

PDF_TRADE_ROW_CAP = 500

# Dark-on-light palette.
_INK = colors.HexColor("#14181d")
_HEADER_BG = colors.HexColor("#16324c")
_ALT_ROW = colors.HexColor("#f2f5f8")
_GRID = colors.HexColor("#c7d0d9")
_MUTED = colors.HexColor("#5a6672")


# --------------------------------------------------------------------------- #
# CSV                                                                          #
# --------------------------------------------------------------------------- #
def _rows_to_csv(rows: Iterable[dict], columns: list[str]) -> bytes:
    """Render dict rows to RFC-4180 CSV bytes (CRLF line endings, header row)."""
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=columns,
        extrasaction="ignore",
        lineterminator="\r\n",
        quoting=csv.QUOTE_MINIMAL,
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({col: row.get(col, "") for col in columns})
    return buffer.getvalue().encode("utf-8")


def trades_csv(rows: list[dict], meta: dict) -> bytes:
    """RFC-4180 CSV of the full trade log (all fill columns, header row).

    ``meta`` carries labeling context (e.g. ``name``, ``strategy_id``) used
    for logging; it is intentionally not embedded in the CSV body so the file
    stays strictly machine-readable.
    """
    payload = _rows_to_csv(rows, TRADE_COLUMNS)
    logger.info(
        "trades CSV rendered: %d row(s), %d bytes (%s)",
        len(rows), len(payload), meta.get("name", "?"),
    )
    return payload


def outcomes_csv(days: list[dict], meta: dict) -> bytes:
    """RFC-4180 CSV of the per-day outcome summary (one row per trading day)."""
    payload = _rows_to_csv(days, OUTCOME_COLUMNS)
    logger.info(
        "outcomes CSV rendered: %d day(s), %d bytes (%s)",
        len(days), len(payload), meta.get("name", "?"),
    )
    return payload


# --------------------------------------------------------------------------- #
# PDF                                                                          #
# --------------------------------------------------------------------------- #
def _inr(value: object) -> str:
    """Format a currency amount (INR) with thousands separators."""
    try:
        return f"INR {float(value):,.2f}"  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "n/a"


def _num(value: object, digits: int = 2) -> str:
    """Format a plain number, or ``'n/a'`` when it is not numeric."""
    try:
        return f"{float(value):,.{digits}f}"  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "n/a"


def _pct(value: object) -> str:
    """Format a 0..1 ratio as a percentage."""
    try:
        return f"{float(value) * 100:.0f}%"  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "n/a"


def _clock(ts: object) -> str:
    """Extract HH:MM:SS from an ISO timestamp string (best effort)."""
    text = str(ts or "")
    return text[11:19] if len(text) >= 19 else text


def _table(data: list[list[str]], col_widths: list[float], numeric_from: int) -> Table:
    """Build a styled table: dark header, zebra body, right-aligned numerics."""
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), _HEADER_BG),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("TEXTCOLOR", (0, 1), (-1, -1), _INK),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, _ALT_ROW]),
                ("GRID", (0, 0), (-1, -1), 0.4, _GRID),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (numeric_from, 1), (-1, -1), "RIGHT"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def strategy_pdf(detail: dict) -> bytes:
    """Render one strategy's full report as PDF bytes (reportlab platypus).

    Expects the ``strategy_detail`` dict shape from ``analytics.results``,
    optionally extended with ``generated_at`` (a preformatted string; this
    function never reads the clock itself). Sections: title block, summary
    stat table, per-day outcomes table, and a trade log capped at
    :data:`PDF_TRADE_ROW_CAP` rows with a truncation note.
    """
    name = str(detail.get("name") or "strategy")
    agent_id = str(detail.get("agent_id") or "?")
    generated_at = str(detail.get("generated_at") or "n/a")
    days = detail.get("days") or []

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ChronosTitle", parent=styles["Title"], fontSize=18,
        textColor=_INK, spaceAfter=2, alignment=0,
    )
    sub_style = ParagraphStyle(
        "ChronosSub", parent=styles["Normal"], fontSize=9,
        textColor=_MUTED, spaceAfter=10,
    )
    heading_style = ParagraphStyle(
        "ChronosHeading", parent=styles["Heading2"], fontSize=12,
        textColor=_HEADER_BG, spaceBefore=12, spaceAfter=4,
    )
    note_style = ParagraphStyle(
        "ChronosNote", parent=styles["Normal"], fontSize=8, textColor=_MUTED,
    )

    story: list = [
        Paragraph(f"Strategy Report — {escape(name)}", title_style),
        Paragraph(
            f"Agent {escape(agent_id)} &nbsp;•&nbsp; "
            f"Script {escape(str(detail.get('script_id') or 'n/a'))} &nbsp;•&nbsp; "
            f"Generated {escape(generated_at)}",
            sub_style,
        ),
    ]

    # -- Summary ----------------------------------------------------------- #
    story.append(Paragraph("Summary", heading_style))
    summary_rows = [
        ["Metric", "Value"],
        ["Status", str(detail.get("status") or "n/a")],
        ["Started at", str(detail.get("started_at") or "n/a")],
        ["Total trades", _num(detail.get("total_trades"), 0)],
        ["Mark-to-market P&L", _inr(detail.get("total_pnl"))],
        ["Realized P&L (avg-cost)", _inr(detail.get("realized_pnl"))],
        ["Trading days", _num(detail.get("n_days"), 0)],
        ["Runs", _num(detail.get("n_runs"), 0)],
    ]
    story.append(_table(summary_rows, [55 * mm, 100 * mm], numeric_from=1))

    # -- Per-day outcomes -------------------------------------------------- #
    story.append(Paragraph("Per-day outcomes", heading_style))
    if days:
        day_rows: list[list[str]] = [
            ["Run", "Day", "Symbol", "Trades", "Start eq", "End eq",
             "Gross P&L", "Max DD", "Win rate"],
        ]
        for day in days:
            run_id = str(day.get("run_id") or "")
            day_rows.append(
                [
                    run_id[:14] + ("…" if len(run_id) > 14 else ""),
                    _num(day.get("day_count"), 0),
                    str(day.get("symbol") or ""),
                    _num(day.get("n_trades"), 0),
                    _num(day.get("start_equity")),
                    _num(day.get("end_equity")),
                    _num(day.get("gross_pnl")),
                    _num(day.get("max_drawdown")),
                    _pct(day.get("win_rate")),
                ]
            )
        story.append(
            _table(
                day_rows,
                [26 * mm, 10 * mm, 16 * mm, 14 * mm, 24 * mm, 24 * mm,
                 22 * mm, 20 * mm, 15 * mm],
                numeric_from=3,
            )
        )
    else:
        story.append(Paragraph("No completed trading days recorded yet.", note_style))

    # -- Trade log --------------------------------------------------------- #
    story.append(Paragraph("Trade log", heading_style))
    all_trades: list[dict] = []
    for day in days:
        all_trades.extend(day.get("trades") or [])
    if all_trades:
        shown = all_trades[:PDF_TRADE_ROW_CAP]
        trade_rows: list[list[str]] = [
            ["Time", "Day", "Min", "Side", "Price", "Qty", "Cash delta",
             "Counterparty"],
        ]
        for trade in shown:
            trade_rows.append(
                [
                    _clock(trade.get("ts")),
                    _num(trade.get("day_count"), 0),
                    _num(trade.get("market_minute"), 0),
                    str(trade.get("side") or ""),
                    _num(trade.get("price")),
                    _num(trade.get("qty"), 0),
                    _num(trade.get("cash_delta")),
                    str(trade.get("counterparty") or ""),
                ]
            )
        story.append(
            _table(
                trade_rows,
                [22 * mm, 10 * mm, 12 * mm, 13 * mm, 22 * mm, 15 * mm,
                 28 * mm, 30 * mm],
                numeric_from=4,
            )
        )
        if len(all_trades) > PDF_TRADE_ROW_CAP:
            story.append(Spacer(1, 2 * mm))
            story.append(
                Paragraph(
                    f"Showing the first {PDF_TRADE_ROW_CAP} of "
                    f"{len(all_trades)} trades — download the CSV for the "
                    f"complete log.",
                    note_style,
                )
            )
    else:
        story.append(Paragraph("No trades recorded yet.", note_style))

    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=f"Strategy Report — {name}",
        author="Chronos",
    )
    document.build(story)
    payload = buffer.getvalue()
    logger.info(
        "strategy PDF rendered: %d day(s), %d trade(s), %d bytes (%s)",
        len(days), len(all_trades), len(payload), name,
    )
    return payload
