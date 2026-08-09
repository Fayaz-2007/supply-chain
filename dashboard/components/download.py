"""
Reusable export/download button components (CSV, Excel, PDF).

The PDF builder writes with fpdf2's built-in Helvetica core font, which can
only encode Latin-1. AI-generated narrative text routinely contains characters
outside that range (em dashes, curly quotes, bullets, emoji) and Markdown
markup, so everything is passed through ``sanitize_for_pdf()`` before it
reaches fpdf2. This keeps the PDF dependency-free (no bundled TTF) while
guaranteeing no text can ever raise an encoding error.

The PDF itself is a structured "executive report" (cover page, bordered
tables, a highlighted best-model callout, optional charts, page numbers) built
from a small ``BusinessReportData`` bundle -- see ``build_business_report_pdf``.
Callers assemble that bundle from data they already have; nothing in this
module computes or reinterprets ML results.
"""

from __future__ import annotations

import io
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import streamlit as st
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from fpdf.fonts import FontFace

from utils.helpers import dataframe_to_csv_bytes, dataframe_to_excel_bytes


def render_csv_download(df: pd.DataFrame, file_name: str, label: str = "Download CSV") -> None:
    """Render a Streamlit download button that exports a DataFrame as CSV."""
    st.download_button(label, data=dataframe_to_csv_bytes(df), file_name=file_name, mime="text/csv")


def render_excel_download(df: pd.DataFrame, file_name: str, label: str = "Download Excel") -> None:
    """Render a Streamlit download button that exports a DataFrame as XLSX."""
    st.download_button(
        label,
        data=dataframe_to_excel_bytes(df),
        file_name=file_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# --------------------------------------------------------------------------- #
# Text sanitization for the Latin-1 core font
# --------------------------------------------------------------------------- #

# Punctuation and symbols LLMs emit constantly that Latin-1 cannot encode.
# Mapped to visually equivalent ASCII rather than dropped, so meaning survives.
_UNICODE_REPLACEMENTS = {
    "–": "-", "—": " - ", "‒": "-", "―": "-",     # dashes
    "‘": "'", "’": "'", "‚": "'", "‛": "'",       # single quotes
    "“": '"', "”": '"', "„": '"', "‟": '"',       # double quotes
    "•": "-", "‣": "-", "●": "-", "▪": "-",       # bullets
    "■": "-", "◦": "-", "⁃": "-", "·": "-",
    "…": "...",                                                   # ellipsis
    "→": "->", "←": "<-", "↔": "<->",                   # arrows
    "↑": "^", "↓": "v", "⇒": "=>",
    "≤": "<=", "≥": ">=", "≠": "!=", "≈": "~",    # math
    "×": "x", "⁄": "/", "−": "-",
    " ": " ", " ": " ", " ": " ", "​": "",        # spaces
    " ": " ", " ": " ",                                      # line/para separators
    "✓": "[x]", "✔": "[x]", "✗": "[ ]", "✘": "[ ]",
    "™": "(TM)", "®": "(R)", "℗": "(P)",
    "﻿": "",                                                      # BOM
}

# Emoji -> short plain-text equivalents. Anything not listed here is removed by
# the catch-all pass below, so this only needs to cover glyphs whose meaning is
# worth preserving in a business report.
_EMOJI_REPLACEMENTS = {
    "\U0001F680": "[growth]", "\U0001F4C8": "[up]", "\U0001F4C9": "[down]",
    "\U0001F4CA": "[chart]", "\U0001F4C9️": "[down]",
    "⚠": "[warning]", "⚠️": "[warning]",
    "✅": "[OK]", "❌": "[X]", "❗": "[!]", "❓": "[?]",
    "\U0001F534": "[high risk]", "\U0001F7E1": "[medium risk]", "\U0001F7E2": "[low risk]",
    "\U0001F535": "[info]", "\U0001F4A1": "[insight]", "\U0001F3AF": "[target]",
    "\U0001F4B0": "[cost]", "\U0001F4B5": "[cost]", "\U0001F50D": "[review]",
    "\U0001F4CC": "[note]", "\U0001F4DD": "[note]", "\U0001F4C4": "[report]",
    "⭐": "[star]", "\U0001F525": "[critical]", "\U0001F44D": "[good]",
    "\U0001F44E": "[poor]", "\U0001F6A8": "[alert]", "⏱": "[time]",
}

# Markdown markup stripped before writing -- fpdf2 would otherwise render the
# literal asterisks/backticks, which looks unprofessional in a report.
_MD_PATTERNS = [
    (re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE), ""),          # ## Heading
    (re.compile(r"\*\*\*(.+?)\*\*\*", re.DOTALL), r"\1"),          # ***bold italic***
    (re.compile(r"\*\*(.+?)\*\*", re.DOTALL), r"\1"),              # **bold**
    (re.compile(r"(?<!\w)\*(?!\s)(.+?)(?<!\s)\*(?!\w)", re.DOTALL), r"\1"),  # *italic*
    (re.compile(r"(?<!\w)__(.+?)__(?!\w)", re.DOTALL), r"\1"),     # __bold__
    (re.compile(r"```[a-zA-Z]*\n?"), ""),                           # fenced code
    (re.compile(r"`([^`]+)`"), r"\1"),                              # `inline code`
    (re.compile(r"!?\[([^\]]*)\]\([^)]*\)"), r"\1"),                # [text](url)
    (re.compile(r"^\s{0,3}>\s?", re.MULTILINE), ""),                # > blockquote
    (re.compile(r"^\s{0,3}([-*+])\s+", re.MULTILINE), "- "),        # list bullets -> "- "
    (re.compile(r"^\s{0,3}(?:-\s*){3,}$", re.MULTILINE), ""),       # --- horizontal rule
]


def _strip_markdown(text: str) -> str:
    """Remove Markdown markup, keeping the underlying words."""
    for pattern, replacement in _MD_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def sanitize_for_pdf(text: Any) -> str:
    """Make arbitrary text safe for fpdf2's Latin-1 core fonts.

    Applied in order: strip Markdown, map known Unicode punctuation/emoji to
    ASCII equivalents, then decompose-or-drop anything still unencodable.
    Latin-1 characters that fpdf2 *can* render (accented letters such as
    "e-acute", currency symbols) are deliberately preserved rather than
    stripped, so non-English column names survive intact.

    Args:
        text: Any value; non-strings are stringified first so numbers/None
            passed from a dataframe cell can go straight through.

    Returns:
        A string guaranteed to encode cleanly as Latin-1.
    """
    if text is None:
        return ""
    text = str(text)
    if not text:
        return ""

    text = _strip_markdown(text)

    for source, target in _EMOJI_REPLACEMENTS.items():
        text = text.replace(source, target)
    for source, target in _UNICODE_REPLACEMENTS.items():
        text = text.replace(source, target)

    # Catch-all for anything still outside Latin-1 (unlisted emoji, CJK, rare
    # symbols): try a compatibility decomposition first so characters like a
    # fullwidth digit degrade to their ASCII form, and drop what remains.
    result = []
    for char in text:
        try:
            char.encode("latin-1")
            result.append(char)
            continue
        except UnicodeEncodeError:
            pass
        decomposed = unicodedata.normalize("NFKD", char).encode("latin-1", "ignore").decode("latin-1")
        result.append(decomposed)

    # Variation selectors / zero-width joiners left behind by emoji sequences
    # decompose to nothing, which can leave doubled spaces -- tidy those up.
    return re.sub(r"[ \t]{2,}", " ", "".join(result))


def _break_long_tokens(pdf: FPDF, text: str, max_width: float) -> str:
    """Hard-split any single token too wide to fit on one line.

    fpdf2 wraps on word boundaries; a token with no break opportunity (a long
    URL, a base64 blob, an unspaced identifier) that exceeds the usable width
    has nowhere to wrap. Splitting by *measured* width rather than a fixed
    character count is what makes this correct for any font size or glyph mix.
    """
    if max_width <= 0:
        return text

    out_lines = []
    for line in text.split("\n"):
        pieces = []
        for token in line.split(" "):
            if pdf.get_string_width(token) <= max_width:
                pieces.append(token)
                continue
            chunk = ""
            for char in token:
                if pdf.get_string_width(chunk + char) > max_width and chunk:
                    pieces.append(chunk)
                    chunk = char
                else:
                    chunk += char
            if chunk:
                pieces.append(chunk)
        out_lines.append(" ".join(pieces))
    return "\n".join(out_lines)


def _write_block(pdf: FPDF, text: str, usable_width: float, line_height: float) -> None:
    """Write one text block, always starting from the left margin.

    ``new_x=XPos.LMARGIN`` is the critical part: fpdf2's ``multi_cell`` default
    is ``XPos.RIGHT``, which parks the cursor at the right edge of the cell it
    just drew. A following ``multi_cell(w=0, ...)`` then resolves its width to
    "right margin minus current x" = 0mm and raises
    "Not enough horizontal space to render a single character". Passing an
    explicit width and resetting x to the left margin makes each block
    independent of whatever was drawn before it.
    """
    safe = _break_long_tokens(pdf, text, usable_width)
    for paragraph in safe.split("\n"):
        if not paragraph.strip():
            pdf.ln(line_height / 2)
            continue
        pdf.multi_cell(
            usable_width, line_height, paragraph,
            new_x=XPos.LMARGIN, new_y=YPos.NEXT,
        )


# --------------------------------------------------------------------------- #
# Chart rendering (matplotlib -> PNG bytes, embedded via fpdf2's image())
# --------------------------------------------------------------------------- #
def build_histogram_png(values: Sequence[float], title: str = "", x_label: str = "", bins: int = 15) -> bytes:
    """Render a simple histogram to PNG bytes for embedding in a PDF (e.g. a
    regression model's predicted-value distribution, where categories don't apply).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.6, 3.0), dpi=150)
    ax.hist(values, bins=bins, color="#2563eb", edgecolor="#1e293b", linewidth=0.4)
    ax.set_title(title, fontsize=11, color="#0f172a", fontweight="bold", pad=10)
    if x_label:
        ax.set_xlabel(x_label, fontsize=9, color="#475569")
    ax.set_ylabel("Count", fontsize=9, color="#475569")
    ax.tick_params(axis="both", labelsize=8, colors="#334155")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.6)
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def build_bar_chart_png(
    categories: Sequence[str],
    values: Sequence[float],
    title: str = "",
    y_label: str = "",
    highlight_index: Optional[int] = None,
) -> bytes:
    """Render a simple bar chart to PNG bytes for embedding in a PDF.

    Uses matplotlib (already a project dependency) rather than exporting a
    Plotly figure, which would require the optional ``kaleido`` package.

    Args:
        categories: Bar labels (x-axis).
        values: Bar heights (y-axis).
        title: Chart title.
        y_label: Y-axis label.
        highlight_index: If given, that bar is drawn in the accent color and
            the rest in a neutral gray, so "the best one" reads at a glance.

    Returns:
        PNG image bytes.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless rendering, no GUI backend required
    import matplotlib.pyplot as plt

    accent = "#2563eb"
    neutral = "#94a3b8"
    colors = [
        accent if highlight_index is not None and i == highlight_index else neutral
        for i in range(len(categories))
    ]

    fig, ax = plt.subplots(figsize=(6.6, 3.0), dpi=150)
    ax.bar(categories, values, color=colors, edgecolor="#1e293b", linewidth=0.5)
    ax.set_title(title, fontsize=11, color="#0f172a", fontweight="bold", pad=10)
    if y_label:
        ax.set_ylabel(y_label, fontsize=9, color="#475569")
    ax.tick_params(axis="both", labelsize=8, colors="#334155")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.6)
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# Structured business report
# --------------------------------------------------------------------------- #
@dataclass
class ReportSection:
    """One AI-narrative heading + body block (e.g. "Executive Summary")."""

    heading: str
    body: str


@dataclass
class ChartImage:
    """One chart to embed, already rendered to PNG bytes."""

    caption: str
    png_bytes: bytes


@dataclass
class BusinessReportData:
    """Everything needed to render one PDF business report.

    Every field is either data the caller already had in session state, or a
    presentation-only derivation of it (e.g. picking the ``Rank == 1`` row) --
    this module never recomputes or reinterprets ML/business results.
    """

    report_title: str = "Business Analytics Summary Report"
    dataset_name: Optional[str] = None
    generated_at: str = ""
    dataset_overview: Optional[Dict[str, Any]] = None
    leaderboard: Optional[pd.DataFrame] = None
    best_model: Optional[Dict[str, Any]] = None
    narrative_sections: List[ReportSection] = field(default_factory=list)
    predictions_overview: Optional[Dict[str, Any]] = None
    charts: List[ChartImage] = field(default_factory=list)


# Brand palette, matching the app's own CSS custom properties (dashboard/components/landing.py)
_NAVY = (15, 23, 42)
_ACCENT = (37, 99, 235)
_ACCENT_DARK = (29, 78, 216)
_GREEN = (5, 150, 105)
_MUTED = (100, 116, 139)
_LIGHT_BG = (239, 246, 255)
_GREEN_BG = (236, 253, 245)
_BORDER = (203, 213, 225)
_WHITE = (255, 255, 255)

_MARGIN = 18  # mm -- more generous than fpdf2's 10mm default, reads as "designed" not "default"
_COVER_BAND_HEIGHT = 62


class _BusinessReportPDF(FPDF):
    """FPDF subclass whose ``footer()`` hook draws page numbers + a running
    footer on every page (fpdf2 calls this automatically per page)."""

    report_title: str = ""
    generated_at: str = ""

    def footer(self) -> None:
        self.set_y(-15)
        self.set_draw_color(*_BORDER)
        self.line(_MARGIN, self.get_y(), self.w - _MARGIN, self.get_y())
        self.set_y(-12)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*_MUTED)
        self.cell(0, 6, sanitize_for_pdf(f"{self.report_title} | Generated {self.generated_at}"), align="L")
        self.set_xy(_MARGIN, -12)
        self.cell(self.w - 2 * _MARGIN, 6, f"Page {self.page_no()} of {{nb}}", align="R")


def _heading(pdf: FPDF, text: str, usable_width: float, level: int = 1) -> None:
    """A consistently-styled section heading (H1/H2) with a small accent rule."""
    pdf.ln(2)
    if level == 1:
        pdf.set_font("Helvetica", "B", 15)
        pdf.set_text_color(*_NAVY)
        _write_block(pdf, sanitize_for_pdf(text), usable_width, 8)
        pdf.set_draw_color(*_ACCENT)
        pdf.set_line_width(0.6)
        pdf.line(pdf.l_margin, pdf.get_y() + 1, pdf.l_margin + 26, pdf.get_y() + 1)
        pdf.set_line_width(0.2)
        pdf.ln(4)
    else:
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(*_ACCENT_DARK)
        _write_block(pdf, sanitize_for_pdf(text), usable_width, 7)
        pdf.ln(1)


def _body_text(pdf: FPDF, text: str, usable_width: float) -> None:
    pdf.set_font("Helvetica", "", 10.5)
    pdf.set_text_color(*_NAVY)
    _write_block(pdf, sanitize_for_pdf(text), usable_width, 5.6)


def _render_stat_cards(pdf: FPDF, cards: List[Tuple[str, str]], usable_width: float) -> None:
    """A row of small bordered "at a glance" stat boxes, used on the cover page."""
    if not cards:
        return
    gap = 4
    n = len(cards)
    card_w = (usable_width - gap * (n - 1)) / n
    card_h = 22
    x0, y0 = pdf.l_margin, pdf.get_y()

    for i, (label, value) in enumerate(cards):
        x = x0 + i * (card_w + gap)
        pdf.set_draw_color(*_BORDER)
        pdf.set_fill_color(*_LIGHT_BG)
        pdf.rect(x, y0, card_w, card_h, style="DF")

        pdf.set_xy(x, y0 + 4)
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(*_ACCENT_DARK)
        pdf.cell(card_w, 7, sanitize_for_pdf(value), align="C", new_x=XPos.LEFT, new_y=YPos.TOP)

        pdf.set_xy(x, y0 + 13)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*_MUTED)
        pdf.cell(card_w, 5, sanitize_for_pdf(label), align="C", new_x=XPos.LEFT, new_y=YPos.TOP)

    pdf.set_xy(x0, y0 + card_h + 6)


def _render_cover(pdf: FPDF, data: BusinessReportData, usable_width: float) -> None:
    """Page 1: title band, dataset/timestamp, and an at-a-glance stat strip."""
    pdf.add_page()

    pdf.set_fill_color(*_NAVY)
    pdf.rect(0, 0, pdf.w, _COVER_BAND_HEIGHT, style="F")

    pdf.set_xy(_MARGIN, 20)
    pdf.set_text_color(*_WHITE)
    pdf.set_font("Helvetica", "B", 22)
    pdf.multi_cell(usable_width, 10, sanitize_for_pdf(data.report_title), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_x(_MARGIN)
    pdf.set_font("Helvetica", "", 11)
    subtitle = f"Dataset: {data.dataset_name}" if data.dataset_name else "AI-Powered Decision Intelligence Platform"
    pdf.cell(usable_width, 7, sanitize_for_pdf(subtitle), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_xy(_MARGIN, _COVER_BAND_HEIGHT + 10)
    pdf.set_text_color(*_MUTED)
    pdf.set_font("Helvetica", "", 9.5)
    pdf.cell(usable_width, 5, sanitize_for_pdf(f"Generated {data.generated_at}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(6)

    cards: List[Tuple[str, str]] = []
    overview = data.dataset_overview or {}
    if "Rows" in overview:
        cards.append(("Dataset Rows", str(overview["Rows"])))
    if "Columns" in overview:
        cards.append(("Dataset Columns", str(overview["Columns"])))
    if data.leaderboard is not None and not data.leaderboard.empty:
        cards.append(("Models Compared", str(len(data.leaderboard))))
    if data.best_model:
        cards.append(("Best Model", str(data.best_model.get("Model", "--"))))
    _render_stat_cards(pdf, cards, usable_width)


def _render_key_value_table(pdf: FPDF, pairs: Sequence[Tuple[str, str]], usable_width: float) -> None:
    """A simple two-column "Metric | Value" table (used for Dataset Overview)."""
    if not pairs:
        return
    pdf.set_font("Helvetica", "", 10)
    with pdf.table(
        col_widths=(usable_width * 0.45, usable_width * 0.55),
        text_align=("LEFT", "LEFT"),
        borders_layout="ALL",
        cell_fill_color=_LIGHT_BG,
        cell_fill_mode="ROWS",
        headings_style=FontFace(emphasis="BOLD", color=_WHITE, fill_color=_ACCENT),
        line_height=6.5,
        first_row_as_headings=True,
    ) as table:
        header = table.row()
        header.cell("Metric")
        header.cell("Value")
        for label, value in pairs:
            row = table.row()
            row.cell(sanitize_for_pdf(label))
            row.cell(sanitize_for_pdf(value))
    pdf.ln(4)


def _leaderboard_col_widths(columns: Sequence[str], usable_width: float) -> List[float]:
    """Give the Model name column more room than narrow numeric metric columns."""
    weights = []
    for col in columns:
        if col == "Rank":
            weights.append(0.55)
        elif col == "Model":
            weights.append(2.1)
        elif col == "Training Time (s)":
            weights.append(1.3)
        else:
            weights.append(1.0)
    total = sum(weights)
    return [usable_width * w / total for w in weights]


def _format_leaderboard_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _render_leaderboard_table(pdf: FPDF, leaderboard: pd.DataFrame, usable_width: float) -> None:
    columns = list(leaderboard.columns)
    col_widths = _leaderboard_col_widths(columns, usable_width)

    pdf.set_font("Helvetica", "", 8.5)
    with pdf.table(
        col_widths=col_widths,
        text_align="CENTER",
        borders_layout="ALL",
        cell_fill_color=_LIGHT_BG,
        cell_fill_mode="ROWS",
        headings_style=FontFace(emphasis="BOLD", color=_WHITE, fill_color=_ACCENT, size_pt=8.5),
        line_height=6,
        first_row_as_headings=True,
    ) as table:
        header = table.row()
        for col in columns:
            header.cell(sanitize_for_pdf(col))
        for _, record in leaderboard.iterrows():
            row = table.row()
            for col in columns:
                row.cell(sanitize_for_pdf(_format_leaderboard_value(record[col])))
    pdf.ln(4)


def _render_best_model_callout(pdf: FPDF, best_model: Dict[str, Any], usable_width: float) -> None:
    """A highlighted box calling out the best-performing model and its metrics."""
    metric_pairs = [
        (k, _format_leaderboard_value(v)) for k, v in best_model.items()
        if k not in ("Rank", "Model", "Training Time (s)")
    ]
    metric_line = "   |   ".join(f"{k}: {v}" for k, v in metric_pairs[:4])

    pdf.set_draw_color(*_GREEN)
    pdf.set_fill_color(*_GREEN_BG)
    pdf.set_line_width(0.5)
    box_y = pdf.get_y()
    box_h = 26 if metric_line else 18
    pdf.rect(pdf.l_margin, box_y, usable_width, box_h, style="DF")
    pdf.set_line_width(0.2)

    pdf.set_xy(pdf.l_margin + 5, box_y + 4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(*_GREEN)
    pdf.cell(usable_width - 10, 7, sanitize_for_pdf(f"Best Performing Model: {best_model.get('Model', 'N/A')}"))

    if metric_line:
        pdf.set_xy(pdf.l_margin + 5, box_y + 13)
        pdf.set_font("Helvetica", "", 9.5)
        pdf.set_text_color(*_NAVY)
        pdf.cell(usable_width - 10, 6, sanitize_for_pdf(metric_line))

    pdf.set_xy(pdf.l_margin, box_y + box_h + 6)


def build_business_report_pdf(data: BusinessReportData) -> bytes:
    """Build a structured, multi-section executive PDF report.

    Layout: cover page (title band + at-a-glance stats) -> Executive Summary
    -> Dataset Overview table -> Model Leaderboard table + optional chart ->
    Best Model callout -> remaining AI narrative sections (Key Insights,
    Business Recommendations, Risk Assessment, Suggested Next Steps) ->
    Predictions Overview + optional chart. Every page carries a footer with
    page numbers and the generation timestamp.

    Args:
        data: The report content -- see ``BusinessReportData``.

    Returns:
        Raw PDF bytes suitable for a Streamlit download button.
    """
    pdf = _BusinessReportPDF()
    pdf.report_title = data.report_title
    pdf.generated_at = data.generated_at
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=22)
    pdf.set_margins(_MARGIN, _MARGIN, _MARGIN)

    usable_width = pdf.w - pdf.l_margin - pdf.r_margin

    _render_cover(pdf, data, usable_width)

    pdf.add_page()

    narrative_by_heading = {s.heading: s.body for s in data.narrative_sections}

    exec_summary = narrative_by_heading.pop("Executive Summary", None)
    if exec_summary:
        _heading(pdf, "Executive Summary", usable_width, level=1)
        _body_text(pdf, exec_summary, usable_width)
        pdf.ln(4)

    if data.dataset_overview:
        _heading(pdf, "Dataset Overview", usable_width, level=1)
        _render_key_value_table(pdf, list(data.dataset_overview.items()), usable_width)

    if data.leaderboard is not None and not data.leaderboard.empty:
        _heading(pdf, "Model Leaderboard", usable_width, level=1)
        _render_leaderboard_table(pdf, data.leaderboard, usable_width)

        leaderboard_chart = next((c for c in data.charts if c.caption == "leaderboard"), None)
        if leaderboard_chart:
            pdf.image(io.BytesIO(leaderboard_chart.png_bytes), x=pdf.l_margin, w=usable_width)
            pdf.ln(4)

    if data.best_model:
        _render_best_model_callout(pdf, data.best_model, usable_width)

    for heading in ("Key Insights", "Business Recommendations", "Risk Assessment", "Suggested Next Steps"):
        body = narrative_by_heading.get(heading)
        if body:
            _heading(pdf, heading, usable_width, level=1)
            _body_text(pdf, body, usable_width)
            pdf.ln(4)

    if data.predictions_overview:
        _heading(pdf, "Predictions Overview", usable_width, level=1)
        _render_key_value_table(pdf, list(data.predictions_overview.items()), usable_width)

        predictions_chart = next((c for c in data.charts if c.caption == "predictions"), None)
        if predictions_chart:
            pdf.image(io.BytesIO(predictions_chart.png_bytes), x=pdf.l_margin, w=usable_width)
            pdf.ln(4)

    raw_output = pdf.output()
    return bytes(raw_output)


def render_pdf_download(data: BusinessReportData, file_name: str, label: str = "Download PDF") -> None:
    """Render a Streamlit download button that exports the structured business report."""
    try:
        pdf_bytes = build_business_report_pdf(data)
    except Exception as exc:  # never let a malformed section break the whole page
        st.error(f"Could not build the PDF report: {exc}")
        return
    st.download_button(label, data=pdf_bytes, file_name=file_name, mime="application/pdf")
