"""Small PDF report builder shared by the dashboard pages (uses fpdf2, no external fonts)."""
from __future__ import annotations

from datetime import datetime

import pandas as pd
from fpdf import FPDF
from fpdf.enums import XPos, YPos

BLUE, GREY = (37, 99, 235), (100, 116, 139)
_SWAP = {"—": "-", "–": "-", "‘": "'", "’": "'", "“": '"', "”": '"', "…": "...",
         "·": "-", "→": "->", "≤": "<=", "≥": ">=", "×": "x", "∞": "inf", "²": "2"}


def clean(text) -> str:
    """The built-in PDF fonts are Latin-1 only: replace anything else."""
    t = str(text)
    for k, v in _SWAP.items():
        t = t.replace(k, v)
    return t.encode("latin-1", "replace").decode("latin-1")


class Report(FPDF):
    def __init__(self, title: str, subtitle: str = ""):
        super().__init__()
        self.report_title, self.subtitle = clean(title), clean(subtitle)
        self.set_auto_page_break(True, margin=16)
        self.set_margins(15, 15, 15)
        self.add_page()
        self.set_font("Helvetica", "B", 20)
        self.set_text_color(15, 23, 42)
        self.cell(0, 11, self.report_title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*GREY)
        line = (self.subtitle + "  |  " if self.subtitle else "") + f"Generated {datetime.now():%d %b %Y %H:%M}"
        self.cell(0, 6, line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*BLUE)
        self.set_line_width(0.6)
        self.line(15, self.get_y() + 2, 195, self.get_y() + 2)
        self.ln(6)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*GREY)
        self.cell(0, 6, f"{self.report_title}  -  page {self.page_no()}", align="C")

    def heading(self, text: str) -> None:
        self.ln(2)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*BLUE)
        self.cell(0, 8, clean(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(15, 23, 42)

    def text(self, body: str, bold: bool = False, size: int = 10) -> None:
        self.set_font("Helvetica", "B" if bold else "", size)
        self.set_text_color(15, 23, 42)
        self.multi_cell(0, 5.2, clean(body), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def bullets(self, items) -> None:
        for it in items:
            self.text(f"-  {it}")

    def facts(self, pairs: dict) -> None:
        """Two-column list of label: value."""
        for k, v in pairs.items():
            self.set_font("Helvetica", "", 10)
            self.set_text_color(*GREY)
            self.cell(62, 6, clean(k))
            self.set_font("Helvetica", "B", 10)
            self.set_text_color(15, 23, 42)
            self.multi_cell(0, 6, clean(v), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def finding(self, tag: str, title: str, detail: str, action: str = "") -> None:
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*BLUE)
        self.cell(0, 5, clean(tag).upper(), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.text(title, bold=True)
        self.text(detail, size=9)
        if action:
            self.text("Recommendation: " + action, size=9)
        self.ln(2)

    def table(self, df: pd.DataFrame, max_rows: int = 25, widths: list[float] | None = None, size: int = 8) -> None:
        df = df.head(max_rows)
        usable = 180
        widths = widths or [usable / len(df.columns)] * len(df.columns)
        self.set_font("Helvetica", "B", size)
        self.set_fill_color(241, 245, 249)
        self.set_text_color(15, 23, 42)
        for col, w in zip(df.columns, widths):
            self.cell(w, 6.5, clean(col)[: int(w / 1.7)], border="B", fill=True)
        self.ln()
        self.set_font("Helvetica", "", size)
        for _, row in df.iterrows():
            for val, w in zip(row, widths):
                if isinstance(val, float):
                    val = f"{val:,.2f}" if abs(val) < 1000 else f"{val:,.0f}"
                self.cell(w, 5.8, clean(val)[: int(w / 1.55)], border="B")
            self.ln()
        self.ln(2)

    def image_bytes(self, png: bytes, width: float = 60) -> None:
        import io
        self.image(io.BytesIO(png), w=width)

    def done(self) -> bytes:
        return bytes(self.output())
