"""ReportLab styles and font setup for ArtInsight PDF reports."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import TableStyle

CHINESE_FONT_CANDIDATES: tuple[Path, ...] = (
    Path("C:/Windows/Fonts/msyh.ttc"),
    Path("C:/Windows/Fonts/simhei.ttf"),
    Path("C:/Windows/Fonts/simsun.ttc"),
)


def register_report_font() -> str:
    """Register the best available local Chinese font for ReportLab."""
    for font_path in CHINESE_FONT_CANDIDATES:
        if not font_path.exists():
            continue
        try:
            pdfmetrics.registerFont(TTFont("ArtInsightSans", str(font_path)))
            return "ArtInsightSans"
        except (OSError, ValueError):
            continue
    return "Helvetica"


def build_report_styles(font_name: str) -> dict[str, ParagraphStyle]:
    """Build ReportLab paragraph styles using the registered Chinese font."""
    base = getSampleStyleSheet()
    return {
        "cover_title": ParagraphStyle(
            "ArtInsightCoverTitle",
            parent=base["Title"],
            fontName=font_name,
            fontSize=28,
            leading=34,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#1F2A30"),
        ),
        "cover_subtitle": ParagraphStyle(
            "ArtInsightCoverSubtitle",
            parent=base["Normal"],
            fontName=font_name,
            fontSize=14,
            leading=20,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#53646B"),
        ),
        "heading": ParagraphStyle(
            "ArtInsightHeading",
            parent=base["Heading1"],
            fontName=font_name,
            fontSize=17,
            leading=23,
            spaceAfter=12,
            textColor=colors.HexColor("#1F2A30"),
        ),
        "subheading": ParagraphStyle(
            "ArtInsightSubheading",
            parent=base["Heading2"],
            fontName=font_name,
            fontSize=13,
            leading=18,
            spaceAfter=8,
            textColor=colors.HexColor("#2F6F73"),
        ),
        "body": ParagraphStyle(
            "ArtInsightBody",
            parent=base["BodyText"],
            fontName=font_name,
            fontSize=10.5,
            leading=17,
            alignment=TA_LEFT,
            spaceAfter=8,
        ),
        "small": ParagraphStyle(
            "ArtInsightSmall",
            parent=base["BodyText"],
            fontName=font_name,
            fontSize=8,
            leading=12,
            alignment=TA_CENTER,
        ),
    }


def build_report_table_style(font_name: str) -> TableStyle:
    """Return the shared report table style."""
    return TableStyle(
        [
            ("FONTNAME", (0, 0), (-1, -1), font_name),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF1F2")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1F2A30")),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D9E1E2")),
            (
                "ROWBACKGROUNDS",
                (0, 1),
                (-1, -1),
                [colors.white, colors.HexColor("#F7FAFA")],
            ),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ],
    )
