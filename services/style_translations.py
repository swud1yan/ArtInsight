"""User-facing Chinese translations for art style labels."""

from __future__ import annotations

import re


STYLE_TRANSLATIONS: dict[str, str] = {
    "abstract art": "抽象艺术",
    "abstract expressionism": "抽象表现主义",
    "academic art": "学院派艺术",
    "action painting": "行动绘画",
    "analytical cubism": "分析立体主义",
    "art brut": "原生艺术",
    "art deco": "装饰艺术",
    "art informel": "非定形艺术",
    "art nouveau": "新艺术运动",
    "baroque": "巴洛克",
    "cartoon": "卡通",
    "chinese ink painting": "中国水墨画",
    "color field painting": "色域绘画",
    "contemporary realism": "当代现实主义",
    "constructivism": "构成主义",
    "cubism": "立体主义",
    "dada": "达达主义",
    "divisionism": "分色主义",
    "early renaissance": "早期文艺复兴",
    "expressionism": "表现主义",
    "fauvism": "野兽派",
    "figurative expressionism": "具象表现主义",
    "futurism": "未来主义",
    "hard edge painting": "硬边绘画",
    "high renaissance": "盛期文艺复兴",
    "impressionism": "印象派",
    "ink and wash painting": "水墨画",
    "international gothic": "国际哥特式",
    "japanese art": "日本艺术",
    "japonism": "日本主义",
    "lyrical abstraction": "抒情抽象",
    "magic realism": "魔幻现实主义",
    "mannerism": "矫饰主义",
    "mannerism late renaissance": "矫饰主义（晚期文艺复兴）",
    "minimalism": "极简主义",
    "naive art primitivism": "朴素艺术/原始主义",
    "naturalism": "自然主义",
    "neoclassicism": "新古典主义",
    "northern renaissance": "北方文艺复兴",
    "op art": "欧普艺术",
    "orientalism": "东方主义",
    "photorealism": "照相写实主义",
    "pointillism": "点彩派",
    "pop art": "波普艺术",
    "post impressionism": "后印象派",
    "post-impressionism": "后印象派",
    "pre raphaelite": "前拉斐尔派",
    "pre-raphaelite": "前拉斐尔派",
    "primitivism": "原始主义",
    "proto renaissance": "前文艺复兴",
    "realism": "现实主义",
    "renaissance": "文艺复兴",
    "rococo": "洛可可",
    "romanticism": "浪漫主义",
    "social realism": "社会现实主义",
    "suprematism": "至上主义",
    "surrealism": "超现实主义",
    "symbolism": "象征主义",
    "synthetic cubism": "综合立体主义",
    "tachisme": "塔希主义",
    "tonalism": "调性主义",
    "ukiyo e": "浮世绘",
    "ukiyo-e": "浮世绘",
    "western medieval": "西方中世纪艺术",
}


def normalize_style_label(label: str) -> str:
    """Return a lookup key that tolerates WikiArt folder naming variants."""
    normalized = label.strip().lower()
    normalized = re.sub(r"[_/]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def translate_style_label(label: str) -> str | None:
    """Return a Chinese translation for a style label when one is known."""
    return STYLE_TRANSLATIONS.get(normalize_style_label(label))


def format_style_label(label: str) -> str:
    """Format a style as ``English（中文）`` when a translation is available."""
    translation = translate_style_label(label)
    if translation is None:
        return label
    return f"{label}（{translation}）"


def format_style_distribution(distribution: dict[str, int]) -> dict[str, int]:
    """Return a display-only distribution with translated style labels."""
    return {
        format_style_label(label): count
        for label, count in distribution.items()
    }
