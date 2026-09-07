"""Tests for user-facing art style translations."""

from unittest import TestCase

from services.style_translations import (
    format_style_distribution,
    format_style_label,
    translate_style_label,
)


class StyleTranslationTests(TestCase):
    """Verify common dataset style labels display with Chinese translations."""

    def test_current_dataset_styles_have_translations(self) -> None:
        """The known WikiArt subset styles should be translated."""
        labels = (
            "Academic_Art",
            "Art_Nouveau",
            "Baroque",
            "Expressionism",
            "Japanese_Art",
            "Neoclassicism",
            "Primitivism",
            "Realism",
            "Renaissance",
            "Rococo",
            "Romanticism",
            "Symbolism",
            "Western_Medieval",
        )

        for label in labels:
            with self.subTest(label=label):
                self.assertIsNotNone(translate_style_label(label))
                self.assertIn("（", format_style_label(label))

    def test_unknown_label_is_left_unchanged(self) -> None:
        """Unexpected custom styles should not be modified."""
        self.assertEqual(format_style_label("CustomStyle"), "CustomStyle")

    def test_distribution_formats_keys_only(self) -> None:
        """Counts should stay unchanged while style names become display labels."""
        formatted = format_style_distribution({"Baroque": 3, "CustomStyle": 2})

        self.assertEqual(formatted["Baroque（巴洛克）"], 3)
        self.assertEqual(formatted["CustomStyle"], 2)
