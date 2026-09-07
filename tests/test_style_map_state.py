"""Tests for PCA style map presentation state."""

from unittest import TestCase

from controllers.gui_dtos import PCAMapPoint, UploadedMapPoint
from views.style_map_state import (
    StyleMapState,
    map_point_to_scatter_point,
    uploaded_to_scatter_point,
)


class StyleMapStateTests(TestCase):
    """Verify style map filtering, searching, and DTO conversion."""

    def test_filter_derives_visible_and_hidden_points(self) -> None:
        """A selected style should hide all other labels."""
        state = StyleMapState(_sample_points())

        state.active_style = "Cubism"

        self.assertEqual(state.visible_count, 1)
        self.assertEqual(state.hidden_labels(), {"Impressionism"})
        self.assertIn("当前显示 1 个", state.status_text())

    def test_search_matches_id_title_artist_or_style(self) -> None:
        """Search should match all user-facing point metadata."""
        state = StyleMapState(_sample_points())

        self.assertEqual(state.find_point("2").title, "Blue Study")
        self.assertEqual(state.find_point("monet").artwork_id, 1)
        self.assertEqual(state.find_point("cubism").artwork_id, 2)
        self.assertEqual(state.find_point("印象派").artwork_id, 1)
        self.assertIsNone(state.find_point("missing"))

    def test_map_point_conversion_preserves_hover_metadata(self) -> None:
        """Converted scatter points should retain metadata used by hover text."""
        point = _sample_points()[0]

        scatter_point = map_point_to_scatter_point(point)

        self.assertEqual(scatter_point.sample_id, "1")
        self.assertEqual(scatter_point.artist_name, "Claude Monet")
        self.assertEqual(scatter_point.image_path, "monet.jpg")

    def test_uploaded_point_conversion_uses_uploaded_id(self) -> None:
        """Uploaded image projections use a stable synthetic sample id."""
        uploaded = UploadedMapPoint(
            x_value=0.5,
            y_value=0.7,
            z_value=None,
            style_label="Impressionism",
            artist_name="上传图片",
            confidence=0.82,
            image_path="query.jpg",
            title="query.jpg",
        )

        scatter_point = uploaded_to_scatter_point(uploaded)

        self.assertEqual(scatter_point.sample_id, "uploaded")
        self.assertEqual(scatter_point.label, "Impressionism")


def _sample_points() -> list[PCAMapPoint]:
    """Return two deterministic style map points."""
    return [
        PCAMapPoint(
            artwork_id=1,
            x_value=0.1,
            y_value=0.2,
            z_value=0.3,
            style_label="Impressionism",
            image_path="monet.jpg",
            title="Water Lilies",
            artist_name="Claude Monet",
        ),
        PCAMapPoint(
            artwork_id=2,
            x_value=1.1,
            y_value=1.2,
            z_value=1.3,
            style_label="Cubism",
            image_path="picasso.jpg",
            title="Blue Study",
            artist_name="Pablo Picasso",
        ),
    ]
