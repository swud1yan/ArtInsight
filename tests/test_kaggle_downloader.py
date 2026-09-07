"""Tests for the pure Python Kaggle downloader integration."""

from __future__ import annotations

from pathlib import Path
import os
import unittest
from unittest import mock

import cv2
import numpy as np

from experiments import kaggle_downloader
from experiments.kaggle_downloader import (
    AUTH_FAILURE_MESSAGE,
    DatasetNotFoundError,
    KaggleDownloaderError,
    KaggleAuthenticationError,
    KaggleDatasetAccessError,
    DatasetVerificationStatus,
    download_wikiart_dataset,
    init_kaggle,
    verify_dataset,
)
from tests.temp_utils import ProjectTempDirectory


class KaggleDownloaderTestCase(unittest.TestCase):
    """Validate Kaggle API auth, integrity checks, and pipeline handoff."""

    def setUp(self) -> None:
        """Create project-local temporary folders."""
        self.temp_dir = ProjectTempDirectory("kaggle_downloader")
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    def test_init_kaggle_sets_env_and_authenticates_without_cli(self) -> None:
        """init_kaggle should use process env credentials and KaggleApi."""
        fake_api = _FakeKaggleApi()
        fake_module = mock.Mock()
        fake_module.KaggleApi.return_value = fake_api
        with mock.patch.object(
            kaggle_downloader.importlib,
            "import_module",
            return_value=fake_module,
        ):
            api = init_kaggle("paper_user", "paper_key")

        self.assertIs(api, fake_api)
        self.assertTrue(fake_api.authenticated)
        self.assertEqual(os.environ["KAGGLE_USERNAME"], "paper_user")
        self.assertEqual(os.environ["KAGGLE_KEY"], "paper_key")
        self.assertNotIn("KAGGLE_API_TOKEN", os.environ)

    def test_init_kaggle_rejects_missing_credentials(self) -> None:
        """Missing user input should produce the required auth message."""
        with self.assertRaisesRegex(KaggleAuthenticationError, AUTH_FAILURE_MESSAGE):
            init_kaggle("", "")

    def test_init_kaggle_treats_auth_probe_403_as_bad_credentials(self) -> None:
        """A 403 during the auth probe should keep the required auth message."""
        fake_api = _FakeKaggleApi(list_error=RuntimeError("403 Forbidden"))
        fake_module = mock.Mock()
        fake_module.KaggleApi.return_value = fake_api
        with mock.patch.object(
            kaggle_downloader.importlib,
            "import_module",
            return_value=fake_module,
        ):
            with self.assertRaisesRegex(
                KaggleAuthenticationError,
                AUTH_FAILURE_MESSAGE,
            ):
                init_kaggle("paper_user", "bad_key")

    def test_verify_dataset_reports_empty_corrupted_and_valid(self) -> None:
        """Integrity check should return the required status strings."""
        empty = self.root / "empty"
        empty.mkdir()
        flat = self.root / "flat"
        self._write_image(flat / "painting.jpg")
        valid = self.root / "valid"
        for class_name in ("cubism", "baroque"):
            for index in range(3):
                self._write_image(valid / class_name / f"{index}.jpg")

        with mock.patch.object(kaggle_downloader, "MIN_SAMPLE_COUNT", 5):
            self.assertEqual(verify_dataset(str(empty)), "EMPTY")
            self.assertEqual(verify_dataset(str(flat)), "CORRUPTED")
            self.assertEqual(verify_dataset(str(valid)), "VALID")

    def test_download_triggers_loader_and_stats_after_valid_download(self) -> None:
        """A valid fake API download should run ArtDatasetLoader and DatasetStats."""
        target = self.root / "wikiart"
        api = _FakeDownloadApi()
        with mock.patch.object(kaggle_downloader, "MIN_SAMPLE_COUNT", 5):
            report = download_wikiart_dataset(
                "paper_user",
                "paper_key",
                path=target,
                api=api,
            )

        self.assertEqual(api.calls[0]["dataset"], "ipythonx/wikiart-dataset")
        self.assertEqual(api.calls[0]["path"], str(target.resolve()))
        self.assertTrue(api.calls[0]["unzip"])
        self.assertTrue(report.success)
        self.assertEqual(report.status, DatasetVerificationStatus.VALID)
        self.assertEqual(report.image_count, 6)
        self.assertEqual(report.class_count, 2)
        self.assertEqual(report.loaded_record_count, 6)
        self.assertIsNotNone(report.statistics)
        assert report.statistics is not None
        self.assertEqual(report.statistics.total_images, 6)

    def test_download_rejects_placeholder_dataset_ref(self) -> None:
        """Documentation placeholders should fail before a Kaggle request."""
        with self.assertRaisesRegex(KaggleDownloaderError, "Replace owner/slug"):
            download_wikiart_dataset(
                "paper_user",
                "paper_key",
                dataset="owner/slug",
                path=self.root / "placeholder",
                api=_FakeDownloadApi(),
            )

    def test_download_normalizes_access_and_missing_dataset_errors(self) -> None:
        """403 downloads and dataset-not-found failures should be explicit."""
        with self.assertRaisesRegex(
            KaggleDatasetAccessError,
            "Kaggle dataset access failed",
        ):
            download_wikiart_dataset(
                "paper_user",
                "paper_key",
                path=self.root / "auth",
                api=_FailingApi(RuntimeError("403 Forbidden")),
            )
        with self.assertRaisesRegex(DatasetNotFoundError, "DatasetNotFound"):
            download_wikiart_dataset(
                "paper_user",
                "paper_key",
                path=self.root / "missing",
                api=_FailingApi(RuntimeError("DatasetNotFound: no such dataset")),
            )

    def _write_image(self, image_path: Path) -> Path:
        """Write a small valid image file."""
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image = np.full((12, 10, 3), 128, dtype=np.uint8)
        if not cv2.imwrite(str(image_path), image):
            msg = f"Failed to write test image: {image_path}"
            raise RuntimeError(msg)
        return image_path


class _FakeKaggleApi:
    """Minimal fake API for auth tests."""

    def __init__(self, list_error: Exception | None = None) -> None:
        self.authenticated = False
        self._list_error = list_error

    def authenticate(self) -> None:
        """Record authentication."""
        self.authenticated = True

    def dataset_list(self, **kwargs: object) -> list[object]:
        """Return a fake authenticated dataset listing."""
        if self._list_error is not None:
            raise self._list_error
        return []


class _FakeDownloadApi:
    """Fake downloader that creates a WikiArt-style folder dataset."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def dataset_download_files(
        self,
        dataset: str,
        path: str,
        unzip: bool = False,
        **_: object,
    ) -> None:
        """Create a deterministic tiny dataset in the requested path."""
        self.calls.append({"dataset": dataset, "path": path, "unzip": unzip})
        root = Path(path)
        for class_name in ("cubism", "baroque"):
            for index in range(3):
                image_path = root / class_name / f"{index}.jpg"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image = np.full((12, 10, 3), 128 + index, dtype=np.uint8)
                if not cv2.imwrite(str(image_path), image):
                    raise RuntimeError(f"Failed to write fake image: {image_path}")


class _FailingApi:
    """Fake downloader that raises a configured exception."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def dataset_download_files(
        self,
        dataset: str,
        path: str,
        unzip: bool = False,
        **kwargs: object,
    ) -> None:
        """Raise the configured exception."""
        raise self._exc


if __name__ == "__main__":
    unittest.main()
