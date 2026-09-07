"""Pure Python Kaggle dataset downloader for ArtInsight.

This module intentionally does not call the Kaggle CLI and does not rely on a
user-level ``kaggle.json`` or OAuth access token. Credentials are supplied to
``init_kaggle`` and written into the process environment immediately before the
Kaggle Python API is imported and authenticated.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import getpass
import importlib
import logging
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Protocol
import zipfile

from requests import RequestException
from requests.exceptions import HTTPError

from experiments.art_dataset_loader import ArtDatasetLoader, ArtDatasetLoaderConfig
from experiments.dataset_compliance import SUPPORTED_IMAGE_SUFFIXES
from experiments.dataset_stats import DatasetStats, DatasetStatistics

LOGGER = logging.getLogger(__name__)

DEFAULT_DATASET = "ipythonx/wikiart-dataset"
DEFAULT_DOWNLOAD_PATH = Path("D:/PCA/data/raw/wikiart")
DEFAULT_COMPETITION = "painter-by-numbers"
DEFAULT_COMPETITION_PATH = Path("D:/PCA/data/raw/painter_by_numbers")
MIN_SAMPLE_COUNT = 500
PROGRESS_INTERVAL_SECONDS = 10.0
AUTH_FAILURE_MESSAGE = (
    "Kaggle authentication failed - please verify username and API key"
)
DATASET_ACCESS_FAILURE_MESSAGE = (
    "Kaggle dataset access failed - credentials were accepted, but Kaggle "
    "returned 403 for this dataset. Open the Dataset URL in a browser while "
    "logged in, accept any rules/terms, and verify the account can access it."
)


class DatasetVerificationStatus(str, Enum):
    """Dataset integrity states returned by ``verify_dataset``."""

    VALID = "VALID"
    EMPTY = "EMPTY"
    CORRUPTED = "CORRUPTED"


@dataclass(frozen=True)
class DatasetVerificationReport:
    """Integrity summary for a downloaded dataset folder."""

    status: DatasetVerificationStatus
    path: Path
    image_count: int
    class_count: int
    byte_count: int
    message: str


@dataclass(frozen=True)
class KaggleDownloadReport:
    """Download outcome for audit logs and reproducible experiments."""

    dataset: str
    output_path: Path
    status: DatasetVerificationStatus
    success: bool
    dataset_size_bytes: int
    image_count: int
    class_count: int
    loaded_record_count: int
    statistics: DatasetStatistics | None
    started_at_utc: str
    finished_at_utc: str
    message: str


class KaggleDownloaderError(RuntimeError):
    """Base exception raised by this downloader."""


class KaggleAuthenticationError(KaggleDownloaderError):
    """Raised when Kaggle credentials are invalid or unauthorized."""


class KaggleDatasetAccessError(KaggleDownloaderError):
    """Raised when credentials work but dataset access is forbidden."""


class DatasetNotFoundError(KaggleDownloaderError):
    """Raised when the requested Kaggle dataset cannot be found."""


class _KaggleApiProtocol(Protocol):
    """Minimal Kaggle API surface used by this module."""

    def dataset_download_files(
        self,
        dataset: str,
        path: str,
        unzip: bool = False,
        **kwargs: Any,
    ) -> None:
        """Download and optionally unzip a Kaggle dataset."""

    def dataset_list(self, **kwargs: Any) -> object:
        """List datasets for a lightweight authenticated API probe."""

    def dataset_list_files(self, dataset: str, **kwargs: Any) -> object:
        """List files for a dataset to verify dataset-specific access."""

    def competition_download_files(
        self,
        competition: str,
        path: str,
        **kwargs: Any,
    ) -> None:
        """Download all files for a Kaggle competition."""

    def competition_list_files(self, competition: str, **kwargs: Any) -> object:
        """List files for a Kaggle competition."""


def init_kaggle(username: str, api_key: str) -> _KaggleApiProtocol:
    """Initialize Kaggle API without relying on kaggle.json.

    Credentials are deliberately supplied through the legacy environment keys
    consumed by ``KaggleApi.authenticate``:

    ``os.environ["KAGGLE_USERNAME"] = username``
    ``os.environ["KAGGLE_KEY"] = api_key``
    """
    username = username.strip()
    api_key = api_key.strip()
    if not username or not api_key:
        raise KaggleAuthenticationError(AUTH_FAILURE_MESSAGE)

    _prepare_kaggle_environment(username, api_key)
    try:
        with _isolated_kaggle_home():
            module = importlib.import_module("kaggle.api.kaggle_api_extended")
            kaggle_api_class = getattr(module, "KaggleApi")
            config_dir = os.environ["KAGGLE_CONFIG_DIR"]
            kaggle_api_class.config_dir = config_dir
            kaggle_api_class.config = str(Path(config_dir) / "kaggle.json")
            kaggle_api_class.config_values = {}
            api = kaggle_api_class()
            api.authenticate()
            _verify_kaggle_credentials(api)
    except SystemExit as exc:
        raise KaggleAuthenticationError(AUTH_FAILURE_MESSAGE) from exc
    except Exception as exc:
        raise _normalize_kaggle_exception(exc, context="auth") from exc
    return api


def verify_dataset(path: str) -> str:
    """Return VALID, EMPTY, or CORRUPTED for a downloaded image dataset."""
    return _verify_dataset_report(Path(path)).status.value


def download_wikiart_dataset(
    username: str,
    api_key: str,
    *,
    dataset: str = DEFAULT_DATASET,
    path: str | Path = DEFAULT_DOWNLOAD_PATH,
    api: _KaggleApiProtocol | None = None,
    progress_interval: float = PROGRESS_INTERVAL_SECONDS,
    expected_size_bytes: int | None = None,
) -> KaggleDownloadReport:
    """Download, verify, and integrate the WikiArt Kaggle dataset."""
    _reject_placeholder_ref(dataset)
    started_at = _utc_now()
    output_path = Path(path).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Starting Kaggle dataset download: dataset=%s path=%s", dataset, output_path)

    try:
        kaggle_api = api if api is not None else init_kaggle(username, api_key)
        expected_bytes = expected_size_bytes or _estimate_dataset_download_size(
            kaggle_api,
            dataset,
        )
        try:
            with _download_progress_monitor(
                output_path,
                label=f"Kaggle dataset {dataset}",
                expected_bytes=expected_bytes,
                interval_seconds=progress_interval,
            ):
                kaggle_api.dataset_download_files(
                    dataset=dataset,
                    path=str(output_path),
                    unzip=True,
                )
        except Exception as exc:
            raise _normalize_kaggle_exception(exc, context="download") from exc
        verification = _verify_dataset_report(output_path)
        LOGGER.info(
            "Kaggle download verification: status=%s images=%s classes=%s size=%s",
            verification.status.value,
            verification.image_count,
            verification.class_count,
            verification.byte_count,
        )
        loaded_records: list[Any] = []
        stats: DatasetStatistics | None = None
        if verification.status == DatasetVerificationStatus.VALID:
            loaded_records, stats = _run_artinsight_pipeline(verification.path)
            message = "Kaggle dataset downloaded, verified, and loaded successfully."
        else:
            message = verification.message
        report = KaggleDownloadReport(
            dataset=dataset,
            output_path=output_path,
            status=verification.status,
            success=verification.status == DatasetVerificationStatus.VALID,
            dataset_size_bytes=verification.byte_count,
            image_count=verification.image_count,
            class_count=verification.class_count,
            loaded_record_count=len(loaded_records),
            statistics=stats,
            started_at_utc=started_at,
            finished_at_utc=_utc_now(),
            message=message,
        )
        _log_report(report)
        return report
    except KaggleDownloaderError:
        LOGGER.error("Download status: FAILED")
        raise
    except Exception as exc:
        normalized = _normalize_kaggle_exception(exc, context="download")
        LOGGER.error("Download status: FAILED")
        LOGGER.error("Kaggle dataset acquisition failed: %s", normalized)
        raise normalized from exc


def main(argv: list[str] | None = None) -> int:
    """Run the downloader from the command line without CLI-based Kaggle auth."""
    parser = argparse.ArgumentParser(
        description="Download WikiArt from Kaggle with Python API credentials.",
    )
    parser.add_argument("--username", help="Kaggle username.")
    parser.add_argument(
        "--api-key",
        help="Kaggle API key. Omit to enter it securely at the prompt.",
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--path", default=str(DEFAULT_DOWNLOAD_PATH))
    parser.add_argument(
        "--source",
        choices=["dataset", "competition"],
        default="dataset",
        help="Kaggle source type.",
    )
    parser.add_argument(
        "--competition",
        default=DEFAULT_COMPETITION,
        help="Kaggle competition slug used when --source competition.",
    )
    parser.add_argument(
        "--check-access",
        action="store_true",
        help="Check whether the account can access the selected dataset.",
    )
    parser.add_argument(
        "--search-candidates",
        action="store_true",
        help="List Kaggle datasets matching --search-term instead of downloading.",
    )
    parser.add_argument(
        "--search-term",
        default="wikiart",
        help="Search term used with --search-candidates.",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=PROGRESS_INTERVAL_SECONDS,
        help="Seconds between progress updates; use 0 to disable.",
    )
    parser.add_argument(
        "--total-bytes",
        type=int,
        help=(
            "Optional expected compressed download size for progress ETA. "
            "Normally this is auto-detected from Kaggle metadata."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    try:
        username = args.username or input("Kaggle username: ").strip()
        api_key = args.api_key or getpass.getpass("Kaggle API key: ").strip()
    except KeyboardInterrupt:
        print("\nDownload cancelled before Kaggle authentication.", file=sys.stderr)
        return 130

    try:
        if args.search_candidates:
            api = init_kaggle(username, api_key)
            candidates = list_kaggle_dataset_candidates(
                api,
                search_term=args.search_term,
            )
            print("Accessible Kaggle dataset candidates:")
            for candidate in candidates:
                print(candidate)
            return 0 if candidates else 3

        if args.check_access:
            api = init_kaggle(username, api_key)
            if args.source == "competition":
                check_kaggle_competition_access(api, args.competition)
                print(f"Competition access OK: {args.competition}")
            else:
                check_kaggle_dataset_access(api, args.dataset)
                print(f"Dataset access OK: {args.dataset}")
            return 0

        if args.source == "competition":
            report = download_kaggle_competition(
                username=username,
                api_key=api_key,
                competition=args.competition,
                path=args.path,
                progress_interval=args.progress_interval,
                expected_size_bytes=args.total_bytes,
            )
            print(f"Status: {'SUCCESS' if report.success else 'FAILED'}")
            print(f"Dataset: {report.dataset}")
            print(f"Dataset size: {report.dataset_size_bytes} bytes")
            print(f"Unzip path: {report.output_path}")
            print(f"Images: {report.image_count}")
            print(f"Classes: {report.class_count}")
            print(f"Loaded records: {report.loaded_record_count}")
            print(f"Message: {report.message}")
            return 0 if report.success else 2

        report = download_wikiart_dataset(
            username=username,
            api_key=api_key,
            dataset=args.dataset,
            path=args.path,
            progress_interval=args.progress_interval,
            expected_size_bytes=args.total_bytes,
        )
    except KaggleDatasetAccessError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        try:
            api = init_kaggle(username, api_key)
            candidates = list_kaggle_dataset_candidates(api, search_term="wikiart")
        except KaggleDownloaderError:
            candidates = []
        if candidates:
            print("\nAccessible WikiArt candidates for this account:", file=sys.stderr)
            for candidate in candidates:
                print(candidate, file=sys.stderr)
            print(
                "\nRetry with: python -m experiments.kaggle_downloader "
                "--username <user> --dataset <owner/slug>",
                file=sys.stderr,
            )
        return 1
    except KaggleDownloaderError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"Status: {'SUCCESS' if report.success else 'FAILED'}")
    print(f"Dataset: {report.dataset}")
    print(f"Dataset size: {report.dataset_size_bytes} bytes")
    print(f"Unzip path: {report.output_path}")
    print(f"Images: {report.image_count}")
    print(f"Classes: {report.class_count}")
    print(f"Loaded records: {report.loaded_record_count}")
    print(f"Message: {report.message}")
    return 0 if report.success else 2


def download_kaggle_competition(
    username: str,
    api_key: str,
    *,
    competition: str = DEFAULT_COMPETITION,
    path: str | Path = DEFAULT_COMPETITION_PATH,
    api: _KaggleApiProtocol | None = None,
    progress_interval: float = PROGRESS_INTERVAL_SECONDS,
    expected_size_bytes: int | None = None,
) -> KaggleDownloadReport:
    """Download and unzip a Kaggle competition archive using Python API."""
    started_at = _utc_now()
    output_path = Path(path).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    LOGGER.info(
        "Starting Kaggle competition download: competition=%s path=%s",
        competition,
        output_path,
    )
    try:
        kaggle_api = api if api is not None else init_kaggle(username, api_key)
        expected_bytes = expected_size_bytes or _estimate_competition_download_size(
            kaggle_api,
            competition,
        )
        try:
            with _download_progress_monitor(
                output_path,
                label=f"Kaggle competition {competition}",
                expected_bytes=expected_bytes,
                interval_seconds=progress_interval,
            ):
                kaggle_api.competition_download_files(
                    competition=competition,
                    path=str(output_path),
                )
        except Exception as exc:
            raise _normalize_kaggle_exception(exc, context="download") from exc
        _extract_zip_archives(output_path)
        verification = _verify_dataset_report(output_path)
        report = KaggleDownloadReport(
            dataset=f"competition:{competition}",
            output_path=output_path,
            status=verification.status,
            success=verification.status != DatasetVerificationStatus.EMPTY,
            dataset_size_bytes=verification.byte_count,
            image_count=verification.image_count,
            class_count=verification.class_count,
            loaded_record_count=0,
            statistics=None,
            started_at_utc=started_at,
            finished_at_utc=_utc_now(),
            message=verification.message,
        )
        _log_report(report)
        return report
    except KaggleDownloaderError:
        LOGGER.error("Download status: FAILED")
        raise
    except Exception as exc:
        normalized = _normalize_kaggle_exception(exc, context="download")
        LOGGER.error("Download status: FAILED")
        LOGGER.error("Kaggle competition acquisition failed: %s", normalized)
        raise normalized from exc


def check_kaggle_dataset_access(
    api: _KaggleApiProtocol,
    dataset: str = DEFAULT_DATASET,
) -> None:
    """Raise a precise error if the authenticated account cannot access data."""
    _reject_placeholder_ref(dataset)
    try:
        api.dataset_list_files(dataset)
    except Exception as exc:
        raise _normalize_kaggle_exception(exc, context="download") from exc


def check_kaggle_competition_access(
    api: _KaggleApiProtocol,
    competition: str = DEFAULT_COMPETITION,
) -> None:
    """Raise a precise error if the account cannot access a competition."""
    try:
        api.competition_list_files(competition)
    except Exception as exc:
        raise _normalize_kaggle_exception(exc, context="download") from exc


def list_kaggle_dataset_candidates(
    api: _KaggleApiProtocol,
    *,
    search_term: str = "wikiart",
    limit: int = 10,
) -> list[str]:
    """Return readable Kaggle dataset refs matching a search term."""
    try:
        datasets = api.dataset_list(search=search_term, page=1)
    except Exception as exc:
        raise _normalize_kaggle_exception(exc, context="auth") from exc
    if datasets is None:
        return []

    refs: list[str] = []
    for dataset in list(datasets)[:limit]:
        ref = _dataset_field(dataset, "ref")
        title = _dataset_field(dataset, "title")
        size = _dataset_field(dataset, "totalBytes", "total_bytes")
        if ref is None:
            continue
        label = str(ref)
        if title:
            label += f" | {title}"
        if size is not None:
            label += f" | {size} bytes"
        refs.append(label)
    return refs


@contextmanager
def _download_progress_monitor(
    output_path: Path,
    *,
    label: str,
    expected_bytes: int | None,
    interval_seconds: float,
) -> Any:
    """Log live download speed and ETA while the Kaggle API writes files."""
    if interval_seconds <= 0:
        yield
        return

    stop_event = threading.Event()
    monitor = threading.Thread(
        target=_monitor_download_progress,
        kwargs={
            "output_path": output_path,
            "label": label,
            "expected_bytes": expected_bytes,
            "interval_seconds": interval_seconds,
            "stop_event": stop_event,
        },
        daemon=True,
    )
    monitor.start()
    try:
        yield
    finally:
        stop_event.set()
        monitor.join(timeout=min(interval_seconds, 2.0))
        current_bytes = _directory_size(output_path)
        LOGGER.info(
            "Progress: %s final observed size=%s",
            label,
            _format_bytes(current_bytes),
        )


def _monitor_download_progress(
    *,
    output_path: Path,
    label: str,
    expected_bytes: int | None,
    interval_seconds: float,
    stop_event: threading.Event,
) -> None:
    """Background worker that calculates speed from byte deltas."""
    started_at = time.monotonic()
    previous_at = started_at
    previous_bytes = _directory_size(output_path)

    LOGGER.info(
        "Progress monitor started: %s total=%s interval=%.1fs",
        label,
        _format_total_bytes(expected_bytes),
        interval_seconds,
    )

    while not stop_event.wait(interval_seconds):
        now = time.monotonic()
        current_bytes = _directory_size(output_path)
        interval_elapsed = max(now - previous_at, 0.001)
        total_elapsed = max(now - started_at, 0.001)
        instant_bps = max(current_bytes - previous_bytes, 0) / interval_elapsed
        average_bps = current_bytes / total_elapsed
        LOGGER.info(
            "Progress: %s downloaded=%s/%s percent=%s speed=%s avg=%s eta=%s",
            label,
            _format_bytes(current_bytes),
            _format_total_bytes(expected_bytes),
            _format_percent(current_bytes, expected_bytes),
            _format_rate(instant_bps),
            _format_rate(average_bps),
            _format_eta(current_bytes, expected_bytes, average_bps),
        )
        previous_at = now
        previous_bytes = current_bytes


def _estimate_dataset_download_size(
    api: _KaggleApiProtocol,
    dataset: str,
) -> int | None:
    """Best-effort compressed size estimate from Kaggle metadata."""
    try:
        files = api.dataset_list_files(dataset)
    except Exception as exc:
        LOGGER.warning(
            "Could not estimate dataset download size before download: %s",
            exc,
        )
        return None
    return _sum_kaggle_file_sizes(files)


def _estimate_competition_download_size(
    api: _KaggleApiProtocol,
    competition: str,
) -> int | None:
    """Best-effort compressed size estimate from competition file metadata."""
    try:
        files = api.competition_list_files(competition)
    except Exception as exc:
        LOGGER.warning(
            "Could not estimate competition download size before download: %s",
            exc,
        )
        return None
    return _sum_kaggle_file_sizes(files)


def _sum_kaggle_file_sizes(files: object) -> int | None:
    """Sum common Kaggle SDK file-size fields from list-like metadata."""
    if files is None:
        return None
    items = getattr(files, "files", files)
    if items is None:
        return None
    total = 0
    matched = False
    for item in list(items):
        value = _dataset_field(
            item,
            "totalBytes",
            "total_bytes",
            "size",
            "fileSize",
            "file_size",
        )
        if value is None:
            continue
        if not isinstance(value, int):
            continue
        total += value
        matched = True
    return total if matched else None


def _format_bytes(value: int) -> str:
    """Return a human-readable byte count."""
    units = ("B", "KB", "MB", "GB", "TB")
    amount = float(max(value, 0))
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.2f} {unit}"
        amount /= 1024
    return f"{amount:.2f} TB"


def _format_total_bytes(value: int | None) -> str:
    """Return a readable total size, preserving unknown totals."""
    return _format_bytes(value) if value else "unknown"


def _format_percent(current_bytes: int, expected_bytes: int | None) -> str:
    """Return a percent string when the total is known."""
    if not expected_bytes:
        return "unknown"
    return f"{min(current_bytes / expected_bytes * 100, 100):.2f}%"


def _format_rate(bytes_per_second: float) -> str:
    """Return a human-readable transfer rate."""
    return f"{_format_bytes(int(bytes_per_second))}/s"


def _format_eta(
    current_bytes: int,
    expected_bytes: int | None,
    bytes_per_second: float,
) -> str:
    """Return an estimated remaining time."""
    if not expected_bytes or bytes_per_second <= 0:
        return "unknown"
    remaining = max(expected_bytes - current_bytes, 0)
    return _format_duration(remaining / bytes_per_second)


def _format_duration(seconds: float) -> str:
    """Format seconds as compact h/m/s."""
    total_seconds = max(int(seconds), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _reject_placeholder_ref(ref: str) -> None:
    """Reject documentation placeholders before making a Kaggle request."""
    if ref.strip().lower() in {"owner/slug", "<owner/slug>", "user/dataset"}:
        raise KaggleDownloaderError(
            "Replace owner/slug with a real Kaggle dataset ref from "
            "--search-candidates output, for example someone/wikiart-data.",
        )


def _dataset_field(dataset: object, *names: str) -> object | None:
    """Read a Kaggle SDK field from public or private attribute names."""
    for name in names:
        value = getattr(dataset, name, None)
        if value is not None:
            return value
        snake_name = _camel_to_snake(name)
        value = getattr(dataset, snake_name, None)
        if value is not None:
            return value
        value = getattr(dataset, f"_{snake_name}", None)
        if value is not None:
            return value
    owner = getattr(dataset, "_owner_slug", None)
    slug = getattr(dataset, "_dataset_slug", None)
    if "ref" in names and owner and slug:
        return f"{owner}/{slug}"
    return None


def _camel_to_snake(value: str) -> str:
    chars: list[str] = []
    for char in value:
        if char.isupper() and chars:
            chars.append("_")
        chars.append(char.lower())
    return "".join(chars)


def _prepare_kaggle_environment(username: str, api_key: str) -> None:
    """Configure process-local Kaggle credentials and isolate config lookup."""
    config_dir = Path("D:/PCA/data/cache/kaggle_config").resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["KAGGLE_CONFIG_DIR"] = str(config_dir)
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = api_key

    for token_key in (
        "KAGGLE_API_TOKEN",
        "KAGGLE_API_V1_TOKEN",
        "KAGGLE_IAP_TOKEN",
    ):
        os.environ.pop(token_key, None)


def _verify_kaggle_credentials(api: _KaggleApiProtocol) -> None:
    """Probe an authenticated endpoint before dataset-specific downloads."""
    try:
        api.dataset_list(search="wikiart", page=1)
    except Exception as exc:
        raise _normalize_kaggle_exception(exc, context="auth") from exc


@contextmanager
def _isolated_kaggle_home() -> Any:
    """Temporarily hide user-level kaggle.json and access_token files."""
    isolated_home = Path("D:/PCA/data/cache/kaggle_home").resolve()
    isolated_home.mkdir(parents=True, exist_ok=True)
    saved_home = os.environ.get("HOME")
    saved_userprofile = os.environ.get("USERPROFILE")
    os.environ["HOME"] = str(isolated_home)
    os.environ["USERPROFILE"] = str(isolated_home)
    try:
        yield
    finally:
        if saved_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = saved_home
        if saved_userprofile is None:
            os.environ.pop("USERPROFILE", None)
        else:
            os.environ["USERPROFILE"] = saved_userprofile


def _verify_dataset_report(path: Path) -> DatasetVerificationReport:
    """Inspect downloaded data for images, class folders, and sample count."""
    root = path.expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return DatasetVerificationReport(
            status=DatasetVerificationStatus.EMPTY,
            path=root,
            image_count=0,
            class_count=0,
            byte_count=0,
            message=f"Dataset path does not exist: {root}",
        )

    dataset_root = _select_dataset_root(root)
    image_paths = [
        candidate
        for candidate in dataset_root.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    ]
    byte_count = _directory_size(root)
    if not image_paths:
        return DatasetVerificationReport(
            status=DatasetVerificationStatus.EMPTY,
            path=dataset_root,
            image_count=0,
            class_count=0,
            byte_count=byte_count,
            message="Downloaded dataset contains no image files.",
        )

    class_counts = _count_images_by_first_level(dataset_root)
    if len(class_counts) < 2:
        return DatasetVerificationReport(
            status=DatasetVerificationStatus.CORRUPTED,
            path=dataset_root,
            image_count=len(image_paths),
            class_count=len(class_counts),
            byte_count=byte_count,
            message="Downloaded images are not organized into category folders.",
        )
    if len(image_paths) <= MIN_SAMPLE_COUNT:
        return DatasetVerificationReport(
            status=DatasetVerificationStatus.CORRUPTED,
            path=dataset_root,
            image_count=len(image_paths),
            class_count=len(class_counts),
            byte_count=byte_count,
            message=(
                "Downloaded dataset is below the minimum reproducibility "
                f"threshold: {len(image_paths)} <= {MIN_SAMPLE_COUNT} images."
            ),
        )

    return DatasetVerificationReport(
        status=DatasetVerificationStatus.VALID,
        path=dataset_root,
        image_count=len(image_paths),
        class_count=len(class_counts),
        byte_count=byte_count,
        message="Dataset satisfies image, category, and sample-count checks.",
    )


def _run_artinsight_pipeline(
    dataset_root: Path,
) -> tuple[list[Any], DatasetStatistics]:
    """Trigger the ArtInsight loader and statistics pipeline."""
    loader = ArtDatasetLoader(ArtDatasetLoaderConfig(dataset_root=dataset_root))
    records = loader.load()
    stats = DatasetStats(dataset_root).generate()
    LOGGER.info(
        "ArtDatasetLoader.load completed: records=%s dataset_root=%s",
        len(records),
        dataset_root,
    )
    LOGGER.info(
        "DatasetStats.generate completed: total_images=%s corrupted=%s",
        stats.total_images,
        stats.corrupted_count,
    )
    return records, stats


def _select_dataset_root(root: Path) -> Path:
    """Return the folder that actually contains class/style directories."""
    if len(_count_images_by_first_level(root)) >= 2:
        return root
    child_dirs = [child for child in root.iterdir() if child.is_dir()]
    if len(child_dirs) != 1:
        return root
    child = child_dirs[0]
    if _count_images_by_first_level(child):
        return child
    return root


def _count_images_by_first_level(root: Path) -> dict[str, int]:
    """Count supported images grouped by first-level directory."""
    counts: dict[str, int] = {}
    for image_path in root.rglob("*"):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        relative_parts = image_path.relative_to(root).parts
        if len(relative_parts) < 2:
            continue
        label = relative_parts[0]
        counts[label] = counts.get(label, 0) + 1
    return counts


def _normalize_kaggle_exception(
    exc: Exception,
    *,
    context: str = "download",
) -> KaggleDownloaderError:
    """Convert Kaggle/HTTP exceptions into actionable downloader errors."""
    status_code = _status_code(exc)
    message = str(exc)
    is_auth_marker = _contains_any(message, ("authentication", "unauthorized"))
    is_forbidden = status_code == 403 or _contains_any(message, ("403", "forbidden"))
    if status_code == 401 or is_auth_marker:
        return KaggleAuthenticationError(AUTH_FAILURE_MESSAGE)
    if is_forbidden:
        if context == "auth":
            return KaggleAuthenticationError(AUTH_FAILURE_MESSAGE)
        return KaggleDatasetAccessError(DATASET_ACCESS_FAILURE_MESSAGE)
    if status_code == 404 or _contains_any(
        message,
        ("datasetnotfound", "dataset not found", "not found", "404"),
    ):
        return DatasetNotFoundError(f"DatasetNotFound: {message}")
    if isinstance(exc, (HTTPError, RequestException)):
        return KaggleDownloaderError(f"Kaggle download failed: {message}")
    return KaggleDownloaderError(f"Kaggle dataset acquisition failed: {message}")


def _status_code(exc: Exception) -> int | None:
    """Return an HTTP status code from common exception shapes."""
    response = getattr(exc, "response", None)
    if response is not None:
        code = getattr(response, "status_code", None)
        if isinstance(code, int):
            return code
    status = getattr(exc, "status", None)
    if isinstance(status, int):
        return status
    return None


def _contains_any(value: str, needles: tuple[str, ...]) -> bool:
    """Return whether a string contains any case-insensitive marker."""
    lowered = value.lower()
    return any(needle in lowered for needle in needles)


def _directory_size(root: Path) -> int:
    """Return total byte size of files under a directory."""
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _extract_zip_archives(root: Path) -> None:
    """Extract all zip archives under a download directory."""
    for archive_path in root.glob("*.zip"):
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(root)


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _log_report(report: KaggleDownloadReport) -> None:
    """Emit the required download audit log fields."""
    LOGGER.info("Download status: %s", "SUCCESS" if report.success else "FAILED")
    LOGGER.info("Dataset size: %s bytes", report.dataset_size_bytes)
    LOGGER.info("Unzip path: %s", report.output_path)
    LOGGER.info("Image count: %s", report.image_count)
    LOGGER.info("Class folder count: %s", report.class_count)


if __name__ == "__main__":
    raise SystemExit(main())
