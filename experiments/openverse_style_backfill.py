"""Backfill missing art styles from Openverse public images."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import mimetypes
import random
import re
import shutil
import time
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "raw" / "public_style_backfill"
DEFAULT_MANIFEST = DEFAULT_OUTPUT_ROOT / "openverse_manifest.csv"
DEFAULT_REPORT = DEFAULT_OUTPUT_ROOT / "openverse_report.txt"
DEFAULT_SEED = 42
DEFAULT_MAX_IMAGES_PER_STYLE = 40
DEFAULT_PAGE_SIZE = 20
DEFAULT_MAX_PAGES_PER_QUERY = 3
DEFAULT_DOWNLOAD_DELAY_SECONDS = 0.2
DEFAULT_MAX_RETRIES = 5
OPENVERSE_API = "https://api.openverse.org/v1/images/"
USER_AGENT = "ArtInsight/1.0 (public style backfill; contact: local)"


STYLE_QUERY_HINTS: dict[str, list[str]] = {
    "Proto Renaissance": [
        "Proto-Renaissance painting",
        "Proto Renaissance art",
        "early Italian painting",
    ],
    "Orientalism": [
        "Orientalist painting",
        "Orientalism art",
        "Orientalist art",
    ],
    "Divisionism": [
        "Divisionism painting",
        "Divisionist art",
        "Pointillism painting",
    ],
    "Tonalism": [
        "Tonalism painting",
        "tonalist art",
    ],
    "Naturalism": [
        "Naturalist painting",
        "Naturalism art",
    ],
    "Social Realism": [
        "Social realism painting",
        "social realist art",
    ],
    "Pre-Raphaelite": [
        "Pre-Raphaelite painting",
        "Pre-Raphaelite Brotherhood",
    ],
    "Futurism": [
        "Futurism painting",
        "Futurist art",
    ],
    "Constructivism": [
        "Constructivism art",
        "Russian constructivism",
    ],
    "Suprematism": [
        "Suprematism art",
        "Suprematist painting",
    ],
    "Dada": [
        "Dada art",
        "Dadaist art",
    ],
    "Surrealism": [
        "Surrealism painting",
        "Surrealist art",
    ],
    "Op Art": [
        "Op art",
        "optical art",
    ],
    "Art Deco": [
        "Art Deco painting",
        "Art Deco illustration",
    ],
    "Abstract Art": [
        "abstract art",
        "abstraction painting",
    ],
    "Lyrical Abstraction": [
        "Lyrical abstraction",
        "lyrical abstract art",
    ],
    "Hard Edge Painting": [
        "hard edge painting",
        "hard-edge abstract art",
    ],
    "Tachisme": [
        "Tachisme",
        "art informel painting",
    ],
    "Art Informel": [
        "Art Informel",
        "informel painting",
    ],
    "Chinese Ink Painting": [
        "Chinese ink painting",
        "ink wash painting",
        "shui mo hua",
    ],
    "Photorealism": [
        "Photorealism",
        "photo-realism painting",
    ],
    "Action Painting": [
        "Action painting",
        "gestural abstraction",
    ],
    "Contemporary Realism": [
        "Contemporary realism painting",
        "contemporary realist art",
    ],
}


@dataclass(frozen=True)
class OpenverseImageCandidate:
    """One downloadable Openverse image candidate."""

    style: str
    query: str
    id: str
    title: str
    detail_url: str | None
    foreign_landing_url: str | None
    download_url: str
    source_url: str | None
    creator: str | None
    creator_url: str | None
    provider: str | None
    source: str | None
    license: str | None
    license_version: str | None
    license_url: str | None
    filesize: int | None
    filetype: str | None
    width: int | None
    height: int | None
    tags: str | None


@dataclass(frozen=True)
class OpenverseStyleReport:
    """Download summary for one style."""

    style: str
    query_count: int
    candidate_count: int
    downloaded_count: int
    downloaded_bytes: int
    output_dir: Path
    status: str


def backfill_public_styles(
    styles: Iterable[str],
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    max_images_per_style: int = DEFAULT_MAX_IMAGES_PER_STYLE,
    seed: int = DEFAULT_SEED,
    progress_interval: int = 10,
    download_delay_seconds: float = DEFAULT_DOWNLOAD_DELAY_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    max_pages_per_query: int = DEFAULT_MAX_PAGES_PER_QUERY,
    page_size: int = DEFAULT_PAGE_SIZE,
    overwrite: bool = False,
) -> list[OpenverseStyleReport]:
    """Download a small public backup set for each missing style."""
    output = output_root.expanduser().resolve()
    if overwrite and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    downloaded_global_keys: set[str] = set()
    reports: list[OpenverseStyleReport] = []
    manifest_rows: list[list[str]] = []

    for style in styles:
        style_dir = output / _safe_style_dir(style)
        style_dir.mkdir(parents=True, exist_ok=True)
        candidates: list[OpenverseImageCandidate] = []
        query_terms = _style_queries(style)

        for query in query_terms:
            for candidate in _search_openverse_candidates(
                style=style,
                query=query,
                limit=max_images_per_style * 4,
                max_pages=max_pages_per_query,
                page_size=page_size,
            ):
                dedupe_key = _candidate_dedupe_key(candidate)
                if dedupe_key in downloaded_global_keys:
                    continue
                if any(existing.id == candidate.id for existing in candidates):
                    continue
                candidates.append(candidate)
                if len(candidates) >= max_images_per_style:
                    break
            if len(candidates) >= max_images_per_style:
                break

        rng = random.Random(f"{seed}:{style}:openverse")
        rng.shuffle(candidates)
        selected = candidates[:max_images_per_style]

        downloaded_count = 0
        downloaded_bytes = 0
        for index, candidate in enumerate(selected, start=1):
            try:
                target_path = _download_candidate(
                    candidate,
                    style_dir,
                    download_delay_seconds=download_delay_seconds,
                    max_retries=max_retries,
                )
            except Exception as exc:  # pragma: no cover - network-dependent
                LOGGER.warning(
                    "Skipping %s (%s) after download failure: %s",
                    candidate.title,
                    candidate.download_url,
                    exc,
                )
                continue
            downloaded_count += 1
            size_bytes = target_path.stat().st_size
            downloaded_bytes += size_bytes
            downloaded_global_keys.add(dedupe_key)
            manifest_rows.append(
                [
                    style,
                    candidate.query,
                    candidate.id,
                    candidate.title,
                    candidate.detail_url or "",
                    candidate.foreign_landing_url or "",
                    candidate.download_url,
                    candidate.source_url or "",
                    candidate.provider or "",
                    candidate.source or "",
                    candidate.license or "",
                    candidate.license_version or "",
                    candidate.license_url or "",
                    candidate.creator or "",
                    candidate.creator_url or "",
                    candidate.tags or "",
                    str(target_path),
                    str(size_bytes),
                ]
            )
            if progress_interval > 0 and index % progress_interval == 0:
                print(
                    f"backfill {style}: {index}/{len(selected)} downloaded",
                    flush=True,
                )

        status = "OK" if downloaded_count > 0 else "EMPTY"
        reports.append(
            OpenverseStyleReport(
                style=style,
                query_count=len(query_terms),
                candidate_count=len(candidates),
                downloaded_count=downloaded_count,
                downloaded_bytes=downloaded_bytes,
                output_dir=style_dir,
                status=status,
            )
        )
        print(
            f"{style}: candidates={len(candidates)} downloaded={downloaded_count} "
            f"bytes={downloaded_bytes} status={status}",
            flush=True,
        )

    _write_manifest(manifest_rows, output / "openverse_manifest.csv")
    _write_report(reports, output / "openverse_report.txt", output, seed)
    return reports


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Backfill art styles with Openverse public images.",
    )
    parser.add_argument("styles", nargs="+", help="Style names to download.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-images-per-style", type=int, default=DEFAULT_MAX_IMAGES_PER_STYLE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--progress-interval", type=int, default=10)
    parser.add_argument("--download-delay-seconds", type=float, default=DEFAULT_DOWNLOAD_DELAY_SECONDS)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--max-pages-per-query", type=int, default=DEFAULT_MAX_PAGES_PER_QUERY)
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Run the public-source backfill downloader."""
    args = parse_args()
    reports = backfill_public_styles(
        args.styles,
        output_root=args.output_root,
        max_images_per_style=args.max_images_per_style,
        seed=args.seed,
        progress_interval=args.progress_interval,
        download_delay_seconds=args.download_delay_seconds,
        max_retries=args.max_retries,
        max_pages_per_query=args.max_pages_per_query,
        page_size=args.page_size,
        overwrite=args.overwrite,
    )
    total = sum(report.downloaded_count for report in reports)
    print(f"Styles requested: {len(reports)}")
    print(f"Downloaded: {total}")
    print(f"Output root: {args.output_root.expanduser().resolve()}")
    return 0 if total > 0 else 2


def _style_queries(style: str) -> list[str]:
    queries = [style, style.replace("-", " "), style.replace(" ", "_")]
    queries.extend(STYLE_QUERY_HINTS.get(style, []))
    queries.extend([f"{style} painting", f"{style} art"])
    normalized: list[str] = []
    seen: set[str] = set()
    for query in queries:
        query = " ".join(query.split())
        if not query or query.lower() in seen:
            continue
        seen.add(query.lower())
        normalized.append(query)
    return normalized


def _search_openverse_candidates(
    *,
    style: str,
    query: str,
    limit: int,
    max_pages: int,
    page_size: int,
) -> list[OpenverseImageCandidate]:
    candidates: list[OpenverseImageCandidate] = []
    seen_ids: set[str] = set()
    for page in range(1, max_pages + 1):
        payload = _openverse_api_request(
            {
                "q": query,
                "page_size": page_size,
                "page": page,
                "mature": "false",
            }
        )
        results = payload.get("results", [])
        if not isinstance(results, list):
            break
        for result in results:
            if not isinstance(result, dict):
                continue
            candidate = _result_to_candidate(style, query, result)
            if candidate is None or candidate.id in seen_ids:
                continue
            seen_ids.add(candidate.id)
            candidates.append(candidate)
            if len(candidates) >= limit:
                return candidates
        if not results:
            break
    return candidates


def _result_to_candidate(
    style: str,
    query: str,
    result: dict[str, object],
) -> OpenverseImageCandidate | None:
    candidate_id = result.get("id")
    title = result.get("title")
    image_url = result.get("url")
    if not isinstance(candidate_id, str) or not isinstance(title, str) or not isinstance(image_url, str):
        return None

    thumbnail = result.get("thumbnail")
    download_url = _extract_string(thumbnail) or image_url
    return OpenverseImageCandidate(
        style=style,
        query=query,
        id=candidate_id,
        title=title,
        detail_url=_extract_string(result.get("detail_url")),
        foreign_landing_url=_extract_string(result.get("foreign_landing_url")),
        download_url=download_url,
        source_url=image_url,
        creator=_extract_string(result.get("creator")),
        creator_url=_extract_string(result.get("creator_url")),
        provider=_extract_string(result.get("provider")),
        source=_extract_string(result.get("source")),
        license=_extract_string(result.get("license")),
        license_version=_extract_string(result.get("license_version")),
        license_url=_extract_string(result.get("license_url")),
        filesize=_extract_int(result.get("filesize")),
        filetype=_extract_string(result.get("filetype")),
        width=_extract_int(result.get("width")),
        height=_extract_int(result.get("height")),
        tags=_tags_to_string(result.get("tags")),
    )


def _download_candidate(
    candidate: OpenverseImageCandidate,
    style_dir: Path,
    *,
    download_delay_seconds: float,
    max_retries: int,
) -> Path:
    suffix = _suffix_from_candidate(candidate)
    safe_title = _slugify(candidate.title)
    target_path = style_dir / f"openverse_{candidate.id}_{safe_title}{suffix}"
    if target_path.exists() and target_path.stat().st_size > 0:
        return target_path
    payload = _download_with_retry(
        candidate.download_url,
        timeout=60,
        max_retries=max_retries,
    )
    target_path.write_bytes(payload)
    if download_delay_seconds > 0:
        time.sleep(download_delay_seconds)
    return target_path


def _openverse_api_request(params: dict[str, object]) -> dict[str, object]:
    url = f"{OPENVERSE_API}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _download_with_retry(
    url: str,
    *,
    timeout: int,
    max_retries: int,
) -> bytes:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=timeout) as response:
                return response.read()
        except HTTPError as exc:
            last_error = exc
            if exc.code not in {429, 500, 502, 503, 504}:
                raise
            wait_seconds = min(30.0, 1.5 * attempt)
            LOGGER.warning(
                "Transient HTTP error %s for %s (attempt %s/%s); retrying in %.1fs",
                exc.code,
                url,
                attempt,
                max_retries,
                wait_seconds,
            )
            time.sleep(wait_seconds)
        except URLError as exc:
            last_error = exc
            wait_seconds = min(30.0, 1.5 * attempt)
            LOGGER.warning(
                "Transient network error for %s (attempt %s/%s); retrying in %.1fs: %s",
                url,
                attempt,
                max_retries,
                wait_seconds,
                exc,
            )
            time.sleep(wait_seconds)
    assert last_error is not None
    raise last_error


def _suffix_from_candidate(candidate: OpenverseImageCandidate) -> str:
    if candidate.source_url:
        suffix = Path(urlparse(candidate.source_url).path).suffix.lower()
        if suffix:
            return suffix
    if candidate.filetype:
        normalized = candidate.filetype.lower().lstrip(".")
        if normalized in {"jpg", "jpeg"}:
            return ".jpg"
        if normalized == "png":
            return ".png"
        if normalized == "webp":
            return ".webp"
        if normalized == "gif":
            return ".gif"
    return ".jpg"


def _candidate_dedupe_key(candidate: OpenverseImageCandidate) -> str:
    return candidate.id.lower()


def _extract_string(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _extract_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    return None


def _tags_to_string(value: object) -> str | None:
    if not isinstance(value, list):
        return None
    names: list[str] = []
    for item in value:
        if isinstance(item, dict):
            name = item.get("name")
            if isinstance(name, str) and name.strip():
                names.append(name.strip())
        elif isinstance(item, str) and item.strip():
            names.append(item.strip())
    if not names:
        return None
    return ", ".join(dict.fromkeys(names))


def _safe_style_dir(style: str) -> str:
    return style.replace(" ", "_").replace("-", "_")


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = value.strip("_")
    return value or "image"


def _write_manifest(rows: list[list[str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "style",
                "query",
                "id",
                "title",
                "detail_url",
                "foreign_landing_url",
                "download_url",
                "source_url",
                "provider",
                "source",
                "license",
                "license_version",
                "license_url",
                "creator",
                "creator_url",
                "tags",
                "local_path",
                "size_bytes",
            ]
        )
        writer.writerows(rows)


def _write_report(
    reports: list[OpenverseStyleReport],
    output_path: Path,
    output_root: Path,
    seed: int,
) -> None:
    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    lines = [
        "Openverse backfill report",
        f"Sampling time: {now}",
        f"Random seed: {seed}",
        f"Output root: {output_root}",
        f"Styles requested: {len(reports)}",
        f"Downloaded: {sum(report.downloaded_count for report in reports)}",
        f"Bytes: {sum(report.downloaded_bytes for report in reports)}",
        "",
        "Per-style summary:",
    ]
    for report in reports:
        lines.append(
            f"- {report.style}: candidates={report.candidate_count}, "
            f"downloaded={report.downloaded_count}, bytes={report.downloaded_bytes}, "
            f"status={report.status}"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
