"""Backfill missing art styles from Wikimedia Commons public images."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
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

from experiments.dataset_compliance import SUPPORTED_IMAGE_SUFFIXES


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "raw" / "public_style_backfill"
DEFAULT_MANIFEST = DEFAULT_OUTPUT_ROOT / "commons_manifest.csv"
DEFAULT_REPORT = DEFAULT_OUTPUT_ROOT / "commons_report.txt"
DEFAULT_SEED = 42
DEFAULT_MAX_IMAGES_PER_STYLE = 40
DEFAULT_QUERY_LIMIT = 50
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "ArtInsight/1.0 (public style backfill; contact: local)"
DEFAULT_DOWNLOAD_DELAY_SECONDS = 2.0
DEFAULT_MAX_RETRIES = 5


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
class CommonsImageCandidate:
    """One downloadable Wikimedia Commons image candidate."""

    style: str
    query: str
    pageid: int
    title: str
    page_url: str
    image_url: str
    size_bytes: int
    mime: str
    license_short_name: str | None
    usage_terms: str | None
    categories: str | None


@dataclass(frozen=True)
class CommonsStyleReport:
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
    overwrite: bool = False,
) -> list[CommonsStyleReport]:
    """Download a small public backup set for each missing style."""
    output = output_root.expanduser().resolve()
    if overwrite and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    downloaded_global_keys: set[str] = set()
    reports: list[CommonsStyleReport] = []
    manifest_rows: list[list[str]] = []

    for style in styles:
        style_dir = output / _safe_style_dir(style)
        style_dir.mkdir(parents=True, exist_ok=True)
        candidates: list[CommonsImageCandidate] = []
        query_terms = _style_queries(style)

        for query in query_terms:
            for candidate in _search_commons_candidates(
                style=style,
                query=query,
                limit=max_images_per_style * 3,
            ):
                dedupe_key = _candidate_dedupe_key(candidate)
                if dedupe_key in downloaded_global_keys:
                    continue
                if any(existing.pageid == candidate.pageid for existing in candidates):
                    continue
                candidates.append(candidate)
                if len(candidates) >= max_images_per_style:
                    break
            if len(candidates) >= max_images_per_style:
                break

        rng = random.Random(f"{seed}:{style}:commons")
        rng.shuffle(candidates)
        selected = candidates[:max_images_per_style]

        downloaded_count = 0
        downloaded_bytes = 0
        for index, candidate in enumerate(selected, start=1):
            target_path = _download_candidate(
                candidate,
                style_dir,
                download_delay_seconds=download_delay_seconds,
                max_retries=max_retries,
            )
            downloaded_count += 1
            size_bytes = target_path.stat().st_size
            downloaded_bytes += size_bytes
            downloaded_global_keys.add(_candidate_dedupe_key(candidate))
            manifest_rows.append(
                [
                    style,
                    candidate.query,
                    str(candidate.pageid),
                    candidate.title,
                    candidate.page_url,
                    candidate.image_url,
                    candidate.license_short_name or "",
                    candidate.usage_terms or "",
                    candidate.categories or "",
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
            CommonsStyleReport(
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

    _write_manifest(manifest_rows, output / "commons_manifest.csv")
    _write_report(reports, output / "commons_report.txt", output, seed)
    return reports


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Backfill art styles with Wikimedia Commons public images.",
    )
    parser.add_argument("styles", nargs="+", help="Style names to download.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-images-per-style", type=int, default=DEFAULT_MAX_IMAGES_PER_STYLE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--progress-interval", type=int, default=10)
    parser.add_argument("--download-delay-seconds", type=float, default=DEFAULT_DOWNLOAD_DELAY_SECONDS)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
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


def _search_commons_candidates(
    *,
    style: str,
    query: str,
    limit: int,
) -> list[CommonsImageCandidate]:
    candidates: list[CommonsImageCandidate] = []
    offset = 0
    while len(candidates) < limit:
        payload = _commons_api_request(
            {
                "action": "query",
                "generator": "search",
                "gsrsearch": query,
                "gsrnamespace": 6,
                "gsrlimit": min(DEFAULT_QUERY_LIMIT, limit - len(candidates)),
                "gsroffset": offset,
                "prop": "imageinfo",
                "iiprop": "url|size|mime|extmetadata",
                "iiurlwidth": 768,
                "format": "json",
                "formatversion": 2,
            }
        )
        pages = payload.get("query", {}).get("pages", [])
        for page in pages:
            candidate = _page_to_candidate(style, query, page)
            if candidate is not None:
                candidates.append(candidate)
        continue_block = payload.get("continue") or {}
        if "gsroffset" not in continue_block:
            break
        offset = int(continue_block["gsroffset"])
    return candidates


def _page_to_candidate(
    style: str,
    query: str,
    page: dict[str, object],
) -> CommonsImageCandidate | None:
    pageid = page.get("pageid")
    title = page.get("title")
    imageinfo = page.get("imageinfo")
    if not isinstance(pageid, int) or not isinstance(title, str):
        return None
    if not isinstance(imageinfo, list) or not imageinfo:
        return None
    first = imageinfo[0]
    if not isinstance(first, dict):
        return None
    image_url = first.get("url")
    thumb_url = first.get("thumburl")
    size_bytes = first.get("size")
    mime = first.get("mime")
    if not isinstance(image_url, str) or not isinstance(size_bytes, int):
        return None
    if not isinstance(mime, str):
        mime = mimetypes.guess_type(image_url)[0] or "image/jpeg"
    extmetadata = first.get("extmetadata")
    if isinstance(extmetadata, dict):
        license_short_name = _extract_extmetadata(extmetadata, "LicenseShortName")
        usage_terms = _extract_extmetadata(extmetadata, "UsageTerms")
        categories = _extract_extmetadata(extmetadata, "Categories")
    else:
        license_short_name = None
        usage_terms = None
        categories = None
    return CommonsImageCandidate(
        style=style,
        query=query,
        pageid=pageid,
        title=title,
        page_url=_page_url(title),
        image_url=thumb_url if isinstance(thumb_url, str) else image_url,
        size_bytes=size_bytes,
        mime=mime,
        license_short_name=license_short_name,
        usage_terms=usage_terms,
        categories=categories,
    )


def _download_candidate(
    candidate: CommonsImageCandidate,
    style_dir: Path,
    *,
    download_delay_seconds: float,
    max_retries: int,
) -> Path:
    suffix = _suffix_from_url(candidate.image_url, candidate.mime)
    safe_title = _slugify(candidate.title.removeprefix("File:"))
    target_path = style_dir / f"commons_{candidate.pageid}_{safe_title}{suffix}"
    if target_path.exists() and target_path.stat().st_size > 0:
        return target_path
    payload = _download_with_retry(
        candidate.image_url,
        timeout=60,
        max_retries=max_retries,
    )
    target_path.write_bytes(payload)
    if download_delay_seconds > 0:
        time.sleep(download_delay_seconds)
    return target_path


def _commons_api_request(params: dict[str, object]) -> dict[str, object]:
    url = f"{COMMONS_API}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload)


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
            wait_seconds = min(30.0, 2.0 * attempt)
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
            wait_seconds = min(30.0, 2.0 * attempt)
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


def _extract_extmetadata(extmetadata: dict[str, object], key: str) -> str | None:
    raw = extmetadata.get(key)
    if not isinstance(raw, dict):
        return None
    value = raw.get("value")
    if not isinstance(value, str):
        return None
    cleaned = re.sub(r"<[^>]+>", " ", value)
    cleaned = " ".join(cleaned.split())
    return cleaned or None


def _candidate_dedupe_key(candidate: CommonsImageCandidate) -> str:
    return hashlib.sha1(candidate.image_url.lower().encode("utf-8")).hexdigest()


def _suffix_from_url(image_url: str, mime: str) -> str:
    path = urlparse(image_url).path
    suffix = Path(path).suffix
    if suffix and suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
        return suffix.lower()
    guessed = mimetypes.guess_extension(mime) or ".jpg"
    if guessed == ".jpe":
        return ".jpg"
    return guessed.lower()


def _page_url(title: str) -> str:
    quoted = title.replace(" ", "_")
    return f"https://commons.wikimedia.org/wiki/{quoted}"


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
                "pageid",
                "title",
                "page_url",
                "image_url",
                "license_short_name",
                "usage_terms",
                "categories",
                "local_path",
                "size_bytes",
            ]
        )
        writer.writerows(rows)


def _write_report(
    reports: list[CommonsStyleReport],
    output_path: Path,
    output_root: Path,
    seed: int,
) -> None:
    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    lines = [
        "Wikimedia Commons backfill report",
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
