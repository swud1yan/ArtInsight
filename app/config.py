"""Configuration loading for the ArtInsight desktop application."""

from __future__ import annotations

from dataclasses import dataclass
import os
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "settings.json"
CONFIG_ENV_VAR = "ARTINSIGHT_CONFIG"


@dataclass(frozen=True)
class PathSettings:
    """Filesystem paths used by the application."""

    dataset_root: Path
    cache_dir: Path
    model_dir: Path
    export_dir: Path


@dataclass(frozen=True)
class DatabaseSettings:
    """Database connection settings."""

    url: str


@dataclass(frozen=True)
class LoggingSettings:
    """Logging output settings."""

    log_dir: Path
    level: str
    file_name: str


@dataclass(frozen=True)
class UISettings:
    """Global UI appearance settings."""

    font_family: str
    font_size: int
    theme_path: Path


@dataclass(frozen=True)
class Config:
    """Application configuration object."""

    paths: PathSettings
    database: DatabaseSettings
    logging: LoggingSettings
    ui: UISettings


class ConfigManager:
    """Load ArtInsight settings from a JSON configuration file."""

    def __init__(self, config_path: Path | None = None) -> None:
        """Create a config manager for the given configuration path."""
        configured_path = config_path or Path(
            os.environ.get(CONFIG_ENV_VAR, str(DEFAULT_CONFIG_PATH)),
        )
        if not configured_path.is_absolute():
            configured_path = PROJECT_ROOT / configured_path
        self._config_path = configured_path

    def load(self) -> Config:
        """Load configuration and convert paths to absolute Path objects."""
        raw_config = self._read_json(self._config_path)
        return self._parse_config(raw_config)

    def _read_json(self, config_path: Path) -> dict[str, Any]:
        """Read raw JSON configuration data."""
        with config_path.open("r", encoding="utf-8-sig") as config_file:
            data = json.load(config_file)
        if not isinstance(data, dict):
            msg = "Configuration root must be a JSON object."
            raise ValueError(msg)
        return data

    def _parse_config(self, raw_config: dict[str, Any]) -> Config:
        """Parse raw configuration values into dataclasses."""
        paths = raw_config.get("paths", {})
        database = raw_config.get("database", {})
        logging_config = raw_config.get("logging", {})
        ui = raw_config.get("ui", {})

        return Config(
            paths=PathSettings(
                dataset_root=self._resolve_path(paths.get("dataset_root")),
                cache_dir=self._resolve_path(paths.get("cache_dir")),
                model_dir=self._resolve_path(paths.get("model_dir")),
                export_dir=self._resolve_path(paths.get("export_dir")),
            ),
            database=DatabaseSettings(
                url=str(database.get("url", "sqlite:///data/artinsight.sqlite3")),
            ),
            logging=LoggingSettings(
                log_dir=self._resolve_path(logging_config.get("log_dir")),
                level=str(logging_config.get("level", "INFO")),
                file_name=str(logging_config.get("file_name", "artinsight.log")),
            ),
            ui=UISettings(
                font_family=str(ui.get("font_family", "Microsoft YaHei UI")),
                font_size=int(ui.get("font_size", 11)),
                theme_path=self._resolve_path(ui.get("theme_path")),
            ),
        )

    def _resolve_path(self, value: object) -> Path:
        """Resolve a configured path relative to the project root."""
        if value is None:
            return PROJECT_ROOT
        candidate = Path(str(value))
        if candidate.is_absolute():
            return candidate
        return PROJECT_ROOT / candidate
