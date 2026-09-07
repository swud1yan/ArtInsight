"""Editable configuration adapter for GUI settings."""

from __future__ import annotations

import json
from pathlib import Path

from app.config import Config, DEFAULT_CONFIG_PATH


class SettingsAdapter:
    """Persist settings-page values to the default JSON configuration."""

    def __init__(self, config: Config) -> None:
        """Create the settings adapter."""
        self._config = config

    def save_config_values(self, values: dict[str, str]) -> None:
        """Save editable configuration values to the default JSON config file."""
        raw = {
            "paths": {
                "dataset_root": values["dataset_root"],
                "cache_dir": values["cache_dir"],
                "model_dir": values["model_dir"],
                "export_dir": values["export_dir"],
            },
            "database": {"url": values["database_url"]},
            "logging": {
                "log_dir": self._path_for_config(self._config.logging.log_dir),
                "level": self._config.logging.level,
                "file_name": self._config.logging.file_name,
            },
            "ui": {
                "font_family": self._config.ui.font_family,
                "font_size": self._config.ui.font_size,
                "theme_path": self._path_for_config(self._config.ui.theme_path),
            },
        }
        DEFAULT_CONFIG_PATH.write_text(
            json.dumps(raw, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _path_for_config(self, path: Path) -> str:
        """Return a stable config path string, relative when possible."""
        try:
            return str(path.relative_to(self._project_root())).replace("\\", "/")
        except ValueError:
            return str(path)

    def _project_root(self) -> Path:
        """Return the project root derived from the config file location."""
        return DEFAULT_CONFIG_PATH.parents[1]
