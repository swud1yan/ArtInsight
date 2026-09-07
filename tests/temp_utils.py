"""Temporary directory helpers for tests.

The Windows sandbox used by the desktop runner rejects child writes inside
directories created with :class:`tempfile.TemporaryDirectory`.  These helpers
create ordinary project-local folders so tests exercise application behavior
without depending on host-specific temporary directory ACLs.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4


class ProjectTempDirectory:
    """Create a writable temporary directory under the project test workspace."""

    def __init__(self, prefix: str = "tmp") -> None:
        """Create the directory immediately and expose its path via ``name``."""
        root = Path(__file__).resolve().parents[1] / ".test_tmp"
        root.mkdir(exist_ok=True)
        self.path = root / f"{prefix}_{uuid4().hex}"
        self.path.mkdir()
        self.name = str(self.path)

    def cleanup(self) -> None:
        """Remove the directory and all files created by the test."""
        shutil.rmtree(self.path, ignore_errors=True)

    def __enter__(self) -> str:
        """Return the temporary directory path as a string."""
        return self.name

    def __exit__(self, *exc_info: object) -> None:
        """Clean up the temporary directory when leaving a context manager."""
        self.cleanup()
