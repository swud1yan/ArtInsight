"""Background task helpers for PySide6 pages."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QObject, QThread, Signal, Slot


class FunctionWorker(QObject):
    """Run one callable on a QThread and emit structured results."""

    succeeded = Signal(object)
    failed = Signal(object)
    finished = Signal()

    def __init__(
        self,
        function: Callable[..., object],
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        """Create a worker for a callable and its arguments."""
        super().__init__()
        self._function = function
        self._args = args
        self._kwargs = kwargs

    @Slot()
    def run(self) -> None:
        """Execute the callable and emit success or failure."""
        try:
            result = self._function(*self._args, **self._kwargs)
            self.succeeded.emit(result)
        except Exception as exc:
            self.failed.emit(exc)
        finally:
            self.finished.emit()


class TaskCallbacks(QObject):
    """Main-thread callback bridge for worker results."""

    def __init__(
        self,
        on_success: Callable[[object], None],
        on_failure: Callable[[object], None],
        on_finished: Callable[[], None],
        parent: QObject,
    ) -> None:
        """Create callback slots owned by the GUI thread."""
        super().__init__(parent)
        self._on_success = on_success
        self._on_failure = on_failure
        self._on_finished = on_finished

    @Slot(object)
    def handle_success(self, result: object) -> None:
        """Handle a successful worker result in the GUI thread."""
        self._on_success(result)

    @Slot(object)
    def handle_failure(self, error: object) -> None:
        """Handle a failed worker result in the GUI thread."""
        self._on_failure(error)

    @Slot()
    def handle_finished(self) -> None:
        """Handle worker completion in the GUI thread."""
        self._on_finished()


def run_in_thread(
    parent: QObject,
    function: Callable[..., object],
    on_success: Callable[[object], None],
    on_failure: Callable[[object], None],
    on_finished: Callable[[], None],
    *args: object,
    **kwargs: object,
) -> QThread:
    """Start a function worker and return the owning thread."""
    thread = QThread(parent)
    worker = FunctionWorker(function, args, kwargs)
    callbacks = TaskCallbacks(on_success, on_failure, on_finished, parent)
    setattr(thread, "_artinsight_worker", worker)
    setattr(thread, "_artinsight_callbacks", callbacks)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.succeeded.connect(callbacks.handle_success)
    worker.failed.connect(callbacks.handle_failure)
    worker.finished.connect(callbacks.handle_finished)
    worker.finished.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    thread.finished.connect(callbacks.deleteLater)
    thread.finished.connect(thread.deleteLater)
    thread.start()
    return thread
