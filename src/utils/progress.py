"""Rich-based progress reporting for bootstrap validation."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


class ProgressReporter:
    """Rich progress reporter for bootstrap validation.

    Usage::

        reporter = ProgressReporter()
        with reporter.bootstrap_progress("LogisticRegression", n_iterations=1000):
            for b in range(1000):
                reporter.advance_iter()
    """

    def __init__(self) -> None:
        self._progress: Progress | None = None
        self._boot_task: TaskID | None = None

    @contextmanager
    def bootstrap_progress(
        self, model_name: str, n_iterations: int
    ) -> Generator[None, None, None]:
        """Context manager that renders a Rich progress bar for bootstrap iterations.

        Args:
            model_name: Label shown in the progress bar header.
            n_iterations: Total number of bootstrap iterations.
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            self._progress = progress
            self._boot_task = progress.add_task(
                f"Bootstrap [{model_name}]", total=n_iterations
            )
            try:
                yield
            finally:
                self._progress = None
                self._boot_task = None

    def set_phase(self, description: str) -> None:
        """Update the progress bar label without advancing the counter.

        Args:
            description: Text to display in the bar header.
        """
        if self._progress is not None and self._boot_task is not None:
            self._progress.update(self._boot_task, description=description)

    def advance_iter(self) -> None:
        """Advance the bootstrap iteration counter by one."""
        if self._progress is not None and self._boot_task is not None:
            self._progress.advance(self._boot_task)
