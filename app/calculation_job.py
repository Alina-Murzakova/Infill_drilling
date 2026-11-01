from __future__ import annotations

import shutil
import threading
import traceback
from collections import deque
from pathlib import Path
from typing import Any

from loguru import logger

from app.exceptions import CalculationCancelled
from app.main import run_model


class CalculationJob:
    """Thread-safe holder for one calculation lifecycle."""

    def __init__(self, parameters: dict[str, Any]):
        self._parameters = parameters
        self._lock = threading.RLock()
        self._cancel = threading.Event()
        self._status = "queued"
        self._progress = 0
        self._logs: deque[str] = deque(maxlen=500)
        self._error: str | None = None
        self._summary = None
        self._result_zip: bytes | None = None
        self._result_zip_name: str | None = None
        self._thread = threading.Thread(target=self._run, name="infill-calculation", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def cancel(self) -> None:
        self._cancel.set()
        self._append_log("Запрошена отмена расчёта. Ожидаем завершения текущего шага…")

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": self._status,
                "progress": self._progress,
                "logs": list(self._logs),
                "error": self._error,
                "cancel_requested": self._cancel.is_set(),
            }

    def result(self):
        with self._lock:
            return self._summary, self._result_zip, self._result_zip_name

    def _append_log(self, message: Any) -> None:
        text = str(message).rstrip()
        if not text:
            return
        with self._lock:
            self._logs.append(text)

    def _update_progress(self, value: int | float) -> None:
        with self._lock:
            self._progress = max(0, min(int(value), 100))

    def _run(self) -> None:
        with self._lock:
            self._status = "running"
        sink_id = logger.add(
            lambda message: self._append_log(message),
            format="{time:HH:mm:ss} | {level} | {message}",
            level="INFO",
        )
        try:
            summary, results_directory = run_model(
                self._parameters,
                total_stages=18,
                progress=self._update_progress,
                is_cancelled=self._cancel.is_set,
            )
            if self._cancel.is_set():
                raise CalculationCancelled()
            results_directory = Path(results_directory)
            archive_base = results_directory.parent / "Результаты_АВНС"
            archive = Path(shutil.make_archive(str(archive_base), "zip", root_dir=results_directory))
            with self._lock:
                self._summary = summary
                self._result_zip = archive.read_bytes()
                self._result_zip_name = archive.name
                self._progress = 100
                self._status = "completed"
        except CalculationCancelled:
            self._append_log("Расчёт отменён пользователем.")
            with self._lock:
                self._status = "cancelled"
        except Exception as error:
            self._append_log(traceback.format_exc())
            with self._lock:
                self._error = str(error)
                self._status = "failed"
        finally:
            logger.remove(sink_id)
