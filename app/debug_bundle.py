from __future__ import annotations

import uuid
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def debug_directory(work_dir: Path) -> Path:
    path = work_dir / "results" / ".debug"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_error_log(work_dir: Path, error: BaseException, traceback_text: str) -> None:
    """Write the same compact error artifact used by the desktop calculation."""
    path = debug_directory(work_dir) / "error.log"
    content = (
        f"Время ошибки (UTC): {datetime.now(timezone.utc).isoformat()}\n"
        f"Тип ошибки: {type(error).__name__}\n"
        f"Сообщение: {error}\n\n"
        f"Трассировка:\n{traceback_text}"
    )
    path.write_text(content, encoding="utf-8")


def create_debug_bundle(*, work_dir: Path) -> tuple[bytes, str]:
    """Archive only the desktop-style ``.debug`` artifacts produced so far."""
    source = debug_directory(work_dir)
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED, allowZip64=True) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source).as_posix())

    created_at = datetime.now(timezone.utc)
    diagnostic_id = uuid.uuid4().hex[:8]
    filename = f"Диагностика_АВНС_{created_at:%Y%m%d_%H%M%SZ}_{diagnostic_id}.zip"
    return buffer.getvalue(), filename
