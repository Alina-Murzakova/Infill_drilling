"""Developer-only data-source configuration.

The web interface intentionally does not expose these values.  Switch the
backend with ``INFILL_DATA_SOURCE=local|kspd`` or change ``DATA_SOURCE`` below.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Developer switch. Environment variables make deployments configurable while
# keeping credentials and infrastructure details out of the user interface.
DATA_SOURCE = os.getenv("INFILL_DATA_SOURCE", "local").strip().casefold()

# Runtime reads only the schema-compatible SQL database. CSV build inputs are
# intentionally configured by the separate local builder, not by the app.
LOCAL_DB_PATH = Path(
    os.getenv("INFILL_LOCAL_DB_PATH", PROJECT_ROOT / ".local_edw" / "edw.db")
)

KSPD = {
    "host": os.getenv("INFILL_KSPD_HOST", "spb99pkl-adbm01"),
    "port": int(os.getenv("INFILL_KSPD_PORT", "5432")),
    "database": os.getenv("INFILL_KSPD_DATABASE", "edw"),
    "user": os.getenv("INFILL_KSPD_USER", os.getenv("USER", "")),
    "password": os.getenv("INFILL_KSPD_PASSWORD", os.getenv("PASSWORD", "")),
}


def repository_config() -> dict[str, object]:
    if DATA_SOURCE == "local":
        return {
            "type": "local",
            "db_path": str(LOCAL_DB_PATH),
        }
    if DATA_SOURCE == "kspd":
        return {"type": "kspd", **KSPD}
    raise ValueError("INFILL_DATA_SOURCE должен быть 'local' или 'kspd'")
