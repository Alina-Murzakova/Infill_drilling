from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from app.config import columns_name

from .local_database import connect_local_database, database_is_current, repair_mojibake
from .models import (
    GridChoice,
    PVT_EDW_FACTORS,
    PVT_FIELDS,
    PVT_RELATIVE_FIELDS,
    classify_grid_type,
)
from .sql_queries import (
    FIELD_OBJECTS_SQL,
    FRACS_SQL,
    GRIDS_SQL,
    GRID_BY_KEY_SQL,
    HISTORY_SQL,
    PVT_SQL,
    SELECTED_KEY_SQL,
)


FRAC_COLUMNS = [
    "well_number",
    "date",
    "object",
    "permeability",
    "P_reservoir",
    "xfr",
    "w_f",
    "skin",
    "perf_upper",
    "perf_lower",
    "comment",
]


class DatabaseRepository(ABC):
    @abstractmethod
    def list_field_objects(self) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def get_history(self, field: str, object_name: str) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def get_fracs(self, field: str, object_name: str) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def get_pvt(self, field: str, object_name: str) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def list_grids(self, field: str, object_name: str) -> list[GridChoice]:
        raise NotImplementedError

    @abstractmethod
    def get_grid(self, locator: str) -> dict[str, Any]:
        raise NotImplementedError

    def test_connection(self) -> None:
        self.list_field_objects()


def _pvt_result(frame: pd.DataFrame) -> dict[str, Any]:
    frame = frame.fillna(0)
    values: dict[str, float] = {}
    sources: dict[str, str] = {}
    if not frame.empty:
        numeric_columns = [source for source, _default in PVT_FIELDS.values() if source in frame]
        score = frame[numeric_columns].apply(pd.to_numeric, errors="coerce").gt(0).sum(axis=1)
        row = frame.loc[score.idxmax()]
    else:
        row = None

    for target, (source, default) in PVT_FIELDS.items():
        value = pd.to_numeric(row[source], errors="coerce") if row is not None and source in row else np.nan
        if pd.isna(value) or float(value) <= 0:
            values[target], sources[target] = float(default), "значение по умолчанию"
        else:
            values[target] = float(value) * PVT_EDW_FACTORS.get(target, 1.0)
            sources[target] = "БД"

    relative_values: dict[str, float] = {}
    relative_sources: dict[str, str] = {}
    for target, (source, default) in PVT_RELATIVE_FIELDS.items():
        value = pd.to_numeric(row[source], errors="coerce") if row is not None and source in row else np.nan
        if pd.isna(value) or float(value) <= 0:
            relative_values[target] = float(default)
            relative_sources[target] = "значение по умолчанию"
        else:
            relative_values[target] = float(value)
            relative_sources[target] = "БД"
    return {
        "values": values,
        "sources": sources,
        "relative_values": relative_values,
        "relative_sources": relative_sources,
        "found": row is not None,
    }


class SqlEdwRepository(DatabaseRepository):
    """Repository whose SELECTs are shared by local DuckDB and KSPD EDW."""

    locator_prefix = "edw"

    @abstractmethod
    def _read_sql(self, query: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        raise NotImplementedError

    def list_field_objects(self) -> pd.DataFrame:
        return self._read_sql(FIELD_OBJECTS_SQL).reindex(
            columns=["field", "ms_1", "objects", "pl_1"]
        )

    def _selected_key(self, field: str, object_name: str) -> tuple[str, str]:
        selected = self._read_sql(
            SELECTED_KEY_SQL,
            {"field": field, "object_name": object_name},
        )
        if selected.empty:
            raise KeyError(f"Не найдена связка месторождение/объект: {field} / {object_name}")
        return str(selected.iloc[0]["ms_1"]), str(selected.iloc[0]["pl_1"])

    def get_history(self, field: str, object_name: str) -> pd.DataFrame:
        ms_code, pl_code = self._selected_key(field, object_name)
        frame = self._read_sql(HISTORY_SQL, {"ms_code": ms_code, "pl_code": pl_code})
        if frame.empty:
            return pd.DataFrame(columns=list(columns_name.values()))
        frame["date"] = pd.to_datetime(
            {
                "year": pd.to_numeric(frame.pop("_year"), errors="coerce"),
                "month": pd.to_numeric(frame.pop("_month"), errors="coerce"),
                "day": 1,
            },
            errors="coerce",
        )
        frame = frame.drop(columns=["_well_id"], errors="ignore")
        text_columns = {"date", "well_number", "field", "object", "objects", "work_marker", "well_status"}
        for column in set(columns_name.values()) - text_columns:
            if column in frame:
                frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
        frame = frame.reindex(columns=list(columns_name.values()))
        return frame.sort_values(["well_number", "date"], ascending=[True, False]).reset_index(drop=True)

    def get_fracs(self, field: str, object_name: str) -> pd.DataFrame:
        ms_code, pl_code = self._selected_key(field, object_name)
        frame = self._read_sql(FRACS_SQL, {"ms_code": ms_code, "pl_code": pl_code})
        if frame.empty:
            return pd.DataFrame(columns=FRAC_COLUMNS)
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        for column in set(FRAC_COLUMNS) - {"well_number", "date", "object", "comment"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
        return frame.reindex(columns=FRAC_COLUMNS).dropna(subset=["well_number"]).reset_index(drop=True)

    def get_pvt(self, field: str, object_name: str) -> dict[str, Any]:
        ms_code, pl_code = self._selected_key(field, object_name)
        frame = self._read_sql(PVT_SQL, {"ms_code": ms_code, "pl_code": pl_code})
        return _pvt_result(frame)

    def list_grids(self, field: str, object_name: str) -> list[GridChoice]:
        ms_code, pl_code = self._selected_key(field, object_name)
        frame = self._read_sql(GRIDS_SQL, {"ms_code": ms_code, "pl_code": pl_code})
        result: list[GridChoice] = []
        for row in frame.to_dict("records"):
            grid_type = repair_mojibake(str(row.get("gridtype", "")))
            map_key = classify_grid_type(grid_type)
            if map_key:
                result.append(
                    GridChoice(
                        locator=f"{self.locator_prefix}:{row['gridkey']}",
                        map_key=map_key,
                        name=repair_mojibake(str(row.get("name", ""))),
                        grid_type=grid_type,
                        date=str(row.get("dt", "")),
                        field=repair_mojibake(str(row.get("field", ""))),
                        object_name=repair_mojibake(str(row.get("object", ""))),
                    )
                )
        return result

    def get_grid(self, locator: str) -> dict[str, Any]:
        prefix = f"{self.locator_prefix}:"
        if not locator.startswith(prefix):
            raise ValueError(f"Некорректный идентификатор карты: {locator}")
        gridkey = locator.removeprefix(prefix)
        frame = self._read_sql(GRID_BY_KEY_SQL, {"gridkey": gridkey})
        if frame.empty:
            raise KeyError(f"Карта не найдена: {gridkey}")
        result = frame.iloc[0].to_dict()
        for key in ("name", "gridtype", "field", "object", "coordsys", "gridkey"):
            if key in result and result[key] is not None:
                result[key] = repair_mojibake(str(result[key]))
        return result


class LocalCsvRepository(SqlEdwRepository):
    """Local SQL imitation of EDW materialized from ``schema.table.csv`` files.

    The legacy class name is retained to avoid breaking the UI configuration.
    No calculation data is read from CSV at runtime.
    """

    locator_prefix = "local"

    def __init__(self, root: str | Path, db_path: str | Path | None = None):
        self.root = Path(root).resolve()
        self.db_path = Path(db_path).resolve() if db_path else self.root / ".local_edw" / "edw.db"
        if not self.root.is_dir():
            raise FileNotFoundError(f"Папка CSV-выгрузок не найдена: {self.root}")
        if not database_is_current(self.root, self.db_path):
            builder = self.root / "build_local_edw.py"
            raise RuntimeError(
                "Локальная EDW не собрана или CSV изменились. "
                f"Запустите: python \"{builder}\""
            )

    def _read_sql(self, query: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        connection = connect_local_database(self.db_path)
        try:
            # SQLAlchemy/Greenplum uses ``:name`` while DuckDB uses ``$name``.
            # Only the parameter marker is adapted; the SQL itself is shared.
            local_query = re.sub(
                r"(?<!:):([A-Za-z_][A-Za-z0-9_]*)", r"$\1", query
            )
            if params:
                return connection.execute(local_query, params).fetchdf()
            return connection.execute(local_query).fetchdf()
        finally:
            connection.close()


class KspdRepository(SqlEdwRepository):
    """Greenplum/EDW repository using the same SELECTs as the local imitation."""

    locator_prefix = "kspd"

    def __init__(self, host: str, port: int, database: str, user: str, password: str):
        self.host = host
        self.port = int(port)
        self.database = database
        self.user = user
        self.password = password
        self._engine = None

    @property
    def engine(self):
        if self._engine is not None:
            return self._engine
        try:
            from gpnlake import greenplumclient as gp

            self._engine = gp.GreenPlumClient(
                self.host, self.port, self.database, self.user, self.password
            ).engine
        except ImportError:
            try:
                from sqlalchemy import create_engine
                from sqlalchemy.engine import URL
            except ImportError as exc:
                raise RuntimeError(
                    "Для подключения к КСПД установите gpnlake либо sqlalchemy и psycopg2-binary."
                ) from exc
            url = URL.create(
                "postgresql+psycopg2",
                username=self.user,
                password=self.password,
                host=self.host,
                port=self.port,
                database=self.database,
            )
            self._engine = create_engine(url, pool_pre_ping=True)
        return self._engine

    def _read_sql(self, query: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        from sqlalchemy import text

        with self.engine.connect() as connection:
            return pd.read_sql(text(query), connection, params=params or {})

    def test_connection(self) -> None:
        self._read_sql("SELECT 1 AS ok")
