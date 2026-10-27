"""Build and open the single-file, schema-compatible local EDW."""

from __future__ import annotations

import csv
import os
import shutil
import tempfile
from collections import defaultdict
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import duckdb
import pandas as pd


CSV_ENCODING = "cp1251"
FORMAT_VERSION = 3
csv.field_size_limit(2**31 - 1)


INDEXES: dict[tuple[str, str], tuple[tuple[str, ...], ...]] = {
    ("dp_rid_ods_corp_oisproduction_db5555", "arx"): (
        ("ms_1", "pl_1", "sk_1", "gg_1", "dd_1"),
        ("sk_1", "ms_1", "pl_1"),
    ),
    ("dp_rid_ods_corp_oisproduction_db5555", "class"): (("cd_1",),),
    ("dp_rid_ods_corp_oisproduction_db5555", "spskv"): (("sk_1",),),
    ("dp_rid_ods_corp_geobd_dwh", "geooistranslation"): (("sk_1",), ("uwi",)),
    ("dp_rid_ods_corp_geobd_super_man", "plastcross"): (
        ("uwi", "object_name", "orig_layers_name"),
    ),
    ("dp_rid_ods_corp_geobd_super_man", "grp_info"): (("uwi", "layer"),),
    ("dp_rid_ods_corp_shtr_wellop", "well_layer_op"): (
        ("well_id", "layer_id", "calc_date"),
    ),
    ("dp_rid_ods_corp_shtr_wellop", "well_layer_op_inj"): (
        ("well_id", "layer_id", "calc_date"),
    ),
    ("dp_rid_ods_corp_ngtsmart_rdfsmart", "pvt"): (("oil_field", "layer"),),
    ("dp_rid_ods_corp_eragrad_api", "gridlist"): (
        ("field", "object", "dt"),
        ("gridkey",),
    ),
}

REQUIRED_TABLES = frozenset(INDEXES)


GRID_TEXT_COLUMNS = {
    "comment",
    "coordsys",
    "docname",
    "executor",
    "field",
    "gridkey",
    "gridtype",
    "name",
    "object",
    "region",
}


def repair_mojibake(value: str) -> str:
    """Repair UTF-8 text that was exported through a cp1251 code page."""

    value = value.strip()
    if not value or ("Р" not in value and "С" not in value):
        return value
    try:
        return value.encode("cp1251").decode("utf-8").strip()
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _source_tables(source_dir: Path) -> list[tuple[str, str, Path]]:
    result: list[tuple[str, str, Path]] = []
    for path in sorted(source_dir.glob("*.csv")):
        parts = path.stem.split(".", 1)
        if len(parts) != 2:
            # field_objects_keys.csv is a query result, not an EDW table.
            continue
        schema, table = parts
        result.append((schema, table, path))
    if not result:
        raise FileNotFoundError(
            f"В {source_dir} нет CSV с именами вида «схема.таблица.csv»"
        )
    return result


def _signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def database_is_current(source_dir: str | Path, db_path: str | Path) -> bool:
    """Check that the one-file DuckDB contains every unchanged source CSV."""

    source_dir = Path(source_dir)
    db_path = Path(db_path)
    if not db_path.is_file():
        return False
    try:
        sources = _source_tables(source_dir)
        with closing(duckdb.connect(str(db_path), read_only=True)) as connection:
            version = connection.execute(
                "SELECT value FROM main.local_edw_metadata WHERE key='format_version'"
            ).fetchone()
            rows = connection.execute(
                """SELECT schema_name, table_name, csv_name, size_bytes, mtime_ns
                   FROM main.local_edw_sources"""
            ).fetchall()
            database_tables = set(
                connection.execute(
                    """SELECT table_schema, table_name
                       FROM information_schema.tables
                       WHERE table_type = 'BASE TABLE'"""
                ).fetchall()
            )
    except (duckdb.Error, FileNotFoundError, OSError):
        return False

    if version is None or int(version[0]) != FORMAT_VERSION:
        return False
    recorded = {
        csv_name: (schema, table, int(size), int(mtime))
        for schema, table, csv_name, size, mtime in rows
    }
    if set(recorded) != {path.name for _, _, path in sources}:
        return False
    for schema, table, path in sources:
        saved_schema, saved_table, saved_size, saved_mtime = recorded[path.name]
        if (saved_schema, saved_table) != (schema, table):
            return False
        if _signature(path) != (saved_size, saved_mtime):
            return False
        if (schema, table) not in database_tables:
            return False
    return True


def validate_local_database(db_path: str | Path) -> Path:
    """Validate a ready-to-use local EDW without consulting source CSV files."""

    db_path = Path(db_path).resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f"Локальная EDW не найдена: {db_path}")

    try:
        with closing(duckdb.connect(str(db_path), read_only=True)) as connection:
            version = connection.execute(
                "SELECT value FROM main.local_edw_metadata WHERE key='format_version'"
            ).fetchone()
            database_tables = set(
                connection.execute(
                    """SELECT table_schema, table_name
                       FROM information_schema.tables
                       WHERE table_type = 'BASE TABLE'"""
                ).fetchall()
            )
    except (duckdb.Error, OSError) as exc:
        raise RuntimeError(
            f"Не удалось открыть локальную EDW {db_path}: {exc}"
        ) from exc

    try:
        format_version = int(version[0]) if version is not None else None
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Локальная EDW {db_path} имеет некорректную версию формата"
        ) from exc
    if format_version != FORMAT_VERSION:
        raise RuntimeError(
            f"Локальная EDW {db_path} имеет версию формата {format_version}; "
            f"ожидается {FORMAT_VERSION}"
        )

    missing_tables = sorted(REQUIRED_TABLES - database_tables)
    if missing_tables:
        formatted = ", ".join(f"{schema}.{table}" for schema, table in missing_tables)
        raise RuntimeError(
            f"В локальной EDW {db_path} отсутствуют обязательные таблицы: {formatted}"
        )
    return db_path


def _batches(
    rows: Iterable[list[str | None]], size: int
) -> Iterable[list[list[str | None]]]:
    batch: list[list[str | None]] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _import_csv(
    connection: duckdb.DuckDBPyConnection,
    schema: str,
    table: str,
    path: Path,
    progress: Callable[[str], None] | None = None,
) -> int:
    # Some grid captions contain isolated bytes that are undefined in cp1251.
    # Replacing such a byte preserves the row and its numeric ``data`` payload.
    with path.open("r", encoding=CSV_ENCODING, errors="replace", newline="") as stream:
        reader = csv.reader(stream)
        try:
            columns = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Пустой CSV: {path.name}") from exc
        if len(columns) != len(set(columns)):
            raise ValueError(f"В CSV {path.name} есть дублирующиеся столбцы")

        qualified_table = f"{_quote_identifier(schema)}.{_quote_identifier(table)}"
        definitions = ", ".join(
            f"{_quote_identifier(column)} VARCHAR" for column in columns
        )
        connection.execute(f"CREATE TABLE {qualified_table} ({definitions})")
        repair_positions = (
            {index for index, name in enumerate(columns) if name in GRID_TEXT_COLUMNS}
            if (schema, table) == ("dp_rid_ods_corp_eragrad_api", "gridlist")
            else set()
        )

        def normalized_rows() -> Iterable[list[str | None]]:
            for line_number, row in enumerate(reader, start=2):
                if len(row) != len(columns):
                    raise ValueError(
                        f"{path.name}, строка {line_number}: "
                        f"ожидалось {len(columns)} полей, получено {len(row)}"
                    )
                values: list[str | None] = [
                    value if value != "" else None for value in row
                ]
                for position in repair_positions:
                    if values[position] is not None:
                        values[position] = repair_mojibake(str(values[position]))
                yield values

        row_count = 0
        batch_size = 1 if table == "gridlist" else 10_000
        for batch in _batches(normalized_rows(), batch_size):
            frame = pd.DataFrame.from_records(batch, columns=columns)
            connection.register("_local_edw_batch", frame)
            try:
                connection.execute(
                    f"INSERT INTO {qualified_table} SELECT * FROM _local_edw_batch"
                )
            finally:
                connection.unregister("_local_edw_batch")
            row_count += len(batch)
            progress_step = 100 if table == "gridlist" else 100_000
            if progress is not None and row_count % progress_step == 0:
                progress(f"    {row_count:,} строк загружено…")

    for number, index_columns in enumerate(INDEXES.get((schema, table), ()), start=1):
        columns_sql = ", ".join(_quote_identifier(column) for column in index_columns)
        index_name = _quote_identifier(f"idx_{schema}_{table}_{number}")
        connection.execute(
            f"CREATE INDEX {index_name} ON {qualified_table} ({columns_sql})"
        )
    connection.execute(f"ANALYZE {qualified_table}")
    connection.commit()
    return row_count


def _remove_legacy_layout(db_path: Path, progress: Callable[[str], None]) -> None:
    """Remove the former multi-file SQLite layout after a successful switch."""

    root = db_path.parent.resolve()
    for name in ("schemas", "manifests"):
        path = (root / name).resolve()
        if path.parent != root or not path.exists():
            continue
        try:
            shutil.rmtree(path)
        except OSError:
            progress(
                f"Не удалось удалить устаревшую папку {path}. "
                "Закройте приложение и удалите её вручную."
            )


def build_local_database(
    source_dir: str | Path,
    db_path: str | Path,
    *,
    force: bool = False,
    progress: Callable[[str], None] = print,
) -> Path:
    """Build all ``schema.table.csv`` sources into one DuckDB file."""

    source_dir = Path(source_dir).resolve()
    db_path = Path(db_path).resolve()
    sources = _source_tables(source_dir)
    if not force and database_is_current(source_dir, db_path):
        _remove_legacy_layout(db_path, progress)
        progress(f"Локальная EDW актуальна: {db_path}")
        return db_path

    db_path.parent.mkdir(parents=True, exist_ok=True)
    build_root = Path(tempfile.mkdtemp(prefix=".local-edw-build-", dir=db_path.parent))
    temporary_db = build_root / db_path.name
    grouped: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for schema, table, path in sources:
        grouped[schema].append((table, path))

    source_rows: list[tuple[str, str, str, int, int, int]] = []
    try:
        with closing(duckdb.connect(str(temporary_db))) as connection:
            for schema, tables in grouped.items():
                progress(f"Схема {schema}")
                connection.execute(f"CREATE SCHEMA {_quote_identifier(schema)}")
                for table, path in tables:
                    progress(f"  {path.name} -> {schema}.{table}")
                    row_count = _import_csv(connection, schema, table, path, progress)
                    size, mtime = _signature(path)
                    source_rows.append(
                        (schema, table, path.name, size, mtime, row_count)
                    )
                    progress(f"    {row_count:,} строк")

            connection.execute(
                "CREATE TABLE main.local_edw_metadata "
                "(key VARCHAR PRIMARY KEY, value VARCHAR NOT NULL)"
            )
            connection.execute(
                """CREATE TABLE main.local_edw_sources (
                    schema_name VARCHAR NOT NULL,
                    table_name VARCHAR NOT NULL,
                    csv_name VARCHAR NOT NULL UNIQUE,
                    size_bytes BIGINT NOT NULL,
                    mtime_ns BIGINT NOT NULL,
                    row_count BIGINT NOT NULL,
                    PRIMARY KEY(schema_name, table_name)
                )"""
            )
            connection.executemany(
                "INSERT INTO main.local_edw_metadata VALUES (?, ?)",
                (
                    ("format_version", str(FORMAT_VERSION)),
                    ("built_at_utc", datetime.now(timezone.utc).isoformat()),
                    ("source_dir", str(source_dir)),
                ),
            )
            connection.executemany(
                "INSERT INTO main.local_edw_sources VALUES (?, ?, ?, ?, ?, ?)",
                source_rows,
            )
            connection.commit()
            connection.execute("CHECKPOINT")

        try:
            os.replace(temporary_db, db_path)
        except PermissionError as exc:
            raise PermissionError(
                f"Не удалось заменить {db_path}. Закройте приложение, которое "
                "использует локальную БД, либо удалите старый edw.db и повторите сборку."
            ) from exc

        _remove_legacy_layout(db_path, progress)
        progress(f"Локальная EDW собрана: {db_path}")
        return db_path
    finally:
        shutil.rmtree(build_root, ignore_errors=True)


def connect_local_database(db_path: str | Path) -> duckdb.DuckDBPyConnection:
    """Open the single-file local EDW in read-only mode."""

    db_path = Path(db_path).resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f"Локальная EDW не найдена: {db_path}")
    return duckdb.connect(str(db_path), read_only=True)
