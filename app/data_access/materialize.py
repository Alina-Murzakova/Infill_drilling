from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.config import columns_name, columns_name_frac
from app.maps_handler.maps import Map


def save_uploaded_file(uploaded_file: Any, destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(uploaded_file, "getvalue"):
        payload = uploaded_file.getvalue()
    elif isinstance(uploaded_file, (bytes, bytearray)):
        payload = bytes(uploaded_file)
    elif hasattr(uploaded_file, "read"):
        payload = uploaded_file.read()
    else:
        raise TypeError("Не удалось прочитать загруженный файл")
    destination.write_bytes(payload)
    return destination


def history_to_excel(history: pd.DataFrame, destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    reverse_columns = {new: old for old, new in columns_name.items()}
    export = history.reindex(columns=list(columns_name.values())).rename(columns=reverse_columns)
    with pd.ExcelWriter(destination, engine="openpyxl") as writer:
        export.to_excel(writer, sheet_name="Данные", index=False)
    return destination


def fracs_to_excel(fracs: pd.DataFrame, destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    reverse_columns = {new: old for old, new in columns_name_frac.items()}
    export = fracs.reindex(columns=list(columns_name_frac.values())).rename(columns=reverse_columns)
    # input_frac_info intentionally reads the second row as the header.
    with pd.ExcelWriter(destination, engine="openpyxl") as writer:
        export.to_excel(writer, sheet_name="ГРП", startrow=1, index=False)
    return destination


def grid_to_grd(grid: dict[str, Any], destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = grid.get("data")
    if isinstance(payload, str):
        payload = json.loads(payload)
    data = np.asarray(payload, dtype=float)
    count_x = int(float(grid["countx"]))
    count_y = int(float(grid["county"]))
    if data.shape == (count_x, count_y):
        data = data.T
    if data.shape != (count_y, count_x):
        if data.size != count_x * count_y:
            raise ValueError(
                f"Размер массива карты {data.shape} не соответствует метаданным "
                f"({count_y}, {count_x})"
            )
        data = data.reshape((count_y, count_x))

    blank = pd.to_numeric(grid.get("blankvalue"), errors="coerce")
    if not pd.isna(blank):
        data[np.isclose(data, float(blank))] = 0
    data[~np.isfinite(data)] = 0
    data[np.abs(data) > 1e30] = 0

    x_min = float(grid.get("x_min") or grid.get("offsetx"))
    y_max = float(grid.get("y_max") or grid.get("offsety"))
    step_x = abs(float(grid.get("stepx") or ((float(grid["x_max"]) - x_min) / max(count_x - 1, 1))))
    step_y = abs(float(grid.get("stepy") or ((y_max - float(grid["y_min"])) / max(count_y - 1, 1))))
    raster = Map(data, [x_min, step_x, 0, y_max, 0, -step_y], "", destination.stem)
    raster.save_grd_file(str(destination))
    return destination
