from __future__ import annotations

import os
from io import BytesIO
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from app.data_access import KspdRepository, LocalDatabaseRepository, MAP_SPECS
from app.data_access.materialize import grid_to_grd, save_uploaded_file
from app.data_access.models import PVT_FIELDS
from app.input_output.input_geo_phys_properties import (
    formatting_dict_geo_phys_properties,
    read_geo_phys_properties,
)


try:
    from dotenv import load_dotenv

    _env_path = Path(os.getenv("INFILL_ENV_FILE", "/home/default/persistent/.env"))
    if _env_path.is_file():
        load_dotenv(_env_path, override=False)
except ImportError:
    # Local mode does not require python-dotenv. KSPD settings can also be
    # supplied through ordinary process environment variables.
    pass

from app.data_access.settings import repository_config as developer_repository_config


SESSION_DEFAULTS: dict[str, Any] = {
    "db_connected": False,
    "selected_field": None,
    "selected_object": None,
    "loaded_object_key": None,
    "history_source": "База данных",
    "history_db_count": 0,
    "frac_db_count": 0,
    "well_frac_source": "Не использовать фрак-лист",
    "well_use_frac_sheet": False,
    "well_frac_type": "МГРП",
    "switch_economy": False,
    "mapping_size_pixel": 50,
    "mapping_radius_interpolate": 1000,
    "mapping_accounting_horwell": True,
    "mapping_auto_frac": False,
    "mapping_min_hor_stress": 45,
    "mapping_frac_half_length": 150.0,
    "mapping_KIN": 0.373,
    "drilling_percent_top": 10,
    "drilling_min_radius": 100,
    "drilling_sensitivity_quality": 20,
    "drilling_init_profit_cum_oil": 20,
    "well_r_w": 0.1,
    "well_t_p": 11,
    "well_efficiency": 0.95,
    "well_KUBS": 1.0,
    "well_KPPP": 0.9,
    "well_skin": 0.0,
    "well_length_frac_stage": 81,
    "well_xfr": 96.9,
    "well_w_f": 3.4,
    "well_k_f": 235.0,
    "well_first_months": 6,
    "well_last_months": 3,
    "well_default_radius_prod": 50,
    "well_default_radius_inj": 100,
    "well_filtration_perm_fact": True,
    "well_L": 1200,
    "well_min_length": 300,
    "well_buffer_project_wells": 200,
    "well_fix_P_well_init": False,
    "well_P_well_init": 0.0,
    "well_water_cut_source": "С карты",
    "well_k": 5,
    "well_threshold": 2500.0,
    "well_period_calculation": 25,
    "resfluid_relative_permeability": False,
    "resfluid_Sor": 0.26,
    "resfluid_Swc": 0.4,
    "resfluid_Fo": 1.0,
    "resfluid_Fw": 0.2,
    "resfluid_m2": 2.2,
    "resfluid_m1": 1.7,
    "resfluid_Bw": 1.01,
    "resfluid_kv_kh": 0.1,
    "calculation_running": False,
    "calculation_requested": False,
    "calculation_job": None,
    "calculation_error": None,
    "calculation_cancelled": False,
    "result_summary": None,
    "result_zip": None,
    "result_zip_name": None,
}

for _target, (_source, _default) in PVT_FIELDS.items():
    SESSION_DEFAULTS[f"pvt_{_target}"] = _default
    SESSION_DEFAULTS[f"pvt_source_{_target}"] = "значение по умолчанию"

for _spec in MAP_SPECS:
    SESSION_DEFAULTS[f"map_source_{_spec.key}"] = "База данных" if _spec.required else "Не использовать"
    SESSION_DEFAULTS[f"map_choice_{_spec.key}"] = None


@dataclass(frozen=True)
class ValidationIssue:
    section: str
    message: str


@dataclass(frozen=True)
class UploadedPayload:
    name: str
    data: bytes

    def getvalue(self) -> bytes:
        return self.data


def remember_upload(widget_key: str, storage_key: str) -> None:
    """Copy an UploadedFile to stable session state before its widget disappears."""
    upload = st.session_state.get(widget_key)
    if upload is not None:
        st.session_state[storage_key] = UploadedPayload(upload.name, upload.getvalue())


def apply_geo_properties_upload() -> None:
    payload = st.session_state.get("gfh_upload_value")
    if payload is None:
        return
    signature = (payload.name, len(payload.data), hash(payload.data))
    if st.session_state.get("gfh_applied_signature") == signature:
        return
    try:
        values = read_geo_phys_properties(
            BytesIO(payload.data),
            st.session_state.selected_field,
            st.session_state.selected_object,
        )
        for name in PVT_FIELDS:
            value = pd.to_numeric(values.get(name), errors="coerce")
            if not pd.isna(value) and float(value) > 0:
                st.session_state[f"pvt_{name}"] = float(value)
                st.session_state[f"pvt_source_{name}"] = "файл ГФХ"
        st.session_state.gfh_applied_signature = signature
        st.session_state.gfh_upload_error = None
    except Exception as error:
        st.session_state.gfh_upload_error = str(error)


def initialize_session() -> None:
    for key, value in SESSION_DEFAULTS.items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault("session_id", uuid.uuid4().hex)
    st.session_state.setdefault("available_grids", [])
    st.session_state.setdefault("pvt_found", False)
    config = repository_config()
    if config["type"] == "local":
        st.session_state.db_connected = Path(config["db_path"]).is_file()
    else:
        st.session_state.db_connected = bool(config.get("host") and config.get("user"))


def repository_config() -> dict[str, Any]:
    return developer_repository_config()


def make_repository(config: dict[str, Any] | None = None):
    config = config or repository_config()
    if config["type"] == "local":
        return LocalDatabaseRepository(config["db_path"])
    return KspdRepository(
        config["host"], config["port"], config["database"], config["user"], config["password"]
    )


@st.cache_data(show_spinner=False, ttl=600)
def cached_field_objects(config: dict[str, Any]) -> pd.DataFrame:
    return make_repository(config).list_field_objects()


@st.cache_data(show_spinner=False, max_entries=6)
def cached_history(config: dict[str, Any], field: str, object_name: str) -> pd.DataFrame:
    return make_repository(config).get_history(field, object_name)


@st.cache_data(show_spinner=False, max_entries=12)
def cached_fracs(config: dict[str, Any], field: str, object_name: str) -> pd.DataFrame:
    return make_repository(config).get_fracs(field, object_name)


@st.cache_data(show_spinner=False, max_entries=12)
def cached_pvt(config: dict[str, Any], field: str, object_name: str) -> dict[str, Any]:
    return make_repository(config).get_pvt(field, object_name)


@st.cache_data(show_spinner=False, max_entries=12)
def cached_grids(config: dict[str, Any], field: str, object_name: str):
    return make_repository(config).list_grids(field, object_name)


def reset_object_data() -> None:
    st.session_state.loaded_object_key = None
    st.session_state.history_db_count = 0
    st.session_state.frac_db_count = 0
    st.session_state.available_grids = []
    st.session_state.pvt_found = False
    for key in (
        "history_upload_value",
        "history_upload",
        "well_frac_upload_value",
        "well_frac_upload",
        "gfh_upload_value",
        "gfh_upload",
        "gfh_applied_signature",
        "gfh_upload_error",
    ):
        st.session_state.pop(key, None)
    for name, (_source, default) in PVT_FIELDS.items():
        st.session_state[f"pvt_{name}"] = float(default)
        st.session_state[f"pvt_source_{name}"] = "значение по умолчанию"
    for spec in MAP_SPECS:
        st.session_state.pop(f"map_upload_value_{spec.key}", None)
        st.session_state.pop(f"map_upload_{spec.key}", None)
        st.session_state[f"map_choice_{spec.key}"] = None
        st.session_state[f"map_source_{spec.key}"] = (
            "База данных" if spec.required else "Не использовать"
        )


def load_selected_object_data(force: bool = False) -> None:
    field = st.session_state.selected_field
    object_name = st.session_state.selected_object
    if not field or not object_name:
        reset_object_data()
        return
    config = repository_config()
    object_key = (tuple(sorted((key, str(value)) for key, value in config.items())), field, object_name)
    if not force and st.session_state.loaded_object_key == object_key:
        return
    history = cached_history(config, field, object_name)
    fracs = cached_fracs(config, field, object_name)
    pvt = cached_pvt(config, field, object_name)
    grids = cached_grids(config, field, object_name)
    st.session_state.history_db_count = int(history["well_number"].nunique()) if not history.empty else 0
    st.session_state.frac_db_count = int(fracs["well_number"].nunique()) if not fracs.empty else 0
    st.session_state.available_grids = grids
    st.session_state.pvt_found = bool(pvt["found"])
    for name, value in pvt["values"].items():
        st.session_state[f"pvt_{name}"] = float(value)
        st.session_state[f"pvt_source_{name}"] = pvt["sources"][name]
    for spec in MAP_SPECS:
        choices = [grid for grid in grids if grid.map_key == spec.key]
        st.session_state[f"map_choice_{spec.key}"] = choices[0].locator if choices else None
        st.session_state[f"map_source_{spec.key}"] = (
            ("База данных" if choices else "Загрузить файл")
            if spec.required
            else "Не использовать"
        )
    if st.session_state.history_db_count:
        st.session_state.history_source = "База данных"
    else:
        st.session_state.history_source = "Загрузить файл"
    if st.session_state.frac_db_count:
        st.session_state.well_frac_source = "База данных"
        st.session_state.well_use_frac_sheet = True
    else:
        st.session_state.well_frac_source = "Не использовать фрак-лист"
        st.session_state.well_use_frac_sheet = False
    st.session_state.loaded_object_key = object_key


def validation_issues() -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if not st.session_state.db_connected:
        issues.append(ValidationIssue("Исходные данные", "Нет подключения к выбранной базе данных"))
    if not st.session_state.selected_field:
        issues.append(ValidationIssue("Исходные данные", "Не выбрано месторождение"))
    if not st.session_state.selected_object:
        issues.append(ValidationIssue("Исходные данные", "Не выбран объект"))
    if st.session_state.selected_field and st.session_state.selected_object and not st.session_state.loaded_object_key:
        issues.append(ValidationIssue("Исходные данные", "Данные выбранного объекта ещё не проверены"))

    if st.session_state.history_source == "База данных":
        if st.session_state.history_db_count <= 0:
            issues.append(ValidationIssue("Исходные данные", "В БД нет истории работы скважин"))
    elif st.session_state.get("history_upload_value") is None:
        issues.append(ValidationIssue("Исходные данные", "Загрузите файл истории работы скважин"))

    for spec in MAP_SPECS:
        if not spec.required:
            continue
        if st.session_state[f"map_source_{spec.key}"] == "База данных":
            if not st.session_state.get(f"map_choice_{spec.key}"):
                issues.append(ValidationIssue("Исходные данные", f"Не выбрана карта «{spec.label}»"))
        elif st.session_state.get(f"map_upload_value_{spec.key}") is None:
            issues.append(ValidationIssue("Исходные данные", f"Загрузите карту «{spec.label}»"))

    for name in PVT_FIELDS:
        value = st.session_state.get(f"pvt_{name}")
        if value is None or float(value) <= 0:
            issues.append(ValidationIssue("Свойства пласта и флюидов", f"Параметр {name} должен быть больше нуля"))

    if st.session_state.well_use_frac_sheet:
        if st.session_state.well_frac_source == "База данных" and st.session_state.frac_db_count <= 0:
            issues.append(ValidationIssue("Параметры скважин", "В БД нет данных ГРП — загрузите файл"))
        if st.session_state.well_frac_source == "Загрузить файл" and st.session_state.get("well_frac_upload_value") is None:
            issues.append(ValidationIssue("Параметры скважин", "Загрузите фрак-лист"))
    return issues


def _materialize_inputs(work_dir: Path, repository) -> tuple[dict[str, str], dict[str, Any]]:
    input_dir = work_dir / "input"
    maps_dir = input_dir / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    field = st.session_state.selected_field
    object_name = st.session_state.selected_object
    config = repository_config()
    preloaded: dict[str, Any] = {}

    if st.session_state.history_source == "База данных":
        preloaded["data_history"] = cached_history(config, field, object_name)
        history_path = ""
    else:
        upload = st.session_state.history_upload_value
        suffix = Path(upload.name).suffix or ".xlsx"
        history_path = save_uploaded_file(upload, input_dir / f"data_history{suffix}")

    for spec in MAP_SPECS:
        source = st.session_state[f"map_source_{spec.key}"]
        if source == "База данных":
            locator = st.session_state.get(f"map_choice_{spec.key}")
            if locator:
                grid_to_grd(repository.get_grid(locator), maps_dir / spec.filename)
        elif source == "Загрузить файл":
            upload = st.session_state.get(f"map_upload_value_{spec.key}")
            if upload is not None:
                save_uploaded_file(upload, maps_dir / spec.filename)

    frac_path = input_dir / "frac.xlsx"
    if st.session_state.well_use_frac_sheet:
        if st.session_state.well_frac_source == "База данных":
            preloaded["frac_info"] = cached_fracs(config, field, object_name)
        else:
            upload = st.session_state.well_frac_upload_value
            suffix = Path(upload.name).suffix or ".xlsx"
            frac_path = save_uploaded_file(upload, input_dir / f"frac{suffix}")
    paths = {
        "data_well_directory": str(history_path),
        "maps_directory": str(maps_dir),
        "path_geo_phys_properties": "",
        "path_frac": str(frac_path) if st.session_state.well_use_frac_sheet else "-",
        "path_economy": "-",
        "save_directory": str(work_dir / "results"),
    }
    return paths, preloaded


def build_parameters(work_dir: Path) -> dict[str, Any]:
    repository = make_repository()
    paths, preloaded = _materialize_inputs(work_dir, repository)
    raw_pvt = {name: float(st.session_state[f"pvt_{name}"]) for name in PVT_FIELDS}
    formatted_pvt = formatting_dict_geo_phys_properties(raw_pvt)
    return {
        "paths": paths,
        "preloaded": {**preloaded, "geo_phys_properties": formatted_pvt},
        "source": {
            "database": repository_config()["type"],
            "field": st.session_state.selected_field,
            "object": st.session_state.selected_object,
        },
        "switches": {
            "switch_fracList_params": bool(st.session_state.well_use_frac_sheet),
            "switch_frac_inj_well": bool(st.session_state.mapping_auto_frac),
            "switch_filtration_perm_fact": bool(st.session_state.well_filtration_perm_fact),
            "switch_economy": False,
            "switch_adaptation_relative_permeability": bool(st.session_state.resfluid_relative_permeability),
            "switch_wc_from_map": st.session_state.well_water_cut_source == "С карты",
            "switch_accounting_horwell": bool(st.session_state.mapping_accounting_horwell),
            "switch_fix_P_well_init": bool(st.session_state.well_fix_P_well_init),
        },
        "maps": {
            "default_size_pixel": int(st.session_state.mapping_size_pixel),
            "radius_interpolate": int(st.session_state.mapping_radius_interpolate),
            "azimuth_sigma_h_min": int(st.session_state.mapping_min_hor_stress),
            "l_half_fracture": float(st.session_state.mapping_frac_half_length),
            "KIN": float(st.session_state.mapping_KIN),
        },
        "drill_zones": {
            "percent_top": int(st.session_state.drilling_percent_top),
            "min_radius": int(st.session_state.drilling_min_radius),
            "sensitivity_quality_drill": int(st.session_state.drilling_sensitivity_quality),
            "init_profit_cum_oil": int(st.session_state.drilling_init_profit_cum_oil),
        },
        "well_params": {
            "general": {
                "t_p": float(st.session_state.well_t_p),
                "r_w": float(st.session_state.well_r_w),
                "well_efficiency": float(st.session_state.well_efficiency),
                "KPPP": float(st.session_state.well_KPPP),
                "skin": float(st.session_state.well_skin),
                "KUBS": float(st.session_state.well_KUBS),
            },
            "fracturing": {
                "Type_Frac": None if st.session_state.well_frac_type == "Без ГРП" else st.session_state.well_frac_type,
                "length_FracStage": float(st.session_state.well_length_frac_stage),
                "k_f": float(st.session_state.well_k_f),
                "xfr": float(st.session_state.well_xfr),
                "w_f": float(st.session_state.well_w_f),
            },
            "fact_wells_params": {
                "first_months": int(st.session_state.well_first_months),
                "last_months": int(st.session_state.well_last_months),
                "default_radius_prod": float(st.session_state.well_default_radius_prod),
                "default_radius_inj": float(st.session_state.well_default_radius_inj),
            },
            "proj_wells_params": {
                "L": int(st.session_state.well_L),
                "min_length": int(st.session_state.well_min_length),
                "buffer_project_wells": int(st.session_state.well_buffer_project_wells),
                "fix_P_well_init": float(st.session_state.well_P_well_init) if st.session_state.well_fix_P_well_init else 0.0,
                "k": int(st.session_state.well_k),
                "threshold": float(st.session_state.well_threshold),
                "period_calculation": int(st.session_state.well_period_calculation),
            },
        },
        "reservoir_fluid_properties": {
            **formatted_pvt,
            "kv_kh": float(st.session_state.resfluid_kv_kh),
            "Swc": float(st.session_state.resfluid_Swc),
            "Sor": float(st.session_state.resfluid_Sor),
            "Fw": float(st.session_state.resfluid_Fw),
            "m1": float(st.session_state.resfluid_m1),
            "Fo": float(st.session_state.resfluid_Fo),
            "m2": float(st.session_state.resfluid_m2),
            "Bw": float(st.session_state.resfluid_Bw),
        },
        "economy_params": {"day_in_month": 29, "start_date": datetime(datetime.now().year + 1, 1, 1)},
    }


def new_work_directory() -> Path:
    base = Path(tempfile.gettempdir()) / "infill_drilling_streamlit" / st.session_state.session_id
    run_dir = base / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


