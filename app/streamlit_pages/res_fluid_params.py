from __future__ import annotations

import streamlit as st

from app.data_access.models import PVT_UI_FIELDS
from app.streamlit_runtime import (
    apply_geo_properties_upload,
    initialize_session,
    remember_upload,
)


def _mark_manual(name: str) -> None:
    st.session_state[f"pvt_source_{name}"] = "введено вручную"


def _pvt_tab() -> None:
    if st.session_state.pvt_found:
        st.success("Для выбранного месторождения и объекта найдена запись PVT")
    else:
        st.warning("Запись PVT не найдена. Установлены физически допустимые значения по умолчанию — проверьте их.")
    upload = st.file_uploader(
        "Загрузить файл ГФХ",
        type=["xlsx", "xls", "xlsm"],
        key="gfh_upload",
        on_change=remember_upload,
        args=("gfh_upload", "gfh_upload_value"),
        disabled=not (st.session_state.selected_field and st.session_state.selected_object),
        help="Значения из файла заполнят форму; после этого их можно изменить вручную.",
    )
    apply_geo_properties_upload()
    stored = st.session_state.get("gfh_upload_value")
    if stored is not None and not st.session_state.get("gfh_upload_error"):
        st.caption(f"Параметры загружены из {stored.name}")
    if st.session_state.get("gfh_upload_error"):
        st.error(f"Не удалось прочитать файл ГФХ: {st.session_state.gfh_upload_error}")

    for name, (label, minimum, maximum) in PVT_UI_FIELDS.items():
        col_label, col_value = st.columns([1.2, 0.8], vertical_alignment="center")
        col_label.markdown(label)
        col_value.number_input(
            label,
            min_value=float(minimum),
            max_value=float(maximum),
            step=0.001,
            format="%.4f",
            key=f"pvt_{name}",
            on_change=_mark_manual,
            args=(name,),
            label_visibility="collapsed",
            persist_state="session",
        )


def _number_row(label: str, key: str, minimum: float, maximum: float, step: float = 0.001, disabled: bool = False):
    col_label, col_value = st.columns([1.2, 0.8], vertical_alignment="center")
    col_label.markdown(label)
    col_value.number_input(
        label,
        min_value=float(minimum),
        max_value=float(maximum),
        step=float(step),
        format="%.3f",
        key=key,
        disabled=disabled,
        label_visibility="collapsed",
        persist_state="session",
    )


def _relative_tab() -> None:
    st.checkbox(
        "Автоадаптация ОФП по истории работы",
        key="resfluid_relative_permeability",
        help="При включении параметры ОФП ниже служат стартовым приближением и уточняются моделью.",
        persist_state="session",
    )
    auto = st.session_state.resfluid_relative_permeability
    fields = [
        ("Остаточная нефтенасыщенность, д.ед.", "resfluid_Sor", 0.0, 1.0),
        ("Связанная водонасыщенность, д.ед.", "resfluid_Swc", 0.0, 1.0),
        ("Концевая точка ОФП по нефти", "resfluid_Fo", 0.0, 1.0),
        ("Концевая точка ОФП по воде", "resfluid_Fw", 0.0, 1.0),
        ("Степень Corey по нефти", "resfluid_m2", 0.001, 10.0),
        ("Степень Corey по воде", "resfluid_m1", 0.001, 10.0),
    ]
    for label, key, minimum, maximum in fields:
        _number_row(label, key, minimum, maximum, disabled=auto)
    st.divider()
    _number_row("Объёмный коэффициент воды, м³/м³", "resfluid_Bw", 0.001, 1.5)
    _number_row("Анизотропия пласта, д.ед.", "resfluid_kv_kh", 0.001, 1.5)


def show():
    initialize_session()
    st.header("Свойства пласта и флюидов")
    st.write("Геолого-физические и PVT-параметры автоматически заполняются из БД. Любое значение можно скорректировать вручную.")
    tab_pvt, tab_relative = st.tabs(["Геолого-физические свойства и PVT", "ОФП и дополнительные параметры"])
    with tab_pvt:
        _pvt_tab()
    with tab_relative:
        _relative_tab()
