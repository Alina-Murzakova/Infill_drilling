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


def _number_row(label: str,
                key: str,
                min_value: float,
                max_value: float,
                step: float = 0.001,
                format: str = "%.3f",
                disabled: bool = False,
                on_change=None,
                args=None,
                ):
    """
    Строка параметра:
    название слева + поле значения справа.
    """
    col_label, col_value = st.columns([0.5, 1.0], gap="small")

    col_label.markdown(
        f'<div class="resfluid-label">{label}</div>',
        unsafe_allow_html=True,
    )

    col_value.number_input(
        label,
        min_value=float(min_value),
        max_value=float(max_value),
        step=float(step),
        format=format,
        key=key,
        disabled=disabled,
        on_change=on_change,
        args=args,
        label_visibility="collapsed",
        persist_state="session",
    )


def _pvt_tab() -> None:
    with st.container(width=950):
        if st.session_state.pvt_found:
            st.success("Для выбранного месторождения и объекта найдена запись PVT.")
        else:
            st.warning("Запись PVT не найдена. Установлены физически допустимые значения по умолчанию — проверьте их.")

    with st.container(width=950):
        st.file_uploader(
            "Загрузить файл ГФХ",
            type=["xlsx", "xls", "xlsm"],
            key="gfh_upload",
            on_change=remember_upload,
            args=("gfh_upload", "gfh_upload_value"),
            disabled=not (st.session_state.selected_field and st.session_state.selected_object),
            help="Для загрузки необходимо выбрать месторождения и объект. "
                 "Значения из файла автоматически подтянутся в форму; после этого их можно изменить вручную.",
        )

    apply_geo_properties_upload()
    stored = st.session_state.get("gfh_upload_value")
    if stored is not None and not st.session_state.get("gfh_upload_error"):
        st.caption(f"Параметры загружены из {stored.name}")

    if st.session_state.get("gfh_upload_error"):
        st.error(f"Не удалось прочитать файл ГФХ: {st.session_state.gfh_upload_error}")

    for name, (label, minimum, maximum) in PVT_UI_FIELDS.items():
        _number_row(
            label=label,
            key=f"pvt_{name}",
            min_value=minimum,
            max_value=maximum,
            step=0.001,
            format="%.3f",
            on_change=_mark_manual,
            args=(name,),
        )


def _relative_tab() -> None:
    st.checkbox(
        "Автоадаптация ОФП по истории работы",
        key="resfluid_relative_permeability",
        help="При включении автоадаптации параметры ОФП ниже служат стартовым приближением и уточняются моделью.",
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
        _number_row(
            label=label,
            key=key,
            min_value=minimum,
            max_value=maximum,
            disabled=auto,
        )

    st.divider()
    _number_row("Объёмный коэффициент воды, м³/м³", "resfluid_Bw", 0.001, 1.5)
    _number_row("Анизотропия пласта, д.ед.", "resfluid_kv_kh", 0.001, 1.5)


def show():
    initialize_session()
    st.header("Свойства пласта и флюидов")

    # Краткое описание раздела
    st.markdown(
        '<div class="resfluid-description">'
        'Настройка свойств пласта и флюидов, используемых при выполнении расчёта.'
        '<br>'
        'Геолого-физические и PVT-параметры автоматически заполняются из БД. '
        'Любое значение можно скорректировать вручную. '
        '</div>',
        unsafe_allow_html=True,
    )

    tab_pvt, tab_relative = st.tabs(["Геолого-физические свойства и PVT", "ОФП и дополнительные параметры"])
    with tab_pvt:
        _pvt_tab()
    with tab_relative:
        _relative_tab()
