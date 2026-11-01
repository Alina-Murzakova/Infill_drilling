from __future__ import annotations

import streamlit as st

from app.data_access import MAP_SPECS
from app.streamlit_runtime import (
    cached_field_objects,
    initialize_session,
    load_selected_object_data,
    remember_upload,
    repository_config,
    reset_object_data,
)


def _field_changed() -> None:
    st.session_state.selected_object = None
    reset_object_data()


def _object_changed() -> None:
    reset_object_data()


def _object_selectors() -> None:
    st.subheader("Месторождение и объект")
    if not st.session_state.db_connected:
        st.error("База данных недоступна. Проверьте конфигурацию приложения.")
        return
    try:
        keys = cached_field_objects(repository_config())
    except Exception as error:
        st.session_state.db_connected = False
        st.error(f"Не удалось получить список месторождений: {error}")
        return

    fields = sorted(keys["field"].dropna().astype(str).unique())
    if st.session_state.selected_field not in fields:
        st.session_state.selected_field = None
    st.selectbox(
        "Месторождение",
        options=[None, *fields],
        format_func=lambda value: "Выберите месторождение" if value is None else value,
        key="selected_field",
        on_change=_field_changed,
        persist_state="session",
    )

    objects = []
    if st.session_state.selected_field:
        objects = sorted(
            keys.loc[keys["field"] == st.session_state.selected_field, "objects"]
            .dropna()
            .astype(str)
            .unique()
        )
    if st.session_state.selected_object not in objects:
        st.session_state.selected_object = None
    st.selectbox(
        "Объект",
        options=[None, *objects],
        format_func=lambda value: "Сначала выберите объект" if value is None else value,
        key="selected_object",
        disabled=not objects,
        on_change=_object_changed,
        persist_state="session",
    )

    if st.session_state.selected_field and st.session_state.selected_object:
        try:
            with st.spinner(
                "Проверяем историю, ГРП, PVT и доступные карты…"
            ):
                load_selected_object_data()
        except Exception as error:
            reset_object_data()
            st.error(f"Не удалось прочитать данные выбранного объекта: {error}")


def _history_block() -> None:
    st.subheader("История работы скважин")
    count = st.session_state.history_db_count
    if count:
        st.success(f"В БД найдена история по {count} скважинам")
        options = ["База данных", "Загрузить файл"]
    else:
        st.warning("В БД не найдено ни одной скважины — загрузите файл")
        options = ["Загрузить файл"]
        st.session_state.history_source = "Загрузить файл"
    if st.session_state.history_source not in options:
        st.session_state.history_source = options[0]
    st.radio(
        "Источник истории", options, horizontal=True,
        key="history_source", persist_state="session",
    )
    if st.session_state.history_source == "Загрузить файл":
        upload = st.file_uploader(
            "Файл истории работы скважин",
            type=["xlsx", "xls", "xlsm"],
            key="history_upload",
            on_change=remember_upload,
            args=("history_upload", "history_upload_value"),
        )
        stored = st.session_state.get("history_upload_value")
        if upload is None and stored is None:
            st.error("Файл обязателен для запуска расчёта")
        elif stored is not None:
            st.caption(f"Загружен файл: {stored.name}")


def _map_block(spec, grids, number: int) -> None:
    choices = [grid for grid in grids if grid.map_key == spec.key]
    source_options = (["База данных"] if choices else []) + ["Загрузить файл"]
    if not spec.required:
        source_options.append("Не использовать")
    source_key = f"map_source_{spec.key}"
    if st.session_state[source_key] not in source_options:
        st.session_state[source_key] = "Не использовать" if not spec.required else source_options[0]

    st.markdown(f"#### {number}. {spec.label}")
    st.radio(
        "Источник", source_options, horizontal=True, key=source_key,
        label_visibility="collapsed", persist_state="session",
    )
    if st.session_state[source_key] == "База данных":
        by_locator = {grid.locator: grid for grid in choices}
        choice_key = f"map_choice_{spec.key}"
        if st.session_state.get(choice_key) not in by_locator:
            st.session_state[choice_key] = choices[0].locator if choices else None
        st.selectbox(
            "Найденные карты",
            options=list(by_locator),
            format_func=lambda locator, lookup=by_locator: lookup[locator].display_name,
            key=choice_key,
            persist_state="session",
        )
    elif st.session_state[source_key] == "Загрузить файл":
        widget_key = f"map_upload_{spec.key}"
        storage_key = f"map_upload_value_{spec.key}"
        upload = st.file_uploader(
            f"Файл {spec.filename}", type=["grd"], key=widget_key,
            on_change=remember_upload, args=(widget_key, storage_key),
        )
        stored = st.session_state.get(storage_key)
        if spec.required and upload is None and stored is None:
            st.error("Обязательная карта не загружена")
        elif stored is not None:
            st.caption(f"Загружен файл: {stored.name}")
    else:
        st.caption("Карта будет рассчитана автоматически")


def _maps_block() -> None:
    st.subheader("Карты")
    st.caption(
        "Пять первых карт обязательны. ОИЗ и обводнённость будут рассчитаны моделью, если их не выбрать."
    )
    grids = st.session_state.available_grids
    required_specs = [item for item in MAP_SPECS if item.required]
    for number, spec in enumerate(required_specs, start=1):
        _map_block(spec, grids, number)
    with st.expander("Дополнительные карты"):
        optional_specs = [item for item in MAP_SPECS if not item.required]
        for number, spec in enumerate(optional_specs, start=len(required_specs) + 1):
            _map_block(spec, grids, number)
            if spec != MAP_SPECS[-1]:
                st.divider()


def show():
    initialize_session()
    st.header("Исходные данные")
    st.write(
        "Выберите месторождение и объект. Найденные данные заполнятся автоматически; "
        "их можно заменить своими файлами."
    )
    _object_selectors()
    if not (st.session_state.selected_field and st.session_state.selected_object):
        return
    st.divider()
    _history_block()
    st.divider()
    _maps_block()
