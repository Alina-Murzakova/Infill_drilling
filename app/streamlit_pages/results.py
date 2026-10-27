from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from app.calculation_job import CalculationJob
from app.streamlit_runtime import build_parameters, new_work_directory


def _numeric_total(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return 0.0
    values = pd.to_numeric(frame.loc[frame["Зона"] != "Всего", column], errors="coerce")
    return float(values.fillna(0).sum())


def _start_requested_calculation() -> None:
    if not st.session_state.get("calculation_requested"):
        return
    if st.session_state.get("calculation_job") is not None:
        st.session_state.calculation_requested = False
        return
    st.session_state.calculation_requested = False
    st.session_state.calculation_error = None
    st.session_state.calculation_cancelled = False
    st.session_state.result_summary = None
    st.session_state.result_zip = None
    try:
        with st.spinner("Подготавливаем исходные данные…"):
            parameters = build_parameters(new_work_directory())
        job = CalculationJob(parameters)
        st.session_state.calculation_job = job
        st.session_state.calculation_running = True
        job.start()
    except Exception as error:
        st.session_state.calculation_running = False
        st.session_state.calculation_error = str(error)


@st.fragment(run_every=1)
def _calculation_panel() -> None:
    job = st.session_state.get("calculation_job")
    if job is None:
        return
    snapshot = job.snapshot()
    status = snapshot["status"]
    if status == "completed":
        summary, result_zip, result_zip_name = job.result()
        st.session_state.result_summary = summary
        st.session_state.result_zip = result_zip
        st.session_state.result_zip_name = result_zip_name
        st.session_state.calculation_running = False
        st.session_state.calculation_job = None
        st.session_state.calculation_logs = snapshot["logs"]
        st.rerun(scope="app")
    if status in {"failed", "cancelled"}:
        st.session_state.calculation_running = False
        st.session_state.calculation_error = snapshot["error"] if status == "failed" else None
        st.session_state.calculation_cancelled = status == "cancelled"
        st.session_state.calculation_logs = snapshot["logs"]
        st.session_state.calculation_job = None
        st.rerun(scope="app")

    with st.container(border=True):
        st.subheader("Расчёт выполняется")
        progress = snapshot["progress"]
        st.progress(progress, text=f"Выполнено {progress}%")
        if snapshot["cancel_requested"]:
            st.warning("Отмена запрошена. Текущий шаг будет завершён безопасно.")
        elif st.button("Отменить расчёт", icon=":material/stop_circle:"):
            job.cancel()
            st.rerun(scope="fragment")
        with st.expander("Журнал расчёта", expanded=True):
            logs = snapshot["logs"]
            st.code("\n".join(logs[-100:]) if logs else "Запуск…", language=None, height=260)


def _zone_chart(frame: pd.DataFrame, column: str, title: str, color: str) -> alt.Chart:
    data = frame[["Зона", column]].copy().reset_index(drop=True)
    data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0)
    data["Зона"] = data["Зона"].astype(str)
    return (
        alt.Chart(data)
        .mark_bar(color=color, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Зона:N", sort=None, axis=alt.Axis(title=None, labelAngle=-90, labelFontWeight="bold", labelLimit=180)),
            y=alt.Y(f"{column}:Q", title=title),
            tooltip=["Зона:N", alt.Tooltip(f"{column}:Q", title=title, format=",.1f")],
        )
        .properties(height=360)
    )


def _render_results(summary: pd.DataFrame) -> None:
    zones = int((summary["Зона"] != "Всего").sum()) if "Зона" in summary else 0
    wells = _numeric_total(summary, "Количество\nскважин")
    reserves = _numeric_total(summary, "Запасы, тыс т")
    oil = _numeric_total(summary, "Накопленная добыча\nнефти, тыс.т")
    with st.container(horizontal=True, horizontal_alignment="distribute"):
        st.metric("Перспективных зон", f"{zones}", border=True, icon=":material/grid_view:")
        st.metric("Проектных скважин", f"{wells:,.0f}".replace(",", " "), border=True, icon=":material/oil_barrel:")
        st.metric("Запасы", f"{reserves:,.1f} тыс. т".replace(",", " "), border=True, icon=":material/database:")
        st.metric("Прогноз добычи нефти", f"{oil:,.1f} тыс. т".replace(",", " "), border=True, icon=":material/trending_up:")

    st.subheader("Краткая сводка")
    st.dataframe(summary.fillna("").astype(str).reset_index(drop=True), width="stretch", hide_index=True)
    zone_rows = summary[summary["Зона"] != "Всего"].copy() if "Зона" in summary else pd.DataFrame()
    if not zone_rows.empty:
        st.subheader("Сравнение зон")
        left, right = st.columns(2)
        if "Запасы, тыс т" in zone_rows:
            left.altair_chart(_zone_chart(zone_rows, "Запасы, тыс т", "Запасы, тыс. т", "#246BCE"), width="stretch")
        if "Количество\nскважин" in zone_rows:
            right.altair_chart(_zone_chart(zone_rows, "Количество\nскважин", "Скважины, шт.", "#E06C3B"), width="stretch")
    if st.session_state.get("result_zip"):
        st.download_button(
            "Скачать результаты", data=st.session_state.result_zip,
            file_name=st.session_state.result_zip_name or "Результаты_АВНС.zip", mime="application/zip",
            type="primary", icon=":material/download:", width="stretch",
        )


def show():
    st.header("Результаты")
    _start_requested_calculation()
    if st.session_state.get("calculation_job") is not None:
        _calculation_panel()
    if st.session_state.get("calculation_error"):
        st.error(f"Расчёт завершился с ошибкой: {st.session_state.calculation_error}")
        with st.expander("Журнал расчёта", expanded=True):
            st.code("\n".join(st.session_state.get("calculation_logs", [])[-100:]), language=None)
    elif st.session_state.get("calculation_cancelled"):
        st.warning("Расчёт отменён.")
        with st.expander("Журнал расчёта"):
            st.code("\n".join(st.session_state.get("calculation_logs", [])[-100:]), language=None)
    summary = st.session_state.get("result_summary")
    if summary is None:
        if (
            not st.session_state.get("calculation_running")
            and not st.session_state.get("calculation_error")
            and not st.session_state.get("calculation_cancelled")
        ):
            st.info("Результаты появятся здесь после завершения расчёта.")
        return
    if not isinstance(summary, pd.DataFrame):
        summary = pd.DataFrame(summary)
    _render_results(summary)
