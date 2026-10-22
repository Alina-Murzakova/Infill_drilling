import sys
import os
import streamlit as st

from pathlib import Path

# Корень проекта: папка, внутри которой находится app
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from version import APP_NAME, APP_VERSION
from streamlit_pages.initial_data import show as show_initial_data
from streamlit_pages.about_program import show as show_about_program
from streamlit_pages.mapping import show as show_mapping
from streamlit_pages.drilling_zone import show as show_drilling_zone
from streamlit_pages.res_fluid_params import show as show_res_fluid_params
from streamlit_pages.well_params import show as show_well_params
from streamlit_pages.results import show as show_results
from app.streamlit_runtime import (
    initialize_session,
    validation_issues,
)


# Иконка приложения
icons = [
    "bi--folder-plus.png",
    "bi--layers-half.png",
    "ep--map-location.png",
    "drilling-rig.png",
    "water-drop (1).png",
    "free-icon-dollars-money-bag-50117.png",
    "ep--histogram.png"]

lbl = "lbl.ico"


def resource_path(relative_path: str) -> str:
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    else:
        # В IDE: убираем первый 'app/' из relative_path
        if relative_path.startswith("app/"):
            relative_path = relative_path[len("app/"):]
        base_path = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_path, relative_path)


base_path_icon = resource_path("app/_internal/icons")
lbl_path = os.path.join(base_path_icon, lbl)

# BASE_DIR = Path(__file__).resolve().parent.parent
# lbl_path = BASE_DIR / "_internal" / "icons" / "lbl.ico"

# Настройки страницы
st.set_page_config(page_title=APP_NAME,
                   page_icon=lbl_path,
                   layout="wide",
                   )

initialize_session()


# Стили
st.markdown(
    """
    <style>
    /* Ширина боковой панели */
    section[data-testid="stSidebar"] {
        width: 350px !important;
    }

    /* Сама кнопка */
    section[data-testid="stSidebar"] button {
        width: 100% !important;
        display: flex !important;
        justify-content: flex-start !important;
        align-items: center !important;
        text-align: left !important;
        padding: 8px 10px !important;
    }
    
    /* Все контейнеры внутри кнопки */
    section[data-testid="stSidebar"] button > div {
        width: 100% !important;
        display: flex !important;
        justify-content: flex-start !important;
        align-items: center !important;
        text-align: left !important;
    }
    
    /* Текст */
    section[data-testid="stSidebar"] button p {
        width: 100% !important;
        margin: 0 !important;
        text-align: left !important;
        font-size: 20px !important;
    }
    
    section[data-testid="stSidebar"] button [data-testid="stIconMaterial"] {
        font-size: 22px !important;
    }

    /* Наведение */
    section[data-testid="stSidebar"] button:hover {
        background-color: rgba(128, 128, 128, 0.15) !important;
    }

    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding-bottom: 5rem;
    }

    section[data-testid="stSidebar"] .st-key-about_bottom {
        position: fixed;
        left: 1rem;
        bottom: 0.75rem;
        width: 318px;
        background: var(--background-color);
        z-index: 10;
    }

    
    </style>
    """,
    unsafe_allow_html=True)

# Разделы приложения
sections = [
    (":material/folder_open:", "Исходные данные"),
    (":material/stacks:", "Картопостроение"),
    (":material/distance:", "Поиск зон бурения"),
    (":material/oil_barrel:", "Параметры скважин"),
    (":material/water_drop:", "Свойства пласта и флюидов"),
    (":material/bar_chart:", "Результаты"),
]

# ===== Состояние выбранного раздела =====
if "current_section" not in st.session_state:
    st.session_state.current_section = "Исходные данные"

# ===== Боковое меню =====
with st.sidebar:
    st.markdown(f"### {APP_NAME}\nВерсия {APP_VERSION}")
    incomplete_sections = {issue.section for issue in validation_issues()}

    for icon, name in sections:
        if st.button(
                name,
                icon=":material/error:" if name in incomplete_sections else icon,
                key=f"section_{name}",
                width="stretch",
                type="primary" if name in incomplete_sections else "tertiary",
        ):
            st.session_state.current_section = name
    with st.container(key="about_bottom"):
        if st.button(
                "О программе",
                icon=":material/info:",
                key="about",
                width="stretch",
                type="tertiary",
        ):
            st.session_state.current_section = "О программе"

# ===== Текущий раздел =====
current_section = st.session_state.current_section

pages = {
    "О программе": show_about_program,
    "Исходные данные": show_initial_data,
    "Картопостроение": show_mapping,
    "Поиск зон бурения": show_drilling_zone,
    "Свойства пласта и флюидов": show_res_fluid_params,
    "Параметры скважин": show_well_params,
    "Результаты": show_results,
}
if st.session_state.current_section not in pages:
    st.session_state.current_section = "Исходные данные"
pages[st.session_state.current_section]()
current_section = st.session_state.current_section


if current_section not in {"О программе", "Результаты"}:
    issues = validation_issues()
    with st.bottom:
        left, right = st.columns([1, 3], vertical_alignment="center")
        run_clicked = left.button(
            "Запустить расчёт",
            type="primary",
            icon=":material/play_arrow:",
            width="stretch",
            disabled=bool(issues) or st.session_state.calculation_running,
        )
        if issues:
            right.caption(f"Заполните обязательные данные. Незаполненные разделы подсвечены в меню ({len(issues)}).")
        else:
            right.success("Все обязательные данные заполнены", icon=":material/check_circle:")
    if run_clicked:
        st.session_state.calculation_requested = True
        st.session_state.current_section = "Результаты"
        st.rerun()
