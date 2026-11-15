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
from app.streamlit_runtime import (initialize_session, validation_issues)

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


def load_css():
    """Загрузка стилей вкладок"""
    css_path = Path(__file__).resolve().parent / "streamlit_pages" / "styles.css"

    with open(css_path, encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True,
        )


# Настройки страницы
st.set_page_config(page_title=APP_NAME,
                   page_icon=lbl_path,
                   layout="wide",
                   )

load_css()
initialize_session()

issues = validation_issues()
incomplete_sections = {issue.section for issue in issues}

# Стили
st.markdown(
    """
    <style>
    /* Ширина боковой панели */
    section[data-testid="stSidebar"] {
        width: 380px !important;
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
        font-size: 18px !important;
    }
    
    section[data-testid="stSidebar"] button [data-testid="stIconMaterial"] {
        font-size: 22px !important;
    }

    /* Наведение */
    section[data-testid="stSidebar"] button:hover {
        background-color: #E0E4EC !important;
        color: #000000 !important;
    }

    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding-bottom: 5rem;
    }

    section[data-testid="stSidebar"] .st-key-about_bottom {
        positiaddon: fixed;
        bottom: 0.75rem;
        width: 325px;
        background: var(--background-color);
        z-index: 10;
    }
    
    /* Активный раздел меню */
    div[class*="st-key-menu_"] button[kind="primary"] {
        background-color: #E0E4EC !important;
        color: #000000 !important;
        border: none !important;
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
    st.session_state.current_section = "О программе"


def select_section(name):
    st.session_state.current_section = name


# ===== Боковое меню =====
with st.sidebar:
    st.markdown(
        f"""
        <div style="margin-bottom: 20px;">
            <div style="font-size: 28px; font-weight: 700;">
                {APP_NAME}
            </div>
            <div style="font-size: 16px; color: #808080;">
                Версия {APP_VERSION}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Основные разделы
    for icon, name in sections:
        is_active = st.session_state.current_section == name

        with st.container(key=f"menu_{name.replace(' ', '_')}"):
            st.button(
                name,
                icon=icon,
                key=f"section_{name}",
                width="stretch",
                type="primary" if is_active else "tertiary",  # if name in incomplete_sections
                on_click=select_section,
                args=(name,),
            )

    # Раздел "О программе"
    with st.container(key="about_bottom"):
        st.button(
            "О программе",
            icon=":material/info:",
            key="about",
            width="stretch",
            type="tertiary",
            on_click=select_section,
            args=("О программе",),
        )

    # горизонтальная разделительная линия
    st.divider()

    # Статус заполнения
    if issues:
        st.warning(
            f"Необходимо заполнить все обязательные данные.\n",
            icon=":material/warning:"
        )
    else:
        st.success(
            "Все обязательные данные заполнены",
            icon=":material/check_circle:"
        )

    # Кнопка запуска расчёта
    run_clicked = st.button(
        "Запустить расчёт",
        type="primary",
        icon=":material/play_arrow:",
        width="stretch",
        disabled=bool(issues) or st.session_state.calculation_running,
    )

    if run_clicked:
        st.session_state.calculation_requested = True
        st.session_state.current_section = "Результаты"
        st.rerun()

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
    st.session_state.current_section = "О программе"
pages[st.session_state.current_section]()
current_section = st.session_state.current_section
