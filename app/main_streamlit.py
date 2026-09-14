import sys
import os
import streamlit as st

from pathlib import Path

# Корень проекта: папка, внутри которой находится app
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pathlib import Path
from version import APP_NAME, APP_VERSION
from streamlit_pages.initial_data import show as show_initial_data
from streamlit_pages.about_program import show as show_about_program
from streamlit_pages.mapping import show as show_mapping
from streamlit_pages.drilling_zone import show as show_drilling_zone
from streamlit_pages.res_fluid_params import show as show_res_fluid_params
from streamlit_pages.well_params import show as show_well_params
from streamlit_pages.economy import show as show_economy


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

st.title(APP_NAME)
st.caption(f"Версия {APP_VERSION}")


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
        border: none !important;
        background: transparent !important;
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
    (":material/money_bag:", "Экономика"),
    (":material/bar_chart:", "Результаты"),
]

# ===== Состояние выбранного раздела =====
if "current_section" not in st.session_state:
    st.session_state.current_section = "О программе"

# ===== Боковое меню =====
with st.sidebar:

    # О программе
    if st.button(
            "О программе",
            icon=":material/info:",
            key="about",
            use_container_width=True,
    ):
        st.session_state.current_section = "О программе"

    # Разделы
    st.markdown("### Разделы")

    for icon, name in sections:
        if st.button(
                name,
                icon=icon,
                key=f"section_{name}",
                use_container_width=True,
        ):
            st.session_state.current_section = name

# ===== Текущий раздел =====
current_section = st.session_state.current_section

# if st.session_state.current_section == "О программе":
#     st.header("О программе")
#     st.write(f"{APP_NAME} v{APP_VERSION}")
# else:
#     st.header(st.session_state.current_section)
#
# st.write(
#     f"Здесь будет раздел «{current_section}»."
# )

pages = {
    "О программе": show_about_program,
    "Исходные данные": show_initial_data,
    "Картопостроение": show_mapping,
    "Поиск зон бурения": show_drilling_zone,
    "Свойства пласта и флюидов": show_res_fluid_params,
    "Параметры скважин": show_well_params,
    "Экономика": show_economy,
}
pages[st.session_state.current_section]()