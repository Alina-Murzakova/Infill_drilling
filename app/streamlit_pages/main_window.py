import sys
import os
import streamlit as st
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
lbl_path = BASE_DIR / "_internal" / "icons" / "lbl.ico"

# Настройки страницы
st.set_page_config(page_title="АВНС",
                   page_icon=lbl_path,
                   layout="wide",
                   )

# Стили
st.markdown(
    """
    <style>
        /* Общий фон */
        .stApp {
            background-color: #f3f5f6;
        }

        /* Заголовок */
        .app-title {
            font-size: 28px;
            font-weight: 600;
            color: #222222;
            margin-bottom: 5px;
        }

        .app-subtitle {
            font-size: 16px;
            color: #666666;
            margin-bottom: 30px;
        }

        /* Заголовок sidebar */
        [data-testid="stSidebar"] {
            background-color: #f3f5f6;
        }

        /* Кнопки навигации */
        [data-testid="stSidebar"] .stButton button {
            width: 100%;
            text-align: left;
            border: none;
            background-color: transparent;
            padding: 10px 12px;
            border-radius: 7px;
            font-size: 15px;
        }

        [data-testid="stSidebar"] .stButton button:hover {
            background-color: #e6eaed;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Разделы приложения
sections = [
    ("📁", "Исходные данные"),
    ("▦", "Картопостроение"),
    ("📍", "Поиск зон бурения"),
    ("🛢️", "Параметры скважин"),
    ("💧", "Свойства пласта и флюидов"),
    ("💰", "Экономика"),
    ("▥", "Результаты"),
]

# Состояние выбранного раздела
if "selected_section" not in st.session_state:
    st.session_state.selected_section = "Исходные данные"

# Sidebar
with st.sidebar:
    st.markdown(
        '<div class="app-title">АВНС</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="app-subtitle">Автоматизированный выбор зон бурения</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    for icon, name in sections:
        if st.button(
                f"{icon}  {name}",
                key=f"section_{name}",
        ):
            st.session_state.selected_section = name
            st.rerun()

# Основная область
st.title(st.session_state.selected_section)


# Отображение выбранного раздела
if st.session_state.selected_section == "Исходные данные":

    from app.streamlit_pages.initial_data import show

    show()

else:

    st.title(st.session_state.selected_section)

    st.info("Этот раздел пока находится в разработке.")

# st.info(
#     f"Сейчас выбран раздел: **{st.session_state.selected_section}**"
# )
