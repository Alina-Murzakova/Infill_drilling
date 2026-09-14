import streamlit as st

from app.input_output.output_functions import get_save_path
from app.version import APP_NAME


def show():

    st.header("Исходные данные")

    st.markdown(
        "Загрузка / подключение основных исходных данных, необходимых для выполнения расчёта, "
        "включая историю работы скважин (МЭР, ТР), геологические карты и карты разработки, "
        "а также геолого-физические характеристики объекта разработки.<br>"
        "Путь для сохранения результатов расчета.",
        unsafe_allow_html=True,
    )
    # -----------------------------------------------------
    # Стили
    # -----------------------------------------------------
    st.markdown(
        """
        <style>
        /* Отступ между строками параметров */
        div[data-testid="stHorizontalBlock"] {
            margin-bottom: 14px;
        }

        /* Названия параметров text_input */
        div[data-testid="stTextInput"] label p {
            font-size: 18px !important;
        }

        /* Краткое описание раздела */
        .input-description {
            font-size: 16px;
            line-height: 1.4;
            margin-top: -8px;
            margin-bottom: 8px;
        }

        /* Поля ввода */
        div[data-testid="stTextInput"] input {
            font-size: 18px !important;
        }

        /* Убираем лишние отступы вокруг поля */
        div[data-testid="stTextInput"] {
            margin-top: 0 !important;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )

    # st.divider()

    # Значение по умолчанию для папки сохранения
    default_save_path = get_save_path(APP_NAME)

    # Инициализируем значения один раз
    if "data_well_directory" not in st.session_state:
        st.session_state.data_well_directory = ""

    if "maps_directory" not in st.session_state:
        st.session_state.maps_directory = ""

    if "path_geo_phys_properties" not in st.session_state:
        st.session_state.path_geo_phys_properties = ""

    if "save_directory" not in st.session_state:
        st.session_state.save_directory = default_save_path

    # История работы скважин
    st.session_state.data_well_directory = st.text_input(
        "История работы скважин",
        value=st.session_state.data_well_directory,
        placeholder=r"C:\...\history.xlsx",
    )

    # Карты
    st.session_state.maps_directory = st.text_input(
        "Карты",
        value=st.session_state.maps_directory,
        placeholder=r"C:\...\maps",
    )

    # Геолого-физические свойства
    st.session_state.path_geo_phys_properties = st.text_input(
        "Геолого-физические свойства",
        value=st.session_state.path_geo_phys_properties,
        placeholder=r"C:\...\properties.xlsx",
    )

    # Папка сохранения
    st.session_state.save_directory = st.text_input(
        "Путь для сохранения результатов расчёта",
        value=st.session_state.save_directory,
        placeholder=r"C:\...\results",
    )

    # st.divider()

    # -----------------------------------------------------
    # Данные для расчёта
    # -----------------------------------------------------

    return {
        "data_well_directory": st.session_state.data_well_directory,
        "maps_directory": st.session_state.maps_directory,
        "path_geo_phys_properties": st.session_state.path_geo_phys_properties,
        "save_directory": st.session_state.save_directory,
    }