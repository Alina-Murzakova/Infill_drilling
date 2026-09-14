import streamlit as st
from datetime import datetime, date


def show():
    """Страница «Экономика»."""

    st.header("Экономика")

    # -----------------------------------------------------
    # Краткое описание раздела
    # -----------------------------------------------------

    st.markdown(
        '<div class="economy-description">'
        "Настройка параметров расчёта экономических показателей."
        "</div>",
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

        /* Названия параметров */
        .economy-label {
            font-size: 18px;
            line-height: 1.3;
            padding-top: 7px;
        }

        /* Описание */
        .economy-description {
            font-size: 16px;
            line-height: 1.4;
            margin-top: -8px;
            margin-bottom: 20px;
        }

        /* Чекбокс */
        div[data-testid="stCheckbox"] label p {
            font-size: 18px !important;
        }

        div[data-testid="stCheckbox"] {
            margin-bottom: 14px !important;
        }

        /* Дата */
        div[data-testid="stDateInput"] input {
            font-size: 18px !important;
        }

        /* Числовое поле */
        div[data-testid="stNumberInput"] {
            width: 350px !important;
            margin-top: 0 !important;
        }

        div[data-testid="stNumberInput"] input {
            font-size: 18px !important;
        }

        /* Текстовое поле */
        div[data-testid="stTextInput"] {
            margin-top: 0 !important;
        }

        div[data-testid="stTextInput"] input {
            font-size: 18px !important;
        }

        /* Кнопки + / - */
        div[data-testid="stNumberInput"] button {
            background-color: #E4E7EC !important;
        }

        div[data-testid="stNumberInput"] button:hover {
            background-color: #D1D5DB !important;
        }

        div[data-testid="stNumberInput"] button:active {
            background-color: #B8BDC6 !important;
        }

        div[data-testid="stNumberInput"] button:focus {
            background-color: #D1D5DB !important;
        }

        /* Кнопка выбора файла */
        div[data-testid="stButton"] button {
            font-size: 18px !important;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------
    # Значения по умолчанию
    # -----------------------------------------------------

    if "switch_economy" not in st.session_state:
        st.session_state.switch_economy = False

    if "start_date" not in st.session_state:
        # Аналог leStartDate.setDate(QDate(текущий год, 1, 1))
        st.session_state.start_date = date(
            datetime.today().year,
            1,
            1,
        )

    if "day_in_month" not in st.session_state:
        # Аналог leNumDays.setText("29")
        st.session_state.day_in_month = 29.0

    if "path_economy" not in st.session_state:
        st.session_state.path_economy = ""

    # -----------------------------------------------------
    # Расчёт экономических показателей
    # -----------------------------------------------------

    switch_economy = st.checkbox(
        "Расчёт экономических показателей",
        value=st.session_state.switch_economy,
        key="switch_economy",
    )

    # -----------------------------------------------------
    # Экономика
    # -----------------------------------------------------

    col_label, col_value = st.columns(
        [0.5, 1.0],
        gap="small",
    )

    with col_label:
        st.markdown(
            '<div class="economy-label">Экономика</div>',
            unsafe_allow_html=True,
        )

    with col_value:
        path_economy = st.text_input(
            "Экономика",
            value=st.session_state.path_economy,
            placeholder=r"C:\...\economy.xlsx",
            disabled=not switch_economy,
            label_visibility="collapsed",
            key="economy_path",
        )

        # Кнопка выбора файла
        if st.button(
            "Выбрать файл",
            disabled=not switch_economy,
            key="btn_economy",
        ):
            st.info(
                "В Streamlit браузер не позволяет открыть "
                "локальный QFileDialog. Для выбора файла можно "
                "использовать загрузку через st.file_uploader()."
            )

    # -----------------------------------------------------
    # Дата начала расчёта
    # -----------------------------------------------------

    col_label, col_value = st.columns(
        [0.5, 1.0],
        gap="small",
    )

    with col_label:
        st.markdown(
            '<div class="economy-label">'
            "Дата начала расчёта денежного потока"
            "</div>",
            unsafe_allow_html=True,
        )

    with col_value:
        start_date = st.date_input(
            "Дата начала расчёта денежного потока",
            value=st.session_state.start_date,
            disabled=not switch_economy,
            label_visibility="collapsed",
            key="economy_start_date",
        )

    # -----------------------------------------------------
    # Количество дней в месяце
    # -----------------------------------------------------

    col_label, col_value = st.columns(
        [0.5, 1.0],
        gap="small",
    )

    with col_label:
        st.markdown(
            '<div class="economy-label">'
            "Количество дней в месяце"
            "</div>",
            unsafe_allow_html=True,
        )

    with col_value:
        day_in_month = st.number_input(
            "Количество дней в месяце",
            min_value=0.0,
            max_value=31.0,
            value=st.session_state.day_in_month,
            step=0.1,
            format="%.3f",
            disabled=not switch_economy,
            label_visibility="collapsed",
            key="economy_day_in_month",
        )

    # -----------------------------------------------------
    # Данные
    # -----------------------------------------------------

    return {
        "switch_economy": switch_economy,
        "start_date": datetime(
            start_date.year + 1,
            start_date.month,
            start_date.day,
        ),
        "day_in_month": float(day_in_month),
        "path_economy": path_economy,
    }