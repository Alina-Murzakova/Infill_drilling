import streamlit as st


def show():
    """Страница «Поиск зон бурения»"""

    st.header("Параметры перспективных зон")

    # Краткое описание раздела
    st.markdown(
        '<div class="drilling-description">'
        'Настройка параметров поиска перспективных зон и определения количества скважин. '
        '</div>',
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

        /* Ширина поля со значением */
        div[data-testid="stNumberInput"] {
            width: 350px !important;
        }

        /* Названия параметров */
        .drilling-label {
            font-size: 18px;
            line-height: 1.3;
            padding-top: 7px;
        }

        /* Краткое описание раздела */
        .drilling-description {
            font-size: 16px;
            line-height: 1.4;
            margin-top: -8px;
            margin-bottom: 20px;
        }

        /* Поля ввода */
        div[data-testid="stNumberInput"] input {
            font-size: 18px !important;
        }

        /* Немного уменьшаем внутренние отступы вокруг полей */
        div[data-testid="stNumberInput"] {
            margin-top: 0 !important;
        }

        /* Кнопки + / - */
        div[data-testid="stNumberInput"] button {
            background-color: #E4E7EC !important;
        }

        /* + / - при наведении */
        div[data-testid="stNumberInput"] button:hover {
            background-color: #D1D5DB !important;
        }

        /* + / - при нажатии */
        div[data-testid="stNumberInput"] button:active {
            background-color: #B8BDC6 !important;
        }

        /* + / - после нажатия */
        div[data-testid="stNumberInput"] button:focus {
            background-color: #D1D5DB !important;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------
    # Вспомогательная функция для параметров
    # -----------------------------------------------------

    def number_parameter(
        label,
        value,
        min_value,
        max_value,
        step,
        key,
        format=None,
    ):
        """
        Строка параметра:
        название слева + поле значения справа.
        """

        col_label, col_value = st.columns(
            [0.5, 1.0],
            gap="small",
        )

        with col_label:
            st.markdown(
                f'<div class="drilling-label">{label}</div>',
                unsafe_allow_html=True,
            )

        with col_value:
            return st.number_input(
                label,
                min_value=min_value,
                max_value=max_value,
                value=value,
                step=step,
                format=format,
                label_visibility="collapsed",
                key=key,
            )

    # -----------------------------------------------------
    # Процент лучших зон
    # -----------------------------------------------------

    percent_top = number_parameter(
        label="Процент лучших точек для кластеризации, %",
        value=10,
        min_value=0,
        max_value=100,
        step=1,
        key="drilling_percent_top",
    )

    # -----------------------------------------------------
    # Минимальный радиус
    # -----------------------------------------------------

    min_radius = number_parameter(
        label="Минимальный радиус зоны для бурения, м",
        value=100,
        min_value=1,
        max_value=999999,
        step=1,
        key="drilling_min_radius",
    )

    # -----------------------------------------------------
    # Чувствительность качества бурения
    # -----------------------------------------------------

    sensitivity_quality_drill = number_parameter(
        label="Чувствительность к качеству зоны бурения, %",
        value=50,
        min_value=0,
        max_value=100,
        step=1,
        key="drilling_sensitivity_quality",
    )

    # -----------------------------------------------------
    # Начальная прибыль по накопленной нефти
    # -----------------------------------------------------

    init_profit_cum_oil = number_parameter(
        label="Минимальные запасы на скважину, тыс.т",
        value=100,
        min_value=1,
        max_value=999999,
        step=1,
        key="drilling_init_profit_cum_oil",
    )

    # -----------------------------------------------------
    # Данные для расчёта
    # -----------------------------------------------------

    return {
        "percent_top": int(percent_top),
        "min_radius": int(min_radius),
        "sensitivity_quality_drill": int(sensitivity_quality_drill),
        "init_profit_cum_oil": int(init_profit_cum_oil),
    }