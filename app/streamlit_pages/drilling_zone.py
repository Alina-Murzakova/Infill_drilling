import streamlit as st

from app.streamlit_widgets import clearable_number_input


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

        col_label, col_value = st.columns([0.5, 1.0], gap="small")

        with col_label:
            st.markdown(
                f'<div class="drilling-label">{label}</div>',
                unsafe_allow_html=True,
            )

        with col_value:
            return clearable_number_input(
                st,
                label,
                key=key,
                min_value=min_value,
                max_value=max_value,
                step=step,
                format=format,
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
        "percent_top": int(percent_top) if percent_top is not None else None,
        "min_radius": int(min_radius) if min_radius is not None else None,
        "sensitivity_quality_drill": int(sensitivity_quality_drill) if sensitivity_quality_drill is not None else None,
        "init_profit_cum_oil": int(init_profit_cum_oil) if init_profit_cum_oil is not None else None,
    }
