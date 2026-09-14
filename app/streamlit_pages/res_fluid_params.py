import streamlit as st


def show():
    """Страница «Свойства пласта и флюидов»."""

    st.header("Свойства пласта и флюидов")

    # Краткое описание раздела
    st.markdown(
        '<div class="resfluid-description">'
        "Настройка свойств пласта и флюидов, используемых при выполнении расчёта. "
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

        /* Ширина поля со значением */
        div[data-testid="stNumberInput"] {
            width: 350px !important;
        }

        /* Названия параметров */
        .resfluid-label {
            font-size: 18px;
            line-height: 1.3;
            padding-top: 7px;
        }

        /* Краткое описание раздела */
        .resfluid-description {
            font-size: 16px;
            line-height: 1.4;
            margin-top: -8px;
            margin-bottom: 20px;
        }

        /* Чекбоксы */
        div[data-testid="stCheckbox"] label p {
            font-size: 18px !important;
        }

        /* Отступ после чекбокса */
        div[data-testid="stCheckbox"] {
            margin-bottom: 14px !important;
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
        disabled=False,
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
                f'<div class="resfluid-label">{label}</div>',
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
                disabled=disabled,
                label_visibility="collapsed",
                key=key,
            )

    # -----------------------------------------------------
    # Учет относительной проницаемости
    # -----------------------------------------------------

    switch_adaptation_relative_permeability = st.checkbox(
        "Автоадаптация ОФП",
        value=True,
        key="resfluid_relative_permeability",
    )

    # -----------------------------------------------------
    # Остаточная нефтенасыщенность
    # -----------------------------------------------------

    Sor = number_parameter(
        label="Остаточная нефтенасыщенность, д.ед.",
        value=0.2,
        min_value=0.0,
        max_value=1.0,
        step=0.001,
        format="%.3f",
        key="resfluid_Sor",
        disabled=not switch_adaptation_relative_permeability,
    )

    # -----------------------------------------------------
    # Остаточная водонасыщенность
    # -----------------------------------------------------

    Swc = number_parameter(
        label="Связанная водонасыщенность, д.ед.",
        value=0.2,
        min_value=0.0,
        max_value=1.0,
        step=0.001,
        format="%.3f",
        key="resfluid_Swc",
        disabled=not switch_adaptation_relative_permeability,
    )

    # -----------------------------------------------------
    # Относительная проницаемость нефти
    # -----------------------------------------------------

    Fo = number_parameter(
        label="Концевая точка ОФП по нефти",
        value=0.8,
        min_value=0.0,
        max_value=1.0,
        step=0.001,
        format="%.3f",
        key="resfluid_Fo",
        disabled=not switch_adaptation_relative_permeability,
    )

    # -----------------------------------------------------
    # Относительная проницаемость воды
    # -----------------------------------------------------

    Fw = number_parameter(
        label="Концевая точка ОФП по воде",
        value=0.8,
        min_value=0.0,
        max_value=1.0,
        step=0.001,
        format="%.3f",
        key="resfluid_Fw",
        disabled=not switch_adaptation_relative_permeability,
    )

    # -----------------------------------------------------
    # Показатель степени Corey для нефти
    # -----------------------------------------------------

    m2 = number_parameter(
        label="Степень Corey по нефти",
        value=2.0,
        min_value=0.0,
        max_value=10.0,
        step=0.001,
        format="%.3f",
        key="resfluid_m2",
        disabled=not switch_adaptation_relative_permeability,
    )

    # -----------------------------------------------------
    # Показатель степени Corey для воды
    # -----------------------------------------------------

    m1 = number_parameter(
        label="Степень Corey по воде",
        value=2.0,
        min_value=0.0,
        max_value=10.0,
        step=0.001,
        format="%.3f",
        key="resfluid_m1",
        disabled=not switch_adaptation_relative_permeability,
    )

    # -----------------------------------------------------
    # Объемный коэффициент воды
    # -----------------------------------------------------

    Bw = number_parameter(
        label="Объёмный коэффициент воды, м³/м³",
        value=1.0,
        min_value=0.0,
        max_value=1.5,
        step=0.001,
        format="%.3f",
        key="resfluid_Bw",
    )

    # -----------------------------------------------------
    # Коэффициент анизотропии
    # -----------------------------------------------------

    kv_kh = number_parameter(
        label="Анизотропия пласта, д.ед.",
        value=1.0,
        min_value=0.0,
        max_value=1.5,
        step=0.001,
        format="%.3f",
        key="resfluid_kv_kh",
    )

    # -----------------------------------------------------
    # Данные для модели
    # -----------------------------------------------------

    return {
        "switch_adaptation_relative_permeability":
            switch_adaptation_relative_permeability,
        "Sor": float(Sor),
        "Swc": float(Swc),
        "Fo": float(Fo),
        "Fw": float(Fw),
        "m2": float(m2),
        "m1": float(m1),
        "Bw": float(Bw),
        "kv_kh": float(kv_kh),
    }