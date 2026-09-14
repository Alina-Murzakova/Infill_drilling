import streamlit as st


def show():
    """Страница «Картопостроение»"""

    st.header("Картопостроение")

    # Краткое описание раздела
    st.markdown(
        '<div class="mapping-description">'
        'Настройка параметров координатной сетки, интерполяции и учета '
        'особенностей горизонтальных скважин и АГРП при построении карт.'
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
        .mapping-label {
            font-size: 18px;
            line-height: 1.3;
            padding-top: 7px;
        }
        
        /* Краткое описание раздела */
        .mapping-description {
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
        
        /* Настройка цветов */     
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
                f'<div class="mapping-label">{label}</div>',
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
    # Размер ячейки
    # -----------------------------------------------------

    size_pixel = number_parameter(
        label="Размер ячейки, м",
        value=50,
        min_value=1,
        max_value=999999,
        step=1,
        key="mapping_size_pixel",
    )

    # -----------------------------------------------------
    # Радиус интерполяции
    # -----------------------------------------------------

    radius_interpolate = number_parameter(
        label="Радиус интерполяции для дискретных карт, м",
        value=1000,
        min_value=1,
        max_value=999999,
        step=1,
        key="mapping_radius_interpolate",
    )

    # -----------------------------------------------------
    # Учет траекторий ГС
    # -----------------------------------------------------

    switch_accounting_horwell = st.checkbox(
        "Учёт траекторий ГС",
        value=True,
        key="mapping_accounting_horwell",
    )

    # -----------------------------------------------------
    # Учет АГРП
    # -----------------------------------------------------

    switch_frac_inj_well = st.checkbox(
        "Учёт АГРП на нагнетательных скважинах на карте ОИЗ",
        value=False,
        key="mapping_auto_frac",
    )

    # -----------------------------------------------------
    # Азимут минимального горизонтального напряжения
    # -----------------------------------------------------

    azimuth_sigma_h_min = number_parameter(
        label="Азимут минимального горизонтального напряжения, град.",
        value=45,
        min_value=0,
        max_value=360,
        step=1,
        key="mapping_min_hor_stress",
        disabled=not switch_frac_inj_well,
    )

    # -----------------------------------------------------
    # Полудлина трещины
    # -----------------------------------------------------

    l_half_fracture = number_parameter(
        label="Полудлина трещины АГРП, м",
        value=150.0,
        min_value=0.0,
        max_value=1500.0,
        step=0.1,
        key="mapping_frac_half_length",
        disabled=not switch_frac_inj_well,
    )

    # -----------------------------------------------------
    # КИН
    # -----------------------------------------------------

    KIN = number_parameter(
        label="КИН, д.ед.",
        value=0.0,
        min_value=0.0,
        max_value=1.0,
        step=0.001,
        format="%.3f",
        key="mapping_KIN",
    )

    # -----------------------------------------------------
    # Данные для модели
    # -----------------------------------------------------

    return {
        "default_size_pixel": int(size_pixel),
        "radius_interpolate": int(radius_interpolate),
        "switch_accounting_horwell": switch_accounting_horwell,
        "switch_frac_inj_well": switch_frac_inj_well,
        "azimuth_sigma_h_min": int(azimuth_sigma_h_min),
        "l_half_fracture": float(l_half_fracture),
        "KIN": float(KIN),
    }