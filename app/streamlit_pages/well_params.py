import streamlit as st


def show():
    """Страница «Параметры скважин»."""

    st.header("Параметры скважин")

    # Краткое описание раздела
    st.markdown(
        '<div class="well-description">'
        "Настройка параметров скважин, параметров ГРП, фактического и проектного фонда скважин "
        "для выполнения расчёта."
        "</div>",
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------
    # Стили
    # -----------------------------------------------------

    st.markdown(
        """
        <style>

        /* Названия вкладок */
        div[data-testid="stTabs"] [role="tab"] {
            font-size: 22px !important; 
            font-weight: 500 !important; 
        }
        
        /* На случай, если Streamlit рендерит текст внутри ссылки/дополнительного тега */
        div[data-testid="stTabs"] [role="tab"] * {
            font-size: 22px !important;
            font-weight: 500 !important;
        }
        
        /* Корректируем высоту панели, чтобы крупные вкладки не обрезались */
        div[data-testid="stTabs"] [role="tablist"] {
            height: auto !important;
            gap: 20px !important;
        }

        /* Отступ между строками параметров */
        div[data-testid="stHorizontalBlock"] {
            margin-bottom: 14px;
        }

        /* Ширина поля со значением */
        div[data-testid="stNumberInput"] {
            width: 350px !important;
        }

        /* Названия параметров */
        .well-label {
            font-size: 18px;
            line-height: 1.3;
            padding-top: 7px;
        }

        /* Краткое описание раздела */
        .well-description {
            font-size: 16px;
            line-height: 1.4;
            margin-top: -8px;
            margin-bottom: 20px;
        }

        /* Чекбоксы */
        div[data-testid="stCheckbox"] label p {
            font-size: 18px !important;
        }

        div[data-testid="stCheckbox"] {
            margin-bottom: 14px !important;
        }

        /* Радиокнопки */
        div[data-testid="stRadio"] label p {
            font-size: 18px !important;
        }

        /* Поля ввода */
        div[data-testid="stNumberInput"] input {
            font-size: 18px !important;
        }

        div[data-testid="stNumberInput"] {
            margin-top: 0 !important;
        }

        /* Подпись поля "Файл параметров ГРП" */
        div[data-testid="stTextInput"] label p {
            font-size: 18px !important;
        }

        /* Текст внутри поля */
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

        </style>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------
    # Вспомогательная функция для числовых параметров
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
        col_label, col_value = st.columns(
            [0.5, 1.0],
            gap="small",
        )

        with col_label:
            st.markdown(
                f'<div class="well-label">{label}</div>',
                unsafe_allow_html=True,
            )

        with col_value:
            if isinstance(value, float):
                min_value = float(min_value)
                max_value = float(max_value)
                step = float(step)
            else:
                min_value = int(min_value)
                max_value = int(max_value)
                step = int(step)

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

    # =====================================================
    # ВКЛАДКИ
    # =====================================================

    tab_common, tab_frac, tab_fact, tab_project = st.tabs(
        [
            "Общие",
            "ГРП",
            "Фактический фонд",
            "Проектный фонд",
        ]
    )

    # =====================================================
    # ОБЩИЕ
    # =====================================================

    with tab_common:

        r_w = number_parameter(
            label="Радиус скважины, м",
            value=0.1,
            min_value=0.0,
            max_value=1.0,
            step=0.001,
            format="%.3f",
            key="well_r_w",
        )

        t_p = number_parameter(
            label="Время на ВНР, сут.",
            value=1,
            min_value=1,
            max_value=999999,
            step=1,
            key="well_t_p",
        )

        well_efficiency = number_parameter(
            label="Коэффициент эксплуатации, д.ед.",
            value=1.0,
            min_value=0.0,
            max_value=1.0,
            step=0.001,
            format="%.3f",
            key="well_efficiency",
        )

        KUBS = number_parameter(
            label="Коэффициент успешности, д.ед.",
            value=1.0,
            min_value=0.0,
            max_value=1.0,
            step=0.001,
            format="%.3f",
            key="well_KUBS",
        )

        KPPP = number_parameter(
            label="Коэффициент для ГС, д.ед.",
            value=1.0,
            min_value=0.0,
            max_value=1.0,
            step=0.001,
            format="%.3f",
            key="well_KPPP",
        )

        skin = number_parameter(
            label="Скин кольматации",
            value=0.0,
            min_value=-99.0,
            max_value=99.0,
            step=0.001,
            format="%.3f",
            key="well_skin",
        )

    # =====================================================
    # ГРП
    # =====================================================

    with tab_frac:

        frac_type = st.radio(
            "Тип ГРП",
            options=[
                "Без ГРП",
                "ГРП",
                "МГРП",
            ],
            horizontal=True,
            index=0,
            key="well_frac_type",
        )

        use_frac_sheet = st.checkbox(
            "Параметры ГРП из фрак-листов",
            value=False,
            key="well_use_frac_sheet",
        )

        frac_file = st.text_input(
            "Файл параметров ГРП",
            value="",
            placeholder=r"C:\...\frac.xlsx",
            disabled=not use_frac_sheet,
            key="well_frac_file",
        )

        length_FracStage = number_parameter(
            label="Длина стадии ГРП, м",
            value=100,
            min_value=1,
            max_value=999999,
            step=1,
            key="well_length_frac_stage",
            disabled=use_frac_sheet,
        )

        xfr = number_parameter(
            label="Полудлина трещины, м",
            value=100,
            min_value=1,
            max_value=999999,
            step=1,
            key="well_xfr",
            disabled=use_frac_sheet,
        )

        w_f = number_parameter(
            label="Ширина трещины, мм",
            value=1,
            min_value=1,
            max_value=999999,
            step=1,
            key="well_w_f",
            disabled=use_frac_sheet,
        )

        k_f = number_parameter(
            label="Проницаемость проппанта, Д",
            value=100,
            min_value=1,
            max_value=999999,
            step=1,
            key="well_k_f",
            disabled=use_frac_sheet,
        )

    # =====================================================
    # ФАКТИЧЕСКИЙ ФОНД
    # =====================================================

    with tab_fact:

        first_months = number_parameter(
            label="Расчёт запускных параметров, как среднее за (мес.)",
            value=1,
            min_value=1,
            max_value=999999,
            step=1,
            key="well_first_months",
        )

        last_months = number_parameter(
            label="Расчёт последних параметров, как среднее за (мес.)",
            value=12,
            min_value=1,
            max_value=999999,
            step=1,
            key="well_last_months",
        )

        default_radius_prod = number_parameter(
            label="Минимальный эффективный радиус добывающих скважин, м",
            value=0,
            min_value=0,
            max_value=999999,
            step=1,
            key="well_default_radius_prod",
        )

        default_radius_inj = number_parameter(
            label="Минимальный эффективный радиус нагнетательных скважин, м",
            value=0,
            min_value=0,
            max_value=999999,
            step=1,
            key="well_default_radius_inj",
        )

        switch_filtration_perm_fact = st.checkbox(
            "Фильтрация выбросов проницаемости через РБ",
            value=False,
            key="well_filtration_perm_fact",
        )

    # =====================================================
    # ПРОЕКТНЫЙ ФОНД
    # =====================================================

    with tab_project:

        L = number_parameter(
            label="Максимальная длина скважины, м",
            value=0,
            min_value=0,
            max_value=999999,
            step=1,
            key="well_L",
        )

        min_length = number_parameter(
            label="Минимальная длина горизонтального участка, м",
            value=0,
            min_value=0,
            max_value=999999,
            step=1,
            key="well_min_length",
        )

        buffer_project_wells = number_parameter(
            label="Условный эффективный радиус проектной скважины, м",
            value=0,
            min_value=0,
            max_value=999999,
            step=1,
            key="well_buffer_project_wells",
        )

        switch_fix_P_well_init = st.checkbox(
            "Зафиксировать забойное давление",
            value=False,
            key="well_fix_P_well_init",
        )

        fix_P_well_init = number_parameter(
            label="Ожидаемое Рзаб, атм",
            value=0.0,
            min_value=0.0,
            max_value=999999,
            step=0.1,
            format="%.1f",
            key="well_P_well_init",
            disabled=not switch_fix_P_well_init,
        )

        # Источник обводнённости
        water_cut_source = st.radio(
            "Обводненность",
            options=[
                "С карты",
                "С окружения",
            ],
            index=0,
            key="well_water_cut_source",
        )

        k = number_parameter(
            label="Количество ближайших фактических скважин, шт",
            value=1,
            min_value=1,
            max_value=999999,
            step=1,
            key="well_k",
        )

        threshold = number_parameter(
            label="Порог для фильтрации ближайших скважин, м",
            value=0,
            min_value=0,
            max_value=999999,
            step=1,
            key="well_threshold",
        )

        period_calculation = number_parameter(
            label="Период прогноза, года",
            value=0,
            min_value=0,
            max_value=999999,
            step=1,
            key="well_period_calculation",
        )

    # =====================================================
    # ДАННЫЕ ДЛЯ РАСЧЁТА
    # =====================================================

    return {
        # ---------- Общие ----------
        "r_w": float(r_w),
        "t_p": float(t_p),
        "well_efficiency": float(well_efficiency),
        "KUBS": float(KUBS),
        "KPPP": float(KPPP),
        "skin": float(skin),

        # ---------- ГРП ----------
        "Type_Frac": (
            None
            if frac_type == "Без ГРП"
            else frac_type
        ),
        "switch_fracList_params": use_frac_sheet,
        "length_FracStage": float(length_FracStage),
        "xfr": float(xfr),
        "w_f": float(w_f),
        "k_f": float(k_f),
        "path_frac": frac_file,

        # ---------- Фактический фонд ----------
        "first_months": int(first_months),
        "last_months": int(last_months),
        "default_radius_prod": float(default_radius_prod),
        "default_radius_inj": float(default_radius_inj),
        "switch_filtration_perm_fact": switch_filtration_perm_fact,

        # ---------- Проектный фонд ----------
        "L": int(L),
        "min_length": int(min_length),
        "buffer_project_wells": int(buffer_project_wells),
        "switch_fix_P_well_init": switch_fix_P_well_init,
        "fix_P_well_init": (
            float(fix_P_well_init)
            if switch_fix_P_well_init
            else 0.0
        ),
        "switch_wc_from_map": (
            water_cut_source == "По карте"
        ),
        "k": int(k),
        "threshold": float(threshold),
        "period_calculation": int(period_calculation),
    }