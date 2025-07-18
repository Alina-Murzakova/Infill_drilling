import math
import numpy as np
import copy
import pandas as pd

from loguru import logger
from scipy.optimize import root_scalar
from .one_phase_model import get_one_phase_model
from .pressure_drop_for_Jd import get_dimensionless_delta_pressure


def calculate_starting_rate(reservoir_params, fluid_params, well_params, coefficients,
                            kv_kh=0.1, Swc=0.2, Sor=0.3, Fw=0.3, m1=1, Fo=1, m2=1, Bw=1):
    """
    Расчет Q_liq - Qж (м3/сут) на Tзап без учета лифта и
           Q_oil - Qн (т/сут) с учетом лифта 3'' на Tзап
    --------------------------------------------------------------------------------------------------------------------
    Свойства пласта - reservoir_params(6):
    f_w - ожидаемая обводненность| %
    c_r - сжимаемость породы | 1/атм
    Phi - пористость | д.ед.
    h - эффективная мощность пласта (общая) | м
    k_h - ожидаемая проницаемость - горизонтальная проницаемость (в плоскости XY) | мД
    Pr - текущее пластовое давление | атм

    Свойства пластовых флюидов - fluid_params (7):
    mu_w - вязкость воды | сП
    mu_o - вязкость нефти | сП
    c_o - сжимаемость нефти | 1/атм
    c_w - сжимаемость воды | 1/атм
    Bo - объемный коэффициент расширения нефти | м3/м3
    Pb - давление насыщения | атм
    rho - плотность нефти | г/см3

    Параметры скважины и трещины - well_params (9):
    L - длина горизонтальной скважины или наклонной скважины | м
    xfr - полудлина трещины ГРП | м
    w_f - раскрытие трещины ГРП | мм
    FracCount - количество трещин ГРП в случае горизонтальной скважины с МГРП | шт.
    k_f - проницаемость проппантной пачки | Д
    Pwf - Забойное давление| атм
    t_p - Tзап - время работы до запуска (на ВНР) | дни
    r_e - Rэфф - радиус области дренирования | м
    r_w - радиус скважины | м

    Экспертные параметры - coefficients(3):
    KPPP - коэффициент для ГС (мощность пласта) | безразмерный
    skin - скин-фактор кольматации | безразмерный
           используется для вертикальной и горизонтальной скважины r_w_eff = r_w * exp(-skin)
           для трещины ГРП и горизонтальной с МГРП - скин на стенке трещины | безразмерный
    KUBS - коэффициент успешности бурения скважины | безразмерный

    Необязательные параметры со значениями по умолчанию(8):
    kv_kh - отношение вертикальной проницаемости k_v к горизонтальной проницаемости k_h (в плоскости XY)
    Swc - связанная водонасыщенность | д.ед
    Sor - остаточная нефтенасыщенность | д.ед
    Fw - концевая точка по воде - k_rw(S_or)
    m1 - Степень Кори по воде
    Fo - концевая точка по нефти - k_ro(S_wc)
    m2 - Степень Кори по нефти
    Bw - объемный м расширения воды | м3/м3

    Рассчитываются по функциям:
    delta_P - депрессия | атм
    well_productivity - коэффициент продуктивности | м3/сут/атм
    --------------------------------------------------------------------------------------------------------------------
    """
    rho = fluid_params['rho']
    f_w = reservoir_params['f_w']
    mu, c_t, B = get_one_phase_model(fluid_params, reservoir_params, Swc, Sor, Fw, m1, Fo, m2, Bw)

    # Коэффициент продуктивности | м3/сут/атм
    well_productivity = get_well_productivity(mu, c_t, B, reservoir_params, well_params, coefficients, kv_kh)

    # Депрессия по Вогелю | атм
    delta_P = get_delta_pressure(reservoir_params, fluid_params, well_params)

    Q_liq = well_productivity * delta_P
    Q_max = -5.3918 * rho * 1000 + 6614.7  # пропускная способность НКТ 89 мм | м3/сут - характеристика напора
    if Q_liq > Q_max:
        Q_liq = Q_max

    Q_oil = Q_liq * rho * (1 - f_w / 100)
    return float(Q_liq), float(Q_oil)


def calculate_permeability_fact_wells(row, dict_parameters_coefficients):
    """
    Находит значение k_h, при котором расчетный Q_liq совпадает с фактическим Q_liq ('init_Ql_rate_TR')
    Parameters
    ----------

    Returns
    -------
    k_h: найденное значение проницаемости (мД)
    """
    # Переопределим параметры из словаря
    kv_kh, Swc, Sor, Fw, m1, Fo, m2, Bw = (
        list(map(lambda name: dict_parameters_coefficients['default_well_params'][name],
                 ['kv_kh', 'Swc', 'Sor', 'Fw', 'm1', 'Fo', 'm2', 'Bw'])))

    reservoir_params = dict_parameters_coefficients['reservoir_params']
    fluid_params = dict_parameters_coefficients['fluid_params']
    coefficients = dict_parameters_coefficients['coefficients']
    well_params = copy.deepcopy(dict_parameters_coefficients['well_params'])
    if dict_parameters_coefficients['well_params']['switch_avg_frac_params']:
        well_params['xfr'] = row['xfr']
        well_params['w_f'] = row['w_f']
        well_params['FracCount'] = row['FracCount']
    else:
        well_params['FracCount'] = check_FracCount(well_params['Type_Frac'],
                                                   well_params['length_FracStage'],
                                                   row['length_geo'])

    reservoir_params['f_w'] = row['init_water_cut_TR']
    reservoir_params['Phi'] = row['m']
    reservoir_params['h'] = row['NNT']
    reservoir_params['Pr'] = row['init_P_reservoir_prod']
    if row['well_type'] == "vertical":
        well_params['L'] = 0
    else:
        well_params['L'] = row['length_geo']
    well_params['Pwf'] = row['init_P_well_prod']
    well_params['r_e'] = row['r_eff_voronoy']
    if (row.init_Ql_rate_TR > 0 and row.init_P_well_prod > 0
            and row.init_P_reservoir_prod > 0 and row.init_P_reservoir_prod > row.init_P_well_prod):
        def error_function(k_h):
            reservoir_params['k_h'] = k_h
            # Расчет дебитов
            Q_liq, _ = calculate_starting_rate(reservoir_params, fluid_params, well_params, coefficients,
                                               kv_kh, Swc, Sor, Fw, m1, Fo, m2, Bw)
            # Ошибка между расчетным и известным Q_liq
            abs_error = float(Q_liq) - row['init_Ql_rate_TR']
            return abs_error

        # Оптимизация - ищем k_h, при котором ошибка минимальна
        try:
            result = root_scalar(error_function, bracket=[1e-3, 1e3], method='brentq', xtol=1e-3, rtol=1e-2,
                                 maxiter=100)
            if result.converged:
                return result.root
            else:
                logger.info(f'Не удалось найти значение k_h для фактической скважины №{row.well_number}')
            return 0
        except ValueError:
            logger.info(f'Не удалось найти значение k_h для фактической скважины №{row.well_number}')
            return 0
    else:
        return 0


def get_df_permeability_fact_wells(data_wells, dict_parameters_coefficients, switch):
    """
    Расчет проницаемости по фактическому фонду через РБ
    Parameters
    ----------
    switch - фильтрация выбросов по статистике в массиве фактических проницаемостей
    """
    data_wells_for_perm = data_wells[(data_wells['m'] > 0) & (data_wells['NNT'] > 0)].copy()
    data_wells_for_perm = data_wells_for_perm[(data_wells_for_perm['init_Ql_rate_TR'] > 0) &
                                              (data_wells_for_perm['init_P_well_prod'] > 0) &
                                              (data_wells_for_perm['init_P_reservoir_prod'] > 0) &
                                              (data_wells_for_perm['init_P_reservoir_prod'] >
                                               data_wells_for_perm['init_P_well_prod'])]
    data_wells_for_perm['permeability_fact'] = data_wells_for_perm.apply(calculate_permeability_fact_wells,
                                                                         args=(dict_parameters_coefficients,),
                                                                         axis=1)
    del data_wells['permeability_fact']
    data_wells = data_wells.merge(data_wells_for_perm[['well_number', 'permeability_fact']],
                                  how='left', on='well_number')
    data_wells['permeability_fact'] = data_wells['permeability_fact'].fillna(0)

    if switch:
        # Верхняя граница для фильтрации выбросов (персентиль q3)
        permeability_lower_bound, permeability_upper_bound = quantile_filter(data_wells, name_column='permeability_fact')
        data_wells['permeability_fact'] = np.where(data_wells['permeability_fact'] > permeability_upper_bound,
                                                   permeability_upper_bound,
                                                   data_wells['permeability_fact'])  # 0 или permeability_upper_bound
    avg_permeability = data_wells[data_wells['permeability_fact'] != 0]['permeability_fact'].mean()
    # Перезапись значения проницаемости по объекту из ГФХ на среднюю по фактическому фонду
    dict_parameters_coefficients['reservoir_params']['k_h'] = avg_permeability
    dict_parameters_coefficients['well_params']['init_P_well'] = data_wells[data_wells['init_P_well_prod']
                                                                            != 0]['init_P_well_prod'].mean()
    return data_wells, dict_parameters_coefficients


def apply_iqr_filter(data_wells, name_column):
    """Функция определения верхней границы выбросов методом межквартильного размаха IQR"""
    column = data_wells[data_wells[name_column] > 0][name_column]
    # Рассчитываем квартили
    q1 = np.percentile(column, 25)
    q3 = np.percentile(column, 75)
    iqr = q3 - q1
    # Определяем порог для отсеивания "выбросов"
    upper_bound = q3 + 1.5 * iqr
    return upper_bound


def quantile_filter(data_wells, name_column):
    """Функция определения верхнего и нижнего квантиля"""
    column = data_wells[data_wells[name_column] > 0][name_column]
    # Рассчитываем квартили
    q1 = np.percentile(column, 25)
    q3 = np.percentile(column, 75)
    return q1, q3


def check_FracCount(type_frac, length_FracStage=1, L=1):
    # определение FracCount
    if type_frac is None:
        return 0
    elif type_frac == 'ГРП':
        return 1
    elif type_frac == 'МГРП':
        if L == 0:
            return 1
        else:
            return int(L / length_FracStage)
    else:
        logger.error(f"Некорректно задан тип ГРП: {type_frac}. Допустимые значения None, ГРМ, МГРП")
        return None


def get_delta_pressure(reservoir_params, fluid_params, well_params):
    """
    Расчет депрессии с учетом газовой фазы и обводненности продукции (Вогель с учетом обводненности)
    Parameters
    ----------
    Pb - давление насыщения, атм
    Pwf - забойное давление, атм
    f_w - ожидаемая обводненность, атм
    Pr - текущее пластовое давление, атм

    Returns
    -------
    dP - депрессия, атм
    """
    Pb = fluid_params['Pb']
    Pwf = well_params['Pwf']
    Pr, f_w = list(map(lambda key: reservoir_params[key], ['Pr', 'f_w']))

    #  корректировка для пластов ниже давления насыщения
    if Pb > Pr:
        Pb = Pr

    if f_w == 100:  # вода
        dP = Pr - Pwf
    else:
        Pwf_G = (4 / 9) * (f_w / 100) * Pb  # Характерное давление на кривой притока при трехфазном потоке
        if Pwf >= Pb:  # нефть / (нефть+ вода)
            dP = Pr - Pwf
        elif Pwf < Pwf_G:  # нефть + вода + газ
            # Промежуточные, необходимые для расчета, параметры
            tgb_r = (81 - 80 * (0.999 * Pb - 0.0018 * (Pr - Pb)) / Pb) ** 0.5  # корень квадратный в выражении для tgb
            tgb = f_w / 100 + (0.125 * (1 - f_w / 100) * Pb * (-1 + tgb_r)) / (0.001 * (Pr - (4 / 9) * Pb))
            dP = (Pwf_G + (Pr - (4 / 9) * Pb) * tgb - Pwf) / tgb
        else:
            # Промежуточные, необходимые для расчета, параметры
            A = (Pwf + 0.125 * (1 - f_w / 100) * Pb - (f_w / 100) * Pr) / (0.125 * (1 - f_w / 100) * Pb)
            B = (f_w / 100) / (0.125 * (1 - f_w / 100) * Pb)
            C = 2 * A * B + 144 / Pb
            D = A ** 2 - 144 * (Pr - Pb) / Pb - 81
            if B == 0:  # нефть + газ (по Вогелю)
                dP = -D / C
            else:  # нефть + вода + газ
                dP = (-C + (C ** 2 - 4 * (B ** 2) * D) ** 0.5) / (2 * (B ** 2))
    return dP


def get_well_productivity(mu, c_t, B, reservoir_params, well_params, coefficients, kv_kh, mode=2):
    """
    Расчет коэффициента продуктивности ствола (м3/сут/атм) в зависимости от типа ствола
    Пояснения к некоторым величинам:
    mode - определяет, что рассчитываем 1 - well flowing pressure, 2 - well flow rate
    Fcd (Dimensionless fracture conductivity) - безразмерная проводимость трещины
    Pd - безразмерный перепад давлений
    Jd - безразмерная продуктивность ствола
    """
    # Выделение необходимых параметров скважины
    L, r_w, xfr, FracCount, w_f, k_f = list(map(lambda key: well_params[key], ['L', 'r_w', 'xfr', 'FracCount',
                                                                               'w_f', 'k_f']))
    t = well_params['t_p'] * 24  # перевод дней в часы
    r_e = well_params['r_e'] * 100  # 100 - коэффициент для масштабирования решений с очень маленьким радиусом,
    # для уменьшения влияния границ дренажного прямоугольника
    # Выделение параметров пласта
    Phi, h, k_h = list(map(lambda key: reservoir_params[key], ['Phi', 'h', 'k_h']))
    # Выделение экспертных параметров
    KUBS, KPPP, skin = list(map(lambda key: coefficients[key], ['KUBS', 'KPPP', 'skin']))

    # размеры дренажного прямоугольника
    xe = (math.pi * r_e ** 2 + L ** 2) ** 0.5
    ye = math.pi * r_e ** 2 / xe

    # координаты скважины
    xw, yw = xe / 2, ye / 2

    K_Jd = 1  # коэффициент масштабирования Jd
    if FracCount > 0:
        if L == 0:  # если ННС возможен сценарий только с 1 трещиной
            L_fr = xfr
        else:  # если ГС
            if FracCount > 1:  # если трещин больше 1 пересчитываем полудлину через L
                xfr = L / (2 * FracCount)
            L_fr = L / 2 + (FracCount - 1) * xfr / 2  # второе слагаемое связано с суммарным воздействием всех трещин
            # полу-длина трещины = половина длины горизонтального ствола скважины
            # максимальная теоретическая длина дренажного воздействия для одной трещины
        Fcd = FracCount * w_f * k_f / (xfr * k_h)
    else:  # нет ГРП, у нас один сток = сверх-проводимая трещина
        Fcd = 10000000
        if L > 0:  # ГС
            L_fr = L / 2
            h_eff = h * (1 / kv_kh) ** 0.5  # эффективная толщина
            r_w_eff = r_w / 2 * (1 + (1 / kv_kh) ** 0.5)  # эффективный радиус эллиптической скважины
            skin_hor = h_eff / L * math.log(h_eff / (2 * math.pi * r_w_eff))  # добавка в скин от анизотропии k пласта
            skin += skin_hor
            K_Jd = KPPP  # коэффициент для ГС
        else:  # Вертикальная скважина без ГРП
            if skin < -3:
                return 0
            # длина трещины, размеры и координаты скважины в данном случае изменяются
            L_fr = 2 * r_w
            ye = xe
            xw, yw = xe / 2, xe / 2

    Jd = K_Jd * get_dimensionless_delta_pressure(t, Fcd, L_fr, k_h, c_t, Phi, mu, xe, ye, xw, yw, mode, skin)
    J = (k_h * h) / (18.42 * B * mu) * Jd  # Продуктивность скважины
    PI = KUBS * J  # Продуктивность скважины с учетом успешности
    return PI
