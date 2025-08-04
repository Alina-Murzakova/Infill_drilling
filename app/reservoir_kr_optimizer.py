from scipy.stats import pearsonr
from scipy.optimize import minimize, fsolve
from loguru import logger

import numpy as np
import pandas as pd
import math


@logger.catch
def get_reservoir_kr(data_history, data_wells, dict_parameters_coefficients, S_or=0.3):
    """Функция для авто-адаптации относительных фазовых проницаемостей объекта на основе history_matching"""
    # Константы
    mu_w = dict_parameters_coefficients['fluid_params']['mu_w']
    mu_o = dict_parameters_coefficients['fluid_params']['mu_o']
    Bw = dict_parameters_coefficients['default_well_params']['Bw']
    Bo = dict_parameters_coefficients['fluid_params']['Bo']

    # Оставим только текущие добывающие
    prod_wells = data_wells[data_wells.work_marker == 'prod']['well_number']
    data_wells = data_wells[data_wells.work_marker == 'prod'].reset_index()
    data_history = data_history[data_history.well_number.isin(prod_wells)]
    data_history = data_history[(data_history['work_marker'] == 'НЕФ')
                                & (data_history['well_status'].isin(['РАБ.', 'ОСТ.']))]

    # Предобработка данных
    data_history['water_cut'] = data_history['water_cut'] / 100
    data_history = data_history.sort_values(by=['well_number', 'date'])
    data_history['Qw_cumsum'] = (data_history['Ql'] - data_history['Qo']).groupby(data_history['well_number']).cumsum()

    # Применяем функцию проверки к каждой скважине и сохраняем результаты
    results_check = data_wells['well_number'].apply(lambda x:
                                                    check_well(data_history[data_history['well_number'] == x]))
    # Преобразуем результаты в DataFrame
    results_check = pd.DataFrame(results_check.tolist(), columns=['quality_check', 'message', 'list_stats'])

    # Объединяем с исходным DataFrame
    data_wells = pd.concat([data_wells, results_check[['quality_check', 'message', 'list_stats']]], axis=1)

    data_history = data_history[data_history['Qo_rate'] > 0]
    data_history = data_history[data_history['water_cut'] < 0.99]

    well_for_adaptation = data_wells[data_wells['quality_check']]['well_number']
    if well_for_adaptation.empty:
        logger.info(f"Нет скважин с допустимой историей работы для адаптации ОФП")
        Swi_opt, no_opt, nw_opt, kro0_opt, krw0_opt = [0.5, 3, 2, 0.05, 0.1]
        logger.info(f"Будут использованы базовые концевые точки: Swc = {Swi_opt}, "
                    f"Sor = {S_or}, Fw = {krw0_opt}, m1 = {nw_opt}, Fo = {kro0_opt}, m2 = {no_opt}")
    else:
        logger.info(f"Для адаптации ОФП используются {well_for_adaptation.shape[0]} скважин")
        average_result = []
        for well in well_for_adaptation:
            r_eff = data_wells[data_wells.well_number == well]['r_eff_not_norm'].iloc[0]
            h = data_wells[data_wells.well_number == well]['NNT'].iloc[0]
            m = data_wells[data_wells.well_number == well]['m'].iloc[0]
            df_well = data_history[data_history.well_number == well]
            Wp_real, fw_real = df_well['Qw_cumsum'].to_numpy(), df_well['water_cut'].to_numpy()

            # Оптимизация
            result = curve_fit_well(Wp_real, fw_real, r_eff, h, m, S_or, mu_o, mu_w, Bo, Bw)
            average_result.append(result)
        average_result = np.mean(np.array(average_result), axis=0)

        Swi_opt, no_opt, nw_opt, kro0_opt, krw0_opt = average_result
        logger.info(f"Адаптированные концевые точки: Swc = {round(Swi_opt, 3)}, Sor = {S_or}, "
                    f"Fw = {round(krw0_opt, 3)}, m1 = {round(nw_opt, 3)}, "
                    f"Fo = {round(kro0_opt, 3)}, m2 = {round(no_opt, 3)}")

    # Обновление значения концевых точек в словаре параметров
    dict_parameters_coefficients['default_well_params']['Swc'] = Swi_opt
    dict_parameters_coefficients['default_well_params']['Sor'] = S_or
    dict_parameters_coefficients['default_well_params']['Fw'] = krw0_opt
    dict_parameters_coefficients['default_well_params']['m1'] = nw_opt
    dict_parameters_coefficients['default_well_params']['Fo'] = kro0_opt
    dict_parameters_coefficients['default_well_params']['m2'] = no_opt

    return dict_parameters_coefficients


def check_well(df_well, min_months=24, threshold_diff=0.4, zero_prod_threshold=0.15, fw0_threshold=0.2,
               threshold_gradient=0.01, threshold_pearson=0.4, threshold_mean_median=0.3):
    """
    Проверка качества истории обводненности скважины для адаптации ОФП
    Parameters
    ----------
    df_well - слайс истории работы скважины по датам
    min_months - минимальное количество месяцев в работе 24
    threshold_diff - максимальный скачок обводненности между месяцами 0.4
    zero_prod_threshold - минимальный процент не рабочих дней от всей истории 0.15
    fw0_threshold - максимальная обводненность на старте 0.2
    threshold_gradient - порог скорости обводнения Δfw/ΔWp 0.01
    threshold_pearson - минимальный коэффициент Пирсона 0.4
    threshold_mean_median - максимальное значение медианы и среднего в распределение обводненности 0.3

    Returns
    -------
    Фрейм скважин с пометкой True|False и сообщением о причине не допуска скважины
    """
    not_null_water_cut = df_well[df_well['Ql_rate'] > 0]['water_cut']
    # 1. Критерии отбраковки скважин
    # Минимальная длина истории
    min_months_well = df_well[df_well['time_work_prod'] > 0].shape[0]
    if min_months_well < min_months:
        return [False, "Короткая история", min_months_well]

    # Аномальные скачки обводненности
    fw_diff = not_null_water_cut.diff().abs()
    if (fw_diff > threshold_diff).any():
        return [False, "Аномальные скачки обводненности", fw_diff.max()]

    # Нестабильная работа
    zero_months = (df_well['Ql_rate'] == 0).mean()
    if zero_months > zero_prod_threshold:
        return [False, "Нестабильная работа", zero_months]

    # Проверка на стартовую обводненность
    fw0 = not_null_water_cut.iloc[0]
    if fw0 > fw0_threshold:
        return [False, "Высокая стратовая обводненность", fw0]

    # Скважины с аномальными градиентами Δfw/ΔWp
    gradient = np.divide(np.diff(not_null_water_cut), np.diff(df_well[df_well['Ql_rate'] > 0]['Qw_cumsum']),
                         out=np.zeros_like(np.diff(not_null_water_cut), dtype=float),
                         where=np.diff(df_well[df_well['Ql_rate'] > 0]['Qw_cumsum']) != 0)
    if np.max(np.abs(gradient)) > threshold_gradient:
        return [False, "Aномальный градиент", np.max(np.abs(gradient))]  # Порог скорости обводнения

    # 2. Статистические метрики для отбора
    # Тренд обводненности
    months = np.arange(len(df_well[df_well['Ql_rate'] > 0]))
    r, _ = pearsonr(months, df_well[df_well['Ql_rate'] > 0]['water_cut'])
    if r < threshold_pearson:
        return [False, "Не монотонный тренд обводненности", r]

    # Медиана и среднее
    mean = np.mean(df_well[df_well['Ql_rate'] > 0]['water_cut'])
    median = np.median(df_well[df_well['Ql_rate'] > 0]['water_cut'])
    if mean > threshold_mean_median or median > threshold_mean_median:
        return [False, "Медиана и средняя больше 50%", [mean, median]]

    return [True, "-", {'длина истории': min_months_well,
                        'max скачок обводненности': round(fw_diff.max(), 2),
                        'нестабильная работа %': round(zero_months, 2),
                        'стратовая обводненность': round(fw0, 2),
                        'коэф пирсона': round(r, 2),
                        'медиана и среднее': [round(mean, 2), round(median, 2)]}]


def brooks_corey_kr(Sw, Swi, Sor=0.3, kro0=1.0, krw0=0.3, no=2, nw=2):
    """
    Расчет относительных фазовых проницаемостей по модели Брукса-Кори.

    Параметры:
    Sw : float or array
        Текущая водонасыщенность.
    Swi : float
        Начальная водонасыщенность.
    Sor : float
        Остаточная нефтенасыщенность.
    kro0 : float, optional
        Максимальная относительная проницаемость нефти (при Sw=Swi).
    krw0 : float, optional
        Максимальная относительная проницаемость воды (при Sw=1-Sor).
    no : float, optional
        Показатель степени для нефти.
    nw : float, optional
        Показатель степени для воды.

    Возвращает:
    kro, krw : tuple
        Относительные проницаемости нефти и воды.
    """
    # Нормированные насыщенности
    S_star = (Sw - Swi) / (1 - Swi - Sor)  # для воды
    So_star = (1 - Sw - Sor) / (1 - Swi - Sor)  # для нефти

    # ОФП (с проверкой на границы)
    krw = np.where(Sw <= Swi, 0, krw0 * S_star ** nw)
    kro = np.where(Sw >= 1 - Sor, 0, kro0 * So_star ** no)

    return kro, krw


def simulate_fw(Wp, Swi, kro0, krw0, no, nw, r_eff, h, m, Sor, mu_o, mu_w, Bo=1.2, Bw=1.0):
    """
    Упрощенная модель обводненности на основе Brooks-Corey.
    Wp: массив накопленной добычи воды.
    Возвращает обводненность (fw(Wp)).
    """
    Sw_сurr = Swi + Wp.max() / (math.pi * (r_eff ** 2) * h * m * (1 - Swi - Sor))
    # Предполагаем линейную связь между Wp и водонасыщенностью Sw
    # Sw = Swi + (1 - Swi - Sor + Sw_сurr) * (Wp / Wp.max())
    Sw = Swi + (Sw_сurr - Swi) * (Wp / Wp.max())
    kro, krw = brooks_corey_kr(Sw, Swi, Sor, kro0, krw0, no, nw)
    # Расчет обводненности
    fw = (krw / (mu_w * Bw)) / ((krw / (mu_w * Bw)) + (kro / (mu_o * Bo)))
    return np.clip(fw, 0, 1)  # Ограничение [0, 1]


def error(params, Wp, fw, r_eff, h, m, Sor, mu_o, mu_w, Bo, Bw):
    Swi, no, nw, kro0, krw0 = params
    fw_model = simulate_fw(Wp, Swi, kro0, krw0, no, nw, r_eff, h, m, Sor, mu_o, mu_w, Bo, Bw)
    return np.mean((fw_model - fw) ** 2)  # MSES


def curve_fit_well(Wp, fw, r_eff, h, m, Sor, mu_o, mu_w, Bo, Bw):
    # Оптимизация
    result = minimize(
        error,
        x0=np.array([0.5, 3, 2, 0.05, 0.01]),  # Swi, no, nw, kro0, krw0
        args=(Wp, fw, r_eff, h, m, Sor, mu_o, mu_w, Bo, Bw),
        method='SLSQP',
        options={'maxiter': 6000},
        bounds=[(0.2, 1 - Sor), (1, 5), (1, 5), (0.01, 1.0), (0.01, 1.0)]
        # Границы параметров Swi, no, nw, kro0, krw0
    )
    return result.x
