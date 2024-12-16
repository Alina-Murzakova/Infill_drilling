import pandas as pd
import numpy as np
import app.decline_rate.config as cfg

from tqdm import tqdm
from loguru import logger

from app.decline_rate.functions import history_processing
from app.decline_rate.residual_reserves import get_reserves_by_characteristic_of_desaturation, get_reserves_by_map
from app.decline_rate.model_Arps import calculation_model_arps


@logger.catch
def get_decline_rates(data_history, data_wells, maps=None, type_reserves=None):
    logger.info("Подготовка истории работы скважин для расчета темпов падения")
    max_delta = cfg.STOPPING_TIME_LIMIT_OF_WELL
    df_initial = history_processing(data_history, max_delta=max_delta)

    #  Загрузка констант расчета ОИЗ
    min_reserves = cfg.MIN_RESIDUAL_RECOVERABLE_RESERVES
    year_min = cfg.MIN_PERIOD_WELL_WORKING
    year_max = cfg.MAX_PERIOD_WELL_WORKING
    r_max = cfg.MAX_RADIUS

    if type_reserves == 'voronoy':
        logger.info(f"Расчет ОИЗ, тип: {type_reserves}")
        if maps is not None:
            # расчет ОИЗ под скважинами через радиус по ячейкам Вороного
            type_map_list = list(map(lambda raster: raster.type_map, maps))
            # инициализация всех необходимых карт
            map_residual_recoverable_reserves = maps[type_map_list.index("residual_recoverable_reserves")]
            data_wells['well_reserves'] = get_reserves_by_map(data_wells,
                                                              map_residual_recoverable_reserves, min_reserves)
    elif type_reserves == 'statistic':
        logger.info(f"Расчет ОИЗ, тип: {type_reserves}")
        well_reserves = get_reserves_by_characteristic_of_desaturation(df_initial,
                                                                       min_reserves, r_max, year_min, year_max)
        data_wells = pd.merge(data_wells.set_index('well_number'), well_reserves, left_index=True, right_index=True)
        data_wells['well_reserves'] = data_wells['well_reserves'].fillna(min_reserves)
        data_wells = data_wells.reset_index()
    else:
        logger.info(f"Аппроксимация темпов без расчета ОИЗ")

    # Создание словаря с ограничениями аппроксимации
    dict_constraints = {"k1_left": cfg.K1_LEFT, "k2_left": cfg.K1_RIGHT,
                        "k1_right": cfg.K2_LEFT, "k2_right": cfg.K2_RIGHT}
    df_wells_decline_rates = pd.DataFrame(columns=["well_number", 'coefficients_Ql_rate', 'coefficients_Qo_rate',
                                                   "cumulative_oil_production"])
    avg_coefficients_Ql_rate, avg_coefficients_Qo_rate, avg_init_Ql_rate, avg_init_Qo_rate = [], [], [], []
    for well in tqdm(df_initial.well_number.unique(), desc='Расчет коэффициентов для темпа падения по истории работы'):
        # Выделение исходных данных для аппроксимации скважины
        df_well = df_initial.loc[df_initial.well_number == well].reset_index(drop=True)
        cumulative_oil_production = df_well.Qo.sum() / 1000
        #  аппроксимация характеристики вытеснения
        coefficients_Ql_rate = calculation_model_arps(df_well.Ql_rate.values, dict_constraints)
        #  аппроксимация кривой добычи жидкости
        coefficients_Qo_rate = calculation_model_arps(df_well.Qo_rate.values, dict_constraints)

        df_wells_decline_rates.loc[df_wells_decline_rates.shape[0] + 1] = \
            [well, coefficients_Ql_rate, coefficients_Qo_rate, cumulative_oil_production]

        # данные по Арпсу Ql
        model_arps_ql = coefficients_Ql_rate
        success_arps_ql = model_arps_ql[0]
        # данные по Арпсу Qo
        model_arps_qo = coefficients_Qo_rate
        success_arps_qo = model_arps_qo[0]
        # если оптимизация сошлась - добавляем коэффициенты к среднему
        if success_arps_ql:
            avg_coefficients_Ql_rate.append(model_arps_ql[1])
            avg_init_Ql_rate.append(model_arps_ql[2])
        if success_arps_qo:
            avg_coefficients_Qo_rate.append(model_arps_qo[1])
            avg_init_Qo_rate.append(model_arps_qo[2])

    avg_coefficients_Ql_rate = np.mean(np.array(avg_coefficients_Ql_rate), axis=0)
    avg_coefficients_Qo_rate = np.mean(np.array(avg_coefficients_Qo_rate), axis=0)
    avg_init_Ql_rate = np.mean(np.array(avg_init_Qo_rate), axis=0)
    avg_init_Qo_rate = np.mean(np.array(avg_init_Qo_rate), axis=0)

    avg_coefficients_Ql_rate = [True, avg_coefficients_Ql_rate, avg_init_Ql_rate,
                                0, f"средний темп скважин месторождения, количество: {df_wells_decline_rates.shape[0]}"]
    avg_coefficients_Qo_rate = [True, avg_coefficients_Qo_rate, avg_init_Qo_rate,
                                0, f"средний темп скважин месторождения, количество: {df_wells_decline_rates.shape[0]}"]

    # Формирование строки в таблице со средним темпом по всем скважинам
    df_wells_decline_rates.loc[df_wells_decline_rates.shape[0] + 1] = ["default_decline_rates",
                                                                       avg_coefficients_Ql_rate,
                                                                       avg_coefficients_Qo_rate,
                                                                       None]
    return df_wells_decline_rates, df_initial, data_wells


def get_avg_decline_rates(data_decline_rate, Ql_start, Qo_start):
    """
    Расчет средних моделей Арпса для кривых дебита жидкости и нефти на основе окружения
    Parameters
    ----------
    data_decline_rate - df с моделями Арпса для ближайших скважин
    Ql_start - стартовый дебит жидкости, т/сут
    Qo_start - стартовый дебит нефти, т/сут

    Returns - массив с коэффициентами аппроксимации
    -------
    [success, [k1, k2], starting_productivity, starting_index, message]
    """
    neighboring_wells = data_decline_rate.well_number.unique()
    count_success_wells = "no"
    success = False
    avg_coeff_arps_ql, avg_coeff_arps_qo = None, None
    message = f"отсутствуют темпы у соседних скважин"
    if len(neighboring_wells) > 1:
        neighboring_wells = np.delete(neighboring_wells, np.where(neighboring_wells == 'default_decline_rates'))
        avg_coeff_arps_ql, avg_coeff_arps_qo, count_success_wells = [], [], 0
        for well in neighboring_wells:
            # Выделение исходных данных для скважины
            decline_rate = data_decline_rate.loc[data_decline_rate.well_number == well]
            # данные по Арпсу Ql
            model_arps_ql = decline_rate["coefficients_Ql_rate"].iloc[0]
            # данные по Арпсу Qo
            model_arps_qo = decline_rate["coefficients_Qo_rate"].iloc[0]
            success_arps_ql, success_arps_qo = model_arps_ql[0], model_arps_qo[0]
            if success_arps_ql and success_arps_qo:
                # если оптимизация сошлась - добавляем коэффициенты к среднему
                avg_coeff_arps_ql.append(model_arps_ql[1])
                avg_coeff_arps_qo.append(model_arps_qo[1])
                count_success_wells += 1
        avg_coeff_arps_ql = np.mean(np.array(avg_coeff_arps_ql), axis=0)
        avg_coeff_arps_qo = np.mean(np.array(avg_coeff_arps_qo), axis=0)
        message = f"средний темп скважин: {neighboring_wells}"
        success = True
    if count_success_wells == 0 or len(neighboring_wells) == 1:
        decline_rate = data_decline_rate.loc[data_decline_rate.well_number == 'default_decline_rates']
        avg_coeff_arps_ql = decline_rate["coefficients_Ql_rate"].iloc[0][1]
        avg_coeff_arps_qo = decline_rate["coefficients_Qo_rate"].iloc[0][1]
        message = decline_rate["coefficients_Qo_rate"].iloc[0][4]
        success = True
    return ([success, avg_coeff_arps_ql, Ql_start, 0, message],
            [success, avg_coeff_arps_qo, Qo_start, 0, message])
