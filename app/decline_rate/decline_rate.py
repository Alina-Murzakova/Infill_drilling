import pandas as pd
import app.decline_rate.config as cfg

from tqdm import tqdm
from loguru import logger

from app.decline_rate.functions import history_processing
from app.decline_rate.residual_reserves import get_reserves_by_characteristic_of_desaturation, get_reserves_by_map
from app.decline_rate.model_Arps import calculation_model_arps


@logger.catch
def get_decline_rate(data_history, data_wells, maps=None, type_reserves=None):
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
    for well in tqdm(df_initial.well_number.unique(), desc='Расчет коэффициентов для темпа падения по истории работы'):
        # Выделение исходных данных для аппроксимации скважины
        df_well = df_initial.loc[df_initial.well_number == well].reset_index(drop=True)
        cumulative_oil_production = df_well.Qo.sum() / 1000
        #  аппроксимация характеристики вытеснения
        coefficients_Ql_rate = calculation_model_arps(df_well.Ql_rate.values, dict_constraints)
        #  аппроксимация кривой добычи жидкости
        coefficients_Qo_rate = calculation_model_arps(df_well.Qo_rate.values, dict_constraints)

        df_wells_decline_rates.loc[well] = [well, coefficients_Ql_rate, coefficients_Qo_rate,
                                            cumulative_oil_production]
    return df_wells_decline_rates, df_initial, data_wells
