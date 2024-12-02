import pandas as pd
import app.decline_rate.config as cfg

from tqdm import tqdm
from loguru import logger

from app.decline_rate.functions import history_processing
from app.decline_rate.residual_reserves import get_reserves_by_characteristic_of_desaturation, get_reserves_by_map
from app.decline_rate.model_Corey import calculation_model_corey
from app.decline_rate.model_Arps import calculation_model_arps


@logger.catch
def get_decline_rate(data_history, data_wells, maps, type_reserves='voronoy'):
    logger.info("Подготовка истории работы скважин для расчета темпов падения")
    max_delta = cfg.STOPPING_TIME_LIMIT_OF_WELL
    df_initial = history_processing(data_history, max_delta=max_delta)

    #  Загрузка констант расчета ОИЗ
    min_reserves = cfg.MIN_RESIDUAL_RECOVERABLE_RESERVES
    year_min = cfg.MIN_PERIOD_WELL_WORKING
    year_max = cfg.MAX_PERIOD_WELL_WORKING
    r_max = cfg.MAX_RADIUS

    logger.info(f"Расчет ОИЗ для оценки темпов, тип: {type_reserves}")
    if type_reserves == 'voronoy':
        # расчет ОИЗ под скважинами через радиус по ячейкам Вороного
        type_map_list = list(map(lambda raster: raster.type_map, maps))
        # инициализация всех необходимых карт
        map_residual_recoverable_reserves = maps[type_map_list.index("residual_recoverable_reserves")]
        data_wells['well_reserves'] = get_reserves_by_map(data_wells,
                                                          map_residual_recoverable_reserves, min_reserves)
    elif type_reserves == 'statistic':
        well_reserves = get_reserves_by_characteristic_of_desaturation(df_initial,
                                                                       min_reserves, r_max, year_min, year_max)
        data_wells = pd.merge(data_wells.set_index('well_number'), well_reserves, left_index=True, right_index=True)
        data_wells['well_reserves'] = data_wells['well_reserves'].fillna(min_reserves)
        data_wells = data_wells.reset_index()

    #  Создание словаря ОИЗ
    dict_reserves = dict(zip(data_wells.well_number, data_wells.well_reserves))

    # Создание словаря с ограничениями аппроксимации
    dict_constraints = {"Arps": {"k1_left": cfg.K1_LEFT, "k2_left": cfg.K1_RIGHT,
                                 "k1_right": cfg.K2_LEFT, "k2_right": cfg.K2_RIGHT},
                        "Corey": {"corey_oil_left": cfg.COREY_OIL_LEFT, "corey_water_left": cfg.COREY_WATER_LEFT,
                                  "mef_left": cfg.MEF_LEFT, "mef_right": cfg.MEF_RIGHT}}
    df_wells_decline_rates = pd.DataFrame(columns=["well_number",
                                                   "coefficients_model_corey", 'coefficients_model_arps',
                                                   "residual_reserves", "cumulative_oil_production"])
    for well in tqdm(df_initial.well_number.unique(), desc='Расчет коэффициентов для темпа падения по истории работы'):
        # Выделение исходных данных для аппроксимации скважины
        df_well = df_initial.loc[df_initial.well_number == well].reset_index(drop=True)
        cumulative_oil_production = df_well.Qo.sum() / 1000
        try:
            residual_reserves = dict_reserves[well]
        except KeyError:
            residual_reserves = min_reserves
        #  аппроксимация характеристики вытеснения
        coefficients_model_corey = calculation_model_corey(df_well.Qo.values, df_well.Ql.values,
                                                           residual_reserves, dict_constraints["Corey"])

        #  аппроксимация кривой добычи жидкости
        coefficients_model_arps = calculation_model_arps(df_well.Ql_rate.values, dict_constraints["Arps"])

        df_wells_decline_rates.loc[well] = [well, coefficients_model_corey, coefficients_model_arps,
                                            residual_reserves, cumulative_oil_production]

    return df_wells_decline_rates, df_initial



