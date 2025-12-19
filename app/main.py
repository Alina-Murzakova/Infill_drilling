import math
import geopandas as gpd
import pandas as pd
from loguru import logger

from app.input_output.input_economy import load_economy_data
from app.input_output.input_frac_info import load_frac_info
from app.input_output.input_geo_phys_properties import load_geo_phys_properties
from app.input_output.input_wells_data import load_wells_data, prepare_wells_data
from app.ranking_drilling.starting_rates import get_df_permeability_fact_wells
from app.decline_rate.decline_rate import get_decline_rates
from app.maps_handler.functions import mapping, calculate_reservoir_state_maps, calculate_score_maps
from app.well_active_zones import calculate_effective_radius
from app.drill_zones.drilling_zones import calculate_drilling_zones
from app.project_wells import calculate_reserves_by_voronoi
from app.input_output.output import get_save_path, upload_data
from app.reservoir_kr_optimizer import get_reservoir_kr
from app.exceptions import CalculationCancelled


def run_model(main_parameters, constants, total_stages, progress=None, is_cancelled=None):
    import logging
    logging.basicConfig(level=logging.INFO, )
    stage_number = 0

    def log_stage(msg):
        """Функция для логирования и определения № шага расчета"""
        nonlocal stage_number
        stage_number += 1
        logger.info(msg)
        if progress:
            progress(round(stage_number / total_stages * 100))  # обновляем прогресс расчета
        if is_cancelled and is_cancelled():
            raise CalculationCancelled()

    log_stage("Инициализация локальных переменных")
    # Пути
    paths = main_parameters['paths']
    # Параметры расчета
    parameters_calculation = main_parameters['parameters_calculation']
    # Параметры для скважин РБ
    well_params = main_parameters['well_params']
    # Переключатели
    switches = main_parameters['switches']

    # Константы расчета
    load_data_param = constants['load_data_param']
    default_coefficients = constants['default_coefficients']
    default_well_params = constants['default_well_params']
    if constants['default_project_well_params']['buffer_project_wells'] <= 0:
        # нижнее ограничение на расстояние до фактических скважин от проектной
        constants['default_project_well_params']['buffer_project_wells'] = 10
    well_params.update(constants['default_project_well_params'])

    log_stage("Загрузка скважинных данных")
    data_history, info_object_calculation = load_wells_data(data_well_directory=paths["data_well_directory"])
    name_field, name_object = info_object_calculation.get("field"), info_object_calculation.get("object_value")
    save_directory = f"{paths['save_directory']}\\{name_field}_{name_object.replace('/', '-')}"
    logger.add(f"{save_directory}/logs.log", mode='w')

    log_stage(f"Загрузка ГФХ по пласту {name_object.replace('/', '-')} месторождения {name_field}")
    dict_parameters_coefficients = load_geo_phys_properties(paths["path_geo_phys_properties"], name_field, name_object)
    dict_parameters_coefficients.update({'well_params': well_params,
                                         'default_well_params': default_well_params,
                                         'coefficients': default_coefficients,
                                         'switches': switches})
    log_stage("Подготовка скважинных данных")
    data_history, data_wells = prepare_wells_data(data_history,
                                                  dict_properties=dict_parameters_coefficients,
                                                  first_months=load_data_param['first_months'])

    if switches['switch_avg_frac_params']:
        log_stage(f"Загрузка фрак-листов")
        data_wells, dict_parameters_coefficients = load_frac_info(paths["path_frac"], data_wells, name_object,
                                                                  dict_parameters_coefficients)

    log_stage("Загрузка и обработка карт")
    maps, data_wells, maps_to_calculate = mapping(maps_directory=paths["maps_directory"],
                                                  data_wells=data_wells,
                                                  **{**load_data_param, **switches})
    default_size_pixel = maps[0].geo_transform[1]  # размер ячейки после загрузки всех карт

    log_stage("Расчет радиусов дренирования и нагнетания для скважин")
    data_wells = calculate_effective_radius(data_wells, dict_properties=dict_parameters_coefficients)

    if switches['switch_adaptation_relative_permeability']:
        log_stage("Авто-адаптация ОФП")
        dict_parameters_coefficients = get_reservoir_kr(data_history.copy(), data_wells.copy(),
                                                        dict_parameters_coefficients)

    if any(maps_to_calculate.values()):
        log_stage("Расчет карт текущего состояния: обводненности и ОИЗ")
        maps = calculate_reservoir_state_maps(data_wells,
                                              maps,
                                              dict_parameters_coefficients,
                                              default_size_pixel,
                                              maps_to_calculate,
                                              maps_directory=paths["maps_directory"])

    log_stage(f"Расчет оценочных карт")
    maps = maps + calculate_score_maps(maps=maps,
                                       dict_properties=dict_parameters_coefficients['reservoir_params'])

    log_stage("Расчет проницаемости для фактических скважин через РБ")
    (data_wells,
     dict_parameters_coefficients) = get_df_permeability_fact_wells(data_wells, dict_parameters_coefficients,
                                                                    save_directory)

    log_stage("Оценка темпов падения для текущего фонда")
    data_decline_rate_stat, _, _ = get_decline_rates(data_history, data_wells)

    log_stage("Расчет зон с высоким индексом бурения")
    # Параметры кластеризации
    epsilon = parameters_calculation["min_radius"] / default_size_pixel
    min_samples = int(parameters_calculation["sensitivity_quality_drill"] / 100 * epsilon ** 2 * math.pi)
    min_samples = 1 if min_samples == 0 else min_samples
    percent_low = 100 - parameters_calculation["percent_top"]
    list_zones, info_clusterization_zones = calculate_drilling_zones(maps=maps,
                                                                     epsilon=epsilon,
                                                                     min_samples=min_samples,
                                                                     percent_low=percent_low,
                                                                     data_wells=data_wells,
                                                                     dict_properties=dict_parameters_coefficients)

    type_map_list = list(map(lambda raster: raster.type_map, maps))
    map_rrr = maps[type_map_list.index('residual_recoverable_reserves')]
    map_opportunity_index = maps[type_map_list.index('opportunity_index')]
    polygon_OI = map_opportunity_index.raster_to_polygon()
    log_stage("Начальное размещение проектных скважин")
    well_params['buffer_project_wells'] = well_params['buffer_project_wells'] / default_size_pixel

    # Проектные скважины с других drill_zone, чтобы исключить пересечения
    gdf_project_wells_all = gpd.GeoDataFrame(columns=["LINESTRING_pix", "buffer"], geometry="LINESTRING_pix")
    for drill_zone in list_zones:
        if drill_zone.rating != -1:
            gdf_project_wells = drill_zone.get_init_project_wells(map_rrr, data_wells, gdf_project_wells_all,
                                                                  polygon_OI, default_size_pixel,
                                                                  parameters_calculation['init_profit_cum_oil'],
                                                                  dict_parameters_coefficients)
            gdf_project_wells_all = pd.concat([gdf_project_wells_all, gdf_project_wells], ignore_index=True)

    log_stage("Расчет запасов для проектных скважин")
    calculate_reserves_by_voronoi(list_zones, data_wells, map_rrr, save_directory)

    if switches['switch_economy']:
        log_stage(f"Загрузка исходных данных для расчета экономики")
        FEM, method_taxes, dict_NDD = load_economy_data(paths['path_economy'], name_field,
                                                        dict_parameters_coefficients['fluid_params']['gor'])
    else:
        FEM, method_taxes, dict_NDD = None, None, None

    for drill_zone in list_zones:
        if drill_zone.rating != -1:
            log_stage(f"Расчет запускных параметров, профиля добычи и экономики  проектных скважин зоны:"
                        f" {drill_zone.rating}")
            drill_zone.calculate_starting_rates(maps, dict_parameters_coefficients)
            drill_zone.calculate_production(data_decline_rate_stat,
                                            parameters_calculation['period_calculation'] * 12,
                                            well_params['day_in_month'],
                                            well_params['well_efficiency'])
            if switches['switch_economy']:
                drill_zone.calculate_economy(FEM, well_params, method_taxes, dict_NDD)

    log_stage(f"Выгрузка данных расчета:")
    summary_table = upload_data(name_field, name_object, save_directory, data_wells, maps, list_zones,
                                info_clusterization_zones, FEM, method_taxes, polygon_OI, data_history,
                                **{**load_data_param, **well_params, **switches})
    return summary_table, save_directory


if __name__ == '__main__':
    from app.local_parameters import main_parameters, constants
    main_parameters['paths']['save_directory'] = get_save_path("Infill_drilling")
    run_model(main_parameters, constants, total_stages=None)