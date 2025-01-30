import math
from loguru import logger

from app.ranking_drilling.starting_rates import get_df_permeability_fact_wells
from local_parameters import main_parameters, constants
from input_output.input import load_wells_data, load_geo_phys_properties, load_frac_info

from app.decline_rate.decline_rate import get_decline_rates
from app.maps_handler.functions import mapping
from well_active_zones import calculate_effective_radius
from drill_zones.drilling_zones import calculate_drilling_zones
from project_wells import calculate_reserves_by_voronoi
from input_output.output import get_save_path, upload_data

if __name__ == '__main__':
    logger.add('logs.log', mode='w')

    logger.info("Инициализация локальных переменных")
    # Пути
    paths = main_parameters['paths']
    # Параметры расчета
    parameters_calculation = main_parameters['parameters_calculation']
    # Параметры для скважин РБ
    well_params = main_parameters['well_params']

    # Константы расчета
    load_data_param = constants['load_data_param']
    default_coefficients = constants['default_coefficients']
    default_well_params = constants['default_well_params']
    well_params.update(constants['default_project_well_params'])

    logger.info("Загрузка скважинных данных")
    (data_history, data_wells,
     info_object_calculation) = load_wells_data(data_well_directory=paths["data_well_directory"],
                                                first_months=load_data_param['first_months'])
    name_field, name_object = info_object_calculation.get("field"), info_object_calculation.get("object_value")
    save_directory = get_save_path("Infill_drilling", name_field, name_object.replace('/', '-'))

    logger.info(f"Загрузка ГФХ по пласту {name_object.replace('/', '-')} месторождения {name_field}")
    dict_parameters_coefficients = load_geo_phys_properties(paths["path_geo_phys_properties"], name_field, name_object)
    dict_parameters_coefficients.update({'well_params': well_params,
                                         'default_well_params': default_well_params,
                                         'coefficients': default_coefficients})
    logger.info(f"Загрузка фрак-листов")
    data_wells, dict_parameters_coefficients = load_frac_info(paths["path_frac"], data_wells, name_object,
                                                              dict_parameters_coefficients)

    logger.info("Загрузка и обработка карт")
    maps, data_wells = mapping(maps_directory=paths["maps_directory"],
                               data_wells=data_wells,
                               dict_properties=dict_parameters_coefficients['reservoir_params'],
                               **load_data_param)
    default_size_pixel = maps[0].geo_transform[1]  # размер ячейки после загрузки всех карт
    type_map_list = list(map(lambda raster: raster.type_map, maps))

    logger.info("Расчет радиусов дренирования и нагнетания для скважин")
    data_wells = calculate_effective_radius(data_wells, dict_properties=dict_parameters_coefficients)

    logger.info("Расчет проницаемости для фактических скважин через РБ")
    (data_wells,
     dict_parameters_coefficients) = get_df_permeability_fact_wells(data_wells, dict_parameters_coefficients,
                                                                    switch=load_data_param['switch_permeability_fact'])

    logger.info("Оценка темпов падения для текущего фонда")
    data_decline_rate_stat, _, _ = get_decline_rates(data_history, data_wells)

    logger.info("Расчет зон с высоким индексом бурения")
    # Параметры кластеризации
    epsilon = parameters_calculation["min_radius"] / default_size_pixel
    min_samples = int(parameters_calculation["sensitivity_quality_drill"] / 100 * epsilon ** 2 * math.pi)
    percent_low = 100 - parameters_calculation["percent_top"]
    list_zones, info_clusterization_zones = calculate_drilling_zones(maps=maps,
                                                                     epsilon=epsilon,
                                                                     min_samples=min_samples,
                                                                     percent_low=percent_low,
                                                                     data_wells=data_wells)

    map_rrr = maps[type_map_list.index('residual_recoverable_reserves')]
    logger.info("Начальное размещение проектных скважин")
    well_params['buffer_project_wells'] = well_params['buffer_project_wells'] / default_size_pixel
    for drill_zone in list_zones:
        if drill_zone.rating != -1:
            drill_zone.get_init_project_wells(map_rrr, data_wells,
                                              default_size_pixel,
                                              parameters_calculation['init_profit_cum_oil'],
                                              dict_parameters_coefficients)

    logger.info("Расчет запасов для проектных скважин")
    calculate_reserves_by_voronoi(list_zones, data_wells, map_rrr, save_directory)

    for drill_zone in list_zones:
        if drill_zone.rating != -1:
            logger.info(f"Расчет запускных параметров и профиля добычи проектных скважин зоны: {drill_zone.rating}")
            for project_well in drill_zone.list_project_wells:
                project_well.get_starting_rates(maps, dict_parameters_coefficients)
                project_well.get_production_profile(data_decline_rate_stat,
                                                    parameters_calculation['period_calculation'] * 12,
                                                    well_params['day_in_month'],
                                                    well_params['well_efficiency'])

    logger.info(f"Выгрузка данных расчета:")
    upload_data(save_directory, data_wells, maps, list_zones, info_clusterization_zones,
                **{**load_data_param, **well_params})
    logger.info(f"Загрузка исходных данных для расчета экономики")