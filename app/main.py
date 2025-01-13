from loguru import logger

from app.ranking_drilling.starting_rates import get_df_permeability_fact_wells
from local_parameters import paths, parameters_calculation, default_well_params, default_coefficients
from input_output.input import load_wells_data, load_geo_phys_properties

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
    data_well_directory = paths["data_well_directory"]
    maps_directory = paths["maps_directory"]
    path_geo_phys_properties = paths["path_geo_phys_properties"]
    path_economy = paths['path_economy']

    # Параметры кластеризации
    default_size_pixel = parameters_calculation["default_size_pixel"]
    epsilon = parameters_calculation["epsilon"]
    min_samples = parameters_calculation["min_samples"]
    percent_low = 100 - parameters_calculation["percent_top"]

    # Параметры для расстановки проектного фонда
    init_profit_cum_oil = parameters_calculation["init_profit_cum_oil"]
    buffer_project_wells = parameters_calculation["buffer_project_wells"] / default_size_pixel

    # Параметры расчета потока
    period_calculation = parameters_calculation["period_calculation"]
    start_date = parameters_calculation["start_date"]

    logger.info("Загрузка скважинных данных")
    data_history, data_wells, info_object_calculation = load_wells_data(data_well_directory=data_well_directory)
    name_field, name_object = info_object_calculation.get("field"), info_object_calculation.get("object_value")
    save_directory = get_save_path("Infill_drilling", name_field, name_object.replace('/', '-'))

    logger.info(f"Загрузка ГФХ по пласту {name_object.replace('/', '-')} месторождения {name_field}")
    dict_parameters_coefficients = load_geo_phys_properties(path_geo_phys_properties, name_field, name_object)
    dict_parameters_coefficients.update({"well_params": default_well_params, 'coefficients': default_coefficients})

    logger.info("Загрузка и обработка карт")
    maps, data_wells = mapping(maps_directory=maps_directory,
                               data_wells=data_wells,
                               dict_properties=dict_parameters_coefficients['reservoir_params'],
                               default_size_pixel=default_size_pixel)
    type_map_list = list(map(lambda raster: raster.type_map, maps))

    logger.info("Расчет проницаемости для фактических скважин через РБ")
    data_wells, dict_parameters_coefficients = get_df_permeability_fact_wells(data_wells, dict_parameters_coefficients,
                                                                              switch=True)

    logger.info("Расчет радиусов дренирования и нагнетания для скважин")
    data_wells = calculate_effective_radius(data_wells, dict_properties=dict_parameters_coefficients['fluid_params'])

    logger.info("Оценка темпов падения для текущего фонда")
    data_decline_rate_stat, _, _ = get_decline_rates(data_history, data_wells)

    logger.info("Расчет зон с высоким индексом бурения")
    list_zones, info_clusterization_zones = calculate_drilling_zones(maps=maps,
                                                                     epsilon=epsilon,
                                                                     min_samples=min_samples,
                                                                     percent_low=percent_low,
                                                                     data_wells=data_wells)

    map_rrr = maps[type_map_list.index('residual_recoverable_reserves')]
    logger.info("Начальное размещение проектных скважин")
    for drill_zone in list_zones:
        if drill_zone.rating != -1:
            drill_zone.get_init_project_wells(map_rrr, data_wells, init_profit_cum_oil,
                                              default_size_pixel, buffer_project_wells, dict_parameters_coefficients)

    logger.info("Расчет запасов для проектных скважин")
    calculate_reserves_by_voronoi(list_zones, data_wells, map_rrr, save_directory)

    for drill_zone in list_zones:
        if drill_zone.rating != -1:
            logger.info(f"Расчет запускных параметров и профиля добычи проектных скважин зоны: {drill_zone.rating}")
            for project_well in drill_zone.list_project_wells:
                project_well.get_starting_rates(maps, dict_parameters_coefficients)
                project_well.get_production_profile(data_decline_rate_stat, period_calculation * 12)

    logger.info(f"Выгрузка данных расчета:")
    upload_data(save_directory, data_wells, maps, list_zones, info_clusterization_zones, buffer_project_wells)

    logger.info(f"Загрузка исходных данных для расчета экономики")
