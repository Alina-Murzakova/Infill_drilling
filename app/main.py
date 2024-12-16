from loguru import logger

from app.ranking_drilling.starting_rates import calculate_permeability_fact_wells
from local_parameters import paths, parameters_calculation, default_well_params, default_coefficients
from input_output import load_wells_data, get_save_path, load_geo_phys_properties

from app.decline_rate.decline_rate import get_decline_rates
from app.maps_handler.functions import mapping, save_map_permeability_fact_wells
from well_active_zones import calculate_effective_radius
from drill_zones.drilling_zones import calculate_drilling_zones, save_picture_clustering_zones
from project_wells import save_ranking_drilling_to_excel, calculate_reserves_by_voronoi

if __name__ == '__main__':
    logger.add('logs.log', mode='w')

    logger.info("Инициализация локальных переменных")
    # Пути
    data_well_directory = paths["data_well_directory"]
    maps_directory = paths["maps_directory"]
    path_geo_phys_properties = paths["path_geo_phys_properties"]
    # Параметры кластеризации
    default_size_pixel = parameters_calculation["default_size_pixel"]
    epsilon = parameters_calculation["epsilon"]
    min_samples = parameters_calculation["min_samples"]
    percent_low = 100 - parameters_calculation["percent_top"]
    # Параметры для расстановки проектного фонда
    init_profit_cum_oil = parameters_calculation["init_profit_cum_oil"]
    buffer_project_wells = parameters_calculation["buffer_project_wells"] / default_size_pixel

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
    data_wells['permeability_fact'] = data_wells.apply(calculate_permeability_fact_wells,
                                                       args=(dict_parameters_coefficients,),
                                                       axis=1)
    logger.info("Сохранение карты фактической проницаемости через РБ")
    map_pressure = maps[type_map_list.index('pressure')]
    save_map_permeability_fact_wells(data_wells, map_pressure, f"{save_directory}/permeability_fact_wells.png")

    logger.info("Расчет радиусов дренирования и нагнетания для скважин")
    data_wells = calculate_effective_radius(data_wells, dict_properties=dict_parameters_coefficients['fluid_params'])

    logger.info("Оценка темпов падения для текущего фонда")
    # data_decline_rate_stat, _, _ = get_decline_rates(data_history, data_wells)

    logger.info("Расчет зон с высоким индексом бурения")
    list_zones, info_clusterization_zones = calculate_drilling_zones(maps=maps,
                                                                     epsilon=epsilon,
                                                                     min_samples=min_samples,
                                                                     percent_low=percent_low,
                                                                     data_wells=data_wells)

    type_calculated_maps = ['reservoir_score', 'potential_score', 'risk_score', 'opportunity_index']
    logger.info(f"Сохраняем исходные карты и рассчитанные в .png и .grd форматах ")
    for i, raster in enumerate(maps):
        if raster.type_map in type_calculated_maps:
            raster.save_img(f"{save_directory}/{raster.type_map}.png", data_wells)
            raster.save_grd_file(f"{save_directory}/{raster.type_map}.grd")
            if raster.type_map == 'opportunity_index':
                logger.info(f"Сохраняем .png карту OI с зонами")
                raster.save_img(f"{save_directory}/map_opportunity_index_with_zones.png", data_wells,
                                list_zones, info_clusterization_zones)
        else:
            raster.save_img(f"{save_directory}/{raster.type_map}.png", data_wells)

    map_rrr = maps[type_map_list.index('residual_recoverable_reserves')]
    logger.info("Начальное размещение проектных скважин")
    for drill_zone in list_zones:
        if drill_zone.rating != -1:
            drill_zone.get_init_project_wells(map_rrr, data_wells, init_profit_cum_oil,
                                              default_size_pixel, buffer_project_wells)

    logger.info("Расчет запасов для проектных скважин")
    calculate_reserves_by_voronoi(list_zones, data_wells, map_rrr, save_directory)

    for drill_zone in list_zones:
        if drill_zone.rating != -1:
            logger.info(f"Расчет запускных параметров проектных скважин зоны: {drill_zone.rating}")
            for project_well in drill_zone.list_project_wells:
                project_well.get_starting_rates(maps, dict_parameters_coefficients)
                project_well.get_production_profile(data_decline_rate_stat)

    logger.info(f"Сохраняем .png с начальным расположением проектного фонда в кластерах и карту ОИ с проектным фондом")
    save_picture_clustering_zones(list_zones, f"{save_directory}/init_project_wells.png", buffer_project_wells)
    map_opportunity_index = maps[type_map_list.index('residual_recoverable_reserves')]
    map_opportunity_index.save_img(f"{save_directory}/map_opportunity_index_with_project_wells.png", data_wells,
                                   list_zones, info_clusterization_zones, project_wells=True)
    save_ranking_drilling_to_excel(list_zones, f"{save_directory}/ranking_drilling.xlsx")
