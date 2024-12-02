from loguru import logger

from local_parameters import paths, parameters_calculation
from input_output import load_wells_data, get_save_path, load_geo_phys_properties
from app.decline_rate.decline_rate import get_decline_rate
from app.maps_handler.functions import mapping
from drilling_zones import calculate_drilling_zones
from well_active_zones import calculate_effective_radius

if __name__ == '__main__':
    logger.add('logs.log', mode='w')

    logger.info("Инициализация локальных переменных")
    data_well_directory = paths["data_well_directory"]
    maps_directory = paths["maps_directory"]
    path_geo_phys_properties = paths["path_geo_phys_properties"]

    default_size_pixel = parameters_calculation["default_size_pixel"]
    epsilon = parameters_calculation["epsilon"]
    min_samples = parameters_calculation["min_samples"]
    percent_low = 100 - parameters_calculation["percent_top"]

    logger.info("Загрузка скважинных данных")
    data_history, data_wells, info_object_calculation = load_wells_data(data_well_directory=data_well_directory)
    name_field, name_object = info_object_calculation.get("field"), info_object_calculation.get("object_value")
    save_directory = get_save_path("Infill_drilling", name_field, name_object.replace('/', '-'))

    logger.info(f"Загрузка ГФХ по пласту {name_object.replace('/', '-')} месторождения {name_field}")
    dict_geo_phys_properties = load_geo_phys_properties(path_geo_phys_properties, name_field, name_object)

    logger.info("Загрузка и обработка карт")
    maps = mapping(maps_directory=maps_directory,
                   data_wells=data_wells,
                   dict_properties=dict_geo_phys_properties,
                   default_size_pixel=default_size_pixel)

    logger.info("Расчет радиусов дренирования и нагнетания для скважин")
    data_wells = calculate_effective_radius(data_wells, dict_geo_phys_properties, maps)

    logger.info("Оценка темпов падения для текущего фонда")
    data_decline_rate_stat, data_history_decline_rate, _ = get_decline_rate(data_history, data_wells)
    data_decline_rate_stat.to_pickle("data_decline_rate.pkl")
    data_history_decline_rate.to_pickle("data_history_decline_rate.pkl")

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
