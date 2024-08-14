from loguru import logger
from local_parameters import paths, parameters_calculation
from input_output import load_wells_data
from map import mapping
from drilling_zones import calculate_zones

if __name__ == '__main__':
    logger.add('logs.log', mode='w')

    logger.info("Инициализация локальных переменных")
    data_well_directory = paths["data_well_directory"]
    maps_directory = paths["maps_directory"]
    save_directory = paths["save_directory"]

    default_size_pixel = parameters_calculation["default_size_pixel"]
    epsilon = parameters_calculation["epsilon"]
    min_samples = parameters_calculation["min_samples"]
    percent_low = 100 - parameters_calculation["percent_top"]

    logger.info("Загрузка скважинных данных")
    _, data_wells = load_wells_data(data_well_directory=data_well_directory)

    logger.info("Загрузка ГФХ по пласту")

    logger.info("Загрузка и обработка карт")
    maps = mapping(maps_directory=maps_directory,
                   save_directory=save_directory,
                   data_wells=data_wells,
                   default_size_pixel=default_size_pixel)

    logger.info("Расчет зон")
    dict_zones = calculate_zones(maps,
                                 epsilon=epsilon,
                                 min_samples=min_samples,
                                 percent_low=percent_low)

    logger.info("Расчет запускных параметров в зонах")
