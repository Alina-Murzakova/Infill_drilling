from loguru import logger
from local_parameters import paths
from input_output import load_wells_data
from map import mapping
from drilling_zones import calculate_zones

if __name__ == '__main__':
    logger.add('logs.log')

    logger.info("Инициализация локальных переменных")
    data_well_directory = paths["data_well_directory"]
    maps_directory = paths["maps_directory"]
    save_directory = paths["save_directory"]

    logger.info("Загрузка скважинных данных")
    data_history, data_wells = load_wells_data(data_well_directory=data_well_directory)

    logger.info("Загрузка и обработка карт")
    maps = mapping(maps_directory=maps_directory, save_directory=save_directory, data_wells=data_wells)

    logger.info("Расчет зон")
    calculate_zones(maps)
