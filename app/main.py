from map import mapping
from input_output import input_data
from loguru import logger

from local_parameters import paths

if __name__ == '__main__':
    logger.add('logs.log')

    logger.info("Инициализация локальных переменных")
    data_well_directory = paths["data_well_directory"]
    maps_directory = paths["maps_directory"]
    save_directory = paths["save_directory"]

    logger.info("Загрузка скважинных данных")
    data_history, data_wells = input_data(data_well_directory=data_well_directory)

    logger.info("Обработка карт")
    mapping(maps_directory=maps_directory, save_directory=save_directory, data_wells=data_wells)
