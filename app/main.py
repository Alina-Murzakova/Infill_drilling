from map import mapping
from loguru import logger


if __name__ == '__main__':
    logger.add('logs.log')
    logger.info("Обработка карт")
    mapping(maps_directory="D:/Work/Programs_Python/Infill_drilling/input/Крайнее_Ю1/Ascii grid grd",
            save_directory="D:/Work/Programs_Python/Infill_drilling/output/")