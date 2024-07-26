from map import mapping
from input_output import input_data
from loguru import logger


if __name__ == '__main__':
    logger.add('logs.log')
    logger.info("Загрузка скважинных данных")
    data_history, data_wells = input_data(data_well_directory=r"C:\Users\Alina\Desktop\Python\!Работа IT ННГ\Infill_drilling\Infill_drilling\input_files\Крайнее_Ю1\Крайнее_Ю1.xlsx")
    logger.info("Обработка карт")
    mapping(maps_directory=r"C:/Users/Alina/Desktop/Python/!Работа IT ННГ/Infill_drilling/Infill_drilling/input_files/Крайнее_Ю1/Ascii grid grd",
            save_directory=r"C:/Users/Alina/Desktop/Python/!Работа IT ННГ/Infill_drilling/Infill_drilling/output/",
            data_wells=data_wells)
