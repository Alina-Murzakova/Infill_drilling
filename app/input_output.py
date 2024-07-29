import os
import pandas as pd
import numpy as np
import xlwings as xw
from config import MER_columns_name


def input_data(data_well_directory):
    """
    Функция, которая обрабатывает выгрузку МЭР (выгрузка по датам по всем скважинам//параметры задаются пользователем)
    Parameters
    ----------
    data_well_directory - путь к выгрузке

    Returns
    -------
    Фрейм с параметрами добычи на последнюю дату работы для всех скважин
    """
    # Загрузка файла
    data_history = pd.read_excel(os.path.join(os.path.dirname(__file__), data_well_directory))

    # Переименовнаие колонок
    data_history = data_history[list(MER_columns_name.keys())]
    data_history.columns = MER_columns_name.values()

    # Подготовка файла
    data_history = data_history.fillna(0)
    # Удаление строк, где скважина еще не пробурена
    data_history = data_history[data_history.work_marker != "не пробурена"]
    data_history = data_history.sort_values(by=['well_number', 'date'], ascending=[True, False]).reset_index(drop=True)

    data_wells = data_history.copy()
    data_wells = data_wells[(data_wells.Ql_rate > 0) | (data_wells.Winj_rate > 0)]
    # Скважины с добычей/закачкой и параметры работы в последний рабочий месяц
    data_wells_last_param = data_wells.groupby('well_number').nth(0).reset_index(drop=True)

    # Все скважины на последнюю дату
    data_wells_last_date = data_history.groupby('well_number').nth(0).reset_index(drop=True)
    df_diff = data_wells_last_date[~data_wells_last_date.well_number.isin(data_wells_last_param.well_number)]

    data_wells = pd.concat([data_wells_last_param, df_diff], ignore_index=True)

    return data_history, data_wells
