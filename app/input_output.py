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
    #  Количество первых месяцев работы для определения стартового дебита нефти
    first_months = 6

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

    # Нахождение среднего стартового дебита за первые "first_months" месяцев
    data_first_rate = (data_history.copy().sort_values(by=['well_number', 'date'], ascending=[True, True])
                       .reset_index(drop=True))
    data_first_rate['cum_rate_liq'] = data_first_rate['Ql_rate'].groupby(data_first_rate['well_number']).cumsum()
    data_first_rate = data_first_rate[data_first_rate['cum_rate_liq'] != 0]
    data_first_rate = data_first_rate.groupby('well_number').head(first_months)
    data_first_rate = (data_first_rate[data_first_rate['Ql_rate'] != 0].groupby('well_number')
                       .agg(init_Qo_rate=('Qo_rate', 'mean'), init_Ql_rate=('Ql_rate', 'mean')).reset_index())

    data_wells = data_wells.merge(data_first_rate, how='left', on='well_number')
    data_wells[['init_Qo_rate', 'init_Ql_rate']] = data_wells[['init_Qo_rate', 'init_Ql_rate']].fillna(0)

    return data_history, data_wells
