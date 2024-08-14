import os
import pandas as pd
import numpy as np
# import xlwings as xw

from config import MER_columns_name


def load_wells_data(data_well_directory, min_length_hor_well=150, first_months=6):
    """
    Функция, которая обрабатывает выгрузку МЭР (выгрузка по датам по всем скважинам//параметры задаются пользователем)
    Parameters
    ----------
    data_well_directory - путь к выгрузке
    min_length_hor_well - максимальная длина ствола ННС
    first_months - количество первых месяцев работы для определения стартового дебита нефти

    Returns
    -------
    Фрейм с обработанной полной историей скважин
    Фрейм с параметрами добычи на последнюю дату работы для всех скважин
    """
    # Загрузка файла
    # data_history = pd.read_excel(os.path.join(os.path.dirname(__file__), data_well_directory))
    data_history = pd.DataFrame()
    xls = pd.ExcelFile(os.path.join(os.path.dirname(__file__), data_well_directory))
    for sheet_name in xls.sheet_names:
        if "Данные" in sheet_name:
            df = xls.parse(sheet_name)
            data_history = pd.concat([data_history, df], ignore_index=True)

    # Переименование колонок
    data_history = data_history[list(MER_columns_name.keys())]
    data_history.columns = MER_columns_name.values()

    # Подготовка файла
    data_history = data_history.fillna(0)
    # Удаление строк, где скважина еще не пробурена
    data_history = data_history[data_history.work_marker != "не пробурена"]
    data_history = data_history.sort_values(by=['well_number', 'date'], ascending=[True, False]).reset_index(drop=True)

    # Обработка координат // разделение на горизонтальные и вертикальные скважины
    data_history.loc[data_history["T3_x"] == 0, 'T3_x'] = data_history.T1_x
    data_history.loc[data_history["T3_y"] == 0, 'T3_y'] = data_history.T1_y
    data_history["length of well T1-3"] = np.sqrt(np.power(data_history.T3_x - data_history.T1_x, 2)
                                                  + np.power(data_history.T3_y - data_history.T1_y, 2))
    data_history["well type"] = ""
    data_history.loc[data_history["length of well T1-3"] < min_length_hor_well, "well type"] = "vertical"
    data_history.loc[data_history["length of well T1-3"] >= min_length_hor_well, "well type"] = "horizontal"
    data_history.loc[data_history["well type"] == "vertical", 'T3_x'] = data_history.T1_x
    data_history.loc[data_history["well type"] == "vertical", 'T3_y'] = data_history.T1_y
    # del data_history["length of well T1-3"]

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
