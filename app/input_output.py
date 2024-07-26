import os
import pandas as pd
import numpy as np
import xlwings as xw

# Названия столбцов в Excel
date = 'Дата'
well_number = '№ скважины'
field = 'Месторождение'
objects = 'Объекты работы'
work_marker = 'Характер работы'
well_status = 'Состояние'
Qo_rate = 'Дебит нефти за последний месяц, т/сут'
Ql_rate = 'Дебит жидкости за последний месяц, т/сут'
water_cut = 'Обводненность за посл.месяц, % (вес)'
Winj_rate = 'Приемистость за последний месяц, м3/сут'
time_work = "Время работы, часы"
P_well = "Забойное давление (ТР), атм"
P_pressure = 'Пластовое давление (ТР), атм'
x1 = "Координата X"
y1 = "Координата Y"
x2 = "Координата забоя Х (по траектории)"
y2 = "Координата забоя Y (по траектории)"

def input_data(data_well_directory):
    data_history = pd.read_excel(os.path.join(os.path.dirname(__file__), data_well_directory))  # Открытие экселя
    data_history = data_history.fillna(0)
    data_history = data_history[data_history[work_marker] != "не пробурена"]  # удаление строк, где скважина еще не пробурена
    data_history = data_history.sort_values(by=[well_number, date], ascending=[True, False]).reset_index(drop=True)

    wells = data_history[well_number].unique() # список скважин
    data_wells = data_history.copy()
    data_wells = data_wells[(data_wells[Ql_rate] > 0) | (data_wells[Winj_rate] > 0)]
    data_wells_last_param = data_wells.groupby(well_number).nth(0).reset_index(drop=True) # скважины с добычей/закачкой и параметры работы в последний рабочий месяц
    data_wells_last_date = data_history.groupby(well_number).nth(0).reset_index(drop=True) # все скважины

    df_diff = data_wells_last_date[~data_wells_last_date[well_number].isin(data_wells_last_param[well_number])]

    data_wells = pd.concat([data_wells_last_param, df_diff], ignore_index=True)
    # data_wells[x2] = np.where(data_wells[x2] == 0, np.nan, data_wells[x2])
    # data_wells[y2] = np.where(data_wells[y2] == 0, np.nan, data_wells[y2])

    return data_history, data_wells

