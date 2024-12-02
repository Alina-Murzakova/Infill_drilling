import os
import win32api

import pandas as pd
import numpy as np
from loguru import logger
# import xlwings as xw

from app.config import MER_columns_name, gpch_column_name, dict_work_marker


@logger.catch
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
    Словарь с данными о расчете {field, object}
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

    # Добавление колонки с указанием как долго не работает скважина
    last_date = sorted(data_history.date.unique())[-1]
    first_date = sorted(data_history.date.unique())[0]
    data_wells_last_param['no_work_time'] = round((last_date - data_wells_last_param.date).dt.days / 29.3)

    # Скважины без добычи/закачки на последнюю дату
    df_diff = data_wells_last_date[~data_wells_last_date.well_number.isin(data_wells_last_param.well_number)]
    import warnings
    with warnings.catch_warnings(action='ignore', category=pd.errors.SettingWithCopyWarning):
        df_diff['no_work_time'] = round((last_date - first_date).days / 29.3)

    data_wells = pd.concat([data_wells_last_param, df_diff], ignore_index=True)

    # Нахождение накопленной добычи нефти и закачки
    df_sort_date = (data_history.copy().sort_values(by=['well_number', 'date'], ascending=[True, True])
                    .reset_index(drop=True))
    for column in ['Qo', 'Winj']:
        data_cumsum = df_sort_date[['well_number', column]].copy()
        data_cumsum[f'{column}_cumsum'] = data_cumsum[column].groupby(data_cumsum['well_number']).cumsum()
        data_cumsum = data_cumsum.groupby('well_number').tail(1)
        data_wells = data_wells.merge(data_cumsum[['well_number', f'{column}_cumsum']], how='left', on='well_number')
    for old, new in dict_work_marker.items():
        data_wells.work_marker = data_wells.work_marker.str.replace(old, new, regex=False)

    # Нахождение среднего стартового дебита за первые "first_months" месяцев
    data_first_rate = df_sort_date.copy()
    data_first_rate['cum_rate_liq'] = data_first_rate['Ql_rate'].groupby(data_first_rate['well_number']).cumsum()
    data_first_rate = data_first_rate[data_first_rate['cum_rate_liq'] != 0]
    data_first_rate = data_first_rate.groupby('well_number').head(first_months)
    data_first_rate = (data_first_rate[data_first_rate['Ql_rate'] != 0].groupby('well_number')
                       .agg(init_Qo_rate=('Qo_rate', 'mean'), init_Ql_rate=('Ql_rate', 'mean')).reset_index())

    data_wells = data_wells.merge(data_first_rate, how='left', on='well_number')
    data_wells[['init_Qo_rate', 'init_Ql_rate']] = data_wells[['init_Qo_rate', 'init_Ql_rate']].fillna(0)

    # Определение запускного забойного давления P_well
    data_P_well_init_prod = (df_sort_date[df_sort_date.Ql_rate > 0][['well_number', 'P_well', 'date']]
                             .groupby('well_number').apply(lambda x: get_P_well_init(x, 'P_well'))
                             .reset_index(name='P_well_init_prod'))

    data_P_well_init_inj = (df_sort_date[df_sort_date.Winj_rate > 0][['well_number', 'P_well', 'date']]
                            .groupby('well_number').apply(lambda x: get_P_well_init(x, 'P_well'))
                            .reset_index(name='P_well_init_inj'))

    data_wells = data_wells.merge(data_P_well_init_prod[['well_number', 'P_well_init_prod']], how='left', on='well_number')
    data_wells = data_wells.merge(data_P_well_init_inj[['well_number', 'P_well_init_inj']], how='left', on='well_number')
    data_wells = data_wells.fillna(0)

    data_wells['azimuth'] = data_wells.apply(calculate_azimuth, axis=1)

    # Словарь с данными о расчете
    field = list(set(data_wells.field.values))
    object_value = list(set(data_wells.object.values))
    if len(field) != 1:
        logger.error(f"Выгрузка содержит не одно месторождение: {field}")
    elif len(object_value) != 1:
        logger.error(f"Выгрузка содержит не один объект: {object_value}")
    else:
        field = field[0]
        object_value = object_value[0]
    info = {'field': field, "object_value": object_value}
    return data_history, data_wells, info

def get_P_well_init(df, column):
    """Получение Рзаб в первый или второй месяц работы скважины, если в первом нет"""
    first_value = df.iloc[0][column]
    if first_value > 0:
        return first_value

    # Проверка, если это единственная запись в скважине
    if len(df) == 1:
        return 0

    second_value = df.iloc[1][column]
    if second_value > 0 and (df['date'].iloc[1] - df['date'].iloc[0]).days / 31 <= 1:
        return second_value
    return 0

def calculate_azimuth(row):
    """Расчет азимута горизонтальной скважины"""
    if row['well type'] != "horizontal":
        return None  # Возвращаем None для вертикальных скважин

    dX = row['T3_x'] - row['T1_x']  # Разность по оси X
    dY = row['T3_y'] - row['T1_y']  # Разность по оси Y
    beta = np.degrees(np.arctan2(abs(dY), abs(dX)))

    # Определяем угол в зависимости от направления
    if dX > 0:
        if dY < 0:
            azimuth = 270 + beta
        else:
            azimuth = 270 - beta
    else:
        if dY < 0:
            azimuth = 90 - beta
        else:
            azimuth = 90 + beta

    # Приведение к диапазону [0, 360)
    azimuth = (360 - azimuth) % 360
    return azimuth


def load_geo_phys_properties(path_geo_phys_properties, name_field, name_object):
    """Создание словаря ГФХ для пласта"""
    # Загрузка файла
    df_geo_phys_properties = pd.read_excel(os.path.join(os.path.dirname(__file__), path_geo_phys_properties))

    # Переименование колонок
    df_geo_phys_properties = df_geo_phys_properties[list(gpch_column_name.keys())]
    df_geo_phys_properties.columns = gpch_column_name.values()
    # dtypes_column = list(map(lambda x: x[1], gpch_column_name.values()))
    # df.astype({'col1': 'int32'}).dtypes Изменить формат config!!!!

    # Подготовка файла
    df_geo_phys_properties = df_geo_phys_properties.fillna(0)
    # Удаление лишних столбцов
    for column in df_geo_phys_properties.columns:
        if 'del' in column:
            del df_geo_phys_properties[column]

    df_geo_phys_properties = df_geo_phys_properties[df_geo_phys_properties.data_type == "в целом"]
    df_geo_phys_properties = df_geo_phys_properties[(df_geo_phys_properties.field == name_field)
                                                    & (df_geo_phys_properties.object == name_object)]
    # # добавляем строки со значениями по умолчанию (среднее по мр) для каждого месторождения
    # type_dct = {str(k): list(v) for k, v in df_geo_phys_properties.groupby(df_geo_phys_properties.dtypes, axis=1)}
    # df_geo_phys_properties_mean = df_geo_phys_properties.groupby('field').mean()
    # df_geo_phys_properties_mean['object'] = "default_properties"

    if df_geo_phys_properties.empty:
        logger.error(f"В файле ГФХ не найден объект {name_field} месторождения {name_field}")
    elif df_geo_phys_properties.shape[0] > 1:
        logger.error(f"В файле ГФХ больше одной строчки для объекта {name_field} месторождения {name_field}")
    else:
        dict_geo_phys_properties = df_geo_phys_properties.iloc[0][5:].to_dict()
        return dict_geo_phys_properties


def create_new_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path)
    return


def get_save_path(program_name: str = "default", field: str = "field", object_value: str = "object") -> str:
    """
    Получение пути на запись
    :return:
    """
    path_program = os.getcwd()
    # Проверка возможности записи в директорию программы
    if os.access(path_program, os.W_OK):
        if "\\app" in path_program:
            path_program = path_program.replace("\\app", "")
        if "\\drill_zones" in path_program:
            path_program = path_program.replace("\\drill_zones", "")
        save_path = f"{path_program}\\output\\{field}_{object_value}"
    else:
        # Поиск другого диска с возможностью записи: D: если он есть и C:, если он один
        # В будущем можно исправить с запросом на сохранение
        drives = win32api.GetLogicalDriveStrings()  # получение списка дисков
        save_drive = []
        list_drives = [drive for drive in drives.split('\\\000')[:-1] if 'D:' in drive]
        if len(list_drives) >= 1:
            save_drive = list_drives[0]
        else:
            list_drives = [drive for drive in drives.split('\\\000')[:-1] if 'C:' in drive]
            if len(list_drives) >= 1:
                save_drive = list_drives[0]
            else:
                logger.error(PermissionError)

        current_user = os.getlogin()
        profile_dir = [dir_ for dir_ in os.listdir(save_drive) if dir_.lower() == "profiles"
                       or dir_.upper() == "PROFILES"]

        if len(profile_dir) < 1:
            save_path = f"{save_drive}\\{program_name}_output\\{field}_{object_value}"
        else:
            save_path = (f"{save_drive}\\{profile_dir[0]}\\{current_user}\\"
                         f"{program_name}_output\\{field}_{object_value}")

    create_new_dir(save_path)
    return save_path
