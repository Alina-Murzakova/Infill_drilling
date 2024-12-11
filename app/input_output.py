import os
import win32api

import pandas as pd
import numpy as np
from loguru import logger
from shapely.geometry import Point, LineString

from app.config import columns_name, gpch_column_name, dict_work_marker, sample_data_wells


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
    data_history = data_history[list(columns_name.keys())]
    data_history.columns = columns_name.values()

    # Подготовка файла
    data_history = data_history.fillna(0)
    # Удаление строк, где скважина еще не пробурена
    data_history = data_history[data_history.work_marker != "не пробурена"]
    data_history = data_history.sort_values(by=['well_number', 'date'], ascending=[True, False]).reset_index(drop=True)

    # Обработка координат // разделение на горизонтальные и вертикальные скважины
    data_history.loc[data_history["T3_x_geo"] == 0, 'T3_x_geo'] = data_history.T1_x_geo
    data_history.loc[data_history["T3_y_geo"] == 0, 'T3_y_geo'] = data_history.T1_y_geo
    data_history["length_geo"] = np.sqrt(np.power(data_history.T3_x_geo - data_history.T1_x_geo, 2)
                                         + np.power(data_history.T3_y_geo - data_history.T1_y_geo, 2))
    data_history["well_type"] = ""
    data_history.loc[data_history["length_geo"] < min_length_hor_well, "well_type"] = "vertical"
    data_history.loc[data_history["length_geo"] >= min_length_hor_well, "well_type"] = "horizontal"
    data_history.loc[data_history["well_type"] == "vertical", 'T3_x_geo'] = data_history.T1_x_geo
    data_history.loc[data_history["well_type"] == "vertical", 'T3_y_geo'] = data_history.T1_y_geo

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

    data_wells['init_water_cut'] = np.where(data_wells['init_Ql_rate'] > 0,
                                            (data_wells['init_Ql_rate'] - data_wells['init_Qo_rate']) /
                                            data_wells['init_Ql_rate'], 0)

    # Определение запускных параметров ТР
    params_TR = {'P_well': 'init_P_well_prod',
                 'P_reservoir': 'init_P_reservoir_prod',
                 'Ql_rate_TR': 'init_Ql_rate_TR',
                 'Qo_rate_TR': 'init_Qo_rate_TR',
                 'water_cut_TR': 'init_water_cut_TR'}

    for param, col in params_TR.items():
        data_param = (df_sort_date[df_sort_date.Ql_rate > 0][['well_number', param, 'date']]
                      .groupby('well_number')
                      .apply(lambda x: get_init_param_TR(x, param))
                      .reset_index(name=col))
        data_wells = data_wells.merge(data_param[['well_number', col]], how='left', on='well_number')
    data_wells = data_wells.fillna(0)

    # Расчет азимута для горизонтальных скважин
    data_wells['azimuth'] = data_wells.apply(calculate_azimuth, axis=1)

    # Словарь с данными о расчете // месторождение и объект
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
    data_wells.drop(columns=['field', "object", "objects"], inplace=True)

    # расчет Shapely объектов
    df_shapely = create_shapely_types(data_wells, list_names=['T1_x_geo', 'T1_y_geo', 'T3_x_geo', 'T3_y_geo'])
    data_wells[['POINT_T1_geo', 'POINT_T3_geo', 'LINESTRING_geo']] = df_shapely

    # дополнение data_wells всеми необходимыми колонками
    data_wells = pd.concat([sample_data_wells, data_wells])
    return data_history, data_wells, info


def load_geo_phys_properties(path_geo_phys_properties, name_field, name_object):
    """Создание словаря ГФХ для пласта"""
    # Загрузка файла
    df_geo_phys_properties = pd.read_excel(os.path.join(os.path.dirname(__file__), path_geo_phys_properties))
    # Переименование колонок
    df_geo_phys_properties = df_geo_phys_properties[list(gpch_column_name.keys())]
    df_geo_phys_properties.columns = list(map(lambda x: list(x.keys())[0], gpch_column_name.values()))
    # Подготовка файла
    df_geo_phys_properties = df_geo_phys_properties.fillna(0)
    df_geo_phys_properties.drop([0], inplace=True)  # Удаление строки с ед. изм.

    list_dict = gpch_column_name.values()
    df_geo_phys_properties = df_geo_phys_properties.astype(
        {k: v for list_item in list_dict for (k, v) in list_item.items()})

    # Удаление лишних столбцов
    list_columns = []
    for column in df_geo_phys_properties.columns:
        if 'del' in column:
            del df_geo_phys_properties[column]
        else:
            list_columns.append(column)

    df_geo_phys_properties = df_geo_phys_properties[df_geo_phys_properties.data_type == "в целом"]

    # добавляем строки со значениями по умолчанию (среднее по мр) для каждого месторождения
    dct = {'number': 'mean', 'object': lambda col: col.mode(), }
    groupby_cols = ['field']
    dct = {k: v for i in
           [{col: agg for col in df_geo_phys_properties.select_dtypes(tp).columns.difference(groupby_cols)} for tp, agg
            in dct.items()] for
           k, v in i.items()}
    agg = df_geo_phys_properties.groupby(groupby_cols).agg(**{k: (k, v) for k, v in dct.items()})
    agg['object'] = "default_properties"
    agg['field'] = agg.index

    df_geo_phys_properties = pd.concat([agg.reset_index(drop=True), df_geo_phys_properties])
    df_geo_phys_properties = df_geo_phys_properties[list_columns]

    df_geo_phys_properties_field_mean = df_geo_phys_properties[(df_geo_phys_properties.field == name_field)
                                                               & (df_geo_phys_properties.object ==
                                                                  "default_properties")]
    if df_geo_phys_properties_field_mean.empty:
        logger.error(f"В файле ГФХ нет данных по месторождению {name_field}")
        return None

    df_geo_phys_properties_field = df_geo_phys_properties[(df_geo_phys_properties.field == name_field)
                                                          & (df_geo_phys_properties.object == name_object)]
    if df_geo_phys_properties_field.shape[0] > 1:
        logger.error(f"В файле ГФХ больше одной строчки для объекта {name_field} месторождения {name_field}")
        return None
    else:
        if df_geo_phys_properties_field.empty:
            logger.info(f"В файле ГФХ не найден объект {name_field} месторождения {name_field}. "
                        f"Используются средние значения по месторождению для объекта.")
            dict_geo_phys_properties_field = df_geo_phys_properties_field_mean.iloc[0][5:].to_dict()
        else:
            dict_geo_phys_properties_field = df_geo_phys_properties_field.iloc[0][5:].to_dict()

        # Проверка наличия требуемых свойств
        list_properties = ['formation_compressibility', 'water_viscosity_in_situ',
                           'oil_viscosity_in_situ', 'oil_compressibility',
                           'water_compressibility', 'Bo', 'bubble_point_pressure',
                           'oil_density_at_surf']
        # для 'init_pressure' есть проверка при построении карты рисков, если оно 0,
        # то используется максимальное значение с карты
        for prop in list_properties:
            value = dict_geo_phys_properties_field[prop]
            value_mean = df_geo_phys_properties_field_mean[prop]
            if value <= 0:
                if value_mean > 0:
                    dict_geo_phys_properties_field[prop] = value_mean
                else:
                    logger.error(f"Свойство {prop} задано некорректно: {value}")
        return formatting_dict_geo_phys_properties(dict_geo_phys_properties_field)


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


"""___________Вспомогательные функции___________"""


def formatting_dict_geo_phys_properties(dict_geo_phys_properties):
    """Формирование словаря со всеми необходимыми свойствами из ГФХ в требуемых размерностях
    !!! указать размерности !!!
    """
    return {'reservoir_params': {'c_r': dict_geo_phys_properties['formation_compressibility'] / 100000,
                                 'P_init': dict_geo_phys_properties['init_pressure'] * 10},
            'fluid_params': {'mu_w': dict_geo_phys_properties['water_viscosity_in_situ'],
                             'mu_o': dict_geo_phys_properties['oil_viscosity_in_situ'],
                             'c_o': dict_geo_phys_properties['oil_compressibility'] / 100000,
                             'c_w': dict_geo_phys_properties['water_compressibility'] / 100000,
                             'Bo': dict_geo_phys_properties['Bo'],
                             'Pb': dict_geo_phys_properties['bubble_point_pressure'] * 10,
                             'rho': dict_geo_phys_properties['oil_density_at_surf']}}


def create_new_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path)
    return


def create_shapely_types(data_wells, list_names):
    """
    Создание фрейма geopandas на базе фрейма pandas
    Parameters
    ----------
    data_wells - DataFrame с основными данными скважин
    list_names - список названий колонок с координатами
    Returns
    -------
    DataFrame[columns = "POINT T1", "POINT T3", "LINESTRING"] - фрейм данных с добавленными shapely объектами
    """
    T1_x, T1_y, T3_x, T3_y = list_names
    df_result = pd.DataFrame()
    import warnings
    with warnings.catch_warnings(action='ignore', category=pd.errors.SettingWithCopyWarning):
        df_result["POINT_T1"] = list(map(lambda x, y: Point(x, y), data_wells[T1_x], data_wells[T1_y]))
        df_result["POINT_T3"] = list(map(lambda x, y: Point(x, y), data_wells[T3_x], data_wells[T3_y]))
        df_result["LINESTRING"] = list(map(lambda x, y: LineString([x, y]),
                                           df_result["POINT_T1"], df_result["POINT_T3"]))
        df_result["LINESTRING"] = np.where(df_result["POINT_T1"] == df_result["POINT_T3"], df_result["POINT_T1"],
                                           list(map(lambda x, y: LineString([x, y]),
                                                    df_result["POINT_T1"], df_result["POINT_T3"])))
    return df_result


def get_init_param_TR(df, column):
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
    if row['well_type'] != "horizontal":
        return None  # Возвращаем None для вертикальных скважин

    dX = row['T3_x_geo'] - row['T1_x_geo']  # Разность по оси X
    dY = row['T1_y_geo'] - row['T3_y_geo']  # Разность по оси Y
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
