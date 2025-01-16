import os
import pandas as pd
import numpy as np

from loguru import logger
from shapely.geometry import Point, LineString

from app.config import columns_name, gpch_column_name, dict_work_marker, sample_data_wells, macroeconomics_rows_name


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


@logger.catch
def load_economy_data(economy_path):
    # Инициализируем необходимые переменные
    with pd.ExcelFile(economy_path) as xls:
        # коэффициенты Кв Кз Кд	Кдв	Ккан Кк	Кман Кабдт
        coefficients = pd.read_excel(xls, sheet_name="Налоги_константы", header=None)
        macroeconomics = pd.read_excel(xls, sheet_name="Макропараметры", header=3)  # Основная макра
        # месторождения с НДД
        reservoirs_NDD = pd.read_excel(xls, sheet_name="МР с НДД", header=None)
        # НРФ для уделок
        # df_fpa = pd.read_excel(economy_path + "\\НРФ.xlsx", sheet_name="Ш-01.02.01.07-01, вер. 1.0",
        #                        usecols=name_columns_FPA, header=4).fillna(0)
        # business_plan = pd.read_excel(economy_path + "\\Макра_оперативная_БП.xlsx", usecols="A, N:R",
        #                               header=3)  # Данные по макре за следующие 5 лет (напр: 2025, 26, 27, 28, 29)
        # business_plan_long = pd.read_excel(economy_path + "\\Макра_долгосрочная.xlsx", usecols="A, H:M",
        #                                    header=3)  # Данные по макре за следующие 6 лет (напр: 2030 - 35)

    # Подготовка файлов
    name_first_column = macroeconomics.columns[0]
    macroeconomics = macroeconomics.iloc[:, ~macroeconomics.columns.str.match('Unnamed').fillna(False)]
    macroeconomics = macroeconomics[macroeconomics[name_first_column].isin(macroeconomics_rows_name.keys())]
    macroeconomics.replace(macroeconomics_rows_name, inplace=True)
    macroeconomics = macroeconomics.fillna(method='ffill', axis=1).reset_index(drop=True)

    urals = macroeconomics[macroeconomics[name_first_column] == 'Urals'].values.flatten()[1:]
    exchange_rate = macroeconomics[macroeconomics[name_first_column] == 'exchange_rate'].values.flatten()[1:]
    base_rate_MET = macroeconomics[macroeconomics[name_first_column] == 'base_rate_MET'].values.flatten()[1:]
    K_k = macroeconomics[macroeconomics[name_first_column] == 'K_k'].values.flatten()[1:]
    K_abdt = macroeconomics[macroeconomics[name_first_column] == 'K_abdt'].values.flatten()[1:]
    K_man = macroeconomics[macroeconomics[name_first_column] == 'K_man'].values.flatten()[1:]
    # Расчет НДПИ
    # k_c = (urals - 15) * exchange_rate / 261
    # MET = base_rate_MET * k_c - 'Кндпи(до маневра)' * k_c * (1 - k_d * k_v * k_z * k_kan) + K_k + K_abdt + K_man
    #
    # macroeconomics = macroeconomics.fillna(method='ffill', axis=1).reset_index(drop=True)
    #
    # macroeconomics.loc[macroeconomics.shape[0] + 1] = \
    #     [well, coefficients_Ql_rate, coefficients_Qo_rate, cumulative_oil_production]
    pass


"""___________Вспомогательные функции___________"""


def formatting_dict_geo_phys_properties(dict_geo_phys_properties):
    """
    Формирование словаря со всеми необходимыми свойствами из ГФХ в требуемых размерностях

    - reservoir_params:
    c_r - сжимаемость породы | (1/МПа)×10-4 --> 1/атм
    P_init - текущее пластовое давление | МПа --> атм

    - fluid_params:
    mu_w - вязкость воды | сП или мПа*с
    mu_o - вязкость нефти | сП или мПа*с
    c_o - сжимаемость нефти | (1/МПа)×10-4 --> 1/атм
    c_w - сжимаемость воды | (1/МПа)×10-4 --> 1/атм
    Bo - объемный коэффициент расширения нефти | м3/м3
    Pb - давление насыщения | МПа --> атм
    rho - плотность нефти | г/см3
    """
    return {'reservoir_params': {'c_r': dict_geo_phys_properties['formation_compressibility'] / 100000,
                                 'P_init': dict_geo_phys_properties['init_pressure'] * 10,
                                 'k_h': dict_geo_phys_properties['permeability']},
            'fluid_params': {'mu_w': dict_geo_phys_properties['water_viscosity_in_situ'],
                             'mu_o': dict_geo_phys_properties['oil_viscosity_in_situ'],
                             'c_o': dict_geo_phys_properties['oil_compressibility'] / 100000,
                             'c_w': dict_geo_phys_properties['water_compressibility'] / 100000,
                             'Bo': dict_geo_phys_properties['Bo'],
                             'Pb': dict_geo_phys_properties['bubble_point_pressure'] * 10,
                             'rho': dict_geo_phys_properties['oil_density_at_surf']}}


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
