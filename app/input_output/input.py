import os
import pandas as pd
import numpy as np

from loguru import logger
from shapely.geometry import Point, LineString

from app.config import (columns_name, gpch_column_name, dict_work_marker, sample_data_wells, columns_name_frac,
                        macroeconomics_rows_name, OPEX_rows_name, workover_wellservice_rows_name)
from app.economy.financial_model import FinancialEconomicModel


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
    data_history['well_number'] = data_history['well_number'].astype(str)

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

    # Скважины с добычей и дата первой добычи
    data_wells_prod = data_wells[(data_wells.Ql_rate > 0)]
    data_wells_first_production = data_wells_prod.groupby('well_number')['date'].min().reset_index()
    data_wells_first_production.rename(columns={'date': 'first_production_date'}, inplace=True)

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
    data_wells = data_wells.merge(data_wells_first_production, on='well_number', how='left')

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
                       .agg(init_Qo_rate=('Qo_rate', 'mean'), init_Ql_rate=('Ql_rate', 'mean'),
                            init_Qo_rate_TR=('Qo_rate_TR', lambda x: x[x != 0].mean()),
                            init_Ql_rate_TR=('Ql_rate_TR', lambda x: x[x != 0].mean()),
                            init_P_well_prod=('P_well', lambda p_well: filter_pressure(p_well, data_first_rate.loc[
                                p_well.index, 'P_reservoir'], type_pressure='P_well')),
                            init_P_reservoir_prod=('P_reservoir', lambda p_res: filter_pressure(
                                data_first_rate.loc[p_res.index, 'P_well'], p_res, type_pressure='P_reservoir')))
                       .reset_index())

    data_wells = data_wells.merge(data_first_rate, how='left', on='well_number')
    data_wells = data_wells.fillna(0)

    data_wells['init_water_cut'] = np.where(data_wells['init_Ql_rate'] > 0,
                                            (data_wells['init_Ql_rate'] - data_wells['init_Qo_rate']) /
                                            data_wells['init_Ql_rate'], 0)

    data_wells['init_water_cut_TR'] = np.where(data_wells['init_Ql_rate_TR'] > 0,
                                               (data_wells['init_Ql_rate_TR'] - data_wells['init_Qo_rate_TR']) /
                                               data_wells['init_Ql_rate_TR'], 0)

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

    # оставляем для расчета только скважины с не нулевой накопленной добычей и закачкой по объекту
    data_wells = (data_wells[(data_wells['Qo_cumsum'] > 0) | (data_wells['Winj_cumsum'] > 0)].reset_index(drop=True))
    return data_history, data_wells, info


def load_frac_info(path_frac, data_wells, name_object, dict_parameters_coefficients):
    """
    Загрузка фрак-листа NGT и получение средних параметров по трещинам
    Parameters
    ----------
    path_frac - путь к выгрузке фрак-лист
    data_wells - фрейм данных по скважинам
    name_object - рассматриваемый объект
    dict_parameters_coefficients - словарь свойств и параметров по умолчанию

    Returns
    -------
    Фрейм данных по скважинам с информацией по фракам, обновленный словарь средних свойств и параметров по умолчанию
    """

    pattern = r"(\d+)\s*из\s*(\d+)"
    # Загрузка файла
    data_frac = pd.read_excel(os.path.join(os.path.dirname(__file__), path_frac), header = 1)
    # Переименование колонок
    data_frac = data_frac[list(columns_name_frac.keys())]
    data_frac.columns = columns_name_frac.values()
    # Подготовка файла
    data_frac['well_number'] = data_frac['well_number'].ffill()  # протягивание номера скважины в объединенных ячейках
    data_frac = data_frac.fillna(0)
    data_frac['well_number'] = data_frac['well_number'].astype(str)
    data_frac = data_frac[data_frac['date'] != 0]
    data_frac["date"] = pd.to_datetime(data_frac["date"], errors="coerce")
    data_frac = data_frac[data_frac['object'] == name_object]  # оставляем фраки на рассматриваемый объект

    # подтягивание дополнительной информации по скважинам
    some_data_wells = data_wells[['well_number', 'well_type', 'length_geo', 'first_production_date']].copy()
    data_frac = data_frac.merge(some_data_wells, on='well_number', how='left')

    # преобразование столбца с датой первой добычи в формат даты
    data_frac['first_production_date'] = data_frac['first_production_date'].replace(0, pd.NaT)
    data_frac['first_production_date'] = pd.to_datetime(data_frac['first_production_date'], errors='coerce')
    # оставляем фраки на начало работы скважины (без рефраков)
    data_frac = data_frac[data_frac['date'].dt.to_period('M') <= (data_frac['first_production_date'] +
                                                                  pd.DateOffset(months=1)).dt.to_period('M')]

    data_frac[["current_frac", "total_Frac"]] = data_frac["comment"].str.extract(pattern)
    data_frac = (data_frac.groupby('well_number').agg(FracCount=('date', 'count'),
                                                      xfr=('xfr', lambda x: round(x[x != 0].mean(), 1)),
                                                      w_f=('w_f', lambda x: round(x[x != 0].mean(), 1)),
                                                      total_Frac=('total_Frac', 'first'),
                                                      well_type=('well_type', 'first'),
                                                      length_geo=('length_geo', 'first')).reset_index())
    data_frac = data_frac.fillna(0)
    data_frac['total_Frac'] = data_frac['total_Frac'].astype(int)
    data_frac['FracCount'] = np.where(data_frac['total_Frac'] != 0, data_frac['total_Frac'], data_frac['FracCount'])
    data_frac['FracCount'] = np.where((data_frac['FracCount'] > 0) & (data_frac['well_type'] == 'vertical'), 1,
                                      data_frac['FracCount'])
    data_frac['length_FracStage'] = np.where(data_frac['well_type'] == 'horizontal',
                                             np.round(data_frac['length_geo'] / data_frac['FracCount'], 0), 0)
    avg_xfr = np.mean(data_frac[data_frac['xfr'] > 0]['xfr'])
    avg_w_f = np.mean(data_frac[data_frac['w_f'] > 0]['w_f'])
    avg_length_FracStage = np.mean(data_frac[data_frac['length_FracStage'] > 0]['length_FracStage'])

    data_frac['xfr'] = np.where((data_frac['xfr'] == 0) & (data_frac['FracCount'] > 0), round(avg_xfr, 1),
                                data_frac['xfr'])
    data_frac['w_f'] = np.where((data_frac['w_f'] == 0) & (data_frac['FracCount'] > 0), round(avg_w_f, 1),
                                data_frac['w_f'])

    data_frac = data_frac.drop(['total_Frac', 'well_type', 'length_geo'], axis=1)
    # Перезапись значений по умолчанию xfr и w_f и length_FracStage по объекту на средние по фактическому фонду
    # if all(x is not pd.isna(x) and x != 0 for x in [avg_xfr, avg_w_f]):
    dict_parameters_coefficients['well_params']['xfr'] = round(avg_xfr, 1)
    dict_parameters_coefficients['well_params']['w_f'] = round(avg_w_f, 1)
    dict_parameters_coefficients['well_params']['length_FracStage'] = round(avg_length_FracStage, 0)

    data_wells = data_wells.merge(data_frac, how='left', on='well_number')
    data_wells[['FracCount', 'xfr', 'w_f', 'length_FracStage']] = data_wells[['FracCount', 'xfr',
                                                                              'w_f', 'length_FracStage']].fillna(0)
    return data_wells, dict_parameters_coefficients


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
def load_economy_data(economy_path, name_field):
    # Инициализируем необходимые переменные
    with pd.ExcelFile(economy_path) as xls:
        # коэффициенты Кв Кз Кд	Кдв	Ккан Кк	Кман Кабдт
        constants = pd.read_excel(xls, sheet_name="Налоги_константы")
        macroeconomics = pd.read_excel(xls, sheet_name="Макропараметры", header=3)  # Основная макра

        # месторождения с НДД, проверка наличия в списке
        reservoirs_NDD = pd.read_excel(xls, sheet_name="МР с НДД", header=None).iloc[:, 0].values.tolist()
        type_tax_calculation = name_field in reservoirs_NDD

        # OPEX
        df_opex = pd.read_excel(xls, sheet_name="Удельный OPEX", header=0)
        df_opex = df_opex[df_opex['Месторождение'] == name_field]
        if df_opex.shape[0] < 3:
            logger.error(f"В исходных данных  ФЭМ нет OPEX по месторождению {name_field}")
            return None
        else:
            del df_opex['Месторождение']

        # потери нефти
        df_oil_loss = pd.read_excel(xls, sheet_name="Нормативы потерь нефти", header=0)
        df_oil_loss = df_oil_loss[df_oil_loss['Месторождение'] == name_field]
        if df_oil_loss.empty:
            logger.error(f"В исходных данных ФЭМ нет потерь нефти по месторождению {name_field}")
            return None
        else:
            del df_oil_loss['Месторождение']
            df_oil_loss = df_oil_loss.set_index([pd.Index(['oil_loss'])])

        # КРС_ПРС
        df_workover_wellservice = pd.read_excel(xls, sheet_name="КРС_ПРС", header=0)
        df_workover_wellservice = df_workover_wellservice[df_workover_wellservice['Месторождение'] == name_field]
        if df_workover_wellservice.shape[0] < 5:
            logger.error(f"В исходных данных ФЭМ нет КРС_ПРС по месторождению {name_field}")
            return None
        else:
            del df_workover_wellservice['Месторождение']

    # Подготовка файлов
    name_first_column = macroeconomics.columns[0]
    macroeconomics = macroeconomics.iloc[:, ~macroeconomics.columns.str.match('Unnamed').fillna(False)]
    macroeconomics = macroeconomics[macroeconomics[name_first_column].isin(macroeconomics_rows_name.keys())]
    macroeconomics.replace(macroeconomics_rows_name, inplace=True)
    macroeconomics = macroeconomics.fillna(method='ffill', axis=1).reset_index(drop=True)
    macroeconomics = formatting_df_economy(macroeconomics)

    df_opex.replace(OPEX_rows_name, inplace=True)
    df_opex = formatting_df_economy(df_opex)

    oil_loss = df_oil_loss.T

    df_workover_wellservice.replace(workover_wellservice_rows_name, inplace=True)
    df_workover_wellservice = formatting_df_economy(df_workover_wellservice)

    FEM = FinancialEconomicModel(macroeconomics, constants,
                                 df_opex, oil_loss, df_workover_wellservice,
                                 type_tax_calculation)
    return FEM


"""___________Вспомогательные функции___________"""


def formatting_df_economy(df):
    df = df.T
    df.columns = df.iloc[0]
    df.drop(df.index[0], inplace=True)
    return df


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


def filter_pressure(p_well, p_reservoir, type_pressure):
    """Функция для фильтрации давлений - исключение месяцев с отрицательной депрессией"""
    # Словарь для выбора рассматриваемого давления
    dict_pressure = {'P_well': p_well,
                      'P_reservoir': p_reservoir}
    pressure = dict_pressure[type_pressure]
    # Условия для поиска подходящих строк
    # исключаем строки с отрицательной депрессией и нулевыми значениями рассматриваемого давления
    valid_rows = ((p_reservoir - p_well) > 0) & (pressure != 0)
    if valid_rows.any():
        return pressure[valid_rows].mean()
    return 0  # Возвращаем 0, если нет подходящих строк


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
