import os
import pandas as pd
import numpy as np
import re

from loguru import logger
from shapely.geometry import Point, LineString, MultiLineString
from sklearn.cluster import DBSCAN

from app.config import (columns_name, gpch_column_name, dict_work_marker, sample_data_wells, columns_name_frac,
                        macroeconomics_rows_name, OPEX_rows_name, workover_wellservice_rows_name, apg_rows_name,
                        df_NDPI_NDD)
from app.economy.financial_model import FinancialEconomicModel
from app.economy.functions import calculation_Kg


@logger.catch
def load_wells_data(data_well_directory, min_length_hor_well=150, first_months=6, last_months=3, priority_radius=200):
    """
    Функция, которая обрабатывает выгрузку МЭР (выгрузка по датам по всем скважинам//параметры задаются пользователем)
    Parameters
    ----------
    data_well_directory - путь к выгрузке
    min_length_hor_well - максимальная длина ствола ННС
    first_months - количество первых месяцев работы для определения стартового дебита нефти
    last_months - количество последних месяцев работы для определения последнего дебита
    priority_radius - радиус для приоритизации скважин (для выделения скважин в одной маленькой зоне)

    Returns
    -------
    Фрейм с обработанной полной историей скважин
    Фрейм с параметрами добычи на последнюю дату работы для всех скважин
    Словарь с данными о расчете {field, object}
    """
    # 1. Загрузка файла
    # data_history = pd.read_excel(os.path.join(os.path.dirname(__file__), data_well_directory))
    data_history = pd.DataFrame()
    xls = pd.ExcelFile(os.path.join(os.path.dirname(__file__), data_well_directory))
    for sheet_name in xls.sheet_names:
        if "Данные" in sheet_name:
            df = xls.parse(sheet_name)
            data_history = pd.concat([data_history, df], ignore_index=True)

    # 2. Подготовка файла
    # Переименование колонок
    data_history = data_history[list(columns_name.keys())]
    data_history.columns = columns_name.values()
    data_history['well_number'] = data_history['well_number'].map(
        lambda x: str(int(float(x))) if isinstance(x, (int, float)) else str(x))

    # Удаление строк, где скважина еще не пробурена
    data_history = data_history[data_history.work_marker != "не пробурена"].fillna(0)
    data_history = data_history.sort_values(by=['well_number', 'date'], ascending=[True, False]).reset_index(drop=True)
    logger.info(f"В исходных данных {len(data_history.well_number.unique())} скважин")
    # Словарь с данными о расчете // месторождение и объект
    field = list(set(data_history.field.values))
    object_value = list(set(data_history.object.values))
    if len(field) != 1:
        logger.error(f"Выгрузка содержит не одно месторождение: {field}")
    elif len(object_value) != 1:
        logger.error(f"Выгрузка содержит не один объект: {object_value}")
    else:
        field = field[0]
        object_value = object_value[0]
    info = {'field': field, "object_value": object_value}

    # Чистим ряд скважин (без характера работы, объекта или статуса)
    data_history = data_history[(data_history.work_marker != 0) & ((data_history.objects != 0) |
                                                                   (data_history.well_status != 0))]
    logger.info(f"После фильтрации осталось {data_history.well_number.nunique()} скважин")

    # 3. Обработка координат // разделение на горизонтальные и вертикальные скважины
    data_history = get_well_type(data_history, min_length_hor_well)

    #  4. Определение ЗБС и МЗС, порядкового номера ствола и разделение добычи для МЗС
    data_history = identification_ZBS_MZS(data_history)
    logger.info(f"Количество МЗС - {data_history[data_history.type_wellbore == 'МЗС'].well_number_digit.nunique()}")
    logger.info(f"Количество ЗБС - {data_history[data_history.type_wellbore == 'ЗБС'].well_number.nunique()}")

    data_history_work = data_history.copy()
    data_history_work = data_history_work[(data_history_work.Ql_rate > 0) | (data_history_work.Winj_rate > 0)]

    # 5. Получение последних параметры работы скважин как среднее за last_months месяцев (добыча/закачка)
    data_wells_last_param = get_avg_last_param(data_history_work, data_history, last_months)

    # 6. Добавление колонки с указанием как долго не работает скважина для скважин с добычей/закачкой
    data_wells_last_param['no_work_time'] = round((data_history['date'].max() - data_wells_last_param.date).dt.days
                                                  / 29.3)
    # Все скважины на последнюю дату (даже если никогда не работали)
    data_wells_last_date = data_history.groupby('well_number').nth(0).reset_index(drop=True)
    # Скважины без добычи/закачки на последнюю дату
    df_diff = data_wells_last_date[~data_wells_last_date.well_number.isin(data_wells_last_param.well_number)]
    import warnings
    with warnings.catch_warnings(action='ignore', category=pd.errors.SettingWithCopyWarning):
        df_diff['no_work_time'] = round((data_history['date'].max() - data_history['date'].min()).days / 29.3)
    # Датафрейм с параметрами по всем скважинам
    data_wells = pd.concat([data_wells_last_param, df_diff], ignore_index=True)

    # 7. Определяем дату первой добычи
    data_wells_prod = data_history_work[(data_history_work.Ql_rate > 0)]
    data_wells_first_production = data_wells_prod.groupby('well_number')['date'].min().reset_index()
    data_wells_first_production.rename(columns={'date': 'first_production_date'}, inplace=True)
    data_wells = data_wells.merge(data_wells_first_production, on='well_number', how='left')

    # 8. Определяем длительность последнего периода работы !!! Пока не используется
    # Разница в месяцах между текущей и предыдущей датой
    data_history_work['month_diff'] = -data_history_work.groupby('well_number')['date'].diff().dt.days.fillna(0) / 31
    # Определяем периоды работы (без перерывов)
    data_history_work['work_period'] = (data_history_work.groupby('well_number')['month_diff']
                                        .transform(lambda x: (x > 1).cumsum()))
    # Продолжительность каждого непрерывного периода работы !!!
    data_history_work['continuous_work_months'] = (data_history_work.groupby(['well_number', 'work_period'])['date']
                                                   .transform('count'))
    data_continuous_work = (data_history_work[data_history_work['work_period'] == 0][['well_number',
                                                                                      'continuous_work_months']]
                            .drop_duplicates())
    data_wells = data_wells.merge(data_continuous_work, on='well_number', how='left')

    # 9. Нахождение накопленной добычи нефти и закачки
    df_sort_date = (data_history.copy().sort_values(by=['well_number', 'date'], ascending=[True, True])
                    .reset_index(drop=True))
    data_wells = calculate_cumsum(data_wells, df_sort_date)

    # 10. Получение среднего стартового дебита за первые "first_months" месяцев
    data_wells = get_avg_first_param(data_wells, df_sort_date, first_months)

    # 11. Расчет азимута для горизонтальных скважин
    data_wells['azimuth'] = data_wells.apply(calculate_azimuth, axis=1)

    # 12. Расчет Shapely объектов
    df_shapely = create_shapely_types(data_wells, list_names=['T1_x_geo', 'T1_y_geo', 'T3_x_geo', 'T3_y_geo'])
    data_wells[['POINT_T1_geo', 'POINT_T3_geo', 'LINESTRING_geo', 'MULTILINESTRING_geo']] = df_shapely

    # 13. Дополнительные преобразования
    data_wells.drop(columns=['field', "object", "objects"], inplace=True)
    # дополнение data_wells всеми необходимыми колонками
    data_wells = pd.concat([sample_data_wells, data_wells])
    # оставляем для расчета только скважины с ненулевой накопленной добычей и закачкой по объекту
    data_wells = (data_wells[(data_wells['Qo_cumsum'] > 0) | (data_wells['Winj_cumsum'] > 0)].reset_index(drop=True))

    # проверка и замена характера работы (нам нужны только НЕФ и НАГ)
    data_wells["work_marker"] = np.where(data_wells.Ql_rate > 0, "НЕФ", np.where(data_wells.Winj_rate > 0, "НАГ",
                                                                                 data_wells["work_marker"]))
    data_wells.work_marker = data_wells.work_marker.replace(dict_work_marker)
    # Определяем неизвестные значения work_marker
    unknown_work_markers = (data_wells[~data_wells["work_marker"].isin(dict_work_marker.values())]["work_marker"]
                            .unique())
    if len(unknown_work_markers) > 0:
        for work_marker in unknown_work_markers:
            well_numbers = data_wells.loc[data_wells["work_marker"] == work_marker, "well_number"].unique()
            logger.error(f"В data_wells появился нераспознанный характер работы - {work_marker}. "
                         f"Скважины {list(well_numbers)} удалены.")

        # Удаляем строки с нераспознанными значениями
        data_wells = data_wells[~data_wells["work_marker"].isin(unknown_work_markers)]

    data_wells['continuous_work_months'] = data_wells['continuous_work_months'].astype(int)

    # 14. Приоритизация скважин в пределах радиуса
    data_wells = range_priority_wells(data_wells, priority_radius)
    logger.info(f"Сумма дебит нефти {data_history.Qo_rate.sum()}")
    logger.info(f"Сумма НДН {data_wells.Qo_cumsum.sum()}")
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
    data_frac = pd.read_excel(os.path.join(os.path.dirname(__file__), path_frac), header=1)
    # Переименование колонок
    data_frac = data_frac[list(columns_name_frac.keys())]
    data_frac.columns = columns_name_frac.values()
    # Подготовка файла
    data_frac['well_number'] = data_frac['well_number'].ffill()  # протягивание номера скважины в объединенных ячейках
    data_frac = data_frac.fillna(0)
    data_frac['well_number'] = data_frac['well_number'].map(
        lambda x: str(int(float(x))) if isinstance(x, (int, float)) else str(x))
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

    if data_frac.empty:
        error_message = (f"Фрак-лист с ГРП на запуске для рассматриваемого объект {name_object} пуст!\n"
                         f"Проверьте данные по ГРП или используйте параметры ГРП/МГРП по умолчанию "
                         f"(switch_avg_frac_params = False)")
        logger.critical(error_message)
        raise ValueError(error_message)

    data_frac['comment'] = data_frac['comment'].astype(str)
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
    if avg_xfr == 0 or pd.isna(avg_xfr):
        avg_xfr = dict_parameters_coefficients['well_params']['xfr']
    if avg_w_f == 0 or pd.isna(avg_w_f):
        avg_w_f = dict_parameters_coefficients['well_params']['w_f']

    data_frac['xfr'] = np.where((data_frac['xfr'] == 0) & (data_frac['FracCount'] > 0), round(avg_xfr, 1),
                                data_frac['xfr'])
    data_frac['w_f'] = np.where((data_frac['w_f'] == 0) & (data_frac['FracCount'] > 0), round(avg_w_f, 1),
                                data_frac['w_f'])

    data_frac = data_frac.drop(['total_Frac', 'well_type', 'length_geo'], axis=1)
    # Перезапись значений по умолчанию xfr и w_f и length_FracStage по объекту на средние по фактическому фонду
    dict_parameters_coefficients['well_params']['xfr'] = round(avg_xfr, 1)
    dict_parameters_coefficients['well_params']['w_f'] = round(avg_w_f, 1)
    if avg_length_FracStage != 0 and not pd.isna(avg_length_FracStage):
        dict_parameters_coefficients['well_params']['length_FracStage'] = round(avg_length_FracStage, 0)
    data_wells.drop(columns=['xfr', 'w_f', 'FracCount', 'length_FracStage'], inplace=True)
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
                           'oil_density_at_surf', 'gas_oil_ratio']
        # для 'init_pressure' есть проверка при построении карты рисков, если оно 0,
        # то используется максимальное значение с карты
        for prop in list_properties:
            value = dict_geo_phys_properties_field[prop]
            value_mean = df_geo_phys_properties_field_mean[prop].iloc[0]
            if value <= 0:
                if value_mean > 0:
                    dict_geo_phys_properties_field[prop] = value_mean
                else:
                    logger.error(f"Свойство {prop} задано некорректно: {value}")
        return formatting_dict_geo_phys_properties(dict_geo_phys_properties_field)


@logger.catch
def load_economy_data(economy_path, name_field, gor):
    """ gor - газосодержание для расчета выручки с пнг, м3/т"""
    # Инициализируем необходимые переменные
    with pd.ExcelFile(economy_path) as xls:
        # коэффициенты Кв Кз Кд	Кдв	Ккан Кк	Кман Кабдт
        constants = pd.read_excel(xls, sheet_name="Налоги")
        macroeconomics = pd.read_excel(xls, sheet_name="Макропараметры", header=3)  # Основная макра

        # месторождения с НДД
        reservoirs_NDD = pd.read_excel(xls, sheet_name="МР с НДД")

        # OPEX
        df_opex = pd.read_excel(xls, sheet_name="Удельный OPEX", header=0)
        df_opex = df_opex[df_opex['Месторождение'] == name_field]
        if df_opex.shape[0] < 3:
            logger.error(f"В исходных данных  ФЭМ нет OPEX по месторождению {name_field}")
            return None
        else:
            del df_opex['Месторождение']

        # CAPEX
        df_capex = pd.read_excel(xls, sheet_name="CAPEX", header=0)
        # Определим цену одной стадии ГРП по массе пропанта
        df_cost_GRP = pd.read_excel(xls, sheet_name="ГРП_цена", header=0)
        # Интерполируем цену
        cost_stage_GRP = np.interp(df_capex.iloc[6, 1], df_cost_GRP['Тонн'], df_cost_GRP['Цена за операцию ГРП, тыс '
                                                                                         'руб. без НДС'])
        df_capex.iloc[6, 1] = cost_stage_GRP
        df_capex.iloc[6, 0] = 'Цена за 1 стадию ГРП, тыс руб'

        # Уд_ОНВСС_бурение
        df_ONVSS_cost_ed = pd.read_excel(xls, sheet_name="Уд_ОНВСС_бурение", header=0)
        df_ONVSS_cost_ed = df_ONVSS_cost_ed[df_ONVSS_cost_ed['Месторождение'] == name_field]
        if df_ONVSS_cost_ed.empty:
            logger.error(f"В исходных данных ФЭМ нет Уд_ОНВСС_бурение по месторождению {name_field}")
            return None
        else:
            ONVSS_cost_ed = df_ONVSS_cost_ed.iloc[0, 1]

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

        # Определим цену ПНГ в зависимости от КС
        df_APG_CS = pd.read_excel(xls, sheet_name="ПНГ_КС", header=0)
        df_APG_CS = df_APG_CS[df_APG_CS['Месторождение'] == name_field]
        if df_APG_CS.shape[0] < 1:
            logger.error(f"В исходных данных ФЭМ нет привязки месторождения {name_field} к КС")
            return None
        else:
            price_APG = df_APG_CS['Цена ПНГ (макра)'].iloc[0]

        # ПНГ
        df_apg = pd.read_excel(xls, sheet_name="ПНГ", header=0)
        df_apg = df_apg[df_apg['Месторождение'] == name_field]
        if df_apg.shape[0] < 3:
            logger.error(f"В исходных данных ФЭМ нет данных ПНГ по месторождению {name_field}")
            return None
        else:
            del df_apg['Месторождение']

        # Схема расчета налогов
    method = "ДНС"
    dict_NDD = {'initial_recoverable_reserves': None,
                'cumulative_production': None,
                'Kg_group': None}

    name_row_NDPI_NDD = None
    if name_field in reservoirs_NDD['Месторождение'].values.tolist():
        method = "НДД"
        initial_recoverable_reserves = constants.iloc[5, 1]
        cumulative_production = constants.iloc[6, 1]
        Kg_group = reservoirs_NDD[reservoirs_NDD['Месторождение'] == name_field]['Кг_номер группы'].iloc[0]

        dict_NDD = {'initial_recoverable_reserves': initial_recoverable_reserves,
                    'cumulative_production': cumulative_production,
                    'Kg_group': Kg_group}

        Kg = calculation_Kg(Kg_group, pd.Series(cumulative_production / initial_recoverable_reserves)).values[0]
        row_NDPI_NDD = df_NDPI_NDD[2]
        name_row_NDPI_NDD = row_NDPI_NDD[row_NDPI_NDD.index <= Kg].iloc[-1]

    # Подготовка файлов
    name_first_column = macroeconomics.columns[0]
    macroeconomics = macroeconomics.iloc[:, ~macroeconomics.columns.str.match('Unnamed').fillna(False)]
    # Определим цену ПНГ в зависимости от КС
    macroeconomics_rows_name[price_APG] = 'price_APG'
    macroeconomics_rows_name[name_row_NDPI_NDD] = 'NDPI_NDD'
    macroeconomics = macroeconomics[macroeconomics[name_first_column].isin(macroeconomics_rows_name.keys())]
    macroeconomics.replace(macroeconomics_rows_name, inplace=True)
    macroeconomics = macroeconomics.fillna(method='ffill', axis=1).reset_index(drop=True)
    macroeconomics = formatting_df_economy(macroeconomics)

    df_opex.replace(OPEX_rows_name, inplace=True)
    df_opex = formatting_df_economy(df_opex)

    oil_loss = df_oil_loss.T

    df_workover_wellservice.replace(workover_wellservice_rows_name, inplace=True)
    df_workover_wellservice = formatting_df_economy(df_workover_wellservice)

    df_apg.replace(apg_rows_name, inplace=True)
    df_apg = formatting_df_economy(df_apg)

    if gor < 0:
        gor = 300  # при отсутствии значения газосодержания в ГФХ
    FEM = FinancialEconomicModel(macroeconomics, constants,
                                 df_opex, oil_loss, df_capex, ONVSS_cost_ed,
                                 df_workover_wellservice, df_apg, gor)
    return FEM, method, dict_NDD


"""___________Вспомогательные функции___________"""


def get_well_type(data_history, min_length_hor_well):
    """Обработка координат // разделение на горизонтальные и вертикальные скважины"""
    data_history.loc[data_history["T3_x_geo"] == 0, 'T3_x_geo'] = data_history.T1_x_geo
    data_history.loc[data_history["T3_y_geo"] == 0, 'T3_y_geo'] = data_history.T1_y_geo
    data_history["length_geo"] = np.sqrt(np.power(data_history.T3_x_geo - data_history.T1_x_geo, 2)
                                         + np.power(data_history.T3_y_geo - data_history.T1_y_geo, 2))
    data_history["well_type"] = ""
    data_history.loc[data_history["length_geo"] < min_length_hor_well, "well_type"] = "vertical"
    data_history.loc[data_history["length_geo"] >= min_length_hor_well, "well_type"] = "horizontal"
    data_history.loc[data_history["well_type"] == "vertical", 'T3_x_geo'] = data_history.T1_x_geo
    data_history.loc[data_history["well_type"] == "vertical", 'T3_y_geo'] = data_history.T1_y_geo
    return data_history


def identification_ZBS_MZS(data_history):
    """
    Идентификация ЗБС и МЗС, разделение добычи для МЗС и определение порядкового номера ствола
    """
    # Выделение числовой части из номера скважины
    data_history['well_number_digit'] = data_history["well_number"].apply(extract_well_number).astype(int)
    # Количество работающих стволов в каждый месяц на одном объекте
    amount_work_well_every_month = (data_history.groupby(['well_number_digit', 'date', 'objects',
                                                          'work_marker', 'well_status'])['well_number']
                                    .transform('count'))  # nunique
    # Если в один месяц работает больше 1 скважины на одном объекте - это МЗС
    data_history['is_mzs'] = amount_work_well_every_month > 1
    # Если в каждый месяц работает только 1 скважина, но их несколько - это ЗБС
    data_history['is_zbs'] = (amount_work_well_every_month == 1) & (data_history.groupby('well_number_digit')
                                                                    ['well_number'].transform('nunique') > 1)
    # Если хотя бы в одной строчке МЗС, то везде МЗС
    data_history['is_mzs'] = data_history.groupby('well_number')['is_mzs'].transform('any')
    data_history['is_zbs'] = np.where(data_history['is_mzs'], False, data_history['is_zbs'])

    data_history['type_wellbore'] = 'Материнский ствол'
    data_history.loc[data_history['is_mzs'], 'type_wellbore'] = 'МЗС'
    data_history.loc[data_history['is_zbs'], 'type_wellbore'] = 'ЗБС'
    data_history = data_history.drop(['is_mzs', 'is_zbs'], axis=1)

    # Копирование и разделение параметров в МЗС
    # Работа только с МЗС
    mask_mzs = data_history['type_wellbore'] == 'МЗС'
    # Копирование ряда параметров в МЗС
    columns_to_copy = ['water_cut', 'water_cut_TR', 'time_work', 'time_work_prod', 'time_work_inj', 'P_well',
                       'P_reservoir']
    # Временно заменяем нули на NaN для корректного заполнения в столбцах columns_to_copy
    data_history.loc[mask_mzs, columns_to_copy] = data_history.loc[mask_mzs, columns_to_copy].replace(0, np.nan)
    grouped_mzs = data_history.loc[mask_mzs].groupby(['well_number_digit', 'date', 'objects'])
    data_history.loc[mask_mzs, columns_to_copy] = (grouped_mzs[columns_to_copy].transform(lambda x: x.ffill().bfill())
                                                   .fillna(0))

    # Разделение добычи/закачки МЗС на количество стволов (пока поровну)
    columns_to_split = ['Qo_rate', 'Qo_rate_TR', 'Ql_rate', 'Ql_rate_TR', 'Qo', 'Ql',
                        'Winj_rate', 'Winj_rate_TR', 'Winj']
    summed_values = grouped_mzs[columns_to_split].transform('sum')
    count_wellbore_mzs = grouped_mzs['well_number'].transform('count')

    for col in columns_to_split:
        data_history.loc[mask_mzs, f'split_{col}'] = summed_values[col] / count_wellbore_mzs
        data_history[col] = np.where(mask_mzs, data_history[f'split_{col}'], data_history[col])
        del data_history[f'split_{col}']

    # Определяем дату появления скважины (первая строка с ненулевым состоянием)
    data_history['first_well_date'] = (data_history.where(data_history['object'] != 0).groupby('well_number')['date']
                                       .transform('min'))
    # На случай, если у МЗС разные даты запуска
    data_history.loc[mask_mzs, 'first_well_date'] = (
        data_history.loc[mask_mzs]
        .groupby(['well_number_digit', 'date', 'type_wellbore'])['first_well_date']
        .transform("min"))

    # Нумеруем стволы в порядке хронологии
    data_history['number_wellbore'] = data_history.groupby('well_number_digit')['first_well_date'].transform(
        lambda x: x.rank(method='dense') - 1).astype(int)

    data_history['type_wellbore'] = np.where((data_history['number_wellbore'] == 0) &
                                             (data_history['type_wellbore'] != "МЗС"), 'Материнский ствол',
                                             data_history['type_wellbore'])
    return data_history


def get_avg_last_param(data_history_work, data_history, last_months):
    """
    Функция для получения фрейма со средними последними параметрами работы скважин (добыча/закачка и обв)
    Parameters
    ----------
    data_history_work - история работы без учета остановок
    data_history - вся история работы
    last_months - количество последних месяцев для осреднения

    Returns
    -------
    Фрейм со средними последними параметрами работы скважин
    """
    # Скважины с добычей/закачкой и параметры работы в последний рабочий месяц
    data_wells_last_param = data_history_work.groupby('well_number').nth(0).reset_index(drop=True)

    # Нахождение среднего последнего дебита за последние "last_months" месяцев
    data_last_rate = data_history.copy()  # сортировка от новых к старым
    data_last_rate['reverse_cum_rate_liq'] = data_last_rate['Ql_rate'].groupby(data_last_rate['well_number']).cumsum()
    data_last_rate['reverse_cum_rate_inj'] = data_last_rate['Winj_rate'].groupby(data_last_rate['well_number']).cumsum()
    # Удаление периода остановки, если скважина не работает
    data_last_rate = data_last_rate[(data_last_rate['reverse_cum_rate_liq'] != 0) |
                                    (data_last_rate['reverse_cum_rate_inj'] != 0)]
    data_last_rate = data_last_rate.groupby('well_number').head(last_months)

    data_last_rate = (data_last_rate.groupby('well_number').agg(Qo_rate=('Qo_rate', lambda x: x[x != 0].mean()),
                                                                Ql_rate=('Ql_rate', lambda x: x[x != 0].mean()),
                                                                Qo_rate_TR=('Qo_rate_TR', lambda x: x[x != 0].mean()),
                                                                Ql_rate_TR=('Ql_rate_TR', lambda x: x[x != 0].mean()),
                                                                Winj_rate=('Winj_rate', lambda x: x[x != 0].mean()),
                                                                Winj_rate_TR=(
                                                                    'Winj_rate_TR', lambda x: x[x != 0].mean()))
                      .fillna(0).reset_index())
    data_last_rate['water_cut'] = (np.where(data_last_rate['Ql_rate'] > 0,
                                            (data_last_rate['Ql_rate'] - data_last_rate['Qo_rate']) * 100 /
                                            data_last_rate['Ql_rate'], 0))

    data_last_rate['water_cut_TR'] = (np.where(data_last_rate['Ql_rate_TR'] > 0,
                                               (data_last_rate['Ql_rate_TR'] - data_last_rate['Qo_rate_TR']) * 100 /
                                               data_last_rate['Ql_rate_TR'], 0))
    # Заменяем последние параметры на последние средние параметры
    data_wells_last_param = data_wells_last_param.merge(data_last_rate, on='well_number', suffixes=('', '_avg'))
    cols_to_replace = ['Qo_rate', 'Ql_rate', 'Qo_rate_TR', 'Ql_rate_TR', 'water_cut', 'water_cut_TR', 'Winj_rate',
                       'Winj_rate_TR']
    for col in cols_to_replace:
        data_wells_last_param[col] = data_wells_last_param[f"{col}_avg"].fillna(0)
    data_wells_last_param.drop(columns=[f"{col}_avg" for col in cols_to_replace], inplace=True)
    return data_wells_last_param


def calculate_cumsum(data_wells, df_sort_date):
    """Расчет накопленной добычи нефти и закачки"""
    for column in ['Qo', 'Winj']:
        data_cumsum = df_sort_date[['well_number', column]].copy()
        data_cumsum[f'{column}_cumsum'] = data_cumsum[column].groupby(data_cumsum['well_number']).cumsum()
        data_cumsum = data_cumsum.groupby('well_number').tail(1)
        data_wells = data_wells.merge(data_cumsum[['well_number', f'{column}_cumsum']], how='left', on='well_number')
    return data_wells


def get_avg_first_param(data_wells, df_sort_date, first_months):
    """
    Функция для получения фрейма со средними стартовыми параметрами работы скважин
    Parameters
    ----------
    data_wells - фрейм со всеми скважинами
    df_sort_date - отсортированная история работы
    first_months - количество первых месяцев для осреднения

    Returns
    -------
    Фрейм со средними стартовыми параметрами работы скважин
    """
    data_first_rate = df_sort_date.copy()
    data_first_rate['cum_rate_liq'] = data_first_rate['Ql_rate'].groupby(data_first_rate['well_number']).cumsum()
    data_first_rate = data_first_rate[data_first_rate['cum_rate_liq'] != 0]
    data_first_rate = data_first_rate.groupby('well_number').head(first_months)

    # Определяем first_months для каждой скважины
    first_months_dict = data_first_rate.groupby('well_number').apply(get_first_months).to_dict()

    # Применяем своё first_months для каждой скважины
    data_first_rate = data_first_rate.groupby('well_number', group_keys=False).apply(
        lambda g: g.head(first_months_dict[g.name]))

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

    # Расчет обводненности
    data_wells['init_water_cut'] = np.where(data_wells['init_Ql_rate'] > 0,
                                            (data_wells['init_Ql_rate'] - data_wells['init_Qo_rate']) /
                                            data_wells['init_Ql_rate'], 0)

    data_wells['init_water_cut_TR'] = np.where(data_wells['init_Ql_rate_TR'] > 0,
                                               (data_wells['init_Ql_rate_TR'] - data_wells['init_Qo_rate_TR']) /
                                               data_wells['init_Ql_rate_TR'], 0)
    return data_wells


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
    P_init - начальное пластовое давление | МПа --> атм

    - fluid_params:
    mu_w - вязкость воды | сП или мПа*с
    mu_o - вязкость нефти | сП или мПа*с
    c_o - сжимаемость нефти | (1/МПа)×10-4 --> 1/атм
    c_w - сжимаемость воды | (1/МПа)×10-4 --> 1/атм
    Bo - объемный коэффициент расширения нефти | м3/м3
    Pb - давление насыщения | МПа --> атм
    rho - плотность нефти | г/см3
    gor - Газосодержание| м3/т
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
                             'rho': dict_geo_phys_properties['oil_density_at_surf'],
                             'gor': dict_geo_phys_properties['gas_oil_ratio']}}


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
        # Добавление MULTILINESTRING для МЗС
        df_result[['well_number_digit', 'type_wellbore']] = data_wells[['well_number_digit', 'type_wellbore']]
        mask_mzs = df_result["type_wellbore"] == "МЗС"
        df_result.loc[mask_mzs, "MULTILINESTRING"] = (df_result.loc[mask_mzs]
                                                      .groupby("well_number_digit")["LINESTRING"]
                                                      .transform(lambda x: MultiLineString(x.tolist())))
        df_result["MULTILINESTRING"] = np.where(df_result["type_wellbore"] == "МЗС", df_result["MULTILINESTRING"],
                                                df_result["LINESTRING"])
    df_result = df_result.drop(['well_number_digit', 'type_wellbore'], axis=1)
    return df_result


def extract_well_number(well_name):
    """Функция для извлечения первой числовой части - номера скважины без доп части"""
    match = re.match(r"(\d+)", str(well_name))  # Приводим к строке на случай NaN
    if match:
        return int(match.group(1))
    else:
        logger.error(f"Не удалось извлечь численную часть номера скважины - {well_name}")
        return None


def get_first_months(data_first_rate):
    """Подсчет количества месяцев с ненулевыми значениями для каждого параметра"""
    nonzero_TR_counts = {
        'Qo_rate_TR': (data_first_rate['Qo_rate_TR'].notna() & (data_first_rate['Qo_rate_TR'] != 0)).sum(),
        'Ql_rate_TR': (data_first_rate['Ql_rate_TR'].notna() & (data_first_rate['Ql_rate_TR'] != 0)).sum(),
        'P_well': (data_first_rate['P_well'].notna() & (data_first_rate['P_well'] != 0)).sum(),
        'P_reservoir': (data_first_rate['P_reservoir'].notna() & (data_first_rate['P_reservoir'] != 0)).sum()
    }
    # Для каждого параметра, если >=3 месяцев ненулевых, берём 3, иначе 6
    first_months = {param: 3 if count >= 3 else 6 for param, count in nonzero_TR_counts.items()}
    # Берем максимум из всех параметров (чтобы не терять важные данные)
    return max(first_months.values())


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


def range_priority_wells(data_wells, epsilon, step_priority_radius=20, ratio_clusters_wells=0.9):
    """
    Функция выделения скважин в пределах epsilon и их приоритизация
    Parameters
    ----------
    data_wells
    epsilon - радиус для поиска скважин в одной зоне, м (priority_radius)
    step_priority_radius
    ratio_clusters_wells

    Returns
    -------
    data_wells
    """
    df = data_wells[['well_number', 'POINT_T1_geo', 'no_work_time']].copy()
    # Преобразуем координаты в массив
    coords = np.array([[point.x, point.y] for point in df["POINT_T1_geo"]])

    while True:
        # Кластеризация DBSCAN, min_samples=1 - одна скважина может быть кластером
        clustering = DBSCAN(eps=epsilon, min_samples=1).fit(coords)
        df["cluster_id"] = clustering.labels_
        df['amount_wells_in_zone'] = df.groupby("cluster_id")['well_number'].transform('count')
        amount_wells = df['well_number'].nunique()
        amount_cluster = df['cluster_id'].nunique()
        if (amount_cluster/amount_wells < ratio_clusters_wells) and (epsilon > step_priority_radius):
            epsilon -= step_priority_radius
        elif ((amount_cluster/amount_wells < ratio_clusters_wells) and
              (epsilon < step_priority_radius) and (epsilon > 0)):
            epsilon = 1
        else:
            break

    def select_priority_wells(gdf):
        priority_wells = []
        for cluster_id, group in gdf.groupby("cluster_id"):
            # Действующие скважины и остановленные <= 3 месяцев назад всегда учитываются
            active_wells = group[group["no_work_time"] <= 3]['well_number'].tolist()
            if len(active_wells) == 0:
                # Иначе берем с минимальным no_work_time
                min_no_work_time = group["no_work_time"].min()
                active_wells = group[group["no_work_time"] == min_no_work_time]['well_number'].tolist()
            priority_wells += active_wells
        return priority_wells

    # Выбираем приоритетные скважины
    first_priority_wells = df[df['amount_wells_in_zone'] == 1]['well_number'].tolist()
    priority_wells = select_priority_wells(df[(df['amount_wells_in_zone'] > 1)])
    first_priority_wells.extend(priority_wells)

    data_wells["priority"] = data_wells["well_number"].isin(first_priority_wells).astype(int)
    # 1 - первый приоритет
    # 0 - второй приоритет
    return data_wells
