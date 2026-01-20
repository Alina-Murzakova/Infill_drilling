import os
import pandas as pd
import numpy as np

from loguru import logger

from app.config import columns_name, dict_work_marker, sample_data_wells
from app.input_output.functions_wells_data import identification_ZBS_MZS, get_avg_last_param, calculate_azimuth, \
    get_avg_first_param, calculate_cumsum, create_shapely_types, range_priority_wells, get_well_type


def load_wells_data(data_well_directory):
    """
    Функция загрузки истории скважин и определения исследуемого объекта (месторождение, пласт)
    Parameters
    ----------
    data_well_directory - путь к выгрузке

    Returns
    -------
    Фрейм с полной историей скважин
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
        error_msg = f"Выгрузка содержит не одно месторождение: {field}"
        logger.critical(error_msg)
        raise ValueError(f"{error_msg}")
    elif len(object_value) != 1:
        error_msg = f"Выгрузка содержит не один объект: {object_value}"
        logger.critical(error_msg)
        raise ValueError(f"{error_msg}")
    else:
        field = field[0]
        object_value = object_value[0]
    info = {'field': field, "object_value": object_value}
    return data_history, info


def prepare_wells_data(data_history, dict_properties, min_length_hor_well=150, first_months=6,
                       last_months=3, priority_radius=200, pho_water=1):
    """
    Функция, которая обрабатывает выгрузку МЭР (выгрузка по датам по всем скважинам//параметры задаются пользователем)
    Parameters
    ----------
    data_history - фрейм с полной историей скважин
    dict_properties - словарь свойств и параметров по умолчанию
    min_length_hor_well - максимальная длина ствола ННС
    first_months - количество первых месяцев работы для определения стартового дебита нефти
    last_months - количество последних месяцев работы для определения последнего дебита
    priority_radius - радиус для приоритизации скважин (для выделения скважин в одной маленькой зоне)

    Returns
    -------
    Фрейм с обработанной полной историей скважин
    Фрейм с параметрами добычи на последнюю дату работы для всех скважин
    """
    # Чистим ряд скважин (без характера работы, объекта или статуса)
    data_history.loc[((data_history['Ql_rate'] > 0) | (data_history['Winj_rate'] > 0)) &
                     (data_history['well_status'] == 0), 'well_status'] = 'РАБ.'
    data_history = data_history[(data_history.work_marker != 0) & ((data_history.objects != 0) |
                                                                   (data_history.well_status != 0))]
    logger.info(f"После фильтрации осталось {data_history.well_number.nunique()} скважин")

    # 3. Обработка координат // разделение на горизонтальные и вертикальные скважины
    data_history = get_well_type(data_history, min_length_hor_well)

    # 4. Определение ЗБС и МЗС, порядкового номера ствола и разделение добычи для МЗС
    data_history = identification_ZBS_MZS(data_history)
    logger.info(f"Количество МЗС - {data_history[data_history.type_wellbore == 'МЗС'].well_number_digit.nunique()}")
    logger.info(f"Количество ЗБС - {data_history[data_history.type_wellbore == 'ЗБС'].well_number.nunique()}")
    logger.info(f"МЗС - {data_history[data_history.type_wellbore == 'МЗС'].well_number.unique()}")

    # 5. Расчет объемной обводненности и плотности нефти для ТР
    data_history['water_cut_V'] = (((data_history['Ql_rate'] - data_history['Qo_rate']) / pho_water) * 100 /
                                   ((data_history['Ql_rate'] - data_history['Qo_rate']) / pho_water +
                                    data_history['Qo_rate'] /
                                    dict_properties['reservoir_fluid_properties']['rho'])).fillna(0)
    data_history['density_oil_TR'] = (data_history['Qo_rate_TR'] /
                                      (data_history['Ql_rate_TR'] * (1 - data_history['water_cut_TR'] / 100))).fillna(0)

    data_history_work = data_history.copy()
    data_history_work = data_history_work[(data_history_work.Ql_rate > 0) | (data_history_work.Winj_rate > 0)]

    # 6. Получение последних параметров работы скважин как среднее за last_months месяцев (добыча/закачка)
    data_wells_last_param = get_avg_last_param(data_history_work, data_history, last_months, dict_properties, pho_water)

    # 7. Добавление колонки с указанием как долго не работает скважина для скважин с добычей/закачкой
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

    # 8. Определяем дату первой добычи
    data_wells_prod = data_history_work[(data_history_work.Ql_rate > 0)]
    data_wells_first_production = data_wells_prod.groupby('well_number')['date'].min().reset_index()
    data_wells_first_production.rename(columns={'date': 'first_production_date'}, inplace=True)
    data_wells = data_wells.merge(data_wells_first_production, on='well_number', how='left')

    # 9. Определяем длительность последнего периода работы !!! Пока не используется
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

    # 10. Нахождение накопленной добычи нефти и закачки
    df_sort_date = (data_history.copy().sort_values(by=['well_number', 'date'], ascending=[True, True])
                    .reset_index(drop=True))
    data_wells = calculate_cumsum(data_wells, df_sort_date)

    # 11. Получение среднего стартового дебита за первые "first_months" месяцев
    data_wells = get_avg_first_param(data_wells, df_sort_date, first_months, dict_properties)

    # 12. Расчет азимута для горизонтальных скважин
    data_wells['azimuth'] = data_wells.apply(calculate_azimuth, axis=1)

    # 13. Расчет Shapely объектов
    df_shapely = create_shapely_types(data_wells, list_names=['T1_x_geo', 'T1_y_geo', 'T3_x_geo', 'T3_y_geo'])
    data_wells[['POINT_T1_geo', 'POINT_T3_geo', 'LINESTRING_geo']] = df_shapely

    # 14. Дополнительные преобразования
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
            logger.warning(f"В data_wells появился нераспознанный характер работы - {work_marker}. "
                           f"Скважины {list(well_numbers)} удалены.")

        # Удаляем строки с нераспознанными значениями
        data_wells = data_wells[~data_wells["work_marker"].isin(unknown_work_markers)]

    data_wells['continuous_work_months'] = data_wells['continuous_work_months'].astype(int)

    # 15. Приоритизация скважин в пределах радиуса
    data_wells = range_priority_wells(data_wells, priority_radius)
    logger.info(f"Сумма дебит нефти {data_history.Qo_rate.sum()}")
    logger.info(f"Сумма НДН {data_wells.Qo_cumsum.sum()}")
    return data_history, data_wells
