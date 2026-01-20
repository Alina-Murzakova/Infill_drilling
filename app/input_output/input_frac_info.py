import os
import pandas as pd
import numpy as np

from loguru import logger

from app.config import columns_name_frac


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
        error_msg = (f"Фрак-лист с ГРП на запуске для рассматриваемого объект {name_object} пуст!\n"
                         f"Проверьте данные по ГРП или используйте параметры ГРП/МГРП по умолчанию "
                         f"(switch_avg_frac_params = False)")
        logger.critical(error_msg)
        raise ValueError(f"{error_msg}")

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
        avg_xfr = dict_parameters_coefficients['well_params']["fracturing"]['xfr']
    if avg_w_f == 0 or pd.isna(avg_w_f):
        avg_w_f = dict_parameters_coefficients['well_params']["fracturing"]['w_f']

    data_frac['xfr'] = np.where((data_frac['xfr'] == 0) & (data_frac['FracCount'] > 0), round(avg_xfr, 1),
                                data_frac['xfr'])
    data_frac['w_f'] = np.where((data_frac['w_f'] == 0) & (data_frac['FracCount'] > 0), round(avg_w_f, 1),
                                data_frac['w_f'])

    data_frac = data_frac.drop(['total_Frac', 'well_type', 'length_geo'], axis=1)
    # Перезапись значений по умолчанию xfr и w_f и length_FracStage по объекту на средние по фактическому фонду
    dict_parameters_coefficients['well_params']["fracturing"]['xfr'] = round(avg_xfr, 1)
    dict_parameters_coefficients['well_params']["fracturing"]['w_f'] = round(avg_w_f, 1)
    if avg_length_FracStage != 0 and not pd.isna(avg_length_FracStage):
        dict_parameters_coefficients['well_params']["fracturing"]['length_FracStage'] = round(avg_length_FracStage, 0)
    data_wells.drop(columns=['xfr', 'w_f', 'FracCount', 'length_FracStage'], inplace=True)
    data_wells = data_wells.merge(data_frac, how='left', on='well_number')
    data_wells[['FracCount', 'xfr', 'w_f', 'length_FracStage']] = data_wells[['FracCount', 'xfr',
                                                                              'w_f', 'length_FracStage']].fillna(0)
    return data_wells, dict_parameters_coefficients

