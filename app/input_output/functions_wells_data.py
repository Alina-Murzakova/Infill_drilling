import pandas as pd
import numpy as np
import re

from loguru import logger
from shapely.geometry import Point, LineString
from sklearn.cluster import DBSCAN


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

    # Определяем дату появления скважины (первая строка с ненулевым состоянием)
    data_history['first_well_date'] = (data_history.where(data_history['object'] != 0).groupby('well_number')['date']
                                       .transform('min'))

    # Копирование и разделение параметров в МЗС
    # Работа только с МЗС
    mask_mzs = data_history['type_wellbore'] == 'МЗС'
    if sum(mask_mzs):
        # Копирование ряда параметров в МЗС
        columns_to_copy = ['water_cut', 'water_cut_TR', 'time_work', 'time_work_prod', 'time_work_inj', 'P_well',
                           'P_reservoir']
        # Временно заменяем нули на NaN для корректного заполнения в столбцах columns_to_copy
        data_history.loc[mask_mzs, columns_to_copy] = data_history.loc[mask_mzs, columns_to_copy].replace(0, np.nan)
        grouped_mzs = data_history.loc[mask_mzs].groupby(['well_number_digit', 'date', 'objects'])
        data_history.loc[mask_mzs, columns_to_copy] = (
            grouped_mzs[columns_to_copy].transform(lambda x: x.ffill().bfill())
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


def get_avg_last_param(data_history_work, data_history, last_months, dict_properties, pho_water):
    """
    Функция для получения фрейма со средними последними параметрами работы скважин (добыча/закачка и обв)
    Parameters
    ----------
    data_history_work - история работы без учета остановок
    data_history - вся история работы
    last_months - количество последних месяцев для осреднения
    dict_properties - словарь свойств и параметров по умолчанию

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
                                                                    'Winj_rate_TR', lambda x: x[x != 0].mean()),
                                                                density_oil_TR=(
                                                                    'density_oil_TR', lambda x: x[x != 0].mean()),
                                                                water_cut_V=(
                                                                    'water_cut_V', lambda x: x[x != 0].mean())
                                                                )
                      .fillna(0).reset_index())
    # Массовая обводненность согласно МЭР
    data_last_rate['water_cut'] = (np.where(data_last_rate['Ql_rate'] > 0,
                                            (data_last_rate['Ql_rate'] - data_last_rate['Qo_rate']) * 100 /
                                            data_last_rate['Ql_rate'], 0))
    # Объемная обводненность согласно ТР
    data_last_rate['density_oil_TR'] = (np.where((data_last_rate['Ql_rate_TR'] != 0) &
                                                 (data_last_rate['density_oil_TR'] == 0),
                                                 dict_properties['fluid_params']['rho'],
                                                 data_last_rate['density_oil_TR']))
    data_last_rate['water_cut_TR'] = (np.where(data_last_rate['Ql_rate_TR'] > 0,
                                               (data_last_rate['Ql_rate_TR'] - data_last_rate['Qo_rate_TR'] /
                                                data_last_rate['density_oil_TR']) * 100 /
                                               data_last_rate['Ql_rate_TR'], 0))
    # Заменяем последние параметры на последние средние параметры
    data_wells_last_param = data_wells_last_param.merge(data_last_rate, on='well_number', suffixes=('', '_avg'))
    cols_to_replace = ['Qo_rate', 'Ql_rate', 'Qo_rate_TR', 'Ql_rate_TR', 'water_cut', 'water_cut_TR', 'Winj_rate',
                       'Winj_rate_TR', 'density_oil_TR', 'water_cut_V']
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


def get_avg_first_param(data_wells, df_sort_date, first_months, dict_properties):
    """
    Функция для получения фрейма со средними стартовыми параметрами работы скважин
    Parameters
    ----------
    data_wells - фрейм со всеми скважинами
    df_sort_date - отсортированная история работы
    first_months - количество первых месяцев для осреднения
    dict_properties - словарь свойств и параметров по умолчанию

    Returns
    -------
    Фрейм со средними стартовыми параметрами работы скважин
    """
    data_first_rate = df_sort_date.copy()
    data_first_rate['cum_rate_liq'] = data_first_rate['Ql_rate'].groupby(data_first_rate['well_number']).cumsum()
    data_first_rate = data_first_rate[data_first_rate['cum_rate_liq'] != 0]
    data_first_rate = data_first_rate.groupby('well_number').head(first_months)
    # # Определяем начальное Рпл, как максимальное за first_months
    data_first_rate['P_reservoir'] = (data_first_rate.groupby('well_number')['P_reservoir']
                                      .transform(clean_p_reservoir))

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
                                data_first_rate.loc[p_res.index, 'P_well'], p_res, type_pressure='P_reservoir')),
                            init_density_oil_TR=('density_oil_TR', lambda x: x[x != 0].mean()))
                       .reset_index())
    # !!! Оценка давлений через распределения для отсечения выбросов
    data_first_rate['init_drawdown'] = np.where((data_first_rate['init_P_reservoir_prod'] > 0) &
                                                (data_first_rate['init_P_well_prod'] > 0),
                                                data_first_rate['init_P_reservoir_prod'] -
                                                data_first_rate['init_P_well_prod'], 0)

    q1, q3 = quantile_filter(data_first_rate, name_column="init_drawdown")
    data_first_rate['init_drawdown'] = np.where((data_first_rate['init_drawdown'] < q1) &
                                                (data_first_rate['init_drawdown'] != 0), q1,
                                                data_first_rate['init_drawdown'])
    data_first_rate['init_drawdown'] = np.where((data_first_rate['init_drawdown'] > q3) &
                                                (data_first_rate['init_drawdown'] != 0), q3,
                                                data_first_rate['init_drawdown'])
    # data_first_rate['init_P_well_prod'] = data_first_rate['init_P_reservoir_prod'] - data_first_rate['init_drawdown']

    data_wells = data_wells.merge(data_first_rate, how='left', on='well_number')
    data_wells = data_wells.fillna(0)

    # Массовая средняя стартовая обводненность согласно МЭР
    data_wells['init_water_cut'] = np.where(data_wells['init_Ql_rate'] > 0,
                                            (data_wells['init_Ql_rate'] - data_wells['init_Qo_rate']) /
                                            data_wells['init_Ql_rate'], 0)
    # Объемная средняя обводненность согласно ТР
    data_wells['init_density_oil_TR'] = (np.where((data_wells['init_Ql_rate_TR'] != 0) &
                                                  (data_wells['init_density_oil_TR'] == 0),
                                                  dict_properties['fluid_params']['rho'],
                                                  data_wells['init_density_oil_TR']))
    data_wells['init_water_cut_TR'] = np.where(data_wells['init_Ql_rate_TR'] > 0,
                                               (data_wells['init_Ql_rate_TR'] - data_wells['init_Qo_rate_TR'] /
                                                data_wells['init_density_oil_TR']) /
                                               data_wells['init_Ql_rate_TR'], 0)
    return data_wells


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


def extract_well_number(well_name):
    """Функция для извлечения первой числовой части - номера скважины без доп части"""
    match = re.match(r"(\d+)", str(well_name))  # Приводим к строке на случай NaN
    if match:
        return int(match.group(1))
    else:
        logger.error(f"Не удалось извлечь численную часть номера скважины - {well_name}")
        return None


def clean_p_reservoir(series):
    s = series.copy()

    for i in range(1, len(s)):
        if (s.iloc[i] > s.iloc[i - 1]) and (s.iloc[i - 1] != 0):
            value = s.iloc[i - 1]
            for j in range(i - 1, -1, -1):
                if s.iloc[j] == value:
                    s.iloc[j] = s.iloc[i]
                else:
                    break
    return s


def get_first_months(data_first_rate):
    """Подсчет количества месяцев с ненулевыми значениями для каждого параметра,
    при этом учитывается, что скважина должна работать (Ql_rate != 0)"""
    params = ['Qo_rate_TR', 'Ql_rate_TR', 'P_well']

    def count_months(series, ql_rate_series):
        nonzero = 0
        for i, (val, ql_rate) in enumerate(zip(series, ql_rate_series)):
            if pd.notna(val) and val != 0 and pd.notna(ql_rate) and ql_rate != 0:
                nonzero += 1
            if nonzero >= 3:
                return i + 1  # +1 потому что индекс с нуля
        # если ненулевых месяцев <3 — берём 6 месяцев
        return min(len(series), 6)

    first_months = {param: count_months(data_first_rate[param], data_first_rate['Ql_rate']) for param in params}
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
        if (amount_cluster / amount_wells < ratio_clusters_wells) and (epsilon > step_priority_radius):
            epsilon -= step_priority_radius
        elif ((amount_cluster / amount_wells < ratio_clusters_wells) and
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


def quantile_filter(data_wells, name_column):
    """Функция определения верхнего и нижнего квантиля"""
    column = data_wells[data_wells[name_column] > 0][name_column]
    # Рассчитываем квартили
    q1 = np.percentile(column, 25)
    q3 = np.percentile(column, 75)
    return q1, q3
