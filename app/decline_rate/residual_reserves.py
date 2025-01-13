import math
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm

from scipy.interpolate import NearestNDInterpolator
from shapely.geometry import Point
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

# from app.well_active_zones import create_gdf_coordinates

pd.options.mode.chained_assignment = None


def get_reserves_by_characteristic_of_desaturation(df, min_reserves, r_max, year_min, year_max):
    """
    Расчет ОИЗ через характеристики вытеснения (история + карта)
    :param df: DataFrame исходных данных: {"well_number", "date", "Qo", "Ql", "T1_x", "T1_y", "T3_x", "T3_y"}
    :param min_reserves: минимальное значение ОИЗ
    :param r_max: максимальный радиус удаления для скважин, рассчитываемых по карте
    :param year_min: минимальное время работы скважины (лет)
    :param year_max: максимальное время работы скважины (лет)

    :return: DataFrame: {"№ скв.","ОИЗ"}
    """
    min_reserves = min_reserves * 1000  # перевод в т
    set_well = set(df.well_number)
    df_reserves = pd.DataFrame()
    well_error = []

    for well in tqdm(set_well, desc='Расчет ОИЗ для скважин по характеристикам вытеснения'):
        df_well = df.loc[df.well_number == well].reset_index(drop=True)
        df_result = calculate_reserves_statistics(df_well, well, marker="all_period")[0]
        if df_result.empty:
            df_result = calculate_reserves_statistics(df_well, well, marker="last_three_dots")[0]
            if df_result.empty:
                well_error.append(well)
                continue

        #  Проверка ограничений
        new_residual_reserves = df_result['ОИЗ']
        if df_result['Время работы (прогноз), лет'].values[0] > year_max:
            new_residual_reserves = (df_result['Добыча за посл мес, т']
                                     + df_result['Добыча за предпосл мес, т']) * year_max * 6
        elif df_result['Время работы (прогноз), лет'].values[0] < year_min:
            new_residual_reserves = (df_result['Добыча за посл мес, т']
                                     + df_result['Добыча за предпосл мес, т']) * year_min * 6
        if df_result['ОИЗ'].values[0] < min_reserves:
            new_residual_reserves = min_reserves

        # Массив запасов для построения карт
        df_reserves = pd.concat([df_reserves, df_result], ignore_index=True)

        df_result['ОИЗ'] = new_residual_reserves
        df_result['Время работы (прогноз), лет'] = new_residual_reserves / (df_result['Добыча за посл мес, т'] * 12)

    """_________________Заключительный этап: расчет для скважин с ошибкой по карте______________________"""

    df_coordinates = df[['well_number', 'T3_x_geo', 'T3_y_geo']]
    df_coordinates = df_coordinates.drop_duplicates(subset=['well_number']).reset_index(drop=True)
    df_all = df_reserves.set_index('Скважина')
    df_coordinates = df_coordinates.set_index('well_number')
    df_field = pd.merge(df_coordinates[['T3_x_geo', 'T3_y_geo']], df_all[['НИЗ']], left_index=True, right_index=True)

    df_errors = pd.DataFrame({'Скважина': well_error, 'well_number': well_error})
    df_errors = df_errors.set_index('well_number')
    df_errors = pd.merge(df_coordinates[['T3_x_geo', 'T3_y_geo']], df_errors[['Скважина']], left_index=True, right_index=True)

    # списки для заполнения
    marker, list_residual_reserves, list_initial_reserves = [], [], []

    for well in tqdm(well_error, desc='Расчет ОИЗ для скважин с ошибкой по карте'):
        # Информация по скважине
        df_well = df.loc[df.well_number == well].reset_index(drop=True)
        x_well, y_well = df_errors['T3_x_geo'][well], df_errors['T3_y_geo'][well]
        cumulative_oil_production = df_well['Qo'].cumsum().values[-1]
        # Добыча нефти за предпоследний и последний месяц
        if df_well.shape[0] > 1:
            Q_next_to_last = float(df_well['Qo'].iloc[-2])
        else:
            Q_next_to_last = 0
        Q_last = df_well['Qo'].values[-1]
        distance = ((x_well - df_field['T3_x_geo']) ** 2 + (y_well - df_field['T3_y_geo']) ** 2) ** 0.5
        r_min = distance.min()
        if r_min > r_max:
            marker.append("!!!Ближайшая скважина на расстоянии " + str(r_min))
        else:
            marker.append("Скважина в пределах ограничений по расстоянию")

        initial_reserves = interpolate_reserves(x_well, y_well,
                                                df_field[['T3_x_geo']], df_field[['T3_y_geo']], df_field[['НИЗ']])
        error_residual_reserves = initial_reserves - cumulative_oil_production

        work_time = error_residual_reserves / (Q_last * 12)
        if work_time < year_min:
            error_residual_reserves = (Q_last + Q_next_to_last) * year_min * 6
        elif work_time > year_max:
            error_residual_reserves = (Q_last + Q_next_to_last) * year_max * 6

        if error_residual_reserves < min_reserves:
            error_residual_reserves = min_reserves
        list_residual_reserves.append(int(error_residual_reserves))
        list_initial_reserves.append(int(error_residual_reserves + cumulative_oil_production))

    df_errors['НИЗ'] = list_initial_reserves
    df_errors['ОИЗ'] = list_residual_reserves
    df_errors['Метка'] = marker

    df_all_reserves = pd.concat([df_errors[['Скважина', 'ОИЗ']], df_reserves[['Скважина', 'ОИЗ']]])
    df_all_reserves['ОИЗ'] = df_all_reserves['ОИЗ'] / 1000
    df_all_reserves.columns = ["well_number", "well_reserves"]
    return df_all_reserves.set_index('well_number')


def get_reserves_by_map(data_wells, map_rrr, min_reserves=2):
    """
    Расчет остаточных запасов под скважинами через карту ОИЗ и эффективный радиус скважин
    Parameters
    ----------

    data_wells - массив скважин с основными данными
    map_rrr - карта ОИЗ, т/Га
    min_reserves - минимальные запасы в тыс.т (по-умолчанию 2)

    Returns
    -------
    Series с ОИЗ
    """
    default_size = map_rrr.geo_transform[1]
    area_cell = default_size ** 2

    # Создание географической и пиксельной сеток
    gdf_mesh, mesh_pixel = prepare_mesh_map_rrr(map_rrr)

    gdf_data_wells = gpd.GeoDataFrame(data_wells, geometry="LINESTRING_geo")
    gdf_data_wells["polygon_r_eff_voronoy"] = gdf_data_wells.buffer(gdf_data_wells["r_eff_voronoy"])
    import warnings
    with warnings.catch_warnings(action='ignore', category=pd.errors.SettingWithCopyWarning):
        gdf_data_wells["reserves"] = None

    for index, row in gdf_data_wells.iterrows():
        points_index = list(gdf_mesh[gdf_data_wells.loc[index, "polygon_r_eff_voronoy"]
                            .contains(gdf_mesh["Mesh_Points"])].index)
        array_rrr = map_rrr.data[mesh_pixel.loc[points_index, 'y_coords'], mesh_pixel.loc[points_index, 'x_coords']]
        value_rrr = np.sum(array_rrr * area_cell / 10000) / 1000
        gdf_data_wells.loc[index, "reserves"] = value_rrr
        if value_rrr == 0 or value_rrr < min_reserves:
            gdf_data_wells.loc[index, "reserves"] = min_reserves
    return gdf_data_wells["reserves"]


def prepare_mesh_map_rrr(map_rrr):
    """Вспомогательная функция для формирования gdp массива гео-координат карты и df сетки пиксельных координат"""
    # Определение границ сетки карты в географических координатах
    x_min, x_max = [map_rrr.geo_transform[0],
                    map_rrr.geo_transform[0] + map_rrr.geo_transform[1] * map_rrr.data.shape[1]]
    y_min, y_max = [map_rrr.geo_transform[3] + map_rrr.geo_transform[5] * map_rrr.data.shape[0],
                    map_rrr.geo_transform[3]]

    # Создание географической и пиксельной сеток
    grid_x, grid_y = np.mgrid[x_min:x_max:map_rrr.geo_transform[1], y_max:y_min:-map_rrr.geo_transform[1]]
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    grid_x_pixel, grid_y_pixel = np.mgrid[0: map_rrr.data.shape[1]:1, 0: map_rrr.data.shape[0]:1]
    grid_points_pixel = np.column_stack((grid_x_pixel.ravel(), grid_y_pixel.ravel()))

    mesh = list(map(lambda x, y: Point(x, y), grid_points[:, 0], grid_points[:, 1]))
    mesh = pd.DataFrame(mesh, columns=['Mesh_Points'])
    gdf_mesh = gpd.GeoDataFrame(mesh, geometry="Mesh_Points")
    mesh_pixel = pd.DataFrame(grid_points_pixel, columns=['x_coords', 'y_coords'])
    return gdf_mesh, mesh_pixel


def calculate_reserves_statistics(df_well, name_well, marker="all_period"):
    """
    Расчет остаточных извлекаемых запасов для скважины на основе истории работы
    :param df_well: DataFrame[columns= {"well_number"; "date"; "Qo"; "Ql"; "time_work_prod"; "objects"; "T3_x"; "T3_y"}]
    :param name_well: название скважины
    :param marker: отметка "all_period" расчет по всем точкам истории, "last_three_dots" для последних трех
    :return: df_well_result: DataFrame[columns= {'Скважина'; "НИЗ"; "ОИЗ"; 'Метод';
                                                'Добыча за посл мес, т'; 'Добыча за предпосл мес, т';
                                                'Накопленная добыча,т'; 'Correlation'; 'Sigma';
                                                'Время работы, прогноз,лет'; 'Время работы, прошло, лет';
                                                'Координата Х'; 'Координата Y'}],
             error: ошибка, из-за которой не были рассчитаны ОИЗ
    """
    error = ""

    #  Подготовка осей
    df_well['Qo_сumsum'] = df_well['Qo'].cumsum()
    df_well['Ql_сumsum'] = df_well['Ql'].cumsum()
    df_well['Qw_сumsum'] = df_well['Ql_сumsum'] - df_well['Qo_сumsum']

    with np.errstate(divide='ignore'):
        df_well['log_Qo'] = np.where(df_well['Qo_сumsum'] > 0, np.log(df_well['Qo_сumsum']), 0)
        df_well['log_Ql'] = np.where(df_well['Ql_сumsum'] > 0, np.log(df_well['Ql_сumsum']), 0)
        df_well['log_Qw'] = np.where(df_well['Qw_сumsum'] > 0, np.log(df_well['Qw_сumsum']), 0)

    df_well['year'] = df_well.date.map(lambda x: x.year)

    if df_well.shape[0] > 2:
        if marker == 'all_period':
            # Добыча нефти за предпоследний месяц
            Q_next_to_last = float(df_well['Qo'].iloc[-2])
        elif marker == 'last_three_dots':
            df_well = df_well.tail(3)
            Q_last, Q_next_to_last = float(df_well['Qo'].iloc[-1]), float(df_well['Qo'].iloc[-2])
            if (Q_last / Q_next_to_last) < 0.25:
                df_well = df_well[:-1]
    else:
        error = "меньше 3 месяцев работы"
        Q_next_to_last = 0

    """Статистические методы"""
    list_methods = ["Nazarov_Sipachev", "Sipachev_Pasevich", "FNI", "Maksimov", "Sazonov"]
    # initial_reserves, residual_reserves, Correlation, Determination
    results = [[], [], [], []]
    for method in list_methods:
        result = linear_model(df_well, method)
        for element in range(4):
            results[element].append(result[element])

    cumulative_oil_production = df_well['Qo_сumsum'].values[-1]
    work_time = int(df_well.year.iloc[-1]) - int(df_well.year.iloc[0])

    #  Формирование итогового DataFrame
    df_well_result = pd.DataFrame(columns=('Скважина', "НИЗ", "ОИЗ", 'Метод',
                                           'Добыча за посл мес, т', 'Добыча за предпосл мес, т',
                                           'Накопленная добыча,т', 'Correlation', 'Sigma',
                                           'Время работы (прогноз), лет', 'Время работы (прошло), лет',
                                           'Координата Х', 'Координата Y'))
    for i in range(len(list_methods)):
        row = [name_well, results[0][i], results[1][i], list_methods[i], df_well['Qo'].values[-1],
               Q_next_to_last, cumulative_oil_production, results[2][i], results[3][i],
               results[1][i] / df_well['Qo'].values[-1], work_time,
               float(df_well.T3_x_geo.iloc[-1]), float(df_well.T3_y_geo.iloc[-1])]
        df_well_result.loc[len(df_well_result)] = row

    # Оценка текущего решения по фильтрам
    df_well_result = df_well_result.loc[df_well_result['ОИЗ'] > 0]
    if df_well_result.empty:
        error = "residual_reserves < 0"

    df_well_result = df_well_result.loc[~((df_well_result['Correlation'] < 0.7)
                                          & (df_well_result['Correlation'] > (-0.7)))]
    if df_well_result.empty:
        error = "Correlation <0.7 & >-0.7"

    df_well_result = df_well_result.loc[df_well_result['Время работы (прогноз), лет'] < 50]
    if df_well_result.empty:
        error = "work_time > 50"

    df_well_result = df_well_result.sort_values('ОИЗ')
    df_well_result = df_well_result.tail(1)

    if not df_well_result.empty:
        if marker == 'all_period':
            df_well_result['Метка'] = 'Расчет по всем точкам'
        elif marker == 'last_three_dots':
            df_well_result['Метка'] = 'Расчет по последним 3м точкам'

    return df_well_result, error


def linear_model(df, method):
    """
    Различные модели характеристик вытеснения для нахождения ОИЗ
    Parameters
    ----------
    df - DataFrame с подготовленными колонками
    method - метод построения характеристики вытеснения

    Returns
    -------
    [НИЗ, ОИЗ, коэффициент корреляции, коэффициент детерминации]
    """
    cumulative_oil_production = df['Qo_сumsum'].values[-1]

    if method == "Nazarov_Sipachev":  # Назаров_Сипачев
        x = df['Qw_сumsum'].values.reshape((-1, 1))
        y = df['Ql_сumsum'] / df['Qo_сumsum']
    elif method == "Sipachev_Pasevich":
        x = df['Ql_сumsum'].values.reshape((-1, 1))
        y = df['Ql_сumsum'] / df['Qo_сumsum']
    elif method == "FNI":
        x = df['Qo_сumsum'].values.reshape((-1, 1))
        y = df['Qw_сumsum'] / df['Qo_сumsum']
    elif method == "Maksimov":
        x = df['Qo_сumsum'].values.reshape((-1, 1))
        y = df['log_Ql']
    elif method == "Sazonov":
        x = df['Qo_сumsum'].values.reshape((-1, 1))
        y = df['log_Qw']

    model = find_linear_end(x, y)[2]
    try:
        a = model.intercept_[0]  # линейный коэф
    except IndexError:
        a = model.intercept_
    b = math.fabs(float(model.coef_))  # угловой коэф

    if b != 0:
        if method == "Nazarov_Sipachev":
            initial_reserves = (1 / b) * (1 - ((a - 1) * (1 - 0.99) / 0.99) ** 0.5)
        elif method == "Sipachev_Pasevich":
            initial_reserves = (1 / b) - ((0.01 * a) / (b ** 2)) ** 0.5
        elif method == "FNI":
            initial_reserves = 1 / (2 * b * (1 - 0.99)) - a / 2 * b
        elif method == "Maksimov":
            initial_reserves = (1 / b) * math.log(0.99 / ((1 - 0.99) * b * math.exp(a)))
        elif method == "Sazonov":
            try:
                initial_reserves = (1 / b) * math.log(0.99 / ((1 - 0.99) * b * math.exp(a)))
            except ZeroDivisionError:
                initial_reserves = 0
        residual_reserves = initial_reserves - cumulative_oil_production
    else:
        initial_reserves, residual_reserves = 0, 0
    Correlation = math.fabs(np.corrcoef(df['Qw_сumsum'], y)[1, 0])
    Determination = model.score(x, y)
    return initial_reserves, residual_reserves, Correlation, Determination


def find_linear_end(x, y):
    """
    Функция поиска линейного участка в нефтяной ВНФ
    :x : Значения по оси X
    :y : Значения по оси Y
    :return : Угловой коэффициент,
              Свободный коэффициент,
              модель
    """
    # Максимальная ошибка по МНК
    max_error = 0.001
    i = 1
    a = -1
    error = 1
    # Проходимся от конца графика и ищем такие 6 точек, идущих подряд, что:
    # 1) Ошибка на МНК по 6 точкам не больше максимальной ошибки
    # 2) Угловой коэффициент положительный
    while a < 0 or error > max_error:
        model = LinearRegression().fit(x[-(i + 6): -i].reshape(-1, 1),
                                       y[-(i + 6): -i].values.reshape(-1, 1))
        a = model.coef_[0][0]
        error = mean_squared_error(model.predict(x[-(6 + i):-i].reshape(-1, 1)),
                                   y[-(6 + i):-i].values.reshape(-1, 1))
        i += 1
    i -= 1
    # Для найденных 6 точек пытаемся увеличить число точек в интервале
    # с сохранением вышеперечисленных ограничений
    for interval_dots in range(6, 20):
        model = LinearRegression().fit(x[-(interval_dots + i): -i].reshape(-1, 1),
                                       y[-(interval_dots + i):-i].values.reshape(-1, 1))
        b = model.intercept_[0]
        a = model.coef_[0][0]
        error = mean_squared_error(model.predict(x[-(interval_dots + i):-i].reshape(-1, 1)),
                                   y[-(interval_dots + i):-i].values.reshape(-1, 1))
        if error > max_error or a < 0:
            model = LinearRegression().fit(x[-(interval_dots + i - 1): -i].reshape(-1, 1),
                                           y[-(interval_dots + i - 1):-i].values.reshape(-1, 1))
            b = model.intercept_[0]
            a = model.coef_[0][0]
            break
    return a, b, model


def interpolate_reserves(x, y, table_x, table_y, table_z):
    """
    Интерполяция значения запасов для скважины по карте значений
    Parameters
    ----------
    x - координата х скважины
    y - координата у скважины
    table_x - таблица координат х скважин
    table_y - таблица координат х скважин
    table_z - таблица запасов скважин соответствующих координатам

    Returns
    -------
    интерполированное значение
    """
    table_x = np.reshape(np.array(table_x), (-1,))
    table_y = np.reshape(np.array(table_y), (-1,))
    table_z = np.reshape(np.array(table_z), (-1,))

    nearest_interp = NearestNDInterpolator(list(zip(table_x, table_y)), table_z)
    z_new = nearest_interp(x, y)
    return z_new
