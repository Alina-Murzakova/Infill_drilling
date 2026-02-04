import math

import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import LineString, Polygon, MultiPolygon
from longsgis import voronoiDiagram4plg

from loguru import logger
from tqdm import tqdm

from app.maps_handler.maps import trajectory_break_points


def calc_r_eff(cumulative_value, B, ro, eff_h, m, So, type_well, len_well, So_min):
    """
    calculate drainage radius of production well
    :param cumulative_value: cumulative oil production or cumulative_water_inj, tons|m3
    :param B: volumetric ratio of oil|water
    :param ro: density oli|water g/cm3
    :param eff_h: effective thickness, m
    :param m: reservoir porosity, units
    :param So: initial oil saturation, units
    :param type_well: "vertical" or "horizontal"
    :param len_well: length of well for vertical well
    :param So_min: minimum oil saturation, units
    :return:
    """
    if So - So_min < 0:
        So = So_min
    if type_well == "vertical":
        a = cumulative_value * B
        b = ro * math.pi * eff_h * m * (So - So_min)
        if b == 0:
            return 0
        else:
            return math.sqrt(a / b)
    elif type_well == "horizontal":
        L = len_well
        a = math.pi * cumulative_value * B
        b = eff_h * m * ro * (So - So_min)
        if b == 0:
            return 0
        else:
            return (-1 * L + math.sqrt(L * L + a / b)) / math.pi
    else:
        raise NameError(f"Wrong well type: {type_well}. Allowed values: vertical or horizontal")


def well_effective_radius(row, So_min, default_radius, default_radius_inj):
    """
    Расчет радиуса дренирования/нагнетания на основе параметров разработки
    Parameters
    ----------
    row - строка из data_wells
    So_min - Sor согласно заданным ОФП в constants
    default_radius - радиус по умолчанию (технически минимальный) для добывающего фонда
    default_radius_inj - радиус по умолчанию (технически минимальный) для нагнетательного фонда

    Returns R_eff
    -------

    """
    work_type_well = row.work_marker
    len_well = row["length_geo"]
    well_type = row["well_type"]
    eff_h = row["NNT"]
    m = row["m"]
    So = row["So"]
    Bo = row.Bo
    ro_oil = row.rho
    if work_type_well == "prod":
        cumulative_oil_prod = row.Qo_cumsum
        R_eff = calc_r_eff(cumulative_oil_prod, Bo, ro_oil, eff_h, m, So, well_type, len_well, So_min)
        if not R_eff or R_eff < default_radius:
            R_eff = default_radius
        return R_eff
    elif work_type_well == "inj":
        cumulative_water_inj = row.V_useful_injection
        R_eff = calc_r_eff(cumulative_water_inj, Bo, ro_oil, eff_h, m, So, well_type, len_well, So_min)
        if not R_eff or R_eff < default_radius_inj:
            R_eff = default_radius_inj
        return R_eff


def get_value_map(well_type, T1_x, T1_y, T3_x, T3_y, length_of_well, raster):
    """Получить среднее значение с карты по точкам в пикселях"""
    x_coord, y_coord = trajectory_break_points(well_type, T1_x, T1_y, T3_x, T3_y, length_of_well, default_size=1)
    value = np.mean(raster.get_values(x_coord, y_coord))
    return value


def get_parameters_voronoi_cells(df_Coordinates, type_coord="geo", default_size_pixel=1) -> pd.DataFrame:
    """
    Функция для расчета эффективного радиуса и площади по ячейкам вороного
    Parameters
    ----------
    gdf_Coordinates - фрейм данных для построения ячеек вороных
        необходимые столбцы ['well_number', "LINESTRING", "length_well"]
    type_coord - тип координат ячеек пиксельные/географические ['pix', 'geo']
    default_size_pixel - указать, если type_coord = 'pix', для пересчета буфера вокруг скважин
    Returns
    -------
    Фрейм с площадью ячейки и эффективным радиусом через данную площадь
    gdf_Coordinates['well_number', ..., 'area_voronoi', 'r_eff_voronoy']
    """
    if type_coord == 'geo':
        LINESTRING = 'LINESTRING_geo'
        length_well = 'length_geo'
    elif type_coord == 'pix':
        LINESTRING = 'LINESTRING_pix'
        length_well = 'length_pix'
    else:
        error_msg = "Неверный тип координат."
        logger.critical(error_msg)
        raise TypeError(f"{error_msg}")

    df_MZS = df_Coordinates[df_Coordinates.type_wellbore == "МЗС"].copy()
    df_Coordinates_other = df_Coordinates[df_Coordinates.type_wellbore != "МЗС"].copy()
    # Проверка на наличие МЗС
    if not df_MZS.empty:
        df_Coordinates_MZS = df_MZS.copy()
        df_Coordinates_MZS[LINESTRING] = df_Coordinates_MZS.groupby("well_number_digit")[LINESTRING].transform(
            combine_to_linestring)
        # Если есть МЗС, то формирование для них одной строки
        df_Coordinates_MZS.drop_duplicates(subset=['well_number_digit'], keep='first', inplace=True)
        df_Coordinates = pd.concat([df_Coordinates_other, df_Coordinates_MZS], ignore_index=True)

    gdf_Coordinates = gpd.GeoDataFrame(df_Coordinates, geometry=LINESTRING)
    # буферизация скважин || тк вороные строятся для полигонов буферизируем точки и линии скважин
    gdf_Coordinates["Polygon"] = gdf_Coordinates.set_geometry(LINESTRING).buffer(1, resolution=3)

    # Выпуклая оболочка - будет служить контуром для ячеек вороного || отступаем от границ фонда на 1000 м
    convex_hull = gdf_Coordinates.set_geometry("Polygon").union_all().convex_hull
    convex_hull = gpd.GeoDataFrame(geometry=[convex_hull]).buffer(1000 / default_size_pixel).boundary

    # Подготовим данные границы и полигонов скважины в нужном формате для алгоритма
    def rounded_geometry(geometry, precision=0):
        """ Округление координат точек в полигоне || на вход voronoiDiagram4plg надо подавать целые координаты """
        if isinstance(geometry, Polygon):
            rounded_exterior = [(round(x, precision), round(y, precision)) for x, y in geometry.exterior.coords]
            return Polygon(rounded_exterior)

    # Данные полигонов скважин polygon
    polygons_wells = gdf_Coordinates[["Polygon"]].copy()
    polygons_wells.columns = ["geometry"]
    polygons_wells["geometry"] = polygons_wells["geometry"].apply(rounded_geometry)

    # Граница в формате MultiPolygon
    boundary = MultiPolygon([rounded_geometry(Polygon(convex_hull[0]))])
    boundary = gpd.GeoDataFrame({'geometry': [boundary]})

    # Вороные
    boundary = boundary.set_geometry('geometry')
    polygons_wells = polygons_wells.set_geometry('geometry')
    vd = voronoiDiagram4plg(polygons_wells, boundary)

    # исходный индекс в функции voronoiDiagram4plg сбрасывается, поэтому восстановим привязку к скважинам
    position_polygons = []
    for _, voronoi_cell in vd.iterrows():
        indexes_contains = polygons_wells[voronoi_cell['geometry'].contains(polygons_wells['geometry'])].index
        position_polygons.append(indexes_contains[0])
        if len(indexes_contains) != 1:
            logger.warning(" Одной ячейке вороного соответствует несколько скважин! \n "
                           " Необходимо проверить траектории скважин и их пересечения. ")

    # Обработка пересекающихся скважин
    for i, polygon in enumerate(position_polygons):
        if type(polygon) is list:
            init_list = polygon.copy()
            for j, well in enumerate(polygon):
                first_part = [item for sublist in position_polygons[:i - 1] for item in
                              (sublist if type(sublist) is list else [sublist])]
                second_part = [item for sublist in position_polygons[i + 1:] for item in
                               (sublist if type(sublist) is list else [sublist])]
                other_wells = set(first_part + second_part)
                if (well in other_wells) and (len(init_list) > 1):
                    del init_list[init_list.index(well)]
            position_polygons[i] = init_list[0]

    vd['position_polygons'] = position_polygons

    # мерджим фреймы - каждой скважине своя ячейка вороного
    gdf_Coordinates['position_polygons'] = gdf_Coordinates.index
    gdf_Coordinates = gdf_Coordinates.merge(vd, on='position_polygons', how='left')
    del gdf_Coordinates['position_polygons']
    gdf_Coordinates = gdf_Coordinates.rename(columns={"geometry": "polygon_voronoi"})
    # расчет площади
    gdf_Coordinates['area_voronoi'] = gdf_Coordinates.set_geometry('polygon_voronoi').area

    # Если есть МЗС восстанавливаем исходный массив скважин
    if not df_MZS.empty:
        df_Coordinates_other = pd.merge(df_Coordinates_other, gdf_Coordinates[['well_number', 'area_voronoi']],
                                        on='well_number', how='left')
        df_Coordinates_MZS = pd.merge(df_MZS, gdf_Coordinates[gdf_Coordinates.type_wellbore
                                                              == "МЗС"][['well_number_digit', 'area_voronoi']],
                                      on='well_number_digit', how='left')
        df_Coordinates_MZS['area_voronoi'] = (df_Coordinates_MZS['area_voronoi'] /
                                              np.array(df_MZS.groupby('well_number_digit')['well_number'].transform(
                                                  'count')))
        df_Coordinates = pd.concat([df_Coordinates_other, df_Coordinates_MZS], ignore_index=True)
    else:
        df_Coordinates = gdf_Coordinates

    # считаем для ГС и ННС радиус через площадь ячейки вороного
    df_Coordinates.loc[df_Coordinates["well_type"] == "vertical", "r_eff_voronoy"] = np.sqrt(
        df_Coordinates['area_voronoi'] / np.pi)
    df_Coordinates.loc[df_Coordinates["well_type"] == "horizontal", "r_eff_voronoy"] = (
            (-df_Coordinates[length_well] + np.sqrt(np.power(df_Coordinates[length_well], 2)
                                                    + df_Coordinates['area_voronoi'] * np.pi)) // np.pi)
    return df_Coordinates[['well_number', 'area_voronoi', 'r_eff_voronoy']]


def voronoi_normalize_r_eff(data_wells, df_parameters_voronoi, buff=1.1):
    """ Нормирование больших эффективных радиусов на площадь ячейки Вороного
        buff - допустимая доля превышения площади ячейки"""

    data_wells_work = data_wells[(data_wells['Qo_cumsum'] > 0) | (data_wells['Winj_cumsum'] > 0)].reset_index(drop=True)
    # расчет вороных для скважин
    del data_wells_work["r_eff_voronoy"]
    data_wells_work = pd.merge(data_wells_work, df_parameters_voronoi, on='well_number', how='left')

    # расчет площадей для нормализации радиуса
    gdf_data_wells_work = gpd.GeoDataFrame(data_wells_work, geometry="LINESTRING_geo")
    gdf_data_wells_work["polygon_r_eff"] = gdf_data_wells_work.buffer(gdf_data_wells_work["r_eff_not_norm"])
    gdf_data_wells_work['area_r_eff'] = gdf_data_wells_work.set_geometry("polygon_r_eff").area

    # if площадь эффективного радиуса > площади ячейки вороного * buff then новый радиус ГС/ННС через площадь ячейки
    gdf_data_wells_work.loc[gdf_data_wells_work['area_r_eff'] > gdf_data_wells_work['area_voronoi']
                            * buff, "r_eff"] = gdf_data_wells_work['r_eff_voronoy']

    # мерджим фреймы gdf_data_wells_work и data_wells
    data_wells['r_eff'] = data_wells[['well_number']].merge(gdf_data_wells_work[['well_number', 'r_eff']],
                                                            on='well_number', how='left')['r_eff']
    data_wells['r_eff_voronoy'] = data_wells[['well_number']].merge(gdf_data_wells_work[['well_number',
                                                                                         'r_eff_voronoy']],
                                                                    on='well_number', how='left')['r_eff_voronoy']
    data_wells['r_eff'] = data_wells['r_eff'].fillna(data_wells['r_eff_not_norm'])
    data_wells['r_eff_voronoy'] = data_wells['r_eff_voronoy'].fillna(data_wells['r_eff_not_norm'])
    return data_wells


def combine_to_linestring(group):
    """Объединение координат МЗС в одну линию"""
    coords = []
    for geom in group:
        if geom.geom_type == 'Point':
            coords.append((geom.x, geom.y))  # добавляем координаты точки
        elif geom.geom_type == 'LineString':
            coords.extend(list(geom.coords))  # добавляем все координаты линии
    return LineString(coords) if coords else None


def calculate_effective_radius(data_wells, dict_properties, is_exe=False):
    """
    Дополнение фрейма data_wells колонкой 'r_eff'
    Parameters
    ----------
    data_wells - фрейм данных по скважинам
    dict_geo_phys_properties - словарь свойств на объект
    maps_handler - обязательные карты NNT|porosity|initial_oil_saturation

    Returns new_data_wells
    -------

    """
    # добавление колонок свойств для расчета из файла ГФХ
    data_wells['Bo'] = dict_properties['reservoir_fluid_properties']['Bo']  # объемный коэффициент нефти, д.ед
    data_wells['rho'] = dict_properties['reservoir_fluid_properties']['rho']  # плотность нефти в поверхностных условиях, г/см3

    # Расчет параметров ячеек вороного
    data_wells_work = data_wells[(data_wells['Qo_cumsum'] > 0) | (data_wells['Winj_cumsum'] > 0)].reset_index(drop=True)
    df_parameters_voronoi = get_parameters_voronoi_cells(data_wells_work)

    # Оценка объемов полезной закачки для нагнетательных скважин
    data_wells = calculate_useful_injection(data_wells, is_exe=is_exe)

    # расчет радиусов по физическим параметрам
    default_radius = dict_properties['well_params']['fact_wells_params']['default_radius_prod']
    default_radius_inj = dict_properties['well_params']['fact_wells_params']['default_radius_inj']
    if dict_properties['switches']['switch_adaptation_relative_permeability']:
        So_min = 0.3
    else:
        So_min = dict_properties['reservoir_fluid_properties']['Sor']
    data_wells['r_eff_not_norm'] = data_wells.apply(well_effective_radius,
                                                    args=(So_min, default_radius, default_radius_inj, ), axis=1)

    # нормировка эффективного радиуса фонда через площади ячеек Вороного
    data_wells = voronoi_normalize_r_eff(data_wells, df_parameters_voronoi)
    del data_wells['Bo']
    del data_wells['rho']
    del data_wells['V_useful_injection']
    return data_wells


def calculate_useful_injection(data_wells, max_distance_inj_prod=1000, is_exe=False):
    """Расчет объема полезной закачки для нагнетательных скважин через коэффициенты влияния
    max_distance_inj_prod - расстояние для поиска соседних скважин
    """
    # Разделяем скважины на нагнетательные и добывающие
    inj_wells = data_wells[data_wells.work_marker == 'inj'].copy()
    prod_wells = data_wells[data_wells.work_marker == 'prod'].copy()

    # Создаем GeoDataFrame для быстрого расчета расстояний
    gdf_inj = gpd.GeoDataFrame(inj_wells, geometry='LINESTRING_pix')
    gdf_prod = gpd.GeoDataFrame(prod_wells, geometry='LINESTRING_pix')

    # Подготовка данных для матричного расчета
    all_inj_wells = inj_wells.well_number.values
    all_prod_wells = prod_wells.well_number.values

    # Векторизованный расчет расстояний между всеми парами скважин
    distance_matrix = np.zeros((len(all_prod_wells), len(all_inj_wells)))

    for i, prod_well in tqdm(enumerate(all_prod_wells), desc='Расчет матрицы расстояний', disable=is_exe):
        prod_point = gdf_prod[gdf_prod.well_number == prod_well].geometry.iloc[0]
        distances = gdf_inj.geometry.distance(prod_point)
        distance_matrix[i, :] = distances

    # Применяем маску для расстояний
    valid_distances = distance_matrix <= max_distance_inj_prod
    distance_matrix[~valid_distances] = np.inf  # чтобы избежать деления на 0

    # Векторизованный расчет коэффициентов влияния
    NNT_values = inj_wells.NNT.values
    Winj_cumsum_values = inj_wells.Winj_cumsum.values

    # Расчет lambda_ij для всех пар
    with np.errstate(divide='ignore', invalid='ignore'):
        lambda_ij = (NNT_values * Winj_cumsum_values) / distance_matrix
        # Нормализация по строкам (для каждой добывающей скважины)
        row_sums = lambda_ij.sum(axis=1, keepdims=True)
        lambda_ij = np.divide(lambda_ij, row_sums, where=row_sums != 0)

    # Расчет полезной закачки (векторизованный)
    Qo_cumsum_values = prod_wells.set_index('well_number')['Qo_cumsum']
    useful_injection = np.zeros(len(all_inj_wells))

    for j in tqdm(range(len(all_inj_wells)), desc='Расчет полезной закачки', disable=is_exe):
        mask = lambda_ij[:, j] > 0
        useful_injection[j] = np.sum(lambda_ij[mask, j] * Qo_cumsum_values.iloc[mask].values)

    # Создаем серию с результатами
    result_series = pd.Series(useful_injection, index=all_inj_wells, name='V_useful_injection')

    # Объединяем с исходными данными
    data_wells = data_wells.merge(result_series, how='left', left_on='well_number', right_index=True)

    data_wells['V_useful_injection'] = data_wells['V_useful_injection'].fillna(0)
    return data_wells
