import math
import numpy as np
import pandas as pd

import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from longsgis import voronoiDiagram4plg

from loguru import logger
from map import trajectory_break_points


def calc_r_eff(cumulative_value, B, ro, eff_h, m, So, type_well, len_well, So_min=0.3):
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
             math.sqrt(a / b)
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


def well_effective_radius(row, default_radius=50, NUMBER_MONTHS = 120):
    """
    Расчет радиуса дренирования/нагнетания на основе параметров разработки
    Parameters
    ----------
    row - строка из data_wells
    default_radius - радиус по умолчанию (технически минимальный)
    NUMBER_MONTHS - Количество месяцев для отнесения скважин к действующим

    Returns R_eff
    -------

    """
    # Проверка на длительность работы
    if row['no_work_time'] > NUMBER_MONTHS:
        return default_radius
    else:
        work_type_well = row.work_marker
        len_well = row["length of well T1-3"]
        well_type = row["well type"]
        eff_h = row["eff_h"]
        m = row["m"]
        So = row["So"]
        R_eff = 0
        if work_type_well == "prod":
            cumulative_oil_prod = row.Qo_cumsum
            Bo = row.Bo
            ro_oil = row.ro_oil
            R_eff = calc_r_eff(cumulative_oil_prod, Bo, ro_oil, eff_h, m, So, well_type, len_well)
        elif work_type_well == "inj":
            cumulative_water_inj = row.Winj_cumsum
            R_eff = calc_r_eff(cumulative_water_inj, 1, 1, eff_h, m, So, well_type, len_well)
        if not R_eff:
            R_eff = default_radius
        return R_eff


def get_value_map(row, raster):
    x_coord, y_coord = trajectory_break_points(row, default_size=raster.geo_transform[1])
    value = np.mean(raster.get_values(x_coord, y_coord))
    return value


def voronoi_normalize_r_eff(data_wells, buff=1.1):
    """ Нормирование больших эффективных радиусов на площадь ячейки Вороного
        buff - допустимая доля превышения площади ячейки"""

    data_wells_work = data_wells[(data_wells['Qo_cumsum'] > 0) | (data_wells['Winj_cumsum'] > 0)].reset_index(drop=True)

    # add shapely types for well coordinates || подготовим фрейм с координатами для работы
    df_Coordinates = data_wells_work[["well_number", 'T1_x', 'T1_y', 'T3_x', 'T3_y',
                                      "r_eff", "length of well T1-3", "well type"]].copy()
    import warnings
    with warnings.catch_warnings(action='ignore', category=pd.errors.SettingWithCopyWarning):
        df_Coordinates["POINT T1"] = list(map(lambda x, y: Point(x, y), df_Coordinates.T1_x, df_Coordinates.T1_y))
        df_Coordinates["POINT T3"] = list(map(lambda x, y: Point(x, y), df_Coordinates.T3_x, df_Coordinates.T3_y))
        df_Coordinates["LINESTRING"] = list(
            map(lambda x, y: LineString([x, y]), df_Coordinates["POINT T1"], df_Coordinates["POINT T3"]))

    gdf_Coordinates = gpd.GeoDataFrame(df_Coordinates, geometry="LINESTRING")

    # буферизация скважин || тк вороные строятся для полигонов буферизируем точки и линии скважин
    gdf_Coordinates["Polygon"] = gdf_Coordinates.set_geometry("LINESTRING").buffer(1, resolution=3)

    # Выпуклая оболочка - будет служить контуром для ячеек вороного || отступаем от границ фонда на 1000 м
    convex_hull = gdf_Coordinates.set_geometry("Polygon").union_all().convex_hull
    convex_hull = gpd.GeoDataFrame(geometry=[convex_hull]).buffer(1000).boundary

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
        if len(indexes_contains) != 1:
            logger.warning(" Одной ячейке вороного соответствует несколько скважин! \n "
                           " Необходимо проверить траектории скважин и их пересечения. ")
        else:
            position_polygons.append(indexes_contains[0])
    vd['position_polygons'] = position_polygons

    # мерджим фреймы - каждой скважине своя ячейка вороного
    gdf_Coordinates['position_polygons'] = gdf_Coordinates.index
    gdf_Coordinates = gdf_Coordinates.merge(vd, on='position_polygons', how='left')
    del gdf_Coordinates['position_polygons']
    gdf_Coordinates = gdf_Coordinates.rename(columns={"geometry": "polygon_voronoi"})

    # расчет площадей для нормализации радиуса
    gdf_Coordinates["polygon_r_eff"] = gdf_Coordinates.set_geometry("LINESTRING").buffer(gdf_Coordinates["r_eff"])
    gdf_Coordinates['area_r_eff'] = gdf_Coordinates.set_geometry('polygon_r_eff').area
    gdf_Coordinates['area_voronoi'] = gdf_Coordinates.set_geometry('polygon_voronoi').area

    # новый радиус
    gdf_Coordinates['new_r_eff'] = gdf_Coordinates['r_eff']

    # if площадь эффективного радиуса > площади ячейки вороного * buff then новый радиус ГС/ННС через площадь ячейки
    gdf_Coordinates.loc[(gdf_Coordinates['area_r_eff'] > gdf_Coordinates['area_voronoi'] * buff) & (
                gdf_Coordinates["well type"] == "vertical"), "new_r_eff"] = np.sqrt(
        gdf_Coordinates['area_voronoi'] / np.pi)
    gdf_Coordinates.loc[(gdf_Coordinates['area_r_eff'] > gdf_Coordinates['area_voronoi'] * buff) & (
                gdf_Coordinates["well type"] == "horizontal"), "new_r_eff"] = (-gdf_Coordinates[
        "length of well T1-3"] + np.sqrt(
        np.power(gdf_Coordinates["length of well T1-3"], 2) + gdf_Coordinates['area_voronoi'] * np.pi)) // np.pi

    # мерджим фреймы gdf_Coordinates и data_wells
    data_wells = data_wells.merge(gdf_Coordinates[['well_number', 'new_r_eff']], on='well_number', how='left')
    data_wells['new_r_eff'] = data_wells['new_r_eff'].fillna(data_wells['r_eff'])
    return data_wells['new_r_eff']


def calculate_effective_radius(data_wells, dict_geo_phys_properties, maps):
    """
    Дополнение фрейма data_wells колонкой 'r_eff'
    Parameters
    ----------
    data_wells - фрейм данных по скважинам
    dict_geo_phys_properties - словарь свойств на объект
    maps - обязательные карты NNT|porosity|initial_oil_saturation

    Returns new_data_wells
    -------

    """
    type_map_list = list(map(lambda raster: raster.type_map, maps))

    # загрузка общих ГФХ для расчета активных зон скважин
    data_wells['Bo'] = dict_geo_phys_properties['Bo']
    data_wells['ro_oil'] = dict_geo_phys_properties['oil_density_at_surf']

    # инициализация всех необходимых карт
    map_NNT = maps[type_map_list.index("NNT")]
    map_porosity = maps[type_map_list.index("porosity")]
    map_initial_oil_saturation = maps[type_map_list.index("initial_oil_saturation")]

    # с карт снимаем значения eff_h, m, So
    data_wells['eff_h'] = data_wells.apply(get_value_map, raster=map_NNT, axis=1)
    data_wells['m'] = data_wells.apply(get_value_map, raster=map_porosity, axis=1)
    data_wells['So'] = data_wells.apply(get_value_map, raster=map_initial_oil_saturation, axis=1)

    # расчет радиусов по физическим параметрам
    data_wells['r_eff'] = data_wells.apply(well_effective_radius, axis=1)

    # нормировка эффективного радиуса фонда через площади ячеек Вороного
    data_wells['r_eff'] = voronoi_normalize_r_eff(data_wells)

    return data_wells