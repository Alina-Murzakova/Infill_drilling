import math

import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import Polygon, MultiPolygon
from longsgis import voronoiDiagram4plg

from loguru import logger
from app.maps_handler.maps import trajectory_break_points


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


def well_effective_radius(row, default_radius=50, NUMBER_MONTHS=120):
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
        len_well = row["length_geo"]
        well_type = row["well_type"]
        eff_h = row["NNT"]
        m = row["m"]
        So = row["So"]
        R_eff = 0
        if work_type_well == "prod":
            cumulative_oil_prod = row.Qo_cumsum
            Bo = row.Bo
            ro_oil = row.rho
            R_eff = calc_r_eff(cumulative_oil_prod, Bo, ro_oil, eff_h, m, So, well_type, len_well)
        elif work_type_well == "inj":
            cumulative_water_inj = row.Winj_cumsum
            R_eff = calc_r_eff(cumulative_water_inj, 1, 1, eff_h, m, So, well_type, len_well)
        if not R_eff:
            R_eff = default_radius
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
        LINESTRING = None
        length_well = None
        logger.error("Неверный тип координат.")
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

    # считаем для ГС и ННС радиус через площадь ячейки вороного
    gdf_Coordinates.loc[gdf_Coordinates["well_type"] == "vertical", "r_eff_voronoy"] = np.sqrt(
        gdf_Coordinates['area_voronoi'] / np.pi)
    gdf_Coordinates.loc[gdf_Coordinates["well_type"] == "horizontal", "r_eff_voronoy"] = (
            (-gdf_Coordinates[length_well] + np.sqrt(np.power(gdf_Coordinates[length_well], 2)
                                                     + gdf_Coordinates['area_voronoi'] * np.pi)) // np.pi)
    return gdf_Coordinates[['area_voronoi', 'r_eff_voronoy']]


def voronoi_normalize_r_eff(data_wells, buff=1.1):
    """ Нормирование больших эффективных радиусов на площадь ячейки Вороного
        buff - допустимая доля превышения площади ячейки"""

    data_wells_work = data_wells[(data_wells['Qo_cumsum'] > 0) | (data_wells['Winj_cumsum'] > 0)].reset_index(drop=True)
    # расчет вороных для скважин
    data_wells_work[['area_voronoi', 'r_eff_voronoy']] = get_parameters_voronoi_cells(data_wells_work)

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


def calculate_effective_radius(data_wells, dict_properties):
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
    data_wells['Bo'] = dict_properties['Bo']  # объемный коэффициент нефти, д.ед
    data_wells['rho'] = dict_properties['rho']  # плотность нефти в поверхностных условиях, г/см3

    # расчет радиусов по физическим параметрам
    data_wells['r_eff_not_norm'] = data_wells.apply(well_effective_radius, axis=1)

    # нормировка эффективного радиуса фонда через площади ячеек Вороного
    data_wells = voronoi_normalize_r_eff(data_wells)
    del data_wells['Bo']
    del data_wells['rho']
    return data_wells
