import os
import pickle
import alphashape
import win32api
from decimal import Decimal
from datetime import datetime
from loguru import logger
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely import Point, Polygon, MultiPolygon, MultiPoint
from typing import Dict, Any, List

from app.maps_handler.maps import read_array
from app.well_active_zones import combine_to_linestring
from longsgis import voronoiDiagram4plg


def summary_table(list_zones, switch_economy):
    """Подготовка краткой сводки по расчету"""

    def round_if_numeric(value, decimal=2):
        # Проверяем все основные числовые типы
        if isinstance(value, (int, float, np.integer, np.floating, Decimal)):
            return round(float(value), decimal)
        return value

    df_summary_table = pd.DataFrame(
        {'Зона': [int(drill_zone.rating) if isinstance(drill_zone.rating, float)
                  else drill_zone.rating for drill_zone in list_zones],
         'Количество\nскважин': [drill_zone.num_project_wells for drill_zone in list_zones],
         'Средний индекс\nуспешности бурения': [round_if_numeric(np.mean(drill_zone.opportunity_index_values)) for
                                                drill_zone in list_zones],
         'Запасы, тыс т': [round_if_numeric(drill_zone.reserves) for drill_zone in list_zones],
         'Средний запускной\nдебит нефти, т/сут': [round_if_numeric(drill_zone.init_avr_Qo_rate) for drill_zone in
                                                   list_zones],
         'Средний запускной\nдебит жидкости, м3/сут':
             [round_if_numeric(drill_zone.init_avr_Ql_rate, 2) for drill_zone in list_zones],
         'Средняя\nобводненность, %':
             [round_if_numeric(drill_zone.init_avr_water_cut, 2) for drill_zone in list_zones],
         'Накопленная добыча\nнефти (25 лет), тыс.т':
             [round(drill_zone.Qo / 1000, 2) if isinstance(drill_zone.Qo, float)
              else drill_zone.Qo for drill_zone in list_zones],
         'Накопленная добыча\nжидкости (25 лет), тыс.т':
             [round(drill_zone.Ql / 1000, 2) if isinstance(drill_zone.Ql, float)
              else drill_zone.Ql for drill_zone in list_zones],
         })

    if switch_economy:
        df_summary_table_economy = pd.DataFrame(
            {'Зона': [int(drill_zone.rating) if isinstance(drill_zone.rating, float)
                  else drill_zone.rating for drill_zone in list_zones],
             'Средний PI зоны': [round(drill_zone.PI, 2) if isinstance(drill_zone.PI, float)
              else drill_zone.PI for drill_zone in list_zones],
             'Суммарный NPV за\nрент. период, тыс.руб.': [round(np.sum(drill_zone.NPV), 2)
                                                          if isinstance(np.sum(drill_zone.NPV), float)
              else drill_zone.NPV for drill_zone in list_zones],
             'Кол-во скважин\nс ГЭП>1': [sum(np.count_nonzero(well.year_economic_limit > 0)
                                             for well in drill_zone.list_project_wells) for drill_zone in list_zones],
             })
        df_summary_table = df_summary_table.merge(df_summary_table_economy, left_on='Зона', right_on='Зона')

    df_summary_table = df_summary_table[df_summary_table['Зона'] != -1]
    if switch_economy:
        df_summary_table.loc['Всего'] = [
            'Всего',
            df_summary_table['Количество\nскважин'].sum(),
            round(df_summary_table['Средний индекс\nуспешности бурения'].mean(), 2),
            round(df_summary_table['Запасы, тыс т'].sum(), 2),
            round(df_summary_table['Средний запускной\nдебит нефти, т/сут'].mean(), 2),
            round(df_summary_table['Средний запускной\nдебит жидкости, м3/сут'].mean(), 2),
            round(df_summary_table['Средняя\nобводненность, %'].mean(), 2),
            round(df_summary_table['Накопленная добыча\nнефти (25 лет), тыс.т'].sum(), 2),
            round(df_summary_table['Накопленная добыча\nжидкости (25 лет), тыс.т'].sum(), 2),
            round(df_summary_table['Средний PI зоны'].mean(), 2),
            round(df_summary_table['Суммарный NPV за\nрент. период, тыс.руб.'].sum(), 2),
            round(df_summary_table['Кол-во скважин\nс ГЭП>1'].sum(), 2)]
    else:
        df_summary_table.loc['Всего'] = [
            'Всего',
            df_summary_table['Количество\nскважин'].sum(),
            round(df_summary_table['Средний индекс\nуспешности бурения'].mean(), 2),
            round(df_summary_table['Запасы, тыс т'].sum(), 2),
            round(df_summary_table['Средний запускной\nдебит нефти, т/сут'].mean(), 2),
            round(df_summary_table['Средний запускной\nдебит жидкости, м3/сут'].mean(), 2),
            round(df_summary_table['Средняя\nобводненность, %'].mean(), 2),
            round(df_summary_table['Накопленная добыча\nнефти (25 лет), тыс.т'].sum(), 2),
            round(df_summary_table['Накопленная добыча\nжидкости (25 лет), тыс.т'].sum(), 2)]
    df_summary_table = df_summary_table.fillna('')
    return df_summary_table


def save_contours(list_zones, map_conv, save_directory_contours, type_calc='buffer', buffer_size=60, alpha=0.01):
    """
    Сохранение контуров зон в формате .txt для загрузки в NGT в отдельную папку
    Parameters
    ----------
    list_zones - список объектов DrillZone
    map_conv - карта для конвертирования пиксельных координат зон в географические
    save_directory_contours -  путь для сохранения файлов в отдельную папку
    type_calc - формат расчета (buffer - буфферезация точек,
                                alpha - через библиотеку alphashape,
                                convex_hull - выпуклая оболочка зоны)
    buffer_size - размер буффера точек
    alpha - параметр для объединения точек alphashape
    """
    for drill_zone in list_zones:
        if drill_zone.rating != -1:
            x_coordinates, y_coordinates = drill_zone.x_coordinates, drill_zone.y_coordinates
            x_coordinates, y_coordinates = map_conv.convert_coord_to_geo((x_coordinates, y_coordinates))
            if type_calc == 'buffer':
                # Создаем список точек
                points = MultiPoint(list(zip(x_coordinates, y_coordinates)))
                # Строим буфер вокруг точек
                buffered = points.buffer(buffer_size).simplify(0.01)
                # Проверяем, что результат — полигон
                if isinstance(buffered, Polygon):
                    x_boundary, y_boundary = buffered.exterior.xy
                else:

                    error_msg = "Не удалось построить границу зоны. Проверьте размер buffer или входные данные."
                    logger.critical(error_msg)
                    raise ValueError(f"{error_msg}")
            elif type_calc == 'alpha':
                # Создаем список точек
                points = np.array(list(zip(x_coordinates, y_coordinates)))
                # Строим alpha shape
                alpha_shape = alphashape.alphashape(points, alpha)
                # Проверяем, что результат — полигон
                if isinstance(alpha_shape, Polygon):
                    x_boundary, y_boundary = alpha_shape.exterior.xy
                elif isinstance(alpha_shape, MultiPolygon):
                    # Выбираем самый большой полигон
                    largest_polygon = max(alpha_shape.geoms, key=lambda p: p.area)
                    x_boundary, y_boundary = largest_polygon.exterior.xy

                    # Выводим площади всех полигонов, чтобы не потерять случайно большой полигон
                    for poly in alpha_shape.geoms:
                        logger.info(f"Площадь полигона Мультиполигона {drill_zone.rating}: {poly.area / 1000000} кв.км")
                else:
                    error_msg = "Не удалось построить границу зоны. Проверьте параметр alpha или входные данные."
                    logger.critical(error_msg)
                    raise ValueError(f"{error_msg}")
            elif type_calc == 'convex_hull':
                mesh = list(map(lambda x, y: Point(x, y), x_coordinates, y_coordinates))
                ob = Polygon(mesh)
                # определяем границу зоны
                boundary_drill_zone = ob.convex_hull
                x_boundary, y_boundary = boundary_drill_zone.exterior.coords.xy
            else:
                error_msg = f"Проверьте значение параметра type_calc: {type_calc}"
                logger.critical(error_msg)
                raise ValueError(f"{error_msg}")
            name_txt = f'{save_directory_contours}/{drill_zone.rating}.txt'
            with open(name_txt, "w") as file:
                file.write(f"/\n")
                for x, y in zip(x_boundary, y_boundary):
                    file.write(f"{x} {y}\n")
                file.write(f"{x_boundary[0]} {y_boundary[0]}\n")
    pass


def get_save_path(program_name: str = "default") -> str:
    """
    Получение пути на запись
    :return:
    """
    path_program = os.getcwd()
    current_datetime = datetime.now().strftime("%d.%m.%Y")
    # Проверка возможности записи в директорию программы
    if os.access(path_program, os.W_OK):
        if "\\app" in path_program:
            path_program = path_program.replace("\\app", "")
        if "\\drill_zones" in path_program:
            path_program = path_program.replace("\\drill_zones", "")
        save_path = f"{path_program}\\output\\{current_datetime}"
    else:
        # Поиск другого диска с возможностью записи: D: если он есть и C:, если он один
        # В будущем можно исправить с запросом на сохранение
        drives = win32api.GetLogicalDriveStrings()  # получение списка дисков
        save_drive = []
        list_drives = [drive for drive in drives.split('\\\000')[:-1] if 'D:' in drive]
        if len(list_drives) >= 1:
            save_drive = list_drives[0]
        else:
            list_drives = [drive for drive in drives.split('\\\000')[:-1] if 'C:' in drive]
            if len(list_drives) >= 1:
                save_drive = list_drives[0]
            else:
                error_msg = f"У пользователя нет прав доступа на запись на диск {save_drive}"
                logger.critical(error_msg)
                raise PermissionError(f"{error_msg}")

        current_user = os.getlogin()
        profile_dir = [dir_ for dir_ in os.listdir(save_drive) if dir_.lower() == "profiles"
                       or dir_.upper() == "PROFILES"]

        if len(profile_dir) < 1:
            save_path = f"{save_drive}\\{program_name}_output\\{current_datetime}"
        else:
            save_path = (f"{save_drive}\\{profile_dir[0]}\\{current_user}\\"
                         f"{program_name}_output\\{current_datetime}")

    create_new_dir(save_path)
    return save_path


def create_new_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path)
    pass


def save_ranking_drilling_to_excel(name_field, name_object, list_zones, filename, switch_economy):
    gdf_result_ranking_drilling = gpd.GeoDataFrame()
    dict_project_wells_Qo, dict_project_wells_Ql = {}, {}
    dict_project_wells_Qo_rate, dict_project_wells_Ql_rate = {}, {}
    (dict_project_wells_cumulative_cash_flow,
     dict_project_wells_CAPEX, dict_project_wells_OPEX, dict_project_wells_NPV) = {}, {}, {}, {}
    for drill_zone in list_zones:
        if drill_zone.rating != -1:
            # gdf_project_wells = gpd.GeoDataFrame([well.__dict__ for well in drill_zone.list_project_wells])
            gdf_project_wells_ranking_drilling = gpd.GeoDataFrame(
                {'Месторождение': [name_field] * len(drill_zone.list_project_wells),
                 'Объект': [name_object] * len(drill_zone.list_project_wells),
                 '№ скважины': [well.well_number for well in drill_zone.list_project_wells],
                 'Координата_T1_x': [round(well.POINT_T1_geo.x, 0) for well in drill_zone.list_project_wells],
                 'Координата_T1_y': [round(well.POINT_T1_geo.y, 0) for well in drill_zone.list_project_wells],
                 'Координата_T3_x': [round(well.POINT_T3_geo.x, 0) for well in drill_zone.list_project_wells],
                 'Координата_T3_y': [round(well.POINT_T3_geo.y, 0) for well in drill_zone.list_project_wells],
                 'Характер работы': ['1'] * len(drill_zone.list_project_wells),  # 1 - добывающая, 2 - нагнетательная
                 'Тип скважины': [well.well_type for well in drill_zone.list_project_wells],
                 'Длина, м': [round(well.length_geo, 1) for well in drill_zone.list_project_wells],
                 'Азимут, градусы': [round(well.azimuth, 1) for well in drill_zone.list_project_wells],
                 'Обводненность (объем), %': [round(well.water_cut, 1) for well in drill_zone.list_project_wells],
                 'Запускной дебит жидкости, м3/сут': [round(well.init_Ql_rate_V, 2) for well in
                                                      drill_zone.list_project_wells],
                 'Запускной дебит нефти, т/сут': [round(well.init_Qo_rate, 2) for well in
                                                  drill_zone.list_project_wells],
                 'Запускное забойное давление, атм': [round(well.P_well_init, 1) for well in
                                                      drill_zone.list_project_wells],
                 'Пластовое давление, атм': [round(well.P_reservoir, 1) for well in drill_zone.list_project_wells],
                 'Нефтенасыщенная толщина, м': [round(well.NNT, 1) for well in drill_zone.list_project_wells],
                 'Начальная нефтенасыщенность, д.ед': [round(well.So_init, 3) for well in
                                                       drill_zone.list_project_wells],
                 'Текущая нефтенасыщенность, д.ед': [round(well.So, 3) for well in drill_zone.list_project_wells],
                 'Пористость, д.ед': [round(well.m, 3) for well in drill_zone.list_project_wells],
                 'Проницаемость, мД': [round(well.permeability, 3) for well in drill_zone.list_project_wells],
                 'Эффективный радиус, м': [round(well.r_eff, 1) for well in drill_zone.list_project_wells],
                 'Запасы, тыс т': [round(well.reserves, 1) for well in drill_zone.list_project_wells],
                 'Накопленная добыча нефти (25 лет), тыс.т': [round(np.sum(well.Qo) / 1000, 1) for well in
                                                              drill_zone.list_project_wells],
                 'Накопленная добыча жидкости (25 лет), тыс.т': [round(np.sum(well.Ql) / 1000, 1) for well in
                                                                 drill_zone.list_project_wells],
                 'Соседние скважины': [well.gdf_nearest_wells.well_number.unique() for
                                       well in drill_zone.list_project_wells]
                 })
            if switch_economy:
                df_project_wells_economy = pd.DataFrame(
                    {'№ скважины': [well.well_number for well in drill_zone.list_project_wells],
                     'PI (Рентабельный период)': [well.PI for well in drill_zone.list_project_wells],
                     'NPV (Рентабельный период), тыс.руб.': [round(np.sum(well.NPV[well.NPV > 0])) for well in
                                                             drill_zone.list_project_wells],
                     'ГЭП': [well.year_economic_limit for well in drill_zone.list_project_wells],
                     })
                gdf_project_wells_ranking_drilling = gdf_project_wells_ranking_drilling.merge(df_project_wells_economy,
                                                                                              left_on='№ скважины',
                                                                                              right_on='№ скважины')
            gdf_result_ranking_drilling = pd.concat([gdf_result_ranking_drilling,
                                                     gdf_project_wells_ranking_drilling], ignore_index=True)

            [dict_project_wells_Qo.update({well.well_number: well.Qo}) for well in drill_zone.list_project_wells]
            [dict_project_wells_Ql.update({well.well_number: well.Ql}) for well in drill_zone.list_project_wells]
            [dict_project_wells_Qo_rate.update({well.well_number: well.Qo_rate})
             for well in drill_zone.list_project_wells]
            [dict_project_wells_Ql_rate.update({well.well_number: well.Ql_rate})
             for well in drill_zone.list_project_wells]

            if switch_economy:
                [dict_project_wells_cumulative_cash_flow.update({well.well_number: well.cumulative_cash_flow})
                 for well in drill_zone.list_project_wells]
                [dict_project_wells_CAPEX.update({well.well_number: well.CAPEX})
                 for well in drill_zone.list_project_wells]
                [dict_project_wells_OPEX.update({well.well_number: well.OPEX})
                 for well in drill_zone.list_project_wells]
                [dict_project_wells_NPV.update({well.well_number: well.NPV})
                 for well in drill_zone.list_project_wells]

    df_result_production_Qo = pd.DataFrame.from_dict(dict_project_wells_Qo, orient='index')
    df_result_production_Ql = pd.DataFrame.from_dict(dict_project_wells_Ql, orient='index')
    df_result_production_Qo_rate = pd.DataFrame.from_dict(dict_project_wells_Qo_rate, orient='index')
    df_result_production_Ql_rate = pd.DataFrame.from_dict(dict_project_wells_Ql_rate, orient='index')
    with pd.ExcelWriter(filename) as writer:
        gdf_result_ranking_drilling.to_excel(writer, sheet_name='РБ', index=False)
        df_result_production_Qo.to_excel(writer, sheet_name='Добыча нефти, т')
        df_result_production_Ql.to_excel(writer, sheet_name='Добыча жидкости, т')
        df_result_production_Qo_rate.to_excel(writer, sheet_name='Дебит нефти, т_сут')
        df_result_production_Ql_rate.to_excel(writer, sheet_name='Дебит жидкости, т_сут')

    if switch_economy:
        df_result_cumulative_cash_flow = pd.DataFrame.from_dict(dict_project_wells_cumulative_cash_flow, orient='index')
        df_result_CAPEX = pd.DataFrame.from_dict(dict_project_wells_CAPEX, orient='index')
        df_result_OPEX = pd.DataFrame.from_dict(dict_project_wells_OPEX, orient='index')
        df_result_NPV = pd.DataFrame.from_dict(dict_project_wells_NPV, orient='index')
        with pd.ExcelWriter(filename, mode='a', engine='openpyxl') as writer:
            df_result_cumulative_cash_flow.to_excel(writer, sheet_name='Накопленный FCF, тыс руб')
            df_result_CAPEX.to_excel(writer, sheet_name='CAPEX, тыс руб')
            df_result_OPEX.to_excel(writer, sheet_name='OPEX, тыс руб')
            df_result_NPV.to_excel(writer, sheet_name='NPV, тыс руб')
    pass


def save_picture_clustering_zones(list_zones, filename, buffer_project_wells):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 10))
    for drill_zone in list_zones:
        if drill_zone.num_project_wells != 0:
            ax = drill_zone.picture_clustering(ax, buffer_project_wells)
    plt.gca().invert_yaxis()
    plt.savefig(filename, dpi=400)
    pass


def save_map_permeability_fact_wells(data_wells, map_pressure, filename, accounting_GS, radius_interpolate):
    map_permeability_fact_wells = read_array(data_wells,
                                             name_column_map="permeability_fact",
                                             type_map="permeability_fact_wells",
                                             geo_transform=map_pressure.geo_transform,
                                             size=map_pressure.data.shape,
                                             accounting_GS=accounting_GS,
                                             radius=radius_interpolate)

    map_permeability_fact_wells.data = np.where(np.isnan(map_permeability_fact_wells.data), 0,
                                                map_permeability_fact_wells.data)
    map_permeability_fact_wells.save_img(filename, data_wells)
    map_permeability_fact_wells.save_grd_file(
        f"{filename.replace('.png', '').replace('/изображения png', '/карты grd')}.grd")
    pass


def create_df_project_wells(list_zones):
    df_result_project_wells = pd.DataFrame()
    for drill_zone in list_zones:
        if drill_zone.rating != -1:
            data_project_wells = pd.DataFrame([well.__dict__ for well in drill_zone.list_project_wells])
            df_result_project_wells = pd.concat([df_result_project_wells, data_project_wells], ignore_index=True)
    if not df_result_project_wells.empty:
        df_result_project_wells['T1_x_geo'] = df_result_project_wells['POINT_T1_geo'].apply(lambda point: point.x)
        df_result_project_wells['T1_y_geo'] = df_result_project_wells['POINT_T1_geo'].apply(lambda point: point.y)
        df_result_project_wells['T3_x_geo'] = df_result_project_wells['POINT_T3_geo'].apply(lambda point: point.x)
        df_result_project_wells['T3_y_geo'] = df_result_project_wells['POINT_T3_geo'].apply(lambda point: point.y)
        df_result_project_wells['T1_x_pix'] = df_result_project_wells['POINT_T1_pix'].apply(lambda point: point.x)
        df_result_project_wells['T1_y_pix'] = df_result_project_wells['POINT_T1_pix'].apply(lambda point: point.y)
        df_result_project_wells['T3_x_pix'] = df_result_project_wells['POINT_T3_pix'].apply(lambda point: point.x)
        df_result_project_wells['T3_y_pix'] = df_result_project_wells['POINT_T3_pix'].apply(lambda point: point.y)
        df_result_project_wells['permeability_fact'] = df_result_project_wells['permeability']
    return df_result_project_wells


def save_picture_voronoi(df_Coordinates, filename, type_coord="geo", default_size_pixel=1):
    """Сохранение картинки с ячейками Вороных"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if type_coord == 'geo':
        LINESTRING = 'LINESTRING_geo'
    elif type_coord == 'pix':
        LINESTRING = 'LINESTRING_pix'
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

    fig, ax = plt.subplots(figsize=(20, 50))

    boundary.plot(color='white', edgecolor='black', ax=ax)
    vd.plot(ax=ax, color='blue')  # cmap="winter"
    vd.boundary.plot(ax=ax, color='white')

    gdf_Coordinates_current = gdf_Coordinates[gdf_Coordinates['work_marker'].notna()].copy()
    gdf_Coordinates_current.set_geometry("LINESTRING_geo").plot(color='black', markersize=50, ax=ax)
    gdf_Coordinates_current.set_geometry("POINT_T1_geo").plot(color='black', markersize=10, ax=ax)

    gdf_Coordinates_project = gdf_Coordinates[gdf_Coordinates['work_marker'].isna()].copy()
    gdf_Coordinates_project.set_geometry("LINESTRING_geo").plot(color='red', markersize=50, ax=ax)
    gdf_Coordinates_project.set_geometry("POINT_T1_geo").plot(color='red', markersize=10, ax=ax)

    # Добавление текста с именами скважин рядом с точками T1
    for point, name in zip(gdf_Coordinates['POINT_T1_geo'], gdf_Coordinates['well_number']):
        if point is not None:  # Проверяем, что линия не пустая
            plt.text(point.x + 30, point.y - 30, name, fontsize=6, ha='left')  # Координаты (x, y)
    plt.savefig(filename + '/.debug/voronoy.png')
    plt.close(fig)
    pass


def remove_keys(data, keys):
    """
    Функция для очистки словаря от лишних ключей
    """
    if isinstance(data, dict):
        for key in keys:
            data.pop(key, None)
        for value in data.values():
            if isinstance(value, (dict, list)):
                remove_keys(value, keys)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                remove_keys(item, keys)
    return data


def dict_to_df(data: Dict[str, Any], translation_dict: Dict[str, str] = None) -> pd.DataFrame:
    """
    Преобразует вложенный словарь в вертикальный DataFrame.

    Args:
        data: Вложенный словарь для преобразования
        translation_dict: Словарь переводов ключей {английский: русский}

    Returns:
        pd.DataFrame с вертикальной структурой

    Пример:
        Вход: {"user": {"name": "John", "info": {"age": 30}}}
        Выход: DataFrame с колонками:
            Level1  Level2  Level3  Value
            user    name    NaN     John
            user    info    age     30
    """

    def translate_key(key: str, translations: Dict[str, str]) -> str:
        """Переводит ключ, если есть перевод"""
        if translations and key in translations:
            return translations[key]
        return key

    def flatten_dict_recursive(
            nested_dict: Dict[str, Any],
            path: List[str] = None,
            result: List[List[str]] = None,
            translations: Dict[str, str] = None
    ) -> List[List[str]]:
        """
        Рекурсивно преобразует вложенный словарь в список строк для DataFrame
        """
        if path is None:
            path = []
        if result is None:
            result = []

        for key, value in nested_dict.items():
            # Переводим ключ
            translated_key = translate_key(key, translations)
            current_path = path + [translated_key]

            if isinstance(value, dict):
                # Рекурсивно обрабатываем вложенный словарь
                flatten_dict_recursive(value, current_path, result, translations)
            elif isinstance(value, list):
                # Обрабатываем списки - создаем отдельную строку для каждого элемента
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        # Если элемент списка - словарь, обрабатываем его
                        flatten_dict_recursive(
                            item,
                            current_path + [f"item_{i + 1}"],
                            result,
                            translations
                        )
                    else:
                        # Если простой элемент списка
                        row = current_path + [""] * (max_levels - len(current_path)) + [item]
                        result.append(row)
            else:
                # Простое значение - добавляем строку
                row = current_path + [""] * (max_levels - len(current_path)) + [value]
                result.append(row)

        return result

    # Сначала определяем максимальную глубину вложенности
    def get_max_depth(d: Dict[str, Any], current_depth: int = 1) -> int:
        """Определяет максимальную глубину вложенности словаря"""
        max_depth = current_depth
        for value in d.values():
            if isinstance(value, dict):
                depth = get_max_depth(value, current_depth + 1)
                max_depth = max(max_depth, depth)
            elif isinstance(value, list):
                # Проверяем элементы списка
                for item in value:
                    if isinstance(item, dict):
                        depth = get_max_depth(item, current_depth + 1)
                        max_depth = max(max_depth, depth)
        return max_depth

    # Определяем максимальную глубину
    max_depth = get_max_depth(data)

    # Глобальная переменная для использования в рекурсивной функции
    global max_levels
    max_levels = max_depth

    # Преобразуем словарь в плоский список
    flat_data = flatten_dict_recursive(data, translations=translation_dict)

    # Создаем имена колонок
    column_names = [f"Уровень_{i + 1}" for i in range(max_depth)] + ["Значение"]

    # Создаем DataFrame
    df = pd.DataFrame(flat_data, columns=column_names)

    return df