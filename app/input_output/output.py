import os

import alphashape
import win32api

from loguru import logger
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely import Point, Polygon, MultiPolygon, MultiPoint

from app.maps_handler.maps import read_array


@logger.catch
def upload_data(save_directory, data_wells, maps, list_zones, info_clusterization_zones, **kwargs):
    """Выгрузка данных после расчета"""
    type_map_list = list(map(lambda raster: raster.type_map, maps))

    dict_calculated_maps = {'residual_recoverable_reserves': "ОИЗ",
                            'water_cut': "обводненность",
                            'reservoir_score': "оценка резервуара",
                            'potential_score': "оценка потенциала пласта",
                            'risk_score': "оценка риска",
                            'opportunity_index': "индекс возможности бурения",
                            'last_rate_oil': "последний дебит",
                            'init_rate_oil': "запускной дебит"}

    logger.info(f"Сохраняем исходные карты и рассчитанные в .png и .grd форматах ")
    for i, raster in enumerate(maps):
        if raster.type_map in dict_calculated_maps.keys():
            raster.save_img(f"{save_directory}/{dict_calculated_maps.get(raster.type_map)}.png", data_wells)
            raster.save_grd_file(f"{save_directory}/{dict_calculated_maps.get(raster.type_map)}.grd")
            if raster.type_map == 'opportunity_index':
                logger.info(f"Сохраняем .png карту OI с зонами")
                raster.save_img(f"{save_directory}/карта индекса возможности бурения с зонами.png", data_wells,
                                list_zones, info_clusterization_zones)

    data_project_wells = create_df_project_wells(list_zones)
    data_all_wells = pd.concat([data_wells, data_project_wells], ignore_index=True)

    logger.info("Сохранение карты фактической проницаемости через РБ в форматах .png и .grd")
    map_pressure = maps[type_map_list.index('pressure')]
    save_map_permeability_fact_wells(data_all_wells, map_pressure,
                                     f"{save_directory}/фактическая проницаемость через РБ.png",
                                     radius_interpolate=kwargs['radius_interpolate'],
                                     accounting_GS=kwargs['accounting_GS'])

    logger.info(f"Сохраняем .png с начальным расположением проектного фонда в кластерах и карту ОИЗ с проектным фондом")
    save_picture_clustering_zones(list_zones, f"{save_directory}/начальное расположение ПФ.png",
                                  buffer_project_wells=kwargs['buffer_project_wells'])
    map_residual_recoverable_reserves = maps[type_map_list.index('residual_recoverable_reserves')]
    map_residual_recoverable_reserves.save_img(f"{save_directory}/карта ОИЗ с ПФ.png", data_wells,
                                   list_zones, info_clusterization_zones, project_wells=True)
    logger.info("Сохранение рейтинга бурения проектных скважин в формате .xlsx")
    save_ranking_drilling_to_excel(list_zones, f"{save_directory}/рейтинг_бурения.xlsx")

    logger.info("Сохранение контуров зон в формате .txt для загрузки в NGT")
    save_directory_contours = f"{save_directory}/контуры зон"
    create_new_dir(save_directory_contours)
    save_contours(list_zones, map_residual_recoverable_reserves, save_directory_contours, type_calc='alpha', buffer_size=40)
    pass


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
                    raise logger.error("Не удалось построить границу зоны. Проверьте размер buffer или входные данные.")
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
                    raise logger.error(
                        "Не удалось построить границу зоны. Проверьте параметр alpha или входные данные.")
            elif type_calc == 'convex_hull':
                mesh = list(map(lambda x, y: Point(x, y), x_coordinates, y_coordinates))
                ob = Polygon(mesh)
                # определяем границу зоны
                boundary_drill_zone = ob.convex_hull
                x_boundary, y_boundary = boundary_drill_zone.exterior.coords.xy
            else:
                raise logger.error(f"Проверьте значение параметра type_calc: {type_calc}")
            name_txt = f'{save_directory_contours}/{drill_zone.rating}.txt'
            with open(name_txt, "w") as file:
                file.write(f"/\n")
                for x, y in zip(x_boundary, y_boundary):
                    file.write(f"{x} {y}\n")
                file.write(f"{x_boundary[0]} {y_boundary[0]}\n")
    pass


def get_save_path(program_name: str = "default", field: str = "field", object_value: str = "object") -> str:
    """
    Получение пути на запись
    :return:
    """
    path_program = os.getcwd()
    # Проверка возможности записи в директорию программы
    if os.access(path_program, os.W_OK):
        if "\\app" in path_program:
            path_program = path_program.replace("\\app", "")
        if "\\drill_zones" in path_program:
            path_program = path_program.replace("\\drill_zones", "")
        save_path = f"{path_program}\\output\\{field}_{object_value}"
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
                logger.error(PermissionError)

        current_user = os.getlogin()
        profile_dir = [dir_ for dir_ in os.listdir(save_drive) if dir_.lower() == "profiles"
                       or dir_.upper() == "PROFILES"]

        if len(profile_dir) < 1:
            save_path = f"{save_drive}\\{program_name}_output\\{field}_{object_value}"
        else:
            save_path = (f"{save_drive}\\{profile_dir[0]}\\{current_user}\\"
                         f"{program_name}_output\\{field}_{object_value}")

    create_new_dir(save_path)
    return save_path


def create_new_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path)
    return


def save_ranking_drilling_to_excel(list_zones, filename):
    gdf_result_ranking_drilling = gpd.GeoDataFrame()
    dict_project_wells_Qo, dict_project_wells_Ql = {}, {}
    dict_project_wells_Qo_rate, dict_project_wells_Ql_rate = {}, {}
    for drill_zone in list_zones:
        if drill_zone.rating != -1:
            # gdf_project_wells = gpd.GeoDataFrame([well.__dict__ for well in drill_zone.list_project_wells])
            gdf_project_wells_ranking_drilling = gpd.GeoDataFrame(
                {'№ скважины': [well.well_number for well in drill_zone.list_project_wells],
                 'Координата_T1_x': [round(well.POINT_T1_geo.x, 0) for well in drill_zone.list_project_wells],
                 'Координата_T1_y': [round(well.POINT_T1_geo.y, 0) for well in drill_zone.list_project_wells],
                 'Координата_T3_x': [round(well.POINT_T3_geo.x, 0)  for well in drill_zone.list_project_wells],
                 'Координата_T3_y': [round(well.POINT_T3_geo.y, 0)  for well in drill_zone.list_project_wells],
                 'Характер работы': ['1'] * len(drill_zone.list_project_wells),  # 1 - добывающая, 2 - нагнетательная
                 'Тип скважины': [well.well_type for well in drill_zone.list_project_wells],
                 'Длина, м': [round(well.length_geo, 1) for well in drill_zone.list_project_wells],
                 'Азимут, градусы': [round(well.azimuth, 1) for well in drill_zone.list_project_wells],
                 'Обводненность, %': [round(well.water_cut, 1) for well in drill_zone.list_project_wells],
                 'Запускной дебит жидкости, т/сут': [round(well.init_Ql_rate, 2) for well in drill_zone.list_project_wells],
                 'Запускной дебит нефти, т/сут': [round(well.init_Qo_rate, 2) for well in drill_zone.list_project_wells],
                 'Запускное забойное давление, атм': [round(well.P_well_init, 1) for well in drill_zone.list_project_wells],
                 'Пластовое давление, атм': [round(well.P_reservoir, 1) for well in drill_zone.list_project_wells],
                 'Нефтенасыщенная толщина, м': [round(well.NNT, 1) for well in drill_zone.list_project_wells],
                 'Начальная нефтенасыщенность, д.ед': [round(well.So, 3) for well in drill_zone.list_project_wells],
                 'Пористость, д.ед': [round(well.m, 3) for well in drill_zone.list_project_wells],
                 'Проницаемость, мД': [round(well.permeability, 3) for well in drill_zone.list_project_wells],
                 'Эффективный радиус, м': [round(well.r_eff, 1) for well in drill_zone.list_project_wells],
                 'Запасы, тыс т': [round(well.reserves, 1) for well in drill_zone.list_project_wells],
                 'Накопленная добыча нефти (25 лет), тыс.т': [round(np.sum(well.Qo) / 1000, 1) for well in
                                                              drill_zone.list_project_wells],
                 'Накопленная добыча жидкости (25 лет), тыс.т': [round(np.sum(well.Ql) / 1000, 1) for well in
                                                                 drill_zone.list_project_wells],
                 'Соседние скважины': [well.gdf_nearest_wells.well_number.unique() for
                                       well in drill_zone.list_project_wells],
                 'Статус аппроксимации Арпса': [well.decline_rates[0][4] for well in drill_zone.list_project_wells]}
            )
            gdf_result_ranking_drilling = pd.concat([gdf_result_ranking_drilling,
                                                     gdf_project_wells_ranking_drilling], ignore_index=True)

            [dict_project_wells_Qo.update({well.well_number: well.Qo}) for well in drill_zone.list_project_wells]
            [dict_project_wells_Ql.update({well.well_number: well.Ql}) for well in drill_zone.list_project_wells]
            [dict_project_wells_Qo_rate.update({well.well_number: well.Qo_rate})
             for well in drill_zone.list_project_wells]
            [dict_project_wells_Ql_rate.update({well.well_number: well.Ql_rate})
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
    pass


def save_picture_clustering_zones(list_zones, filename, buffer_project_wells):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 10))
    for drill_zone in list_zones:
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
    map_permeability_fact_wells.save_grd_file(f"{filename.replace('.png', '')}.grd")
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
