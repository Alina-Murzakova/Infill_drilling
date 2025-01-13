import os
import win32api

from loguru import logger
import geopandas as gpd
import pandas as pd
import numpy as np

from app.maps_handler.maps import read_array


def upload_data(save_directory, data_wells, maps, list_zones, info_clusterization_zones, buffer_project_wells):
    """Выгрузка данных после расчета"""
    type_map_list = list(map(lambda raster: raster.type_map, maps))

    type_calculated_maps = ['reservoir_score', 'potential_score', 'risk_score', 'opportunity_index']
    logger.info(f"Сохраняем исходные карты и рассчитанные в .png и .grd форматах ")
    for i, raster in enumerate(maps):
        if raster.type_map in type_calculated_maps:
            raster.save_img(f"{save_directory}/{raster.type_map}.png", data_wells)
            raster.save_grd_file(f"{save_directory}/{raster.type_map}.grd")
            if raster.type_map == 'opportunity_index':
                logger.info(f"Сохраняем .png карту OI с зонами")
                raster.save_img(f"{save_directory}/map_opportunity_index_with_zones.png", data_wells,
                                list_zones, info_clusterization_zones)
        else:
            raster.save_img(f"{save_directory}/{raster.type_map}.png", data_wells)

    data_project_wells = create_df_project_wells(list_zones)
    data_all_wells = pd.concat([data_wells, data_project_wells], ignore_index=True)

    logger.info("Сохранение карты фактической проницаемости через РБ")
    map_pressure = maps[type_map_list.index('pressure')]
    save_map_permeability_fact_wells(data_all_wells, map_pressure, f"{save_directory}/permeability_fact_wells.png")

    logger.info(f"Сохраняем .png с начальным расположением проектного фонда в кластерах и карту ОИ с проектным фондом")
    save_picture_clustering_zones(list_zones, f"{save_directory}/init_project_wells.png", buffer_project_wells)
    map_opportunity_index = maps[type_map_list.index('residual_recoverable_reserves')]
    map_opportunity_index.save_img(f"{save_directory}/map_opportunity_index_with_project_wells.png", data_wells,
                                   list_zones, info_clusterization_zones, project_wells=True)
    logger.info("Сохранение рейтинг бурения проектных скважин в формате .xlsx")
    save_ranking_drilling_to_excel(list_zones, f"{save_directory}/ranking_drilling.xlsx")
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
                 'Тип скважины': [well.well_type for well in drill_zone.list_project_wells],
                 'Длина, м': [well.length_geo for well in drill_zone.list_project_wells],
                 'Координата T1': [well.POINT_T1_geo for well in drill_zone.list_project_wells],
                 'Координата T3': [well.POINT_T3_geo for well in drill_zone.list_project_wells],
                 'Азимут, градусы': [well.azimuth for well in drill_zone.list_project_wells],
                 'Обводненность, %': [well.water_cut for well in drill_zone.list_project_wells],
                 'Запускной дебит жидкости, т/сут': [well.init_Ql_rate for well in drill_zone.list_project_wells],
                 'Запускной дебит нефти, т/сут': [well.init_Qo_rate for well in drill_zone.list_project_wells],
                 'Запускное забойное давление, атм': [well.P_well_init for well in drill_zone.list_project_wells],
                 'Нефтенасыщенная толщина, м': [well.NNT for well in drill_zone.list_project_wells],
                 'Начальная нефтенасыщенность, д.ед': [well.So for well in drill_zone.list_project_wells],
                 'Пористость, д.ед': [well.m for well in drill_zone.list_project_wells],
                 'Проницаемость, мД': [well.permeability for well in drill_zone.list_project_wells],
                 'Эффективный радиус, м': [well.r_eff for well in drill_zone.list_project_wells],
                 'Запасы, тыс т': [well.P_reservoir for well in drill_zone.list_project_wells],
                 'Накопленная добыча нефти (25 лет), т': [np.sum(well.Qo) for well in drill_zone.list_project_wells],
                 'Накопленная добыча жидкости (25 лет), т': [np.sum(well.Ql) for well in drill_zone.list_project_wells],
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


def save_map_permeability_fact_wells(data_wells, map_pressure, filename):
    map_permeability_fact_wells = read_array(data_wells,
                                             name_column_map="permeability_fact",
                                             type_map="permeability_fact_wells",
                                             geo_transform=map_pressure.geo_transform,
                                             size=map_pressure.data.shape)

    map_permeability_fact_wells.data = np.where(np.isnan(map_permeability_fact_wells.data), 0,
                                                map_permeability_fact_wells.data)
    map_permeability_fact_wells.save_img(filename, data_wells)
    pass


def create_df_project_wells(list_zones):
    df_result_project_wells = pd.DataFrame()
    for drill_zone in list_zones:
        if drill_zone.rating != -1:
            data_project_wells = pd.DataFrame([well.__dict__ for well in drill_zone.list_project_wells])
            df_result_project_wells = pd.concat([df_result_project_wells, data_project_wells], ignore_index=True)

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
