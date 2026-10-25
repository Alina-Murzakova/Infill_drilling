import pickle
from loguru import logger
import pandas as pd

from app.config import translation_dict_local_parameters
from app.input_output.output_functions import summary_table, create_new_dir, save_map_permeability_fact_wells, \
    save_ranking_drilling_to_excel, save_picture_clustering_zones, dict_to_df, create_df_project_wells, \
    save_local_parameters, save_contours, save_excel_permeability_fact_wells, save_excel_inj_wells


def upload_data(name_field, name_object, save_directory, data_wells, maps, list_zones, info_clusterization_zones,
                FEM, method_taxes, polygon_OI, data_history, data_wells_permeability_excel, parameters,
                default_size_pixel):
    """Выгрузка данных после расчета"""
    name_field = name_field.replace('/', "_")
    name_object = name_object.replace('/', "_")
    type_map_list = list(map(lambda raster: raster.type_map, maps))
    df_summary_table = summary_table(list_zones, parameters['switches']['switch_economy'])

    # Создание дополнительных директорий
    create_new_dir(f"{save_directory}/карты grd")
    create_new_dir(f"{save_directory}/изображения png")

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
            raster.save_img(f"{save_directory}/изображения png/{dict_calculated_maps.get(raster.type_map)}.png",
                            data_wells)
            raster.save_grd_file(f"{save_directory}/карты grd/{dict_calculated_maps.get(raster.type_map)}.grd")
            if raster.type_map == 'opportunity_index':
                logger.info(f"Сохраняем .png карту OI с зонами")
                raster.save_img(f"{save_directory}/изображения png/карта индекса возможности бурения с зонами.png",
                                data_wells, list_zones, info_clusterization_zones)

    data_project_wells = create_df_project_wells(list_zones)
    data_all_wells = pd.concat([data_wells, data_project_wells], ignore_index=True)

    logger.info("Сохранение карты фактической проницаемости через РБ в форматах .png и .grd")
    map_pressure = maps[type_map_list.index('pressure')]

    if not data_all_wells[data_all_wells['permeability_fact'] != 0].empty:
        save_map_permeability_fact_wells(data_all_wells, map_pressure,
                                         f"{save_directory}/изображения png/фактическая проницаемость через РБ.png",
                                         radius_interpolate=parameters['maps']['radius_interpolate'],
                                         accounting_GS=parameters['switches']['switch_accounting_horwell'])

    logger.info(f"Сохраняем .png с начальным расположением проектного фонда в кластерах и карту ОИЗ с проектным фондом")
    save_picture_clustering_zones(list_zones, f"{save_directory}/изображения png/начальное расположение ПФ.png",
                                  buffer_project_wells=parameters['well_params']['proj_wells_params']
                                                       ['buffer_project_wells'] / default_size_pixel)
    map_residual_recoverable_reserves = maps[type_map_list.index('residual_recoverable_reserves')]
    map_residual_recoverable_reserves.save_img(f"{save_directory}/изображения png/карта ОИЗ с ПФ.png", data_wells,
                                               list_zones, info_clusterization_zones, project_wells=True)
    logger.info("Сохранение рейтинга бурения проектных скважин в формате .xlsx")
    save_ranking_drilling_to_excel(name_field, name_object, list_zones,
                                   f"{save_directory}/Рейтинг_бурения_{name_field}_{name_object}.xlsx",
                                   parameters['switches']['switch_economy'])

    logger.info("Сохранение pickle файлов")
    with open(f'{save_directory}/.debug/data_wells.pickle', 'wb') as file:
        pickle.dump(data_wells, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'{save_directory}/.debug/list_zones.pickle', 'wb') as file:
        pickle.dump(list_zones, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'{save_directory}/.debug/maps.pickle', 'wb') as file:
        pickle.dump(maps, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'{save_directory}/.debug/data_history.pickle', 'wb') as file:
        pickle.dump(data_history, file, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("Сохранение фактической проницаемости по факт фонду")
    save_excel_permeability_fact_wells(data_wells_permeability_excel, save_directory)

    logger.info("Сохранение полезной закачки")
    save_excel_inj_wells(data_wells, save_directory)

    logger.info("Сохранение контуров зон в формате .txt для загрузки в NGT")
    save_directory_contours = f"{save_directory}/контуры зон"
    create_new_dir(save_directory_contours)

    saved_zones, failed_zones = save_contours(
        list_zones,
        map_residual_recoverable_reserves,
        save_directory_contours,
        type_calc='alpha',
        buffer_size=40,
        alpha=0.01
    )

    logger.info(f"Сохранено контуров: {len(saved_zones)}")

    if failed_zones:
        logger.warning(f"Контуры не построены для зон: {failed_zones}")

    logger.info("Сохранение local_parameters")
    save_local_parameters(parameters, f"{save_directory}/.debug/local_parameters.py")

    logger.info("Сохранение .xlsx с основными параметрами расчета и сводной таблицей")
    # Переводим parameters в df
    df_parameters = dict_to_df(parameters, translation_dict_local_parameters)
    # Сохраняем в Excel
    with pd.ExcelWriter(f"{save_directory}/info.xlsx") as writer:
        df_summary_table.to_excel(writer, sheet_name='Сводная таблица', index=False)
        df_parameters.to_excel(writer, sheet_name='Параметры расчета', index=False)

    return df_summary_table
