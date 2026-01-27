import pickle
from loguru import logger
import pandas as pd

from app.config import translation_dict_local_parameters
from app.input_output.output_functions import summary_table, create_new_dir, save_map_permeability_fact_wells, \
    save_ranking_drilling_to_excel, save_picture_clustering_zones, remove_keys, dict_to_df, create_df_project_wells


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
    save_map_permeability_fact_wells(data_all_wells, map_pressure,
                                     f"{save_directory}/изображения png/фактическая проницаемость через РБ.png",
                                     radius_interpolate=parameters['maps']['radius_interpolate'],
                                     accounting_GS=parameters['switches']['switch_accounting_horwell'])

    logger.info(f"Сохраняем .png с начальным расположением проектного фонда в кластерах и карту ОИЗ с проектным фондом")
    save_picture_clustering_zones(list_zones, f"{save_directory}/изображения png/начальное расположение ПФ.png",
                                  buffer_project_wells=parameters['well_params']['proj_wells_params']
                                  ['buffer_project_wells']/default_size_pixel)
    map_residual_recoverable_reserves = maps[type_map_list.index('residual_recoverable_reserves')]
    map_residual_recoverable_reserves.save_img(f"{save_directory}/изображения png/карта ОИЗ с ПФ.png", data_wells,
                                               list_zones, info_clusterization_zones, project_wells=True)
    logger.info("Сохранение рейтинга бурения проектных скважин в формате .xlsx")
    save_ranking_drilling_to_excel(name_field, name_object, list_zones,
                                   f"{save_directory}/рейтинг_бурения_{name_field}_{name_object}.xlsx",
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
    data_wells_permeability_excel = data_wells_permeability_excel[data_wells_permeability_excel['permeability_fact']
                                                                  != 0]
    data_wells_permeability_excel.columns = ['Номер скважины', 'характер', 'состояние', 'тип', 'дата',
                                             'эффективный радиус через площадь ячейки вороного, м',
                                             'длина ствола скважины T1-T3, м', 'количество стадий ГРП, шт',
                                             'полудлина трещины ГРП, м', 'раскрытие трещины ГРП, мм',
                                             'запускной Qж ТР, т/сут', 'стартовая обводненность ТР (объем), д.ед.',
                                             'запускное забойное давление добывающей скважины, атм',
                                             'стартовое пластовое давление ТР, атм',
                                             'нефтенасыщенная толщина, м', 'пористость, д.ед',
                                             'проницаемость c карты, мД', 'проницаемость обратным счетом через РБ, мД']
    with pd.ExcelWriter(f"{save_directory}/Фактическая_проницаемость_скважин.xlsx") as writer:
        data_wells_permeability_excel.to_excel(writer)

    # logger.info("Сохранение контуров зон в формате .txt для загрузки в NGT")
    # save_directory_contours = f"{save_directory}/контуры зон"
    # create_new_dir(save_directory_contours)
    # save_contours(list_zones, map_residual_recoverable_reserves, save_directory_contours, type_calc='alpha',
    #               buffer_size=40)

    logger.info("Сохранение local_parameters")
    # Удаляем параметры, которые были рассчитаны
    list_keys = ['Bo', 'P_init', 'Pb', 'c_o', 'c_r', 'c_w', 'gor', 'k_h', 'mu_o', 'mu_w', 'rho',
                 'save_directory', 'all_P_wells_init']
    parameters = remove_keys(parameters, list_keys)
    # Сохранение local_parameters
    save_path = f"{save_directory}/.debug/local_parameters.py"
    with open(save_path, 'w', encoding='utf-8') as f:
        # Используем pprint для красивого форматирования
        import pprint

        f.write('import datetime\n\n')
        f.write('parameters = ')
        pprint.pprint(parameters, f, indent=4, width=100, depth=None)

    logger.info("Сохранение .xlsx с основными параметрами расчета и сводной таблицей")
    # Переводим parameters в df
    df_parameters = dict_to_df(parameters, translation_dict_local_parameters)
    # Сохраняем в Excel
    with pd.ExcelWriter(f"{save_directory}/info.xlsx") as writer:
        df_summary_table.to_excel(writer, sheet_name='Сводная таблица', index=False)
        df_parameters.to_excel(writer, sheet_name='Параметры расчета', index=False)

    return df_summary_table

