import math
import sys
import geopandas as gpd
import pandas as pd
from loguru import logger
from tqdm import tqdm

from app.input_output.input_economy import load_economy_data
from app.input_output.input_frac_info import load_frac_info
from app.input_output.input_geo_phys_properties import load_geo_phys_properties
from app.input_output.input_wells_data import load_wells_data, prepare_wells_data
from app.ranking_drilling.starting_rates import get_df_permeability_fact_wells
from app.decline_rate.decline_rate import get_decline_rates
from app.maps_handler.functions import mapping, calculate_reservoir_state_maps, calculate_score_maps
from app.well_active_zones import calculate_effective_radius
from app.drill_zones.drilling_zones import calculate_drilling_zones
from app.project_wells import calculate_reserves_by_voronoi
from app.input_output.output_functions import get_save_path, create_new_dir, save_local_parameters
from app.input_output.output import upload_data
from app.reservoir_kr_optimizer import get_reservoir_kr
from app.exceptions import CalculationCancelled


def run_model(parameters, total_stages, progress=None, is_cancelled=None):
    import logging
    logging.basicConfig(level=logging.INFO, )
    # Настраиваем библиотеку reservoir_maps
    my_library_logger = logging.getLogger('reservoir_maps')
    my_library_logger.setLevel(logging.INFO)
    stage_number = -1

    def log_stage(msg):
        """Функция для логирования и определения № шага расчета"""
        nonlocal stage_number
        stage_number += 1
        logger.info(msg)
        if progress:
            progress(round(stage_number / total_stages * 100))  # обновляем прогресс расчета
        if is_cancelled and is_cancelled():
            raise CalculationCancelled()

    log_stage("ИНИЦИАЛИЗАЦИЯ ЛОКАЛЬНЫХ ПЕРЕМЕННЫХ")
    # Пути
    paths = parameters['paths']
    # Параметры расчета
    drill_zones_params = parameters['drill_zones']
    is_exe = getattr(sys, 'frozen', False) or hasattr(sys, '_MEIPASS')

    if parameters['well_params']['proj_wells_params']['buffer_project_wells'] <= 0:
        # нижнее ограничение на расстояние до фактических скважин от проектной
        parameters['well_params']['proj_wells_params']['buffer_project_wells'] = 10

    log_stage("ЗАГРУЗКА СКВАЖИННЫХ ДАННЫХ")
    data_history, info_object_calculation = load_wells_data(data_well_directory=paths["data_well_directory"])
    name_field, name_object = info_object_calculation.get("field"), info_object_calculation.get("object_value")
    save_directory = paths['save_directory']
    # Создание локальной папки Месторождение_Объект
    # save_directory = f"{paths['save_directory']}\\{name_field}_{name_object.replace('/', '-')}"
    # создание пути под системные файлы
    create_new_dir(f"{save_directory}/.debug")
    logger.add(f"{save_directory}/.debug/logs.log", mode='w', encoding='utf-8')

    log_stage(f"ЗАГРУЗКА ГФХ ПО ПЛАСТУ {name_object.replace('/', '-')} МЕСТОРОЖДЕНИЯ {name_field}")
    dict_geo_phys_properties = load_geo_phys_properties(paths["path_geo_phys_properties"], name_field, name_object)
    parameters["reservoir_fluid_properties"].update(dict_geo_phys_properties)

    log_stage("ПОДГОТОВКА СКВАЖИННЫХ ДАННЫХ")
    data_history, data_wells = (
        prepare_wells_data(data_history, dict_properties=parameters,
                           first_months=parameters['well_params']['fact_wells_params']['first_months']))

    log_stage(f"ЗАГРУЗКА ФРАК-ПАРАМЕТРОВ")
    data_wells, parameters = load_frac_info(paths["path_frac"], data_wells, name_object, parameters)

    log_stage("ЗАГРУЗКА И ОБРАБОТКА КАРТ")
    maps, data_wells, maps_to_calculate = mapping(maps_directory=paths["maps_directory"],
                                                  data_wells=data_wells,
                                                  **{**parameters['maps'], **parameters['switches']})
    default_size_pixel = maps[0].geo_transform[1]  # размер ячейки после загрузки всех карт

    log_stage("РАСЧЕТ РАДИУСОВ ДРЕНИРОВАНИЯ И НАГНЕТАНИЯ ДЛЯ СКВАЖИН")
    data_wells = calculate_effective_radius(data_wells, dict_properties=parameters, is_exe=is_exe)

    log_stage(f"ЗАГРУЗКА ОФП, {parameters['switches']['switch_adaptation_relative_permeability']}")
    if parameters['switches']['switch_adaptation_relative_permeability']:
        parameters = get_reservoir_kr(data_history.copy(), data_wells.copy(), parameters)

    log_stage(f"РАСЧЕТ КАРТ ТЕКУЩЕГО СОСТОЯНИЯ: ОБВОДНЕННОСТИ И ОИЗ, {any(maps_to_calculate.values())}")
    if any(maps_to_calculate.values()):
        maps = calculate_reservoir_state_maps(data_wells,
                                              maps,
                                              parameters,
                                              default_size_pixel,
                                              maps_to_calculate,
                                              maps_directory=paths["maps_directory"])

    log_stage(f"РАСЧЕТ ОЦЕНОЧНЫХ КАРТ")
    maps = maps + calculate_score_maps(maps=maps, dict_properties=parameters['reservoir_fluid_properties'])

    log_stage("РАСЧЕТ ПРОНИЦАЕМОСТИ ДЛЯ ФАКТИЧЕСКИХ СКВАЖИН ЧЕРЕЗ РБ")
    data_wells, parameters, data_wells_permeability_excel = get_df_permeability_fact_wells(data_wells, parameters)

    log_stage("ОЦЕНКА ТЕМПОВ ПАДЕНИЯ ДЛЯ ТЕКУЩЕГО ФОНДА")
    data_decline_rate_stat, _, _ = get_decline_rates(data_history, data_wells, is_exe=is_exe)

    log_stage("РАСЧЕТ ЗОН С ВЫСОКИМ ИНДЕКСОМ БУРЕНИЯ")
    # Параметры кластеризации
    epsilon = drill_zones_params["min_radius"] / default_size_pixel
    min_samples = int(drill_zones_params["sensitivity_quality_drill"] / 100 * epsilon ** 2 * math.pi)
    min_samples = 1 if min_samples == 0 else min_samples
    percent_low = 100 - drill_zones_params["percent_top"]
    list_zones, info_clusterization_zones = calculate_drilling_zones(maps=maps,
                                                                     epsilon=epsilon,
                                                                     min_samples=min_samples,
                                                                     percent_low=percent_low,
                                                                     data_wells=data_wells,
                                                                     dict_properties=parameters)

    type_map_list = list(map(lambda raster: raster.type_map, maps))
    map_rrr = maps[type_map_list.index('residual_recoverable_reserves')]
    map_opportunity_index = maps[type_map_list.index('opportunity_index')]
    polygon_OI = map_opportunity_index.raster_to_polygon()
    log_stage("НАЧАЛЬНОЕ РАЗМЕЩЕНИЕ ПРОЕКТНЫХ СКВАЖИН")
    # Проектные скважины с других drill_zone, чтобы исключить пересечения
    gdf_project_wells_all = gpd.GeoDataFrame(columns=["LINESTRING_pix", "buffer"], geometry="LINESTRING_pix")
    for drill_zone in list_zones:
        if drill_zone.rating != -1:
            gdf_project_wells = drill_zone.get_init_project_wells(map_rrr, data_wells, gdf_project_wells_all,
                                                                  polygon_OI, default_size_pixel,
                                                                  drill_zones_params['init_profit_cum_oil'],
                                                                  parameters)
            gdf_project_wells_all = pd.concat([gdf_project_wells_all, gdf_project_wells], ignore_index=True)

    log_stage("РАСЧЕТ ЗАПАСОВ ДЛЯ ПРОЕКТНЫХ СКВАЖИН")
    calculate_reserves_by_voronoi(list_zones, data_wells, map_rrr, save_directory)

    log_stage(f"ЗАГРУЗКА ИСХОДНЫХ ДАННЫХ ДЛЯ РАСЧЕТА ЭКОНОМИКИ, {parameters['switches']['switch_economy']}")
    if parameters['switches']['switch_economy']:
        FEM, method_taxes, dict_NDD = load_economy_data(paths['path_economy'], name_field,
                                                        parameters['reservoir_fluid_properties']['gor'])
    else:
        FEM, method_taxes, dict_NDD = None, None, None

    well_params_economy = (parameters['economy_params'] | parameters['well_params']["fracturing"]
                           | parameters['well_params']["proj_wells_params"])

    log_stage(f"РАСЧЕТ ЗАПУСКНЫХ ПАРАМЕТРОВ, ПРОФИЛЯ ДОБЫЧИ И ЭКОНОМИКИ ПРОЕКТНЫХ СКВАЖИН")
    for drill_zone in list_zones:
        if drill_zone.rating != -1:
            logger.info(f"Зона {drill_zone.rating}: расчет запускных, профиля и оценка экономики")
            drill_zone.calculate_starting_rates(maps, parameters)
            drill_zone.calculate_production(data_decline_rate_stat, well_params_economy['period_calculation'] * 12,
                                            well_params_economy['day_in_month'],
                                            parameters['well_params']["general"]['well_efficiency'])
            if parameters['switches']['switch_economy']:
                drill_zone.calculate_economy(FEM, well_params_economy, method_taxes, dict_NDD)

    log_stage(f"ВЫГРУЗКА ДАННЫХ РАСЧЕТА:")
    summary_table = upload_data(name_field, name_object, save_directory, data_wells, maps, list_zones,
                                info_clusterization_zones, FEM, method_taxes, polygon_OI, data_history,
                                data_wells_permeability_excel, parameters, default_size_pixel)
    log_stage(f"ЭКСПОРТ ЗАВЕРШЕН")

    return summary_table, save_directory


if __name__ == '__main__':
    from app.local_parameters import parameters
    from pathlib import Path
    from datetime import datetime
    import traceback
    import sys
    import time

    save_dir = get_save_path("Infill_drilling")
    parameters['paths']['save_directory'] = save_dir

    start_time = time.time()

    try:
        run_model(parameters, total_stages=None)

    except Exception as e:
        # ОШИБКА - сохраняем логи

        if save_dir:
            log_path = Path(save_dir) / ".debug" / f"error.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)

            # Формируем содержимое лога с ошибкой
            error_content = f"Ошибка выполнения\n"
            error_content += f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            error_content += f"Директория: {save_dir}\n"
            error_content += f"Ошибка: {str(e)}\n"
            error_content += f"Тип ошибки: {type(e).__name__}\n\n"
            error_content += "Traceback:\n"
            error_content += traceback.format_exc()
            error_content += f"\n\nВремя выполнения до ошибки: {time.time() - start_time:.2f} секунд"

            # Записываем в файл с режимом 'w'
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(error_content)

            logger.info("Сохранение local_parameters")
            save_local_parameters(parameters, f"{save_dir}/.debug/local_parameters.py")

            logger.info(f"Логи ошибки сохранены в: {log_path}")

        sys.exit(1)
