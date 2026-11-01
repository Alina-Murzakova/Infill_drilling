import os
import alphashape
import platform
from decimal import Decimal
from datetime import datetime
from loguru import logger
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely import Point, Polygon, MultiPolygon, MultiPoint
from typing import Dict, Any, List

from app.maps_handler.maps import read_array
from app.well_active_zones import combine_mzs_to_polygon
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
         'Накопленная добыча\nнефти, тыс.т':
             [round(drill_zone.Qo / 1000, 2) if isinstance(drill_zone.Qo, float)
              else drill_zone.Qo for drill_zone in list_zones],
         'Накопленная добыча\nжидкости, тыс.т':
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
            round(df_summary_table['Накопленная добыча\nнефти, тыс.т'].sum(), 2),
            round(df_summary_table['Накопленная добыча\nжидкости, тыс.т'].sum(), 2),
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
            round(df_summary_table['Накопленная добыча\nнефти, тыс.т'].sum(), 2),
            round(df_summary_table['Накопленная добыча\nжидкости, тыс.т'].sum(), 2)]
    df_summary_table = df_summary_table.fillna('')
    return df_summary_table


def save_contours(
        list_zones,
        map_conv,
        save_directory_contours,
        type_calc='buffer',
        buffer_size=60,
        alpha=0.01
):
    """
    Сохранение контуров зон в формате .txt для загрузки в NGT в отдельную папку.

    Если stop_on_error=False, ошибки по отдельным зонам не останавливают расчет.
    Функция возвращает два списка:
    - saved_zones: успешно сохраненные зоны
    - failed_zones: зоны, которые не удалось сохранить
    """

    saved_zones = []
    failed_zones = []

    for drill_zone in list_zones:
        if drill_zone.rating == -1:
            continue

        try:
            logger.info(f"Сохранение контура зоны {drill_zone.rating}")

            x_coordinates = drill_zone.x_coordinates
            y_coordinates = drill_zone.y_coordinates

            # Проверка входных координат
            if x_coordinates is None or y_coordinates is None:
                raise ValueError("Координаты зоны отсутствуют")

            if len(x_coordinates) == 0 or len(y_coordinates) == 0:
                raise ValueError("Пустой массив координат зоны")

            if len(x_coordinates) != len(y_coordinates):
                raise ValueError(
                    f"Разная длина массивов координат: "
                    f"x={len(x_coordinates)}, y={len(y_coordinates)}"
                )

            x_coordinates, y_coordinates = map_conv.convert_coord_to_geo(
                (x_coordinates, y_coordinates)
            )

            if type_calc == 'buffer':
                points = MultiPoint(list(zip(x_coordinates, y_coordinates)))

                if points.is_empty:
                    raise ValueError("MultiPoint пустой после конвертации координат")

                buffered = points.buffer(buffer_size).simplify(0.01)

                if isinstance(buffered, Polygon):
                    x_boundary, y_boundary = buffered.exterior.xy
                elif isinstance(buffered, MultiPolygon):
                    largest_polygon = max(buffered.geoms, key=lambda p: p.area)
                    x_boundary, y_boundary = largest_polygon.exterior.xy

                    logger.warning(
                        f"Зона {drill_zone.rating}: buffer вернул MultiPolygon. "
                        f"Выбран самый большой полигон площадью "
                        f"{largest_polygon.area / 1000000:.3f} кв.км"
                    )
                else:
                    raise ValueError(
                        "Не удалось построить границу зоны через buffer. "
                        "Проверьте buffer_size или входные данные."
                    )

            elif type_calc == 'alpha':
                points = np.array(list(zip(x_coordinates, y_coordinates)))

                if len(points) < 3:
                    raise ValueError(
                        f"Недостаточно точек для построения полигона: {len(points)}"
                    )

                alpha_shape = alphashape.alphashape(points, alpha)

                if isinstance(alpha_shape, Polygon):
                    x_boundary, y_boundary = alpha_shape.exterior.xy

                elif isinstance(alpha_shape, MultiPolygon):
                    largest_polygon = max(alpha_shape.geoms, key=lambda p: p.area)
                    x_boundary, y_boundary = largest_polygon.exterior.xy

                    for poly in alpha_shape.geoms:
                        logger.info(
                            f"Площадь полигона MultiPolygon зоны "
                            f"{drill_zone.rating}: {poly.area / 1000000:.3f} кв.км"
                        )

                    logger.warning(
                        f"Зона {drill_zone.rating}: alpha_shape вернул MultiPolygon. "
                        f"Выбран самый большой полигон."
                    )

                else:
                    raise ValueError(
                        f"Не удалось построить границу зоны через alpha_shape. "
                        f"Тип результата: {type(alpha_shape)}. "
                        f"Проверьте alpha или входные данные."
                    )

            elif type_calc == 'convex_hull':
                if len(x_coordinates) < 3:
                    raise ValueError(
                        f"Недостаточно точек для convex_hull: {len(x_coordinates)}"
                    )

                mesh = [
                    Point(x, y)
                    for x, y in zip(x_coordinates, y_coordinates)
                ]

                ob = Polygon(mesh)
                boundary_drill_zone = ob.convex_hull

                if not isinstance(boundary_drill_zone, Polygon):
                    raise ValueError(
                        f"convex_hull не вернул Polygon. "
                        f"Тип результата: {type(boundary_drill_zone)}"
                    )

                x_boundary, y_boundary = boundary_drill_zone.exterior.coords.xy

            else:
                raise ValueError(
                    f"Некорректное значение параметра type_calc: {type_calc}"
                )

            # Дополнительная проверка результата
            if len(x_boundary) == 0 or len(y_boundary) == 0:
                raise ValueError("Получена пустая граница зоны")

            name_txt = f'{save_directory_contours}/{drill_zone.rating}.txt'

            with open(name_txt, "w", encoding="utf-8") as file:
                file.write("/\n")

                for x, y in zip(x_boundary, y_boundary):
                    file.write(f"{x} {y}\n")

                file.write(f"{x_boundary[0]} {y_boundary[0]}\n")

            saved_zones.append(drill_zone.rating)

            logger.info(
                f"Контур зоны {drill_zone.rating} успешно сохранен: {name_txt}"
            )

        except ValueError as error:
            failed_zones.append(drill_zone.rating)
            logger.warning(f"Контур зоны {drill_zone.rating} не построен: {error}")
            continue

        except Exception as error:
            failed_zones.append(drill_zone.rating)
            logger.error(f"Неожиданная ошибка при сохранении контура зоны {drill_zone.rating}:"
                         f" {type(error).__name__}: {error}")
            continue

    logger.info(
        f"Сохранение контуров завершено. "
        f"Успешно: {len(saved_zones)}, "
        f"с ошибками: {len(failed_zones)}"
    )

    if failed_zones:
        logger.warning(
            f"Не удалось сохранить контуры для зон: {failed_zones}"
        )

    return saved_zones, failed_zones


def get_save_path(program_name: str = "default") -> str:
    """
    Получение пути для сохранения результатов.
    """
    path_program = os.getcwd()
    current_datetime = datetime.now().strftime("%d.%m.%Y")

    # Определяем корень проекта
    if os.name == "nt":
        path_program = path_program.replace("\\app", "")
        path_program = path_program.replace("\\drill_zones", "")
        save_path = os.path.join(
            path_program,
            "output",
            current_datetime
        )

        try:
            create_new_dir(save_path)
        except PermissionError:
            # Резервная логика для Windows
            import win32api

            drives = win32api.GetLogicalDriveStrings()
            drives = drives.split('\000')

            list_drives = [
                drive for drive in drives
                if drive and "D:" in drive
            ]

            if not list_drives:
                list_drives = [
                    drive for drive in drives
                    if drive and "C:" in drive
                ]

            if not list_drives:
                error_msg = "У пользователя нет прав доступа на запись"
                logger.critical(error_msg)
                raise PermissionError(error_msg)

            save_drive = list_drives[0]

            current_user = os.getlogin()

            profile_dir = [
                dir_
                for dir_ in os.listdir(save_drive)
                if dir_.lower() == "profiles"
            ]

            if not profile_dir:
                save_path = os.path.join(
                    save_drive,
                    f"{program_name}_output",
                    current_datetime
                )
            else:
                save_path = os.path.join(
                    save_drive,
                    profile_dir[0],
                    current_user,
                    f"{program_name}_output",
                    current_datetime
                )

            create_new_dir(save_path)

    else:
        # Linux / Streamlit Cloud
        save_path = os.path.join(
            path_program,
            "output",
            current_datetime
        )
        create_new_dir(save_path)
    return save_path


def create_new_dir(path: str) -> None:
    """
    Создает директорию.

    Args:
        path: Путь к директории
    """
    # Создаем директорию (не вызовет ошибку если уже существует)
    os.makedirs(path, exist_ok=True)


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
                 'Накопленная добыча нефти, тыс.т': [round(np.sum(well.Qo) / 1000, 1) for well in
                                                     drill_zone.list_project_wells],
                 'Накопленная добыча жидкости, тыс.т': [round(np.sum(well.Ql) / 1000, 1) for well in
                                                        drill_zone.list_project_wells],
                 'Соседние скважины': [well.gdf_nearest_wells.well_number.unique() for
                                       well in drill_zone.list_project_wells],
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


def save_picture_voronoi(
        df_Coordinates,
        filename,
        type_coord="geo",
        default_size_pixel=1,
        label_mode="all",  # "all" | "corners" | "none"
        labels_count=4,
        voronoi_linewidth=1.4,
        well_linewidth=1.2,
        zone_alpha=0.22,
        label_fontsize=7,
        single_well_markersize=1.2,
        t1_markersize=2.5,
):
    """
    Сохранение прозрачной картинки с ячейками Вороного:
    - черные границы ячеек Вороного
    - прозрачный фон
    - стволы и точки T1
    - подписи скважин:
        * all     -> весь фонд
        * corners -> 4 фактические по углам области
        * none    -> без подписей
    - зоны эффективной добычи/закачки по r_eff_voronoy

    Для МЗС используется исходная логика:
    объединение через groupby(...).transform(combine_to_linestring)
    с последующим drop_duplicates по well_number_digit.
    """

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import geopandas as gpd

    from shapely.geometry import Polygon, MultiPolygon
    from shapely.geometry.base import BaseGeometry

    # ---------------------------
    # Выбор колонок геометрии
    # ---------------------------
    if type_coord == "geo":
        LINESTRING = "LINESTRING_geo"
        POINT_T1 = "POINT_T1_geo"
    elif type_coord == "pix":
        LINESTRING = "LINESTRING_pix"
        POINT_T1 = "POINT_T1_pix"
    else:
        error_msg = "Неверный тип координат."
        logger.critical(error_msg)
        raise TypeError(error_msg)

    df_Coordinates = df_Coordinates.copy()

    # ---------------------------
    # Вспомогательные функции
    # ---------------------------
    def is_valid_non_empty_geometry(g):
        return g is not None and hasattr(g, "is_empty") and not g.is_empty

    def filter_non_empty_geometry(gdf, geom_col):
        return gdf[gdf[geom_col].apply(is_valid_non_empty_geometry)].copy()

    def rounded_geometry(geometry, precision=0):
        """
        Округление координат полигона.
        На вход voronoiDiagram4plg лучше подавать целые координаты.
        """
        if geometry is None or geometry.is_empty:
            return geometry

        if isinstance(geometry, Polygon):
            rounded_exterior = [(round(x, precision), round(y, precision)) for x, y in geometry.exterior.coords]
            rounded_interiors = [
                [(round(x, precision), round(y, precision)) for x, y in interior.coords]
                for interior in geometry.interiors
            ]
            return Polygon(rounded_exterior, rounded_interiors)

        return geometry

    def get_zone_color(work_marker, is_project=False):
        if is_project:
            return "#dc143c"  # алый
        if pd.isna(work_marker):
            return "#dc143c"

        marker = str(work_marker).strip().lower()
        if marker == "inj":
            return "#43bff0"  # голубой
        if marker == "prod":
            return "#b87333"  # рыже-коричневый

        return "#808080"

    def plot_well_geometry(gdf, geom_col, ax, color, line_width=1.2, point_markersize=1.2, zorder=5):
        """
        Рисует геометрию скважин:
        - LineString / MultiLineString как линии
        - Point как маленькие точки
        """
        if gdf.empty:
            return

        gdf_geom = gdf[gdf[geom_col].notna()].copy()
        if gdf_geom.empty:
            return

        gdf_geom = gdf_geom[gdf_geom[geom_col].apply(is_valid_non_empty_geometry)].copy()
        if gdf_geom.empty:
            return

        gdf_geom = gdf_geom.set_geometry(geom_col)

        # Линии
        gdf_lines = gdf_geom[gdf_geom.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()
        if not gdf_lines.empty:
            gdf_lines.plot(
                ax=ax,
                color=color,
                linewidth=line_width,
                zorder=zorder
            )

        # Одиночные точки
        gdf_points = gdf_geom[gdf_geom.geometry.geom_type == "Point"].copy()
        if not gdf_points.empty:
            gdf_points.plot(
                ax=ax,
                color=color,
                markersize=point_markersize,
                zorder=zorder
            )

    def select_corner_wells_for_labels(gdf_points, point_col, count=4):
        """
        Выбирает скважины, ближайшие к 4 углам bounding box.
        Только из фактического фонда.
        """
        if gdf_points.empty:
            return gdf_points.iloc[0:0].copy()

        gdf_valid = gdf_points[gdf_points[point_col].notna()].copy()
        if gdf_valid.empty:
            return gdf_valid

        gdf_valid = filter_non_empty_geometry(gdf_valid, point_col)
        if gdf_valid.empty:
            return gdf_valid

        gdf_valid["_x"] = gdf_valid[point_col].apply(lambda p: p.x)
        gdf_valid["_y"] = gdf_valid[point_col].apply(lambda p: p.y)

        minx = gdf_valid["_x"].min()
        maxx = gdf_valid["_x"].max()
        miny = gdf_valid["_y"].min()
        maxy = gdf_valid["_y"].max()

        corners = [
            ("bottom_left", minx, miny),
            ("top_left", minx, maxy),
            ("top_right", maxx, maxy),
            ("bottom_right", maxx, miny),
        ]

        selected_idx = []
        used_idx = set()

        for _, cx, cy in corners:
            gdf_valid["_dist_corner"] = (gdf_valid["_x"] - cx) ** 2 + (gdf_valid["_y"] - cy) ** 2
            for idx in gdf_valid.sort_values("_dist_corner").index:
                if idx not in used_idx:
                    selected_idx.append(idx)
                    used_idx.add(idx)
                    break

        target_count = min(count, len(gdf_valid))
        if len(selected_idx) < target_count:
            center_x = (minx + maxx) / 2
            center_y = (miny + maxy) / 2
            gdf_valid["_dist_center"] = (gdf_valid["_x"] - center_x) ** 2 + (gdf_valid["_y"] - center_y) ** 2

            for idx in gdf_valid.sort_values("_dist_center", ascending=False).index:
                if idx not in used_idx:
                    selected_idx.append(idx)
                    used_idx.add(idx)
                if len(selected_idx) >= target_count:
                    break

        result = gdf_valid.loc[selected_idx].copy()
        drop_cols = [c for c in ["_x", "_y", "_dist_corner", "_dist_center"] if c in result.columns]
        result.drop(columns=drop_cols, inplace=True, errors="ignore")
        return result.head(count)

    def get_label_gdf(gdf_all, gdf_current, point_col, mode="all", count=4):
        if mode == "none":
            return gdf_all.iloc[0:0].copy()

        if mode == "corners":
            return select_corner_wells_for_labels(gdf_current, point_col, count=count)

        if mode == "all":
            gdf_valid = gdf_all[gdf_all[point_col].notna()].copy()
            if gdf_valid.empty:
                return gdf_valid
            return filter_non_empty_geometry(gdf_valid, point_col)

        raise ValueError("label_mode должен быть одним из: 'all', 'corners', 'none'")

    # ---------------------------
    # МЗС — возвращаем исходную логику
    # ---------------------------
    df_MZS = df_Coordinates[df_Coordinates.type_wellbore == "МЗС"].copy()
    df_other = df_Coordinates[df_Coordinates.type_wellbore != "МЗС"].copy()
    # буферизация обычных скважин || тк вороные строятся для полигонов буферизируем точки и линии скважин
    df_other["Polygon"] = (df_other.set_geometry(LINESTRING).buffer(1, resolution=3))

    if not df_MZS.empty:
        df_Coordinates_MZS = df_MZS.copy()
        # буферизация МЗС
        df_polygons_mzs = (df_Coordinates_MZS.groupby("well_number_digit")[LINESTRING].apply(combine_mzs_to_polygon)
                           .reset_index(name="Polygon"))
        df_Coordinates_MZS = (df_Coordinates_MZS.drop_duplicates(subset=["well_number_digit"], keep="first")
                              .drop(columns=["Polygon"], errors="ignore")
                              .merge(df_polygons_mzs, on="well_number_digit", how="left"))
        df_Coordinates = pd.concat([df_other, df_Coordinates_MZS], ignore_index=True)
    else:
        df_Coordinates = df_other.copy()

    gdf_Coordinates = gpd.GeoDataFrame(df_Coordinates, geometry=LINESTRING)

    # ---------------------------
    # Буферизация для Вороного
    # ---------------------------
    gdf_Coordinates["Polygon"] = gdf_Coordinates.set_geometry(LINESTRING).buffer(1, resolution=3)

    # ---------------------------
    # Внешняя граница
    # ---------------------------
    convex_hull = gdf_Coordinates.set_geometry("Polygon").union_all().convex_hull
    convex_hull = gpd.GeoDataFrame(geometry=[convex_hull]).buffer(1000 / default_size_pixel).boundary

    # ---------------------------
    # Подготовка входа для voronoiDiagram4plg
    # ---------------------------
    polygons_wells = gdf_Coordinates[["Polygon"]].copy()
    polygons_wells.columns = ["geometry"]
    polygons_wells["geometry"] = polygons_wells["geometry"].apply(rounded_geometry)

    hull_geom = convex_hull.iloc[0]
    if hasattr(hull_geom, "geoms"):
        hull_geom = list(hull_geom.geoms)[0]

    boundary_poly = MultiPolygon([rounded_geometry(Polygon(hull_geom))])
    boundary = gpd.GeoDataFrame({"geometry": [boundary_poly]})

    polygons_wells = polygons_wells.set_geometry("geometry")
    boundary = boundary.set_geometry("geometry")

    # ---------------------------
    # Вороной
    # ---------------------------
    vd = voronoiDiagram4plg(polygons_wells, boundary)

    # ---------------------------
    # Разделение фонда
    # ---------------------------
    gdf_current = gdf_Coordinates[gdf_Coordinates["work_marker"].notna()].copy()
    gdf_project = gdf_Coordinates[gdf_Coordinates["work_marker"].isna()].copy()

    # ---------------------------
    # Зоны эффективной работы
    # ---------------------------
    zone_geoms = []
    zone_colors = []

    gdf_Coordinates.loc[gdf_Coordinates["work_marker"].isna(), "r_eff"] = gdf_Coordinates["r_eff_voronoy"]
    for _, row in gdf_Coordinates.iterrows():
        geom = row.get(LINESTRING)
        r_eff = row.get("r_eff")

        if geom is None or pd.isna(r_eff):
            continue
        if not isinstance(geom, BaseGeometry) or geom.is_empty:
            continue
        if r_eff <= 0:
            continue

        zone = geom.buffer(r_eff / default_size_pixel, resolution=16)
        if zone.is_empty:
            continue

        is_project = pd.isna(row.get("work_marker"))
        zone_geoms.append(zone)
        zone_colors.append(get_zone_color(row.get("work_marker"), is_project=is_project))

    gdf_zones = None
    if zone_geoms:
        gdf_zones = gpd.GeoDataFrame({"color": zone_colors}, geometry=zone_geoms)

    # ---------------------------
    # Рисование
    # ---------------------------
    fig, ax = plt.subplots(figsize=(12, 16))
    fig.patch.set_alpha(0)
    ax.set_facecolor((1, 1, 1, 0))

    # Граница фонда
    boundary.plot(
        ax=ax,
        facecolor="none",
        edgecolor="black",
        linewidth=1.6,
        zorder=1
    )

    # Зоны
    if gdf_zones is not None and not gdf_zones.empty:
        for color in gdf_zones["color"].unique():
            gdf_part = gdf_zones[gdf_zones["color"] == color]
            gdf_part.plot(
                ax=ax,
                facecolor=color,
                edgecolor="none",
                alpha=zone_alpha,
                zorder=2
            )

    # Вороной
    vd.plot(
        ax=ax,
        facecolor="none",
        edgecolor="black",
        linewidth=voronoi_linewidth,
        zorder=3
    )

    vd.boundary.plot(
        ax=ax,
        color="black",
        linewidth=voronoi_linewidth,
        zorder=4
    )

    # Фактические: линии / точки
    if not gdf_current.empty:
        plot_well_geometry(
            gdf=gdf_current,
            geom_col=LINESTRING,
            ax=ax,
            color="black",
            line_width=well_linewidth,
            point_markersize=single_well_markersize,
            zorder=5
        )

        gdf_t1_current = gdf_current[gdf_current[POINT_T1].notna()].copy()
        if not gdf_t1_current.empty:
            gdf_t1_current = filter_non_empty_geometry(gdf_t1_current, POINT_T1)
            if not gdf_t1_current.empty:
                gdf_t1_current.set_geometry(POINT_T1).plot(
                    ax=ax,
                    color="black",
                    markersize=t1_markersize,
                    zorder=6
                )

    # Проектные: линии / точки
    if not gdf_project.empty:
        plot_well_geometry(
            gdf=gdf_project,
            geom_col=LINESTRING,
            ax=ax,
            color="red",
            line_width=well_linewidth,
            point_markersize=single_well_markersize,
            zorder=5
        )

        gdf_t1_project = gdf_project[gdf_project[POINT_T1].notna()].copy()
        if not gdf_t1_project.empty:
            gdf_t1_project = filter_non_empty_geometry(gdf_t1_project, POINT_T1)
            if not gdf_t1_project.empty:
                gdf_t1_project.set_geometry(POINT_T1).plot(
                    ax=ax,
                    color="red",
                    markersize=t1_markersize,
                    zorder=6
                )

    # ---------------------------
    # Подписи
    # ---------------------------
    gdf_labels = get_label_gdf(
        gdf_all=gdf_Coordinates,
        gdf_current=gdf_current,
        point_col=POINT_T1,
        mode=label_mode,
        count=labels_count
    )

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    dx = (xlim[1] - xlim[0]) * 0.010
    dy = (ylim[1] - ylim[0]) * 0.010
    cx = (xlim[0] + xlim[1]) / 2
    cy = (ylim[0] + ylim[1]) / 2

    for _, row in gdf_labels.iterrows():
        pt = row[POINT_T1]
        name = row.get("well_number", "")
        if not is_valid_non_empty_geometry(pt):
            continue

        if label_mode == "all":
            local_dx = dx if pt.x >= cx else -dx
            local_dy = dy if pt.y >= cy else -dy
            ha = "left" if pt.x >= cx else "right"
            va = "bottom" if pt.y >= cy else "top"
        else:
            local_dx = dx
            local_dy = dy
            ha = "left"
            va = "bottom"

        ax.text(
            pt.x + local_dx,
            pt.y + local_dy,
            str(name),
            fontsize=label_fontsize,
            color="black",
            ha=ha,
            va=va,
            zorder=7,
            bbox=dict(
                facecolor=(1, 1, 1, 0.45),
                edgecolor="none",
                pad=1.0
            )
        )

    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout(pad=0)

    plt.savefig(
        filename,
        dpi=300,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0
    )
    plt.close(fig)


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


def save_local_parameters(parameters, save_path):
    """Сохранение файла local_parameters.py"""
    # Удаляем параметры, которые были рассчитаны
    list_keys = ['Bo', 'P_init', 'Pb', 'c_o', 'c_r', 'c_w', 'gor', 'k_h', 'mu_o', 'mu_w', 'rho',
                 'save_directory', 'all_P_wells_init']
    parameters = remove_keys(parameters, list_keys)
    with open(save_path, 'w', encoding='utf-8') as f:
        # Используем pprint для красивого форматирования
        import pprint

        f.write('import datetime\n\n')
        f.write('parameters = ')
        pprint.pprint(parameters, f, indent=4, width=100, depth=None)
    pass


def save_excel_permeability_fact_wells(data_wells_permeability_excel, save_directory):
    data_wells_permeability_excel = data_wells_permeability_excel[data_wells_permeability_excel['permeability_fact']
                                                                  != 0]
    data_wells_permeability_excel.columns = ['номер скважины', 'характер', 'состояние', 'тип', 'последняя дата работы',
                                             'эффективный радиус скважины рассчитанный на основе порового объема, м',
                                             'эффективный радиус через площадь ячейки вороного, м',
                                             'эффективный радиус нормированный на ячейку вороного, м',
                                             'длина ствола скважины T1-T3, м', 'количество стадий ГРП, шт',
                                             'полудлина трещины ГРП, м', 'раскрытие трещины ГРП, мм',
                                             'запускной Qж ТР, т/сут', 'стартовая обводненность ТР (объем), д.ед.',
                                             'запускное забойное давление добывающей скважины, атм',
                                             'стартовое пластовое давление ТР, атм',
                                             'нефтенасыщенная толщина, м', 'пористость, д.ед',
                                             'проницаемость c карты, мД', 'проницаемость обратным счетом через РБ, мД']
    with pd.ExcelWriter(f"{save_directory}/Фактическая_проницаемость_скважин.xlsx") as writer:
        data_wells_permeability_excel.to_excel(writer, index=False)
    pass


def save_excel_inj_wells(data_wells, save_directory):
    data_inj_wells = data_wells[data_wells['work_marker'] == 'inj']
    data_inj_wells = data_inj_wells[["well_number", "well_status", "well_type", "date",
                                     "Winj_rate_TR", "Winj_rate", "Winj",
                                     "time_work_inj", "no_work_time",
                                     "r_eff_not_norm", "r_eff_voronoy", "r_eff",
                                     "Winj_cumsum", "V_useful_injection"]]
    data_inj_wells.columns = ['номер скважины', 'состояние', 'тип', 'последняя дата работы',
                              'приемистость ТР, м3/сут', 'приемистость, м3/сут', 'закачка, м3',
                              'время работы в закачке, часы', 'количество месяцев в простое',
                              'эффективный радиус скважины рассчитанный на основе порового объема, м',
                              'эффективный радиус через площадь ячейки вороного, м',
                              'эффективный радиус нормированный на ячейку вороного, м',
                              'накопленная закачка, м3', 'объем полезной закачки через Куч, м3']
    with pd.ExcelWriter(f"{save_directory}/Полезная_закачка.xlsx") as writer:
        data_inj_wells.to_excel(writer, index=False)
    pass
