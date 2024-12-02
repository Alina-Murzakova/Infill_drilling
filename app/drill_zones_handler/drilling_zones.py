import pandas as pd
import numpy as np
import geopandas as gpd
import time

from loguru import logger
from sklearn.cluster import DBSCAN, KMeans
from shapely.geometry import Point, LineString, Polygon, MultiPolygon

from app.maps_handler.functions import apply_wells_mask
from app.project_wells import ProjectWell
from app.drill_zones_handler.functions import (create_gdf_with_polygons, get_params_nearest_wells,
                                               compute_t1_t3_points, update_and_shift_proj_wells)

@logger.catch
def calculate_drilling_zones(maps, epsilon, min_samples, percent_low, data_wells):
    """
    Выделение зон для уверенного бурения с высоким индексом возможности OI
    Parameters
    ----------
    maps - Cписок всех необходимых карт для расчета (пока используется только OI)
    eps - Максимальное расстояние между двумя точками, чтобы одна была соседкой другой, расстояние по сетке
    min_samples - Минимальное количество точек для образования плотной области, шт
    percent_top - процент лучших точек для кластеризации, %
    dict_properties - ГФХ пласта
    data_wells - фрейм скважин
    Returns
    -------
    dict_zones - словарь с зонами и параметрами кластеризации
     {label: [x, y, z, mean_OI, max_OI], DBSCAN_parameters: [epsilon, min_samples]}
    """
    type_maps_list = list(map(lambda raster: raster.type_map, maps))
    # инициализация всех необходимых карт из списка
    map_opportunity_index = maps[type_maps_list.index("opportunity_index")]

    logger.info("Создание на карте области исключения (маски) на основе действующего фонда")
    modified_map_opportunity_index = apply_wells_mask(map_opportunity_index, data_wells)

    logger.info("Кластеризация зон")
    list_zones, info = clusterization_zones(modified_map_opportunity_index, epsilon, min_samples, percent_low)
    return list_zones, info


class DrillZone:
    def __init__(self, x_coordinates, y_coordinates, z_values, rating):
        self.x_coordinates = x_coordinates
        self.y_coordinates = y_coordinates
        self.opportunity_index_values = z_values
        self.rating = rating

        self.reserves = None
        self.area = None
        self.list_project_wells = []
        self.num_project_wells = None

    def calculate_reserves(self, map_rrr):
        """Расчет ОИЗ перспективной зоны, тыс.т"""
        array_rrr = map_rrr.data[self.y_coordinates, self.x_coordinates]
        self.reserves = np.sum(array_rrr * map_rrr.geo_transform[1] ** 2 / 10000) / 1000

    def calculate_area(self, map_rrr):
        """Расчет площади перспективной зоны, км**2"""
        self.area = len(self.x_coordinates) * map_rrr.geo_transform[1] ** 2 / 1000000

    def get_init_project_wells(self, map_rrr, data_wells, init_profit_cum_oil, init_area_well, default_size_pixel,
                                    buffer_project_wells):
        """Расчет количества проектных скважин в перспективной зоне"""
        self.calculate_reserves(map_rrr)
        self.calculate_area(map_rrr)
        # Начальное количество скважин на основе запасов/площади
        # num_project_wells = int(self.reserves // init_profit_cum_oil) # !!! Возможность выбора или убрать площадь
        num_project_wells = int(self.area // init_area_well)
        points = np.column_stack((self.x_coordinates, self.y_coordinates))

        # Рассматриваем только зоны, куда можно вписать >=1 скважину
        if num_project_wells >= 1:
            logger.info(f"Кластеризация перспективной зоны {self.rating}")
            num_project_wells, labels = self.clusterize_drill_zone(map_rrr, num_project_wells, points,
                                                                   init_profit_cum_oil)

            logger.info("Преобразование точек кластеров в полигоны и создание GeoDataFrame с кластерами")
            gdf_clusters = create_gdf_with_polygons(points, labels)

            logger.info("Получение GeoDataFrame с проектными скважинами из кластеров")
            gdf_project_wells = self.get_project_wells_from_clusters(gdf_clusters, data_wells, default_size_pixel,
                                                                     buffer_project_wells)
            # Количество проектных скважин в перспективной зоне
            self.num_project_wells = len(gdf_project_wells)

            # Преобразуем строки gdf_project_wells в объекты ProjectWell
            self.get_list_project_wells(gdf_project_wells)

    def get_list_project_wells(self, gdf_project_wells):
        """Добавляет объекты ProjectWell в list_project_wells из GeoDataFrame"""
        for _, row in gdf_project_wells.iterrows():
            project_well = ProjectWell(row["well_number"], row["cluster"], row["POINT T1"], row["POINT T3"],
                                       row["LINESTRING"], row["azimuth"], row["well_type"])
            self.list_project_wells.append(project_well)

    def clusterize_drill_zone(self, map_rrr, num_project_wells, points, init_profit_cum_oil):
        """Кластеризация перспективной зоны методом k-means"""
        array_rrr = map_rrr.data[self.y_coordinates, self.x_coordinates]

        # Уменьшаем количество скважин и кластеризуем пока во всех кластерах не будет достаточно запасов
        while True:
            kmeans = KMeans(n_clusters=num_project_wells, max_iter=300, random_state=42)
            kmeans.fit(points)  # , sample_weight=value_OI)
            labels = kmeans.labels_
            cluster_reserves = np.array([np.sum(array_rrr[labels == i]) * map_rrr.geo_transform[1] ** 2 / 10000 / 1000
                                         for i in range(num_project_wells)])
            # Фильтрация кластеров по количеству запасов
            valid_clusters = np.where(cluster_reserves >= init_profit_cum_oil)[0]

            if len(valid_clusters) == num_project_wells or num_project_wells == 1:  # минимальное кол-во кластеров = 1
                break
            else:
                num_project_wells = len(valid_clusters)

        return num_project_wells, labels

    def get_project_wells_from_clusters(self, gdf_clusters, data_wells, default_size_pixel, buffer_project_wells):
        """Получаем GeoDataFrame с начальными координатами проектных скважин"""
        # Подготовка GeoDataFrame с проектными скважинами
        gdf_project = gdf_clusters.copy()
        gdf_project['well_number'] = [f'{self.rating}_{i}' for i in range(1, len(gdf_project) + 1)]
        gdf_project['well_marker'] = 'project'
        gdf_project.rename(columns={'centers': 'POINT T2', 'geometry': 'cluster'}, inplace=True)
        gdf_project.set_geometry("POINT T2", inplace=True)

        # Подготовка GeoDataFrame с фактическими скважинами
        data_wells_work = (data_wells[(data_wells['Qo_cumsum'] > 0) | (data_wells['Winj_cumsum'] > 0)]
                           .reset_index(drop=True))
        df_fact_wells = data_wells_work[["well_number", 'T1_x_conv', 'T1_y_conv', 'T3_x_conv', 'T3_y_conv', 'azimuth',
                                         "length of well T1-3", "well type", "P_well_init_prod", "r_eff"]].copy()
        import warnings
        with warnings.catch_warnings(action='ignore', category=pd.errors.SettingWithCopyWarning):
            df_fact_wells["POINT T1"] = list(map(lambda x, y: Point(x, y),
                                                 df_fact_wells.T1_x_conv, df_fact_wells.T1_y_conv))
            df_fact_wells["POINT T3"] = list(map(lambda x, y: Point(x, y),
                                                 df_fact_wells.T3_x_conv, df_fact_wells.T3_y_conv))
            df_fact_wells["LINESTRING"] = list(map(lambda x, y: LineString([x, y]),
                                                   df_fact_wells["POINT T1"], df_fact_wells["POINT T3"]))
            df_fact_wells["LINESTRING"] = np.where(df_fact_wells["POINT T1"] == df_fact_wells["POINT T3"],
                                                   df_fact_wells["POINT T1"],
                                                   list(map(lambda x, y: LineString([x, y]),
                                                            df_fact_wells["POINT T1"], df_fact_wells["POINT T3"])))

        gdf_fact_wells = gpd.GeoDataFrame(df_fact_wells, geometry="LINESTRING")
        gdf_fact_wells['length_conv'] = gdf_fact_wells.apply(lambda row:
                                                             row['POINT T1'].distance(row['POINT T3']), axis=1)
        # GeoDataFrame с фактическими ГС
        gdf_fact_hor_wells = gdf_fact_wells[gdf_fact_wells["well type"] == "horizontal"].reset_index(drop=True)
        if gdf_fact_hor_wells.empty:
            logger.warning("На объекте нет фактических горизонтальных скважин! \n "
                           "Необходимо задать азимут, длину, Рзаб проектных скважин вручную. ")

        # Находим ближайшие фактические ГС для проектных точек и рассчитываем параметры по окружению
        gdf_project[["azimuth", 'length_conv']] = (gdf_project["POINT T2"].apply(
            lambda center: pd.Series(get_params_nearest_wells(center, gdf_fact_hor_wells, default_size_pixel,
                                                              name_param="траектория"))))

        # Получаем точки T1 и T3 на основе центров кластеров (T2)
        gdf_project[['POINT T1', 'POINT T3']] = gdf_project.apply(compute_t1_t3_points, axis=1,
                                                                  result_type='expand')
        # Получаем линии (горизонтальные стволы)
        gdf_project['LINESTRING'] = gdf_project.apply(lambda row: LineString([row['POINT T1'], row['POINT T3']]),
                                                      axis=1)
        gdf_project.set_geometry("LINESTRING", inplace=True)

        # Выпуклая оболочка для того, чтобы проектная скважина не выходила за её пределы
        gdf_project['convex_hull'] = gdf_project['cluster'].apply(lambda x: x.convex_hull)

        gdf_fact_wells.set_geometry("LINESTRING", inplace=True)
        gdf_project.set_geometry("LINESTRING", inplace=True)

        # Создаем буферы, чтобы искать пересечения зон скважин (для фактических скважин - радиусы дренирования)
        gdf_fact_wells['buffer'] = gdf_fact_wells.geometry.buffer(gdf_fact_wells["r_eff"] / default_size_pixel)
        gdf_project['buffer'] = gdf_project.geometry.buffer(buffer_project_wells)

        # Пересекающийся с проектным и/или фактическим фондом проектный фонд скважин
        intersecting_proj_wells = gdf_project[
            gdf_project.apply(lambda row: (gdf_fact_wells['buffer'].intersects(row['buffer']).any() or
                                           gdf_project[gdf_project.index != row.name]['buffer'].intersects(
                                               row['buffer']).any()), axis=1)]

        # Смещаем, вращаем, сокращаем пересекающихся проектный фонд скважин, при наличии такового
        if not intersecting_proj_wells.empty:
            update_and_shift_proj_wells(gdf_project, gdf_fact_wells, intersecting_proj_wells, default_size_pixel,
                                        buffer_project_wells)
            gdf_project = gdf_project[gdf_project['well_marker'] != 'удалить']

        # gdf_project.loc[gdf_project["length_conv"] < 1, "well_type"] = "vertical"
        # gdf_project.loc[gdf_project["length_conv"] >= 1, "well_type"] = "horizontal"
        gdf_project["POINT T1"] = gdf_project["LINESTRING"].apply(lambda x: Point(x.coords[0]))
        gdf_project["POINT T3"] = gdf_project["LINESTRING"].apply(lambda x: Point(x.coords[-1]))
        gdf_project.loc[gdf_project["POINT T1"] == gdf_project["POINT T3"], "well_type"] = "vertical"
        gdf_project.loc[gdf_project["POINT T1"] != gdf_project["POINT T3"], "well_type"] = "horizontal"

        return gdf_project


def clusterization_zones(map_opportunity_index, epsilon, min_samples, percent_low):
    """Кластеризация зон бурения на основе карты индекса возможности с помощью метода DBSCAN"""
    data_opportunity_index = map_opportunity_index.data

    # Фильтрация карты индекса вероятности по процентилю
    nan_opportunity_index = np.where(data_opportunity_index == 0, np.nan, data_opportunity_index)
    threshold_value = np.nanpercentile(nan_opportunity_index, percent_low)
    data_opportunity_index_threshold = np.where(data_opportunity_index > threshold_value, data_opportunity_index, 0)

    # Массив для кластеризации
    drilling_index_map = data_opportunity_index_threshold.copy()

    # Преобразование карты в двумерный массив координат, значений и вытягивание их в вектор
    X, Y = np.meshgrid(np.arange(drilling_index_map.shape[1]), np.arange(drilling_index_map.shape[0]))
    X, Y, Z = X.flatten(), Y.flatten(), drilling_index_map.flatten()

    # Фильтрация точек для обучающего набора данных - на fit идут точки с не нулевым индексом
    X, Y, Z = np.array(X[Z > 0]), np.array(Y[Z > 0]), np.array(Z[Z > 0])

    # Комбинирование координат и значений в один массив
    dataset = pd.DataFrame(np.column_stack((X, Y, Z)), columns=["X", "Y", "Z"])
    training_dataset = dataset[["X", "Y"]]

    # eps: Максимальное расстояние между двумя точками, чтобы одна была соседкой другой
    # min_samples: Минимальное количество точек для образования плотной области
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples).fit(training_dataset)
    labels = sorted(list(set(dbscan.labels_)))

    # Создание параметра по которому будет произведена сортировка кластеров // среднее значение индекса в кластере
    mean_indexes = list(map(lambda label: np.mean(Z[np.where(dbscan.labels_ == label)]), labels[1:]))
    mean_indexes = pd.DataFrame({"labels": labels[1:], "mean_indexes": mean_indexes})
    mean_indexes = mean_indexes.sort_values(by=['mean_indexes'], ascending=False).reset_index(drop=True)

    # Создание списка объектов DrillZone
    list_zones = []
    for label in labels:
        idx = np.where(dbscan.labels_ == label)
        x_label, y_label, z_label = X[idx], Y[idx], Z[idx]
        position = -1
        if label != -1:
            position = mean_indexes[mean_indexes["labels"] == label].index[0]
        zone = DrillZone(x_label, y_label, z_label, position)
        list_zones.append(zone)
    # Добавление в словарь гиперпараметров кластеризации
    info = {'epsilon': epsilon, "min_samples": min_samples, "n_clusters": len(labels) - 1}
    return list_zones, info


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from app.input_output import load_wells_data, load_geo_phys_properties, get_save_path
    from app.local_parameters import paths, parameters_calculation
    from app.maps_handler.functions import mapping
    from app.well_active_zones import calculate_effective_radius

    data_well_directory = paths["data_well_directory"]
    maps_directory = paths["maps_directory"]
    path_geo_phys_properties = paths["path_geo_phys_properties"]
    epsilon = parameters_calculation["epsilon"]
    min_samples = parameters_calculation["min_samples"]
    percent_low = 100 - parameters_calculation["percent_top"]
    default_size_pixel = parameters_calculation["default_size_pixel"]
    init_profit_cum_oil = parameters_calculation["init_profit_cum_oil"]
    init_area_well = parameters_calculation["init_area_well"]
    buffer_project_wells = parameters_calculation["buffer_project_wells"] / default_size_pixel

    _, data_wells, info = load_wells_data(data_well_directory=data_well_directory)
    name_field, name_object = info["field"], info["object_value"]

    save_directory = get_save_path("Infill_drilling", name_field, name_object.replace('/', '-'))

    dict_geo_phys_properties = load_geo_phys_properties(path_geo_phys_properties, name_field, name_object)

    maps = mapping(maps_directory=maps_directory,
                   data_wells=data_wells,
                   dict_properties=dict_geo_phys_properties,
                   default_size_pixel=default_size_pixel)

    data_wells = calculate_effective_radius(data_wells, dict_geo_phys_properties, maps)
    type_maps_list = list(map(lambda raster: raster.type_map, maps))
    # инициализация всех необходимых карт из списка
    map_opportunity_index = maps[type_maps_list.index("opportunity_index")]

    map_residual_recoverable_reserves = maps[type_maps_list.index("residual_recoverable_reserves")]

    list_zones, info_clusterization_zones = calculate_drilling_zones(maps=maps,
                                                                     epsilon=epsilon,
                                                                     min_samples=min_samples,
                                                                     percent_low=percent_low,
                                                                     data_wells=data_wells)

    list_zones = list_zones[1:]
    list_project_wells = []
    start_time = time.time()

    for i, drill_zone in enumerate(list_zones):
        logger.info(f"Обработка {i + 1} зоны из {len(list_zones)} зон")
        drill_zone.get_init_project_wells(map_residual_recoverable_reserves, data_wells, init_profit_cum_oil,
                                               init_area_well, default_size_pixel, buffer_project_wells)
        list_project_wells.extend(drill_zone.list_project_wells)

    end_time = time.time()
    print(f"Время выполнения: {end_time - start_time} секунд")

    # Создаем GeoDataFrame из списка проектных скважин
    gdf_project_wells = gpd.GeoDataFrame({'geometry': [well.linestring for well in list_project_wells],
                                          'cluster': [well.cluster for well in list_project_wells],
                                          'POINT T1': [well.point_T1 for well in list_project_wells],
                                          'well_number': [well.well_number for well in list_project_wells]})

    gdf_project_wells['buffer'] = gdf_project_wells.geometry.buffer(buffer_project_wells)

    fig, ax = plt.subplots(figsize=(10, 10))

    gdf_project_wells.set_geometry("cluster").plot(ax=ax, cmap="viridis", linewidth=0.5)
    gdf_project_wells.set_geometry("buffer").plot(edgecolor="gray", facecolor="white", alpha=0.5, ax=ax)
    gdf_project_wells.set_geometry("geometry").plot(ax=ax, color='red', linewidth=1, markersize=1)

    # Отображение точек T1 на графике
    gdf_project_wells.set_geometry('POINT T1').plot(color='red', markersize=10, ax=ax)

    # Добавление текста с именами скважин рядом с точками T1
    for point, name in zip(gdf_project_wells['POINT T1'], gdf_project_wells['well_number']):
        if point is not None:
            plt.text(point.x + 2, point.y - 2, name, fontsize=6, ha='left')

    plt.gca().invert_yaxis()
    plt.savefig(f"{save_directory}/init_project_wells.png", dpi=400)

    pass


# if __name__ == '__main__':
#     # Скрипт для перебора гиперпараметров DBSCAN по карте cut_map_opportunity_index.grd
#     import matplotlib.pyplot as plt
#     from input_output import load_wells_data, load_geo_phys_properties
#     from local_parameters import paths, parameters_calculation
#     from input_output import get_save_path
#     from app.well_active_zones import calculate_effective_radius
#     from app.maps_handler.functions import mapping
#
#     data_well_directory = paths["data_well_directory"]
#     maps_directory = paths["maps_directory"]
#     path_geo_phys_properties = paths["path_geo_phys_properties"]
#
#     _, data_wells, info = load_wells_data(data_well_directory=data_well_directory)
#     name_field, name_object = info["field"], info["object_value"]
#
#     save_directory = get_save_path("Infill_drilling", name_field, name_object.replace('/', '-'))
#
#     percent_low = 100 - parameters_calculation["percent_top"]
#
#     dict_geo_phys_properties = load_geo_phys_properties(path_geo_phys_properties, name_field, name_object)
#
#     default_size_pixel = parameters_calculation["default_size_pixel"]
#
#     maps = mapping(maps_directory=maps_directory,
#                    data_wells=data_wells,
#                    dict_properties=dict_geo_phys_properties,
#                    default_size_pixel=default_size_pixel)
#
#     type_maps_list = list(map(lambda raster: raster.type_map, maps))
#     # инициализация всех необходимых карт из списка
#     map_opportunity_index = maps[type_maps_list.index("opportunity_index")]
#     data_wells = calculate_effective_radius(data_wells, dict_geo_phys_properties, maps)
#
#     # Перебор параметров DBSCAN c сеткой графиков 5 х 3
#     # pairs_of_hyperparams = [[5, 20], [5, 50], [5, 100],
#     #                         [10, 20], [10, 50], [10, 100],
#     #                         [15, 20], [15, 50], [15, 100],
#     #                         [20, 20], [20, 50], [20, 100],
#     #                         [30, 20], [30, 50], [30, 100], ]
#     pairs_of_hyperparams = [[4, 10], [4, 30], [4, 50],
#                             [5, 10], [5, 30], [5, 50],
#                             [5, 70], [5, 80], [5, 100],
#                             [7, 10], [7, 30], [7, 50],
#                             [7, 70], [7, 80], [7, 100], ]
#
#     fig = plt.figure()
#     fig.set_size_inches(20, 50)
#
#     for i, s in enumerate(pairs_of_hyperparams):
#         list_zones,_ = calculate_drilling_zones(maps, epsilon=s[0], min_samples=s[1],
#                                               percent_low=percent_low, data_wells=data_wells)
#
#         ax_ = fig.add_subplot(5, 3, i + 1)
#
#         # Определение размера осей
#         x = (map_opportunity_index.geo_transform[0], map_opportunity_index.geo_transform[0] +
#              map_opportunity_index.geo_transform[1] * map_opportunity_index.data.shape[1])
#         y = (map_opportunity_index.geo_transform[3] + map_opportunity_index.geo_transform[5] *
#              map_opportunity_index.data.shape[0], map_opportunity_index.geo_transform[3])
#
#         d_x = x[1] - x[0]
#         d_y = y[1] - y[0]
#
#         element_size = min(d_x, d_y) / 10 ** 5
#         font_size = min(d_x, d_y) / 10 ** 3
#
#         plt.imshow(map_opportunity_index.data, cmap='viridis')
#         cbar = plt.colorbar()
#         cbar.ax.tick_params(labelsize=font_size)
#
#         if data_wells is not None:
#             # Отображение списка скважин на карте
#             column_lim_x = ['T1_x', 'T3_x']
#             for column in column_lim_x:
#                 data_wells = data_wells.loc[((data_wells[column] <= x[1]) & (data_wells[column] >= x[0]))]
#             column_lim_y = ['T1_y', 'T3_y']
#             for column in column_lim_y:
#                 data_wells = data_wells.loc[((data_wells[column] <= y[1]) & (data_wells[column] >= y[0]))]
#
#             # Преобразование координат скважин в пиксельные координаты
#             x_t1, y_t1 = map_opportunity_index.convert_coord((data_wells.T1_x, data_wells.T1_y))
#             x_t3, y_t3 = map_opportunity_index.convert_coord((data_wells.T3_x, data_wells.T3_y))
#
#             # Отображение скважин на карте
#             plt.plot([x_t1, x_t3], [y_t1, y_t3], c='black', linewidth=element_size)
#             plt.scatter(x_t1, y_t1, s=element_size, c='black', marker="o")
#
#             # Отображение имен скважин рядом с точками T1
#             for x, y, name in zip(x_t1, y_t1, data_wells.well_number):
#                 plt.text(x + 3, y - 3, name, fontsize=font_size / 10, ha='left')
#
#         plt.title(map_opportunity_index.type_map, fontsize=font_size * 1.2)
#         plt.tick_params(axis='both', which='major', labelsize=font_size)
#         plt.contour(map_opportunity_index.data, levels=8, colors='black', origin='lower', linewidths=font_size / 100)
#
#         labels = list(map(lambda zone: zone.rating, list_zones))
#         # Выбираем теплую цветовую карту
#         cmap = plt.get_cmap('Wistia', len(set(labels)))
#         # Генерируем список цветов
#         colors = [cmap(i) for i in range(len(set(labels)))]
#
#         if len(labels) == 1 and labels[0] == -1:
#             colors = {0: "gray"}
#         else:
#             colors = dict(zip(labels, colors))
#             colors.update({-1: "gray"})
#
#         for lab, c in zip(labels, colors.values()):
#             zone = list_zones[labels.index(lab)]
#             x_zone = zone.x_coordinates
#             y_zone = zone.y_coordinates
#             mean_index = np.mean(zone.opportunity_index_values)
#             max_index = np.max(zone.opportunity_index_values)
#             plt.scatter(x_zone, y_zone, color=c, alpha=0.6, s=1)
#
#             x_middle = x_zone[int(len(x_zone) / 2)]
#             y_middle = y_zone[int(len(y_zone) / 2)]
#
#             if lab != -1:
#                 # Отображение среднего и максимального индексов рядом с кластерами
#                 plt.text(x_middle, y_middle, f"OI_mean = {np.round(mean_index, 2)}",
#                          fontsize=font_size / 10, ha='left', color="black")
#                 plt.text(x_middle, y_middle, f"OI_max = {np.round(max_index, 2)}",
#                          fontsize=font_size / 10, ha='left', color="black")
#
#         plt.xlim(0, map_opportunity_index.data.shape[1])
#         plt.ylim(0, map_opportunity_index.data.shape[0])
#         plt.xlabel("x")
#         plt.ylabel("y")
#         plt.gca().invert_yaxis()
#
#         n_clusters = len(labels) - 1
#
#         plt.title(f"Epsilon = {s[0]}\n min_samples = {s[1]} \n with {n_clusters} clusters")
#
#     fig.tight_layout()
#     plt.savefig(f"{save_directory}/drilling_index_map", dpi=300)
#     plt.close()
#     pass
