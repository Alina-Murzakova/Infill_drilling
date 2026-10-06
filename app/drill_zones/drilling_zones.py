import copy

import numpy as np
import geopandas as gpd
import pandas as pd

from sklearn.cluster import DBSCAN
from loguru import logger
from shapely import Point, LineString

from app.drill_zones.init_project_wells import get_project_wells_from_clusters
from app.maps_handler.functions import apply_wells_mask, calculate_reservoir_state_maps, calculate_score_maps
from app.maps_handler.maps import Map
from app.project_wells import ProjectWell
from app.drill_zones.init_project_wells import create_gdf_with_polygons, clusterize_drill_zone
from app.reservoir_kr_optimizer import get_reservoir_kr


class DrillZone:
    def __init__(self, x_coordinates, y_coordinates, z_values, rating):
        self.x_coordinates = x_coordinates
        self.y_coordinates = y_coordinates
        self.opportunity_index_values = z_values
        self.rating = rating

        self.reserves = None
        self.area = None
        self.list_project_wells = []
        self.num_project_wells = 0

        # Средние атрибуты с фонда
        self.init_avr_Qo_rate = None
        self.init_avr_Ql_rate = None
        self.init_avr_water_cut = None
        # Суммарная добыча
        self.Qo = None
        self.Ql = None

        # Экономика
        self.cumulative_cash_flow = None  # Накопленный поток наличности
        self.CAPEX = None  # CAPEX
        self.OPEX = None  # OPEX
        self.NPV = None
        self.PVI = None
        self.PI = None

    def calculate_reserves(self, map_rrr):
        """Расчет ОИЗ перспективной зоны, тыс.т"""
        array_rrr = map_rrr.data[self.y_coordinates, self.x_coordinates]
        # 10000 - перевод т/га в т/м2
        # 1000 - перевод т в тыс.т
        self.reserves = np.sum(array_rrr * map_rrr.geo_transform[1] ** 2 / 10000) / 1000
        pass

    def calculate_area(self, map_rrr):
        """Расчет площади перспективной зоны, км**2"""
        self.area = len(self.x_coordinates) * map_rrr.geo_transform[1] ** 2 / 1000000
        pass

    def get_init_project_wells(self, map_rrr, data_wells, gdf_project_wells_all, polygon_OI, default_size_pixel,
                               init_profit_cum_oil, dict_parameters):
        """Расчет количества проектных скважин в перспективной зоне"""
        # Инициализация параметров
        gdf_project_wells = gpd.GeoDataFrame(geometry=[])
        buffer_project_wells = (dict_parameters['well_params']["proj_wells_params"]['buffer_project_wells']
                                / default_size_pixel)
        threshold = dict_parameters['well_params']['proj_wells_params']['threshold']
        k_wells = dict_parameters['well_params']['proj_wells_params']['k']
        min_length = dict_parameters['well_params']['proj_wells_params']['min_length']
        max_length = dict_parameters['well_params']['proj_wells_params']['L']
        # Площадь буфера
        min_area_proj_cluster = (2 * (buffer_project_wells * default_size_pixel) * max_length +
                                 np.pi * (buffer_project_wells * default_size_pixel) ** 2) / 1000000

        # threshold = 2500 - максимальное расстояние для исключения скважины из ближайших скважин
        # при расчете азимутов, пиксели
        self.calculate_reserves(map_rrr)
        self.calculate_area(map_rrr)
        # Начальное количество скважин на основе запасов/площади
        num_project_wells = int(self.reserves // init_profit_cum_oil)

        # Рассматриваем только зоны, куда можно вписать >=1 скважину
        if num_project_wells:
            logger.info(f"Проверка зоны на размещение {num_project_wells} скважин в зоне {self.rating}")
            logger.info(f"Кластеризация перспективной зоны {self.rating}")
            num_project_wells, labels = clusterize_drill_zone((self.x_coordinates, self.y_coordinates),
                                                              map_rrr, num_project_wells, init_profit_cum_oil,
                                                              default_size_pixel, min_area_proj_cluster)

            logger.info("Преобразование точек кластеров в полигоны и создание GeoDataFrame с кластерами")
            gdf_clusters = create_gdf_with_polygons((self.x_coordinates, self.y_coordinates), labels)

            logger.info("Получение GeoDataFrame с проектными скважинами из кластеров")
            gdf_project_wells = get_project_wells_from_clusters(self.rating, polygon_OI, gdf_clusters, data_wells,
                                                                gdf_project_wells_all, default_size_pixel,
                                                                buffer_project_wells,
                                                                threshold, k_wells,
                                                                max_length, min_length)
            # Подготовка GeoDataFrame с фактическими скважинами с добычей
            df_fact_wells = (data_wells[(data_wells['Qo_cumsum'] > 0)].reset_index(drop=True))
            # Преобразуем строки gdf_project_wells в объекты ProjectWell
            for _, row in gdf_project_wells.iterrows():
                project_well = ProjectWell(row["well_number"], row["cluster"], row["POINT_T1_pix"], row["POINT_T3_pix"],
                                           row["LINESTRING_pix"], row["azimuth"], row["well_type"])
                project_well.length_pix = project_well.LINESTRING_pix.length

                # Преобразование координат скважин в географические координаты
                project_well.POINT_T1_geo = Point(map_rrr.convert_coord_to_geo((project_well.POINT_T1_pix.x,
                                                                                project_well.POINT_T1_pix.y)))
                project_well.POINT_T3_geo = Point(map_rrr.convert_coord_to_geo((project_well.POINT_T3_pix.x,
                                                                                project_well.POINT_T3_pix.y)))
                if project_well.POINT_T1_geo == project_well.POINT_T3_geo:
                    project_well.LINESTRING_geo = project_well.POINT_T1_geo
                else:
                    project_well.LINESTRING_geo = LineString([project_well.POINT_T1_geo, project_well.POINT_T3_geo])
                project_well.length_geo = project_well.LINESTRING_geo.length
                # Определение ближайшего окружения и параметров с него
                project_well.get_nearest_wells(df_fact_wells, threshold, k=k_wells)
                project_well.get_params_nearest_wells(dict_parameters)
                self.list_project_wells.append(project_well)
            # Количество проектных скважин в перспективной зоне
            self.num_project_wells = len(self.list_project_wells)
        return gdf_project_wells

    def picture_clustering(self, ax, buffer_project_wells):
        """ ax объект с кластерами скважин зоны"""
        import matplotlib.pyplot as plt
        # Создаем GeoDataFrame из списка проектных скважин
        gdf_project_wells = gpd.GeoDataFrame({'well_number': [well.well_number for well in self.list_project_wells],
                                              'well_type': [well.well_type for well in self.list_project_wells],
                                              'geometry': [well.LINESTRING_pix for well in self.list_project_wells],
                                              'cluster': [well.cluster for well in self.list_project_wells],
                                              'POINT_T1_pix': [well.POINT_T1_pix for well in self.list_project_wells]})

        gdf_project_wells['buffer'] = gdf_project_wells.geometry.buffer(buffer_project_wells)
        gdf_project_wells.set_geometry("cluster").plot(ax=ax, cmap="viridis", linewidth=0.5)
        gdf_project_wells.set_geometry("buffer").plot(edgecolor="gray", facecolor="white", alpha=0.5, ax=ax)
        gdf_project_wells.set_geometry("geometry").plot(ax=ax, color='red', linewidth=1, markersize=1)

        # Отображение точек T1 на графике
        gdf_project_wells.set_geometry('POINT_T1_pix').plot(color='red', markersize=10, ax=ax)

        # Добавление текста с именами скважин рядом с точками T1
        for point, name in zip(gdf_project_wells['POINT_T1_pix'], gdf_project_wells['well_number']):
            if point is not None:
                plt.text(point.x + 2, point.y - 2, name, fontsize=6, ha='left')
        return ax

    def calculate_starting_rates(self, maps, dict_parameters_coefficients):
        """ Расчет запускных дебитов для всех проектных скважин в зоне"""
        for project_well in self.list_project_wells:
            project_well.get_starting_rates(maps, dict_parameters_coefficients)
        pass

    def calculate_production(self, data_decline_rate_stat, period, day_in_month, well_efficiency):
        """ Расчет профиля каждой скважины зоны"""
        for project_well in self.list_project_wells:
            project_well.get_production_profile(data_decline_rate_stat, period, day_in_month, well_efficiency)
        self.get_production_profile()
        pass

    def get_production_profile(self):
        """ Расчет атрибутов профилей зоны"""
        logger.info(f"Расчет атрибутов профилей зоны {self.rating}")
        Qo_rate, Ql_rate, water_cut = [], [], []
        for project_well in self.list_project_wells:
            Qo_rate.append(project_well.init_Qo_rate)
            Ql_rate.append(project_well.init_Ql_rate_V)
            water_cut.append(copy.deepcopy(project_well.water_cut))
            if self.Qo is None:
                self.Qo = project_well.Qo.copy()
            else:
                if self.Qo is not None:
                    self.Qo += project_well.Qo
            if self.Ql is None:
                self.Ql = project_well.Ql.copy()
            else:
                if self.Ql is not None:
                    self.Ql += project_well.Ql
        self.init_avr_Qo_rate = np.mean(Qo_rate)
        self.init_avr_Ql_rate = np.mean(Ql_rate)
        self.init_avr_water_cut = np.mean(water_cut)

        self.Qo = np.sum(self.Qo)
        self.Ql = np.sum(self.Ql)
        pass

    def calculate_economy(self, FEM, well_params, method, dict_NDD=None):
        """ Расчет профиля экономики каждой скважины зоны"""
        for project_well in self.list_project_wells:
            project_well.calculate_economy(FEM, well_params, method, dict_NDD)
        self.get_economic_profile()
        pass

    def get_economic_profile(self):
        """ Расчет экономических атрибутов зоны"""
        logger.info(f"Расчет экономических атрибутов зоны {self.rating}")
        for project_well in self.list_project_wells:
            if self.cumulative_cash_flow is None:
                self.cumulative_cash_flow = project_well.cumulative_cash_flow.copy()
            else:
                if self.cumulative_cash_flow is not None:
                    self.cumulative_cash_flow += project_well.cumulative_cash_flow
            if self.CAPEX is None:
                self.CAPEX = project_well.CAPEX.copy()
            else:
                if self.CAPEX is not None:
                    self.CAPEX += project_well.CAPEX
            if self.OPEX is None:
                self.OPEX = project_well.OPEX.copy()
            else:
                if self.OPEX is not None:
                    self.OPEX += project_well.OPEX
            if self.NPV is None:
                self.NPV = project_well.NPV.copy()
            else:
                if self.NPV is not None:
                    self.NPV += project_well.NPV
            if self.PVI is None:
                self.PVI = project_well.PVI.copy()
            else:
                if self.PVI is not None:
                    self.PVI += project_well.PVI
            self.PI = self.NPV.max() / self.PVI.sum() + 1
        pass


def calculate_drilling_zones(maps, epsilon, min_samples, percent_low, data_wells, dict_properties):
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
    modified_map_opportunity_index = apply_wells_mask(map_opportunity_index, data_wells, dict_properties)

    logger.info("Кластеризация зон")
    list_zones, info = clusterization_zones(modified_map_opportunity_index, epsilon, min_samples, percent_low)
    return list_zones, info


def clusterization_zones(map_opportunity_index, epsilon, min_samples, percent_low):
    """Кластеризация зон бурения на основе карты индекса возможности с помощью метода DBSCAN"""
    data_opportunity_index = map_opportunity_index.data

    # Фильтрация карты индекса вероятности по процентилю
    nan_opportunity_index = np.where(data_opportunity_index == 0, np.nan, data_opportunity_index)
    threshold_value = np.nanpercentile(nan_opportunity_index, percent_low)
    data_opportunity_index_threshold = np.where(data_opportunity_index >= threshold_value, data_opportunity_index, 0)

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
    position = 0
    if -1 in labels:
        position = 1
    mean_indexes = list(map(lambda label: np.mean(Z[np.where(dbscan.labels_ == label)]), labels[position:]))
    mean_indexes = pd.DataFrame({"labels": labels[position:], "mean_indexes": mean_indexes})
    mean_indexes = mean_indexes.sort_values(by=['mean_indexes'], ascending=False).reset_index(drop=True)
    # Добавление не кластеризованных точек
    mean_indexes.loc[len(mean_indexes)] = [-1, 0]

    # Создание списка объектов DrillZone
    list_zones = []
    for label in mean_indexes.labels:
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
    import math
    import itertools
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    from app.input_output.input_wells_data import load_wells_data, prepare_wells_data
    from app.input_output.input_geo_phys_properties import load_geo_phys_properties
    from app.local_parameters import parameters
    from app.well_active_zones import calculate_effective_radius
    from app.maps_handler.functions import mapping
    from app.input_output.output_functions import get_save_path, create_new_dir

    # ---------------------------
    # Параметры перебора
    # ---------------------------
    # min_radius = list(range(50, 501, 50))
    # sensitivity_quality_drill = list(range(0, 101, 10))
    min_radius = [100]
    sensitivity_quality_drill = [20]

    # None — брать percent_top из parameters['drill_zones']
    # или задай диапазон, например:
    # percent_top_range = list(range(10, 51, 10))
    percent_top_range = list(range(10, 101, 10))

    cols = 3
    rows = 4
    plots_per_page = cols * rows

    save_dir = get_save_path("Infill_drilling")
    parameters['paths']['save_directory'] = save_dir
    paths = parameters['paths']
    drill_zones_params = parameters['drill_zones']

    if percent_top_range is None:
        percent_top_values = [drill_zones_params["percent_top"]]
    else:
        percent_top_values = percent_top_range

    combinations = list(itertools.product(
        min_radius,
        sensitivity_quality_drill,
        percent_top_values
    ))

    if parameters['well_params']['proj_wells_params']['buffer_project_wells'] <= 0:
        parameters['well_params']['proj_wells_params']['buffer_project_wells'] = 10

    logger.info("Загрузка скважинных данных")
    data_history, info_object_calculation = load_wells_data(
        data_well_directory=paths["data_well_directory"]
    )

    name_field = info_object_calculation.get("field")
    name_object = info_object_calculation.get("object_value")

    save_directory = paths['save_directory']
    create_new_dir(f"{save_directory}/.debug")

    logger.info(
        f"Загрузка ГФХ по пласту {name_object.replace('/', '-')} "
        f"месторождения {name_field}"
    )
    dict_geo_phys_properties = load_geo_phys_properties(
        paths["path_geo_phys_properties"],
        name_field,
        name_object
    )
    parameters["reservoir_fluid_properties"].update(dict_geo_phys_properties)

    logger.info("Подготовка скважинных данных")
    data_history, data_wells = prepare_wells_data(
        data_history,
        dict_properties=parameters,
        first_months=parameters['well_params']['fact_wells_params']['first_months']
    )

    logger.info("Загрузка и обработка карт")
    maps, data_wells, maps_to_calculate = mapping(
        maps_directory=paths["maps_directory"],
        data_wells=data_wells,
        **{**parameters['maps'], **parameters['switches']}
    )

    default_size_pixel = maps[0].geo_transform[1]

    logger.info("Расчет радиусов дренирования и нагнетания для скважин")
    data_wells = calculate_effective_radius(
        data_wells,
        dict_properties=parameters
    )

    logger.info(
        f"Загрузка ОФП, "
        f"{parameters['switches']['switch_adaptation_relative_permeability']}"
    )
    if parameters['switches']['switch_adaptation_relative_permeability']:
        parameters = get_reservoir_kr(
            data_history.copy(),
            data_wells.copy(),
            parameters
        )

    logger.info(f"Расчет карт текущего состояния: {any(maps_to_calculate.values())}")
    if any(maps_to_calculate.values()):
        maps = calculate_reservoir_state_maps(
            data_wells,
            maps,
            parameters,
            default_size_pixel,
            maps_to_calculate,
            maps_directory=paths["maps_directory"]
        )

    logger.info("Расчет оценочных карт")
    maps = maps + calculate_score_maps(
        maps=maps,
        dict_properties=parameters['reservoir_fluid_properties']
    )

    type_map_list = [raster.type_map for raster in maps]
    map_opportunity_index = maps[type_map_list.index("opportunity_index")]

    # ---------------------------
    # Границы карты
    # ---------------------------
    x_geo = (
        map_opportunity_index.geo_transform[0],
        map_opportunity_index.geo_transform[0]
        + map_opportunity_index.geo_transform[1]
        * map_opportunity_index.data.shape[1]
    )

    y_geo = (
        map_opportunity_index.geo_transform[3]
        + map_opportunity_index.geo_transform[5]
        * map_opportunity_index.data.shape[0],
        map_opportunity_index.geo_transform[3]
    )

    # ---------------------------
    # Общая цветовая шкала
    # ---------------------------
    vmin = np.nanmin(map_opportunity_index.data)
    vmax = np.nanmax(map_opportunity_index.data)

    pages = [
        combinations[i:i + plots_per_page]
        for i in range(0, len(combinations), plots_per_page)
    ]

    logger.info(
        f"Всего комбинаций: {len(combinations)}. "
        f"Картинок будет сохранено: {len(pages)}"
    )

    progress = tqdm(
        total=len(combinations),
        desc="Расчет карт зон бурения",
        unit="map"
    )

    try:
        for page_idx, page_combinations in enumerate(pages, start=1):
            fig, axes = plt.subplots(
                rows,
                cols,
                figsize=(5.5 * cols, 4.8 * rows),
                sharex=True,
                sharey=True,
                constrained_layout=True
            )

            axes = np.atleast_1d(axes).ravel()
            im_for_cbar = None

            for i, (radius, quality, percent_top) in enumerate(page_combinations):
                ax = axes[i]

                percent_low = 100 - percent_top
                epsilon = radius / default_size_pixel

                # DBSCAN требует eps > 0
                if epsilon <= 0:
                    ax.set_title(
                        f"R={radius}, Q={quality}, top={percent_top}, skipped",
                        fontsize=8
                    )
                    ax.axis('off')
                    progress.update(1)
                    continue

                # DBSCAN требует min_samples >= 1
                min_samples = max(
                    1,
                    int(quality / 100 * epsilon**2 * math.pi)
                )

                list_zones, _ = calculate_drilling_zones(
                    maps=maps,
                    epsilon=epsilon,
                    min_samples=min_samples,
                    percent_low=percent_low,
                    data_wells=data_wells,
                    dict_properties=parameters
                )

                im = ax.imshow(
                    map_opportunity_index.data,
                    cmap='viridis',
                    vmin=vmin,
                    vmax=vmax,
                    origin='upper'
                )

                if im_for_cbar is None:
                    im_for_cbar = im

                ax.contour(
                    map_opportunity_index.data,
                    levels=8,
                    colors='black',
                    origin='lower',
                    linewidths=0.25
                )

                # ---------------------------
                # Скважины
                # ---------------------------
                if data_wells is not None and not data_wells.empty:
                    wells_plot = data_wells.copy()

                    for column in ['T1_x_geo', 'T3_x_geo']:
                        wells_plot = wells_plot.loc[
                            (wells_plot[column] >= x_geo[0])
                            & (wells_plot[column] <= x_geo[1])
                        ]

                    for column in ['T1_y_geo', 'T3_y_geo']:
                        wells_plot = wells_plot.loc[
                            (wells_plot[column] >= y_geo[0])
                            & (wells_plot[column] <= y_geo[1])
                        ]

                    if not wells_plot.empty:
                        x_t1, y_t1 = map_opportunity_index.convert_coord_to_pix(
                            (wells_plot.T1_x_geo, wells_plot.T1_y_geo)
                        )
                        x_t3, y_t3 = map_opportunity_index.convert_coord_to_pix(
                            (wells_plot.T3_x_geo, wells_plot.T3_y_geo)
                        )

                        ax.plot(
                            [x_t1, x_t3],
                            [y_t1, y_t3],
                            c='black',
                            linewidth=0.4
                        )
                        ax.scatter(
                            x_t1,
                            y_t1,
                            s=5,
                            c='black',
                            marker='o'
                        )

                        if len(wells_plot) <= 20:
                            for xx, yy, name in zip(
                                x_t1,
                                y_t1,
                                wells_plot.well_number
                            ):
                                ax.text(
                                    xx + 2,
                                    yy - 2,
                                    name,
                                    fontsize=5,
                                    ha='left',
                                    color='black'
                                )

                # ---------------------------
                # Кластеры
                # ---------------------------
                labels = [zone.rating for zone in list_zones]
                unique_labels = sorted(set(labels))

                if unique_labels == [-1]:
                    color_map = {-1: "gray"}
                else:
                    cluster_labels = [
                        lab for lab in unique_labels
                        if lab != -1
                    ]

                    cmap_clusters = plt.get_cmap(
                        'Wistia',
                        max(len(cluster_labels), 1)
                    )

                    color_map = {
                        lab: cmap_clusters(j)
                        for j, lab in enumerate(cluster_labels)
                    }
                    color_map[-1] = "gray"

                for zone in list_zones:
                    ax.scatter(
                        zone.x_coordinates,
                        zone.y_coordinates,
                        color=color_map[zone.rating],
                        alpha=0.65,
                        s=1,
                        rasterized=True
                    )

                n_clusters = len({
                    lab for lab in labels
                    if lab != -1
                })

                ax.set_title(
                    f"R={radius}, Q={quality}, top={percent_top}, clusters={n_clusters}",
                    fontsize=8
                )

                ax.set_xlim(0, map_opportunity_index.data.shape[1])
                ax.set_ylim(0, map_opportunity_index.data.shape[0])
                ax.invert_yaxis()

                if i % cols == 0:
                    ax.set_ylabel("y", fontsize=8)
                else:
                    ax.set_ylabel("")

                if i >= len(page_combinations) - cols:
                    ax.set_xlabel("x", fontsize=8)
                else:
                    ax.set_xlabel("")

                ax.tick_params(
                    axis='both',
                    which='major',
                    labelsize=7
                )

                progress.update(1)

            # Скрываем пустые ячейки на последней странице
            for j in range(len(page_combinations), len(axes)):
                axes[j].axis('off')

            if im_for_cbar is not None:
                cbar = fig.colorbar(
                    im_for_cbar,
                    ax=axes[:len(page_combinations)],
                    location='right',
                    shrink=0.98,
                    pad=0.02
                )
                cbar.set_label("opportunity_index", fontsize=10)
                cbar.ax.tick_params(labelsize=8)

            fig.suptitle(
                f"Сравнение параметров кластеризации зон бурения — "
                f"страница {page_idx}/{len(pages)}",
                fontsize=14
            )

            out_path = (
                f"{save_directory}/.debug/"
                f"drilling_index_map_grid_page_{page_idx:03d}.png"
            )

            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

            logger.info(f"Сетка карт сохранена: {out_path}")

    finally:
        progress.close()