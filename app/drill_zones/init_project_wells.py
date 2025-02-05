import numpy as np
import pandas as pd
import geopandas as gpd

from loguru import logger
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely.affinity import translate, rotate
from shapely.ops import unary_union, nearest_points
from sklearn.cluster import KMeans


def get_project_wells_from_clusters(name_cluster, gdf_clusters, data_wells, default_size_pixel, buffer_project_wells,
                                    threshold, k_wells, max_length, min_length):
    """Получаем GeoDataFrame с начальными координатами проектных скважин"""
    # threshold - максимальное расстояние для исключения скважины из ближайших скважин, пиксели

    # Подготовка GeoDataFrame с проектными скважинами
    gdf_project = gdf_clusters.copy()
    gdf_project['well_number'] = [f'{name_cluster}_{i}' for i in range(1, len(gdf_project) + 1)]
    gdf_project['well_marker'] = 'project'
    gdf_project.rename(columns={'centers': 'POINT_T2_pix', 'geometry': 'cluster'}, inplace=True)
    gdf_project.set_geometry("POINT_T2_pix", inplace=True)

    # Подготовка GeoDataFrame с фактическими скважинами
    gdf_fact_wells = gpd.GeoDataFrame(data_wells, geometry="LINESTRING_pix")

    # GeoDataFrame с фактическими ГС
    gdf_fact_hor_wells = gdf_fact_wells[gdf_fact_wells["well_type"] == "horizontal"].reset_index(drop=True)
    if gdf_fact_hor_wells.empty:
        logger.warning("На объекте нет фактических горизонтальных скважин! \n "
                       "Необходимо задать азимут, длину, Рзаб проектных скважин вручную.")

    # Находим ближайшие фактические ГС для проектных точек и рассчитываем параметры по окружению
    gdf_project["azimuth"] = (gdf_project["POINT_T2_pix"].apply(
        lambda center: pd.Series(get_well_path_nearest_wells(center, gdf_fact_hor_wells,
                                                             threshold / default_size_pixel, k=k_wells))))
    gdf_project['length_pix'] = max_length / default_size_pixel
    # Получаем точки T1 и T3 на основе центров кластеров (T2)
    gdf_project[['POINT_T1_pix', 'POINT_T3_pix']] = gdf_project.apply(compute_t1_t3_points, axis=1,
                                                                      result_type='expand')
    # Получаем линии (горизонтальные стволы)
    gdf_project['LINESTRING_pix'] = gdf_project.apply(lambda row: LineString([row['POINT_T1_pix'],
                                                                              row['POINT_T3_pix']]), axis=1)
    # Выпуклая оболочка для того, чтобы проектная скважина не выходила за её пределы
    gdf_project['convex_hull'] = gdf_project['cluster'].apply(lambda x: x.convex_hull)
    # Создаем буферы, чтобы искать пересечения зон скважин (для фактических скважин - радиусы дренирования)
    gdf_fact_wells.set_geometry("LINESTRING_pix", inplace=True)
    gdf_project.set_geometry("LINESTRING_pix", inplace=True)
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
                                    buffer_project_wells, min_length)
        gdf_project = gdf_project[gdf_project['well_marker'] != 'удалить']

    gdf_project["POINT_T1_pix"] = gdf_project["LINESTRING_pix"].apply(lambda x: Point(x.coords[0]))
    gdf_project["POINT_T3_pix"] = gdf_project["LINESTRING_pix"].apply(lambda x: Point(x.coords[-1]))
    if not gdf_project.empty:
        gdf_project.loc[gdf_project["POINT_T1_pix"] == gdf_project["POINT_T3_pix"], "well_type"] = "vertical"
        gdf_project.loc[gdf_project["POINT_T1_pix"] != gdf_project["POINT_T3_pix"], "well_type"] = "horizontal"
    return gdf_project


def clusterize_drill_zone(tuple_points, map_rrr, num_project_wells, init_profit_cum_oil, default_size_pixel,
                          min_area_proj_cluster):
    """Кластеризация зоны методом k-means, критерий: в каждой зоне должно быть достаточно запасов"""
    points = np.column_stack(tuple_points)
    array_rrr = map_rrr.data[tuple_points[1], tuple_points[0]]
    weights = array_rrr / np.max(array_rrr)
    area_pixel = default_size_pixel ** 2

    # Уменьшаем количество скважин и кластеризуем пока во всех кластерах не будет достаточно запасов
    while True:
        kmeans = KMeans(n_clusters=num_project_wells, max_iter=300, random_state=42)
        kmeans.fit(points, sample_weight=weights)
        labels = kmeans.labels_
        cluster_reserves = np.array([np.sum(array_rrr[labels == i]) * map_rrr.geo_transform[1] ** 2 / 10000 / 1000
                                     for i in range(num_project_wells)])
        cluster_area = np.array([np.sum(labels == i) * area_pixel / 1000000 for i in range(num_project_wells)])
        # Фильтрация кластеров только по количеству запасов
        if num_project_wells == 1:
            valid_clusters = np.where(cluster_reserves >= init_profit_cum_oil)[0]
        else:
            valid_clusters = np.where((cluster_reserves >= init_profit_cum_oil) &
                                      (cluster_area >= min_area_proj_cluster))[0]
        if len(valid_clusters) == num_project_wells or num_project_wells == 1:  # минимальное кол-во кластеров = 1
            break
        else:
            num_project_wells -= 1

    return num_project_wells, labels


def create_gdf_with_polygons(tuple_points, labels):
    """Преобразование точек кластеров в полигоны и создание gdf с кластерами для расположения в них проектных скважин"""
    points = np.column_stack(tuple_points)
    # Преобразуем точки кластера в GeoDataFrame
    gdf_points = gpd.GeoDataFrame(geometry=[Point(p) for p in points])
    # Создаем буферы вокруг точек (1 - минимальный буфер, необходимый затем для объединения)
    gdf_buffers = gdf_points.buffer(1)
    # Объединяем буферы по меткам кластеров в полигоны
    cluster_polygons = gdf_buffers.groupby(labels).apply(lambda point: point.union_all())
    # GeoDataFrame с полученными кластерами/полигонами
    gdf_clusters = gpd.GeoDataFrame(geometry=cluster_polygons).reset_index(drop=True)

    # Добавляем столбец для геометрий MultiPolygon, чтобы перераспределить несвязанные кластеры
    # zones - зоны мультиполигона
    gdf_clusters['zones'] = gdf_clusters['geometry'].apply(
        lambda geom: list(geom.geoms) if isinstance(geom, MultiPolygon) else [])

    import copy
    gdf_clusters['start_zones'] = gdf_clusters['zones'].apply(lambda x: copy.deepcopy(x))

    # Выделяем зоны в несвязанных кластерах (MultiPolygon) и перераспределяем их
    gdf_clusters.apply(lambda row: convert_multipolygon(row, gdf_clusters), axis=1)

    # Объединяем перераспределенные зоны и преобразуем в Polygon
    gdf_clusters['geometry'] = gdf_clusters.apply(
        lambda row: unary_union(row['zones']) if row['zones'] else row['geometry'], axis=1)
    gdf_clusters = gdf_clusters.drop(columns=['zones', 'start_zones'])
    gdf_clusters['centers'] = gdf_clusters['geometry'].centroid  # Центры новых полигонов

    return gdf_clusters


def convert_multipolygon(row, gdf_clusters, coef_area=0.4):
    """
    Функция преобразования MultiPolygon в Polygon - перераспределения несвязанных кластеров (MultiPolygon)
    Parameters
    ----------
    row - строка gdf_clusters
    gdf_clusters - GeoDataFrame с полигонами/мультиполигонами под проектные скважины
    coef_area - доля площади, необходимая для выделения самостоятельного полигона из мультиполигона

    Returns
    -------
    """
    geometry = row['geometry']
    if not isinstance(geometry, MultiPolygon):
        return  # Если не MultiPolygon, ничего не делаем
    zones = row['start_zones']  # исходные зоны мультиполигона
    cluster_area = geometry.area  # Общая площадь мультиполигона
    # Перебираем зоны в мультиполигоне
    for zone in zones:
        if zone.area / cluster_area >= coef_area:
            continue  # Зона достаточно крупная, остается самостоятельным кластером

        # Находим пересекающиеся полигоны, исключая текущий
        intersecting_polygons = gdf_clusters[(gdf_clusters.index != row.name) &
                                             (gdf_clusters.geometry.intersects(zone))].copy()
        if intersecting_polygons.empty:
            gdf_clusters.at[row.name, 'zones'].remove(zone)
            continue  # Нет пересекающихся кластеров

        # Добавляем расстояние до текущей зоны для сортировки
        intersecting_polygons['distance'] = intersecting_polygons.geometry.apply(zone.distance)
        intersecting_polygons = intersecting_polygons.sort_values(by='distance')

        zone_added = False  # Флаг для отслеживания успешного добавления зоны
        # Проверяем каждый пересекающийся кластер
        for idx, poly_row in intersecting_polygons.iterrows():
            surround_poly = poly_row['geometry']

            # Обработка зоны в случае MultiPolygon
            if isinstance(surround_poly, MultiPolygon):
                # Оставляет только зоны на расстоянии 0 (касаются или пересекаются)
                intersecting_polygon_zones = [nearest_zone for nearest_zone in surround_poly.geoms
                                              if zone.distance(nearest_zone) == 0]
            # Обработка зоны в случае Polygon
            elif isinstance(surround_poly, Polygon):
                intersecting_polygon_zones = [surround_poly]
            else:
                intersecting_polygon_zones = []
                logger.error(f"Кластер скважины не Polygon/MultiPolygon")

            # Проверяем пересечение и размер зон
            for nearest_zone in intersecting_polygon_zones:
                # Проверяем, пересекается ли nearest_zone с zone и проверяем площадь зоны
                if (nearest_zone.intersects(zone) and nearest_zone.overlaps(zone) and
                        nearest_zone.area / surround_poly.area >= coef_area):
                    # Добавляем зону в gdf, если она еще не добавлена
                    if nearest_zone not in gdf_clusters.at[idx, 'zones']:
                        gdf_clusters.at[idx, 'zones'].append(nearest_zone)
                    gdf_clusters.at[idx, 'zones'].append(zone)

                    # Удаляем зону из текущего кластера
                    gdf_clusters.at[row.name, 'zones'].remove(zone)
                    zone_added = True
                    break
                else:
                    continue  # Если не нашли подходящую зону, продолжаем

            if zone_added:
                break  # Выход из цикла по intersecting_polygons

        if not zone_added:
            if zone.area / cluster_area < coef_area:
                # Удаляем зону из текущего кластера
                gdf_clusters.at[row.name, 'zones'].remove(zone)
                logger.info(f"Зона {row.name} удалена. Её площадь составляет {zone.area / cluster_area} "
                            f"от всего кластера")
            else:
                logger.warning(f"Несвязанный кластер {row.name} (MultiPolygon) не перераспределился!")


def get_well_path_nearest_wells(center, gdf_fact_wells, threshold, k=5):
    """
    Получаем параметры k - ближайших фактических горизонтальных скважин
    Parameters
    ----------
    center - центр кластера скважины (POINT_T2_pix)
    gdf_fact_wells - gdf с фактическими скважинами
    threshold = 2500 / default_size_pixel - максимальное расстояние для исключения скважины
                                                из ближайших скважин, пиксели
    Returns
    -------
    avg_azimuth - средний азимут по окружению
    avg_length - средняя длина по окружению
    avg_param - средний параметр name_param по окружению
    """
    # Вычисляем расстояния до всех ГС
    distances = center.distance(gdf_fact_wells["LINESTRING_pix"])
    sorted_distances = distances.nsmallest(k)
    nearest_hor_wells_index = [sorted_distances.index[0]]  # Ближайшая скважина всегда включается

    for i in range(1, len(sorted_distances)):
        # Проверяем разницу с первой добавленной ближайшей скважиной
        if sorted_distances.iloc[i] - sorted_distances.iloc[0] < threshold:
            nearest_hor_wells_index.append(sorted_distances.index[i])

    # Извлечение строк GeoDataFrame по индексам
    gdf_nearest_wells = gdf_fact_wells.loc[nearest_hor_wells_index]

    # Расчет средний параметров по выбранному окружению
    avg_azimuth = gdf_nearest_wells.loc[(abs(gdf_nearest_wells['azimuth']
                                             - gdf_nearest_wells['azimuth'].iloc[0]) <= 90)]['azimuth'].mean()
    return avg_azimuth


def compute_t1_t3_points(row):
    """Расчет точек Т1 и Т3 на основе Т2 (центра) и азимута"""
    azimuth_rad = np.radians(row['azimuth'])
    half_length = row['length_pix'] / 2

    # Вычисление координат начальной и конечной точек
    x1 = row['POINT_T2_pix'].x - half_length * np.sin(azimuth_rad)
    y1 = row['POINT_T2_pix'].y + half_length * np.cos(azimuth_rad)
    x2 = row['POINT_T2_pix'].x + half_length * np.sin(azimuth_rad)
    y2 = row['POINT_T2_pix'].y - half_length * np.cos(azimuth_rad)

    return Point(x1, y1), Point(x2, y2)


def update_and_shift_proj_wells(gdf_project, gdf_fact_wells, intersecting_proj_wells, default_size_pixel,
                                buffer_project_wells, min_length, step_length=0.1):
    """
    Функция поиска начального расположения проектных скважин без пересечений
    gdf_project
    gdf_fact_wells
    intersecting_proj_wells - пересекающиеся проектные скважины
    min_length=300 - минимальная длина ГС, иначе ННС
    step_length=0.1 - шаг уменьшения длины ГС для вписывания в зону кластеризации
    """
    # Перебираем пересекающиеся с другим фондом проектные скважины
    for proj_idx, proj_row in intersecting_proj_wells.iterrows():
        other_proj_wells = gdf_project.drop(proj_idx)

        intersected_fact_wells = gdf_fact_wells[gdf_fact_wells['buffer'].intersects(proj_row['buffer'])]
        intersected_proj_wells = other_proj_wells[other_proj_wells['buffer'].intersects(proj_row['buffer'])]
        intersected_wells = pd.concat([intersected_fact_wells, intersected_proj_wells])

        # Находим ближайшую пересеченную скважину
        if not intersected_wells.empty:
            intersected_wells['distance'] = intersected_wells['LINESTRING_pix'].apply(
                lambda x: proj_row['LINESTRING_pix'].distance(x))
            min_distance = intersected_wells['distance'].min()
            nearest_intersected_well = intersected_wells[intersected_wells['distance'] == min_distance].copy().iloc[0]
        else:
            continue

        original_position = proj_row['LINESTRING_pix']
        new_position = original_position
        is_updated = False  # Флаг для управления завершением цикла
        original_length = original_position.length  # Сохраняем первоначальную длину
        new_length = original_length

        while not is_updated:
            # Двигаем проектную скважину
            new_position, is_updated = shift_project_well(proj_row, nearest_intersected_well, gdf_fact_wells,
                                                          other_proj_wells, buffer_project_wells)
            # Вращаем проектную скважину, если решение не нашлось и она не ННС
            if not is_updated and proj_row['LINESTRING_pix'].length > 0:
                new_position, is_updated = rotate_project_well(original_position, gdf_fact_wells,
                                                               other_proj_wells, buffer_project_wells)
            # Удаление проектной скважины из-за невозможности расположить её
            if not is_updated and new_length == 0:
                gdf_project.at[proj_idx, 'well_marker'] = 'удалить'
                # gdf_project.drop(proj_idx, inplace=True)
                break

            # Уменьшаем длины проектной скважины
            new_length -= step_length * original_length  # Уменьшаем длину на 10%
            if new_length < min_length / default_size_pixel:
                new_length = 0

            if not is_updated:
                new_position = short_well(proj_row['LINESTRING_pix'], new_length)
                proj_row['LINESTRING_pix'] = new_position
                proj_row['length_pix'] = new_position.length
                proj_row['buffer'] = new_position.buffer(buffer_project_wells)

        # Обновление данных
        if is_updated:
            gdf_project.at[proj_idx, 'buffer'] = new_position.buffer(buffer_project_wells)
            gdf_project.at[proj_idx, 'LINESTRING_pix'] = new_position
            gdf_project.at[proj_idx, 'length_pix'] = new_position.length


def shift_project_well(proj_row, nearest_intersected_well, gdf_fact_wells, other_proj_wells,
                       buffer_project_wells, part_line_in=0.7, step_shift=1, max_attempts=100):
    """
    Сдвиг проектной скважины в указанном направлении
    proj_row - строка gdf с проектными скважинами
    nearest_intersected_well - пересекающиеся скважины
    gdf_fact_wells - gdf с фактическими скважинами
    other_proj_wells - gdf с другими проектными скважинами
    part_line_in - необходимая доля проектной скважины в кластере
    step_shift - шаг сдвига, 1 пиксель
    """
    # Словарь с направлениями и сдвигами
    dict_directions = {'up': (0, -step_shift),
                       'down': (0, step_shift),
                       'right': (step_shift, 0),
                       'left': (-step_shift, 0)}

    original_position = proj_row['LINESTRING_pix']

    # Если исходная проектная скважина - не LINESTRING (не ГС)
    if not original_position.is_valid:
        original_position = Point(original_position.coords[0])

    # Определяем порядок направлений сдвига (вверх/вниз/влево/вправо)
    directions = determine_shift_direction(original_position, nearest_intersected_well['LINESTRING_pix'])
    Flag = False
    attempt_count = 0  # Счетчик попыток сдвига

    # Перебираем направления
    for direction in directions:
        new_position = original_position
        x_off, y_off = dict_directions[direction]

        # Сдвигаем в данном направлении пока есть пересечения с фактическим и/или проектным фондом
        while (gdf_fact_wells['buffer'].intersects(new_position.buffer(buffer_project_wells)).any() or
               other_proj_wells['buffer'].intersects(new_position.buffer(buffer_project_wells)).any()):
            # Сдвигаем в зависимости от направления
            new_position = translate(new_position, xoff=x_off, yoff=y_off)
            attempt_count += 1  # Увеличиваем счетчик попыток

            # Если количество попыток превышает лимит, выходим
            if attempt_count > max_attempts:
                logger.warning("Превышен лимит попыток сдвига, скважина не может быть сдвинута.")
                return original_position, False  # Сдвиг не удался

            # Смена Flag в случае, если изначально центр и больше 0,7*ГС вне кластера
            if proj_row['cluster'].intersection(new_position).length >= (part_line_in * proj_row['length_pix']):
                if proj_row['cluster'].contains(new_position.centroid):
                    Flag = True

            # Меняем направление, если скважина начала выходить из кластера на >30%
            if (not proj_row['cluster'].intersection(new_position).length >=
                    (part_line_in * proj_row['length_pix']) and Flag or
                    not proj_row['cluster'].contains(new_position.centroid) and Flag or
                    not proj_row['convex_hull'].intersection(new_position).length >=
                        (part_line_in * proj_row['length_pix']) or
                    not proj_row['convex_hull'].contains(new_position.centroid)):
                Flag = False
                break

        # Проверка, что сдвинутая скважина не пересекает другие буферы
        if not (gdf_fact_wells['buffer'].intersects(new_position.buffer(buffer_project_wells)).any() or
                other_proj_wells['buffer'].intersects(new_position.buffer(buffer_project_wells)).any()):
            return new_position, True  # Сдвиг удался

    return original_position, False  # Сдвиг не удался


def rotate_project_well(original_position, gdf_fact_wells, other_proj_wells, buffer_project_wells,
                        max_angle=46, step_angle=5):
    """
    Поворот проектной скважины в интервале +45/-45 градусов
    original_position - исходное положение проектной скважины
    gdf_fact_wells - gdf с фактическими скважинами
    other_proj_wells - gdf с другими проектными скважинами
    max_angle - максимальный угол вращения
    step_angle - шаг вращения
    """
    angle = 0
    change_direction = False

    # Вращаем скважины до предельного угла
    while (max_angle >= 0 and angle < max_angle) or (max_angle <= 0 and angle > max_angle):
        # Поворачиваем скважины вокруг её центра
        new_position = rotate(original_position, angle, origin=original_position.centroid)

        # Если нет пересечений, то решение найдено
        if not (gdf_fact_wells['buffer'].intersects(new_position.buffer(buffer_project_wells)).any() or
                other_proj_wells['buffer'].intersects(new_position.buffer(buffer_project_wells)).any()):
            return new_position, True

        angle += step_angle

        if abs(angle) > abs(max_angle):
            if change_direction:
                break
            change_direction = True
            angle = 0
            max_angle = -max_angle
            step_angle = -step_angle

    return original_position, False  # если не удалось повернуть


def get_orientation(line):
    """Определение ориентации скважины (вдоль x или y)"""
    x_diff = abs(line.coords[-1][0] - line.coords[0][0])
    y_diff = abs(line.coords[-1][1] - line.coords[0][1])

    if x_diff >= y_diff:  # что если x_diff == y_diff?!
        return 'x'
    return 'y'


def determine_shift_direction(proj_well, intersect_well):
    """Определение направления для сдвига (вверх/вниз/влево/вправо)
    в зависимости от расположения ближайшей пересекающей скважины"""
    proj_point, intersect_point = nearest_points(proj_well, intersect_well)
    directions = []
    # Если ориентация вдоль x
    if get_orientation(proj_well) == 'x':
        if intersect_point.y < proj_point.y:
            directions = ['down', 'up']
        else:
            directions = ['up', 'down']

        if intersect_point.x < proj_point.x:
            directions += ['right', 'left']
        else:
            directions += ['left', 'right']

    # Если ориентация вдоль y
    elif get_orientation(proj_well) == 'y':
        if intersect_point.x < proj_point.x:
            directions = ['right', 'left']
        else:
            directions = ['left', 'right']

        if intersect_point.y < proj_point.y:
            directions += ['down', 'up']
        else:
            directions += ['up', 'down']

    return directions


def short_well(trajectory, new_length):
    """Функция для сокращения линии до определенной длины"""
    # Проверка длины линии
    if trajectory.length < new_length:
        return Point(trajectory.centroid)  # ННС, если ГС уже короче 300 м

    half_length = new_length / 2
    start_point = trajectory.interpolate(trajectory.length / 2 + half_length, normalized=False)
    end_point = trajectory.interpolate(trajectory.length / 2 - half_length, normalized=False)

    # Проверка, нужно ли поменять местами start_point и end_point
    if start_point.distance(Point(trajectory.coords[0])) > end_point.distance(Point(trajectory.coords[0])):
        start_point, end_point = end_point, start_point

    # Создаем новую траекторию с учетом сокращения относительно центра
    new_trajectory = LineString([start_point, end_point])
    return new_trajectory
