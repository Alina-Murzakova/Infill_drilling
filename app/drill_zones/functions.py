import numpy as np
import pandas as pd
import geopandas as gpd

from loguru import logger
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely.affinity import translate, rotate
from shapely.ops import unary_union, nearest_points


def create_gdf_with_polygons(points, labels):
    """Преобразование точек кластеров в полигоны и создание gdf с кластерами"""

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
    row - строка
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
                intersecting_polygon_zones = sorted(surround_poly.geoms, key=lambda p: zone.distance(p))
            # Обработка зоны в случае Polygon
            elif isinstance(surround_poly, Polygon):
                intersecting_polygon_zones = [surround_poly]
            else:
                intersecting_polygon_zones = []
                logger.error(f"Кластер скважины не Polygon/MultiPolygon")

            # Проверяем пересечение и размер зон
            for nearest_zone in intersecting_polygon_zones:
                # Проверяем, пересекается ли nearest_zone с zone и проверяем площадь зоны
                if nearest_zone.intersects(zone) and nearest_zone.area / surround_poly.area >= coef_area:
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
            logger.warning("Несвязанный кластер (MultiPolygon) не перераспределился!")


def get_nearest_wells(geometry, gdf_fact_wells, default_size_pixel, k=5, threshold=2500):
    """
    Получаем параметры ближайших фактических горизонтальных скважин
    Parameters
    ----------
    geometry - центр кластера скважины (POINT T2) или сама проектная скважина
    gdf_fact_wells - gdf с фактическими скважинами
    k=5 - количество ближайших фактических скважин
    threshold=2500 - максимальное расстояние для исключения скважины из ближайших скважин, м
    Returns
    -------
    gdf_nearest_wells - gdf с ближайшими фактическими скважинами
    """
    threshold = threshold / default_size_pixel
    # Вычисляем расстояния до всех ГС
    distances = geometry.distance(gdf_fact_wells["LINESTRING"])
    sorted_distances = distances.nsmallest(k)
    nearest_hor_wells_index = [sorted_distances.index[0]]  # Ближайшая скважина всегда включается

    for i in range(1, len(sorted_distances)):
        # Проверяем разницу с первой добавленной ближайшей скважиной
        if sorted_distances.iloc[i] - sorted_distances.iloc[0] < threshold:
            nearest_hor_wells_index.append(sorted_distances.index[i])

    # Извлечение строк GeoDataFrame по индексам
    gdf_nearest_wells = gdf_fact_wells.loc[nearest_hor_wells_index]

    return gdf_nearest_wells


def get_params_nearest_wells(center, gdf_fact_wells, default_size_pixel, name_param):
    """
    Получаем параметры ближайших фактических горизонтальных скважин
    Parameters
    ----------
    center - центр кластера скважины (POINT T2)
    gdf_fact_wells - gdf с фактическими скважинами
    Returns
    -------
    avg_azimuth - средний азимут по окружению
    avg_length - средняя длина по окружению
    avg_param - средний параметр name_param по окружению
    """
    gdf_nearest_hor_wells = get_nearest_wells(center, gdf_fact_wells, default_size_pixel)

    # Расчет средний параметров по выбранному окружению в зависимости от типа параметры
    if name_param == "траектория":
        avg_azimuth = gdf_nearest_hor_wells.loc[(abs(gdf_nearest_hor_wells['azimuth'] - gdf_nearest_hor_wells['azimuth']
                                                 .iloc[0]) <= 90)]['azimuth'].mean()
        avg_length = np.mean(gdf_nearest_hor_wells['length_conv'])
        return avg_azimuth, avg_length

    else:
        avg_param = np.mean(gdf_nearest_hor_wells[name_param])
        return avg_param


def compute_t1_t3_points(row):
    """Расчет точек Т1 и Т3 на основе Т2 (центра) и азимута"""
    azimuth_rad = np.radians(row['azimuth'])
    half_length = row['length_conv'] / 2

    # Вычисление координат начальной и конечной точек
    x1 = row['POINT T2'].x - half_length * np.sin(azimuth_rad)
    y1 = row['POINT T2'].y + half_length * np.cos(azimuth_rad)
    x2 = row['POINT T2'].x + half_length * np.sin(azimuth_rad)
    y2 = row['POINT T2'].y - half_length * np.cos(azimuth_rad)

    return Point(x1, y1), Point(x2, y2)


def update_and_shift_proj_wells(gdf_project, gdf_fact_wells, intersecting_proj_wells, default_size_pixel,
                                buffer_project_wells, min_length=300):
    """
    Функция поиска начального расположения проектных скважин без пересечений
    gdf_project
    gdf_fact_wells
    intersecting_proj_wells - пересекающиеся проектные скважины
    min_length=300 - минимальная длина ГС, иначе ННС
    """
    # Перебираем пересекающиеся с другим фондом проектные скважины
    for proj_idx, proj_row in intersecting_proj_wells.iterrows():
        other_proj_wells = gdf_project.drop(proj_idx)

        intersected_fact_wells = gdf_fact_wells[gdf_fact_wells['buffer'].intersects(proj_row['buffer'])]
        intersected_proj_wells = other_proj_wells[other_proj_wells['buffer'].intersects(proj_row['buffer'])]
        intersected_wells = pd.concat([intersected_fact_wells, intersected_proj_wells])

        # Находим ближайшую пересеченную скважину
        if not intersected_wells.empty:
            nearest_intersected_well = intersected_wells.loc[intersected_wells['LINESTRING'].apply(
                lambda x: proj_row['LINESTRING'].distance(x)).idxmin()].copy()
        else:
            continue

        original_position = proj_row['LINESTRING']
        is_updated = False  # Флаг для управления завершением цикла
        original_length = original_position.length  # Сохраняем первоначальную длину
        new_length = original_length

        while not is_updated:
            # Двигаем проектную скважину
            new_position, is_updated = shift_project_well(proj_row, nearest_intersected_well, gdf_fact_wells,
                                                          other_proj_wells, buffer_project_wells)
            # Вращаем проектную скважину, если решение не нашлось и она не ННС
            if not is_updated and proj_row['LINESTRING'].length > 0:
                new_position, is_updated = rotate_project_well(original_position, gdf_fact_wells,
                                                               other_proj_wells, buffer_project_wells)
            # Удаление проектной скважины из-за невозможности расположить её
            if not is_updated and new_length == 0:
                gdf_project.at[proj_idx, 'well_type'] = 'удалить'
                # gdf_project.drop(proj_idx, inplace=True)
                break

            # Уменьшаем длины проектной скважины
            if new_length >= min_length / default_size_pixel:
                new_length -= 0.1 * original_length # Уменьшаем длину на 10%
            else:
                new_length = 0

            if not is_updated:
                new_position = short_well(proj_row['LINESTRING'], new_length)
                proj_row['LINESTRING'] = new_position
                proj_row['length_conv'] = new_position.length
                proj_row['buffer'] = new_position.buffer(buffer_project_wells)

        # Обновление данных
        if is_updated:
            gdf_project.at[proj_idx, 'buffer'] = new_position.buffer(buffer_project_wells)
            gdf_project.at[proj_idx, 'LINESTRING'] = new_position


def shift_project_well(proj_row, nearest_intersected_well, gdf_fact_wells, other_proj_wells,
                       buffer_project_wells, part_line_in=0.7, step_shift=1, max_attempts = 100):
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

    original_position = proj_row['LINESTRING']

    # Если исходная проектная скважина - не LINESTRING (не ГС)
    if not original_position.is_valid:
        original_position = Point(original_position.coords[0])

    # Определяем порядок направлений сдвига (вверх/вниз/влево/вправо)
    directions = determine_shift_direction(original_position, nearest_intersected_well['LINESTRING'])
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
                logger.warning("НПревышен лимит попыток сдвига, скважина не может быть сдвинута.")
                return original_position, False  # Сдвиг не удался

            # Смена Flag в случае, если изначально центр и больше 0,7*ГС вне кластера
            if proj_row['cluster'].intersection(new_position).length >= (part_line_in * proj_row['length_conv']):
                if proj_row['cluster'].contains(new_position.centroid):
                    Flag = True

            # Меняем направление, если скважина начала выходить из кластера на >30%
            if (not proj_row['cluster'].intersection(new_position).length >=
                    (part_line_in * proj_row['length_conv']) and Flag or
                    not proj_row['cluster'].contains(new_position.centroid) and Flag or
                    not proj_row['convex_hull'].intersection(new_position).length >=
                        (part_line_in * proj_row['length_conv']) or
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
    proj_centroid, intersect_centroid = nearest_points(proj_well, intersect_well)
    directions = []
    # Если ориентация вдоль x
    if get_orientation(proj_well) == 'x':
        if intersect_centroid.y < proj_centroid.y:
            directions = ['down', 'up']
        else:
            directions = ['up', 'down']

        if intersect_centroid.x < proj_centroid.x:
            directions += ['right', 'left']
        else:
            directions += ['left', 'right']

    # Если ориентация вдоль y
    elif get_orientation(proj_well) == 'y':
        if intersect_centroid.x < proj_centroid.x:
            directions = ['right', 'left']
        else:
            directions = ['left', 'right']

        if intersect_centroid.y < proj_centroid.y:
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
