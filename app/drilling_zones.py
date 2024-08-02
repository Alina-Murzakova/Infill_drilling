import numpy as np
from loguru import logger
import pandas as pd
from osgeo import gdal, ogr, osr
import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import numpy as np

from map import Map

from local_parameters import paths
from input_output import load_wells_data

"""________БЛОК ДЛЯ УДАЛЕНИЯ_______"""
maps_directory = paths["maps_directory"]
data_well_directory = paths["data_well_directory"]
save_directory = paths["save_directory"]
_, data_wells = load_wells_data(data_well_directory=data_well_directory)
"""________БЛОК ДЛЯ УДАЛЕНИЯ_______"""


def calculate_zones(maps):
    type_map_list = list(map(lambda raster: raster.type_map, maps))

    map_NNT = maps[type_map_list.index("NNT")]
    map_permeability = maps[type_map_list.index("permeability")]
    map_residual_recoverable_reserves = maps[type_map_list.index("residual_recoverable_reserves")]
    map_pressure = maps[type_map_list.index("pressure")]
    map_initial_oil_saturation = maps[type_map_list.index("initial_oil_saturation")]

    map_water_cut = maps[type_map_list.index("water_cut")]
    map_last_rate_oil = maps[type_map_list.index("last_rate_oil")]
    map_init_rate_oil = maps[type_map_list.index("init_rate_oil")]

    logger.info("Расчет карты оценки пласта")
    map_reservoir_score = reservoir_score(map_NNT, map_permeability)

    logger.info("Расчет карты оценки показателей разработки")
    map_potential_score = potential_score(map_residual_recoverable_reserves, map_pressure,
                                          map_last_rate_oil, map_init_rate_oil)

    logger.info("Расчет карты оценки проблем")
    map_risk_score = risk_score(map_water_cut, map_initial_oil_saturation)

    logger.info("Расчет карты индекса возможностей")
    map_opportunity_index = opportunity_index(map_reservoir_score, map_potential_score, map_risk_score)

    # где нет толщин, проницаемости и давления opportunity_index = 0
    map_opportunity_index.data[(map_NNT.data == 0) & (map_permeability.data == 0) & (map_pressure.data == 0)] = 0

    map_opportunity_index.save_img(f"{save_directory}/map_opportunity_index.png", data_wells)
    map_opportunity_index.save_grd_file(f"{save_directory}/opportunity_index.grd")

    logger.info("Кластеризация зон")
    clusterization_zones(map_opportunity_index)

    pass


def reservoir_score(map_NNT, map_permeability) -> Map:
    """
    Оценка пласта
    -------
    Map(type_map=reservoir_score)
    """
    norm_map_NNT = map_NNT.normalize_data()
    norm_map_permeability = map_permeability.normalize_data()

    data_reservoir_score = (norm_map_NNT.data + norm_map_permeability.data) / 2

    map_reservoir_score = Map(data_reservoir_score,
                              norm_map_NNT.geo_transform,
                              norm_map_NNT.projection,
                              "reservoir_score")

    # Пример использования функций
    logger.info("Расчет буфера вокруг действующих скважин")
    union_buffer = active_well_outline()

    logger.info("Создание маски буфера")
    if union_buffer:
        mask = create_mask_from_buffers(map_reservoir_score, union_buffer)
    else:
        mask = np.empty(map_reservoir_score.data.shape)

    logger.info("Бланкование карты согласно буферу")
    blank_value = np.nan
    modified_map = cut_map_by_mask(map_reservoir_score, mask, blank_value)
    modified_map.save_img(f"{save_directory}/cut_map_reservoir_score.png", data_wells)

    return map_reservoir_score


def potential_score(map_residual_recoverable_reserves, map_pressure, map_last_rate_oil, map_init_rate_oil) -> Map:
    """
    Оценка показателей разработки
    -------
    Map(type_map=potential_score)
    """
    P_init = 40 * 9.87  # атм

    map_last_rate_oil.data = np.nan_to_num(map_last_rate_oil.data)
    map_init_rate_oil.data = np.nan_to_num(map_init_rate_oil.data)
    norm_last_rate_oil = map_last_rate_oil.normalize_data()
    norm_init_rate_oil = map_init_rate_oil.normalize_data()

    norm_residual_recoverable_reserves = map_residual_recoverable_reserves.normalize_data()
    map_delta_P = Map(P_init - map_pressure.data, map_pressure.geo_transform, map_pressure.projection,
                      type_map="delta_P").normalize_data()

    data_potential_score = (map_delta_P.data + norm_residual_recoverable_reserves.data
                            + norm_last_rate_oil.data + norm_init_rate_oil.data) / 4
    map_potential_score = Map(data_potential_score, map_pressure.geo_transform, map_pressure.projection,
                              "potential_score")

    map_potential_score.save_img(f"{save_directory}/map_potential_score.png", data_wells)

    return map_potential_score


def risk_score(map_water_cut, map_initial_oil_saturation) -> Map:
    """
    Оценка проблем
    -------
    Map(type_map=risk_score)
    """
    data_last_oil_saturation = 1 - map_water_cut.data / 100
    mask = np.isnan(map_water_cut.data)
    data_oil_saturation = np.where(mask, map_initial_oil_saturation.data, data_last_oil_saturation)

    # # 1) Применение морфологической операции расширения
    # dilated_mask = binary_dilation(mask, iterations=3)
    # # Заполнение NaN значений
    # filled = np.where(dilated_mask, map_initial_oil_saturation.data, data_last_oil_saturation)
    # map_filled = Map(filled, map_water_cut.geo_transform, map_water_cut.projection,
    #                          "filled")
    # map_filled.save_img(f"{save_directory}/map_filled", data_wells)

    # 2) Применение гауссова фильтра для сглаживания
    sigma = 5  # параметр для определения степени сглаживания
    data_oil_saturation = gaussian_filter(data_oil_saturation, sigma=sigma)
    # map_oil_saturation = Map(data_oil_saturation, map_water_cut.geo_transform, map_water_cut.projection,
    #                                                "oil_saturation")
    # map_oil_saturation.save_img(f"{save_directory}/map_oil_saturation", data_wells)

    # data_risk_score = (1 - map_water_cut.data / 100 + map_initial_oil_saturation.data) / 2
    # data_risk_score[np.isnan(data_risk_score)] = 0
    data_risk_score = data_oil_saturation
    map_risk_score = Map(data_risk_score, map_water_cut.geo_transform, map_water_cut.projection,
                         "risk_score")

    map_risk_score.save_img(f"{save_directory}/map_risk_score.png", data_wells)

    return map_risk_score


def opportunity_index(map_reservoir_score, map_potential_score, map_risk_score) -> Map:
    """
    Оценка индекса возможностей
    -------
    Map(type_map=opportunity_index)
    """
    k_reservoir = k_potential = k_risk = 1
    data_opportunity_index = (k_reservoir * map_reservoir_score.data +
                              k_potential * map_potential_score.data +
                              k_risk * map_risk_score.data) / 3
    map_opportunity_index = Map(data_opportunity_index, map_reservoir_score.geo_transform,
                                map_reservoir_score.projection, "opportunity_index")
    return map_opportunity_index


def clusterization_zones(map_opportunity_index, percent_low=60):
    data_opportunity_index = map_opportunity_index.data.copy()

    nan_opportunity_index = np.where(data_opportunity_index == 0, np.nan, data_opportunity_index)
    threshold_value = np.nanpercentile(nan_opportunity_index, percent_low)

    mask = data_opportunity_index > threshold_value
    data_opportunity_index_threshold = np.where(mask, 1, 0)

    map_opportunity_index_threshold = Map(data_opportunity_index_threshold, map_opportunity_index.geo_transform,
                                          map_opportunity_index.projection, "opportunity_index_threshold")
    map_opportunity_index_threshold.save_img(f"{save_directory}/map_opportunity_index_threshold = "
                                             f"{round(threshold_value, 2)}.png", data_wells)
    pass


if __name__ == '__main__':
    from map import read_raster

    map_opportunity_index = read_raster(f"{save_directory}/opportunity_index.grd", no_value=0)
    data_opportunity_index = map_opportunity_index.data.copy()
    percent_low = 60
    nan_opportunity_index = np.where(data_opportunity_index == 0, np.nan, data_opportunity_index)
    threshold_value = np.nanpercentile(nan_opportunity_index, percent_low)

    mask = data_opportunity_index > threshold_value
    # data_opportunity_index_threshold = np.where(mask, 1, 0)
    data_opportunity_index_threshold = np.where(mask, data_opportunity_index, 0)

    # # Использование методов анализа изображений
    # import cv2
    # import matplotlib.pyplot as plt
    #
    # # Применение порогового преобразования
    # _, thresholded_map = cv2.threshold(data_opportunity_index, 0.4, 1, cv2.THRESH_BINARY)
    #
    # # Поиск контуров
    # contours, _ = cv2.findContours((thresholded_map * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #
    # # Рисуем контуры на изображении
    # contour_img = np.zeros_like(data_opportunity_index)
    # cv2.drawContours(contour_img, contours, -1, (255, 255, 255), 2)
    #
    # # Визуализация
    # plt.imshow(contour_img, cmap='gray')
    # plt.colorbar()
    # plt.savefig("D:/Work/Programs_Python/Infill_drilling/output/contour_img", dpi=300)
    # plt.close()

    # data_opportunity_index = data_opportunity_index_threshold
    #
    # кластеризация с использованием DBSCAN
    import matplotlib.pyplot as plt
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler

    # карта индекса возможности бурения
    drilling_index_map = data_opportunity_index_threshold[:int(data_opportunity_index.shape[0]),
                         :int(data_opportunity_index.shape[1])]

    # Преобразование карты в двумерный массив координат и значений
    X, Y = np.meshgrid(np.arange(drilling_index_map.shape[1]), np.arange(drilling_index_map.shape[0]))
    X_vec = X.flatten()
    Y_vec = Y.flatten()
    Z = drilling_index_map.flatten()

    X_filtered, Y_filtered, Z_filtered = [], [], []

    for i, z in enumerate(Z):
        if z > 0:
            X_filtered.append(X_vec[i])
            Y_filtered.append(Y_vec[i])
            Z_filtered.append(z)

    X_filtered = np.array(X_filtered)
    Y_filtered = np.array(Y_filtered)
    Z_filtered = np.array(Z_filtered)

    # Комбинирование координат и значений в один массив
    # data = np.column_stack((X_filtered, Y_filtered, Z_filtered))
    data = np.column_stack((X_filtered, Y_filtered))

    # Нормализация данных
    # scaler = StandardScaler()
    # data_scaled = scaler.fit_transform(data)
    # Пока без нормализации
    data_scaled = data

    # Применение DBSCAN
    """eps: Максимальное расстояние между двумя точками, чтобы одна была соседкой другой.
    min_samples: Минимальное количество точек для образования плотной области."""
    db = DBSCAN(eps=20, min_samples=20)  # Параметры могут варьироваться
    db.fit(data_scaled)

    # Получение меток кластеров
    labels = db.labels_

    # Количество кластеров (без учета шума)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # Визуализация Точки из разных кластеров окрашиваются в разные цвета, точки-шум — в чёрный.
    plt.imshow(drilling_index_map, cmap='viridis', interpolation='nearest')
    plt.colorbar()

    # Нанесение кластеров на карту
    unique_labels = set(labels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Цвет для шума
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)
        xy = data[class_member_mask]

        plt.plot(xy[:, 0], xy[:, 1], markerfacecolor=tuple(col), markersize=5 if k != -1 else 1)
        # markeredgecolor='k', 'o',

    plt.title(f'Карта индекса возможности бурения с кластеризацией DBSCAN (число кластеров: {n_clusters})'
              ,fontsize=5)
    plt.savefig("D:/Work/Programs_Python/Infill_drilling/output/drilling_index_map", dpi=300)
    plt.close()

    pass


def active_well_outline():
    """
    Создание буфера вокруг действующих скважин
    Returns
    -------
    union_buffer
    """
    # Радиус
    NUMBER_MONTHS = 12
    BUFFER_RADIUS = 500

    last_date = data_wells.date.sort_values()
    last_date = last_date.unique()[-1]
    active_date = last_date - relativedelta(months=NUMBER_MONTHS)
    # check_date = last_date + relativedelta(months=1)

    data_wells_with_work = data_wells.copy()
    data_wells_with_work = data_wells_with_work[(data_wells_with_work.Ql_rate > 0) | (data_wells_with_work.Winj_rate > 0)]
    data_active_wells = data_wells_with_work[data_wells_with_work.date > active_date]
    # data_active_wells = data_wells_with_work[data_wells_with_work.date == check_date]
    # data_active_wells = data_wells_with_work[data_wells_with_work.well_number == 1026]

    if data_active_wells.empty:
        print('Нет действующих скважин')
        return

    # Создание геометрии для каждой скважины
    def create_buffer(row):
        if row.T3_x != 0 and row.T3_y != 0:
            # Горизонтальная скважина (линия)
            line = ogr.Geometry(ogr.wkbLineString)
            line.AddPoint(row.T1_x, row.T1_y)
            line.AddPoint(row.T3_x, row.T3_y)
            buffer = line.Buffer(BUFFER_RADIUS)
        else:
            # Вертикальная скважина (точка)
            point = ogr.Geometry(ogr.wkbPoint)
            point.AddPoint(row.T1_x, row.T1_y)
            buffer = point.Buffer(BUFFER_RADIUS)
        return buffer

    # Создаём список буферов
    buffers = [create_buffer(row) for _, row in data_active_wells.iterrows()]

    # Создаем объединенный буфер
    union_buffer = ogr.Geometry(ogr.wkbGeometryCollection)
    for buffer in buffers:
        union_buffer = union_buffer.Union(buffer)

    return union_buffer


def create_mask_from_buffers(map, buffer):
    """
    Функция создания маски из буфера в видe array
    Parameters
    ----------
    map - карта, с которой получаем geo_transform для определения сетки для карты с буфером
    buffer - буфер вокруг действующих скважин

    Returns
    -------
    mask - array
    """
    # Размер карты
    cols = map.data.shape[1]
    rows = map.data.shape[0]

    # Информация о трансформации
    transform = map.geo_transform
    projection = map.projection

    # Создаем временный растровый слой
    driver = gdal.GetDriverByName('MEM')
    dataset = driver.Create('', cols, rows, 1, gdal.GDT_Float32)
    dataset.SetGeoTransform(transform)
    dataset.SetProjection(projection)

    # Создаем векторный слой в памяти для буфера
    mem_driver = ogr.GetDriverByName('Memory')
    mem_ds = mem_driver.CreateDataSource('')  # Создаем временный источник данных
    mem_layer = mem_ds.CreateLayer('', None, ogr.wkbPolygon) # Создаем слой в этом источнике данных
    feature = ogr.Feature(mem_layer.GetLayerDefn()) # Создание нового объекта
    feature.SetGeometry(buffer) # Установка геометрии для объекта
    mem_layer.CreateFeature(feature) # Добавление объекта в слой

    # Заполнение растера значениями по умолчанию
    band = dataset.GetRasterBand(1) # первый слой
    band.SetNoDataValue(0)
    band.Fill(0)  # Установка всех значений пикселей в 0

    # Растеризация буферов
    gdal.RasterizeLayer(dataset, [1], mem_layer, burn_values=[1], options=['ALL_TOUCHED=TRUE'])

    # Чтение данных из растрового бэнда в массив
    mask = band.ReadAsArray()

    # Закрываем временные данные
    dataset = None
    mem_ds = None

    return mask


def cut_map_by_mask(map, mask, blank_value):
    """
    Обрезаем карту согласно маске (буферу вокруг действующих скважин)
    Parameters
    ----------
    map - карта для обрезки
    mask - маска буфера
    blank_value - значение бланка

    Returns
    -------
    modified_map - обрезанная карта
    """

    # Создаем копию данных карты и заменяем значения внутри маски
    modified_data = map.data.copy()
    modified_data[mask == 1] = blank_value

    # Создаем новый объект Map с модифицированными данными
    modified_map = Map(modified_data,
                       map.geo_transform,
                       map.projection,
                       map.type_map)

    return modified_map