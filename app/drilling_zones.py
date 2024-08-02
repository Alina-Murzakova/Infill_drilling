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
from config import MER_columns_name

"""________БЛОК ДЛЯ УДАЛЕНИЯ_______"""
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
    map_potential_score = potential_score(map_residual_recoverable_reserves, map_pressure)

    logger.info("Расчет карты оценки проблем")
    map_risk_score = risk_score(map_water_cut, map_initial_oil_saturation)

    logger.info("Расчет карты индекса возможностей")
    map_opportunity_index = opportunity_index(map_reservoir_score, map_potential_score, map_risk_score)

    pass


def reservoir_score(map_NNT, map_permeability) -> Map:
    """
    Оценка пласта
    Parameters
    ----------
    map_NNT - карта ННТ
    map_permeability  - карта проницаемости

    Returns
    -------
    Map(type_map=reservoir_score)
    """

    norm_map_NNT = map_NNT.normalize_data()
    norm_map_permeability = map_permeability.normalize_data()

    norm_map_NNT.save_img(f"{save_directory}/norm_map_NNT.png", data_wells)
    norm_map_permeability.save_img(f"{save_directory}/norm_map_permeability.png", data_wells)

    # data_reservoir_score = (norm_map_NNT.data * norm_map_permeability.data) ** 1/2
    data_reservoir_score = (norm_map_NNT.data + norm_map_permeability.data) / 2

    map_reservoir_score = Map(data_reservoir_score,
                              norm_map_NNT.geo_transform,
                              norm_map_NNT.projection,
                              "reservoir_score")
    map_reservoir_score = map_reservoir_score.normalize_data()
    map_reservoir_score.save_img(f"{save_directory}/norm_map_reservoir_score.png", data_wells)

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


def potential_score(map_residual_recoverable_reserves, map_pressure) -> Map:

    P_init = 40 * 9.87  # атм

    map_delta_P = Map(P_init - map_pressure.data, map_pressure.geo_transform, map_pressure.projection,
                      type_map="delta_P")
    norm_residual_recoverable_reserves = map_residual_recoverable_reserves.normalize_data()
    pass


def risk_score(map_water_cut, map_initial_oil_saturation) -> Map:
    pass


def opportunity_index(map_reservoir_score, map_potential_score, map_risk_score) -> Map:
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