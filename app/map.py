import os
import numpy as np
import pandas as pd
from osgeo import gdal
from loguru import logger
from scipy.interpolate import RBFInterpolator
from scipy.spatial import KDTree

from config import list_names_map


@logger.catch
def mapping(maps_directory, data_wells, default_size_pixel):
    """Загрузка и подготовка всех необходимых карт"""
    logger.info(f"path: {maps_directory}")
    content = os.listdir(path=maps_directory)
    if content:
        logger.info(f"maps: {len(content)}")
    else:
        raise logger.critical("no maps!")

    logger.info(f"Загрузка карт из папки: {maps_directory}")
    maps, default_size_pixel = maps_load(maps_directory, data_wells, default_size_pixel)

    logger.info(f"Преобразование карт к единому размеру и сетке")
    dst_geo_transform, dst_projection, shape = final_resolution(maps, default_size_pixel)

    res_maps = list(map(lambda raster: raster.resize(dst_geo_transform, dst_projection, shape), maps))
    return res_maps


def maps_load(maps_directory, data_wells, default_size_pixel):
    maps = []

    logger.info(f"Загрузка карты ННТ")
    try:
        maps.append(read_raster(f'{maps_directory}/NNT.grd', no_value=0))
    except FileNotFoundError:
        logger.error(f"в папке отсутствует файл с картой ННТ: NNT.grd")

    logger.info(f"Загрузка карты проницаемости")
    try:
        maps.append(read_raster(f'{maps_directory}/permeability.grd', no_value=0))
    except FileNotFoundError:
        logger.error(f"в папке отсутствует файл с картой проницаемости: permeability.grd")

    logger.info(f"Загрузка карты ОИЗ")
    try:
        maps.append(read_raster(f'{maps_directory}/residual_recoverable_reserves.grd', no_value=0))
    except FileNotFoundError:
        logger.error(f"в папке отсутствует файл с картой ОИЗ: residual_recoverable_reserves.grd")

    logger.info(f"Загрузка карты изобар")
    try:
        maps.append(read_raster(f'{maps_directory}/pressure.grd', no_value=0))
    except FileNotFoundError:
        logger.error(f"в папке отсутствует файл с картой изобар: pressure.grd")

    logger.info(f"Загрузка карты начальной нефтенасыщенности")
    try:
        maps.append(read_raster(f'{maps_directory}/initial_oil_saturation.grd', no_value=0))
    except FileNotFoundError:
        logger.error(f"в папке отсутствует файл с картой изобар: initial_oil_saturation.grd")

    # Вычисление минимального размера пикселя, если он None при загрузке
    if not default_size_pixel:
        dst_geo_transform, _, _ = final_resolution(maps, default_size_pixel)
        default_size_pixel = dst_geo_transform[1]

    #  Загрузка карт из "МЭР"
    logger.info(f"Загрузка карты обводненности на основе выгрузки МЭР")
    maps.append(read_array(data_wells, name_column_map="water_cut", type_map="water_cut",
                           default_size=default_size_pixel))

    logger.info(f"Загрузка карты последних дебитов нефти на основе выгрузки МЭР")
    maps.append(read_array(data_wells, name_column_map="Qo_rate", type_map="last_rate_oil",
                           default_size=default_size_pixel))

    logger.info(f"Загрузка карты стартовых дебитов нефти на основе выгрузки МЭР")
    maps.append(read_array(data_wells, name_column_map="init_Qo_rate", type_map="init_rate_oil",
                           default_size=default_size_pixel))

    return maps, default_size_pixel


def read_raster(file_path, no_value=0):
    """
    Создание объекта класса MAP через загрузку из файла .grd
    Parameters
    ----------
    file_path - путь к файлу
    no_value - значение для заполнения пустот на карте

    Returns
    -------
    Map(type_map)
    """
    dataset = gdal.Open(file_path)
    ndv = dataset.GetRasterBand(1).GetNoDataValue()
    band = dataset.GetRasterBand(1)
    data = band.ReadAsArray()
    if ndv is not None:
        data = np.where(data >= ndv, no_value, data)
    geo_transform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()
    name_file = os.path.basename(file_path).replace(".grd", "")
    if name_file not in list_names_map:
        raise logger.critical(f"Неверный тип карты! {name_file}\n"
                              f"Переименуйте карту в соответствии со списком допустимых названий: {list_names_map}")
    return Map(data, geo_transform, projection, type_map=name_file)


def read_array(data_wells, name_column_map, type_map, default_size, accounting_GS=True, radius=1000, expand=0.3):
    """
    Создание объекта класса MAP из DataFrame
    Parameters
    ----------
    data_wells - DataFrame
    name_column_map - наименование колонок, по значениям котрой строится карта
    type_map - тип карты
    radius - радиус экстраполяции за крайние скважины
    expand - Расширение границ

    Returns
    -------
    Map(type_map)
    """
    # Очистка фрейма от скважин не в работе
    import warnings
    if type_map == "water_cut":
        data_wells_with_work = data_wells[(data_wells.Ql_rate > 0) | (data_wells.Winj_rate > 0)]
        with warnings.catch_warnings(action='ignore', category=pd.errors.SettingWithCopyWarning):
            data_wells_with_work.water_cut = np.where(data_wells_with_work.Winj_rate > 0,
                                                      100, data_wells_with_work.water_cut)
        # !!! приоритизация точек по последней дате в работе
    elif type_map == "last_rate_oil" or type_map == "init_rate_oil":
        data_wells_with_work = data_wells[(data_wells.Ql_rate > 0)]

        with warnings.catch_warnings(action='ignore', category=pd.errors.SettingWithCopyWarning):
            data_wells_with_work[name_column_map] = np.where(data_wells_with_work['well type'] == 'horizontal',
                                                      data_wells_with_work[name_column_map]/data_wells_with_work["length of well T1-3"] ,
                                                    data_wells_with_work[name_column_map])
        data_wells_with_work[name_column_map] = data_wells_with_work.groupby('well type')[name_column_map].transform(
            lambda x: (x - x.min()) / (x.max() - x.min()))
        # NNS_mean_init_Ql = data_wells[(data_wells['init_Ql_rate'] != 0) &
        #                               (data_wells['well type'] == 'vertical')]['init_Ql_rate'].mean()
        # GS_mean_init_Ql = data_wells[(data_wells['init_Ql_rate'] != 0) &
        #                              (data_wells['well type'] == 'horizontal')]['init_Ql_rate'].mean()
        # coefficient_GS_to_NNS = round(NNS_mean_init_Ql / GS_mean_init_Ql, 1)
        # logger.info(f'Коэффициент соотношения дебитов жидкости ННС и ГС: {coefficient_GS_to_NNS}')
        #
        # with warnings.catch_warnings(action='ignore', category=pd.errors.SettingWithCopyWarning):
        #     data_wells_with_work[name_column_map] = np.where(data_wells_with_work['well type'] == 'horizontal',
        #                                                  data_wells_with_work[name_column_map] * coefficient_GS_to_NNS,
        #                                                  data_wells_with_work[name_column_map])
    else:
        if type_map not in list_names_map:
            raise logger.critical(f"Неверный тип карты! {type_map}")

    if accounting_GS:
        # Формирование списка точек для ствола каждой скважины
        coordinates = data_wells_with_work.apply(trajectory_break_points, default_size=default_size, axis=1)
        # Объединяем координаты и с исходным df
        data_wells_with_work = pd.concat([data_wells_with_work, coordinates], axis=1)

        x, y, values = [], [], []
        for _, row in data_wells_with_work.iterrows():
            x.extend(row['x_coords'])
            y.extend(row['y_coords'])
            values.extend([row[name_column_map]] * len(row['x_coords']))

        x, y, values = np.array(x), np.array(y), np.array(values)
        well_coord = np.column_stack((x, y))

    else:
        # Построение карт по значениям T1
        x, y = np.array(data_wells_with_work.T1_x), np.array(data_wells_with_work.T1_y)
        well_coord = np.column_stack((x, y))
        # Выделяем значения для карты
        values = np.array(data_wells_with_work[name_column_map])

    # Определение границ интерполяции
    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)

    # Расширение границ
    x_min -= (x_max - x_min) * expand
    x_max += (x_max - x_min) * expand
    y_min -= (y_max - y_min) * expand
    y_max += (y_max - y_min) * expand

    # Создание сетки для интерполяции
    grid_x, grid_y = np.mgrid[x_min:x_max:default_size, y_max:y_min:-default_size]
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    # Создание KDTree для координат скважин
    tree = KDTree(well_coord)

    # Поиск точек сетки, которые находятся в пределах заданного радиуса от любой скважины
    points_in_radius = tree.query_ball_point(grid_points, r=radius)

    # Создаем маску для этих точек
    points_mask = np.array([len(indices) > 0 for indices in points_in_radius])
    grid_points_mask = grid_points[points_mask]

    # Использование RBFInterpolator
    rbfi = RBFInterpolator(well_coord, values, kernel='linear')  # сглаживание smoothing=0.5

    # Предсказание значений на сетке
    valid_grid_z = rbfi(grid_points_mask)
    grid_z = np.full(grid_x.shape, np.nan)
    grid_z.ravel()[points_mask] = valid_grid_z

    # Применяем ограничения на минимальное и максимальное значения карты
    grid_z = np.clip(grid_z, 0, 100).T

    # Определение геотрансформации
    geo_transform = [x_min, (x_max - x_min) / grid_z.shape[1], 0, y_max, 0, -((y_max - y_min) / grid_z.shape[0])]

    return Map(grid_z, geo_transform, projection='', type_map=type_map)


class Map:
    def __init__(self, data, geo_transform, projection, type_map):
        self.data = data
        self.geo_transform = geo_transform
        self.projection = projection
        self.type_map = type_map

    def normalize_data(self):
        # Возвращение новой карты
        new_data = (self.data - np.min(self.data)) / (np.max(self.data) - np.min(self.data))
        return Map(new_data, self.geo_transform, self.projection, self.type_map)

    def resize(self, dst_geo_transform, dst_projection, dst_shape):
        # Создание исходного GDAL Dataset в памяти
        src_driver = gdal.GetDriverByName('MEM')
        src_ds = src_driver.Create('', self.data.shape[1], self.data.shape[0], 1, gdal.GDT_Float32)
        src_ds.SetGeoTransform(self.geo_transform)
        src_ds.SetProjection(self.projection)
        src_ds.GetRasterBand(1).WriteArray(self.data)

        # Создание результирующего GDAL Dataset в памяти
        dst_driver = gdal.GetDriverByName('MEM')
        dst_ds = dst_driver.Create('', dst_shape[1], dst_shape[0], 1, gdal.GDT_Float32)
        dst_ds.SetGeoTransform(dst_geo_transform)
        dst_ds.SetProjection(dst_projection)

        # Выполнение перепроекции
        gdal.ReprojectImage(src_ds, dst_ds, self.projection, dst_projection, gdal.GRA_Bilinear)

        # Возвращение результирующего массива
        data = dst_ds.GetRasterBand(1).ReadAsArray()

        # Возвращение новой карты
        return Map(data, dst_geo_transform, dst_projection, self.type_map)

    def show(self):
        import matplotlib.pyplot as plt
        plt.imshow(self.data, cmap='viridis')
        plt.colorbar()
        plt.show()

    def save_grd_file(self, filename):
        filename_copy = filename.replace(".grd", "") + "_copy.grd"
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(filename_copy, self.data.shape[1], self.data.shape[0], 1, gdal.GDT_Float32)
        dataset.SetGeoTransform(self.geo_transform)
        dataset.SetProjection(self.projection)
        dataset.GetRasterBand(1).WriteArray(self.data)
        dataset.FlushCache()
        src_dataset = gdal.Open(filename_copy, gdal.GA_ReadOnly)
        # driver = gdal.GetDriverByName('XYZ') можно использовать для формата .dat
        driver = gdal.GetDriverByName('GSAG')
        driver.CreateCopy(filename, src_dataset, 0, options=['TFW=NO'])
        # Удаляем временный файл
        src_dataset = None
        dataset = None
        os.remove(filename_copy)
        os.remove(filename.replace(".grd", "") + ".grd.aux.xml")

    def convert_coord(self, array):
        # Преобразование координат массива в пиксельные координаты в соответствии с geo_transform карты
        x, y = array
        conv_x = np.where(x != 0, ((x - self.geo_transform[0]) / self.geo_transform[1]).astype(int), np.nan)
        conv_y = np.where(y != 0, ((self.geo_transform[3] - y) / abs(self.geo_transform[5])).astype(int), np.nan)
        return conv_x, conv_y

    def save_img(self, filename, data_wells=None, dict_zones=None):
        import matplotlib.pyplot as plt

        # Определение размера осей
        x = (self.geo_transform[0], self.geo_transform[0] + self.geo_transform[1] * self.data.shape[1])
        y = (self.geo_transform[3] + self.geo_transform[5] * self.data.shape[0], self.geo_transform[3])

        d_x = x[1] - x[0]
        d_y = y[1] - y[0]

        plt.figure(figsize=(d_x / 2540, d_y / 2540))
        element_size = min(d_x, d_y) / 10 ** 5
        font_size = min(d_x, d_y) / 10 ** 3

        plt.imshow(self.data, cmap='viridis', origin="upper")
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=font_size)

        # Отображение списка скважин на карте
        if data_wells is not None:
            # Преобразование координат скважин в пиксельные координаты
            x_t1, y_t1 = self.convert_coord((data_wells.T1_x, data_wells.T1_y))
            x_t3, y_t3 = self.convert_coord((data_wells.T3_x, data_wells.T3_y))

            # Отображение скважин на карте
            plt.plot([x_t1, x_t3], [y_t1, y_t3], c='black', linewidth=element_size)
            plt.scatter(x_t1, y_t1, s=element_size, c='black', marker="o")

            # Отображение имен скважин рядом с точками T1
            for x, y, name in zip(x_t1, y_t1, data_wells.well_number):
                plt.text(x + 3, y - 3, name, fontsize=font_size / 10, ha='left')

        # Отображение зон кластеризации на карте
        title = ""
        if dict_zones is not None:
            labels = list(set(dict_zones.keys()) - {"DBSCAN_parameters"})
            # Выбираем теплую цветовую карту
            cmap = plt.get_cmap('Wistia', len(set(labels)))
            # Генерируем список цветов
            colors = [cmap(i) for i in range(len(set(labels)))]

            if len(labels) == 1 and labels[0] == -1:
                colors = {0: "gray"}
            else:
                colors = dict(zip(labels, colors))
                colors.update({-1: "gray"})

            for i, c in zip(labels, colors.values()):
                x_zone = dict_zones[i][0]
                y_zone = dict_zones[i][1]
                mean_index = dict_zones[i][3]
                max_index = dict_zones[i][4]
                plt.scatter(x_zone, y_zone, color=c, alpha=0.6, s=1)

                if i != -1:
                    # Отображение среднего и максимального индексов рядом с кластерами
                    plt.text(x_zone[int(len(x_zone) / 2)], y_zone[int(len(y_zone) / 2)],
                             f"OI_mean = {np.round(mean_index, 3)}",
                             fontsize=font_size / 10, ha='left', color="black")
                    plt.text(x_zone[int(len(x_zone) / 2)], y_zone[int(len(y_zone) / 2)] - 10,
                             f"OI_max = {np.round(max_index, 3)}",
                             fontsize=font_size / 10, ha='left', color="black")

            n_clusters = len(labels) - 1
            title = (f"Epsilon = {dict_zones['DBSCAN_parameters'][0]}\n "
                     f"min_samples = {dict_zones['DBSCAN_parameters'][1]} \n "
                     f"with {n_clusters} clusters")

        plt.title(f"{self.type_map}\n {title}", fontsize=font_size * 1.2)
        plt.tick_params(axis='both', which='major', labelsize=font_size)
        plt.contour(self.data, levels=8, colors='black', origin='lower', linewidths=font_size / 100)
        plt.savefig(filename, dpi=300)
        plt.close()


def final_resolution(list_rasters, pixel_sizes):
    """Поиск наименьшего размера карты, который станет целевым для расчета

    list_rasters - список карт
    pixel_sizes - шаг сетки int/None (по-умолчанию 50/поиск наименьшего шага среди сеток)

    return: geo_transform, projection, shape
    """
    data_list = list(map(lambda raster: raster.data, list_rasters))
    geo_transform_list = list(map(lambda raster: raster.geo_transform, list_rasters))
    projection_list = list(map(lambda raster: raster.projection, list_rasters))

    min_x = max(list(map(lambda geo_transform: geo_transform[0], geo_transform_list)))
    max_x = min(list(map(lambda geo_transform, data: geo_transform[0] + geo_transform[1] * data.shape[1],
                         geo_transform_list, data_list)))
    min_y = max(list(map(lambda geo_transform, data: geo_transform[3] + geo_transform[5] * data.shape[0],
                         geo_transform_list, data_list)))
    max_y = min(list(map(lambda geo_transform: geo_transform[3], geo_transform_list)))

    if not pixel_sizes:
        pixel_size_x = round(min(list(map(lambda geo_transform: geo_transform[1], geo_transform_list))) / 5) * 5
        pixel_size_y = round(min(list(map(lambda geo_transform: abs(geo_transform[5]), geo_transform_list))) / 5) * 5
        pixel_sizes = min(pixel_size_x, pixel_size_y)

    cols = int((max_x - min_x) / pixel_sizes)
    rows = int((max_y - min_y) / pixel_sizes)
    shape = (rows, cols)

    dst_geo_transform = (min_x, pixel_sizes, 0, max_y, 0, -pixel_sizes)
    dst_projection = max(set(projection_list), key=projection_list.count)

    return dst_geo_transform, dst_projection, shape


"""Вспомогательные функции"""


def trajectory_break_points(row, default_size):
    """Формирование списка точек для ствола ГС"""
    if row['well type'] == 'vertical':
        return pd.Series({'x_coords': [row['T1_x']], 'y_coords': [row['T1_y']]})
    elif row['well type'] == 'horizontal':
        # Количество точек вдоль ствола
        num_points = int(np.ceil(row['length of well T1-3'] / default_size))
        # Для ГС создаем списки координат вдоль ствола
        x_coords = np.linspace(row['T1_x'], row['T3_x'], num_points)
        y_coords = np.linspace(row['T1_y'], row['T3_y'], num_points)
    return pd.Series({'x_coords': x_coords.tolist(), 'y_coords': y_coords.tolist()})
