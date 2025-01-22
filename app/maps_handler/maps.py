import os
import numpy as np
import pandas as pd
from osgeo import gdal
from loguru import logger
from scipy.interpolate import RBFInterpolator, RegularGridInterpolator
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter

from .config import list_names_map


class Map:
    def __init__(self, data, geo_transform, projection, type_map):
        self.data = data
        self.geo_transform = geo_transform
        self.projection = projection
        self.type_map = type_map

    def normalize_data(self):
        """MinMax нормализация карты, метод возвращает новую карту"""
        new_data = (self.data - np.min(self.data)) / (np.max(self.data) - np.min(self.data))
        return Map(new_data, self.geo_transform, self.projection, self.type_map)

    def resize(self, dst_geo_transform, dst_projection, dst_shape):
        """Изменение размеров карты"""
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

    def convert_coord_to_pix(self, array):
        """Преобразование координат массива в пиксельные координаты в соответствии с geo_transform карты"""
        x, y = array
        conv_x = np.where(x != 0, ((x - self.geo_transform[0]) / self.geo_transform[1]).astype(int), np.nan)
        conv_y = np.where(y != 0, ((self.geo_transform[3] - y) / abs(self.geo_transform[5])).astype(int), np.nan)
        return conv_x, conv_y

    def convert_coord_to_geo(self, array):
        """Преобразование координат массива из пиксельных координат в соответствии с geo_transform карты"""
        conv_x, conv_y = array
        x = np.where(conv_x != 0, (conv_x * self.geo_transform[1] + self.geo_transform[0]), np.nan)
        y = np.where(conv_y != 0, (self.geo_transform[3] - conv_y * abs(self.geo_transform[5])), np.nan)
        return x, y

    def get_values(self, x, y):
        """
        Получить значения с карты по спискам x и y
        Parameters
        ----------
        x, y - array координаты x, y в пикселях

        Returns
        -------
        Список значений по координатам с карты
        """
        interpolated_values, values_out = [], []

        # Создаем массивы координат
        x_coords = np.arange(self.data.shape[1])
        y_coords = np.arange(self.data.shape[0])

        # Проверка, чтобы не выйти за пределы изображения
        mask_x_in = (x >= x_coords[0]) & (x <= x_coords[-1])
        mask_y_in = (y >= y_coords[0]) & (y <= y_coords[-1])
        mask_all_in = mask_x_in & mask_y_in
        x_in, y_in = list(np.array(x)[mask_all_in]), list(np.array(y)[mask_all_in])

        if len(x_in):
            # Создаем интерполятор
            interpolator = RegularGridInterpolator((y_coords, x_coords), self.data, method='linear')
            # Получаем интерполированные значения в точках (x, y)
            interpolated_values = interpolator((y_in, x_in))

        x_out, y_out = list(np.array(x)[~mask_all_in]), list(np.array(y)[~mask_all_in])
        if len(x_out):
            values_out = [0] * len(x_out)

        values = list(interpolated_values) + values_out
        return values

    def save_img(self, filename, data_wells=None, list_zones=None, info_clusterization_zones=None, project_wells=None):
        import matplotlib.pyplot as plt

        # Определение размера осей
        x = (self.geo_transform[0], self.geo_transform[0] + self.geo_transform[1] * self.data.shape[1])
        y = (self.geo_transform[3] + self.geo_transform[5] * self.data.shape[0], self.geo_transform[3])
        x_plt = self.convert_coord_to_pix((np.array(x), np.array(y)))[0]
        y_plt = self.convert_coord_to_pix((np.array(x), np.array(y)))[1]
        d_x, d_y = abs(x_plt[1] - x_plt[0]), abs(y_plt[1] - y_plt[0])
        count = len(str(int(min(d_x, d_y))))

        plt.figure(figsize=(d_x / 10 ** (count - 1), d_y / 10 ** (count - 1)))
        element_size = min(d_x, d_y) / 10 ** (count - 0.2)
        font_size = min(d_x, d_y) / 10 ** (count - 0.5)

        plt.imshow(self.data, cmap='viridis', origin="upper")
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=font_size * 8)

        # Отображение зон кластеризации на карте
        title = ""
        if list_zones is not None:
            labels = list(map(lambda zone: zone.rating, list_zones))
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
                zone = list_zones[labels.index(i)]
                x_zone = zone.x_coordinates
                y_zone = zone.y_coordinates
                plt.scatter(x_zone, y_zone, color=c, alpha=0.6, s=1)

                if i != -1:
                    #  Отрисовка проектного фонда
                    if project_wells:
                        for well in zone.list_project_wells:
                            # координаты скважин в пиксельных координатах
                            x_t1, y_t1 = (well.POINT_T1_pix.x, well.POINT_T1_pix.y)
                            x_t3, y_t3 = (well.POINT_T3_pix.x, well.POINT_T3_pix.y)

                            # Отображение скважин на карте
                            plt.plot([x_t1, x_t3], [y_t1, y_t3], c='red', linewidth=element_size * 0.3)
                            plt.scatter(x_t1, y_t1, s=element_size, c='red', marker="o", linewidths=0.1)

                            # Отображение имен скважин рядом с точками T1
                            plt.text(x_t1 + 3, y_t1 - 3, well.well_number, fontsize=font_size, ha='left')

            if info_clusterization_zones is not None:
                title = (f"Epsilon = {info_clusterization_zones['epsilon']}\n "
                         f"min_samples = {info_clusterization_zones['min_samples']} \n "
                         f"with {info_clusterization_zones['n_clusters']} clusters")

        # Отображение списка скважин на карте
        if data_wells is not None:
            column_lim_x = ['T1_x_geo', 'T3_x_geo']
            for column in column_lim_x:
                data_wells = data_wells.loc[((data_wells[column] <= x[1]) & (data_wells[column] >= x[0]))]
            column_lim_y = ['T1_y_geo', 'T3_y_geo']
            for column in column_lim_y:
                data_wells = data_wells.loc[((data_wells[column] <= y[1]) & (data_wells[column] >= y[0]))]

            # координаты скважин в пиксельных координатах
            x_t1, y_t1 = (data_wells.T1_x_pix, data_wells.T1_y_pix)
            x_t3, y_t3 = (data_wells.T3_x_pix, data_wells.T3_y_pix)

            # Отображение скважин на карте
            plt.plot([x_t1, x_t3], [y_t1, y_t3], c='black', linewidth=element_size * 0.3)
            plt.scatter(x_t1, y_t1, s=element_size, c='black', marker="o", linewidths=0.1)

            # Отображение имен скважин рядом с точками T1
            for x, y, name in zip(x_t1, y_t1, data_wells.well_number):
                plt.text(x + 3, y - 3, name, fontsize=font_size, ha='left')

        plt.title(f"{self.type_map}\n {title}", fontsize=font_size * 8)
        plt.tick_params(axis='both', which='major', labelsize=font_size * 8)
        plt.contour(self.data, levels=8, colors='black', origin='lower', linewidths=font_size / 100)
        plt.xlim(x_plt)
        plt.ylim([y_plt[1], y_plt[0]])
        plt.gca().invert_yaxis()
        plt.savefig(filename, dpi=400)
        plt.close()


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


def read_array(data_wells, name_column_map, type_map, geo_transform, size,
               accounting_GS=True,
               radius=1000):
    """
    Создание объекта класса MAP из DataFrame
    Parameters
    ----------
    data_wells - DataFrame
    name_column_map - наименование колонок, по значениям котрой строится карта
    type_map - тип карты
    radius - радиус экстраполяции за крайние скважины
    geo_transform - геотрансформация карты
    size - размер массива (x, y)
    accounting_GS - учет ствола горизонтальных скважин при построении карты
    radius - радиус интерполяции от точки

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
            data_wells_with_work[name_column_map] = np.where(data_wells_with_work['well_type'] == 'horizontal',
                                                             data_wells_with_work[name_column_map] /
                                                             data_wells_with_work["length_geo"],
                                                             data_wells_with_work[name_column_map])
            data_wells_with_work[name_column_map] = data_wells_with_work.groupby('well_type')[
                name_column_map].transform(lambda x: (x - x.min()) / (x.max() - x.min()))

        """Обработка колонок last_rate_oil и init_rate_oil через коэффициент
        NNS_mean_init_Ql = data_wells[(data_wells['init_Ql_rate'] != 0) &
                                      (data_wells['well_type'] == 'vertical')]['init_Ql_rate'].mean()
        GS_mean_init_Ql = data_wells[(data_wells['init_Ql_rate'] != 0) &
                                     (data_wells['well_type'] == 'horizontal')]['init_Ql_rate'].mean()
        coefficient_GS_to_NNS = round(NNS_mean_init_Ql / GS_mean_init_Ql, 1)
        logger.info(f'Коэффициент соотношения дебитов жидкости ННС и ГС: {coefficient_GS_to_NNS}')

        with warnings.catch_warnings(action='ignore', category=pd.errors.SettingWithCopyWarning):
            data_wells_with_work[name_column_map] = np.where(data_wells_with_work['well_type'] == 'horizontal',
                                                         data_wells_with_work[name_column_map] * coefficient_GS_to_NNS,
                                                         data_wells_with_work[name_column_map])"""
    elif type_map == "permeability_fact_wells":
        # permeability_column = data_wells[data_wells['permeability_fact'] > 0]['permeability_fact']
        # # Рассчитываем квартили
        # q1 = np.percentile(permeability_column, 25)
        # q3 = np.percentile(permeability_column, 75)
        # iqr = q3 - q1
        # # Определяем порог для отсеивания "выбросов"
        # upper_bound = q3 + 1.5 * iqr
        # # Выбираем только ту проницаемость, которая меньше или равна верхнему пределу
        # data_wells = data_wells[(data_wells['permeability_fact'] <= upper_bound)
        #                         & (data_wells['permeability_fact'] > 0)].reset_index(drop=True)
        data_wells = data_wells[(data_wells['permeability_fact'] > 0)].reset_index(drop=True)
        data_wells_with_work = data_wells
    else:
        data_wells_with_work = pd.DataFrame()
        if type_map not in list_names_map:
            raise logger.critical(f"Неверный тип карты! {type_map}")

    if accounting_GS:
        # Формирование списка точек для ствола каждой скважины
        coordinates = data_wells_with_work.apply(
            lambda row: pd.Series(trajectory_break_points(row['well_type'], row['T1_x_geo'],
                                                          row['T1_y_geo'], row['T3_x_geo'],
                                                          row['T3_y_geo'], row['length_geo'],
                                                          default_size=geo_transform[1]),
                                                          index=['x_coords', 'y_coords']), axis=1)
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
        x, y = np.array(data_wells_with_work.T1_x_geo), np.array(data_wells_with_work.T1_y_geo)
        well_coord = np.column_stack((x, y))
        # Выделяем значения для карты
        values = np.array(data_wells_with_work[name_column_map])

    # Определение границ интерполяции
    x_min, x_max = [geo_transform[0], geo_transform[0] + geo_transform[1] * size[1]]
    y_min, y_max = [geo_transform[3] + geo_transform[5] * size[0], geo_transform[3]]

    default_size = geo_transform[1]

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
    rbf_interpolator = RBFInterpolator(well_coord, values, kernel='linear', epsilon=10)  # сглаживание smoothing=0.5

    # Предсказание значений на сетке
    valid_grid_z = rbf_interpolator(grid_points_mask)
    grid_z = np.full(grid_x.shape, np.nan)
    grid_z.ravel()[points_mask] = valid_grid_z

    # Применяем ограничения на минимальное и максимальное значения карты
    grid_z = np.clip(grid_z, values.min(), values.max()).T

    # Определение геотрансформации
    geo_transform = [x_min, (x_max - x_min) / grid_z.shape[1], 0, y_max, 0, -((y_max - y_min) / grid_z.shape[0])]

    return Map(grid_z, geo_transform, projection='', type_map=type_map)


def get_map_reservoir_score(map_NNT, map_permeability) -> Map:
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
    return map_reservoir_score


def get_map_potential_score(map_residual_recoverable_reserves, map_last_rate_oil, map_init_rate_oil) -> Map:
    """
    Оценка показателей разработки
    -------
    Map(type_map=potential_score)
    """
    map_last_rate_oil.data = np.nan_to_num(map_last_rate_oil.data)
    map_init_rate_oil.data = np.nan_to_num(map_init_rate_oil.data)

    norm_last_rate_oil = map_last_rate_oil.normalize_data()
    norm_init_rate_oil = map_init_rate_oil.normalize_data()
    norm_residual_recoverable_reserves = map_residual_recoverable_reserves.normalize_data()

    data_potential_score = (norm_residual_recoverable_reserves.data
                            + norm_last_rate_oil.data
                            + norm_init_rate_oil.data) / 3

    map_potential_score = Map(data_potential_score,
                              norm_last_rate_oil.geo_transform,
                              norm_last_rate_oil.projection,
                              "potential_score")
    return map_potential_score


def get_map_risk_score(map_water_cut, map_initial_oil_saturation, map_pressure, init_pressure, sigma=5) -> Map:
    """
    Оценка проблем
    Parameters
    ----------
    P_init - начально давление в атм
    sigma - параметр для определения степени сглаживания
    -------
    Map(type_map=risk_score)
    """
    # Подготовка карты снижения давлений
    data_delta_P = init_pressure - map_pressure.data
    data_delta_P = np.where(data_delta_P < 0, 0, data_delta_P)
    map_delta_P = Map(data_delta_P, map_pressure.geo_transform, map_pressure.projection,
                      type_map="delta_P").normalize_data()
    # Подготовка карты текущей обводненности
    data_init_water_cut = (1 - map_initial_oil_saturation.data) * 100
    mask = np.isnan(map_water_cut.data)
    data_water_cut = np.where(mask, data_init_water_cut, map_water_cut.data)

    # Применение гауссова фильтра для сглаживания при объединении карт обводненности и начальной нефтенасыщенности
    data_water_cut = gaussian_filter(data_water_cut, sigma=sigma)
    map_water_cut.data = data_water_cut
    norm_water_cut = map_water_cut.normalize_data()

    data_risk_score = (norm_water_cut.data + map_delta_P.data) / 2
    map_risk_score = Map(data_risk_score, map_water_cut.geo_transform, map_water_cut.projection, "risk_score")
    return map_risk_score


def get_map_opportunity_index(map_reservoir_score, map_potential_score, map_risk_score) -> Map:
    """
    Оценка индекса возможностей
    -------
    Map(type_map=opportunity_index)
    """
    k_reservoir = k_potential = k_risk = 1  # все карты оценок имеют равные веса
    data_opportunity_index = (k_reservoir * map_reservoir_score.data +
                              k_potential * map_potential_score.data -
                              k_risk * map_risk_score.data)
    map_opportunity_index = Map(data_opportunity_index,
                                map_reservoir_score.geo_transform,
                                map_reservoir_score.projection, "opportunity_index")
    map_opportunity_index = map_opportunity_index.normalize_data()
    return map_opportunity_index


"""___Вспомогательная функция___"""


def trajectory_break_points(well_type, T1_x, T1_y, T3_x, T3_y, length_of_well, default_size):
    """Формирование списка точек для ствола ГС"""
    if well_type == 'vertical':
        return [T1_x], [T1_y]
    elif well_type == 'horizontal':
        # Количество точек вдоль ствола
        num_points = int(np.ceil(length_of_well / default_size))
        # Для ГС создаем списки координат вдоль ствола
        x_coords = np.linspace(T1_x, T3_x, num_points)
        y_coords = np.linspace(T1_y, T3_y, num_points)
    return x_coords.tolist(), y_coords.tolist()