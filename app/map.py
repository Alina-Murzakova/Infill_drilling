import os
import numpy as np
from osgeo import gdal
from loguru import logger
from scipy.interpolate import griddata, RBFInterpolator, LinearNDInterpolator
from scipy.spatial import KDTree

# Названия столбцов в Excel
date = 'Дата'
well_number = '№ скважины'
field = 'Месторождение'
objects = 'Объекты работы'
work_marker = 'Характер работы'
well_status = 'Состояние'
Qo_rate = 'Дебит нефти за последний месяц, т/сут'
Ql_rate = 'Дебит жидкости за последний месяц, т/сут'
water_cut = 'Обводненность за посл.месяц, % (вес)'
Winj_rate = 'Приемистость за последний месяц, м3/сут'
time_work = "Время работы, часы"
P_well = "Забойное давление (ТР), атм"
P_pressure = 'Пластовое давление (ТР), атм'
x1 = "Координата X"
y1 = "Координата Y"
x2 = "Координата забоя Х (по траектории)"
y2 = "Координата забоя Y (по траектории)"

@logger.catch
def mapping(maps_directory, save_directory, data_wells):
    logger.info(f"path: {maps_directory}")
    content = os.listdir(path=maps_directory)
    if content:
        logger.info(f"maps: {len(content)}")
        maps = []
    else:
        raise logger.critical("no maps!")

    for map_file in content:
        maps.append(read_raster(f'{maps_directory}/{map_file}', no_value=0))

    maps.append(read_array(data_wells))

    for i, raster in enumerate(maps):
        raster.save_img(f"{save_directory}/map{i + 1}.png", data_wells)

    dst_geo_transform, dst_projection, shape = final_resolution(maps)

    res_maps = list(map(lambda raster: raster.resize(dst_geo_transform, dst_projection, shape), maps))

    for i, raster in enumerate(res_maps):
        raster.save_img(f"{save_directory}/res_map{i + 1}.png", data_wells)

    read_array(data_wells)

pass


def read_raster(file_path, no_value=np.nan):
    """
    no_value значение для заполнения пустот на карте
    """
    dataset = gdal.Open(file_path)
    ndv = dataset.GetRasterBand(1).GetNoDataValue()
    band = dataset.GetRasterBand(1)
    data = band.ReadAsArray()
    data = np.where(data >= ndv, no_value, data)
    geo_transform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()
    name_file = os.path.basename(file_path).replace(".grd", "")
    return Map(data, geo_transform, projection, type_map=name_file)

def read_array(data_wells): # потом добавлю тип строящейся карты в переменные функции и входные параметры изменяемые в зависимости от карты

    default_size = 50.0
    radius = 2000

    data_wells_with_work = data_wells[(data_wells[Ql_rate] > 0) | (data_wells[Winj_rate] > 0)]
    # data_wells_with_work[water_cut] = np.where(data_wells_with_work[Winj_rate] > 0, 100.0, data_wells_with_work[water_cut])
    data_wells_with_work.loc[data_wells_with_work[Winj_rate] > 0, water_cut] = 100.0
    # well_coord = data_wells_with_work[[x1, y1]].values
    X = np.array(data_wells_with_work[x1])
    Y = np.array(data_wells_with_work[y1])
    well_coord = np.column_stack((X, Y))
    values = np.array(data_wells_with_work[water_cut])

    # Определение границ интерполяции
    x_min, x_max = min(X), max(X)
    y_min, y_max = min(Y), max(Y)

    #Расширение границ
    expand = 0.2
    x_min -= (x_max - x_min) * expand
    x_max += (x_max - x_min) * expand
    y_min -= (y_max - y_min) * expand
    y_max += (y_max - y_min) * expand

    # Создание KDTree для координат скважин
    tree = KDTree(well_coord)

    # Создание сетки для интерполяции
    grid_x, grid_y = np.mgrid[x_min:x_max:default_size, y_min:y_max:default_size]
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    # Поиск точек сетки, которые находятся в пределах заданного радиуса от любой скважины
    points_in_radius = tree.query_ball_point(grid_points, r=radius)

    # Создаем маску для этих точек
    points_mask = np.array([len(indices) > 0 for indices in points_in_radius])
    grid_points_mask = grid_points[points_mask]

    # # Интерполяция griddata
    # grid_z = griddata(well_coord, values, (grid_x, grid_y), method='linear', fill_value=0)

    # Использование RBFInterpolator
    rbfi = RBFInterpolator(well_coord, values, kernel='linear') # , smoothing=0.5
    # lin = LinearNDInterpolator(well_coord, values)

    # Предсказание значений на сетке
    valid_grid_z = rbfi(grid_points_mask)
    grid_z = np.full(grid_x.shape, np.nan)
    grid_z.ravel()[points_mask] = valid_grid_z
    # grid_z = rbfi(grid_points).reshape(grid_x.shape) # использовать без маски
    # grid_z = lin(grid_points).reshape(grid_x.shape)

    # Применяем ограничения на минимальное и максимальное значения
    grid_z = np.clip(grid_z, 0, 100).T

    # Определение геотрансформации
    geo_transform = [x_min, (x_max - x_min) / grid_z.shape[1], 0, y_max, 0, -((y_max - y_min) / grid_z.shape[0])]
    projection = ''
    return Map(grid_z, geo_transform, projection, type_map="interpolated_water_cut")

class Map:
    def __init__(self, data, geo_transform, projection, type_map):
        self.data = data
        self.geo_transform = geo_transform
        self.projection = projection
        self.type_map = type_map

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

    def add(self, other_map):
        if self.data.shape != other_map.data.shape:
            raise ValueError("Map shapes do not match.")
        self.data += other_map.data

    def show(self):
        import matplotlib.pyplot as plt
        plt.imshow(self.data, cmap='viridis')
        plt.colorbar()
        plt.show()

    def save_grd_file(self, filename):
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(filename, self.data.shape[1], self.data.shape[0], 1, gdal.GDT_Float32)
        dataset.SetGeoTransform(self.geo_transform)
        dataset.SetProjection(self.projection)
        dataset.GetRasterBand(1).WriteArray(self.data)
        dataset.FlushCache()

    def convert_coord(self, data_wells):
        # Преобразование координат скважин в пиксельные координаты
        x_t1 = ((data_wells[x1] - self.geo_transform[0]) / self.geo_transform[1]).astype(int)
        y_t1 = ((self.geo_transform[3] - data_wells[y1]) / abs(self.geo_transform[5])).astype(int)
        x_t2 = np.where(data_wells[x2] != 0,
                        ((data_wells[x2] - self.geo_transform[0]) / self.geo_transform[1]).astype(int), np.nan)
        y_t2 = np.where(data_wells[y2] != 0,
                        ((self.geo_transform[3] - data_wells[y2]) / abs(self.geo_transform[5])).astype(int), np.nan)
        return x_t1, y_t1, x_t2, y_t2

    def save_img(self, filename, data_wells):
        import matplotlib.pyplot as plt
        # plt.imsave(filename, self.data, cmap='viridis')

        plt.imshow(self.data, cmap='viridis')
        plt.colorbar()

        if data_wells is not None:
            # Преобразование координат скважин в пиксельные координаты
            x_t1 = ((data_wells[x1] - self.geo_transform[0]) / self.geo_transform[1]).astype(int)
            y_t1 = ((self.geo_transform[3] - data_wells[y1]) / abs(self.geo_transform[5])).astype(int)
            x_t2 = np.where(data_wells[x2] != 0, ((data_wells[x2] - self.geo_transform[0]) / self.geo_transform[1]).astype(int), np.nan)
            y_t2 = np.where(data_wells[y2] != 0, ((self.geo_transform[3] - data_wells[y2]) / abs(self.geo_transform[5])).astype(int), np.nan)

            # Отображение скважин на карте
            plt.plot([x_t1, x_t2], [y_t1, y_t2], c='black', linewidth=0.3)
            plt.scatter(x_t1, y_t1, s=0.3, c='black', marker="o")

            # Отображение имен скважин рядом с точками T1
            for x, y, name in zip(x_t1, y_t1, data_wells[well_number]):
                plt.text(x + 3, y - 3, name, fontsize=1.8, ha='left')

        # plt.legend()
        # plt.ylim(min_y1, max_y1)
        # plt.xlim(min_x1, max_x1)
        plt.title(self.type_map, fontsize=10)
        plt.tick_params(axis='both', which='major', labelsize=5)
        plt.savefig(filename, dpi=300)
        plt.close()



def final_resolution(list_rasters, pixel_sizes="default"):
    """Поиск наименьшего размера карты, который станет целевым для расчета

    list_rasters - список карт
    pixel_sizes ("default"/anything) - шаг сетки (по-умолчанию 50/поиск наименьшего шага среди сеток)

    return: geo_transform, projection, shape
    """
    default_size = 50

    data_list = list(map(lambda raster: raster.data, list_rasters))
    geo_transform_list = list(map(lambda raster: raster.geo_transform, list_rasters))
    projection_list = list(map(lambda raster: raster.projection, list_rasters))

    min_x = max(list(map(lambda geo_transform: geo_transform[0], geo_transform_list)))
    max_x = min(list(map(lambda geo_transform, data: geo_transform[0] + geo_transform[1] * data.shape[1],
                         geo_transform_list, data_list)))
    min_y = max(list(map(lambda geo_transform, data: geo_transform[3] + geo_transform[5] * data.shape[0],
                         geo_transform_list, data_list)))
    max_y = min(list(map(lambda geo_transform: geo_transform[3], geo_transform_list)))

    if pixel_sizes == "default":
        pixel_size_x = pixel_size_y = default_size
    else:
        pixel_size_x = min(list(map(lambda geo_transform: geo_transform[1], geo_transform_list)))
        pixel_size_y = min(list(map(lambda geo_transform: abs(geo_transform[5]), geo_transform_list)))

    cols = int((max_x - min_x) / pixel_size_x)
    rows = int((max_y - min_y) / pixel_size_y)
    shape = (rows, cols)

    dst_geo_transform = (min_x, pixel_size_x, 0, max_y, 0, -pixel_size_y)
    dst_projection = max(set(projection_list), key=projection_list.count)

    return dst_geo_transform, dst_projection, shape


def calculator():
    return print('Opportunity map')
