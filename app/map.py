import os
import numpy as np
from osgeo import gdal
from loguru import logger


@logger.catch
def mapping(maps_directory, save_directory):
    logger.info(f"path: {maps_directory}")
    content = os.listdir(path=maps_directory)
    if content:
        logger.info(f"maps: {len(content)}")
        maps = []
    else:
        raise logger.critical("no maps!")

    for map_file in content:
        maps.append(read_raster(f'{maps_directory}/{map_file}', no_value=0))

    for i, raster in enumerate(maps):
        raster.save_img(f"{save_directory}/map{i + 1}.png")

    dst_geo_transform, dst_projection, shape = final_resolution(maps)

    res_maps = list(map(lambda raster: raster.resize(dst_geo_transform, dst_projection, shape), maps))

    for i, raster in enumerate(res_maps):
        raster.save_img(f"{save_directory}/res_map{i + 1}.png")


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

    def save_img(self, filename):
        import matplotlib.pyplot as plt
        plt.imsave(filename, self.data, cmap='viridis')


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
