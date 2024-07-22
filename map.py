import numpy as np
from osgeo import gdal

class Map:
    def __init__(self, file_path):
        no_value = 0  # значение для заполнения пустот на карте
        self.file_path = file_path
        self.data, self.geo_transform, self.projection = self.read_raster(no_value)
        self.type_map = ""

    def read_raster(self, no_value=np.nan):
        dataset = gdal.Open(self.file_path)
        ndv = dataset.GetRasterBand(1).GetNoDataValue()
        band = dataset.GetRasterBand(1)
        data = band.ReadAsArray()
        data = np.where(data >= ndv, no_value, data)
        geo_transform = dataset.GetGeoTransform()
        projection = dataset.GetProjection()
        return data, geo_transform, projection

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
        # self.data = dst_ds.GetRasterBand(1).ReadAsArray()
        return

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
        plt.imshow(self.data, cmap='viridis')
        plt.colorbar()
        plt.savefig(filename)


def max_resolution(list_rasters, type_resize="min"):
    """ Поиск самого большого разрешения среди загруженных карт"""
    data_list = list(map(lambda raster: raster.data, list_rasters))
    geo_transform_list = list(map(lambda raster: raster.geo_transform, list_rasters))

    min_x = min(geo_transform[0] for geo_transform in geo_transform_list)
    max_y = max(geo_transform[3] for geo_transform in geo_transform_list)

    # Загрузка двух сеток данных
    array1, geo_transform1, projection1 = map1.data, map1.geo_transform, map1.projection
    array2, geo_transform2, projection2 = map2.data, map2.geo_transform, map2.projection

    # Determine the resolution and bounding box of the output
    min_x = min(geo_transform1[0], geo_transform2[0])
    max_x = max(geo_transform1[0] + geo_transform1[1] * array1.shape[1],
                geo_transform2[0] + geo_transform2[1] * array2.shape[1])
    min_y = min(geo_transform1[3] + geo_transform1[5] * array1.shape[0],
                geo_transform2[3] + geo_transform2[5] * array2.shape[0])
    max_y = max(geo_transform1[3], geo_transform2[3])

    pixel_size_x = min(geo_transform1[1], geo_transform2[1])
    pixel_size_y = min(abs(geo_transform1[5]), abs(geo_transform2[5]))

    cols = int((max_x - min_x) / pixel_size_x)
    rows = int((max_y - min_y) / pixel_size_y)

    dst_geo_transform = (min_x, pixel_size_x, 0, max_y, 0, -pixel_size_y)

    # Reproject both arrays to the output resolution and bounds
    map1.resize(dst_geo_transform, projection1, (rows, cols))

    dst_geo_transform, projection1,(rows, cols)


def calculator():
    return print('Opportunity map')