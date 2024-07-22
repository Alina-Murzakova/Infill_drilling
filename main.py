from map import Map


def run():
    # Загрузка карт
    map1 = Map('input_files/Крайнее_Ю1/Ascii grid grd/Изобары.grd')
    map2 = Map('input_files/Крайнее_Ю1/Ascii grid grd/Пористость.grd')

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
    map1.show()
    # Reproject both arrays to the output resolution and bounds
    map1.resize(dst_geo_transform, projection1,(rows, cols))
    map2.resize(dst_geo_transform, projection2,(rows, cols))

    # Арифметическая операция (суммирование) // нужна нормализация
    map1.add(map2)

    # Визуализация результата
    map1.show()


if __name__ == '__main__':  # avoid run on import
    run()