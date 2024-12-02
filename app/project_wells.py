import pandas as pd
import numpy as np
import time
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from loguru import logger
from app.input_output import get_save_path
from app.drill_zones_handler.functions import (get_nearest_wells, get_params_nearest_wells)
from app.well_active_zones import get_value_map
from app.ranking_drilling.starting_rates import (get_geo_phys_and_default_params, calculate_starting_rate)


@logger.catch
def init_locate_project_wells(maps, list_zones, data_wells, save_directory, init_profit_cum_oil, init_area_well,
                              default_size_pixel, buffer_project_wells):
    """
    Размещение проектного фонда в зонах с высоким индексом возможности OI
    Parameters
    ----------
    maps - список всех карт
    data_wells - фрейм фактических скважин
    Returns
    -------
    gdf_project_wells - фрейм проектных скважин
    """
    list_zones = list_zones[1:]
    type_maps_list = list(map(lambda raster: raster.type_map, maps))
    # инициализация карты ОИЗ
    map_residual_recoverable_reserves = maps[type_maps_list.index("residual_recoverable_reserves")]

    list_project_wells = []

    start_time = time.time()
    for drill_zone in list_zones:
        drill_zone.get_init_project_wells(map_residual_recoverable_reserves, data_wells, init_profit_cum_oil,
                                               init_area_well, default_size_pixel, buffer_project_wells)
        list_project_wells.extend(drill_zone.list_project_wells)

    end_time = time.time()
    print(f"Время выполнения: {end_time - start_time} секунд")
    # Создаем GeoDataFrame из списка проектных скважин
    gdf_project_wells = gpd.GeoDataFrame({'well_number': [well.well_number for well in list_project_wells],
                                          'well_type': [well.well_type for well in list_project_wells],
                                          'geometry': [well.linestring for well in list_project_wells],
                                          'cluster': [well.cluster for well in list_project_wells],
                                          'POINT T1': [well.point_T1 for well in list_project_wells]})

    gdf_project_wells['buffer'] = gdf_project_wells.geometry.buffer(buffer_project_wells)

    logger.info("Сохраняем .png карту с начальным расположением проектного фонда")
    fig, ax = plt.subplots(figsize=(10, 10))

    gdf_project_wells.set_geometry("cluster").plot(ax=ax, cmap="viridis", linewidth=0.5)
    gdf_project_wells.set_geometry("buffer").plot(edgecolor="gray", facecolor="white", alpha=0.5, ax=ax)
    gdf_project_wells.set_geometry("geometry").plot(ax=ax, color='red', linewidth=1, markersize=1)

    # Отображение точек T1 на графике
    gdf_project_wells.set_geometry('POINT T1').plot(color='red', markersize=10, ax=ax)

    # Добавление текста с именами скважин рядом с точками T1
    for point, name in zip(gdf_project_wells['POINT T1'], gdf_project_wells['well_number']):
        if point is not None:
            plt.text(point.x + 2, point.y - 2, name, fontsize=6, ha='left')

    plt.gca().invert_yaxis()

    plt.savefig(f"{save_directory}/init_project_wells.png", dpi=400)

    return list_project_wells, gdf_project_wells


class ProjectWell:
    def __init__(self, well_number, cluster, point_T1, point_T3, linestring, azimuth, well_type):
        self.well_number = well_number
        self.cluster = cluster
        self.point_T1 = point_T1
        self.point_T3 = point_T3
        self.linestring = linestring
        self.azimuth = azimuth
        self.well_type = well_type
        self.P_well_init = None
        self.gdf_nearest_wells = None
        self.P_res = None
        self.NNT = None # должен быть эффективный для РБ
        self.initial_oil_saturation = None
        self.water_cut = None
        self.porosity = None
        self.permeability = None
        self.Q_liq = None
        self.Q_oil = None
        self.point_T1_geo = None
        self.point_T3_geo = None

    def get_params(self, gdf_fact_wells, maps, dict_geo_phys_properties, default_size_pixel):
        # Получение параметров со скважин окружения
        gdf_fact_wells_one_type = gdf_fact_wells[gdf_fact_wells["well type"] == self.well_type].reset_index(drop=True)

        self.P_well_init = get_params_nearest_wells(self.linestring, gdf_fact_wells_one_type, default_size_pixel,
                                                    'P_well_init_prod')
        # Получение параметров с карт
        type_map_list = list(map(lambda raster: raster.type_map, maps))

        # Инициализация всех необходимых карт
        map_pressure = maps[type_map_list.index("pressure")]
        map_NNT = maps[type_map_list.index("NNT")]
        map_initial_oil_saturation = maps[type_map_list.index("initial_oil_saturation")]
        map_water_cut = maps[type_map_list.index("water_cut")]
        map_porosity = maps[type_map_list.index("porosity")]
        map_permeability = maps[type_map_list.index("permeability")]

        # Преобразование координат скважин в пиксельные координаты
        self.point_T1_geo = Point(map_pressure.convert_coord_from_pixel((self.point_T1.x, self.point_T1.y)))
        self.point_T3_geo = Point(map_pressure.convert_coord_from_pixel((self.point_T3.x, self.point_T3.y)))


        # Снимаем значения с карт
        self.P_res = get_value_map(self.well_type, self.point_T1_geo.x, self.point_T1_geo.y, self.point_T3_geo.x, self.point_T3_geo.y,
                                   self.linestring.length * default_size_pixel, raster=map_pressure)
        self.NNT = get_value_map(self.well_type, self.point_T1_geo.x, self.point_T1_geo.y, self.point_T3_geo.x, self.point_T3_geo.y,
                                 self.linestring.length * default_size_pixel, raster=map_NNT)
        self.initial_oil_saturation = get_value_map(self.well_type, self.point_T1_geo.x, self.point_T1_geo.y, self.point_T3_geo.x,
                                                    self.point_T3_geo.y, self.linestring.length * default_size_pixel,
                                                    raster=map_initial_oil_saturation)
        self.water_cut = get_value_map(self.well_type, self.point_T1_geo.x, self.point_T1_geo.y, self.point_T3_geo.x,
                                       self.point_T3_geo.y, self.linestring.length * default_size_pixel, raster=map_water_cut)
        self.porosity = get_value_map(self.well_type, self.point_T1_geo.x, self.point_T1_geo.y, self.point_T3_geo.x,
                                      self.point_T3_geo.y, self.linestring.length * default_size_pixel, raster=map_porosity)
        self.permeability = get_value_map(self.well_type, self.point_T1_geo.x, self.point_T1_geo.y, self.point_T3_geo.x,
                                          self.point_T3_geo.y, self.linestring.length * default_size_pixel, raster=map_permeability)

        # Параметры для расчета РБ - !!! надо этот вызов переместить и добавить параметры в аргументы
        reservoir_params, fluid_params, well_params, coefficients = get_geo_phys_and_default_params(dict_geo_phys_properties)
        reservoir_params['f_w'] = self.water_cut
        reservoir_params['Phi'] = self.porosity
        reservoir_params['h'] = self.NNT
        reservoir_params['k_h'] = self.permeability
        reservoir_params['Pr'] = self.P_res

        well_params['L'] = self.linestring.length * default_size_pixel
        well_params['Pwf'] = self.P_well_init
        print(reservoir_params, fluid_params, well_params, coefficients)
        print("Расчет запускных")
        self.Q_liq, self.Q_oil = calculate_starting_rate(reservoir_params, fluid_params, well_params, coefficients)
        print(f"жидкость - {self.Q_liq}, нефть - {self.Q_oil}")


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from app.input_output import load_wells_data, load_geo_phys_properties, get_save_path
    from app.local_parameters import paths, parameters_calculation, default_well_params, default_coefficients
    from app.maps_handler.functions import mapping
    from app.well_active_zones import calculate_effective_radius
    from app.drill_zones_handler.drilling_zones import (calculate_drilling_zones)

    data_well_directory = paths["data_well_directory"]
    maps_directory = paths["maps_directory"]
    path_geo_phys_properties = paths["path_geo_phys_properties"]
    epsilon = parameters_calculation["epsilon"]
    min_samples = parameters_calculation["min_samples"]
    percent_low = 100 - parameters_calculation["percent_top"]
    default_size_pixel = parameters_calculation["default_size_pixel"]
    init_profit_cum_oil = parameters_calculation["init_profit_cum_oil"]
    init_area_well = parameters_calculation["init_area_well"]
    buffer_project_wells = parameters_calculation["buffer_project_wells"] / default_size_pixel

    _, data_wells, info = load_wells_data(data_well_directory=data_well_directory)
    name_field, name_object = info["field"], info["object_value"]

    save_directory = get_save_path("Infill_drilling", name_field, name_object.replace('/', '-'))

    dict_geo_phys_properties = load_geo_phys_properties(path_geo_phys_properties, name_field, name_object)

    maps = mapping(maps_directory=maps_directory,
                   data_wells=data_wells,
                   dict_properties=dict_geo_phys_properties,
                   default_size_pixel=default_size_pixel)

    data_wells = calculate_effective_radius(data_wells, dict_geo_phys_properties, maps)
    type_maps_list = list(map(lambda raster: raster.type_map, maps))
    # инициализация всех необходимых карт из списка
    map_opportunity_index = maps[type_maps_list.index("opportunity_index")]

    map_residual_recoverable_reserves = maps[type_maps_list.index("residual_recoverable_reserves")]

    list_zones, info_clusterization_zones = calculate_drilling_zones(maps=maps,
                                                                     epsilon=epsilon,
                                                                     min_samples=min_samples,
                                                                     percent_low=percent_low,
                                                                     data_wells=data_wells)

    list_project_wells, data_project_wells = init_locate_project_wells(maps=maps,
                                                                       list_zones=list_zones,
                                                                       data_wells=data_wells,
                                                                       save_directory=save_directory,
                                                                       init_profit_cum_oil=init_profit_cum_oil,
                                                                       init_area_well=init_area_well,
                                                                       default_size_pixel=default_size_pixel,
                                                                       buffer_project_wells=buffer_project_wells)

    data_wells_work = data_wells[(data_wells['Qo_cumsum'] > 0)|(data_wells['Winj_cumsum'] > 0)].reset_index(drop=True)
    import warnings

    with warnings.catch_warnings(action='ignore', category=pd.errors.SettingWithCopyWarning):
        data_wells_work["POINT T1"] = list(
            map(lambda x, y: Point(x, y), data_wells_work.T1_x_conv, data_wells_work.T1_y_conv))
        data_wells_work["POINT T3"] = list(
            map(lambda x, y: Point(x, y), data_wells_work.T3_x_conv, data_wells_work.T3_y_conv))
        data_wells_work["LINESTRING"] = list(
            map(lambda x, y: LineString([x, y]), data_wells_work["POINT T1"], data_wells_work["POINT T3"]))
        data_wells_work["LINESTRING"] = np.where(data_wells_work["POINT T1"] == data_wells_work["POINT T3"],
                                                data_wells_work["POINT T1"],
                                                list(map(lambda x, y: LineString([x, y]), data_wells_work["POINT T1"],
                                                         data_wells_work["POINT T3"])))
    gdf_wells_work = gpd.GeoDataFrame(data_wells_work, geometry="LINESTRING")
    gdf_wells_work['length_conv'] = gdf_wells_work.apply(lambda row: row['POINT T1'].distance(row['POINT T3']),
                                                           axis=1)

    for well in list_project_wells:
        well.get_params(gdf_wells_work, maps, dict_geo_phys_properties, default_size_pixel)

    gdf_project_wells = gpd.GeoDataFrame({'well_number': [well.well_number for well in list_project_wells],
                                          'well_type': [well.well_type for well in list_project_wells],
                                          'length': [well.linestring.length for well in list_project_wells],
                                          'water_cut': [well.water_cut for well in list_project_wells],
                                          'Q_liq': [well.Q_liq for well in list_project_wells],
                                          'Q_oil': [well.Q_oil for well in list_project_wells]}
    )

    gdf_project_wells.to_excel(save_directory + r'\all.xlsx', sheet_name='РБ', index=False)


    pass

