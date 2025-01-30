import geopandas as gpd
import pandas as pd
import numpy as np

from loguru import logger

from app.decline_rate.decline_rate import get_avg_decline_rates
from app.well_active_zones import get_parameters_voronoi_cells, save_picture_voronoi
from app.decline_rate.functions import production_model
from app.well_active_zones import get_value_map
from app.ranking_drilling.starting_rates import calculate_starting_rate, check_FracCount
from app.decline_rate.residual_reserves import prepare_mesh_map_rrr


class ProjectWell:
    def __init__(self, well_number, cluster, POINT_T1_pix, POINT_T3_pix, LINESTRING_pix, azimuth, well_type):
        self.well_number = well_number
        self.cluster = cluster
        self.POINT_T1_pix = POINT_T1_pix
        self.POINT_T3_pix = POINT_T3_pix
        self.LINESTRING_pix = LINESTRING_pix
        self.azimuth = azimuth
        self.well_type = well_type

        self.length_geo = None
        self.length_pix = None
        self.POINT_T1_geo = None
        self.POINT_T3_geo = None
        self.LINESTRING_geo = None

        self.P_well_init = None
        self.gdf_nearest_wells = None
        self.P_reservoir = None
        self.NNT = None  # должен быть эффективный для РБ
        self.So = None
        self.water_cut = None
        self.m = None
        self.permeability = None
        self.init_Ql_rate = None
        self.init_Qo_rate = None
        self.decline_rates = None
        self.r_eff = None
        self.reserves = None  # тыс. т

        self.Qo_rate = None
        self.Ql_rate = None
        self.Qo = None
        self.Ql = None

    def get_nearest_wells(self, df_wells, threshold, k=5):
        """
        Получаем ближайшее окружение скважины
        Parameters
        ----------
        gdf_fact_wells - gdf с фактическими скважинами
        k=5 - количество ближайших фактических скважин
        threshold = 2500 / default_size_pixel - максимальное расстояние для исключения скважины
                                                из ближайших скважин, пиксели
        """
        gdf_wells = gpd.GeoDataFrame(df_wells, geometry="LINESTRING_pix")
        # GeoDataFrame со скважинами того же типа
        gdf_fact_wells = gdf_wells[gdf_wells["well_type"] == self.well_type].reset_index(drop=True)
        if gdf_fact_wells.empty:
            logger.warning(f"На объекте нет фактических скважин типа {self.well_type}! \n "
                           "Необходимо задать азимут, длину, Рзаб проектных скважин вручную.")

        # Вычисляем расстояния до всех скважин
        distances = self.LINESTRING_pix.distance(gdf_fact_wells["LINESTRING_pix"])
        sorted_distances = distances.nsmallest(k)
        nearest_hor_wells_index = [sorted_distances.index[0]]  # Ближайшая скважина всегда включается

        for i in range(1, len(sorted_distances)):
            # Проверяем разницу с первой добавленной ближайшей скважиной
            if sorted_distances.iloc[i] - sorted_distances.iloc[0] < threshold:
                nearest_hor_wells_index.append(sorted_distances.index[i])

        # Извлечение строк GeoDataFrame по индексам
        self.gdf_nearest_wells = gdf_fact_wells.loc[nearest_hor_wells_index]
        pass

    def get_params_nearest_wells(self, dict_parameters_coefficients):
        """Получаем параметры для проектной скважины с ближайших фактических"""
        # Проверка на наличие ближайшего окружения
        if self.gdf_nearest_wells.empty:
            logger.warning(f"Проверьте наличие окружения для проектной скважины {self.well_number}!")
        # Расчет средних параметров по выбранному окружению.
        # Выбираем только те строки, где значение Рзаб больше 0
        self.P_well_init = np.mean(self.gdf_nearest_wells[(self.gdf_nearest_wells['init_P_well_prod'] > 0) &
                                                          (self.gdf_nearest_wells['init_P_well_prod'].notna())]
                                   ['init_P_well_prod'])
        if np.isnan(self.P_well_init):
            self.P_well_init = dict_parameters_coefficients['well_params']['init_P_well']
        # Выбираем только те строки, где значение проницаемости больше 0 и не nan
        self.permeability = np.mean(self.gdf_nearest_wells[(self.gdf_nearest_wells['permeability_fact'] > 0) &
                                                           (self.gdf_nearest_wells['permeability_fact'].notna())]
                                    ['permeability_fact'])
        if np.isnan(self.permeability):
            self.permeability = dict_parameters_coefficients['reservoir_params']['k_h']
        pass

    def get_params_maps(self, maps):
        """Значения с карт для скважины"""
        type_map_list = list(map(lambda raster: raster.type_map, maps))
        list_arguments = (self.well_type, self.POINT_T1_pix.x, self.POINT_T1_pix.y,
                          self.POINT_T3_pix.x, self.POINT_T3_pix.y, self.length_pix)
        self.P_reservoir = get_value_map(*list_arguments, raster=maps[type_map_list.index('pressure')])
        self.NNT = get_value_map(*list_arguments, raster=maps[type_map_list.index('NNT')])
        self.m = get_value_map(*list_arguments, raster=maps[type_map_list.index('porosity')])
        self.So = get_value_map(*list_arguments, raster=maps[type_map_list.index('initial_oil_saturation')])
        self.water_cut = get_value_map(*list_arguments, raster=maps[type_map_list.index('water_cut')])
        pass

    def get_starting_rates(self, maps, dict_parameters_coefficients):
        self.get_params_maps(maps)

        kv_kh, Swc, Sor, Fw, m1, Fo, m2, Bw = (
            list(map(lambda name: dict_parameters_coefficients['default_well_params'][name],
                     ['kv_kh', 'Swc', 'Sor', 'Fw', 'm1', 'Fo', 'm2', 'Bw'])))

        reservoir_params = dict_parameters_coefficients['reservoir_params']
        fluid_params = dict_parameters_coefficients['fluid_params']
        well_params = dict_parameters_coefficients['well_params']
        coefficients = dict_parameters_coefficients['coefficients']

        reservoir_params['f_w'] = self.water_cut
        reservoir_params['Phi'] = self.m
        reservoir_params['h'] = self.NNT
        reservoir_params['k_h'] = self.permeability
        reservoir_params['Pr'] = self.P_reservoir

        well_params['L'] = self.length_geo
        well_params['Pwf'] = self.P_well_init
        well_params['r_e'] = self.r_eff
        well_params['FracCount'] = check_FracCount(well_params['Type_Frac'],
                                                   well_params['length_FracStage'],
                                                   well_params['L'])
        self.init_Ql_rate, self.init_Qo_rate = calculate_starting_rate(reservoir_params, fluid_params,
                                                                       well_params, coefficients,
                                                                       kv_kh, Swc, Sor, Fw, m1, Fo, m2, Bw)
        logger.info(f"Для проектной скважины {self.well_number}: Q_liq = {self.init_Ql_rate},"
                    f" Q_oil = {self.init_Qo_rate}")

    def get_production_profile(self, data_decline_rate_stat, period=25 * 12, day_in_month=29, well_efficiency=0.95):
        if self.init_Qo_rate is None or self.init_Ql_rate is None:
            logger.warning(f"Проверьте расчет запускных для проектной скважины {self.well_number}!")
        else:
            # Рассчитываем средние коэффициенты скважин из окружения
            list_nearest_wells = self.gdf_nearest_wells.well_number.unique()
            list_nearest_wells = np.append(list_nearest_wells, 'default_decline_rates')
            data_decline_rate_stat = data_decline_rate_stat[data_decline_rate_stat.well_number.isin(list_nearest_wells)]
            self.decline_rates = get_avg_decline_rates(data_decline_rate_stat, self.init_Ql_rate, self.init_Qo_rate)
            # Восстанавливаем профиль для проектной скважины
            model_arps_ql = self.decline_rates[0]
            model_arps_qo = self.decline_rates[1]

            success_arps_ql = model_arps_ql[0]
            success_arps_qo = model_arps_qo[0]
            if success_arps_ql and success_arps_qo:
                rates, productions = production_model(period, model_arps_ql, model_arps_qo,
                                                      self.reserves * 1000, day_in_month, well_efficiency)
                self.Ql_rate, self.Qo_rate = rates
                self.Ql, self.Qo = productions
            else:
                logger.warning(f"Проверьте расчет среднего темпа для проектной скважины {self.well_number}!")
        pass

    def calculate_reserves(self, map_rrr, gdf_mesh, mesh_pixel):
        """Расчет ОИЗ проектной скважины, тыс.т"""

        # Создаем буфер вокруг скважины
        buffer = self.LINESTRING_geo.buffer(self.r_eff)

        # Проверка принадлежности точек буферу
        points_index = list(gdf_mesh[buffer.contains(gdf_mesh["Mesh_Points"])].index)
        array_rrr = map_rrr.data[mesh_pixel.loc[points_index, 'y_coords'], mesh_pixel.loc[points_index, 'x_coords']]
        self.reserves = np.sum(array_rrr * map_rrr.geo_transform[1] ** 2 / 10000) / 1000
        pass

    def calculate_economy(self, economy_info, start_date, period=25):

        pass


def calculate_reserves_by_voronoi(list_zones, df_fact_wells, map_rrr, save_directory=None):
    """Расчет запасов для проектных скважин с помощью ячеек Вороных"""
    df_fact_wells = (df_fact_wells[(df_fact_wells['Qo_cumsum'] > 0) |
                                   (df_fact_wells['Winj_cumsum'] > 0)].reset_index(drop=True))[['well_number',
                                                                                                'well_type',
                                                                                                'work_marker',
                                                                                                'LINESTRING_geo',
                                                                                                'length_geo',
                                                                                                'POINT_T1_geo']]
    gdf_project_wells = gpd.GeoDataFrame()
    for drill_zone in list_zones:
        if drill_zone.rating != -1:
            gdf_project_wells_zone = gpd.GeoDataFrame(
                {'well_number': [well.well_number for well in drill_zone.list_project_wells],
                 'well_type': [well.well_type for well in drill_zone.list_project_wells],
                 'LINESTRING_geo': [well.LINESTRING_geo for well in drill_zone.list_project_wells],
                 'length_geo': [well.length_geo for well in drill_zone.list_project_wells],
                 'POINT_T1_geo': [well.POINT_T1_geo for well in drill_zone.list_project_wells]}
            )
            gdf_project_wells = pd.concat([gdf_project_wells, gdf_project_wells_zone], ignore_index=True)
    # gdf со всеми скважинами: проектными и фактическими
    gdf_all_wells = gpd.GeoDataFrame(pd.concat([df_fact_wells, gdf_project_wells], ignore_index=True))
    # расчет Вороных
    gdf_all_wells[['area_voronoi', 'r_eff_voronoy']] = get_parameters_voronoi_cells(gdf_all_wells)
    # сохранение картинки Вороных
    if save_directory:
        save_picture_voronoi(gdf_all_wells, f"{save_directory}/voronoy.png",
                             type_coord="geo", default_size_pixel=1)
    # оставляем только проектные скважины
    gdf_project_wells = gdf_all_wells[gdf_all_wells['work_marker'].isna()].reset_index(drop=True)
    # Подготовка сетки точек к расчету запасов
    gdf_mesh, mesh_pixel = prepare_mesh_map_rrr(map_rrr)

    for drill_zone in list_zones:
        if drill_zone.rating != -1:
            logger.info(f"Расчет ОИЗ проектных скважин зоны: {drill_zone.rating}")
            for project_well in drill_zone.list_project_wells:
                # Радиус дренирования проектной скважины согласно Вороным
                project_well.r_eff = \
                    gdf_project_wells[gdf_project_wells['well_number'] == project_well.well_number][
                        'r_eff_voronoy'].iloc[0]
                # Запасы проектной скважины согласно Вороным
                project_well.calculate_reserves(map_rrr, gdf_mesh, mesh_pixel)
    pass
