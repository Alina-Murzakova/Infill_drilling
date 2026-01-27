import geopandas as gpd
import pandas as pd
import numpy as np

from loguru import logger
from copy import deepcopy
from app.decline_rate.decline_rate import get_avg_decline_rates
from app.input_output.output_functions import save_picture_voronoi
from app.well_active_zones import get_parameters_voronoi_cells

from app.decline_rate.functions import production_model
from app.well_active_zones import get_value_map
from app.ranking_drilling.starting_rates import calculate_starting_rate
from app.input_output.input_frac_info import check_FracCount
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
        self.So_init = None  # начальная нефтенасыщенность
        self.So = None  # текущая нефтенасыщенность
        self.water_cut = None
        self.m = None
        self.permeability = None
        self.init_Ql_rate = None  # т/сут
        self.init_Ql_rate_V = None  # м3/сут
        self.init_Qo_rate = None
        self.decline_rates = None
        self.r_eff = None
        self.reserves = None  # тыс. т

        # Профили
        self.Qo_rate = None
        self.Ql_rate = None
        self.Qo = None
        self.Ql = None

        # Экономика
        self.cumulative_cash_flow = None  # Накопленный поток наличности
        self.CAPEX = None  # CAPEX
        self.OPEX = None  # OPEX
        self.NPV = None
        self.PVI = None
        self.PI = None
        self.year_economic_limit = None

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
        gdf_fact_wells = gpd.GeoDataFrame(df_wells, geometry="LINESTRING_pix")

        # Если требуется GeoDataFrame со скважинами ближайшего окружения того же типа
        # gdf_fact_wells = gdf_fact_wells[gdf_fact_wells["well_type"] == self.well_type].reset_index(drop=True)
        # if gdf_fact_wells.empty:
        #     logger.warning(f"На объекте нет фактических скважин типа {self.well_type}! \n "
        #                    "Необходимо задать азимут, длину, Рзаб проектных скважин вручную.")

        # Вычисляем расстояния до всех скважин
        distances = self.LINESTRING_pix.distance(gdf_fact_wells["LINESTRING_pix"])
        gdf_fact_wells['distances'] = distances
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
        mask_distance = (self.gdf_nearest_wells['distances'] != 0) & self.gdf_nearest_wells['distances'].notna()
        # Рзаб - Выбираем только те строки, где значение Рзаб больше 0
        mask = ((self.gdf_nearest_wells['init_P_well_prod'] > 0) &
                self.gdf_nearest_wells['init_P_well_prod'].notna() & mask_distance)
        if sum(mask) == 0:
            self.P_well_init = dict_parameters_coefficients['well_params']["proj_wells_params"]['all_P_wells_init']
        else:
            self.P_well_init = np.average(self.gdf_nearest_wells.loc[mask, 'init_P_well_prod'],
                                          weights=1 / np.square(self.gdf_nearest_wells.loc[mask, 'distances']))
        # Проницаемость - Выбираем только те строки, где значение проницаемости больше 0 и не nan
        mask = ((self.gdf_nearest_wells['permeability_fact'] > 0) &
                self.gdf_nearest_wells['permeability_fact'].notna() & mask_distance)
        if sum(mask) == 0:
            self.permeability = dict_parameters_coefficients['reservoir_fluid_properties']['k_h']
        else:
            self.permeability = np.average(self.gdf_nearest_wells.loc[mask, 'permeability_fact'],
                                           weights=1 / np.square(self.gdf_nearest_wells.loc[mask, 'distances']))
        # Пористость на случай если в зоне нет карты
        # выбираем только те строки, где значение пористости больше 0 и не nan
        mask = ((self.gdf_nearest_wells['m'] > 0) & self.gdf_nearest_wells['m'].notna() & mask_distance)
        if not sum(mask) == 0:
            self.m = np.average(self.gdf_nearest_wells.loc[mask, 'm'],
                                weights=1 / np.square(self.gdf_nearest_wells.loc[mask, 'distances']))

        # Если установлен switch_wc_from_map = False (расчет обв-ти с окружения)
        if not dict_parameters_coefficients['switches']['switch_wc_from_map']:
            # Обводненность - Выбираем только те скважины, которые остановлены не более 10 лет назад
            mask = (self.gdf_nearest_wells['no_work_time'] <= 12 * 10) & (self.gdf_nearest_wells['Ql_rate'] > 0)
            if not sum(mask) == 0:
                self.water_cut = np.average(self.gdf_nearest_wells.loc[mask, 'water_cut_V'],
                                            weights=1 / np.square(self.gdf_nearest_wells.loc[mask, 'distances']))
            else:
                self.water_cut = 100
                logger.error(f"У скважины {self.well_number}, отсутствует окружение для актуального расчета "
                             f"обводненности!"
                             f"\nЗаданное значение обводненности 100%,"
                             f" скважины окружения:{self.gdf_nearest_wells.well_number.to_list()}")
        pass

    def get_params_maps(self, maps):
        """Значения с карт для скважины"""
        type_map_list = list(map(lambda raster: raster.type_map, maps))
        list_arguments = (self.well_type, self.POINT_T1_pix.x, self.POINT_T1_pix.y,
                          self.POINT_T3_pix.x, self.POINT_T3_pix.y, self.length_pix)
        self.P_reservoir = get_value_map(*list_arguments, raster=maps[type_map_list.index('pressure')])
        self.NNT = get_value_map(*list_arguments, raster=maps[type_map_list.index('NNT')])
        # !!! Если в зоне нет карты пористости, то берется с окружения.
        # Либо изначально в OI добавить фильтр в т.ч. по пористости
        if get_value_map(*list_arguments, raster=maps[type_map_list.index('porosity')]) != 0:
            self.m = get_value_map(*list_arguments, raster=maps[type_map_list.index('porosity')])
        self.So_init = get_value_map(*list_arguments, raster=maps[type_map_list.index('initial_oil_saturation')])
        if pd.isna(self.water_cut):
            self.water_cut = get_value_map(*list_arguments, raster=maps[type_map_list.index('water_cut')])
        pass

    def get_starting_rates(self, maps, dict_parameters_coefficients):
        self.get_params_maps(maps)

        # Создаем локальную копию
        local_dict = deepcopy(dict_parameters_coefficients)

        kv_kh, Swc, Sor, Fw, m1, Fo, m2, Bw = (
            list(map(lambda name: local_dict['reservoir_fluid_properties'][name],
                     ['kv_kh', 'Swc', 'Sor', 'Fw', 'm1', 'Fo', 'm2', 'Bw'])))

        reservoir_params = local_dict['reservoir_fluid_properties']
        fluid_params = local_dict['reservoir_fluid_properties']
        well_params = (local_dict['well_params']['general'] | local_dict['well_params']["fracturing"]
                       | local_dict['well_params']["proj_wells_params"])
        coefficients = local_dict['well_params']['general']

        # Проверка на отрицательную депрессию и фиксированную забойку:
        if (self.P_reservoir - self.P_well_init) < 0 or local_dict['switches']['switch_fix_P_well_init']:
            self.P_well_init = well_params['fix_P_well_init']
            if (self.P_reservoir - self.P_well_init) < 0:
                self.P_well_init = 0.4 * self.P_reservoir

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
                                                   well_params['L'],
                                                   self.well_type)
        self.init_Ql_rate_V, self.init_Qo_rate, self.So = calculate_starting_rate(reservoir_params, fluid_params,
                                                                                  well_params, coefficients,
                                                                                  kv_kh, Swc, Sor, Fw, m1, Fo, m2, Bw)
        self.init_Ql_rate = self.init_Ql_rate_V * (
                self.water_cut / 100 * 1 + (1 - self.water_cut / 100) * fluid_params['rho'])
        logger.info(f"Для проектной скважины {self.well_number}: Q_liq = {round(self.init_Ql_rate_V, 2)} м3/сут,"
                    f" Q_oil = {round(self.init_Qo_rate, 2)} т/сут")
        pass

    def get_production_profile(self, data_decline_rate_stat, period=25 * 12, day_in_month=29, well_efficiency=0.95):
        if self.init_Qo_rate is None or self.init_Ql_rate is None:
            logger.warning(f"Проверьте расчет запускных для проектной скважины {self.well_number}!")
        else:
            # Рассчитываем средние коэффициенты скважин из окружения
            list_nearest_wells = self.gdf_nearest_wells.well_number.unique()
            list_nearest_wells = np.append(list_nearest_wells, 'default_decline_rates')
            data_decline_rate_stat = data_decline_rate_stat[data_decline_rate_stat.well_number.isin(list_nearest_wells)]
            logger.info(f"Оценка темпа падения для проектной скважины {self.well_number}")
            self.decline_rates = get_avg_decline_rates(data_decline_rate_stat, self.init_Ql_rate, self.init_Qo_rate)
            logger.info(f"Восстанавливаем профиль для проектной скважины {self.well_number}")
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
        logger.info(f"Расчет ОИЗ для проектной скважины {self.well_number}")
        # Создаем буфер вокруг скважины
        buffer = self.LINESTRING_geo.buffer(self.r_eff)

        # Проверка принадлежности точек буферу
        points_index = list(gdf_mesh[buffer.contains(gdf_mesh["Mesh_Points"])].index)
        array_rrr = map_rrr.data[mesh_pixel.loc[points_index, 'y_coords'], mesh_pixel.loc[points_index, 'x_coords']]
        self.reserves = np.sum(array_rrr * map_rrr.geo_transform[1] ** 2 / 10000) / 1000
        pass

    def calculate_economy(self, FEM, well_params, method, dict_NDD):
        logger.info(f"Расчет экономики для проектной скважины {self.well_number}")
        start_date = well_params['start_date']
        self.CAPEX, self.OPEX, self.cumulative_cash_flow, self.NPV, self.PVI, self.PI, self.year_economic_limit = (
            FEM.calculate_economy_well(self.Qo, self.Ql, start_date, self.well_type, well_params, method, dict_NDD))
        pass


def calculate_reserves_by_voronoi(list_zones, df_fact_wells, map_rrr, save_directory=None, max_radius_project_well=500):
    """Расчет запасов для проектных скважин с помощью ячеек Вороных"""
    df_fact_wells = (df_fact_wells[(df_fact_wells['Qo_cumsum'] > 0) |
                                   (df_fact_wells['Winj_cumsum'] > 0)].reset_index(drop=True))[['well_number',
                                                                                                'well_type',
                                                                                                'work_marker',
                                                                                                'LINESTRING_geo',
                                                                                                'length_geo',
                                                                                                'POINT_T1_geo',
                                                                                                'well_number_digit',
                                                                                                'type_wellbore']]
    gdf_project_wells = gpd.GeoDataFrame()
    for drill_zone in list_zones:
        if drill_zone.rating != -1:
            gdf_project_wells_zone = gpd.GeoDataFrame(
                {'well_number': [well.well_number for well in drill_zone.list_project_wells],
                 'well_type': [well.well_type for well in drill_zone.list_project_wells],
                 'LINESTRING_geo': [well.LINESTRING_geo for well in drill_zone.list_project_wells],
                 'length_geo': [well.length_geo for well in drill_zone.list_project_wells],
                 'POINT_T1_geo': [well.POINT_T1_geo for well in drill_zone.list_project_wells],
                 'well_number_digit': [well.well_number for well in drill_zone.list_project_wells],
                 'type_wellbore': ['Материнский ствол'] * len(drill_zone.list_project_wells)}
            )
            gdf_project_wells = pd.concat([gdf_project_wells, gdf_project_wells_zone], ignore_index=True)
    # gdf со всеми скважинами: проектными и фактическими
    gdf_all_wells = gpd.GeoDataFrame(pd.concat([df_fact_wells, gdf_project_wells], ignore_index=True))
    # расчет Вороных
    df_parameters_voronoi = get_parameters_voronoi_cells(gdf_all_wells)
    gdf_all_wells = pd.merge(gdf_all_wells, df_parameters_voronoi, on='well_number', how='left')
    # сохранение картинки Вороных
    if save_directory:
        save_picture_voronoi(gdf_all_wells, f"{save_directory}", type_coord="geo", default_size_pixel=1)
    # оставляем только проектные скважины
    gdf_project_wells = gdf_all_wells[gdf_all_wells['work_marker'].isna()].reset_index(drop=True)
    # Подготовка сетки точек к расчету запасов
    gdf_mesh, mesh_pixel = prepare_mesh_map_rrr(map_rrr)

    for drill_zone in list_zones:
        if drill_zone.rating != -1:
            logger.info(f"Расчет ОИЗ проектных скважин зоны: {drill_zone.rating}")
            for project_well in drill_zone.list_project_wells:
                # Радиус дренирования проектной скважины согласно Вороным или max_radius_project_well м, если он больше
                project_well.r_eff = min(
                    gdf_project_wells[gdf_project_wells['well_number'] ==
                                      project_well.well_number]['r_eff_voronoy'].iloc[0], max_radius_project_well)
                # Запасы проектной скважины согласно Вороным
                project_well.calculate_reserves(map_rrr, gdf_mesh, mesh_pixel)
    pass
