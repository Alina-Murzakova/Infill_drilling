import geopandas as gpd
import pandas as pd
import numpy as np

from loguru import logger

from app.decline_rate.decline_rate import get_avg_decline_rates
from app.well_active_zones import get_value_map
from app.ranking_drilling.starting_rates import calculate_starting_rate


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
        self.Ql = None
        self.Qo = None
        self.decline_rates = None

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
        self.permeability = gdf_fact_wells.loc[nearest_hor_wells_index]
        pass

    def get_params_nearest_wells(self):
        """Получаем параметры для проектной скважины с ближайших фактических"""
        # Проверка на наличие ближайшего окружения
        if self.gdf_nearest_wells.empty:
            logger.warning(f"Проверьте наличие окружения для проектной скважины {self.well_number}!")
        # Расчет средних параметров по выбранному окружению
        self.P_well_init = np.mean(self.gdf_nearest_wells['init_P_well_prod'])
        self.permeability = np.mean(self.gdf_nearest_wells['permeability_fact'])
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
        self.Ql, self.Qo = calculate_starting_rate(reservoir_params, fluid_params, well_params, coefficients)
        logger.info(f"Для проектной скважины {self.well_number}: Q_liq = {self.Ql}, Q_oil = {self.Qo}")

    def get_production_profile(self, data_decline_rate_stat, period=25*12):
        if self.Qo is None or self.Ql is None:
            logger.warning(f"Проверьте расчет запускных для проектной скважины {self.well_number}!")
        else:
            # Рассчитываем средние коэффициенты скважин из окружения
            list_nearest_wells = self.gdf_nearest_wells.well_number.unique()
            data_decline_rate_stat = data_decline_rate_stat[data_decline_rate_stat.well_number.is_in(list_nearest_wells)]
            self.decline_rates = get_avg_decline_rates(data_decline_rate_stat, self.Ql, self.Qo)
            # Восстанавливаем профиль для проектной скважины
            model_arps_ql = self.decline_rates[0]
            model_arps_qo = self.decline_rates[1]

            success_arps_ql = model_arps_ql[0]
            success_arps_qo = model_arps_qo[0]
            if success_arps_ql and success_arps_qo:
                print(1)
            else:
                logger.warning(f"Проверьте среднего темпа для проектной скважины {self.well_number}!")
        pass


def save_ranking_drilling_to_excel(list_zones, filename):
    gdf_result = gpd.GeoDataFrame()
    for drill_zone in list_zones:
        if drill_zone.rating != -1:
            gdf_project_wells = gpd.GeoDataFrame(
                {'well_number': [well.well_number for well in drill_zone.list_project_wells],
                 'well_type': [well.well_type for well in drill_zone.list_project_wells],
                 'length': [well.length_geo for well in drill_zone.list_project_wells],
                 'water_cut': [well.water_cut for well in drill_zone.list_project_wells],
                 'Q_liq': [well.Ql for well in drill_zone.list_project_wells],
                 'Q_oil': [well.Qo for well in drill_zone.list_project_wells]}
            )
            gdf_result = pd.concat([gdf_result, gdf_project_wells], ignore_index=True)
    gdf_result.to_excel(filename, sheet_name='РБ', index=False)
    pass
