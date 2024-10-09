from ranking_drilling.ranking_drillling import calculate_starting_rate
import numpy as np


def calculate_flow_rates(maps, dict_zones, dict_properties):
    # снимаем значения с карт
    type_map_list = list(map(lambda raster: raster.type_map, maps))

    # инициализация всех необходимых карт
    map_water_cut = maps[type_map_list.index('water_cut')]
    map_porosity = maps[type_map_list.index('porosity')]
    map_NNT = maps[type_map_list.index('NNT')]
    map_permeability = maps[type_map_list.index('permeability')]
    map_pressure = maps[type_map_list.index('pressure')]

    # задаваемые параметры
    L = 0
    r_e = 300
    xfr = 0
    w_f = 0
    FracCount = 0
    k_f = 0
    Pwf = 90
    t_p = 11
    r_w = 0.1
    S = 0
    KPPP = 1
    KUBS = 1

    # параметры из ГФХ для объекта
    c_r, mu_w, mu_o, c_o, c_w, Bo, Pb, rho = list(map(lambda key: dict_properties[key],
                                                      ['formation_compressibility',
                                                       'water_viscosity_in_situ',
                                                       'oil_viscosity_in_situ',
                                                       'oil_compressibility',
                                                       'water_compressibility',
                                                       'Bo', 'bubble_point_pressure',
                                                       'oil_density_at_surf']))

    # для каждого кластера считаем пятно запускных:
    for zone in dict_zones.keys():
        x_coord, y_coord = dict_zones[zone][0], dict_zones[zone][1]

        array_f_w = map_water_cut.data[y_coord, x_coord]
        array_Phi = map_porosity.data[y_coord, x_coord]
        array_h = map_NNT.data[y_coord, x_coord]
        array_k_h = map_permeability.data[y_coord, x_coord]
        array_Pr = map_pressure.data[y_coord, x_coord]

        array_Q_oil, array_Q_liq = [], []

        fluid_params = {'mu_w': mu_w, 'mu_o': mu_o, 'c_o': c_o, 'c_w': c_w, 'Bo': Bo, 'Pb': Pb, 'rho': rho}
        well_params = {'L': L, 'xfr': xfr, 'w_f': w_f, 'FracCount': FracCount, 'k_f': k_f, 'Pwf': Pwf, 't_p': t_p,
                       'r_e': r_e, 'r_w': r_w}
        coefficients = {'KPPP': KPPP, 'skin': S, 'KUBS': KUBS}

        for f_w, Phi, h, k_h, Pr in zip(array_f_w, array_Phi, array_h, array_k_h, array_Pr):
            reservoir_params = {'f_w': f_w, 'c_r': c_r, 'Phi': Phi, 'h': h, 'k_h': k_h, 'Pr': Pr}
            Q_oil, Q_liq = calculate_starting_rate(reservoir_params, fluid_params, well_params, coefficients)
            array_Q_oil.append(Q_oil)
            array_Q_liq.append(Q_liq)
        print(1)

    pass
