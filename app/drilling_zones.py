from loguru import logger

from map import Map

from local_parameters import paths
from input_output import load_wells_data

"""________БЛОК ДЛЯ УДАЛЕНИЯ_______"""
data_well_directory = paths["data_well_directory"]
save_directory = paths["save_directory"]
_, data_wells = load_wells_data(data_well_directory=data_well_directory)
"""________БЛОК ДЛЯ УДАЛЕНИЯ_______"""


def calculate_zones(maps):
    type_map_list = list(map(lambda raster: raster.type_map, maps))

    map_NNT = maps[type_map_list.index("NNT")]
    map_permeability = maps[type_map_list.index("permeability")]
    map_residual_recoverable_reserves = maps[type_map_list.index("residual_recoverable_reserves")]
    map_pressure = maps[type_map_list.index("pressure")]
    map_initial_oil_saturation = maps[type_map_list.index("initial_oil_saturation")]

    map_water_cut = maps[type_map_list.index("water_cut")]
    map_last_rate_oil = maps[type_map_list.index("last_rate_oil")]
    map_init_rate_oil = maps[type_map_list.index("init_rate_oil")]

    logger.info("Расчет карты оценки пласта")
    map_reservoir_score = reservoir_score(map_NNT, map_permeability)

    logger.info("Расчет карты оценки показателей разработки")
    map_potential_score = potential_score(map_residual_recoverable_reserves, map_pressure)

    logger.info("Расчет карты оценки проблем")
    map_risk_score = risk_score(map_water_cut, map_initial_oil_saturation)

    logger.info("Расчет карты индекса возможностей")
    map_opportunity_index = opportunity_index(map_reservoir_score, map_potential_score, map_risk_score)

    pass


def reservoir_score(map_NNT, map_permeability) -> Map:
    """
    Оценка пласта
    Parameters
    ----------
    map_NNT - карта ННТ
    map_permeability  - карта проницаемости

    Returns
    -------
    Map(type_map=reservoir_score)
    """

    norm_map_NNT = map_NNT.normalize_data()
    norm_map_permeability = map_permeability.normalize_data()

    norm_map_NNT.save_img(f"{save_directory}/norm_map_NNT.png", data_wells)
    norm_map_permeability.save_img(f"{save_directory}/norm_map_permeability.png", data_wells)

    # data_reservoir_score = (norm_map_NNT.data * norm_map_permeability.data) ** 1/2
    data_reservoir_score = (norm_map_NNT.data + norm_map_permeability.data) / 2

    map_reservoir_score = Map(data_reservoir_score,
                              norm_map_NNT.geo_transform,
                              norm_map_NNT.projection,
                              "reservoir_score")
    map_reservoir_score = map_reservoir_score.normalize_data()
    map_reservoir_score.save_img(f"{save_directory}/norm_map_reservoir_score.png", data_wells)

    return map_reservoir_score


def potential_score(map_residual_recoverable_reserves, map_pressure) -> Map:

    P_init = 40 * 9.87  # атм

    map_delta_P = Map(P_init - map_pressure.data, map_pressure.geo_transform, map_pressure.projection,
                      type_map="delta_P")
    norm_residual_recoverable_reserves = map_residual_recoverable_reserves.normalize_data()
    pass


def risk_score(map_water_cut, map_initial_oil_saturation) -> Map:
    pass


def opportunity_index(map_reservoir_score, map_potential_score, map_risk_score) -> Map:
    pass
