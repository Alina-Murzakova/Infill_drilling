from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MapSpec:
    key: str
    label: str
    filename: str
    required: bool
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class GridChoice:
    """A lightweight grid descriptor; the potentially huge data array is loaded later."""

    locator: str
    map_key: str
    name: str
    grid_type: str
    date: str
    field: str
    object_name: str

    @property
    def display_name(self) -> str:
        parts = [self.date or "дата не указана", self.name or self.grid_type]
        return " — ".join(parts)


# The calculation engine requires the first five grids. The last two are generated
# by the reservoir-state model when they are not supplied.
MAP_SPECS: tuple[MapSpec, ...] = (
    MapSpec(
        "NNT",
        "Начальная нефтенасыщенная толщина (ННТ)",
        "NNT.grd",
        True,
        ("нефтенасыщенн", "толщин"),
    ),
    MapSpec(
        "initial_oil_saturation",
        "Начальная нефтенасыщенность",
        "initial_oil_saturation.grd",
        True,
        ("начальн", "нефтенасыщенност"),
    ),
    MapSpec(
        "permeability",
        "Проницаемость",
        "permeability.grd",
        True,
        ("проницаемост",),
    ),
    MapSpec(
        "porosity",
        "Пористость",
        "porosity.grd",
        True,
        ("пористост",),
    ),
    MapSpec(
        "pressure",
        "Давление",
        "pressure.grd",
        True,
        ("давлен",),
    ),
    MapSpec(
        "residual_recoverable_reserves",
        "Остаточные извлекаемые запасы (ОИЗ)",
        "residual_recoverable_reserves.grd",
        False,
        ("запас",),
    ),
    MapSpec(
        "water_cut",
        "Обводнённость",
        "water_cut.grd",
        False,
        ("обводненност",),
    ),
)


def classify_grid_type(grid_type: str) -> str | None:
    normalized = str(grid_type).strip().casefold().replace("ё", "е")
    for spec in MAP_SPECS:
        if all(keyword in normalized for keyword in spec.keywords):
            # Thickness must not be mistaken for saturation: both contain the
            # word "нефтенасыщ...".
            if spec.key == "initial_oil_saturation" and "толщин" in normalized:
                continue
            return spec.key
    return None


PVT_FIELDS = {
    "formation_compressibility": ("rockcompressibility", 0.4),
    "init_pressure": ("initialpressure", 22.5),
    "permeability": ("permeability", 50.0),
    "water_viscosity_in_situ": ("waterviscosityipc", 0.5),
    "oil_viscosity_in_situ": ("oilviscosityipc", 2.0),
    "oil_compressibility": ("oilcompressibility", 10.0),
    "water_compressibility": ("watercompressibility", 4.2),
    "Bo": ("conversioncoefficientoil", 1.2),
    "bubble_point_pressure": ("saturationpressure", 10.0),
    "oil_density_at_surf": ("oildensitysc", 0.84),
    "gas_oil_ratio": ("gasfactor", 60.0),
}


PVT_RELATIVE_FIELDS = {
    "Sor": ("residualoilsaturation", 0.26),
    "Swc": ("connatewatersaturation", 0.40),
    "Fo": ("relativeoilpermeability", 1.0),
    "Fw": ("relativewaterpermeability", 0.20),
    "Bw": ("conversioncoefficientwater", 1.01),
}


# According to ``Соотнесение данных.xlsx``, EDW stores pressures in
# atmospheres. The existing calculation form accepts MPa and later converts
# those values back to atmospheres, so normalize them at the repository edge.
PVT_EDW_FACTORS = {
    "init_pressure": 0.1,
    "bubble_point_pressure": 0.1,
}


PVT_UI_FIELDS = {
    "formation_compressibility": ("Сжимаемость породы, 1/МПа × 10⁻⁴", 0.001, 1000.0),
    "init_pressure": ("Начальное пластовое давление, МПа", 0.001, 1000.0),
    "permeability": ("Проницаемость, мД", 0.001, 1_000_000.0),
    "water_viscosity_in_situ": ("Вязкость воды, мПа·с", 0.001, 1000.0),
    "oil_viscosity_in_situ": ("Вязкость нефти, мПа·с", 0.001, 100_000.0),
    "oil_compressibility": ("Сжимаемость нефти, 1/МПа × 10⁻⁴", 0.001, 1000.0),
    "water_compressibility": ("Сжимаемость воды, 1/МПа × 10⁻⁴", 0.001, 1000.0),
    "Bo": ("Объёмный коэффициент нефти, м³/м³", 0.001, 10.0),
    "bubble_point_pressure": ("Давление насыщения, МПа", 0.001, 1000.0),
    "oil_density_at_surf": ("Плотность нефти, г/см³", 0.001, 2.0),
    "gas_oil_ratio": ("Газосодержание, м³/т", 0.001, 100_000.0),
}
