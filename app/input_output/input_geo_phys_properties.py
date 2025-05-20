import os
import pandas as pd

from loguru import logger

from app.config import gpch_column_name


def load_geo_phys_properties(path_geo_phys_properties, name_field, name_object):
    """Создание словаря ГФХ для пласта"""
    # Загрузка файла
    df_geo_phys_properties = pd.read_excel(os.path.join(os.path.dirname(__file__), path_geo_phys_properties))
    # Переименование колонок
    df_geo_phys_properties = df_geo_phys_properties[list(gpch_column_name.keys())]
    df_geo_phys_properties.columns = list(map(lambda x: list(x.keys())[0], gpch_column_name.values()))
    # Подготовка файла
    df_geo_phys_properties = df_geo_phys_properties.fillna(0)
    df_geo_phys_properties.drop([0], inplace=True)  # Удаление строки с ед. изм.

    list_dict = gpch_column_name.values()
    df_geo_phys_properties = df_geo_phys_properties.astype(
        {k: v for list_item in list_dict for (k, v) in list_item.items()})

    # Удаление лишних столбцов
    list_columns = []
    for column in df_geo_phys_properties.columns:
        if 'del' in column:
            del df_geo_phys_properties[column]
        else:
            list_columns.append(column)

    df_geo_phys_properties = df_geo_phys_properties[df_geo_phys_properties.data_type == "в целом"]

    # добавляем строки со значениями по умолчанию (среднее по мр) для каждого месторождения
    dct = {'number': 'mean', 'object': lambda col: col.mode(), }
    groupby_cols = ['field']
    dct = {k: v for i in
           [{col: agg for col in df_geo_phys_properties.select_dtypes(tp).columns.difference(groupby_cols)} for tp, agg
            in dct.items()] for
           k, v in i.items()}
    agg = df_geo_phys_properties.groupby(groupby_cols).agg(**{k: (k, v) for k, v in dct.items()})
    agg['object'] = "default_properties"
    agg['field'] = agg.index

    df_geo_phys_properties = pd.concat([agg.reset_index(drop=True), df_geo_phys_properties])
    df_geo_phys_properties = df_geo_phys_properties[list_columns]

    df_geo_phys_properties_field_mean = df_geo_phys_properties[(df_geo_phys_properties.field == name_field)
                                                               & (df_geo_phys_properties.object ==
                                                                  "default_properties")]
    if df_geo_phys_properties_field_mean.empty:
        logger.error(f"В файле ГФХ нет данных по месторождению {name_field}")
        return None

    df_geo_phys_properties_field = df_geo_phys_properties[(df_geo_phys_properties.field == name_field)
                                                          & (df_geo_phys_properties.object == name_object)]
    if df_geo_phys_properties_field.shape[0] > 1:
        logger.error(f"В файле ГФХ больше одной строчки для объекта {name_field} месторождения {name_field}")
        return None
    else:
        if df_geo_phys_properties_field.empty:
            logger.info(f"В файле ГФХ не найден объект {name_field} месторождения {name_field}. "
                        f"Используются средние значения по месторождению для объекта.")
            dict_geo_phys_properties_field = df_geo_phys_properties_field_mean.iloc[0][5:].to_dict()
        else:
            dict_geo_phys_properties_field = df_geo_phys_properties_field.iloc[0][5:].to_dict()

        # Проверка наличия требуемых свойств
        list_properties = ['formation_compressibility', 'water_viscosity_in_situ',
                           'oil_viscosity_in_situ', 'oil_compressibility',
                           'water_compressibility', 'Bo', 'bubble_point_pressure',
                           'oil_density_at_surf', 'gas_oil_ratio']
        # для 'init_pressure' есть проверка при построении карты рисков, если оно 0,
        # то используется максимальное значение с карты
        for prop in list_properties:
            value = dict_geo_phys_properties_field[prop]
            value_mean = df_geo_phys_properties_field_mean[prop].iloc[0]
            if value <= 0:
                if value_mean > 0:
                    dict_geo_phys_properties_field[prop] = value_mean
                else:
                    logger.error(f"Свойство {prop} задано некорректно: {value}")
        return formatting_dict_geo_phys_properties(dict_geo_phys_properties_field)


def formatting_dict_geo_phys_properties(dict_geo_phys_properties):
    """
    Формирование словаря со всеми необходимыми свойствами из ГФХ в требуемых размерностях

    - reservoir_params:
    c_r - сжимаемость породы | (1/МПа)×10-4 --> 1/атм
    P_init - начальное пластовое давление | МПа --> атм
    k_h - проницаемость | мД

    - fluid_params:
    mu_w - вязкость воды | сП или мПа*с
    mu_o - вязкость нефти | сП или мПа*с
    c_o - сжимаемость нефти | (1/МПа)×10-4 --> 1/атм
    c_w - сжимаемость воды | (1/МПа)×10-4 --> 1/атм
    Bo - объемный коэффициент расширения нефти | м3/м3
    Pb - давление насыщения | МПа --> атм
    rho - плотность нефти | г/см3
    gor - Газосодержание| м3/т
    """
    return {'reservoir_params': {'c_r': dict_geo_phys_properties['formation_compressibility'] / 100000,
                                 'P_init': dict_geo_phys_properties['init_pressure'] * 10,
                                 'k_h': dict_geo_phys_properties['permeability']},
            'fluid_params': {'mu_w': dict_geo_phys_properties['water_viscosity_in_situ'],
                             'mu_o': dict_geo_phys_properties['oil_viscosity_in_situ'],
                             'c_o': dict_geo_phys_properties['oil_compressibility'] / 100000,
                             'c_w': dict_geo_phys_properties['water_compressibility'] / 100000,
                             'Bo': dict_geo_phys_properties['Bo'],
                             'Pb': dict_geo_phys_properties['bubble_point_pressure'] * 10,
                             'rho': dict_geo_phys_properties['oil_density_at_surf'],
                             'gor': dict_geo_phys_properties['gas_oil_ratio']}}