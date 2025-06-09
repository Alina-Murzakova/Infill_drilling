import pickle
import numpy as np
import pandas as pd
from loguru import logger
import ast

from app.local_parameters import main_parameters, constants
from app.input_output.input import load_wells_data, load_geo_phys_properties, load_frac_info

from app.decline_rate.decline_rate import get_decline_rates
from app.maps_handler.functions import mapping
from app.input_output.output import get_save_path, upload_data

if __name__ == '__main__':
    logger.add(r'D:\Work\Programs_Python\Infill_drilling\app\logs.log', mode='w')

    logger.info("Инициализация локальных переменных")
    # Пути
    paths = main_parameters['paths']
    # Параметры расчета
    parameters_calculation = main_parameters['parameters_calculation']
    # Параметры для скважин РБ
    well_params = main_parameters['well_params']

    # Константы расчета
    load_data_param = constants['load_data_param']
    default_coefficients = constants['default_coefficients']
    default_well_params = constants['default_well_params']
    if constants['default_project_well_params']['buffer_project_wells'] <= 0:
        # нижнее ограничение на расстояние до фактических скважин от проектной
        constants['default_project_well_params']['buffer_project_wells'] = 10
    well_params.update(constants['default_project_well_params'])

    logger.info("Загрузка скважинных данных")
    (data_history, data_wells,
     info_object_calculation) = load_wells_data(data_well_directory=paths["data_well_directory"],
                                                first_months=load_data_param['first_months'])
    name_field, name_object = info_object_calculation.get("field"), info_object_calculation.get("object_value")
    save_directory = get_save_path("Infill_drilling", name_field, name_object.replace('/', '-'))

    logger.info(f"Загрузка ГФХ по пласту {name_object.replace('/', '-')} месторождения {name_field}")
    dict_parameters_coefficients = load_geo_phys_properties(paths["path_geo_phys_properties"], name_field, name_object)
    dict_parameters_coefficients.update({'well_params': well_params,
                                         'default_well_params': default_well_params,
                                         'coefficients': default_coefficients})

    if dict_parameters_coefficients['well_params']['switch_avg_frac_params']:
        logger.info(f"Загрузка фрак-листов")
        data_wells, dict_parameters_coefficients = load_frac_info(paths["path_frac"], data_wells, name_object,
                                                                  dict_parameters_coefficients)

    logger.info("Загрузка и обработка карт")
    _, data_wells, maps_to_calculate = mapping(maps_directory=paths["maps_directory"],
                                               data_wells=data_wells,
                                               dict_properties=dict_parameters_coefficients['reservoir_params'],
                                               **load_data_param)
    # default_size_pixel = maps[0].geo_transform[1]  # размер ячейки после загрузки всех карт
    # type_map_list = list(map(lambda raster: raster.type_map, maps))

    logger.info("Оценка темпов падения для текущего фонда")
    data_decline_rate_stat, df_initial, _ = get_decline_rates(data_history, data_wells)


    def unpack_list(value):
        # Функция для распаковки списка
        if isinstance(value, str):  # Если данные считались как строка
            value = ast.literal_eval(value)  # Преобразуем в список

        return pd.Series([value[0], value[1][0], value[1][1], value[2], value[3], value[4]])


    # Разворачиваем столбцы
    df_qi = data_decline_rate_stat["coefficients_Ql_rate"].apply(unpack_list)
    df_qo = data_decline_rate_stat["coefficients_Qo_rate"].apply(unpack_list)

    # Переименовываем столбцы
    df_qi.columns = ["Ql_success", "Ql_k1", "Ql_k2", "Ql_value", "Ql_iter", "Ql_status"]
    df_qo.columns = ["Qo_success", "Qo_k1", "Qo_k2", "Qo_value", "Qo_iter", "Qo_status"]

    # Объединяем с оригинальной таблицей
    df_final = pd.concat([data_decline_rate_stat[["well_number", "cumulative_oil_production"]], df_qi, df_qo], axis=1)

    # Вычисляем средние значения
    k1_mean_Qo = df_final.loc[df_final.well_number == 'default_decline_rates']["Qo_k1"].values[0]
    k2_mean_Qo = df_final.loc[df_final.well_number == 'default_decline_rates']["Qo_k2"].values[0]
    k1k2_mean_Qo = k1_mean_Qo * k2_mean_Qo
    k1_mean_Ql = df_final.loc[df_final.well_number == 'default_decline_rates']["Ql_k1"].values[0]
    k2_mean_Ql = df_final.loc[df_final.well_number == 'default_decline_rates']["Ql_k2"].values[0]
    k1k2_mean_Ql = k1_mean_Ql * k2_mean_Ql
    Ql_value_mean = df_final.loc[df_final.well_number == 'default_decline_rates']["Ql_value"].values[0]
    Qo_value_mean = df_final.loc[df_final.well_number == 'default_decline_rates']["Qo_value"].values[0]

    df_final['> среднего темпа Ql'] = ((df_final.Ql_k1 * df_final.Ql_k2 < k1k2_mean_Ql)
                                       & (df_final.Ql_k2 > k2_mean_Ql))
    df_final['> среднего темпа Qo'] = ((df_final.Qo_k1 * df_final.Qo_k2 < k1k2_mean_Qo)
                                       & (df_final.Qo_k2 > k2_mean_Qo))


    def f(x, k1, k2, Qst):
        return Qst * (1 + k1 * k2 * x) ** (-1 / k2)


    percent_low = 60
    n = 120
    x_values = np.array(list(range(0, n, 1)))  # Разбиваем интервал на 100 точек
    # Вычисляем среднюю линию
    f_mean_values_Qo = f(x_values, k1_mean_Qo, k2_mean_Qo, Qo_value_mean)
    f_mean_values_Ql = f(x_values, k1_mean_Ql, k2_mean_Ql, Ql_value_mean)

    df_final[f'> среднего темпа Ql за период {n} месяцев'] = df_final.apply(lambda x: np.all(f(x_values, x.Ql_k1,
                                                                                               x.Ql_k2, Ql_value_mean)
                                                                                             >= f_mean_values_Ql),
                                                                            axis=1)
    df_final[f'> среднего темпа Qo за период {n} месяцев'] = df_final.apply(lambda x: np.all(f(x_values, x.Qo_k1,
                                                                                               x.Qo_k2,
                                                                                               Qo_value_mean) >=
                                                                                             f_mean_values_Qo), axis=1)

    threshold_value_Qo = np.nanpercentile(df_final.loc[df_final.well_number != 'default_decline_rates']["Qo_value"],
                                          percent_low)
    threshold_value_Ql = np.nanpercentile(df_final.loc[df_final.well_number != 'default_decline_rates']["Ql_value"],
                                          percent_low)

    df_final[f'Стартовый дебит Qo 40% лучших'] = df_final.Qo_value > threshold_value_Qo
    df_final[f'Стартовый дебит Ql 40% лучших'] = df_final.Ql_value > threshold_value_Ql

    logger.info("Сохранения файла")
    name_field = name_field.replace('/', "_")
    name_object = name_object.replace('/', "_")
    filename = f"{save_directory}/темпы_падения_{name_field}_{name_object}.xlsx"
    with pd.ExcelWriter(filename) as writer:
        df_final.to_excel(writer, index=False)
    pass

    logger.info("Сохранение pickle файлов")
    with open(f'{save_directory}/data_decline_rate.pkl', 'wb') as file:
        pickle.dump(data_decline_rate_stat, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'{save_directory}/data_history_decline_rate.pkl', 'wb') as file:
        pickle.dump(df_initial, file, protocol=pickle.HIGHEST_PROTOCOL)
