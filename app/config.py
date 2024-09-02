# Названия колонок в файле МЭР
MER_columns_name = {'Дата': 'date',
                    '№ скважины': 'well_number',
                    'Месторождение': 'field',
                    'Объект': 'object',
                    'Объекты работы': 'objects',
                    'Характер работы': 'work_marker',
                    'Состояние': 'well_status',
                    'Дебит нефти за последний месяц, т/сут': 'Qo_rate',
                    'Дебит жидкости за последний месяц, т/сут': 'Ql_rate',
                    'Добыча нефти за посл.месяц, т': 'Qo',
                    'Обводненность за посл.месяц, % (вес)': 'water_cut',
                    'Приемистость за последний месяц, м3/сут': 'Winj_rate',
                    'Закачка за посл.месяц, м3': 'Winj',
                    "Время работы, часы": 'time_work',
                    "Забойное давление (ТР), атм": 'P_well',
                    'Пластовое давление (ТР), атм': 'P_pressure',
                    "Координата X": 'T1_x',
                    "Координата Y": 'T1_y',
                    "Координата забоя Х (по траектории)": 'T3_x',
                    "Координата забоя Y (по траектории)": 'T3_y'}

dict_work_marker = {"НЕФ":	"prod",
                    "НАГ": "inj"}

# Стандартные типы карт, которые используются в модуле
list_names_map = ['NNT',  # ННТ, расчет карты ОИ
                  'permeability',  # проницаемость, расчет карты ОИ
                  'residual_recoverable_reserves',  # ОИЗ, расчет карты ОИ
                  'pressure',  # Изобары, расчет карты ОИ
                  'water_cut',  # Обводненность, расчет карты ОИ
                  'last_rate_oil',  # последний дебит нефти, расчет карты ОИ
                  'init_rate_oil',  # запускные, расчет карты ОИ
                  'initial_oil_saturation',  # начальная нефтенасыщенности, расчет карты ОИ
                  'porosity',  # Пористость, зоны продуктивности/нагнетания
                  # набор карт создаваемых в результате расчета
                  'reservoir_score',
                  'potential_score',
                  'risk_score',
                  'opportunity_index']

# Названия колонок в файле ГФХ
gpch_column_name = {'Месторождение': 'field',
                    'Тип данных': 'data_type',
                    'Объект': 'object',
                    'Пласт': 'pool',
                    'Район': 'area',
                    'Средняя глубина залегания кровли': 'top_depth',
                    'Абсолютная отметка ВНК': 'oil-water_contact_depth',
                    'Абсолютная отметка ГНК': 'gas-oil_contact_depth',
                    'Абсолютная отметка ГВК': 'gas-water_contact_depth',
                    'Тип залежи': 'type_pool',
                    'Тип коллектора': 'reservoir_type',
                    'Площадь нефтеносности': 'oil_productive_area',
                    'Площадь газоносности': 'gas_productive_area',
                    'Средняя общая толщина': 'Avg_total_h',
                    'Средняя эффективная нефтенасыщенная толщина': 'eff_oil_h',
                    'Средняя эффективная газонасыщенная толщина': 'eff_gas_h',
                    'Средняя эффективная водонасыщенная толщина': 'eff_water_h',
                    'Коэффициент пористости': 'porosity',
                    'Коэффициент нефтенасыщенности ЧНЗ': 'oil_saturation_oil_zone',
                    'Коэффициент нефтенасыщенности ВНЗ': 'oil_saturation_water_oil_zone',
                    'Коэффициент нефтенасыщенности пласта': 'total_oil_saturation',
                    'Коэффициент газонасыщенности пласта': 'total_gas_saturation',
                    'Проницаемость': 'permeability',
                    'Коэффициент песчанистости': 'gross_sand_ratio ',
                    'Расчлененность': 'stratification_factor',
                    'Начальная пластовая температура': 'init_temperature',
                    'Начальное пластовое давление': 'init_pressure',
                    'Вязкость нефти в пластовых условиях': 'oil_viscosity_in_situ',
                    'Плотность нефти в пластовых условиях': 'oil_density_in_situ',
                    'Плотность нефти в поверхностных условиях': 'oil_density_at_surf',
                    'Объемный коэффициент нефти': 'Bo',
                    'Содержание серы в нефти': 'sulphur_content',
                    'Содержание парафина в нефти': 'paraffin_content',
                    'Давление насыщения нефти газом': 'bubble_point_pressure',
                    'Газосодержание': 'gas_oil_ratio',
                    'Давление начала конденсации': 'dewpoint_pressure',
                    'Плотность конденсата в стандартных условиях': 'condensate_density_in_st',
                    'Вязкость конденсата в стандартных условиях': 'condensate_viscosity_in_st',
                    'Потенциальное содержание стабильного конденсата в газе (С5+)': 'stabilized_condensate_content_gas',
                    'Содержание сероводорода': 'hydrogen_sulfide_content',
                    'Вязкость газа в пластовых условиях': 'gas_viscosity_in_situ',
                    'Плотность газа в пластовых условиях': 'gas_density_in_situ',
                    'Коэффициент сверхсжимаемости газа': 'z_factor',
                    'Вязкость воды в пластовых условиях': 'water_viscosity_in_situ',
                    'Плотность воды в поверхностных условиях': 'water_density_at_surf',
                    'Сжимаемость': 'compressibility_del',
                    'нефти': 'oil_compressibility',
                    'воды': 'water_compressibility',
                    'породы': 'formation_compressibility',
                    'Коэффициент вытеснения (водой)': 'water_flood_displacement_efficiency',
                    'Коэффициент вытеснения (газом)': 'gas_flood_displacement_efficiency',
                    'Коэффициент продуктивности': 'productivity_factor',
                    'Коэффициенты фильтрационных сопротивлений:': 'flow_coefficient_del',
                    'А': 'flow_coefficient_A',
                    'В': 'flow_coefficient_B'}
