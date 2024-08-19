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
                    'Обводненность за посл.месяц, % (вес)': 'water_cut',
                    'Приемистость за последний месяц, м3/сут': 'Winj_rate',
                    "Время работы, часы": 'time_work',
                    "Забойное давление (ТР), атм": 'P_well',
                    'Пластовое давление (ТР), атм': 'P_pressure',
                    "Координата X": 'T1_x',
                    "Координата Y": 'T1_y',
                    "Координата забоя Х (по траектории)": 'T3_x',
                    "Координата забоя Y (по траектории)": 'T3_y'}

# Стандартные типы кар, которые используются в модуле
list_names_map = ['NNT',
                  'permeability',
                  'residual_recoverable_reserves',
                  'pressure',
                  'water_cut',
                  'last_rate_oil',
                  'init_rate_oil',
                  'initial_oil_saturation',
                  # набор карт создаваемых в результате расчета
                  'reservoir_score',
                  'potential_score',
                  'risk_score',
                  'opportunity_index']
