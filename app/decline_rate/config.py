import numpy as np

# Константы для аппроксимации кривой Арпса
K1_LEFT = 0.0001
K1_RIGHT = 5
K2_LEFT = 0.0001
K2_RIGHT = 50

# Константы для аппроксимации кривой Кори
COREY_OIL_LEFT = 0.0
COREY_WATER_LEFT = 0.0
MEF_LEFT = 0.0
MEF_RIGHT = np.inf

# Предельное время остановки скважины для обрезки истории
STOPPING_TIME_LIMIT_OF_WELL = 730

# Константы для расчета ОИЗ
MIN_RESIDUAL_RECOVERABLE_RESERVES = 2  # тыс.т
MIN_PERIOD_WELL_WORKING = 5
MAX_PERIOD_WELL_WORKING = 80
MAX_RADIUS = 1_000


