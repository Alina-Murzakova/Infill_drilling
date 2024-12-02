import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint, fsolve

import warnings
warnings.filterwarnings("ignore")


class CharacteristicOfDesaturation:
    """
    Класс функции характеристики вытеснения по модели Кори
    """

    def __init__(self, oil_production, liq_production,
                 initial_reserves, water_cut, oil_recovery_factor):
        self.oil_production = oil_production
        self.liq_production = liq_production
        self.initial_reserves = initial_reserves
        self.water_cut_fact = water_cut
        self.oil_recovery_factor = oil_recovery_factor

        self.last_three_points = False

    def solver(self, correlation_coeff):
        """
        Аппроксимация кривой добычи жидкости Арпса
        :param correlation_coeff: параметры корреляции
        :return: сумма квадратов отклонений точек функции от фактических
        """
        corey_oil, corey_water, mef = correlation_coeff

        v1 = np.cumsum(self.oil_production) / self.initial_reserves / 1000
        v1 = np.delete(v1, 0)
        v1 = np.insert(v1, 0, 0)
        k1 = (1 - v1) ** corey_oil / ((1 - v1) ** corey_oil + mef * v1 ** corey_water)
        v2 = v1 + self.liq_production * k1 / 2 / self.initial_reserves / 1000
        k2 = (1 - v2) ** corey_oil / ((1 - v2) ** corey_oil + mef * v2 ** corey_water)
        v3 = v1 + self.liq_production * k2 / 2 / self.initial_reserves / 1000
        k3 = (1 - v3) ** corey_oil / ((1 - v3) ** corey_oil + mef * v3 ** corey_water)
        v4 = v1 + self.liq_production * k3 / 2 / self.initial_reserves / 1000
        k4 = (1 - v4) ** corey_oil / ((1 - v4) ** corey_oil + mef * v4 ** corey_water)

        oil_production_model = self.liq_production / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        oil_production_model[(oil_production_model == -np.inf) | (oil_production_model == np.inf)] = 0
        deviation = [(oil_production_model - self.oil_production) ** 2]
        return np.sum(deviation)

    def conditions_cd(self, correlation_coeff):
        """Привязка (binding) к последним фактическим точкам"""
        corey_oil, corey_water, mef = correlation_coeff
        water_cut_last_dot = self.water_cut_fact[-1]
        # self.last_three_points = True - привязка к последним трем точкам, False - привязка к последним трем
        if self.last_three_points:
            if self.water_cut_fact.size >= 3:
                water_cut_last_dot = np.average(self.water_cut_fact[-3:-1])
            elif self.water_cut_fact.size == 2:
                water_cut_last_dot = np.average(self.water_cut_fact[-2:-1])

        water_cut_model = mef * self.oil_recovery_factor ** corey_water / (
                (1 - self.oil_recovery_factor) ** corey_oil + mef * self.oil_recovery_factor ** corey_water)
        binding = water_cut_model - water_cut_last_dot
        return [binding]


def calculation_model_corey(Qo_well, Ql_well, residual_reserves, constraints_model_corey):
    """
    Функция для аппроксимации характеристики вытеснения
    :param Qo_well: добыча нефти по скважине, т
    :param Ql_well: добыча жидкости по скважине, т
    :param residual_reserves: ОИЗ на входе, тыс. т
    :param constraints_model_corey: ограничения на аппроксимацию
    :return: output - массив с коэффициентами аппроксимации
    [success, [corey_oil, corey_water, mef], message, method]
    """
    # Текущая выработка скважины
    cumulative_oil_production = Qo_well.sum() / 1000
    initial_reserves = residual_reserves + cumulative_oil_production
    oil_recovery_factor = cumulative_oil_production / initial_reserves
    # начальные параметры [corey_oil, corey_water, mef]
    d_set = np.array([3, 2, 3])
    output = [False, d_set, "ValueError: аппроксимация не сошлась", '']
    # Ограничения на аппроксимацию
    corey_oil_right, corey_water_right = np.inf, np.inf
    function_boundaries = Bounds([constraints_model_corey["corey_oil_left"],
                                  constraints_model_corey["corey_water_left"],
                                  constraints_model_corey["mef_left"]],
                                 [corey_oil_right, corey_water_right, constraints_model_corey["mef_right"]])
    # Проверка точек обводненности
    water_cut = (Ql_well - Qo_well) / Ql_well
    water_cut[(water_cut == -np.inf) | (water_cut == np.inf)] = 0
    # Больше ли точек с обводненностью выше последней?
    condition = len(water_cut[np.where(water_cut < water_cut[-1])]) > len(
        water_cut[np.where(water_cut > water_cut[-1])])
    if condition:
        mask = np.full(water_cut.shape, True)
        water_cut_for_model, Qo_for_model, Ql_for_model = water_cut[mask], Qo_well[mask], Ql_well[mask]
        CD = CharacteristicOfDesaturation(Qo_for_model, Ql_for_model,
                                          initial_reserves, water_cut_for_model, oil_recovery_factor)

        non_linear_con = NonlinearConstraint(CD.conditions_cd, [-0.00001], [0.00001])
        # Оптимизация функции
        try:
            result_minimize = minimize(CD.solver, d_set,
                                       method='trust-constr',
                                       bounds=function_boundaries,
                                       constraints=non_linear_con)
            output = [result_minimize.success, result_minimize.x,  result_minimize.message,
                      'Расчет на всех точках']
        except ValueError:
            pass
        # logger.info(output)

    # Проверка характеристик вытеснения, если был расчет
    if (output[1][0] < 0.01 and output[1][1] < 0.01) or not condition:
        mask = np.where(water_cut <= water_cut[-1])
        water_cut_for_model, Qo_for_model, Ql_for_model = water_cut[mask], Qo_well[mask], Ql_well[mask]
        CD = CharacteristicOfDesaturation(Qo_for_model, Ql_for_model,
                                          initial_reserves, water_cut_for_model, oil_recovery_factor)
        non_linear_con = NonlinearConstraint(CD.conditions_cd, [-0.00001], [0.00001])
        try:
            result_minimize = minimize(CD.solver, d_set,
                                       method='trust-constr',
                                       bounds=function_boundaries,
                                       constraints=non_linear_con)
            output = [result_minimize.success, result_minimize.x, result_minimize.message,
                      'Отброшены точки выше последней']
        except ValueError:
            pass
    return output
