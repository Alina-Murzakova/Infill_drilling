import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint

import warnings
warnings.filterwarnings("ignore")


class FunctionFluidProduction:
    """Класс функции добычи"""

    def __init__(self, fluid_production):
        self.fluid_production = fluid_production

        self.first_month = -1
        self.starting_productivity = -1
        self.starting_index = -1
        self.base_correction = -1

        self.last_three_points = False

    def adaptation(self, correlation_coeff):
        """
        :param correlation_coeff: коэффициенты корреляции функции
        :return: сумма квадратов отклонений фактических точек от модели
        """
        k1, k2 = correlation_coeff

        self.base_correction = self.fluid_production[-1]
        if self.last_three_points:
            if self.fluid_production.size >= 3:
                self.base_correction = np.average(self.fluid_production[-3:-1])
            elif self.fluid_production.size == 2:
                self.base_correction = np.average(self.fluid_production[-2:-1])

        self.starting_productivity = np.amax(self.fluid_production)
        self.starting_index = np.where(self.fluid_production == self.starting_productivity)[0][0]

        if self.starting_index > (self.fluid_production.size - 4) and self.fluid_production.size > 3:
            if self.base_correction < np.amax(self.fluid_production[:-3]):
                self.starting_productivity = np.amax(self.fluid_production[:-3])
                self.starting_index = np.where(self.fluid_production == self.starting_productivity)[0][0]

        indexes = np.arange(start=self.starting_index, stop=self.fluid_production.size, step=1) - self.starting_index
        day_fluid_production_month = self.starting_productivity * (1 + k1 * k2 * indexes) ** (-1 / k2)
        deviation = [(self.fluid_production[self.starting_index:] - day_fluid_production_month) ** 2]
        self.first_month = self.fluid_production.size - self.starting_index + 1
        return np.sum(deviation)

    def conditions_fp(self, correlation_coeff):
        """Привязка (binding) к последним фактическим точкам"""
        k1, k2 = correlation_coeff

        self.base_correction = self.fluid_production[-1]
        # self.last_three_points = True - привязка к последним трем точкам, False - привязка к последним трем
        if self.last_three_points:
            if self.fluid_production.size >= 3:
                self.base_correction = np.average(self.fluid_production[-3:-1])
            elif self.fluid_production.size == 2:
                self.base_correction = np.average(self.fluid_production[-2:-1])

        self.starting_productivity = np.amax(self.fluid_production)
        self.starting_index = np.where(self.fluid_production == self.starting_productivity)[0][0]

        if self.starting_index > (self.fluid_production.size - 4) and self.fluid_production.size > 3:
            if self.base_correction < np.amax(self.fluid_production[:-3]):
                self.starting_productivity = np.amax(self.fluid_production[:-3])
                self.starting_index = np.where(self.fluid_production == self.starting_productivity)[0][0]

        last_prod = (self.starting_productivity *
                     (1 + k1 * k2 * (self.fluid_production.size - 1 - self.starting_index)) ** (-1 / k2))
        binding = self.base_correction - last_prod
        return binding


def calculation_model_arps_with_condition(rate_well, constraints_model_arps):
    """
    Функция для аппроксимации кривой добычи c привязкой к фактическим точкам кривой
    :param rate_well: дебит по скважине, т/сут
    :param constraints_model_arps: ограничения на аппроксимацию
    :return: output - массив с коэффициентами аппроксимации
    [success, [k1, k2], message, first_month, starting_productivity, starting_index]
    """
    # начальные параметры [k1, k2]
    c_cet = np.array([0.0001, 0.0001])
    output = [False, c_cet, "ValueError: аппроксимация не сошлась", 0, 0, 0]
    #  Условие, если в расчете только одна точка или последняя точка максимальная
    if rate_well.size > 1 and np.amax(rate_well) != rate_well[-1]:
        FP = FunctionFluidProduction(rate_well)
        # Ограничения на аппроксимацию
        function_boundaries = Bounds([constraints_model_arps['k1_left'], constraints_model_arps['k2_left']],
                                     [constraints_model_arps['k1_right'], constraints_model_arps['k2_right']])
        try:
            for i in range(10):
                non_linear_con = NonlinearConstraint(FP.conditions_fp, [-0.00001], [0.00001])
                result_minimize = minimize(FP.adaptation, c_cet, method='trust-constr', bounds=function_boundaries,
                                           constraints=non_linear_con, options={'disp': False})
                if result_minimize.nit > 900:
                    break
            output = [result_minimize.success, result_minimize.x, result_minimize.message,
                      FP.first_month, FP.starting_productivity, FP.starting_index]
        except ValueError:
            pass
        # logger.info(output)
    return output


def fun_arps(correlation_coeff, production, starting_index, starting_productivity):
    """
    Parameters
    ----------
    correlation_coeff коэффициенты корреляции функции
    starting_productivity стартовый дебит для аппроксимации
    starting_index стартовый индекс
    production дебит
    :return: сумма квадратов отклонений фактических точек от модели
    """
    k1, k2 = correlation_coeff
    indexes = np.arange(start=starting_index, stop=production.size, step=1) - starting_index
    day_fluid_production_month = starting_productivity * (1 + k1 * k2 * indexes) ** (-1 / k2)
    deviation = [(production[starting_index:] - day_fluid_production_month) ** 2]
    return np.sum(deviation)


def calculation_model_arps(rate_well, constraints_model_arps, min_time_work=7):
    """
    Функция для аппроксимации кривой добычи без привязки
    :param rate_well: дебит по скважине, т/сут
    :param constraints_model_arps: ограничения на аппроксимацию
    :min_time_work: минимальное количество месяцев работы скважины для аппроксимации
    :return: output - массив с коэффициентами аппроксимации
    [success, [k1, k2], starting_productivity, starting_index, message]
    """
    # начальные параметры [k1, k2]
    list_ks = [np.array([0.5, 0.5]), np.array([0.001, 0.001])]
    output = [False, list_ks[1], 0, 0, f"TimeError: скважина работает меньше {min_time_work} месяцев"]
    #  Условие, если в расчете точек < min_time_work
    if rate_well.size >= min_time_work:
        # Ограничения на аппроксимацию
        function_boundaries = Bounds([constraints_model_arps['k1_left'], constraints_model_arps['k1_right']],
                                     [constraints_model_arps['k2_left'], constraints_model_arps['k2_right']])
        # поиск месяца и индекса максимальной добычи
        starting_productivity = np.amax(rate_well)
        starting_index = np.where(rate_well == starting_productivity)[0][0]
        if starting_index > min_time_work:
            starting_productivity = np.amax(rate_well[:-min_time_work+1])
            starting_index = np.where(rate_well == starting_productivity)[0][0]
        for c_cet in list_ks:
            try:
                from functools import partial
                partial_arps = partial(fun_arps,
                                       production=rate_well,
                                       starting_index=starting_index,
                                       starting_productivity=starting_productivity)
                result_minimize = minimize(partial_arps, c_cet,
                                           method='trust-constr',
                                           bounds=function_boundaries,
                                           options={'gtol': 1e-6, 'maxiter': 10000})
                output = [result_minimize.success, result_minimize.x,
                          starting_productivity, starting_index, result_minimize.message]
            except ValueError:
                pass
            if output[0]:
                break
    return output
