import os

import pandas as pd
import numpy as np
import xlwings as xw

from loguru import logger


class FinancialEconomicModel:
    """
    ФЭМ для процесса бурения
    """

    def __init__(self, macroeconomics, constants, unit_costs, oil_loss, workover_wellservice, type_tax_calculation):
        #  Макро-параметры
        self.dollar_exchange = macroeconomics['dollar_exchange']  # курс доллара
        self.urals = macroeconomics['urals']  # Urals для расчета налогов
        self.netback = macroeconomics['netback']
        self.cost_transportation = macroeconomics['cost_transportation']  # Транспортные расходы

        #  Налоги
        self.customs_duty = macroeconomics['customs_duty']  # Экспортная пошлина на нефть
        self.base_rate_MET = macroeconomics['base_rate_MET']  # Базовая ставка НДПИ на нефть, руб/т
        self.K_man = macroeconomics['K_man']  # Коэффициент Кман для расчета НДПИ и акциза, руб./т
        self.K_abdt = macroeconomics['K_abdt']  # КАБДТ с учетом НБУГ для расчета НДПИ на нефть, руб./т
        self.K_k = macroeconomics['K_k']  # Кк (НДПИ - корректировка по закону), руб./т
        self.r = macroeconomics['r']  # Ставка дисконтирования по Группе ГПН реальная
        self.type_tax_calculation = type_tax_calculation  # Тип расчета налогов НДПИ/НДД
        self.k_c = ((self.urals - 15) * self.dollar_exchange / 261).rename("k_c")
        self.mineral_extraction_tax = None

        #  Константы
        self.k_d = constants.iloc[0, 1]
        self.k_v = constants.iloc[1, 1]
        self.k_z = constants.iloc[2, 1]
        self.k_kan = constants.iloc[3, 1]
        self.K_ndpi = constants.iloc[4, 1]

        # OPEX
        self.unit_costs_oil = unit_costs['unit_costs_oil']
        self.unit_cost_fluid = unit_costs['unit_cost_fluid']
        self.cost_prod_well = unit_costs['cost_prod_well']
        self.oil_loss = oil_loss['oil_loss']

        # КРС И ПРС на 1 скважину сдф, МРП - не учитывается
        self.workover_one_cost = workover_wellservice['workover_one_cost']
        self.wellservice_one_cost = workover_wellservice['wellservice_one_cost']
        self.workover_number = workover_wellservice['workover_number']
        self.wellservice_number = workover_wellservice['wellservice_number']
        self.workover_wellservice_cost = None

        # CAPEX
        self.ONVSS_one_cost = workover_wellservice['ONVSS_one_cost']
        self.ONVSS_cost = None
        # + данные на амортизацию

        # Вычисления параметров модели
        self.calculate_mineral_extraction_tax()
        self.calculate_workover_wellservice_cost()
        self.calculate_ONVSS_cost()

    def calculate_mineral_extraction_tax(self):
        """Расчет НДПИ"""
        self.mineral_extraction_tax = (self.base_rate_MET * self.k_c - self.K_ndpi * self.k_c *
                                       (1 - self.k_d * self.k_v * self.k_z * self.k_kan)
                                       + self.K_k + self.K_abdt + self.K_man).rename("mineral_extraction_tax")
        pass

    def calculate_workover_wellservice_cost(self):
        """Удельные затраты на скважину КРС_ПРС"""
        self.workover_wellservice_cost = (self.workover_number * self.workover_one_cost
                                          + self.wellservice_number *
                                          self.wellservice_one_cost).rename("workover_wellservice_cost")
        pass

    def calculate_ONVSS_cost(self):
        self.ONVSS_cost = ((self.workover_number + self.wellservice_number)
                           * self.ONVSS_one_cost * 0.92).rename("ONVSS_cost")
        pass

    def calculate_revenue_side(self, Qo_yearly):
        """
        Доходная часть
        :param Qo_yearly: добыча нефти по годам, т
        """
        df_income = Qo_yearly.to_frame().join(self.netback).fillna(method='ffill')
        df_income = df_income.join(self.oil_loss).fillna(method='ffill')
        df_income['income'] = df_income.Qo_yearly * (1-df_income.oil_loss) * df_income.netback
        return df_income['income']

    def calculate_OPEX(self, Qo_yearly, Ql_yearly):
        """
        Расходная часть, руб
        :param Qo_yearly: добыча нефти по годам, т
        :param Ql_yearly: добыча жидкости по годам, т
        """
        df_cost_oil = Qo_yearly.to_frame().join(self.unit_costs_oil).fillna(method='ffill')
        df_cost_oil['cost_oil'] = df_cost_oil.Qo_yearly * df_cost_oil.unit_costs_oil

        df_cost_fluid = Ql_yearly.to_frame().join(self.unit_cost_fluid).fillna(method='ffill')
        df_cost_fluid['cost_fluid'] = df_cost_oil.Ql_yearly * df_cost_oil.unit_cost_fluid

        #
        # self.cost_prod_well
        # self.oil_loss
        #
        #
        # cost_prod_wells = (costs_oil + cost_fluid + cost_water) * num_days
        # cost_prod_wells = pd.DataFrame(cost_prod_wells, columns=Qo_yearly.columns[5:])
        # cost_prod_wells = pd.concat([Ql_yearly.iloc[:, :4], cost_prod_wells], axis=1)
        #
        #
        # all_cost = np.array(cost_cells) + np.array(cost_inj_wells.iloc[:, 1:])
        # all_cost = pd.DataFrame(all_cost, columns=cost_cells.columns)
        # all_cost = pd.concat([cost_inj_wells['Ячейка'], all_cost], axis=1)
        # all_cost = all_cost.groupby(['Ячейка']).sum().sort_index()
        # return cost_prod_wells, cost_inj_wells, all_cost

        pass


def calculate_economy(reservoir):

    logger.info(f"Расходная часть")
    cost_prod_wells, cost_inj_wells, all_cost = expenditure_side(df_Qoil, df_Qliq, df_W, unit_costs_oil,
                                                                 unit_costs_injection, unit_cost_fluid,
                                                                 unit_cost_water)

    netback = macroeconomics[macroeconomics["Параметр"] == "Netback"]
    Urals = macroeconomics[macroeconomics["Параметр"] == "Urals"]
    dollar_rate = macroeconomics[macroeconomics["Параметр"] == "exchange_rate"]

    logger.info(f"Доходная часть")
    income = revenue_side(netback, df_Qoil)
    logger.info(f"Расчетная выручка для расчета налога по схеме НДД")
    income_taxes = estimated_revenue(df_Qoil, Urals, dollar_rate)

    export_duty = macroeconomics[macroeconomics["Параметр"] == "customs_duty"]
    cost_transportation = macroeconomics[macroeconomics["Параметр"] == "cost_transportation"]
    K_man = macroeconomics[macroeconomics["Параметр"] == "K_man"]
    K_dt = macroeconomics[macroeconomics["Параметр"] == "K_dt"]

    coefficients_res = coefficients[coefficients["Месторождение"] == reservoir]
    if coefficients_res.empty:
        raise ValueError(f"Wrong name for coefficients: {reservoir}")
    coefficients_res = coefficients_res.values.tolist()[0][1:]

    method = "mineral_extraction_tax"
    if reservoir in reservoirs_NDD.values:
        method = "income_tax_additional"
    K = [coefficients_res, K_man, K_dt, K_d]

    logger.info(f"Налоги")
    all_taxes = taxes(df_Qoil, Urals, dollar_rate, export_duty, cost_transportation, income_taxes, all_cost, *K,
                      method=method)
    logger.info(f"Прибыль")
    profit = Profit(income, all_cost, all_taxes)
    logger.info(f"FCF")
    fcf = FCF(profit, profits_tax=0.2)
    r = macroeconomics[macroeconomics["Параметр"] == "r"]
    dcf = DCF(fcf, r, 2023)
    npv = dcf.cumsum(axis=1)

    # Выгрузка для Маши
    num_days = np.reshape(np.array(pd.to_datetime(df_Qoil.columns[5:]).days_in_month), (1, -1))
    years = pd.Series(df_Qoil.columns[5:], name='year').dt.year
    incremental_oil_production = np.array(df_Qoil.iloc[:, 5:]) * num_days
    incremental_oil_production = pd.DataFrame(incremental_oil_production, columns=df_Qoil.columns[5:])
    incremental_oil_production = pd.concat([df_Qoil.iloc[:, :4], incremental_oil_production], axis=1)
    incremental_oil_production = incremental_oil_production.groupby(['Ячейка'])[
        incremental_oil_production.columns[4:]].sum().sort_index()

    # Start print in Excel for one reservoir
    app1 = xw.App(visible=False)
    new_wb = xw.Book()

    add_on_sheet(new_wb, f"закачка", df_W)
    add_on_sheet(new_wb, f"Qн добывающих", df_Qoil)
    add_on_sheet(new_wb, f"Qж добывающих", df_Qliq)
    add_on_sheet(new_wb, f"Qн нагнетательных", incremental_oil_production)

    add_on_sheet(new_wb, f"Прибыль", profit)
    add_on_sheet(new_wb, f"доходная часть", income)
    add_on_sheet(new_wb, f"macroeconomics", macroeconomics)
    add_on_sheet(new_wb, f"НРФ", merged_FPA)
    add_on_sheet(new_wb, f"коэффициенты из НРФ", coefficients)
    add_on_sheet(new_wb, f"налоги", all_taxes)
    add_on_sheet(new_wb, f"затраты", all_cost)
    add_on_sheet(new_wb, f"затраты на добывающие", cost_prod_wells)
    add_on_sheet(new_wb, f"затраты на нагнетательные", cost_inj_wells)

    add_on_sheet(new_wb, f"{reservoir} FCF", fcf)
    add_on_sheet(new_wb, f"{reservoir} DCF", dcf)
    add_on_sheet(new_wb, f"{reservoir} NPV", npv)

    logger.info(f"Запись .xlsx")
    new_wb.save(path_to_save + f"\\{reservoir}_экономика.xlsx")
    app1.kill()
    return


if __name__ == "__main__":
    from app.local_parameters import main_parameters, constants
    from app.input_output.input import load_economy_data
    import pickle
    import sys

    name_field = 'Крайнее'
    paths = main_parameters['paths']
    path_economy = paths['path_economy']

    # Константы расчета
    start_date = constants['default_project_well_params']['start_date']

    sys.path.append(os.path.abspath(r"D:\Work\Programs_Python\Infill_drilling\app"))

    with open('D:\Work\Programs_Python\Infill_drilling\other_files\\test\data_wells.pkl', 'rb') as inp:
        data_wells = pickle.load(inp)
    with open('D:\Work\Programs_Python\Infill_drilling\other_files\\test\list_zones.pkl', 'rb') as inp:
        list_zones = pickle.load(inp)

    # для консольного расчета экономики
    FEM = load_economy_data(path_economy, name_field)

    # Для тестового расчета вытащим информацию по одной скважине
    project_well = list_zones[0].list_project_wells[0]

    # Сведение добыч по годам
    Qo = project_well.Qo

    # Создаем временной ряд с месячным интервалом
    date_range = pd.date_range(start=start_date, periods=len(Qo), freq='M')
    # Создаем DataFrame
    series_Qo = pd.Series(Qo, index=date_range, name='Qo_yearly')
    # Группируем по годам и суммируем добычу
    Qo_yearly = series_Qo.resample('Y').sum()
    # Переименовываем индекс для удобства
    Qo_yearly.index = Qo_yearly.index.year

    FEM.calculate_revenue_side(Qo_yearly)
    print(1)
