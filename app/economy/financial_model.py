import os

import pandas as pd
import numpy as np
import xlwings as xw

from loguru import logger

from app.ranking_drilling.starting_rates import check_FracCount

import pandas as pd
import numpy as np


def calculate_depreciation_base(capex_series, lifetime):
    """
    Расчет базы для амортизации по методу уменьшаемого остатка.

    capex_series: pd.Series - капитальные затраты (CAPEX) по годам.
    lifetime: int - срок амортизации в годах.

    Возвращает: pd.Series - база для амортизации по годам.
    """
    start_year = capex_series.index.min()
    end_year = capex_series.index.max()

    depreciation_base = pd.Series(0, index=range(start_year, end_year + lifetime + 1))  # Создаем пустую серию

    for year in range(start_year, end_year + 1):
        for age in range(lifetime + 1):  # 0 - текущий год, 1 - год назад и т.д.
            contribution_year = year - age
            if contribution_year in capex_series.index:
                if age == lifetime:
                    depreciation_base[year] += 0.5 * capex_series[contribution_year]  # Половина CAPEX за lifetime назад
                elif age == 0:
                    depreciation_base[year] += 0.5 * capex_series[year]  # Половина текущего CAPEX
                else:
                    depreciation_base[year] += capex_series[contribution_year]  # Полный CAPEX за прошлые годы

    return depreciation_base[start_year:end_year + 1]  # Ограничиваем диапазон

def linear_depreciation(cost, salvage, life):
    """
    Рассчитывает величину амортизации актива за один период, используя линейный метод.

    :param cost: Начальная стоимость актива.
    :param salvage: Ликвидационная стоимость актива (стоимость в конце срока службы).
    :param life: Срок службы актива (количество периодов).
    :return: Величина амортизации за один период.
    """
    if life <= 0:
        raise ValueError("Срок службы должен быть больше 0")

    return (cost - salvage) / life


def bring_arrays_to_one_date(*series):
    """ Привести датированные показатели к одной дате в массиве"""
    joint_data_frame = series[0].to_frame()
    for part in series[1:]:
        joint_data_frame = joint_data_frame.join(part).fillna(method='ffill')
    return joint_data_frame

def calculate_production_by_years(production_by_month, start_date, type):
        if type == 'Qo':
            name = 'Qo_yearly'
        elif type == 'Ql':
            name = 'Ql_yearly'
        # Создаем временной ряд с месячным интервалом
        date_range = pd.date_range(start=start_date, periods=len(production_by_month), freq='M')
        # Создаем DataFrame
        series_production_by_month = pd.Series(production_by_month, index=date_range, name=name)
        # Группируем по годам и суммируем добычу
        production_by_years = series_production_by_month.resample('Y').sum()/1000 # в тыс. т
        # Переименовываем индекс для удобства
        production_by_years.index = production_by_years.index.year
        return production_by_years

class FinancialEconomicModel:
    """
    ФЭМ для процесса бурения
    """

    def __init__(self, macroeconomics, constants, unit_costs, oil_loss, df_capex,
                 workover_wellservice, type_tax_calculation):
        #  Макро-параметры
        self.dollar_exchange = macroeconomics['dollar_exchange']  # курс доллара
        self.urals = macroeconomics['urals']  # Urals для расчета налогов
        self.netback = macroeconomics['netback']
        self.cost_transportation = macroeconomics['cost_transportation']  # Транспортные расходы

        #  Налоги
        self.export_duty = macroeconomics['export_duty']  # Экспортная пошлина на нефть
        self.base_rate_MET = macroeconomics['base_rate_MET']  # Базовая ставка НДПИ на нефть, руб/т
        self.K_man = macroeconomics['K_man']  # Коэффициент Кман для расчета НДПИ и акциза, руб./т
        self.K_abdt = macroeconomics['K_abdt']  # КАБДТ с учетом НБУГ для расчета НДПИ на нефть, руб./т
        self.K_k = macroeconomics['K_k']  # Кк (НДПИ - корректировка по закону), руб./т
        self.r = macroeconomics['r']  # Ставка дисконтирования по Группе ГПН реальная
        self.type_tax_calculation = type_tax_calculation  # Тип расчета налогов НДПИ/НДД
        self.k_c = ((self.urals - 15) * self.dollar_exchange / 261).rename("k_c")
        self.rate_mineral_extraction_tax = None

        #  Константы
        self.k_d = constants.iloc[0, 1]
        self.k_v = constants.iloc[1, 1]
        self.k_z = constants.iloc[2, 1]
        self.k_kan = constants.iloc[3, 1]
        self.K_ndpi = constants.iloc[4, 1]

        # OPEX
        self.unit_cost_oil = unit_costs['unit_costs_oil']
        self.unit_cost_fluid = unit_costs['unit_cost_fluid']
        self.cost_prod_well = unit_costs['cost_prod_well']

        # КРС И ПРС на 1 скважину сдф, МРП - не учитывается
        self.workover_one_cost = workover_wellservice['workover_one_cost']
        self.wellservice_one_cost = workover_wellservice['wellservice_one_cost']
        self.workover_number = workover_wellservice['workover_number']
        self.wellservice_number = workover_wellservice['wellservice_number']
        self.workover_wellservice_cost = None

        # CAPEX
        self.ONVSS_one_cost = workover_wellservice['ONVSS_one_cost']
        self.ONVSS_cost = None
        self.cost_production_drilling_vertical=df_capex.iloc[3, 1]
        self.cost_production_drilling_horizontal=df_capex.iloc[2, 1]
        self.cost_stage_GRP = df_capex.iloc[4, 1]
        self.cost_secondary_material_resources = df_capex.iloc[1, 1]
        self.cost_completion =df_capex.iloc[0, 1]

        # Дополнительные данные для расчета
        self.oil_loss = oil_loss['oil_loss']

        # Вычисления параметров модели
        self.calculate_rate_mineral_extraction_tax()
        self.calculate_workover_wellservice_cost()
        self.calculate_ONVSS_cost()

    def calculate_rate_mineral_extraction_tax(self):
        """Расчет НДПИ"""
        self.rate_mineral_extraction_tax = (self.base_rate_MET * self.k_c - self.K_ndpi * self.k_c *
                                           (1 - self.k_d * self.k_v * self.k_z * self.k_kan)
                                           + self.K_k + self.K_abdt + self.K_man).rename("rate_mineral_extraction_tax")
        pass

    def calculate_workover_wellservice_cost(self):
        """Удельные затраты на скважину КРС_ПРС = ТКРС (нефть)"""
        self.workover_wellservice_cost = (self.workover_number * self.workover_one_cost
                                          + self.wellservice_number *
                                          self.wellservice_one_cost).rename("workover_wellservice_cost")
        pass

    def calculate_ONVSS_cost(self):
        self.ONVSS_cost = ((self.workover_number + self.wellservice_number)
                           * self.ONVSS_one_cost * 0.92).rename("ONVSS_cost")
        pass

    def calculate_Qo_yearly_for_sale(self, Qo_yearly):
        """
        Доходная часть
        :param Qo_yearly: добыча нефти по годам, тыс. т
        """
        df_Qo_yearly_for_sale = bring_arrays_to_one_date(Qo_yearly, self.oil_loss)
        df_Qo_yearly_for_sale['Qo_yearly_for_sale'] = ((df_Qo_yearly_for_sale.Qo_yearly *
                                                         (1-df_Qo_yearly_for_sale.oil_loss))
                                                         .rename("Qo_yearly_for_sale"))
        return df_Qo_yearly_for_sale.Qo_yearly_for_sale

    def calculate_revenue_side(self, Qo_yearly):
        """
        Выручка (доходная часть)
        :param Qo_yearly: добыча нефти по годам, тыс. т
        """
        Qo_yearly_for_sale = self.calculate_Qo_yearly_for_sale(Qo_yearly)
        df_income = bring_arrays_to_one_date(Qo_yearly_for_sale, self.netback)
        df_income['income'] = df_income.Qo_yearly_for_sale * df_income.netback
        return df_income.income

    def calculate_OPEX(self, Qo_yearly, Ql_yearly):
        """
        Расходная опексовая часть, руб
        :param Qo_yearly: добыча нефти по годам, тыс. т
        :param Ql_yearly: добыча жидкости по годам, тыс. т
        """
        df_OPEX = bring_arrays_to_one_date(Qo_yearly, self.unit_cost_oil,
                                                Ql_yearly, self.unit_cost_fluid,
                                                self.cost_prod_well,
                                                self.workover_wellservice_cost)

        df_OPEX['cost_oil'] = df_OPEX.Qo_yearly * df_OPEX.unit_costs_oil
        df_OPEX['cost_fluid'] = df_OPEX.Ql_yearly * df_OPEX.unit_cost_fluid
        df_OPEX['OPEX'] = (df_OPEX.cost_oil + df_OPEX.cost_fluid +
                           df_OPEX.cost_prod_well + df_OPEX.workover_wellservice_cost)
        return df_OPEX.OPEX

    def calculate_CAPEX(self, project_well, well_params, Qo_yearly):
        """
        Расходная капитализируемая часть, руб
        состав:
        Скважины (Бурение_ГС|ННС + ГРП_за 1 стадию) cost_production_drilling
        ОНСС (в первый месяц + 5 638 тыс.р на скв ? узнать что это) ONVSS_cost
        Обустройство (Обустройство + ВМР) cost_infrastructure
        """
        cost_production_drilling = self.cost_production_drilling_vertical
        if project_well.well_type == 'horizontal':
            cost_production_drilling = self.cost_production_drilling_horizontal
        FracCount = check_FracCount(well_params['Type_Frac'], well_params['length_FracStage'], well_params['L'])
        cost_production_drilling += FracCount * self.cost_stage_GRP

        cost_infrastructure = self.cost_secondary_material_resources + self.cost_completion

        df_CAPEX = bring_arrays_to_one_date(Qo_yearly, self.ONVSS_cost)
        del df_CAPEX['Qo_yearly']
        df_CAPEX.ONVSS_cost.iloc[0] += 5638

        df_CAPEX['cost_production_drilling'] = 0
        df_CAPEX['cost_infrastructure'] = 0
        df_CAPEX['cost_production_drilling'] .iloc[0] += cost_production_drilling
        df_CAPEX['cost_infrastructure'].iloc[0] += cost_infrastructure
        df_CAPEX['CAPEX']= df_CAPEX.cost_infrastructure + df_CAPEX.cost_production_drilling + df_CAPEX.ONVSS_cost
        return df_CAPEX

    def calculate_depreciation(self, df_CAPEX, lifetime_well = 7, lifetime_ONVSS = 4):
        start_year, end_year = df_CAPEX.index.min(), df_CAPEX.index.max()
        df_depreciation = pd.DataFrame(index=range(start_year, end_year + max(lifetime_well, lifetime_ONVSS) + 1))
        # Базы для расчета амортизации
        df_depreciation['production_drilling_depreciation_base'] = (
            calculate_depreciation_base(df_CAPEX.cost_production_drilling, lifetime_well))
        pass

    def calculate_income_NDD(self, Qo_yearly):
        """
        Расчетная выручка для расчета налога по схеме НДД, тыс. руб
        :param Qo_yearly: добыча нефти по годам, тыс. т
        """
        df_income_NDD = bring_arrays_to_one_date(Qo_yearly, self.urals, self.dollar_exchange)
        df_income_NDD['income_NDD'] = (df_income_NDD.Qo_yearly * df_income_NDD.urals
                                                   * df_income_NDD.dollar_exchange * 7.3)
        return df_income_NDD.income_NDD

    def calculate_taxes(self, Qo_yearly_for_sale, method="НДД"):
        """
        Расчет общей суммы налогов по схеме НДД или просто НДПИ, тыс. руб.
        :param Qo_yearly_for_sale: Добыча нефти по годам с вычетом потерь, тыс. т
        :param method: "НДПИ" или "НДД"
        """
        # + Налог на имущество: Амортизация * налог на имущество
        # + Налог на прибыль: база налога на прибыль * налог на прибыль
        # База по НП без учета переноса убытков = Выручка-Opex-НДПИ-налог на имущество-Амортизация
        # Итого для расчета НДПИ - сумма трех налогов
        if method == "НДПИ":
            df_mineral_extraction_tax = bring_arrays_to_one_date(Qo_yearly_for_sale,
                                                                 self.rate_mineral_extraction_tax)
            df_mineral_extraction_tax['mineral_extraction_tax'] = (df_mineral_extraction_tax.Qo_yearly *
                                                                  df_mineral_extraction_tax.rate_mineral_extraction_tax)
            return df_mineral_extraction_tax['mineral_extraction_tax']


        elif method == "НДД":
            df_export_duty_cash = bring_arrays_to_one_date(self.export_duty, self.dollar_exchange)
            df_export_duty_cash[
                'export_duty_cash'] = df_export_duty_cash.export_duty * df_export_duty_cash.dollar_exchange  # руб/т


            rate_mineral_extraction_tax = 0.5 * (Urals_line - 15) * 7.3 * dollar_rate_line * K_g - export_duty_line
            mineral_extraction_tax = rate_mineral_extraction_tax * incremental_oil_production
            mineral_extraction_tax = pd.DataFrame(mineral_extraction_tax, columns=Q_oil.columns[5:])
            mineral_extraction_tax = pd.concat([Q_oil.iloc[:, :4], mineral_extraction_tax], axis=1)
            mineral_extraction_tax = mineral_extraction_tax.groupby(['Ячейка'])[
                mineral_extraction_tax.columns[4:]].sum().sort_index()

            income_tax_additional = - 0.5 * (rate_mineral_extraction_tax + export_duty_line + cost_transportation_line) \
                                    * incremental_oil_production

            income_tax_additional = pd.DataFrame(income_tax_additional, columns=Q_oil.columns[5:])
            income_tax_additional = pd.concat([Q_oil.iloc[:, :4], income_tax_additional], axis=1)
            income_tax_additional = income_tax_additional.groupby(['Ячейка']
                                                                  )[
                income_tax_additional.columns[4:]].sum().sort_index()

            income_tax_additional = 0.5 * (estimated_revenue - expenditure_side) + income_tax_additional

            return mineral_extraction_tax + income_tax_additional
        else:
            return None



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
    # Параметры для скважин РБ
    well_params = main_parameters['well_params']

    # Константы расчета
    start_date = constants['default_project_well_params']['start_date']

    sys.path.append(os.path.abspath(r"C:\Users\Вячеслав\Desktop\Work\Python\Infill_drilling\app"))

    with open('C:\\Users\Вячеслав\Desktop\Work\Python\Infill_drilling\output\Крайнее_2БС10\data_wells.pickle', 'rb') as inp:
        data_wells = pickle.load(inp)
    with open('C:\\Users\Вячеслав\Desktop\Work\Python\Infill_drilling\output\Крайнее_2БС10\list_zones.pickle', 'rb') as inp:
        list_zones = pickle.load(inp)

    # для консольного расчета экономики
    FEM = load_economy_data(path_economy, name_field)

    # Для тестового расчета вытащим информацию по одной скважине
    project_well = list_zones[0].list_project_wells[0]

    Qo = project_well.Qo
    Ql = project_well.Ql

    # Сведение добычи по годам
    Qo_yearly = calculate_production_by_years(Qo, start_date, type='Qo')
    Ql_yearly = calculate_production_by_years(Ql, start_date, type='Ql')

    df_income = FEM.calculate_revenue_side(Qo_yearly)
    df_OPEX = FEM.calculate_OPEX(Qo_yearly, Ql_yearly)
    df_CAPEX = FEM.calculate_CAPEX(project_well, well_params, Qo_yearly)

    # Амортизация
    df_depreciation = FEM.calculate_depreciation(df_CAPEX)

    # Налоги
    df_income_NDD = FEM.calculate_income_NDD(Qo_yearly)

    # Показатели эффективности

    print(1)
