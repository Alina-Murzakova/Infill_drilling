import os
import pandas as pd
import numpy as np
import xlwings as xw

from loguru import logger

from app.economy.functions import bring_arrays_to_one_date, calculate_depreciation_base, linear_depreciation, \
    calculate_irr_root_scalar, calculate_mirr, calculate_production_by_years, calculate_performance_indicators
from app.ranking_drilling.starting_rates import check_FracCount


class FinancialEconomicModel:
    """
    ФЭМ для процесса бурения
    """

    def __init__(self, macroeconomics, constants, unit_costs, oil_loss, df_capex, workover_wellservice):
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
        self.k_c = ((self.urals - 15) * self.dollar_exchange / 261).rename("k_c")
        self.rate_mineral_extraction_tax = None
        self.rate_profits_tax = macroeconomics['rate_profits_tax']  # Ставка налога на прибыль
        self.rate_property_tax = macroeconomics['rate_property_tax']  # Ставка налога на имущество

        #  Константы
        self.k_d = constants.iloc[0, 1]
        self.k_v = constants.iloc[1, 1]
        self.k_z = constants.iloc[2, 1]
        self.k_kan = constants.iloc[3, 1]
        self.K_ndpi = constants.iloc[4, 1]

        # Налоги (НДД)
        self.rate_NDD_tax = macroeconomics['rate_NDD_tax']  # Ставка налога на дополнительный доход, %

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
        self.cost_completion = df_capex.iloc[0, 1]
        self.cost_secondary_material_resources = df_capex.iloc[1, 1]
        self.cost_production_drilling_horizontal = df_capex.iloc[2, 1]
        self.cost_production_drilling_vertical = df_capex.iloc[3, 1]
        self.cost_stage_GRP = df_capex.iloc[4, 1]
        self.lifetime_well = df_capex.iloc[5, 1]
        self.lifetime_ONVSS = df_capex.iloc[6, 1]
        self.lifetime_infrastructure = df_capex.iloc[7, 1]

        # Дополнительные данные для расчета
        self.oil_loss = oil_loss['oil_loss']

        # Дисконтирование
        self.r = macroeconomics['r']  # Ставка дисконтирования по Группе ГПН реальная
        self.start_discount_year = 2023  # к половине 2023 года

        # Вычисления параметров модели
        self.calculate_rate_mineral_extraction_tax()
        self.calculate_workover_wellservice_cost()
        self.calculate_ONVSS_cost()

    def calculate_rate_mineral_extraction_tax(self):
        """Расчет льготного НДПИ"""
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
        """ Расчет ОНВСС при ПРС"""
        self.ONVSS_cost = ((self.workover_number + self.wellservice_number)
                           * self.ONVSS_one_cost * 0.92).rename("ONVSS_cost")
        pass

    def calculate_Qo_yearly_for_sale(self, Qo_yearly):
        """
         Добыча нефти по годам с вычетом потерь, тыс. т
        :param Qo_yearly: добыча нефти по годам, тыс. т
        """
        df_Qo_yearly_for_sale = bring_arrays_to_one_date(Qo_yearly, self.oil_loss)
        df_Qo_yearly_for_sale['Qo_yearly_for_sale'] = ((df_Qo_yearly_for_sale.Qo_yearly *
                                                        (1 - df_Qo_yearly_for_sale.oil_loss))
                                                       .rename("Qo_yearly_for_sale"))
        return df_Qo_yearly_for_sale.Qo_yearly_for_sale

    def calculate_income_side(self, Qo_yearly_for_sale):
        """
        Выручка (доходная часть)
        :param Qo_yearly_for_sale: Добыча нефти по годам с вычетом потерь, тыс. т
        """
        df_income = bring_arrays_to_one_date(Qo_yearly_for_sale, self.netback)
        df_income['income'] = df_income.Qo_yearly_for_sale * df_income.netback
        return df_income.income
        # return df_income

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
        # return df_OPEX

    def calculate_CAPEX(self, well_type, well_params, Qo_yearly):
        """
        Расходная капитализируемая часть, руб
        состав:
        Скважины (Бурение_ГС|ННС + ГРП_за 1 стадию) cost_production_drilling
        ОНСС (в первый месяц + 5 638 тыс.р на скв ? узнать что это) ONVSS_cost
        Обустройство (Обустройство + ВМР) cost_infrastructure
        """
        cost_production_drilling = self.cost_production_drilling_vertical
        if well_type == 'horizontal':
            cost_production_drilling = self.cost_production_drilling_horizontal
        FracCount = check_FracCount(well_params['Type_Frac'], well_params['length_FracStage'], well_params['L'])
        cost_production_drilling += FracCount * self.cost_stage_GRP

        cost_infrastructure = self.cost_secondary_material_resources + self.cost_completion

        df_CAPEX = bring_arrays_to_one_date(Qo_yearly, self.ONVSS_cost)
        del df_CAPEX['Qo_yearly']
        df_CAPEX.ONVSS_cost.iloc[0] += 5638

        df_CAPEX['cost_production_drilling'] = 0
        df_CAPEX['cost_infrastructure'] = 0
        df_CAPEX['cost_production_drilling'].iloc[0] += cost_production_drilling
        df_CAPEX['cost_infrastructure'].iloc[0] += cost_infrastructure
        df_CAPEX['CAPEX'] = df_CAPEX.sum(axis=1)
        return df_CAPEX

    def calculate_depreciation(self, df_CAPEX):
        """Расчет Амортизации по трем категориям
        Скважины  cost_production_drilling
        ОНСС  ONVSS_cost
        Обустройство cost_infrastructure"""
        start_year, end_year = df_CAPEX.index.min(), df_CAPEX.index.max()
        columns_for_depreciation = ['cost_production_drilling', 'cost_infrastructure', 'ONVSS_cost']
        lifetimes = {'cost_production_drilling': self.lifetime_well,
                     'cost_infrastructure': self.lifetime_infrastructure,
                     'ONVSS_cost': self.lifetime_ONVSS}

        df_base_depreciation = pd.DataFrame(index=range(start_year, end_year + max(lifetimes.values()) + 1))
        df_depreciation = pd.DataFrame(index=range(start_year, end_year + max(lifetimes.values()) + 1))

        # База для расчета амортизации
        for column in columns_for_depreciation:
            series_base_depreciation = calculate_depreciation_base(df_CAPEX[column], lifetimes[column])
            series_base_depreciation.name = column
            df_base_depreciation = df_base_depreciation.join(series_base_depreciation).fillna(0)

        # Амортизация линейным способом
        for column in columns_for_depreciation:
            series_depreciation = df_base_depreciation[column].apply(linear_depreciation, args=(0, lifetimes[column],))
            series_depreciation.name = 'depreciation_' + column
            df_depreciation = df_depreciation.join(series_depreciation).fillna(0)
        df_depreciation['depreciation'] = df_depreciation.sum(axis=1)
        df_depreciation = df_depreciation.join(df_CAPEX).fillna(0)

        # Остаточная стоимость
        df_depreciation['residual_cost'] = df_depreciation.CAPEX.cumsum() - df_depreciation.depreciation.cumsum()

        # База для исчисления Налога на имущество
        df_depreciation['base_property_tax'] = df_depreciation['residual_cost'].rolling(window=2).mean()
        df_depreciation['base_property_tax'].iloc[0] = df_depreciation['residual_cost'].iloc[0]

        # Амортизация для Налога на прибыль (с учетом премии 30%)
        for column in columns_for_depreciation:
            series_depreciation = df_depreciation[column] * 0.3 + (df_base_depreciation[column] * 0.7).apply(
                linear_depreciation, args=(0, lifetimes[column],))
            series_depreciation.name = 'depreciation_income_tax_' + column
            df_depreciation = df_depreciation.join(series_depreciation).fillna(0)
        columns_for_depreciation_income_tax = ['depreciation_income_tax_' + column for column in
                                               columns_for_depreciation]
        df_depreciation['depreciation_income_tax'] = df_depreciation[columns_for_depreciation_income_tax].sum(axis=1)
        return df_depreciation[['depreciation', 'base_property_tax', 'depreciation_income_tax']]
        # return df_depreciation

    def calculate_income_NDD(self, Qo_yearly_for_sale):
        """
        Расчетная выручка для расчета налога по схеме НДД, тыс. руб за вычетом транспортных расходов
        :param Qo_yearly_for_sale: добыча нефти по годам с вычетом потерь, тыс. т
        """
        df_income_NDD = bring_arrays_to_one_date(Qo_yearly_for_sale,
                                                 self.urals, self.dollar_exchange, self.cost_transportation)
        df_income_NDD['income_NDD'] = (df_income_NDD.urals * df_income_NDD.dollar_exchange * 7.3
                                       - df_income_NDD.cost_transportation) * df_income_NDD.Qo_yearly_for_sale
        return df_income_NDD.income_NDD

    def calculate_taxes(self, Qo_yearly, df_depreciation, income, OPEX, CAPEX, method="НДД", **kwargs):
        """
        Расчет общей суммы налогов по схеме НДД или просто льготный НДПИ, тыс. руб.
        :param Qo_yearly: Добыча нефти по годам, тыс. т
        :param method: "НДПИ" или "НДД"
        :param df_depreciation: фрейм Амортизации
        """
        df_taxes = bring_arrays_to_one_date(Qo_yearly, df_depreciation, income, OPEX, CAPEX,
                                            self.rate_mineral_extraction_tax,
                                            self.rate_property_tax,
                                            self.rate_profits_tax,
                                            self.rate_NDD_tax, self.urals, self.dollar_exchange, self.export_duty)

        df_taxes['Qo_yearly_for_sale'] = self.calculate_Qo_yearly_for_sale(Qo_yearly)
        # Налог на имущество
        df_taxes['property_tax'] = df_taxes.base_property_tax * df_taxes.rate_property_tax

        if method == "НДПИ":
            # НДПИ нефть
            df_taxes['mineral_extraction_tax'] = (df_taxes.Qo_yearly_for_sale * df_taxes.rate_mineral_extraction_tax)
            # База для налога на прибыль без учета переноса убытков
            # = Выручка - Опекс - НДПИ нефть - Налог на имущество - Амортизация для Налога на прибыль
            df_taxes['base_profits_tax'] = (df_taxes.income - df_taxes.OPEX -
                                            df_taxes.mineral_extraction_tax - df_taxes.property_tax -
                                            df_taxes.depreciation_income_tax)
            # Налог на прибыль
            df_taxes['profits_tax'] = df_taxes.base_profits_tax * df_taxes.rate_profits_tax
            # Налоги сумма
            columns_taxes = ['mineral_extraction_tax', 'property_tax', 'profits_tax']
            df_taxes['taxes'] = df_taxes[columns_taxes].sum(axis=1)
            return df_taxes[columns_taxes + ['taxes']]
            # return df_taxes

        elif method == "НДД":
            initial_recoverable_reserves = kwargs.get('initial_recoverable_reserves')
            cumulative_production = kwargs.get('cumulative_production')
            Kg_group = kwargs.get('Kg_group')

            # Выработка ЛУ
            df_taxes['production_reserves'] = ((cumulative_production + df_taxes.Qo_yearly.cumsum())
                                               / initial_recoverable_reserves)
            df_taxes['year_commercial_development'] = (df_taxes.production_reserves > 0.01).cumsum()

            if Kg_group == 1 or Kg_group == 2:
                df_taxes['Kg'] = (df_taxes['year_commercial_development']
                                  .apply(lambda x: 0.4 if x < 7 else 0.6 if x == 7 else 0.8 if x == 8 else 1))
            elif Kg_group == 3:
                df_taxes['Kg'] = 1
            elif Kg_group == 4:
                df_taxes['Kg'] = (df_taxes['year_commercial_development']
                                  .apply(lambda x: 0.5 if x < 3 else 0.75 if x < 4 else 1))
            else:
                logger.warning(f"Неверное значение Кг_номер группы: {Kg_group}")
            # Ставка НДПИ в НДД с уч. Кг
            # почему в расчете фиксированное значение - НДПИ на нефть для участков V группы НДД (при Кг=0,6)
            df_taxes['rate_mineral_extraction_tax_NDD_Kg'] = ((0.5 * (df_taxes.urals - 15) * 7.3)
                                                              * df_taxes.dollar_exchange
                                                              * df_taxes.Kg - df_taxes.export_duty).clip(lower=0)
            # НДПИ нефть
            df_taxes['mineral_extraction_tax'] = (df_taxes.Qo_yearly_for_sale
                                                  * df_taxes.rate_mineral_extraction_tax_NDD_Kg)

            # База для налога на прибыль без учета переноса убытков
            # = Выручка - Опекс - НДПИ нефть - Налог на имущество - Амортизация для Налога на прибыль
            df_taxes['base_profits_tax'] = (df_taxes.income - df_taxes.OPEX -
                                            df_taxes.mineral_extraction_tax - df_taxes.property_tax -
                                            df_taxes.depreciation_income_tax)
            # Налог на прибыль
            df_taxes['profits_tax'] = df_taxes.base_profits_tax * df_taxes.rate_profits_tax

            # Расчетная выручка-Тр = (Urals*Курс доллара*7,3-'ТРАНСПОРТНЫЕ РАСХОДЫ ДЛЯ НДД)*Нефть товарная
            df_taxes['income_NDD'] = self.calculate_income_NDD(df_taxes.Qo_yearly_for_sale)
            # Фактические расходы = CAPEX, OPEX, Налог на имущество
            df_taxes['actual_expenditures'] = df_taxes.CAPEX + df_taxes.OPEX + df_taxes.property_tax
            # Расчетные расходы = ЕСЛИ(И(Кг<100%;группа<3);0;Нефть товарная*Экспортная пошлина на нефть*Курс доллара)
            # + НДПИ в НДД с уч. Кг
            df_taxes['calculated_expenditures'] = (df_taxes.apply(
                lambda row: 0 if row.Kg < 1 and Kg_group < 3 else row.Qo_yearly_for_sale
                                                                  * row.export_duty
                                                                  * row.dollar_exchange, axis=1)
                                                   + df_taxes.mineral_extraction_tax)

            # База НДД = (Расчетная выручка-Тр) - Фактические расходы - Расчетные расходы
            df_taxes['base_NDD'] = df_taxes.income_NDD - df_taxes.actual_expenditures - df_taxes.calculated_expenditures
            # налог по схеме НДД
            df_taxes['NDD_tax'] = df_taxes.base_NDD * df_taxes.rate_NDD_tax
            # Налоги сумма
            columns_taxes = ['mineral_extraction_tax', 'NDD_tax', 'property_tax', 'profits_tax']
            df_taxes['taxes'] = df_taxes[columns_taxes].sum(axis=1)
            return df_taxes[columns_taxes + ['taxes']]
            # return df_taxes
        else:
            return None

    def calculate_discount_period(self, FCF):
        """Расчет периода дисконтирования"""
        start_year = self.start_discount_year
        end_year = FCF.index.max()
        years = list(range(start_year, end_year + 1))
        periods = np.arange(0.5, 0.5 + len(years), 1)
        return pd.Series(periods, index=years, name='discount_period')

    def calculate_discounted_measures(self, FCF, CAPEX, discount_period):
        """ Дисконтированные показатели: DCF, NPV, PVI, PI, IRR, MIRR """
        df_discounted_measures = bring_arrays_to_one_date(FCF, CAPEX, discount_period, self.r)
        # Стоимость денежной единицы
        df_discounted_measures['value_monetary_unit'] = (
                1 / (1 + df_discounted_measures.r) ** df_discounted_measures.discount_period)
        df_discounted_measures['DCF'] = df_discounted_measures.value_monetary_unit * df_discounted_measures.FCF
        df_discounted_measures['NPV'] = df_discounted_measures.DCF.cumsum()
        #  Дисконтированные инвестиции
        df_discounted_measures['PVI'] = df_discounted_measures.value_monetary_unit * df_discounted_measures.CAPEX
        # Проектный период
        PI = df_discounted_measures.NPV.max() / df_discounted_measures.PVI.sum() + 1
        # Расчет IRR
        if df_discounted_measures.FCF.sum() < 0:
            IRR = None
        elif df_discounted_measures.FCF.min() > 0:
            IRR = np.inf
        else:
            IRR = calculate_irr_root_scalar(df_discounted_measures.FCF.values)
        # Расчет MIRR
        MIRR = calculate_mirr(df_discounted_measures.FCF.values)
        return df_discounted_measures[['DCF', 'NPV', 'PVI']], PI, IRR, MIRR
        # return df_discounted_measures, PI, IRR, MIRR

    def calculate_economy_well(self, Qo, Ql, start_date, well_type, well_params, method, dict_NDD):
        """
        Расчет ФЭМ для процесса бурения скважины
        Parameters
        ----------
        Qo - профиль добычи нефти, т
        Ql - профиль добычи жидкости, т
        start_date - дата начала бурения
        well_type - тип скважины
        well_params - параметры по-умолчанию
        method - схема расчета налогов
        dict_NDD - параметры для расчета налогов по схеме НДД
        Returns
        -------
        [CAPEX, OPEX, Накопленный поток наличности, NPV, PVI, PI]
        """
        # Сведение добычи по годам
        Qo_yearly = calculate_production_by_years(Qo, start_date, type='Qo')
        Ql_yearly = calculate_production_by_years(Ql, start_date, type='Ql')
        Qo_yearly_for_sale = self.calculate_Qo_yearly_for_sale(Qo_yearly)

        df_income = self.calculate_income_side(Qo_yearly_for_sale)
        df_OPEX = self.calculate_OPEX(Qo_yearly, Ql_yearly)
        df_CAPEX = self.calculate_CAPEX(well_type, well_params, Qo_yearly)

        # Амортизация
        df_depreciation = self.calculate_depreciation(df_CAPEX)

        # Налоги
        df_taxes = self.calculate_taxes(Qo_yearly, df_depreciation, df_income, df_OPEX, df_CAPEX.CAPEX,
                                        method, **dict_NDD)
        # Показатели эффективности
        performance_indicators = calculate_performance_indicators(df_income, df_OPEX, df_CAPEX.CAPEX, df_taxes,
                                                                  df_depreciation)
        # Дисконтированные показатели
        discount_period = self.calculate_discount_period(performance_indicators.FCF)
        df_discounted_measures, PI, IRR, MIRR = self.calculate_discounted_measures(performance_indicators.FCF,
                                                                                   df_CAPEX.CAPEX,
                                                                                   discount_period)
        return (df_CAPEX.CAPEX, df_OPEX, performance_indicators.cumulative_cash_flow,
                df_discounted_measures.NPV, df_discounted_measures.PVI, PI)


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

    sys.path.append(os.path.abspath(r"D:\Work\Programs_Python\Infill_drilling\app"))

    with open('D:\Work\Programs_Python\Infill_drilling\output\Крайнее_2БС10\data_wells.pickle', 'rb') as inp:
        data_wells = pickle.load(inp)
    with open('D:\Work\Programs_Python\Infill_drilling\output\Крайнее_2БС10\list_zones.pickle', 'rb') as inp:
        list_zones = pickle.load(inp)

    # для консольного расчета экономики
    FEM, method, dict_NDD = load_economy_data(path_economy, name_field)

    # Для тестового расчета вытащим информацию по одной скважине
    project_well = list_zones[3].list_project_wells[4]

    Qo = project_well.Qo
    Ql = project_well.Ql

    # Сведение добычи по годам
    Qo_yearly = calculate_production_by_years(Qo, start_date, type='Qo')
    Ql_yearly = calculate_production_by_years(Ql, start_date, type='Ql')
    Qo_yearly_for_sale = FEM.calculate_Qo_yearly_for_sale(Qo_yearly)

    df_production = bring_arrays_to_one_date(Qo_yearly, Ql_yearly, Qo_yearly_for_sale)
    df_production.column = ['добыча нефти, тыс. т', 'добыча жидкости, тыс. т', 'нефть товарная, тыс. т']

    df_OPEX = FEM.calculate_OPEX(Qo_yearly, Ql_yearly)
    df_CAPEX = FEM.calculate_CAPEX(project_well, well_params, Qo_yearly)
    df_income = FEM.calculate_income_side(Qo_yearly_for_sale)

    # Амортизация
    df_depreciation = FEM.calculate_depreciation(df_CAPEX)

    # Налоги
    # method = "НДПИ"
    df_taxes = FEM.calculate_taxes(Qo_yearly,
                                   df_depreciation[['depreciation', 'base_property_tax', 'depreciation_income_tax']],
                                   df_income.income, df_OPEX.OPEX, df_CAPEX.CAPEX, method, **dict_NDD)

    # Показатели эффективности
    performance_indicators = calculate_performance_indicators(df_income.income, df_OPEX.OPEX, df_CAPEX.CAPEX,
                                                              df_taxes[['taxes', 'profits_tax']],
                                                              df_depreciation[['depreciation', 'base_property_tax',
                                                                               'depreciation_income_tax']])

    # Дисконтированные показатели
    discount_period = FEM.calculate_discount_period(performance_indicators.FCF)
    df_discounted_measures, PI, IRR, MIRR = FEM.calculate_discounted_measures(performance_indicators.FCF,
                                                                              df_CAPEX.CAPEX,
                                                                              discount_period)
    df_measures = pd.DataFrame(data=[PI, IRR, MIRR], index=['PI', 'IRR', 'MIRR'])

    # Переименование колонок
    dict_columns = {'Qo_yearly': 'добыча нефти, тыс. т',
                    'Ql_yearly': 'добыча жидкости, тыс. т',
                    'Qo_yearly_for_sale': 'нефть товарная, тыс. т',
                    'unit_costs_oil': 'Удельные затраты на 1 тонну нефти',
                    'unit_cost_fluid': 'Удельные затраты на 1 тонну жидкости',
                    'cost_prod_well': 'Удельные затраты на 1 скв.СДФ (доб.)',
                    'workover_wellservice_cost': 'Удельные затраты на скважину КРС_ПРС (ТКРС)',
                    'cost_oil': 'OPEX нефть',
                    'cost_fluid':  'OPEX жидкость',
                    'ONVSS_cost': 'ОНСС',
                    'cost_production_drilling': 'Скважины (Бурение_ГС|ННС + ГРП_за 1 стадию * кол-во стадий)',
                    'cost_infrastructure': 'Обустройство (Обустройство + ВМР)',
                    'income': 'Выручка',
                    'depreciation_cost_production_drilling': 'Амортизация Скважины',
                    'depreciation_cost_infrastructure': 'Амортизация Обустройство',
                    'depreciation_ONVSS_cost': 'Амортизация ОНСС',
                    'depreciation'	: 'Амортизация сумма',
                    'residual_cost'	: 'Остаточная стоимость',
                    'base_property_tax'	: 'База для исчисления Налога на имущество',
                    'depreciation_income_tax_cost_production_drilling': 'Амортизация для Налога на прибыль (Скважины)',
                    'depreciation_income_tax_cost_infrastructure': 'Амортизация для Налога на прибыль (Обустройство)',
                    'depreciation_income_tax_ONVSS_cost'	: 'Амортизация для Налога на прибыль (ОНСС)',
                    'depreciation_income_tax': 'Амортизация для Налога на прибыль (с учетом премии 30%)',
			        'rate_mineral_extraction_tax': 'льготный НДПИ',
                    'rate_property_tax': 'Ставка налога на имущество',
                    'rate_profits_tax': 'Ставка налога на прибыль',
                    'rate_NDD_tax': 'Ставка НДД',
                    'dollar_exchange': 'Доллар США',
                    'export_duty': 'Экспортная пошлина на нефть',
                    'property_tax': 'Налог на имущество',
                    'mineral_extraction_tax': 'НДПИ',
                    'base_profits_tax': 'База для налога на прибыль без учета переноса убытков',
                    'profits_tax': 'Налог на прибыль',
                    'taxes': 'Налоги',
                    'cumulative_cash_flow': 'Накопленный поток наличности',
                    'discount_period': 'период дисконтирования',
                    'value_monetary_unit': 'Стоимость денежной единицы',
                    'production_reserves': 'Выработка ЛУ',
                    'year_commercial_development': 'Год промышленной разработки',
                    'Kg': 'Кг',
                    'rate_mineral_extraction_tax_NDD_Kg': 'Ставка НДПИ в НДД с уч. Кг',
                    'income_NDD': 'Расчетная выручка для НДД',
                    'actual_expenditures': 'Фактические расходы',
                    'calculated_expenditures': 'Расчетные расходы',
                    'base_NDD': 'База НДД',
                    'NDD_tax': 'НДД'}

    df_production.rename(columns=dict_columns, inplace=True)
    df_OPEX.rename(columns=dict_columns, inplace=True)
    df_CAPEX.rename(columns=dict_columns, inplace=True)
    df_income.rename(columns=dict_columns, inplace=True)
    df_depreciation.rename(columns=dict_columns, inplace=True)
    df_taxes.rename(columns=dict_columns, inplace=True)
    performance_indicators.rename(columns=dict_columns, inplace=True)
    df_discounted_measures.rename(columns=dict_columns, inplace=True)
    df_measures.rename(columns=dict_columns, inplace=True)

    # Сохранение материалов для экономиста
    filename = f"D:\Work\Programs_Python\Infill_drilling\other_files\ФЭМ\ФЭМ_Крайнее_2БС10_ВНС_3_5_{method}.xlsx"
    with (pd.ExcelWriter(filename) as writer):

        df_production.to_excel(writer, sheet_name='профиль')
        df_OPEX.to_excel(writer, sheet_name='OPEX')
        df_CAPEX.to_excel(writer, sheet_name='CAPEX')
        df_income.to_excel(writer, sheet_name='Выручка')
        df_depreciation.to_excel(writer, sheet_name='Амортизация')
        df_taxes.to_excel(writer, sheet_name='Налоги')
        performance_indicators.to_excel(writer, sheet_name='Показатели эффективности')
        df_discounted_measures.to_excel(writer, sheet_name='Дисконтированные показатели')
        df_measures.to_excel(writer, sheet_name='PI_IRR_MIRR')
