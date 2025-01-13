import os
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from loguru import logger
import xlwings as xw
#
# from economy.economy_functions import (select_analogue, expenditure_side, revenue_side, estimated_revenue, taxes,
#                                        Profit, FCF, DCF)
# from economy.economy_utilities import (add_on_sheet, check_economy_data_is_exist, dict_business_plan,
#                                        dict_macroeconomics, name_columns_FPA, preparation_business_plan,
#                                        preparation_coefficients, prepare_long_business_plan,
#                                        preparation_macroeconomics, prepare_fpa)




def calculate_economy(reservoir):
    years = pd.Series(df_prod_well.columns[7:]).dt.to_period("A")
    last_year = years.iloc[-1]
    last_data = df_prod_well.columns[-1]
    index_start_year = years[years == last_year].index[0] + 7
    df_prod_well = df_prod_well.iloc[:, [1, 2, 3, 6]].merge(df_prod_well.iloc[:, index_start_year:],
                                                            right_index=True, left_index=True)
    df_prod_well = df_prod_well[df_prod_well["Параметр"].isin(["delta_Qliq, tons/day",
                                                               "delta_Qoil, tons/day",
                                                               "Qliq_fact, tons/day"])]
    df_prod_well.insert(loc=0, column="Месторождение", value=reservoir)
    df_Qliq = df_prod_well[df_prod_well["Параметр"] == "delta_Qliq, tons/day"].reset_index(drop=True)
    df_Qoil = df_prod_well[df_prod_well["Параметр"] == "delta_Qoil, tons/day"].reset_index(drop=True)
    df_Qliq_fact = df_prod_well[df_prod_well["Параметр"] == "Qliq_fact, tons/day"].iloc[:, [2, -1]].reset_index(
        drop=True)
    df_Qliq_fact = df_Qliq_fact.groupby("№ добывающей").sum()

    df_forecasts = pd.read_excel(file_data_reservoir, sheet_name="Прогноз_суммарный").iloc[:, 1:]
    df_forecasts['Ячейка'] = df_forecasts['Ячейка'].astype("str")


    logger.info(f"Проверка наличия всех скважин в НРФ")
    df_fpa.rename(columns={'№скв.': '№ добывающей'}, inplace=True)
    merged_FPA = df_Qliq[["Месторождение", "№ добывающей", ]].merge(df_fpa.drop_duplicates(
                                                                  subset=["Месторождение", "№ добывающей"]),
                                                                  left_on=["Месторождение", "№ добывающей"],
                                                                  right_on=["Месторождение", "№ добывающей"],
                                                                  how='left', validate = 'm:1')

    columns_start = merged_FPA.columns
    merged_FPA = merged_FPA.apply(
        lambda row: select_analogue(df_fpa.copy(), row, df_Qliq_fact.loc[row["№ добывающей"]][0], liquid_groups)
        if np.isnan(row['liquid_group']) else row, axis=1)
    merged_FPA = merged_FPA[columns_start]

    df_inj_well = pd.read_excel(file_data_reservoir, sheet_name="Прирост_наг_суммарный")
    df_inj_well[['Ячейка']] = df_inj_well[['Ячейка']].astype("str")

    index_start_year = years[years == last_year].index[0] + 3
    df_inj_well = df_inj_well.iloc[:, [1, 2]].merge(df_inj_well.iloc[:, index_start_year:], right_index=True,
                                                    left_index=True)
    df_W = df_inj_well[df_inj_well["Параметр"] == "Injection, m3/day"].reset_index(drop=True)

    logger.info(f"Объединение факта с прогнозом для закачки")
    if df_forecasts.shape[1] > 3:
        for column in df_forecasts.columns[1:]:
            df_W[column] = df_W.iloc[:, -1]

    unit_costs_oil = merged_FPA["Уделка на нефть, руб/тн.н"]
    unit_costs_injection = df_fpa[df_fpa["Месторождение"] == reservoir]['Уделка на закачку, руб/м3'].mean()
    unit_cost_fluid = merged_FPA['Уделка на жидкость, руб/т']
    unit_cost_water = merged_FPA["Уделка на воду, руб/м3"]
    K_d = merged_FPA["Кд"]

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
    from app.local_parameters import paths
    from app.input_output.input import load_economy_data
    path_economy = paths['path_economy']
    # для консольного расчета экономики
    load_economy_data(path_economy)