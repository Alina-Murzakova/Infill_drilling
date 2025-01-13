import os
import pandas as pd
import numpy as np
import yaml
from dateutil.relativedelta import relativedelta
from loguru import logger
import xlwings as xw

from Utility_function import get_save_path

from economy.economy_functions import (select_analogue, expenditure_side, revenue_side, estimated_revenue, taxes,
                                       Profit, FCF, DCF)
from economy.economy_utilities import (add_on_sheet, check_economy_data_is_exist, dict_business_plan,
                                       dict_macroeconomics, name_columns_FPA, preparation_business_plan,
                                       preparation_coefficients, prepare_long_business_plan,
                                       preparation_macroeconomics, prepare_fpa)


def calculate_economy(reservoir):
    logger.info("1. Подготовка данных для расчета экономики")
    dir_path = f"{os.getcwd()}"
    economy_path = os.path.join(dir_path, "economy\\Экономика")

    with open(os.path.join(dir_path, "conf_files\\liquid_groups.yml")) as f:
        liquid_groups = pd.DataFrame(yaml.safe_load(f))
    liquid_groups = liquid_groups.astype("float")

    check_economy_data_is_exist(economy_path)

    # Инициализируем необходимые переменные
    coefficients = pd.read_excel(economy_path + "\\НРФ.xlsx", sheet_name="Расчет НДПИ", header=1, nrows=41)
    macroeconomics = pd.read_excel(economy_path + "\\Макра_оперативная_текущий_год.xlsx", nrows=15,
                                   usecols="A, B, O")  # Данные по макре за текущий год (напр: 2024)
    df_fpa = pd.read_excel(economy_path + "\\НРФ.xlsx", sheet_name="Ш-01.02.01.07-01, вер. 1.0",
                           usecols=name_columns_FPA, header=4).fillna(0)
    reservoirs_NDD = pd.read_excel(economy_path + "\\НРФ.xlsx", sheet_name="МР с НДД",
                                   header=None).replace({'2': '', '6': '', ' ЮЛ': ''}).drop_duplicates(keep='last')
    business_plan = pd.read_excel(economy_path + "\\Макра_оперативная_БП.xlsx", usecols="A, N:R",
                                  header=3)  # Данные по макре за следующие 5 лет (напр: 2025, 26, 27, 28, 29)
    business_plan_long = pd.read_excel(economy_path + "\\Макра_долгосрочная.xlsx", usecols="A, H:M",
                                       header=3)  # Данные по макре за следующие 6 лет (напр: 2030 - 35)

    logger.info("Предварительная обработка и подготовка файлов")
    macroeconomics = preparation_macroeconomics(macroeconomics, dict_macroeconomics)
    business_plan = preparation_business_plan(business_plan, dict_business_plan)
    coefficients = preparation_coefficients(coefficients)
    df_fpa = prepare_fpa(df_fpa, liquid_groups)

    macroeconomics = macroeconomics.merge(business_plan, left_on='Параметр', right_on='Параметр', how='outer')
    macroeconomics = macroeconomics.fillna(method='bfill', axis=1)
    macroeconomics.at[
        macroeconomics.loc[macroeconomics["Параметр"] == "r", "Параметр"].index[0], "Ед.изм."] = "Д.ед."

    dict_business_plan[
        'Нетбэк нефти для  Хантоса, СПД, Томскнефти, Мегиона, ГПН-Востока, Пальян, Толедо'] = 'Netback'
    business_plan_long = prepare_long_business_plan(business_plan_long, dict_business_plan)

    macroeconomics = macroeconomics.merge(business_plan_long, left_on='Параметр', right_on='Параметр', how='outer')
    macroeconomics = macroeconomics.fillna(method='ffill', axis=1)

    logger.info("check the content of output")
    if __name__ == "__main__":
        path_to_save = os.path.join(dir_path, "injection_profitability_output")
    else:
        try:
            path_to_save = get_save_path("injection_profitability")
        except PermissionError:
            return "Нет прав доступа для записи"

    file_data_reservoir = path_to_save + f"\\{reservoir}.xlsx"

    if not os.path.isfile(file_data_reservoir):
        logger.warning(f"Отсутствует расчет физики для месторождения - {reservoir}")
        raise f"Отсутствует расчет физики для месторождения - {reservoir}"

    # Dataframes:
    df_prod_well = pd.read_excel(file_data_reservoir, sheet_name="Прирост доб")
    df_prod_well[['№ добывающей', 'Ячейка']] = df_prod_well[['№ добывающей', 'Ячейка']].astype("str")

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

    logger.info(f"Объединение факта с прогнозом для добычи")
    if df_forecasts.shape[1] > 3:

        df_forecasts.columns = df_forecasts.columns[:3].to_list() + \
                               [pd.to_datetime(last_data) + relativedelta(months=i + 1) for i in
                                range(df_forecasts.shape[1] - 3)]
        del df_forecasts["Последняя дата работы"]
        df_forecasts.set_index("Ячейка", inplace=True)

        df_fQliq = df_forecasts[df_forecasts["Параметр"] == "delta_Qliq, tons/day"]
        df_fQoil = df_forecasts[df_forecasts["Параметр"] == "delta_Qoil, tons/day"]

        sum_Qliq = df_Qliq[['Ячейка', last_data]].groupby(by=['Ячейка']).sum()
        sum_Qoil = df_Qoil[['Ячейка', last_data]].groupby(by=['Ячейка']).sum()

        df_ratio_liq = df_Qliq[['Ячейка', "№ добывающей", last_data]]
        df_ratio_oil = df_Qoil[['Ячейка', "№ добывающей", last_data]]

        df_ratio_liq["ratio"] = df_ratio_liq.apply(lambda row: row[last_data] / sum_Qliq.loc[row['Ячейка']],
                                                   axis=1).fillna(0)
        df_ratio_oil["ratio"] = df_ratio_oil.apply(lambda row: row[last_data] / sum_Qoil.loc[row['Ячейка']],
                                                   axis=1).fillna(0)
        for column in df_forecasts.columns[1:]:
            df_Qliq[column] = df_ratio_liq.apply(lambda row: row["ratio"] * df_fQliq[column].loc[row['Ячейка']],
                                                 axis=1).fillna(0)
            df_Qoil[column] = df_ratio_oil.apply(lambda row: row["ratio"] * df_fQoil[column].loc[row['Ячейка']],
                                                 axis=1).fillna(0)

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
    # для консольного расчета экономики
    calculate_economy("Западно-Усть-Балыкское")