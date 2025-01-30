import itertools
import pandas as pd
import numpy as np


def revenue_side(Netback, Q_oil):
    """
    Доходная часть
    :param Netback: руб/т
    :param Q_oil: дебит нефти, т/сут
    """
    num_days = np.reshape(np.array(pd.to_datetime(Q_oil.columns[5:]).days_in_month), (1, -1))
    years = pd.Series(Q_oil.columns[5:], name='year').dt.year

    Netback = pd.DataFrame(data={'Netback': Netback.iloc[:, 2:].values.tolist()[0], 'year': Netback.columns[2:]})
    Netback = Netback.astype({'year': np.int64})
    Netback_line = np.array(pd.merge_asof(years, Netback, on="year", direction="nearest").iloc[:, -1])

    incremental_oil_production = np.array(Q_oil.iloc[:, 5:]) * num_days

    income = incremental_oil_production * Netback_line
    income = pd.DataFrame(income, columns=Q_oil.columns[5:])
    income = pd.concat([Q_oil.iloc[:, :4], income], axis=1)
    income = income.groupby(['Ячейка'])[income.columns[4:]].sum().sort_index()
    return income


def expenditure_side(Q_oil, Q_fluid, W,
                     unit_costs_oil, unit_costs_injection, unit_cost_fluid, unit_cost_water):
    """
    Расходная часть, руб
    :param unit_cost_water: уд. расходы на воду, руб/м3
    :param Q_oil: дебит нефти, т/сут
    :param Q_fluid: дебит жидкости, т/сут
    :param W: закачка, м3/сут
    :param unit_costs_oil: уд. расходы на нефть, руб/т
    :param unit_costs_injection: уд. расходы на закачку, руб/м3
    :param unit_cost_fluid: уд. расходы на жидкость, руб/т
    """
    # W = W.loc[:, Q_oil.iloc[:, 5:].columns.to_list()]
    diff_data = set(W.iloc[:, 2:].columns.to_list()).difference(Q_oil.iloc[:, 5:].columns.to_list())
    W = W.drop(columns=diff_data)

    num_days = np.reshape(np.array(pd.to_datetime(Q_oil.columns[5:]).days_in_month), (1, -1))

    costs_oil = np.array(Q_oil.iloc[:, 5:]) * np.reshape(np.array(unit_costs_oil), (-1, 1))
    cost_fluid = np.array(Q_fluid.iloc[:, 5:]) * np.reshape(np.array(unit_cost_fluid), (-1, 1))
    cost_water = (np.array(Q_fluid.iloc[:, 5:]) - np.array(Q_oil.iloc[:, 5:])) * np.reshape(np.array(unit_cost_water),
                                                                                            (-1, 1))

    cost_prod_wells = (costs_oil + cost_fluid + cost_water) * num_days
    cost_prod_wells = pd.DataFrame(cost_prod_wells, columns=Q_oil.columns[5:])
    cost_prod_wells = pd.concat([Q_fluid.iloc[:, :4], cost_prod_wells], axis=1)

    cost_inj_wells = np.array(W.iloc[:, 2:]) * unit_costs_injection * num_days
    cost_inj_wells = pd.DataFrame(cost_inj_wells, columns=W.columns[2:])
    cost_inj_wells = pd.concat([W.iloc[:, :1], cost_inj_wells], axis=1)
    cost_inj_wells = cost_inj_wells.loc[cost_inj_wells["Ячейка"].isin(cost_prod_wells["Ячейка"].unique()), :]

    cost_cells = cost_prod_wells.groupby(['Ячейка'])[cost_prod_wells.columns[4:]].sum().sort_index()
    cost_inj_wells = cost_inj_wells.sort_values(by='Ячейка')

    all_cost = np.array(cost_cells) + np.array(cost_inj_wells.iloc[:, 1:])
    all_cost = pd.DataFrame(all_cost, columns=cost_cells.columns)
    all_cost = pd.concat([cost_inj_wells['Ячейка'], all_cost], axis=1)
    all_cost = all_cost.groupby(['Ячейка']).sum().sort_index()
    return cost_prod_wells, cost_inj_wells, all_cost


def estimated_revenue(Q_oil, Urals, dollar_rate):
    """
    Расчетная выручка для расчета налога по схеме НДД, руб
    :param Q_oil: дебит нефти, т/сут
    :param Urals: стоимость дол/бар
    :param dollar_rate: Обменный курс, руб/дол
    """
    num_days = np.reshape(np.array(pd.to_datetime(Q_oil.columns[5:]).days_in_month), (1, -1))
    years = pd.Series(Q_oil.columns[5:], name='year').dt.year

    Urals = pd.DataFrame(data={'Urals': Urals.iloc[:, 2:].values.tolist()[0], 'year': Urals.columns[2:]})
    Urals = Urals.astype({'year': np.int64})
    Urals_line = np.array(pd.merge_asof(years, Urals, on="year", direction="nearest").iloc[:, -1])

    dollar_rate = pd.DataFrame(
        data={'dollar_rate': dollar_rate.iloc[:, 2:].values.tolist()[0], 'year': dollar_rate.columns[2:]})
    dollar_rate = dollar_rate.astype({'year': np.int64})
    dollar_rate_line = np.array(pd.merge_asof(years, dollar_rate, on="year", direction="nearest").iloc[:, -1])

    incremental_oil_production = np.array(Q_oil.iloc[:, 5:]) * num_days

    income_taxes = incremental_oil_production * Urals_line * dollar_rate_line * 7.3
    income_taxes = pd.DataFrame(income_taxes, columns=Q_oil.columns[5:])
    income_taxes = pd.concat([Q_oil.iloc[:, :4], income_taxes], axis=1)
    income_taxes = income_taxes.groupby(['Ячейка'])[income_taxes.columns[4:]].sum().sort_index()
    return income_taxes


def taxes(Q_oil, Urals, dollar_rate, export_duty,
          cost_transportation, estimated_revenue, expenditure_side, *K,
          method="mineral_extraction_tax", K_g=1):
    """
    Расчет налогов по схеме НДД или просто НДПИ, руб.
    :param Q_oil: дебит нефти, т/сут
    :param expenditure_side: Расходная часть, руб.
    :param estimated_revenue: Расчетная выручка для расчета налога по схеме НДД, руб
    :param Urals: стоимость дол/бар
    :param dollar_rate: Обменный курс, руб/дол
    :param export_duty: Экспортная пошлина, $/т
    :param cost_transportation: Транспортные расходы, руб./т
    :param method: "mineral_extraction_tax" (НДПИ) или "income_tax_additional" (НДД)
    :param K_g: зависит от группы м/р (для Мегиона всегда 1, 3я группа)
    :param K: K_d, K_v, K_z, K_an, K_man, K_dt
    :return:
    """
    num_days = np.reshape(np.array(pd.to_datetime(Q_oil.columns[5:]).days_in_month), (1, -1))
    years = pd.Series(Q_oil.columns[5:], name='year').dt.year

    incremental_oil_production = np.array(Q_oil.iloc[:, 5:]) * num_days

    Urals = pd.DataFrame(data={'Urals': Urals.iloc[:, 2:].values.tolist()[0], 'year': Urals.columns[2:]})
    Urals = Urals.astype({'year': np.int64})
    Urals_line = np.array(pd.merge_asof(years, Urals, on="year", direction="nearest").iloc[:, -1])

    dollar_rate = pd.DataFrame(
        data={'dollar_rate': dollar_rate.iloc[:, 2:].values.tolist()[0], 'year': dollar_rate.columns[2:]})
    dollar_rate = dollar_rate.astype({'year': np.int64})
    dollar_rate_line = np.array(pd.merge_asof(years, dollar_rate, on="year", direction="nearest").iloc[:, -1])

    export_duty = pd.DataFrame(
        data={'export_duty': export_duty.iloc[:, 2:].values.tolist()[0], 'year': export_duty.columns[2:]})
    export_duty = export_duty.astype({'year': np.int64})
    export_duty_line = np.array(pd.merge_asof(years, export_duty, on="year", direction="nearest").iloc[:, -1])

    cost_transportation = pd.DataFrame(
        data={'cost_transportation': cost_transportation.iloc[:, 2:].values.tolist()[0],
              'year': cost_transportation.columns[2:]})
    cost_transportation = cost_transportation.astype({'year': np.int64})
    cost_transportation_line = np.array(
        pd.merge_asof(years, cost_transportation, on="year", direction="nearest").iloc[:, -1])

    export_duty_line = export_duty_line * dollar_rate_line  # руб/т
    if method == "mineral_extraction_tax":
        # НДПИ/mineral_extraction_tax
        [K_v, K_z, K_an], K_man, K_dt, K_d = K

        K_man = pd.DataFrame(
            data={'K_man': K_man.iloc[:, 2:].values.tolist()[0], 'year': K_man.columns[2:]})
        K_man = K_man.astype({'year': np.int64})
        K_man_line = np.array(pd.merge_asof(years, K_man, on="year", direction="nearest").iloc[:, -1])

        K_dt = pd.DataFrame(
            data={'K_dt': K_dt.iloc[:, 2:].values.tolist()[0], 'year': K_dt.columns[2:]})
        K_dt = K_dt.astype({'year': np.int64})
        K_dt_line = np.array(pd.merge_asof(years, K_dt, on="year", direction="nearest").iloc[:, -1])

        K_d = np.reshape(np.array(K_d), (-1, 1))
        K_man_line = np.reshape(np.array(K_man_line), (1, -1))
        K_dt_line = np.reshape(np.array(K_dt_line), (1, -1))

        K_c = np.reshape((Urals_line - 15) * dollar_rate_line / 261, (1, -1))
        rate_mineral_extraction_tax = 919 * K_c - 559 * K_c * (
                1 - K_d * K_v * K_z * K_an) + K_man_line + 428 + K_dt_line
        mineral_extraction_tax = rate_mineral_extraction_tax * incremental_oil_production
        mineral_extraction_tax = pd.DataFrame(mineral_extraction_tax, columns=Q_oil.columns[5:])
        mineral_extraction_tax = pd.concat([Q_oil.iloc[:, :4], mineral_extraction_tax], axis=1)
        mineral_extraction_tax = mineral_extraction_tax.groupby(['Ячейка'])[
            mineral_extraction_tax.columns[4:]].sum().sort_index()
        return mineral_extraction_tax
    elif method == "income_tax_additional":
        # НДД/income_tax_additional
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
                                                             )[income_tax_additional.columns[4:]].sum().sort_index()

        income_tax_additional = 0.5 * (estimated_revenue - expenditure_side) + income_tax_additional

        return mineral_extraction_tax + income_tax_additional

    else:
        return None


def Profit(revenue_side, expenditure_side, taxes):
    """
    Прибыль, рб
    :param revenue_side: доход, руб
    :param expenditure_side: расход, руб
    :param taxes: налоги, руб
    """
    return revenue_side - expenditure_side - taxes


def FCF(profit, profits_tax=0.2):
    return profit * (1 - profits_tax)


def DCF(FCF, r, year_start):
    years = pd.Series(FCF.columns, name='year').dt.year

    r = pd.DataFrame(data={'r': r.iloc[:, 2:].values.tolist()[0], 'year': r.columns[2:]})
    r = r.astype({'year': np.int64})
    r_line = np.array(pd.merge_asof(years, r, on="year", direction="nearest").iloc[:, -1])

    count = 0
    k = 0.5 + years[0] - year_start  # к середине 2023 года, каждый год +1
    t = []
    while count < FCF.shape[1]:
        if FCF.shape[1] - count < 12:
            t += list(itertools.repeat(k, FCF.shape[1] - count))
        else:
            t += list(itertools.repeat(k, 12))
        count = len(t)
        k += 1 + (years[count-1] - year_start)
    dcf = np.array(FCF) * 1 / (1 + r_line) ** np.array(t)
    dcf = pd.DataFrame(dcf, columns=FCF.columns, index=FCF.index)
    return dcf

