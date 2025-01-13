import numpy as np
import pandas as pd


def history_processing(history, max_delta, month_stop=120):
    """
    Предварительная обработка МЭР
    :param history: DataFrame
    :param max_delta Максимальный период остановки, дни
    :param month_stop Количество месяцев остановки для отнесения скважин к действующим (10 лет)
    :return: измененный DataFrame
    """
    last_date = np.sort(history.date.unique())[-1]
    # Пересчет дебитов через добычу
    history['Ql_rate'] = history.Ql / history.time_work_prod * 24
    history['Qo_rate'] = history.Qo / history.time_work_prod * 24
    history = history[(history.Ql_rate != 0) & (history.Qo_rate != 0) &
                      (history.time_work_prod != 0) & (history.objects != 0)]  # Оставляем не нулевые строки

    history = history.sort_values(['well_number', 'date'], ascending=True)
    history = history.reset_index(drop=True)

    unique_wells = history.well_number.unique()  # уникальный список скважин (без нулевых значений)
    history_new = pd.DataFrame()

    for well in unique_wells:
        slice_well = history.loc[history.well_number == well].copy()
        work_object = slice_well['objects'].iloc[-1]
        slice_well = slice_well[slice_well['objects'] == work_object]
        # удаление скважины, если она остановлена больше month_stop от текущей даты расчета
        no_work_time = round((last_date - slice_well.date.iloc[-1]).days / 29.3)
        if no_work_time > month_stop:
            continue

        next_dates = slice_well.date.iloc[1:]
        next_dates.loc[-1] = slice_well.date.iloc[-1]
        slice_well["next_date"] = list(next_dates)
        slice_well["date_delta"] = slice_well.next_date - slice_well.date

        # если скважина работала меньше суток в последний месяц - он удаляется
        if slice_well.time_work_prod.iloc[-1] < 24:
            slice_well = slice_well.iloc[:-1]

        slice_well = slice_well.reset_index()
        # обрезка истории, если скважина была остановлена больше max_delta
        if not slice_well[slice_well.date_delta > np.timedelta64(max_delta, 'D')].empty:
            last_index = slice_well[slice_well.date_delta > np.timedelta64(max_delta, 'D')].index.tolist()[-1]
            slice_well = slice_well.loc[last_index + 1:]
        history_new = pd.concat([history_new, slice_well], ignore_index=True)
    del history_new["date_delta"]
    del history_new["next_date"]
    return history_new


def production_model(period, model_arps_ql, model_arps_qo, reserves, day_in_month=29, well_efficiency=0.95):
    """
    Построение профиля жидкости и нефти
    Parameters
    ----------
    period период расчета, мес
    model_arps_ql параметры Арпса жидкости
    model_arps_qo параметры Арпса нефти
    reserves - запасы на скважину, т
    well_efficiency - коэффициент эксплуатации скважины
    day_in_month -  количество дней в месяце
    Returns
    -------
    [Ql_rates, Qo_rates], [Ql, Qo]
    """
    rates, productions = [], []
    # восстановление кривой Арпса
    for model in [model_arps_ql, model_arps_qo]:
        k1, k2 = model[1]
        starting_productivity = model[2]
        starting_index = model[3]

        range_period = np.array(range(starting_index, starting_index + period + 1))
        rate = starting_productivity * (1 + k1 * k2 * range_period) ** (-1 / k2)
        production = rate * day_in_month * well_efficiency
        rates.append(rate)
        productions.append(production)

    # проверка превышения запасов
    Qo_сumsum = np.cumsum(productions[1])
    mask_reserves = Qo_сumsum > reserves
    if True in mask_reserves:
        index_argmax = np.argmax(mask_reserves)
        Qo_argmax = rates[1][index_argmax]
        rates[1] = np.where(mask_reserves, 0, rates[1])
        productions[1] = np.where(mask_reserves, 0, productions[1])
        if index_argmax > 0:
            if Qo_сumsum[index_argmax - 1] != reserves:
                rates[1][index_argmax] = Qo_argmax
                productions[1][index_argmax] = reserves - Qo_сumsum[index_argmax - 1]
        else:
            rates[1][index_argmax] = Qo_argmax
            productions[1][index_argmax] = reserves

    # проверка снижение обводненности - полка по жидкости
    Wc = (rates[0] - rates[1]) / rates[0]
    mask_wc = (Wc[1:] - Wc[:-1]) < 0
    if True in mask_wc:
        index_argmax = np.argmax(mask_wc)
        rates[0][index_argmax:] = rates[0][index_argmax - 1]
        productions[0] = rates[0] * day_in_month * well_efficiency
    return rates, productions
