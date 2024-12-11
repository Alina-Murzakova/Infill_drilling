import numpy as np
import pandas as pd


def history_processing(history, max_delta):
    """
    Предварительная обработка МЭР
    :param history: DataFrame
    :param max_delta Максимальный период остановки, дни
    :return: измененный DataFrame
    """
    last_data = np.sort(history.date.unique())[-1]
    # Пересчет дебитов через добычу
    history['Ql_rate'] = history.Ql / history.time_work_prod * 24
    history['Qo_rate'] = history.Qo / history.time_work_prod * 24
    history = history[(history.Ql_rate != 0) & (history.Qo_rate != 0) &
                      (history.time_work_prod != 0) & (history.objects != 0)]  # Оставляем не нулевые строки

    # оставляем скважины с историей по объектам на дату расчета
    list_objects = history[history.date == last_data].groupby(['well_number'])['objects'].apply(list)
    history = history[history.well_number.isin(list_objects.index.unique())]
    history = history[history.apply(lambda x: x.objects in list_objects[x.well_number], axis=1)]
    history = history.sort_values(['well_number', 'date'], ascending=True)
    history = history.reset_index(drop=True)

    unique_wells = history.well_number.unique()  # уникальный список скважин (без нулевых значений)
    history_new = pd.DataFrame()

    for well in unique_wells:
        slice_well = history.loc[history.well_number == well].copy()

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

# from datetime import (date, timedelta)
# import calendar
# def fluid_production_mod(period, model_corey_well, model_arps_well, date_last, oil_recovery_factor, initial_reserves):
#     """
#     Построение профиля жидкости !!!! требует изменения под новый формат
#     :param period: период расчета (мес.)
#     :param model_corey_well: параметры функции Кори для скважины
#     :param model_arps_well: параметры Арпса
#     :param date_last: дата начала прогноза
#     :param oil_recovery_factor: текущая выработка
#     :param initial_reserves: НИЗ
#     :return: [Qo_rate, Ql_rate] - набор списков добыча нефти и добыча жидкости
#     """
#     Co, Cw, mef = model_corey_well
#     k1, k2, starting_index, starting_productivity = model_arps_well
#     k1 = float(k1)
#     k2 = float(k2)
#     starting_index = int(float(starting_index))
#     starting_productivity = float(starting_productivity)
#     date_last = date(date_last[0], date_last[1], date_last[2])
#
#     Qo, Qo_rate, Ql_rate, water_cut = [0], [], [], []
#     for month in range(period):
#         oil_recovery_factor = oil_recovery_factor + Qo[-1] / initial_reserves / 1000
#         if oil_recovery_factor >= 1:
#             oil_recovery_factor = 0.99999999999
#         water_cut.append(mef * oil_recovery_factor ** Cw /
#                          ((1 - oil_recovery_factor) ** Co + mef * oil_recovery_factor ** Cw))
#         Ql_rate.append(starting_productivity * (1 + k1 * k2 * (starting_index - 1)) ** (-1 / k2))
#         starting_index += 1
#         Qo_rate.append(Ql_rate[-1] * (1 - water_cut[-1]))
#         days_in_month = calendar.monthrange(date_last.year, date_last.month)[1]
#         Qo.append(Qo_rate[-1] * days_in_month)
#         date_last += timedelta(days=days_in_month)
#     return Qo_rate, Ql_rate
