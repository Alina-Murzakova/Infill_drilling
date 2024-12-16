import os
import pandas as pd
import sys
import xlwings as xw

from loguru import logger


dict_macroeconomics = {'Обменный курс, руб/дол': 'exchange_rate',
                       'Таможенная пошлина на нефть из России, дол/тонна': 'customs_duty',
                       'Базовая ставка НДПИ на нефть, руб/т': 'base_rate_MET',
                       'Коэффициент Кман для расчета НДПИ и акциза, руб/т': 'K_man',
                       'КАБДТ с учетом НБУГ и НДФО для расчета НДПИ на нефть, руб/т': 'K_dt',
                       'Netback УУН Холмогоры - FOB Новороссийск, руб/тн (с учетом дисконта Трейдингу)': 'Netback',
                       'Urals (средний), дол/бар': 'Urals',
                       'Транспортные расходы для лицензионных участков III-IV группы НДД:'
                       ' поставка УУН Холмогоры - FOB Новороссийск, руб./т': 'cost_transportation'}

dict_business_plan = {'Доллар США': 'exchange_rate',
                      'Urals для расчета налогов': 'Urals',
                      'Нетбэк нефти для  Хантоса, СПД, Томскнефти, Мегиона, ГПН-Востока, Пальян, Толедо': 'Netback',
                      'Экспортная пошлина на нефть': 'customs_duty',
                      'Коэффициент Кман для расчета НДПИ и акциза, руб./т': 'K_man',
                      'КАБДТ с учетом НБУГ для расчета НДПИ на нефть, руб./т': 'K_dt',
                      'Базовая ставка НДПИ на нефть, руб/т': 'base_rate_MET',
                      'Транспортные расходы для м/р районов сдачи нефти (т.е. региона РФ, где расположен СИКН, '
                      'на который сдается нефть с данного м/р) - Республики Башкортостан, Республики Коми, '
                      'Удмуртской Республики, Пермского края, Тюменской области, Ненецкого автономного округа, '
                      'Ханты-Мансийского автономного округа - Югры; Ямало-Ненецкого автономного округа '
                      '(для участков недр, расположенных полностью или частично севернее 65 градуса северной широты, '
                      'южнее 70 градуса северной широты и западнее 80 градуса восточной долготы в границах '
                      'Ямало-Ненецкого автономного округа) за исключением м/р, приведенных ниже': 'cost_transportation',
                      'Ставка дисконтирования по Группе ГПН реальная': 'r'}

name_columns_FPA = "A:D, F, H, I:K, N:P, AM:AS, BA, BB, BD, BG, CG, DL, JW"


def check_economy_data_is_exist(path_to_economy_data: str) -> None:
    """Функция проверяет наличие данных, необходимых для расчета экономики"""

    if not os.path.isdir(path_to_economy_data) or len(os.listdir(path_to_economy_data)) == 0:
        logger.warning("Не загружены данные по экономике")
        sys.exit()

    if "НРФ.xlsx" not in os.listdir(path=path_to_economy_data):
        raise FileExistsError("НРФ.xlsx")
    elif "Макра_долгосрочная.xlsx" not in os.listdir(path=path_to_economy_data):
        raise FileExistsError("Макра_долгосрочная.xlsx")
    elif "Макра_оперативная_БП.xlsx" not in os.listdir(path=path_to_economy_data):
        raise FileExistsError("Макра_оперативная_БП.xlsx")
    elif "Макра_оперативная_текущий_год.xlsx" not in os.listdir(path=path_to_economy_data):
        raise FileExistsError("Макра_оперативная_текущий_год.xlsx")

    return


def preparation_macroeconomics(df_macroeconomics: pd.DataFrame, dict_columns: dict) -> pd.DataFrame:
    """Предварительная обработка файла Макра_оперативная_текущий_год"""

    df_temp = df_macroeconomics
    df_temp.columns = ["Параметр", df_temp.columns[1], df_temp.columns[2]]
    df_temp = df_temp[df_temp["Параметр"].isin(dict_columns.keys())]
    df_temp.replace(dict_columns, inplace=True)

    return df_temp


def preparation_business_plan(df_business_plan: pd.DataFrame, dict_columns: dict) -> pd.DataFrame:
    """Предварительная обработка файла Макра_оперативная_БП"""

    df_temp = df_business_plan
    df_temp.drop([0], inplace=True)
    df_temp.columns = ["Параметр"] + list(df_temp.columns[1:])
    df_temp = df_temp[df_temp["Параметр"].isin(dict_columns.keys())]
    df_temp.replace(dict_columns, inplace=True)
    df_temp = df_temp.fillna(method='ffill', axis=1).reset_index(drop=True)

    return df_temp


def prepare_long_business_plan(df_long_macroeconomics: pd.DataFrame, dict_columns: dict) -> pd.DataFrame:
    """Предварительная обработка файла Макра_долгосрочная"""

    dt_temp = df_long_macroeconomics
    dt_temp.drop([0], inplace=True)
    dt_temp.columns = ["Параметр"] + list(dt_temp.columns[1:])
    dt_temp = dt_temp[dt_temp["Параметр"].isin(dict_columns.keys())]
    dt_temp.replace(dict_columns, inplace=True)

    return dt_temp


def preparation_coefficients(df_coefficients: pd.DataFrame) -> pd.DataFrame:
    """Предварительная обработка файла НРФ"""

    df_temp = df_coefficients
    df_temp["Наименование участка недр/Общества"] = df_temp["Наименование участка недр/Общества"] \
        .replace(regex={'2': '', '6': '', ' ЮЛ': ''})
    df_temp = df_temp[["Наименование участка недр/Общества", "Кв", "Кз", "Ккан"]]
    df_temp = df_temp.groupby("Наименование участка недр/Общества").mean().reset_index()
    df_temp.rename({"Наименование участка недр/Общества": "Месторождение"}, axis=1, inplace=True)

    return df_temp


def prepare_fpa(df_fpa_nrf: pd.DataFrame, liquid_groups: pd.DataFrame) -> pd.DataFrame:
    """Предварительная обработка файла НРФ"""

    df_tmp = df_fpa_nrf
    df_tmp.drop([0, 1, 2], inplace=True)
    df_tmp.drop(df_tmp.columns[[0, 2, 3]], inplace=True, axis=1)
    df_tmp.reset_index(inplace=True, drop=True)
    df_tmp.columns = ['Месторождение', '№скв.', '№куста', 'ДНС', 'Пласты', 'Состояние',
                      'Дебит жидк., м3/сут', 'Дебит жидк., т/сут', 'Дебит нефти, т/сут', 'Тариф на электроэнергию',
                      'УРЭ на ППД', 'УРЭ на подг. нефти', 'УРЭ Транспорт жидкости', 'УРЭ трансп. нефти',
                      'УРЭ трансп. подт. воды', 'УРЭ внешний транспорт нефти', 'Переменные расходы ППД',
                      'Переменные расходы по подготовке нефти', 'Переменные расходы по транспортировке нефти',
                      'Переменные коммерческие расходы', 'Удельные от нефти', 'Удельный расход ЭЭ на МП', 'Кд']
    df_tmp['Кд'] = df_tmp['Кд'].replace(regex={'ТРИЗ ': '', ",": "."}).astype("float")
    df_tmp['Кд'] = df_tmp['Кд'].replace(0, 1)
    df_tmp["Уделка на нефть, руб/тн.н"] = (
            df_tmp["Тариф на электроэнергию"] * (df_tmp['УРЭ на подг. нефти'] + df_tmp['УРЭ трансп. нефти']
                                                 + df_tmp['УРЭ внешний транспорт нефти'])
            + df_tmp["Переменные расходы по подготовке нефти"] + df_tmp["Переменные расходы по транспортировке нефти"]
            + df_tmp["Переменные коммерческие расходы"] + df_tmp["Удельные от нефти"])
    df_tmp["Уделка на закачку, руб/м3"] = (df_tmp["Тариф на электроэнергию"] * df_tmp["УРЭ на ППД"]
                                           + df_tmp["Переменные расходы ППД"])
    df_tmp["Уделка на жидкость, руб/т"] = df_tmp["Тариф на электроэнергию"] * (df_tmp["УРЭ Транспорт жидкости"]
                                                                               + df_tmp["Удельный расход ЭЭ на МП"])
    df_tmp["Уделка на воду, руб/м3"] = df_tmp["Тариф на электроэнергию"] * df_tmp["УРЭ трансп. подт. воды"]
    df_tmp = df_tmp[['Месторождение', '№скв.', '№куста', 'ДНС', 'Пласты', 'Состояние', 'Дебит жидк., м3/сут',
                     'Дебит жидк., т/сут', 'Дебит нефти, т/сут', "Уделка на нефть, руб/тн.н",
                     'Уделка на закачку, руб/м3', 'Уделка на жидкость, руб/т', "Уделка на воду, руб/м3", 'Кд']]
    df_tmp['№скв.'] = df_tmp['№скв.'].astype("str")

    df_tmp.iloc[:, 6:] = df_tmp.iloc[:, 6:].astype("float")
    df_tmp = df_tmp.rename(columns={'Дебит жидк., т/сут': "Qliq_min"})
    df_tmp = df_tmp.sort_values(by=["Qliq_min"])
    df_tmp["liquid_group"] = pd.merge_asof(df_tmp["Qliq_min"],
                                           liquid_groups, on="Qliq_min", direction="nearest").iloc[:, -1]

    return df_tmp


def add_on_sheet(wb: xw.Book, name: str, df) -> None:
    if name in wb.sheets:
        xw.Sheet[name].delete()
    wb.sheets.add(name)
    sht = wb.sheets(name)
    sht.range('A1').options().value = df

    return
