import pandas as pd
import numpy as np

from loguru import logger

from app.config import (macroeconomics_rows_name, OPEX_rows_name, workover_wellservice_rows_name, apg_rows_name,
                        df_NDPI_NDD)
from app.economy.financial_model import FinancialEconomicModel
from app.economy.functions import calculation_Kg


@logger.catch
def load_economy_data(economy_path, name_field, gor):
    """ gor - газосодержание для расчета выручки с пнг, м3/т"""
    # Инициализируем необходимые переменные
    with pd.ExcelFile(economy_path) as xls:
        # коэффициенты Кв Кз Кд	Кдв	Ккан Кк	Кман Кабдт
        constants = pd.read_excel(xls, sheet_name="Налоги")
        macroeconomics = pd.read_excel(xls, sheet_name="Макропараметры", header=3)  # Основная макра

        # месторождения с НДД
        reservoirs_NDD = pd.read_excel(xls, sheet_name="МР с НДД")

        # OPEX
        df_opex = pd.read_excel(xls, sheet_name="Удельный OPEX", header=0)
        df_opex = df_opex[df_opex['Месторождение'] == name_field]
        if df_opex.shape[0] < 3:
            logger.error(f"В исходных данных  ФЭМ нет OPEX по месторождению {name_field}")
            return None
        else:
            del df_opex['Месторождение']

        # CAPEX
        df_capex = pd.read_excel(xls, sheet_name="CAPEX", header=0)
        # Определим цену одной стадии ГРП по массе пропанта
        df_cost_GRP = pd.read_excel(xls, sheet_name="ГРП_цена", header=0)
        # Интерполируем цену
        cost_stage_GRP = np.interp(df_capex.iloc[6, 1], df_cost_GRP['Тонн'], df_cost_GRP['Цена за операцию ГРП, тыс '
                                                                                         'руб. без НДС'])
        df_capex.iloc[6, 1] = cost_stage_GRP
        df_capex.iloc[6, 0] = 'Цена за 1 стадию ГРП, тыс руб'

        # Уд_ОНВСС_бурение
        df_ONVSS_cost_ed = pd.read_excel(xls, sheet_name="Уд_ОНВСС_бурение", header=0)
        df_ONVSS_cost_ed = df_ONVSS_cost_ed[df_ONVSS_cost_ed['Месторождение'] == name_field]
        if df_ONVSS_cost_ed.empty:
            logger.error(f"В исходных данных ФЭМ нет Уд_ОНВСС_бурение по месторождению {name_field}")
            return None
        else:
            ONVSS_cost_ed = df_ONVSS_cost_ed.iloc[0, 1]

        # потери нефти
        df_oil_loss = pd.read_excel(xls, sheet_name="Нормативы потерь нефти", header=0)
        df_oil_loss = df_oil_loss[df_oil_loss['Месторождение'] == name_field]
        if df_oil_loss.empty:
            logger.error(f"В исходных данных ФЭМ нет потерь нефти по месторождению {name_field}")
            return None
        else:
            del df_oil_loss['Месторождение']
            df_oil_loss = df_oil_loss.set_index([pd.Index(['oil_loss'])])

        # КРС_ПРС
        df_workover_wellservice = pd.read_excel(xls, sheet_name="КРС_ПРС", header=0)
        df_workover_wellservice = df_workover_wellservice[df_workover_wellservice['Месторождение'] == name_field]
        if df_workover_wellservice.shape[0] < 5:
            logger.error(f"В исходных данных ФЭМ нет КРС_ПРС по месторождению {name_field}")
            return None
        else:
            del df_workover_wellservice['Месторождение']

        # Определим цену ПНГ в зависимости от КС
        df_APG_CS = pd.read_excel(xls, sheet_name="ПНГ_КС", header=0)
        df_APG_CS = df_APG_CS[df_APG_CS['Месторождение'] == name_field]
        if df_APG_CS.shape[0] < 1:
            logger.error(f"В исходных данных ФЭМ нет привязки месторождения {name_field} к КС")
            return None
        else:
            price_APG = df_APG_CS['Цена ПНГ (макра)'].iloc[0]

        # ПНГ
        df_apg = pd.read_excel(xls, sheet_name="ПНГ", header=0)
        df_apg = df_apg[df_apg['Месторождение'] == name_field]
        if df_apg.shape[0] < 3:
            logger.error(f"В исходных данных ФЭМ нет данных ПНГ по месторождению {name_field}")
            return None
        else:
            del df_apg['Месторождение']

        # Схема расчета налогов
    method = "ДНС"
    dict_NDD = {'initial_recoverable_reserves': None,
                'cumulative_production': None,
                'Kg_group': None}

    name_row_NDPI_NDD = None
    if name_field in reservoirs_NDD['Месторождение'].values.tolist():
        method = "НДД"
        initial_recoverable_reserves = constants.iloc[5, 1]
        cumulative_production = constants.iloc[6, 1]
        Kg_group = reservoirs_NDD[reservoirs_NDD['Месторождение'] == name_field]['Кг_номер группы'].iloc[0]

        dict_NDD = {'initial_recoverable_reserves': initial_recoverable_reserves,
                    'cumulative_production': cumulative_production,
                    'Kg_group': Kg_group}

        Kg = calculation_Kg(Kg_group, pd.Series(cumulative_production / initial_recoverable_reserves)).values[0]
        row_NDPI_NDD = df_NDPI_NDD[2]
        name_row_NDPI_NDD = row_NDPI_NDD[row_NDPI_NDD.index <= Kg].iloc[-1]

    # Подготовка файлов
    name_first_column = macroeconomics.columns[0]
    macroeconomics = macroeconomics.iloc[:, ~macroeconomics.columns.str.match('Unnamed').fillna(False)]
    # Определим цену ПНГ в зависимости от КС
    macroeconomics_rows_name[price_APG] = 'price_APG'
    macroeconomics_rows_name[name_row_NDPI_NDD] = 'NDPI_NDD'
    macroeconomics = macroeconomics[macroeconomics[name_first_column].isin(macroeconomics_rows_name.keys())]
    macroeconomics.replace(macroeconomics_rows_name, inplace=True)
    macroeconomics = macroeconomics.fillna(method='ffill', axis=1).reset_index(drop=True)
    macroeconomics = formatting_df_economy(macroeconomics)

    df_opex.replace(OPEX_rows_name, inplace=True)
    df_opex = formatting_df_economy(df_opex)

    oil_loss = df_oil_loss.T

    df_workover_wellservice.replace(workover_wellservice_rows_name, inplace=True)
    df_workover_wellservice = formatting_df_economy(df_workover_wellservice)

    df_apg.replace(apg_rows_name, inplace=True)
    df_apg = formatting_df_economy(df_apg)

    if gor < 0:
        gor = 300  # при отсутствии значения газосодержания в ГФХ
    FEM = FinancialEconomicModel(macroeconomics, constants,
                                 df_opex, oil_loss, df_capex, ONVSS_cost_ed,
                                 df_workover_wellservice, df_apg, gor)
    return FEM, method, dict_NDD


def formatting_df_economy(df):
    df = df.T
    df.columns = df.iloc[0]
    df.drop(df.index[0], inplace=True)
    return df