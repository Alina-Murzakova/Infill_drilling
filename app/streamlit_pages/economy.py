from datetime import datetime

import streamlit as st


def show():
    """Economics is intentionally unavailable in the web application."""

    st.session_state.switch_economy = False
    st.header("Экономика")
    st.info(
        "Экономический расчёт в веб-версии отключён. Расчёт зон бурения и профилей добычи "
        "выполняется без экономических показателей.",
        icon=":material/info:",
    )
    return {
        "switch_economy": False,
        "start_date": datetime(datetime.now().year + 1, 1, 1),
        "day_in_month": 29.0,
        "path_economy": "-",
    }
