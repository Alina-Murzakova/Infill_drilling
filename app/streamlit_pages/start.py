import streamlit as st

st.title("Мое приложение")

name = st.text_input("Введите имя")

if st.button("Рассчитать"):
    st.write(f"Привет, {name}!")