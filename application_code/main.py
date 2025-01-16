import streamlit as st
from frequency_deployment import frequency
from fleet_deployment import fleet
from eta_tab import  eta_tab
st.set_page_config(layout="wide")

st.title("Maritime AI Awarweness")

tab1, tab2,tab3 = st.tabs(["Frequency Deployment", "Fleet Deployment","ETA"])

with tab1:
    frequency()

with tab2:
    fleet()

with tab3:
    eta_tab()
    print("ETA")