import pandas as pd
import streamlit as st
from util import process_dataset, display_dataset

st.set_page_config(layout="wide")

if 'df' not in st.session_state:
    st.session_state.df = None

uploaded = st.file_uploader(label='Upload the CSV file', type='csv')

if uploaded is not None:
    df = process_dataset(uploaded)
    
    if isinstance(df, str):  
        st.error(df)
    else:
        st.success('File successfully uploaded and processed.')
        st.session_state.df = df 
        # dp = display_dataset(df)

    #---- SIDEBAR ----
    st.sidebar.header("Filter :")

    col1, col2 = st.sidebar.columns(2)
    
    if 'class_type' not in st.session_state:
        st.session_state.class_type = df["Overall class"].unique().tolist()

    if col1.button("Select All"):
        st.session_state.class_type = df["Overall class"].unique().tolist()
    
    if col2.button("Clear All"):
        st.session_state.class_type = []

    class_type = st.sidebar.multiselect(
        "Select the Overall class:",
        options=df["Overall class"].unique(),
        default=st.session_state.class_type
    )

    df_selection = df[df["Overall class"].isin(class_type)].head()

    if df_selection.empty:
        st.warning("No data available based on the current filters!")
        st.stop()

    st.dataframe(df_selection)

    col1, col2 = st.columns(2)
    if col1.button("View Plots"):
        st.switch_page("pages/plots.py")
    if col2.button("View EDA"):
        st.switch_page("pages/EDA.py")

else:
    st.warning("Please upload a CSV file to get started.")
