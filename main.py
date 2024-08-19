import pandas as pd
import streamlit as st
from util import process_dataset, display_dataset

st.set_page_config(layout="wide")

# Create a session state to store the dataframe
if 'df' not in st.session_state:
    st.session_state.df = None

uploaded = st.file_uploader(label='Upload the excel file', type='csv')

if uploaded is not None:
    st.success('File successfully uploaded and processed.')

    df = process_dataset(uploaded)
    st.session_state.df = df  # Store the dataframe in session state
    dp = display_dataset(df)

    #---- SIDEBAR ----
    st.sidebar.header("Filter :")

    # Add "Select All" and "Clear All" buttons
    col1, col2 = st.sidebar.columns(2)
    
    if 'class_type' not in st.session_state:
        st.session_state.class_type = dp["Overall class"].unique().tolist()

    if col1.button("Select All"):
        st.session_state.class_type = dp["Overall class"].unique().tolist()
    
    if col2.button("Clear All"):
        st.session_state.class_type = []

    class_type = st.sidebar.multiselect(
        "Select the Overall class:",
        options=dp["Overall class"].unique(),
        default=st.session_state.class_type
    )

    dp_selection = dp[dp["Overall class"].isin(class_type)]

    # Checking if the dataframe is empty:
    if dp_selection.empty:
        st.warning("No data available based on the current filters!")
        st.stop()

    # Display the filtered dataframe
    st.dataframe(dp_selection)

    # st.dataframe(dp.head())

    # Add a button to navigate to the plots page
    if st.button("View Plots"):
        st.switch_page("pages/plots.py")

else:
    st.warning("Please upload a CSV file to get started.")