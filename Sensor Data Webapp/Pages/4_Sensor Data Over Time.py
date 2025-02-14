import streamlit as st
import pandas as pd
import plotly.express as px

from util import tummy_time_duration, plot_exercise_durations

st.set_page_config(layout="wide")

if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

# Check if the dataframe is in the session state
if 'df' not in st.session_state or st.session_state.df is None:
    st.error("Please upload a file on the main page first.")
    st.stop()

df = st.session_state.df

# Check if the df_plot is in the session state
if 'df_plot' not in st.session_state or st.session_state.df_plot is None:
    st.error("Please process the file on the Prone page first!")
    st.stop()

df_plot = st.session_state.df_plot

# Compute tummy time durations
buckets, bucket_ls, durations = tummy_time_duration(df_plot, min_size=60)

# Input for prone tolerance value
prone_tolerance_value = st.text_input(
    "Enter Prone Tolerance Value (format: Xm Ys, e.g., 9m 30s):",
    value="9m 30s"
)

# Input for number of rows to exclude
exclude_rows = st.number_input(
    "Enter how many of the first few bars to exclude:",
    min_value=0,
    value=0,
    step=1
)

# Plot with the option to exclude rows
fig = plot_exercise_durations(prone_tolerance_value, durations[exclude_rows:])

# Display the Plotly figure
st.plotly_chart(fig, use_container_width=True)

if st.button('Back to Main Page'):
    st.switch_page("main.py")