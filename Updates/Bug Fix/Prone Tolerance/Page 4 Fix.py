import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
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

# Create columns for the input parameters
col1, col2, col3 = st.columns(3)

with col1:
    # Input for minimum duration (min_size)
    min_size = st.number_input(
        "Minimum Duration (seconds):",
        min_value=1,
        value=60,
        help="Minimum duration in seconds for a sequence to be considered valid"
    )

with col2:
    # Input for start time
    start_time = st.time_input(
        "Start Time (optional):",
        value=None,
        help="Filter data to only include readings after this time"
    )
    # Convert time input to string format required by the function
    start_time_str = start_time.strftime("%H:%M") if start_time else None

with col3:
    # Input for end time
    end_time = st.time_input(
        "End Time (optional):",
        value=None,
        help="Filter data to only include readings before this time"
    )
    # Convert time input to string format required by the function
    end_time_str = end_time.strftime("%H:%M") if end_time else None

# Validate time inputs
if start_time and end_time and start_time > end_time:
    st.error("Start time must be before end time!")
    st.stop()

# Compute tummy time durations with the new parameters
buckets, bucket_ls, durations = tummy_time_duration(
    df_plot, 
    min_size=min_size,
    start_time=start_time_str,
    end_time=end_time_str
)

# Create columns for the remaining inputs
col4, col5 = st.columns(2)

with col4:
    # Input for prone tolerance value
    prone_tolerance_value = st.text_input(
        "Enter Prone Tolerance Value (format: Xm Ys, e.g., 9m 30s):",
        value="9m 30s"
    )

with col5:
    # Input for number of rows to exclude
    exclude_rows = st.number_input(
        "Enter how many of the first few bars to exclude:",
        min_value=0,
        value=0,
        step=1
    )

# Check if we have any durations to plot
if not durations:
    st.warning("No valid tummy time durations found with the current parameters.")
    st.stop()

# Plot with the option to exclude rows
fig = plot_exercise_durations(prone_tolerance_value, durations[exclude_rows:])

# Display the Plotly figure
st.plotly_chart(fig, use_container_width=True)

# if st.button('Back to Main Page'):
    # st.switch_page("main.py")
