import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, time
from util import tummy_time_duration, plot_exercise_durations

st.set_page_config(layout="wide")

def validate_datetime(input_datetime, min_datetime, max_datetime):
    """Validate if input datetime is within the dataset range"""
    if input_datetime < min_datetime or input_datetime > max_datetime:
        st.error(f"Please input datetime between {min_datetime.strftime('%Y-%m-%d %H:%M:%S')} and {max_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        return False
    return True

# Initialize default values in session state if not present
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

# Initialize parameters with default values if not in session state
if 'default_values_set' not in st.session_state:
    st.session_state.wake_time = "06:00"
    st.session_state.sleep_time = "20:00"
    st.session_state.min_size = 60
    st.session_state.no_filters = False
    st.session_state.default_values_set = True

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

# Get the date range from the dataset
df_plot['A'] = pd.to_datetime(df_plot['A'])
min_datetime = df_plot['A'].min()
max_datetime = df_plot['A'].max()

# Display the date range information at the top
st.header("Time")
st.info(f"Data Collection starts from {min_datetime.strftime('%Y-%m-%d %H:%M:%S')} to {max_datetime.strftime('%Y-%m-%d %H:%M:%S')}")

# Add buttons for reset and no filter
col_buttons = st.columns(2)
with col_buttons[0]:
    if st.button("Reset All Filters"):
        st.session_state.wake_time = None
        st.session_state.sleep_time = None
        st.session_state.no_filters = False
        st.rerun()

with col_buttons[1]:
    if st.button("No Filters"):
        st.session_state.no_filters = True
        st.rerun()

if not st.session_state.no_filters:
    # Create two columns for sleep/wake time inputs
    col3, col4 = st.columns(2)

    with col3:
        wake_time_str = st.text_input(
            "Enter wake time (HH:MM)",
            value="06:00"
        )
        try:
            wake_time = datetime.strptime(wake_time_str, "%H:%M").strftime("%H:%M")
        except ValueError:
            st.error("Please enter time in HH:MM format")
            wake_time = None

    with col4:
        sleep_time_str = st.text_input(
            "Enter sleep time (HH:MM)",
            value="20:00"
        )
        try:
            sleep_time = datetime.strptime(sleep_time_str, "%H:%M").strftime("%H:%M")
        except ValueError:
            st.error("Please enter time in HH:MM format")
            sleep_time = None

    # Input for minimum duration with default value
    min_size = st.number_input(
        "Minimum duration (seconds):",
        min_value=1,
        value=60,
        step=1
    )

else:
    # When no filters are active
    wake_time = None
    sleep_time = None
    min_size = 60

# Input for prone tolerance value
prone_tolerance_value = st.text_input(
    "Enter Prone Tolerance Value (format: Xm Ys, e.g., 9m 30s):",
    value="9m 30s"
)

# # Input for number of rows to exclude (Old Code for excluding bars)
# exclude_rows = st.number_input(
#     "Enter how many of the first few bars to exclude:",
#     min_value=0,
#     value=0,
#     step=1
# )

removed_bars_input = st.text_input(
    "Enter bar numbers to remove (comma-separated, e.g., 1,2,5):", 
    value=""
)

# Parse the input string into a list of integers
try:
    removed_bars = [int(x.strip()) for x in removed_bars_input.split(',')] if removed_bars_input else []
except ValueError:
    st.error("Please enter valid numbers separated by commas")
    removed_bars = []

# Compute tummy time durations with all parameters
buckets, bucket_ls, durations = tummy_time_duration(
    df_plot,
    min_size=min_size,
    sleep_time=sleep_time,
    wake_time=wake_time
)

# Plot with the option to exclude specific bars
fig = plot_exercise_durations(prone_tolerance_value, durations, removed_bars)

# Display the Plotly figure
st.plotly_chart(fig, use_container_width=True)
