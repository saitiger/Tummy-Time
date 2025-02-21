# Old 
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


# Adding Date pick for sensor put on and off 
# Adding a filter for sleep and wake time 
# Excluding Rows still retained for anamolies (Defaults to zero)

# New 

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

# Create two columns for the datetime inputs
col1, col2 = st.columns(2)

with col1:
    start_datetime = st.date_input("Select sensor start date", value=None)
    if start_datetime:
        start_time = st.time_input("Select sensor start time", value=time(8, 0))
        start_datetime = datetime.combine(start_datetime, start_time)

with col2:
    end_datetime = st.date_input("Select sensor end date", value=None)
    if end_datetime:
        end_time = st.time_input("Select sensor end time", value=time(20, 0))
        end_datetime = datetime.combine(end_datetime, end_time)

# Create two columns for the daily sleep/wake time inputs
col3, col4 = st.columns(2)

with col3:
    wake_time = st.time_input("Daily wake time", value=time(6, 0))
    wake_time = wake_time.strftime("%H:%M")

with col4:
    sleep_time = st.time_input("Daily sleep time", value=time(20, 0))
    sleep_time = sleep_time.strftime("%H:%M")

# Input for minimum duration
min_size = st.number_input(
    "Minimum duration (seconds):",
    min_value=1,
    value=60,
    step=1
)

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

# Compute tummy time durations with all parameters
buckets, bucket_ls, durations = tummy_time_duration(
    df_plot,
    min_size=min_size,
    start_datetime=start_datetime,
    end_datetime=end_datetime,
    sleep_time=sleep_time,
    wake_time=wake_time
)

# Plot with the option to exclude rows
fig = plot_exercise_durations(prone_tolerance_value, durations[exclude_rows:])

# Display the Plotly figure
st.plotly_chart(fig, use_container_width=True)

# Error with wrong date picker {Gives options which are not available in the dataset}

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, time
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

# Get the date range from the dataset
df_plot['A'] = pd.to_datetime(df_plot['A'])
min_date = df_plot['A'].min().date()
max_date = df_plot['A'].max().date()

# Display the date range information at the top
st.header("Time")
st.info(f"Data Collection starts from {min_date} to {max_date}")

# Create two columns for the datetime inputs
col1, col2 = st.columns(2)

with col1:
    start_datetime = st.date_input(
        "Select sensor start date",
        value=min_date,
        min_value=min_date,
        max_value=max_date
    )
    if start_datetime:
        # Get the min and max times for the selected date
        date_data = df_plot[df_plot['A'].dt.date == start_datetime]
        min_time = date_data['A'].min().time()
        max_time = date_data['A'].max().time()
        start_time = st.time_input(
            "Select sensor start time",
            value=min_time
        )
        start_datetime = datetime.combine(start_datetime, start_time)

with col2:
    # Ensure end_date is not before start_date
    end_datetime = st.date_input(
        "Select sensor end date",
        value=max_date,
        min_value=start_datetime if start_datetime else min_date,
        max_value=max_date
    )
    if end_datetime:
        # Get the min and max times for the selected date
        date_data = df_plot[df_plot['A'].dt.date == end_datetime]
        min_time = date_data['A'].min().time()
        max_time = date_data['A'].max().time()
        end_time = st.time_input(
            "Select sensor end time",
            value=max_time
        )
        end_datetime = datetime.combine(end_datetime, end_time)

# Create two columns for the daily sleep/wake time inputs
col3, col4 = st.columns(2)

with col3:
    wake_time = st.time_input("Daily wake time", value=time(6, 0))
    wake_time = wake_time.strftime("%H:%M")

with col4:
    sleep_time = st.time_input("Daily sleep time", value=time(20, 0))
    sleep_time = sleep_time.strftime("%H:%M")

# Input for minimum duration
min_size = st.number_input(
    "Minimum duration (seconds):",
    min_value=1,
    value=60,
    step=1
)

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

# Compute tummy time durations with all parameters
buckets, bucket_ls, durations = tummy_time_duration(
    df_plot,
    min_size=min_size,
    start_datetime=start_datetime,
    end_datetime=end_datetime,
    sleep_time=sleep_time,
    wake_time=wake_time
)

fig = plot_exercise_durations(prone_tolerance_value, durations[exclude_rows:])

st.plotly_chart(fig, use_container_width=True)
