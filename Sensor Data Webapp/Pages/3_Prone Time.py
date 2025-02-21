import streamlit as st
import plotly.graph_objs as go

from util import plot_sensor_data

st.set_page_config(layout="wide")

# Check if the dataframe is in the session state
if 'df' not in st.session_state or st.session_state.df is None:
    st.error("Please upload a file on the main page first.")
    st.stop()

df = st.session_state.df

st.title("Sensor Data Over Time")

st.subheader("Acceleration and X-axis")

# Generate and store the plot in session state
if 'df_plot' not in st.session_state or 'plotly_fig' not in st.session_state:
    # plot_sensor_data returns both dataframe and figure
    df_plot = plot_sensor_data(df)[0]
    plotly_fig = plot_sensor_data(df)[1]
    st.session_state.df_plot = df_plot
    st.session_state.plotly_fig = plotly_fig

# Display the plot
plotly_fig = st.session_state.plotly_fig
st.plotly_chart(plotly_fig, use_container_width=False)

st.subheader("Acceleration Filter")
acc_filter = st.slider(
        "Acceleration Filter",
        min_value=0.5,
        max_value=3.0,
        value=1.5)
st.write(df[df['Acceleration'] > acc_filter])

# if st.button('Back to Main Page'):
    # st.switch_page("main.py")