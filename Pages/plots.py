import streamlit as st
import matplotlib.pyplot as plt
from util import create_plot, plot_bins

st.set_page_config(layout="wide")

# Check if the dataframe is in the session state
if 'df' not in st.session_state or st.session_state.df is None:
    st.error("Please upload a file on the main page first.")
    st.stop()

df = st.session_state.df

st.title("Data Visualization")

# Add create_plot
st.subheader("Duration of Each Class")
create_plot(df)

# Add plot_bins
st.subheader("Distribution of Time Duration for Each Class")
selected_class = st.selectbox("Select a class to view its distribution:", options=df["Overall class"].unique())
plot_bins(df, selected_class)

# Add a button to navigate back to the main page
if st.button("Back to Main Page"):
    st.switch_page("pages/main.py")