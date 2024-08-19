import streamlit as st
import matplotlib.pyplot as plt
from util import dataset_description, overall_class_stats

st.set_page_config(layout="wide")

# Check if the dataframe is in the session state
if 'df' not in st.session_state or st.session_state.df is None:
    st.error("Please upload a file on the main page first.")
    st.stop()

df = st.session_state.df

st.title("EDA")

c,t = dataset_description(df)

# Total Duration of Video
st.subheader("Total Duration of Video")
st.write(t)

# Class Count
st.subheader("Count of Each Class")
st.write(c)

# Stats
st.subheader("Column Stats")
st.write(df.describe())

if st.button("Back to Main Page"):
    st.switch_page("pages/main.py")