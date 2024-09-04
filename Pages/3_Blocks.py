import streamlit as st
from util import overall_class_stats

st.set_page_config(layout="wide")

# Check if the dataframe is in the session state
if 'df' not in st.session_state or st.session_state.df is None:
    st.error("Please upload a file on the main page first.")
    st.stop()

df = st.session_state.df

st.title("Blocks")

# Overall Class Stats
st.subheader("Overall Class Time Blocks")
selected_class = st.selectbox("Select a class for detailed stats:", options=df["Overall class"].unique())
max_sequence,cnt_arr = overall_class_stats(df, selected_class)

st.write("Consecutive Blocks :")
for segment in cnt_arr:
    st.write(segment)

if st.button('Back to Main Page'):
    st.switch_page("main.py")
