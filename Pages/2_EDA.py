import streamlit as st
from util import dataset_description, overall_class_stats

st.set_page_config(layout="wide")

# Check if the dataframe is in the session state
if 'df' not in st.session_state or st.session_state.df is None:
    st.error("Please upload a file on the main page first.")
    st.stop()

df = st.session_state.df

st.title("Exploratory Data Analysis")

class_counts, duration_str = dataset_description(df)

# Total Duration of Video
st.subheader("Total Duration of Video")
st.write(duration_str)

# Class Count
st.subheader("Count of Each Class")
st.dataframe(class_counts)

# Stats
st.subheader("Column Stats")
st.write(df.describe())

# Overall Class Stats
st.subheader("Overall Class Statistics")
selected_class = st.selectbox("Select a class for detailed stats:", options=df["Overall class"].unique())
max_sequence,cnt_arr = overall_class_stats(df, selected_class)
preview = cnt_arr[:5]
st.write(f"Preview of Segments for {selected_class}: {preview}")
st.write(f"Maximum Consecutive Rows {max_sequence}")

# Moved to Blocks Page for better viewing experience based on the Lab Requirements 
# for easy debuggging experience.
# st.write("All segments:")
# for segment in cnt_arr:
#     st.write(segment)

if st.button("Back to Main Page"):
    st.switch_page("main.py")
