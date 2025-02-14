import streamlit as st
import plotly.graph_objects as go
from util import create_plot,plot_bins,overall_class_stats,plot_contiguous_blocks,plot_contiguous_blocks_scatter

st.set_page_config(layout="wide",page_icon="📈")

# Check if the dataframe is in the session state
if 'df' not in st.session_state or st.session_state.df is None:
    st.error("Please upload a file on the main page first.")
    st.stop()

df = st.session_state.df

st.title("Data Visualization")

# Add create_plot
st.subheader("Duration of Each Class")
fig_duration = create_plot(df)
st.plotly_chart(fig_duration)

# Add plot_bins
st.subheader("Distribution of Time Duration for Each Class")
selected_class = st.selectbox("Select a class to view its distribution:", options=df["Overall class"].unique())
fig_bins = plot_bins(df, selected_class)
if isinstance(fig_bins, str):
    st.warning(fig_bins)
else:
    st.plotly_chart(fig_bins)

# Overall Class Stats
st.subheader("Overall Class Time Blocks")
selected_class = st.selectbox("Select a class for detailed stats:", options=df["Overall class"].unique())
max_sequence,cnt_arr = overall_class_stats(df, selected_class)

# Slider for threshold
threshold = st.slider(
        "Select threshold for Contigous Block",
        min_value=5,
        max_value=50,
        value=10)
fig_contigous_counts = plot_contiguous_blocks(cnt_arr,threshold)
fig_contigous_counts_scatter = plot_contiguous_blocks_scatter(cnt_arr,threshold)

st.plotly_chart(fig_contigous_counts)

st.plotly_chart(fig_contigous_counts_scatter)

col1, col2 = st.columns(2)

with col1:
    if st.button('Back to Main Page'):
        st.switch_page("main.py")

with col2:
    if st.button("View Blocks Page"):
        st.switch_page("pages/3_Blocks.py")    
