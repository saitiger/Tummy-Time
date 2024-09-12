import pandas as pd
import streamlit as st
import os 
import time
from util import process_dataset, overall_class_stats  # Import the function from util

st.set_page_config(layout="wide")

if 'df' not in st.session_state:
    st.session_state.df = None

uploaded = st.file_uploader(label='Upload the CSV file', type='csv')

if uploaded is not None:
    with st.spinner("Processing"):
        df = process_dataset(uploaded)
    
    if isinstance(df, str):  
        st.error(df)
    else:
        message_placeholder = st.empty()
        message_placeholder.success('File successfully uploaded and processed.')
        time.sleep(3)
        message_placeholder.empty()  # Clear the success message after 3 seconds
        st.session_state.df = df  

    #---- SIDEBAR ----
    st.sidebar.header("Filter :")

    col1, col2 = st.sidebar.columns(2)
    
    if 'class_type' not in st.session_state:
        st.session_state.class_type = df["Overall class"].unique().tolist()

    if col1.button("Select All"):
        st.session_state.class_type = df["Overall class"].unique().tolist()
    
    if col2.button("Clear All"):
        st.session_state.class_type = []

    class_type = st.sidebar.multiselect(
        "Select the Overall class:",
        options=df["Overall class"].unique(),
        default=st.session_state.class_type
    )

    df_preview = df[df["Overall class"].isin(class_type)].head()

    if df_preview.empty:
        st.warning("No data available based on the current filters!")
        st.stop()

    st.dataframe(df_preview)

    # Create the new filename
    original_filename = os.path.splitext(uploaded.name)[0]
    processed_filename = f"{original_filename}_processed.csv"

     # Add a download button for the complete dataset
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Processed Dataset",
        data=csv,
        file_name=processed_filename,
        mime="text/csv",
    )

    col1, col2,col3 = st.columns(3)
    with col1:
        if col1.button("View Plots"):
            st.switch_page("pages/1_Plots.py")
    with col2:
        if col2.button("View EDA"):
            st.switch_page("pages/2_EDA.py")
    with col3:
        if col3.button("View Blocks"):
            st.switch_page("pages/3_Blocks.py")

    # Depreciated 
    # if st.sidebar.button("Generate Class Statistics"):
    #     overall_class = st.sidebar.selectbox("Select Overall Class for Stats", options=df["Overall class"].unique())
        
    #     file_path = 'results.txt'
    #     cnt_arr, max_sequence = overall_class_stats(df, overall_class, file_path)
        
#         with open(file_path, 'r') as file:
#             st.download_button(
#                 label="Download Class Statistics",
#                 data=file,
#                 file_name=file_path,
#                 mime='text/plain'
#             )

else:
    st.warning("Please upload a CSV file to get started.")
