import pandas as pd
import streamlit as st
import os 
import time
from util import process_dataset, overall_class_stats

st.set_page_config(layout="wide")

# Initialize session state for dataframe
if 'df' not in st.session_state:
    st.session_state.df = None

# File uploader
uploaded = st.file_uploader(label='Upload the CSV file', type='csv')

if uploaded is not None:
    with st.spinner("Processing"):
        df = process_dataset(uploaded)
    # Handle processing errors
    if isinstance(df, str):  
        st.error(df)
    else:
        message_placeholder = st.empty()
        message_placeholder.success('File successfully uploaded and processed.')
        time.sleep(3)
        message_placeholder.empty()  # Clear the success message after 3 seconds
        st.session_state.df = df  

    # Sidebar for filtering
    st.sidebar.header("Filter :")

    col1, col2 = st.sidebar.columns(2)
    
    if 'class_type' not in st.session_state:
        st.session_state.class_type = df["Overall class"].unique().tolist()

    # Select All and Clear All buttons
    if col1.button("Select All"):
        st.session_state.class_type = df["Overall class"].unique().tolist()
    
    if col2.button("Clear All"):
        st.session_state.class_type = []

    # Multiselect for class filtering
    class_type = st.sidebar.multiselect(
        "Select the Overall class:",
        options=df["Overall class"].unique(),
        default=st.session_state.class_type
    )

    # Display filtered dataframe preview
    df_preview = df[df["Overall class"].isin(class_type)].head()

    if df_preview.empty:
        st.warning("No data available based on the current filters!")
        st.stop()

    st.dataframe(df_preview)

    # Create download button for processed dataset
    original_filename = os.path.splitext(uploaded.name)[0]
    processed_filename = f"{original_filename}_processed.csv"
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Processed Dataset",
        data=csv,
        file_name=processed_filename,
        mime="text/csv",
    )

    # Navigation buttons
    col1, col2, col3,col4 = st.columns(4)
    with col1:
        if col1.button("View Plots"):
            st.switch_page("pages/1_Plots.py")
    with col2:
        if col2.button("View EDA"):
            st.switch_page("pages/2_EDA.py")
    with col3:
        if col3.button("Prone Time Visualization"):
            st.switch_page("pages/3_Prone.py")  
    with col4:
        if col4.button("Sensor Data Over Time"):
            st.switch_page("pages/4_Sensor Data Over Time.py")                     
    
    ### Depreciated : Only for inital analysis done using jupyter notebook
    # with col3:
        # if col3.button("View Blocks"):
            # st.switch_page("pages/3_Blocks.py")
    # with col4:
        # if col4.button("View Blocks Over Time Plot"):
            # st.switch_page("pages/4_Blocks_Over_Time.py")

else:
    st.warning("Please upload a CSV file to get started.")
