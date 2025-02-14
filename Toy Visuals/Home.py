import streamlit as st

st.set_page_config(layout="wide")

st.title("Data Analysis Application")

st.write("""
Toy Data Analysis
         
### Getting Started:
1. Go to the **Validation** page to upload and validate your data
2. Then proceed to the **Plots** page to visualize the analysis

Use the sidebar to navigate between pages.
""")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None