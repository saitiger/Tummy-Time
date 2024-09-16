import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from util import parse_datetime, create_data_blocks, plot_block

st.set_page_config(layout="wide", page_title="Blocks Over Time")

# Check if the dataframe is in the session state
if 'df' not in st.session_state:
    st.error("Please upload a file on the main page first.")
    st.stop()

@st.cache_data
def load_data():
    return st.session_state.df

df = load_data()

# Diagnostic information
# st.subheader("Diagnostic Information")
# st.write("DataFrame shape:", df.shape)
# st.write("DataFrame columns:", df.columns.tolist())
# st.write("Data types of all columns:")
# st.write(df.dtypes)

# st.write("\nInformation about 'A' column:")
# st.write("Data type of 'A' column:", df['A'].dtype)
# st.write("First few values of 'A' column:", df['A'].head())
# st.write("Number of null values in 'A' column:", df['A'].isnull().sum())
# st.write("Number of non-null values in 'A' column:", df['A'].notnull().sum())

# if df['A'].notnull().sum() > 0:
#     st.write("Sample of non-null values in 'A' column:")
#     st.write(df[df['A'].notnull()]['A'].head())


st.title("Blocks Over Time")

# Convert 'A' column to datetime
df['A'] = df['A'].apply(parse_datetime)

min_date = df['A'].min().date()
max_date = df['A'].max().date()

if min_date == max_date:
    max_date += timedelta(days=1)

start_date = st.date_input("Select start date", min_value=min_date, max_value=max_date, value=min_date)
start_time = st.time_input("Select start time", value=datetime.min.time())

start_datetime = datetime.combine(start_date, start_time)

# Create blocks
blocks = create_data_blocks(df, start_datetime)

# Display blocks
if not blocks:
    st.warning("No data blocks found after the selected start time.")
else:
    for i, block in enumerate(blocks):
        # with st.expander(f"Block {i+1}"):
        fig = plot_block(block)
        st.plotly_chart(fig, use_container_width=True)
        st.write("Summary Statistics:")
        st.write(block['Overall class'].value_counts())

if st.button('Back to Main Page'):
    st.switch_page("main.py")

# @st.cache_data
# def load_and_process_data(file):
#     df['A'] = df['A'].apply(parse_datetime)
#     return df

# @st.cache_data
# def create_blocks_parallel(df, start_datetime, block_size=50000):
#     with mp.Pool(processes=mp.cpu_count()) as pool:
#         blocks = pool.starmap(create_data_blocks, [(df, start_datetime, block_size)])
#     return blocks[0]

# # Check if the dataframe is in the session state
# if 'df' not in st.session_state or st.session_state.df is None:
#     st.error("Please upload a file on the main page first.")
#     st.stop()

# df = st.session_state.df

# df = load_and_process_data(st.session_state.df)

# st.title("Blocks Over Time")


# # Attempt to convert 'A' column to datetime using the custom parser
# try:
#     df['A'] = df['A'].apply(parse_datetime)
#     # st.success("Successfully processed 'A' column as datetime.")
# except Exception as e:
#     st.error(f"Error processing 'A' column: {str(e)}")
#     st.write("Unable to process 'A' column as datetime. Please check the format of your data.")
    
#     # problematic_values = df[pd.to_datetime(df['A'], errors='coerce').isnull()]['A'].head()
#     # st.write("Sample of problematic values:")
#     # st.write(problematic_values)
    
#     st.stop()

# # Check for any valid datetime values
# if df['A'].notna().sum() == 0:
#     st.error("No valid datetime values found in the 'A' column. Please check your data.")
#     st.stop()

# min_date = df['A'].min().date()
# max_date = df['A'].max().date()

# if min_date == max_date:
#     max_date += timedelta(days=1)

# start_date = st.date_input("Select start date", min_value=min_date, max_value=max_date, value=min_date)
# start_time = st.time_input("Select start time", value=datetime.min.time())

# start_datetime = datetime.combine(start_date, start_time)

# # Create blocks using parallel processing
# blocks = create_blocks_parallel(df, start_datetime)

# # Display blocks with lazy loading
# if not blocks:
#     st.warning("No data blocks found after the selected start time.")
# else:
#     for i, block in enumerate(blocks):
#         if st.button(f"Load Block {i+1}"):
#             fig = plot_block(block)
#             st.plotly_chart(fig, use_container_width=True)

#             st.write("Summary Statistics:")
#             st.write(block['Overall class'].value_counts())
#             st.markdown("---")

# if st.button('Back to Main Page'):
#     st.switch_page("main.py")
