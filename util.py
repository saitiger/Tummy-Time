import pandas as pd
import numpy as np
import streamlit as st
import csv
from functools import partial
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import io
from datetime import datetime, timedelta

csv.field_size_limit(int(1e9)) 

# Depreciated 
# def processed_file_analysis(file):
#     """
#     Analyze a processed CSV file.
    
#     :param file: Streamlit UploadedFile object
#     :return: Processed DataFrame
#     """
#     df = pd.read_csv(file, skiprows=99)
#     df.rename(columns={'Unnamed: 0': 'Timing'}, inplace=True)
#     df['Timing'] = pd.to_datetime(df['Timing'], format='%Y-%m-%d %H:%M:%S:%f')
#     df['Milliseconds'] = df['Timing'].dt.microsecond // 1000
#     df['Time Diff'] = df['Milliseconds'].diff().fillna(10.0)
#     df.loc[df['Time Diff'] < 0, 'Time Diff'] = 10.0
    
#     return df

# Depreciated 
# def process_row(row, S5=140):
#     try:
#         A = row[0]
#         B, C, D = map(float, row[1:4])
#     except (ValueError, IndexError):
#         return None

#     if B == 0 and C == 0 and D == 0:
#         return None
    
#     angle_360 = np.sign(B) * np.arccos(-D / np.sqrt(B**2 + D**2)) * 180 / np.pi + 180
#     angle_updown = np.arcsin(C / np.sqrt(B**2 + C**2 + D**2)) * 180 / np.pi
#     body_rotation = "supine-recline" if S5 < angle_360 < (S5 + 180) else "prone-sit"
    
#     if body_rotation == "prone-sit":
#         prone_sit_class = np.select([
#             angle_updown > 0,
#             angle_updown > -23,
#             angle_updown > -63
#         ], [
#             "prone",
#             "prone supported",
#             "upright"
#         ], default="sitting")
#         supine_recline_class = ""
#     else:
#         supine_recline_class = np.select([
#             angle_updown > 15,
#             angle_updown < -36,
#             angle_360 < (S5 + 69),
#             angle_360 > (S5 + 101)
#         ], [
#             "upsidedown",
#             "reclined",
#             "left side",
#             "right side"
#         ], default="supine")
#         prone_sit_class = ""
    
#     # Ensure prone_sit_class and supine_recline_class are strings
#     prone_sit_class = str(prone_sit_class) if isinstance(prone_sit_class, np.ndarray) else prone_sit_class
#     supine_recline_class = str(supine_recline_class) if isinstance(supine_recline_class, np.ndarray) else supine_recline_class
    
#     overall_class = (prone_sit_class + supine_recline_class).strip()
    
#     return [A, B, C, D, angle_360, angle_updown, body_rotation, prone_sit_class, supine_recline_class, overall_class]


@st.cache_data(show_spinner=False)
def process_dataset(file):
    """
    Process a large dataset from a CSV file.

    Parameter : Streamlit UploadedFile object
   
    Returns : 
    
    Processed DataFrame
    """
    try:
        df = pd.read_csv(file, encoding='utf-8', skiprows=99, on_bad_lines='skip', header=None)
    except UnicodeDecodeError:
        try:
            # If UTF-8 fails, try 'latin-1'
            df = pd.read_csv(file, encoding='latin-1', skiprows=99, on_bad_lines='skip', header=None)
        except UnicodeDecodeError:
            return "Error: Unable to decode the file. Please ensure it's a valid CSV."

    # Keep only the first 4 columns
    df = df.iloc[:, :4]
    df.columns = ['A', 'B', 'C', 'D']

    # Add new columns with default values
    df['360 angle'] = np.nan
    df['Up/down angle'] = np.nan
    df['Body Rotation'] = ""
    df['Prone-sit class'] = ""
    df['Supine-recline class'] = ""
    df['Overall class'] = ""

    S5 = 140

    # Calculate '360 angle' and 'Up/down angle' with error handling for division by zero
    df['360 angle'] = np.where(
        (df['B']**2 + df['D']**2) != 0,
        np.sign(df['B']) * np.degrees(np.arccos(-df['D'] / np.sqrt(df['B']**2 + df['D']**2))) + 180,
        np.nan
    )

    df['Up/down angle'] = np.where(
        (df['B']**2 + df['C']**2 + df['D']**2) != 0,
        np.degrees(np.arcsin(df['C'] / np.sqrt(df['B']**2 + df['C']**2 + df['D']**2))),
        np.nan
    )

    # Determine 'Body Rotation' status
    df['Body Rotation'] = np.where(
        (df['360 angle'] > S5) & (df['360 angle'] < (S5 + 180)),
        'supine-recline',
        'prone-sit'
    )

    # Calculate 'Prone-sit class' using np.select for the entire dataframe
    prone_sit_conditions = [
        df['Up/down angle'] > 0,
        df['Up/down angle'] > -23,
        df['Up/down angle'] > -63
    ]
    prone_sit_choices = ['prone', 'prone supported', 'upright']

    df['Prone-sit class'] = np.select(
        prone_sit_conditions, 
        prone_sit_choices, 
        default='sitting'
    )

    # Apply condition only to rows where Body Rotation is 'prone-sit'
    df.loc[df['Body Rotation'] != 'prone-sit', 'Prone-sit class'] = ""

    # Calculate 'Supine-recline class' using np.select for the entire dataframe
    supine_recline_conditions = [
        df['Up/down angle'] > 15,
        df['Up/down angle'] < -36,
        df['360 angle'] < (S5 + 69),
        df['360 angle'] > (S5 + 101)
    ]
    supine_recline_choices = ['upsidedown', 'reclined', 'left side', 'right side']

    df['Supine-recline class'] = np.select(
        supine_recline_conditions, 
        supine_recline_choices, 
        default='supine'
    )

    # Apply condition only to rows where Body Rotation is 'supine-recline'
    df.loc[df['Body Rotation'] != 'supine-recline', 'Supine-recline class'] = ""

    # Combine 'Prone-sit class' and 'Supine-recline class' to form 'Overall class'
    df['Overall class'] = (df['Prone-sit class'] + ' ' + df['Supine-recline class']).str.strip()

    # Drop rows with missing or NaN values in specific columns
    df = df.dropna(subset=['A', 'B', 'C', 'D'])

    return df

# def display_dataset(df):
#     # return df.iloc[:, :-2]
#     return df.head()

def dataset_description(df):
    """
    Provides a description of the dataset including the duration of each class
    
    Parameters:

    df (pandas.DataFrame): The processed dataframe.

    Returns:

    tuple: (pandas.DataFrame, str)

    DataFrame containing class counts and durations.
    String describing the total duration of the video.
    """
    class_counts = df['Overall class'].fillna('NaN').groupby(df['Overall class'].fillna('Missing Rows')).count().reset_index(name='Class Count')
    class_counts['Duration in seconds'] = class_counts['Class Count'] / 100
    class_counts = class_counts[['Overall class', 'Duration in seconds']]
    
    total_duration_seconds = class_counts['Duration in seconds'].sum()
    
    if total_duration_seconds >= 3600:
        total_duration_hours = total_duration_seconds / 3600
        duration_str = f"Duration of Video: {total_duration_hours:.2f} Hours"
    elif total_duration_seconds >= 60 and total_duration_seconds<3600 :
        total_duration_minutes = total_duration_seconds / 60
        duration_str = f"Duration of Video: {total_duration_minutes:.2f} Minutes"
    else:
        duration_str = f"Duration of Video: {total_duration_seconds:.2f} Seconds"
    
    return class_counts, duration_str

def create_plot(df):
    """
    Creates a bar plot showing the duration of each class

    Parameters:

    df (pandas.DataFrame): The processed dataframe.

    Returns:

    Figure representing the bar plot.
    """
    class_counts = df['Overall class'].fillna('NaN').groupby(df['Overall class'].fillna('Missing Rows')).count().reset_index(name='Class Count')
    class_counts['Duration in seconds'] = class_counts['Class Count'] / 100
    
    fig = px.bar(class_counts, x='Overall class', y='Duration in seconds',
                 labels={'Overall class': 'Category', 'Duration in seconds': 'Duration in Seconds'},
                 title='Duration of Each Class')
    
    fig.update_layout(
        xaxis_tickangle=-45, 
        yaxis_title='Duration in Seconds',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False) 
        )
    
    for i in range(len(fig.data[0].x)):
        fig.add_annotation(
            x=fig.data[0].x[i],
            y=fig.data[0].y[i],
            text=f"{fig.data[0].y[i]:.2f}",
            showarrow=False,
            yshift=10
        )
    
    return fig

def plot_bins(df, class_name):
    """
    Creates a bar plot showing the distribution of time durations for a specific class.
    0.1 seconds spaced bins are being plooted.

    Parameters:

    df (pandas.DataFrame): The processed dataframe.
    class_name (str): The name of the class to plot.

    Returns:

    Figure or str: Figure if successful, or an error message string if no data is available. 
    """
    same_class_mask = df['Overall class'] == df['Overall class'].shift(1)
    df['Increment'] = np.where(same_class_mask, 10, 0)
    df['Rolling Sum'] = df['Increment'].groupby((~same_class_mask).cumsum()).cumsum() / 1000
    df.drop(columns=['Increment'], inplace=True)
    
    d = df[df['Overall class'] == class_name].copy()
    
    # If there are no rows for the given class
    if d.empty:
        return f"No values for class '{class_name}' exist."
    
    max_val = d['Rolling Sum'].max()
    
    fixed_bins = [0.1, 0.2, 0.3, 0.4, 0.5, 1]
    variable_bins = np.linspace(1.5, max(max_val, 1.5), num=5)
    bins = np.unique(np.sort(np.concatenate((fixed_bins, variable_bins))))
    
    d['duration_bin'] = pd.cut(d['Rolling Sum'], bins, include_lowest=True)
    
    cnt_bin = d.groupby(['Overall class', 'duration_bin']).size().reset_index(name='bin_count')
    
    # Check if all bin counts are zero
    if cnt_bin['bin_count'].sum() == 0:
        return f"No values for class '{class_name}' exist."
    
    cnt_bin['duration_bin'] = cnt_bin['duration_bin'].astype(str)
    
    fig = px.bar(cnt_bin, x='duration_bin', y='bin_count',
                 labels={'duration_bin': 'Duration (seconds)', 'bin_count': 'Count'},
                 title=f"Buckets for: {class_name}")
    
    fig.update_layout(xaxis_tickangle=-45,
                      xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False) 
        )
    
    for i in range(len(fig.data[0].x)):
        fig.add_annotation(
            x=fig.data[0].x[i],
            y=fig.data[0].y[i],
            text=f"{fig.data[0].y[i]:.0f}",
            showarrow=False,
            yshift=10
        )
    
    return fig

def overall_class_stats(df, overall_class):
    """
    Returns the longest contiguous segment of a given overall class along with its start and end indices.
    
    Parameters:

    df (pandas.DataFrame): The processed dataframe.
    overall_class (str): The name of the class to analyze.

    Returns:

    tuple: (str, list)

    String describing the maximum consecutive sequence.
    List of strings describing all contiguous segments.
    """
    class_indices = df[df['Overall class'] == overall_class].index
    cnt_arr = []
    cnt = max_cnt = 1
    start = end = class_indices[0]
    
    for i in range(len(class_indices) - 1):
        if class_indices[i + 1] == class_indices[i] + 1:
            cnt += 1
            max_cnt = max(cnt, max_cnt)
        else:
            end = class_indices[i]
            formatted_output = f"{start} to {end}: {cnt}"
            cnt_arr.append(formatted_output)
            start = class_indices[i + 1]
            cnt = 1
    
    end = class_indices[-1]
    formatted_output = f"{start} to {end}: {cnt}"
    cnt_arr.append(formatted_output)
    
    max_sequence = max(cnt_arr, key=lambda x: int(x.split(': ')[1]))
    
    # Save results to a text file for logging (Depreciated)
    # Created a separated page on Streamlit based on the requirements
    
    # with open(file_path, 'w') as f:
    #     for line in cnt_arr:  
    #         f.write(line + '\n')

    # Show only the first ten rows and save the others for logging 
    # cnt_arr_display = sorted(cnt_arr,key=lambda x: int(x.split(': ')[1]),reverse=True)[:10]

    return max_sequence,cnt_arr

def plot_contiguous_blocks(contiguous_blocks, threshold, selected_option):
    """
    Plot a bar chart showing the count of contiguous blocks for a given overall class.
    
    Args:
        contiguous_blocks (list): A list of formatted strings representing the contiguous blocks,
                                 e.g. ["0 to 10: 11", "12 to 15: 4", ...]
        threshold (int): The threshold for filtering data.
        selected_option (str): The option selected to filter the data.
    """
    sequence_lengths = [int(x.split(': ')[1]) for x in contiguous_blocks]
    length_counts = pd.Series(sequence_lengths).value_counts(ascending=False).reset_index()
    length_counts.columns = ['Sequence Length', 'Count']

    df_filtered = length_counts[length_counts[selected_option] > threshold]

    # Generate dynamic labels
    dynamic_x_label = f'Sequence Length (Filtered by {selected_option})'
    dynamic_y_label = f'Count (Threshold: {threshold})'
    dynamic_title = f'Bar Plot of Sequence Length vs Count (Filtered by {selected_option} > {threshold})'

    # Create the bar plot
    fig = px.bar(df_filtered, x='Sequence Length', y='Count', 
                 title=dynamic_title,
                 labels={'Sequence Length': dynamic_x_label, 'Count': dynamic_y_label},
                 color='Count', color_continuous_scale='Blues')

    fig.update_layout(xaxis_title=dynamic_x_label,
                      yaxis_title=dynamic_y_label,
                      xaxis_tickangle=-90)

    return fig 

def plot_contiguous_blocks_scatter(contiguous_blocks,threshold,selected_option):
    """
    Plot a scatter plot showing the count of contiguous blocks for a given overall class.
    Y-axis is log scale.
    Args:
        contiguous_blocks (list): A list of formatted strings representing the contiguous blocks,
                                 e.g. ["0 to 10: 11", "12 to 15: 4", ...]
    """
    sequence_lengths = [int(x.split(': ')[1]) for x in contiguous_blocks]
    length_counts = pd.Series(sequence_lengths).value_counts(ascending=False).reset_index()
    length_counts.columns = ['Sequence Length', 'Count']

    df_filtered = length_counts[length_counts[selected_option] > threshold]

    fig2 = px.scatter(df_filtered, x='Sequence Length', y='Count', 
                 title='Scatter Plot of Sequence Length vs Count',
                 labels={'x': 'Sequence Length', 'y': 'Count'},
                 opacity=0.6)

    # Set the y-axis to a logarithmic scale
    fig2.update_layout(yaxis_type="log", 
                  xaxis_title='Sequence Length', 
                  yaxis_title='Count (Log Scale)',
                  height=600, width=1200)

    return fig2 

def create_data_blocks(df, start_time, block_size=50000):
    """
    Create blocks of data based on starting time.
    
    :param df: DataFrame with processed data
    :param start_time: Starting time for the blocks (datetime object)
    :param block_size: Size of each block (default: 50000)
    :return: List of DataFrames, each representing a block
    """
    df_sorted = df[df['A'] >= start_time].sort_values('A')
    total_blocks = (len(df_sorted) + block_size - 1) // block_size
    return [df_sorted.iloc[i*block_size:(i+1)*block_size] for i in range(total_blocks)]

def plot_block(block):
    """
    Create a scatter plot for a given block of data.
    
    :param block: DataFrame representing a block of data
    :return: Plotly Figure object
    """
    # Use a more efficient color mapping
    unique_classes = block['Overall class'].unique()
    color_map = {cls: f'rgb({np.random.randint(0, 256)}, {np.random.randint(0, 256)}, {np.random.randint(0, 256)})' 
                 for cls in unique_classes}

    # Create subplots for better performance
    fig = make_subplots(rows=1, cols=1)

    # Use go.Scattergl for better performance with large datasets
    for cls in unique_classes:
        df_class = block[block['Overall class'] == cls]
        fig.add_trace(go.Scattergl(
            x=df_class['A'],
            y=df_class['Up/down angle'],
            mode='markers',
            marker=dict(color=color_map[cls], size=4, opacity=0.7),
            name=cls,
            legendgroup=cls,
            showlegend=True
        ))

    fig.update_layout(
        title={
            'text': 'Body Position Analysis Over Time',
            'font': dict(size=28, color='#FFFFFF', family="Arial, sans-serif"),
            'x': 0.5,
            'xanchor': 'center',
            'y': 0.95,
            'yanchor': 'top'
        },
        xaxis_title='Time',
        yaxis_title='Up/Down Angle (degrees)',
        font=dict(family="Arial, sans-serif", size=14, color="#FFFFFF"),
        legend_title='Body Position',
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
            font=dict(color="#FFFFFF")
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode='closest',
        margin=dict(t=100, b=50, l=50, r=150),
    )

    fig.update_xaxes(
        showgrid=True, gridwidth=1, gridcolor='#555555',
        showline=True, linewidth=2, linecolor='#FFFFFF',
        tickformat='%H:%M:%S\n%Y-%m-%d',
        tickfont=dict(color="#FFFFFF")
    )
    fig.update_yaxes(
        showgrid=True, gridwidth=1, gridcolor='#555555',
        showline=True, linewidth=2, linecolor='#FFFFFF',
        zeroline=True, zerolinewidth=2, zerolinecolor='#FFFFFF',
        tickfont=dict(color="#FFFFFF")
    )

    return fig

def parse_datetime(date_val):
    if isinstance(date_val, str):
        return pd.to_datetime(date_val.rsplit(':', 1)[0], format='%Y-%m-%d %H:%M:%S')
    elif isinstance(date_val, pd.Timestamp):
        return date_val
    else:
        raise ValueError(f"Unexpected data type: {type(date_val)}")
