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
from scipy.stats import circmean

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
# def display_dataset(df):
#     # return df.iloc[:, :-2]
#     return df.head()

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

# def parse_datetime(date_val):
#     """
#     Parse datetime values consistently across the application.
#     """
#     if isinstance(date_val, str):
#         return pd.to_datetime(date_val.rsplit(':', 1)[0], format='%Y-%m-%d %H:%M:%S')
#     elif isinstance(date_val, pd.Timestamp):
#         return date_val
#     else:
#         raise ValueError(f"Unexpected data type: {type(date_val)}")

@st.cache_data(show_spinner=False)
def process_dataset(file):
    """
    Process a large dataset from a CSV file.

    Parameter : Streamlit UploadedFile object
   
    Returns : Processed DataFrame
    """
    try:
        df = pd.read_csv(
            file,
            encoding='utf-8',
            skiprows=99,
            on_bad_lines='skip',
            header=None,
            parse_dates=[0],
            date_format='%Y-%m-%d %H:%M:%S:%f'
        )
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(
                file,
                encoding='latin-1',
                skiprows=99,
                on_bad_lines='skip',
                header=None,
                parse_dates=[0],
            )
        except UnicodeDecodeError:
            return "Error: Unable to decode the file. Please ensure it's a valid CSV."
    
    # Manually convert the date column using the correct format
    df[0] = pd.to_datetime(df[0], format='%Y-%m-%d %H:%M:%S:%f')
    
    df = df.iloc[1:, :5]
    columns = ['A', 'B', 'C', 'D', 'E']
    df.columns = columns
    
    # Initialize columns
    df['360 angle'] = np.nan
    df['Up/down angle'] = np.nan
    df['Body Rotation'] = ""
    df['Prone-sit class'] = ""
    df['Supine-recline class'] = ""
    df['Overall class'] = ""
    df['Acceleration'] = ""
    
    S5 = 140
    
    df['Acceleration'] = np.sqrt(df['B']**2 + df['C']**2 + df['D']**2)
    
    denominator = np.sqrt(df['B']**2 + df['D']**2)
    mask = denominator != 0
    df.loc[mask, '360 angle'] = np.sign(df.loc[mask, 'B']) * np.degrees(
        np.arccos(-df.loc[mask, 'D'] / denominator[mask])
    ) + 180
    
    denominator = np.sqrt(df['B']**2 + df['C']**2 + df['D']**2)
    mask = denominator != 0
    df.loc[mask, 'Up/down angle'] = np.degrees(
        np.arcsin(df.loc[mask, 'C'] / denominator[mask])
    )
    
    df['Body Rotation'] = np.where(
        (df['360 angle'] > S5) & (df['360 angle'] < (S5 + 180)),
        'supine-recline',
        'prone-sit'
    )
    
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
    df.loc[df['Body Rotation'] != 'prone-sit', 'Prone-sit class'] = ""
    
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
    df.loc[df['Body Rotation'] != 'supine-recline', 'Supine-recline class'] = ""
    
    df['Overall class'] = (df['Prone-sit class'] + ' ' + df['Supine-recline class']).str.strip()
    df = df.dropna(subset=['A', 'B', 'C', 'D', 'E'])
    
    return df

def dataset_description(df):
    """
    Provides a description of the dataset including the duration of each class.
    """
    class_counts = df['Overall class'].fillna('NaN').groupby(df['Overall class'].fillna('Missing Rows')).count().reset_index(name='Class Count')
    
    class_counts['Duration in seconds'] = class_counts['Class Count'] / 100
    
    def format_duration(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        
        if hours > 0:
            return f"{hours} Hours {minutes} Minutes {seconds} Seconds"
        elif minutes > 0:
            return f"{minutes} Minutes {seconds} Seconds"
        else:
            return f"{seconds} Seconds"
    
    class_counts['Total Duration'] = class_counts['Duration in seconds'].apply(format_duration)
    
    total_duration_seconds = class_counts['Duration in seconds'].sum()
    duration_str = format_duration(total_duration_seconds)
    duration_str = f"Duration of Video: {duration_str}"
    
    return class_counts[['Overall class', 'Total Duration']], duration_str

def create_plot(df):
    """
    Creates a bar plot showing the duration of each class in HH:MM:SS format
    """
    class_counts = df['Overall class'].fillna('NaN').groupby(df['Overall class'].fillna('Missing Rows')).count().reset_index(name='Class Count')
    
    # Convert count to seconds (assuming 100 readings per second)
    seconds = class_counts['Class Count'] / 100
    
    # Convert seconds to HH:MM:SS format for annotations
    def format_duration(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    class_counts['Duration'] = seconds.apply(format_duration)
    
    fig = px.bar(class_counts, x='Overall class', y=seconds,
                 labels={'Overall class': 'Category'},
                 title='Duration of Each Class')
    
    fig.update_layout(
        xaxis_tickangle=-45,
        yaxis=dict(
            showticklabels=False,  # Hide y-axis tick labels
            title=None,            # Remove y-axis title
            showgrid=False         # Hide y-axis grid
        ),
        xaxis=dict(
            showgrid=False         # Hide x-axis grid
        )
    )
    
    # Add duration labels on top of each bar
    for i in range(len(fig.data[0].x)):
        fig.add_annotation(
            x=fig.data[0].x[i],
            y=fig.data[0].y[i],
            text=class_counts['Duration'].iloc[i],
            showarrow=False,
            yshift=10
        )
    
    return fig

def plot_bins(df, class_name):
    """
    Creates a bar plot showing the distribution of time durations for a specific class.
    0.1 seconds spaced bins are being plotted.
    """
    same_class_mask = df['Overall class'] == df['Overall class'].shift(1)
    df['Increment'] = np.where(same_class_mask, 10, 0)
    df['Rolling Sum'] = df['Increment'].groupby((~same_class_mask).cumsum()).cumsum() / 1000
    df.drop(columns=['Increment'], inplace=True)
    
    d = df[df['Overall class'] == class_name].copy()
    
    if d.empty:
        return f"No values for class '{class_name}' exist."
    
    max_val = d['Rolling Sum'].max()
    
    fixed_bins = [0.1, 0.2, 0.3, 0.4, 0.5, 1]
    variable_bins = np.linspace(1.5, max(max_val, 1.5), num=5)
    bins = np.unique(np.sort(np.concatenate((fixed_bins, variable_bins))))
    
    d['duration_bin'] = pd.cut(d['Rolling Sum'], bins, include_lowest=True)
    cnt_bin = d.groupby(['Overall class', 'duration_bin']).size().reset_index(name='bin_count')
    
    if cnt_bin['bin_count'].sum() == 0:
        return f"No values for class '{class_name}' exist."
    
    cnt_bin['duration_bin'] = cnt_bin['duration_bin'].astype(str)
    
    fig = px.bar(cnt_bin, x='duration_bin', y='bin_count',
                 labels={'duration_bin': 'Duration (seconds)', 'bin_count': 'Count'},
                 title=f"Buckets for: {class_name}")
    
    fig.update_layout(
        xaxis_tickangle=-45,
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
    """
    class_indices = df[df['Overall class'] == overall_class].index
    if len(class_indices) == 0:
        return "No data found for this class", []
        
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
    
    return max_sequence, cnt_arr

def plot_contiguous_blocks(contiguous_blocks, threshold, selected_option):
    """
    Plot a bar chart showing the count of contiguous blocks for a given overall class.
    """
    sequence_lengths = [int(x.split(': ')[1]) for x in contiguous_blocks]
    length_counts = pd.Series(sequence_lengths).value_counts(ascending=False).reset_index()
    length_counts.columns = ['Sequence Length', 'Count']
    
    df_filtered = length_counts[length_counts[selected_option] > threshold]
    
    dynamic_x_label = f'Sequence Length (Filtered by {selected_option})'
    dynamic_y_label = f'Count (Threshold: {threshold})'
    dynamic_title = f'Bar Plot of Sequence Length vs Count (Filtered by {selected_option} > {threshold})'
    
    fig = px.bar(df_filtered, x='Sequence Length', y='Count',
                 title=dynamic_title,
                 labels={'Sequence Length': dynamic_x_label, 'Count': dynamic_y_label},
                 color='Count', color_continuous_scale='Blues')
    
    fig.update_layout(
        xaxis_title=dynamic_x_label,
        yaxis_title=dynamic_y_label,
        xaxis_tickangle=-90
    )
    
    return fig

def plot_contiguous_blocks_scatter(contiguous_blocks, threshold, selected_option):
    """
    Plot a scatter plot showing the count of contiguous blocks for a given overall class.
    Y-axis is log scale.
    """
    sequence_lengths = [int(x.split(': ')[1]) for x in contiguous_blocks]
    length_counts = pd.Series(sequence_lengths).value_counts(ascending=False).reset_index()
    length_counts.columns = ['Sequence Length', 'Count']
    
    df_filtered = length_counts[length_counts[selected_option] > threshold]
    
    fig2 = px.scatter(df_filtered, x='Sequence Length', y='Count',
                      title='Scatter Plot of Sequence Length vs Count',
                      labels={'x': 'Sequence Length', 'y': 'Count'},
                      opacity=0.6)
    
    fig2.update_layout(
        yaxis_type="log",
        xaxis_title='Sequence Length',
        yaxis_title='Count (Log Scale)',
        height=600,
        width=1200
    )
    
    return fig2

def create_data_blocks(df, start_time, block_size=50000):
    """
    Create blocks of data based on starting time.
    """
    df_sorted = df[df['A'] >= start_time].sort_values('A')
    total_blocks = (len(df_sorted) + block_size - 1) // block_size
    return [df_sorted.iloc[i*block_size:(i+1)*block_size] for i in range(total_blocks)]

def plot_block(block):
    """
    Create a scatter plot for a given block of data.
    """
    unique_classes = block['Overall class'].unique()
    color_map = {
        cls: f'rgb({np.random.randint(0, 256)}, {np.random.randint(0, 256)}, {np.random.randint(0, 256)})'
        for cls in unique_classes
    }
    
    fig = make_subplots(rows=1, cols=1)
    
    for cls in unique_classes:
        df_class = block[block['Overall class'] == cls]
        fig.add_trace(
            go.Scattergl(
                x=df_class['A'],
                y=df_class['Up/down angle'],
                mode='markers',
                marker=dict(color=color_map[cls], size=4, opacity=0.7),
                name=cls,
                legendgroup=cls,
                showlegend=True
            )
        )
    
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
        margin=dict(t=100, b=50, l=50, r=150)
    )
    
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='#555555',
        showline=True,
        linewidth=2,
        linecolor='#FFFFFF',
        tickformat='%H:%M:%S\n%Y-%m-%d',
        tickfont=dict(color="#FFFFFF")
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='#555555',
        showline=True,
        linewidth=2,
        linecolor='#FFFFFF',
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor='#FFFFFF',
        tickfont=dict(color="#FFFFFF")
    )
    
    return fig

def plot_sensor_data(df):
    """
    Used to visualize the sensor data to identify wear, non-wear time and sensor drop.
    """
    # Group by timestamp (floor to seconds for aggregation)
    df_plot = (df.groupby(df['A'].dt.floor('s'))
               .agg({
                   'Overall class': pd.Series.mode,
                   'B': 'mean',
                   'Acceleration': 'mean'
               })
               .round(4)
               .reset_index())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_plot['A'],
        y=df_plot['B'],
        mode='lines',
        name='X-axis',
        hoverinfo='text',
        text=[
            f"Overall class: {row['Overall class']}<br>X-axis: {row['B']}<br>Time:{row['A']}"
            for _, row in df_plot.iterrows()
        ]
    ))
    
    fig.add_trace(go.Scatter(
        x=df_plot['A'],
        y=df_plot['Acceleration'],
        mode='lines',
        name='Acceleration',
        hoverinfo='text',
        text=[
            f"Overall class: {row['Overall class']}<br>Acceleration: {row['Acceleration']}"
            for _, row in df_plot.iterrows()
        ]
    ))
    
    fig.update_layout(
        width=1500,
        height=600,
        title='X Axis and Acceleration',
        xaxis_title='Time',
        yaxis_title='Values',
        legend_title='Metrics'
    )
    
    total_seconds = len(df_plot)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    print(f"Total Video Length {hours} hours and {minutes} minutes")
    
    return df_plot, fig

def tummy_time_duration(df, min_size=60, sleep_time=None, wake_time=None):
    """    
    Preparing data for visualization for tummy time at home and prone tolerance test comparison.
    Only processes rows with 'prone' or 'prone supported' values in Overall class.
    
    Parameters:
    df (DataFrame): Input DataFrame with timestamp column 'A'
    min_size (int): Minimum duration for visualization (default: 60 seconds)
    start_datetime (datetime): Start datetime for sensor data collection
    end_datetime (datetime): End datetime for sensor data collection
    sleep_time (str): Daily sleep time in 'HH:MM' format
    wake_time (str): Daily wake time in 'HH:MM' format
    """
    # Convert 'timestamp' or 'time' column to datetime if it is not already
    df['A'] = pd.to_datetime(df['A'])
        
    # Filter by daily sleep/wake times if provided
    if sleep_time and wake_time:
        sleep_time = pd.to_datetime(sleep_time, format='%H:%M').time()
        wake_time = pd.to_datetime(wake_time, format='%H:%M').time()
        
        # Keep rows where time is before sleep_time OR after wake_time
        df = df[
            (df['A'].dt.time <= sleep_time) | 
            (df['A'].dt.time >= wake_time)
        ]

    # Filter for only prone or prone supported positions
    mask = df['Overall class'].isin(['prone', 'prone supported'])
    prone_or_supported = df[mask].copy()
    
    # Reset index to create 'row_number'
    prone_or_supported = prone_or_supported.reset_index()
    
    if prone_or_supported.empty:
        return [], [], []
        
    # Find gaps in the row numbers to identify sequences
    prone_or_supported['gap'] = prone_or_supported['index'].diff()
    sequence_starts = prone_or_supported[prone_or_supported['gap'] > 1].index.tolist()
    sequence_starts.insert(0, 0)  # Add the first sequence start
    
    # Create buckets
    buckets = []
    for i in range(len(sequence_starts)):
        start_idx = sequence_starts[i]
        if i < len(sequence_starts) - 1:
            end_idx = sequence_starts[i + 1]
            sequence = prone_or_supported.iloc[start_idx:end_idx]
        else:
            sequence = prone_or_supported.iloc[start_idx:]
            
        if len(sequence) >= min_size:
            bucket = sequence['index'].tolist()
            buckets.append((bucket, len(bucket)))
    
    # Sort buckets by size
    sorted_buckets = sorted(buckets, key=lambda x: x[1], reverse=True)
    
    # Create validation and duration lists
    validate_list = []
    duration_ls = []
    
    for bucket, size in sorted_buckets:
        if size >= min_size:
            start_idx = bucket[0]
            end_idx = bucket[-1]
            
            duration = (df.iloc[end_idx]['A'] - df.iloc[start_idx]['A']).total_seconds()
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            
            timestamp_val_1 = df.iloc[start_idx]['A']
            timestamp_val_2 = df.iloc[end_idx]['A']
            
            validate_list.append([start_idx, end_idx, f"{minutes}m {seconds}s"])
            duration_ls.append([f"{minutes}m {seconds}s", timestamp_val_1, timestamp_val_2])
    
    return sorted_buckets, validate_list, duration_ls

def plot_exercise_durations(prone_tolerance_value, durations, removed_bars=None):
    """
    Create a bar plot comparing exercise durations to a Prone Tolerance Value.
    
    Parameters:
    prone_tolerance_value (str): Duration in format 'Xm Ys'
    durations (list): List of durations and timestamps
    removed_bars (list): List of bar indices to remove (1-based indexing)
    """
    def time_to_seconds(time_str):
        minutes, seconds = map(int, time_str.replace('s', '').split('m'))
        return minutes * 60 + seconds
    def format_duration(seconds):
        minutes = seconds // 60
        sec = seconds % 60
        return f"{minutes}m {sec}s"
    prone_tolerance_value = time_to_seconds(prone_tolerance_value)
    
    # Convert durations to seconds and format timestamps
    duration_seconds = [time_to_seconds(dur[0]) for dur in durations]
    formatted_durations = [format_duration(d) for d in duration_seconds]
    tummy_time_timestamps = [f"{dur[1].strftime('%Y-%m-%d %H:%M:%S')} to {dur[2].strftime('%Y-%m-%d %H:%M:%S')}" for dur in durations]
    # Create the full dataset first
    data = pd.DataFrame({
        'Session': [f"Session {i+1}" for i in range(len(durations))],
        'Duration (seconds)': duration_seconds,
        'Formatted Duration': formatted_durations,
        'Timestamps': tummy_time_timestamps,
        'Index': range(1, len(durations) + 1)  # 1-based indexing
    })
    # Filter out removed bars if specified
    if removed_bars:
        data = data[~data['Index'].isin(removed_bars)]
    
    fig = px.bar(
        data,
        x='Session',
        y='Duration (seconds)',
        title='Tummy Time',
        labels={'Session': 'Tummy Time Sessions', 'Duration (seconds)': 'Duration (seconds)'},
        hover_data={'Timestamps': True, 'Duration (seconds)': False}
    )
    
    fig.add_hline(
        y=prone_tolerance_value,
        line_dash='dash',
        line_color='red',
        annotation_text=f"Prone Tolerance Value: {format_duration(prone_tolerance_value)}",
        annotation_position='top right'
    )
    
    fig.update_traces(
        marker_color='skyblue',
        text=data['Formatted Duration'],  # Display formatted duration above bars
        textposition='outside',
        width=0.6
    )
    
    fig.update_layout(
        title={'text': 'Tummy Time', 'x': 0.5, 'xanchor': 'center'},
        xaxis_title="Tummy Time Sessions",
        yaxis_title="Duration (seconds)",
        showlegend=False,
        bargap=0.3
    )
    
    # Remove all x-axis ticks and labels
    fig.update_xaxes(showticklabels=False, showgrid=False)
    
    return fig
