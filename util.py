import pandas as pd
import numpy as np
import streamlit as st
import csv
from functools import partial
import plotly.graph_objects as go
import plotly.express as px
import io

csv.field_size_limit(int(1e9)) 

@st.cache_data
def process_dataset(file):
    """
    Process a large dataset from a CSV file.

    :param file: Streamlit UploadedFile object
    :return: Processed DataFrame
    """
    try:
        # Attempt to read the file directly with pandas
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

    # Define constants (example values)
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

def display_dataset(df):
    # return df.iloc[:, :-2]
    return df.head()

def dataset_description(df):
    """
    Provides a description of the dataset including the duration of each class
    """
    class_counts = df['Overall class'].fillna('NaN').groupby(df['Overall class'].fillna('Missing Rows')).count().reset_index(name='Class Count')
    class_counts['Duration in seconds'] = class_counts['Class Count'] / 100
    class_counts = class_counts[['Overall class', 'Duration in seconds']]
    
    total_duration_seconds = class_counts['Duration in seconds'].sum()

    if total_duration_seconds >= 60:
        total_duration_minutes = total_duration_seconds / 60
        duration_str = f"Duration of Video: {total_duration_minutes:.2f} minutes"
    else:
        duration_str = f"Duration of Video: {total_duration_seconds:.2f} seconds"
    
    return class_counts, total_duration_seconds, duration_str

def create_plot(df):
    class_counts = df['Overall class'].fillna('NaN').groupby(df['Overall class'].fillna('Missing Rows')).count().reset_index(name='Class Count')
    class_counts['Duration in seconds'] = class_counts['Class Count'] / 100
    
    fig = px.bar(class_counts, x='Overall class', y='Duration in seconds',
                 labels={'Overall class': 'Category', 'Duration in seconds': 'Duration in Seconds'},
                 title='Duration of Each Class')
    
    fig.update_layout(xaxis_tickangle=-45, yaxis_title='Duration in Seconds')
    
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
    Plots bins/blocks based on the duration of the segment of the provided class.
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
    
    fig.update_layout(xaxis_tickangle=-45)
    
    for i in range(len(fig.data[0].x)):
        fig.add_annotation(
            x=fig.data[0].x[i],
            y=fig.data[0].y[i],
            text=f"{fig.data[0].y[i]:.0f}",
            showarrow=False,
            yshift=10
        )
    
    return fig

def overall_class_stats(df, overall_class, file_path='results.txt'):
    """
    Returns the longest continuous segment of a given overall class along with its start and end indices,
    and saves the results to a text file.
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
    
    # Save results to a text file
    with open(file_path, 'w') as f:
        for line in cnt_arr[:5]:  # Save only the first 5 lines
            f.write(line + '\n')
    
    return cnt_arr, max_sequence
