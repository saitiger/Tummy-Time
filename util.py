import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st

def clean_dataset(file):
    """
    This is for csv files where the file has been processed using formulas
    file : Is the name of the csv that has the data from the sensor
    Example : A1_right hip_102373_2024-07-31 10-52-18 
    NOTE: There is no need to add .csv in front of the filename 
    """
    df = pd.read_csv(file,skiprows = 99)
    df.drop(df.columns[[5,7,8,9,10,11,12,15]], axis=1, inplace = True)    
    df.rename(columns={'Unnamed: 0': 'Timing', 'Overall class ': 'Overall class','360 angle ':'360 angle','Supine-recline class ':'Supine-recline class'}, inplace=True)
    df['Timing'] = pd.to_datetime(df['Timing'], format='%Y-%m-%d %H:%M:%S:%f')
    df['Milliseconds'] = df['Timing'].dt.microsecond // 1000
    df['Time Diff'] = df['Milliseconds'].diff().fillna(10.0)
    df.loc[df['Time Diff'] < 0, 'Time Diff'] = 10.0
    
    return df

def process_dataset(file):
    """
    file : Is the name of the csv that has the data from the sensor
    Example : A1_right hip_102373_2024-07-31 10-52-18 
    NOTE: There is no need to add .csv in front of the filename 
    """
    # df = pd.read_csv(file + '.csv', skiprows=99)
    df = pd.read_csv(file,skiprows = 99,names = ['A','B','C','D','E','F','G','H','I','J','K','L','M','360 angle','Up/down angle','Observed position','Body Rotation','Prone-sit class','Supine-recline class','Overall class'])  

    S5 = 0

    df['360 angle'] = np.sign(df['B']) * np.arccos(-df['D'] / np.sqrt(df['B']**2 + df['D']**2)) * 180 / np.pi + 180

    df['Up/down angle'] = np.arcsin(df['C'] / np.sqrt(df['B']**2 + df['C']**2 + df['D']**2)) * 180 / 3.14

    df['Body Rotation'] = np.where((S5 < df['360 angle']) & (df['360 angle'] < (S5 + 180)), "supine-recline", "prone-sit")

    df['Prone-sit class'] = np.select([
    (df['Body Rotation'] == "prone-sit") & (df['Up/down angle'] > 0),
    (df['Body Rotation'] == "prone-sit") & (df['Up/down angle'] > -23),
    (df['Body Rotation'] == "prone-sit") & (df['Up/down angle'] > -63),
    df['Body Rotation'] == "prone-sit"
    ], [
    "prone",
    "prone supported",
    "upright",
    "sitting"
    ], default="")

    df['Supine-recline class'] = np.select([
    (df['Body Rotation'] == "supine-recline") & (df['Up/down angle'] > 15),
    (df['Body Rotation'] == "supine-recline") & (df['Up/down angle'] < -36),
    (df['Body Rotation'] == "supine-recline") & (df['360 angle'] < (S5 + 69)),
    (df['Body Rotation'] == "supine-recline") & (df['360 angle'] > (S5 + 101)),
    df['Body Rotation'] == "supine-recline"
    ], [
    "upsidedown",
    "reclined",
    "left side",
    "right side",
    "supine"
    ], default="")

    df['Overall class'] = df['Prone-sit class'] + df['Supine-recline class']

    # Save the result
    # df.to_csv('df_processed.csv', index=False)
    # print("Processing complete. Results saved to 'df_processed.csv'")  

    df.drop(columns = ['H','I','J','K','L','M'],inplace = True)
    df.rename(columns = {'A':'Timing'},inplace = True)
    df['Timing'] = pd.to_datetime(df['Timing'], format='%Y-%m-%d %H:%M:%S:%f')
    df['Milliseconds'] = df['Timing'].dt.microsecond // 1000
    df['Time Diff'] = df['Milliseconds'].diff().fillna(10.0)
    df.loc[df['Time Diff'] < 0, 'Time Diff'] = 10.0
    
    return df

def display_dataset(df):
    return df.iloc[:,:-2]
    # return df[['Timing','360 angle','Up/down angle','Body Rotation','Prone-sit class','Supine-recline class','Overall class']]

def dataset_description(df):
    """
    Provides a description of the dataset including the duration of each class
    """
    class_counts = df['Overall class'].fillna('NaN').groupby(df['Overall class'].fillna('Missing Rows')).count().reset_index(name='Class Count')
    class_counts['Duration in seconds'] = class_counts['Class Count'] / 100
    class_counts = class_counts[['Overall class', 'Duration in seconds']]
    
    total_duration = class_counts['Duration in seconds'].sum()
    # print(f"Duration of Video: {total_duration}")
    
    # statistics = df.describe()
    return class_counts, total_duration

def create_plot(df):
    class_counts = df['Overall class'].fillna('NaN').groupby(df['Overall class'].fillna('Missing Rows')).count().reset_index(name='Class Count')
    class_counts['Duration in seconds'] = class_counts['Class Count'] / 100
    
    fig, ax = plt.subplots(figsize=(8, 6))  # Create figure and axes objects
    bars = ax.bar(class_counts['Overall class'], class_counts['Duration in seconds'])
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 1, round(yval, 2), ha='center', va='bottom')
    
    ax.set_xticklabels(class_counts['Overall class'], rotation=45)
    ax.set_yticks([])
    ax.set_xlabel('Category')
    ax.set_ylabel('Duration in Seconds')
    sns.despine(ax=ax, bottom=True, left=True)
    
    st.pyplot(fig)  # Provide the figure to st.pyplot

def plot_bins(df, class_name):
    same_class_mask = df['Overall class'] == df['Overall class'].shift(1)
    df['Increment'] = np.where(same_class_mask, 10, 0)
    df['Rolling Sum'] = df['Increment'].groupby((~same_class_mask).cumsum()).cumsum() / 1000
    df.drop(columns=['Increment'], inplace=True)
    
    d = df[df['Overall class'] == class_name].copy()
    
    # If there are no rows for the given class
    if d.empty:
        st.warning(f"No values for class '{class_name}' exist.")
        return
    
    max_val = d['Rolling Sum'].max()
    
    fixed_bins = [0.1, 0.2, 0.3, 0.4, 0.5, 1]
    variable_bins = np.linspace(1.5, max(max_val, 1.5), num=5)
    bins = np.unique(np.sort(np.concatenate((fixed_bins, variable_bins))))
    
    d['duration_bin'] = pd.cut(d['Rolling Sum'], bins, include_lowest=True)
    
    cnt_bin = d.groupby(['Overall class', 'duration_bin']).size().reset_index(name='bin_count')
    
    # Check if all bin counts are zero
    if cnt_bin['bin_count'].sum() == 0:
        st.warning(f"No values for class '{class_name}' exist.")
        return
    
    cnt_bin['duration_bin'] = cnt_bin['duration_bin'].astype(str)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x='duration_bin', height='bin_count', data=cnt_bin)
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 1, round(yval, 2), ha='center', va='bottom')
    
    ax.set_title(f"Buckets for: {class_name}")
    ax.set_xticklabels(cnt_bin['duration_bin'], rotation=45, ha='right')
    ax.set_yticks([])
    ax.set_xlabel('Duration (seconds)')
    ax.set_ylabel('Count')
    sns.despine(ax=ax, bottom=True, left=True)
    plt.tight_layout()
    
    st.pyplot(fig)
    
def overall_class_stats(df, overall_class):
    """
    Returns the longest continuous segment of a given overall class along with its start and end indices
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
            cnt_arr.append((cnt, start, end))
            start = class_indices[i + 1]
            cnt = 1
    
    cnt_arr.append((cnt, start, end))  # To account for the last sequence
    max_sequence = max(cnt_arr, key=lambda x: x[0])
    
    return max_sequence