import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import io

def add_threshold_label(ax, threshold, df_plot, position='smart'):
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    
    ax.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5)
    vertical_offset = (y_max - y_min) * 0.05
    y_pos = threshold + vertical_offset    
    y_pos = min(y_pos, y_max - vertical_offset)
    
    if position == 'smart':
        if isinstance(df_plot.index[0], pd.Timestamp):
            data_series = df_plot['current height'] if 'current height' in df_plot else df_plot.iloc[:, 1]
            first_half = data_series.iloc[:len(data_series)//2]
            second_half = data_series.iloc[len(data_series)//2:]
            
            first_half_near = first_half[(first_half > threshold - vertical_offset*2) & 
                                         (first_half < threshold + vertical_offset*2)].count()
            second_half_near = second_half[(second_half > threshold - vertical_offset*2) & 
                                          (second_half < threshold + vertical_offset*2)].count()
            
            x_pos = x_max - (x_max - x_min) * 0.01 if first_half_near > second_half_near else x_min + (x_max - x_min) * 0.01
            ha = 'right' if first_half_near > second_half_near else 'left'
        else:
            x_pos = x_max - (x_max - x_min) * 0.01
            ha = 'right'
    elif position == 'right':
        x_pos = x_max - (x_max - x_min) * 0.01
        ha = 'right'
    elif position == 'left':
        x_pos = x_min + (x_max - x_min) * 0.01
        ha = 'left'
    elif position == 'avoid_data':
        bins = 20
        hist, bin_edges = np.histogram(df_plot['current height'], bins=bins)
        bin_index = np.digitize(threshold, bin_edges) - 1
        
        left_density = np.sum(hist[:bins//2])
        right_density = np.sum(hist[bins//2:])
        
        if left_density > right_density:
            x_pos = x_max - (x_max - x_min) * 0.01
            ha = 'right'
        else:
            x_pos = x_min + (x_max - x_min) * 0.01
            ha = 'left'
    
    ax.text(
        x=x_pos,
        y=y_pos,
        s=f"Threshold: {threshold:.2f}",
        fontweight='bold',
        fontsize=15,
        ha=ha,
        va='bottom',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3)
    )

def contingent_head_above_threshold(df, chunk_time=120, save_path=None):
    """
    Analyze head position relative to threshold in contingent mode using matplotlib and seaborn
    Returns a tuple of (matplotlib figure, results dictionary, filename)
    """
    results = {}
    
    threshold = df['threshold'].unique()[0]

    df_plot = df[df['face_detection_flag'] == True][['Time', 'current height', 'head above the threshold', 'Toy status']]
    
    # Ensure sorting before resampling
    df_plot = df_plot.sort_values(by='Time')
    
    if not pd.api.types.is_datetime64_any_dtype(df_plot['Time']):
        df_plot['Time'] = pd.to_datetime(df_plot['Time'], format='%Y-%m-%d %H:%M:%S.%f')
    
    # Resampling
    height_resampled = df_plot.set_index('Time')[['current height']].resample('500ms').mean()
    threshold_resampled = df_plot.set_index('Time')[['head above the threshold']].resample('500ms').last()
    
    # Combine resampled data
    df_resampled = pd.concat([height_resampled, threshold_resampled], axis=1).reset_index()
    
    total_video_length = len(df_plot) * 50 / 1000  
    head_above_threshold_total_time = df_plot[df_plot['head above the threshold'] == True].shape[0] * 50 / 1000  # in seconds
    
    # Count Inversions 
    results['Inversion Count'] = df_plot[(df_plot['head above the threshold']==True) & (df_plot['head above the threshold'].shift(1)==False)]['head above the threshold'].count()

    results['video_stats'] = {
        'Video Duration': f"{int(total_video_length // 60)} minutes {round(total_video_length % 60, 2)} seconds",
        'Head above the threshold duration': f"{int(head_above_threshold_total_time // 60)} minutes {round(head_above_threshold_total_time % 60, 2)} seconds",
        'Percentage above threshold': f"{round(head_above_threshold_total_time / total_video_length * 100, 2)}%"
    }
                               
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
    
    sns.lineplot(
        data=df_resampled,
        x='Time',
        y='current height',
        ax=ax1,
        linewidth=1.5,
        color='blue'
    )
    ax1.set_ylabel('Current Height')
    ax1.grid(False)
    ax1.set_xticks([]) 
    ax1.set_xlabel('')
    sns.despine(ax=ax1, bottom=True)
    
    add_threshold_label(ax1, threshold, df_resampled, position='smart')
    
    sns.lineplot(
        data=df_resampled,
        x='Time',
        y='head above the threshold',
        ax=ax2,
        linewidth=1.5,
        color='red',
        drawstyle='steps-post'
    )
    ax2.set_ylabel('Toy Status')
    ax2.set_ylim(-0.1, 1.1)
    ax2.grid(False)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Off', 'On'])
    sns.despine(ax=ax2)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"contingent_analysis_{timestamp}.png"
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/{filename}", dpi=300, bbox_inches='tight')
    
    return fig, results, filename

def non_contingent_head_above_threshold(df, save_path=None):
    """
    Create non-contingent analysis plot using matplotlib and seaborn
    Returns a tuple of (matplotlib figure, results dictionary, filename)
    """
    results = {}
    
    # Create df_plot first before using it
    df_plot = df[df['face_detection_flag']==True][['Time', 'current height', 'head above the threshold', 'Toy status']].reset_index(drop=True)
    
    total_video_length = len(df_plot) * 50 / 1000  

    results['Video Duration'] = f"{int(total_video_length // 60)} minutes {round(total_video_length % 60, 2)} seconds"
    
    if not pd.api.types.is_datetime64_any_dtype(df_plot['Time']):
        df_plot['Time'] = pd.to_datetime(df_plot['Time'], format='%Y-%m-%d %H:%M:%S.%f')
    
    # Add row numbers
    df_plot['row_number'] = range(1, len(df_plot) + 1)
    
    # Check if Toy status has multiple values or needs to be created
    if df_plot['Toy status'].nunique() > 1:
        if not set(df_plot['Toy status'].unique()).issubset({0, 1, "0", "1", "On", "Off", True, False}):
            status_mapping = {status: i % 2 for i, status in enumerate(df_plot['Toy status'].unique())}
            df_plot['Toy status'] = df_plot['Toy status'].map(status_mapping)
        
        if df_plot['Toy status'].dtype == 'object':
            df_plot['Toy status'] = df_plot['Toy status'].map({"On": 1, "Off": 0, "True": 1, "False": 0, "1": 1, "0": 0})
    else:
        # Create alternating toy status every 300 rows
        df_plot['Toy status'] = (df_plot.index // 300 % 2).astype(int)
    
    # Status Changes
    status_changes = (df_plot['Toy status'] != df_plot['Toy status'].shift(1)).cumsum()
    period_counts = status_changes.value_counts()

    # More efficient way to count toy status periods
    on_periods = df_plot['Toy status'].diff().fillna(df_plot['Toy status'].iloc[0]).gt(0).sum() + (df_plot['Toy status'].iloc[0] == 1)
    off_periods = df_plot['Toy status'].diff().fillna(df_plot['Toy status'].iloc[0]).lt(0).sum() + (df_plot['Toy status'].iloc[0] == 0)
    
    # For consistency, use 'status_stats' instead of 'status_change'
    results['status_stats'] = {
        'On periods': on_periods,
        'Off periods': off_periods,
        'Number of status changes': len(period_counts) - 1
    }

    # More efficient inversion count
    inversion_counts = ((df_plot['current height'] > 0) & (df_plot['current height'].shift(1) <= 0)).sum()
    inversion_by_toy_status = df_plot[
        (df_plot['current height'] > 0) & (df_plot['current height'].shift(1) <= 0)
    ].groupby('Toy status').size().to_dict()
    
    results['Inversions'] = {
        'Count': inversion_counts,
        'Count by Toy Status': inversion_by_toy_status
    }
    
    head_above_zero = df_plot[df_plot['current height'] > 0]
    head_above_zero_total_time = head_above_zero.shape[0] * 50 / 1000

    results['Head above Zero'] = {
        'Count': head_above_zero.shape[0],
        'Count by Toy Status': head_above_zero.groupby('Toy status').size().to_dict(),
        'Duration': f"{int(head_above_zero_total_time // 60)} minutes {round(head_above_zero_total_time % 60, 2)} seconds", 
        '% Of Video': f"{round(head_above_zero_total_time / total_video_length * 100, 2)}%"
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
    # fig.suptitle('Non-Contingent Analysis', fontsize=16)
    
    sns.lineplot(
        data=df_plot,
        x='row_number',
        y='current height',
        ax=ax1,
        linewidth=1.5,
        color='blue'
    )
    ax1.set_ylabel('Current Height')
    ax1.grid(False)
    ax1.set_xticks([])  
    ax1.set_xlabel('')
    sns.despine(ax=ax1, bottom=True)
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
    
    sns.lineplot(
        data=df_plot,
        x='row_number',
        y='Toy status',
        ax=ax2,
        linewidth=1.5,
        color='red',
        drawstyle='steps-post'
    )
    ax2.set_ylabel('Toy Status')
    ax2.set_ylim(-0.1, 1.1)
    ax2.grid(False)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Off', 'On'])
    
    # Remove the x-axis ticks and labels
    ax2.set_xticks([])
    ax2.set_xticklabels([])
    ax2.set_xlabel('')
    
    sns.despine(ax=ax2, bottom=True)
    
    status_change_indices = df_plot.loc[df_plot['Toy status'] != df_plot['Toy status'].shift()].index
    for idx in status_change_indices:
        if idx > 0:  
            ax1.axvline(x=df_plot.loc[idx, 'row_number'], color='gray', linestyle='-', alpha=0.3)
            ax2.axvline(x=df_plot.loc[idx, 'row_number'], color='gray', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"non_contingent_analysis_{timestamp}.png"
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/{filename}", dpi=300, bbox_inches='tight')
    
    return fig, results, filename

def validation_baseline(df, save_path=None):
    """
    Validate data for baseline phase
    Returns a tuple of (matplotlib figure, results dictionary, filename)
    """
    results = {}
    
    # Drop these values
    false_flags = round(100.0 * df[df['face_detection_flag']==False]['Time'].count()/df['Time'].count(), 2)
    duration = df[df['face_detection_flag']==False]['Time'].count()/20
    total_duration = df['Time'].count()/20
    
    results['false_flags'] = {
        'percentage': f"{false_flags}%",
        'duration': f"{duration//60} Minutes {round(duration%60,2)} Seconds",
        'total_duration': f"{total_duration//60} Minutes {round(total_duration%60,2)} Seconds"
    }

    results['Threshold Value'] = {
        'Threshold' : df['threshold'].iloc[-1] if 'threshold' in df.columns else 'N/A'
    }

    # Frequency Check
    if not pd.api.types.is_datetime64_any_dtype(df['Time']):
        df['Time'] = pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S.%f')
        
    time_diffs = df['Time'].diff().dt.total_seconds().dropna()
    std = time_diffs.std()

    lower_bound = 0.05 - std
    upper_bound = 0.05 + std

    above_count = time_diffs[time_diffs > upper_bound].count()
    below_count = time_diffs[time_diffs < lower_bound].count()
    
    results['frequency_check'] = {
        'above_count': f"Count above {upper_bound:.5f} seconds: {above_count}",
        'below_count': f"Count below {lower_bound:.5f} seconds: {below_count}"
    }

    # Height mismatch check
    if 'current height' in df.columns and 'filtered height' in df.columns:
        chk = df[(~df['current height'].isna()) & (~df['filtered height'].isna())]
        chk_2 = chk[~np.isclose(chk['current height'], chk['filtered height'], atol=1e-9)]
        results['height_mismatch'] = f"Number of Rows with current and filtered height mismatch: {chk_2.shape[0]}"
    
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.histplot(time_diffs, bins=50, ax=ax, kde=True)
    ax.set_title('Distribution of Time Differences', fontsize=16)
    ax.set_xlabel('Time Difference (seconds)')
    ax.set_ylabel('Frequency')
    
    plt.axvline(x=0.05, color='red', linestyle='--', label='Expected (0.05s)')
    plt.axvline(x=lower_bound, color='green', linestyle='--', label=f'Lower Bound ({lower_bound:.5f}s)')
    plt.axvline(x=upper_bound, color='orange', linestyle='--', label=f'Upper Bound ({upper_bound:.5f}s)')
    
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"validation_baseline_{timestamp}.png"
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/{filename}", dpi=300, bbox_inches='tight')
    
    return fig, results, filename

def phase_two_validation(df, filename, save_path=None):
    """
    Validate data for phase two
    Returns a tuple of (matplotlib figure, results dictionary, filename)
    """
    results = {}

    # Check if the filename contains "contingent" and only then add "Threshold"
    if filename.split('_')[1]=="Contingent":
        results['Threshold'] = df['threshold'].unique()[0]

    # Time uniqueness check
    results['time_check'] = f"More than one values for Time: {df['Time'].nunique() != 1}"
    
    # Face detection flag counts
    results['face_detection_counts'] = df['face_detection_flag'].value_counts().to_dict()
   
    # Null checks
    null_height_count = df[df['current height'].isnull()].shape[0]
    false_face_count = df[df['face_detection_flag'] == False].shape[0]
    results['null_checks'] = {
        'null_height_count': null_height_count,
        'false_face_count': false_face_count,
        'match': 'Values match' if null_height_count == false_face_count else 'Values do not match'
    }

    # False flags analysis
    false_flags = round(100.0 * df[df['face_detection_flag'] == False]['Time'].count() / df['Time'].count(), 2)
    duration = df[df['face_detection_flag'] == False]['Time'].count() // 20

    # Toy Status 
    results['Toy Status Value Counts'] = df['Toy status'].value_counts()                     
    

    # Total Duration
    total_duration = df['Time'].count() / 20
    
    results['false_flags'] = {
        'percentage': f"{false_flags}%",
        'duration': f"{round(duration // 60, 2)} Minutes {round(duration % 60, 2)} Seconds",
    }

    results['Total Duration'] = f"{round(total_duration // 60, 2)} Minutes {round(total_duration % 60, 2)} Seconds"

    if not pd.api.types.is_datetime64_any_dtype(df['Time']):
        df['Time'] = pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S.%f')

    time_diffs = df['Time'].diff().dt.total_seconds().dropna()

    fig, ax = plt.subplots(figsize=(16, 8))
    sns.histplot(time_diffs, bins=50, ax=ax, kde=True)
    ax.set_title('Distribution of Time Differences', fontsize=16)
    ax.set_xlabel('Time Difference (seconds)')
    ax.set_ylabel('Frequency')

    plt.axvline(x=0.05, color='red', linestyle='--', label='Expected (0.05s)')

    std = time_diffs.std()
    lower_bound = 0.05 - std
    upper_bound = 0.05 + std

    plt.axvline(x=lower_bound, color='green', linestyle='--', label=f'Lower Bound ({lower_bound:.5f}s)')
    plt.axvline(x=upper_bound, color='orange', linestyle='--', label=f'Upper Bound ({upper_bound:.5f}s)')

    plt.legend()
    plt.grid(False)
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"phase_two_validation_{timestamp}.png"

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/{output_filename}", dpi=300, bbox_inches='tight')

    return fig, results, output_filename