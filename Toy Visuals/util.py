import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import numpy as np

def validation_baseline(df):
    """
    Validate data for baseline phase
    Returns a tuple of (plotly figure, results dictionary)
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

    # Frequency Check
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
    chk = df[(~df['current height'].isna()) & (~df['filtered height'].isna())]
    chk_2 = chk[~np.isclose(chk['current height'], chk['filtered height'], atol=1e-9)]
    results['height_mismatch'] = f"Number of Rows with current and filtered height mismatch: {chk_2.shape[0]}"
    
    # Frequency histogram
    fig = px.histogram(
        time_diffs,
        nbins=50,
        title='Distribution of Time Differences',
        labels={'value': 'Time Difference (seconds)', 'count': 'Frequency'},
    )
    fig.update_layout(
        showlegend=False,
        title_x=0.5,
        title_font_size=20,
        plot_bgcolor='white'
    )

    return fig, results

def phase_two_validation(df):
    """
    Validate data for phase two
    Returns a tuple of (plotly figure, results dictionary)
    """
    results = {}
    
    # Time uniqueness check
    results['time_check'] = f"More than one values for Time: {df['Time'].nunique()!=1}"
    
    # Face detection flag counts
    results['face_detection_counts'] = df['face_detection_flag'].value_counts().to_dict()
    
    # Null checks
    null_height_count = df[df['current height'].isnull()].shape[0]
    false_face_count = df[df['face_detection_flag']==False].shape[0]
    results['null_checks'] = {
        'null_height_count': null_height_count,
        'false_face_count': false_face_count,
        'match': 'Values match' if null_height_count == false_face_count else 'Values do not match'
    }

    # False flags analysis
    false_flags = round(100.0 * df[df['face_detection_flag']==False]['Time'].count()/df['Time'].count(), 2)
    duration = df[df['face_detection_flag']==False]['Time'].count()//20
    total_duration = df['Time'].count()/20
    
    results['false_flags'] = {
        'percentage': f"{false_flags}%",
        'duration': f"{round(duration//60,2)} Minutes {round(duration%60,2)} Seconds",
        'total_duration': f"{round(total_duration//60,2)} Minutes {round(total_duration%60,2)} Seconds"
    }

    df['Time'] = pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S.%f')
    time_diffs = df['Time'].diff().dt.total_seconds().dropna()
    
    # Frequency histogram
    fig = px.histogram(
        time_diffs,
        nbins=50,
        title='Distribution of Time Differences',
        labels={'value': 'Time Difference (seconds)', 'count': 'Frequency'},
    )
    fig.update_layout(
        showlegend=False,
        title_x=0.5,
        title_font_size=20,
        plot_bgcolor='white'
    )

    return fig, results

def contingent_head_above_threshold(df, chunk_time=120):
    """
    Analyze head position relative to threshold in contingent mode
    Returns a tuple of (plotly figure, results dictionary)
    """
    results = {}
    
    df_plot = df[['head above the threshold', 'Toy status', 'Time']]
    chunk_size = chunk_time * 20
    chunk_results = []

    total_video_length = df_plot['Time'].count() * 50 / 1000
    head_above_threshold_total_time = df_plot[df_plot['head above the threshold'] == True]['Time'].count() * 50 / 1000
    
    results['video_stats'] = {
        'total_length': f"{total_video_length // 60} minutes {round(total_video_length % 60, 2)} seconds",
        'head_above_time': f"{head_above_threshold_total_time // 60} minutes {round(head_above_threshold_total_time % 60, 2)} seconds"
    }

    for i, chunk_start in enumerate(range(0, len(df), chunk_size)):
        chunk = df.iloc[chunk_start:chunk_start + chunk_size]

        count = ((chunk['head above the threshold'].shift(1) == False) &
                 (chunk['head above the threshold'] == True)).sum()
        chunk_time = chunk['Time'].count() * 50 / 1000
        head_above_threshold_time = chunk[chunk['head above the threshold'] == True]['Time'].count() * 50 / 1000

        chunk_results.append({
            'Chunk': f'Chunk {i + 1}',
            'Chunk Time': chunk_time if len(chunk) < chunk_size else chunk_time,
            'Head Above Time': head_above_threshold_time,
            'Count': count
        })

    # Create DataFrame for plotting
    results_df = pd.DataFrame(chunk_results)
    
    # Create grouped bar plot using plotly
    fig = go.Figure()
    
    # Add bars for each metric
    fig.add_trace(go.Bar(
        name='Chunk Time',
        x=results_df['Chunk'],
        y=results_df['Chunk Time'],
        text=results_df['Chunk Time'].round(1),
        textposition='auto',
    ))
    
    fig.add_trace(go.Bar(
        name='Head Above Time',
        x=results_df['Chunk'],
        y=results_df['Head Above Time'],
        text=results_df['Head Above Time'].round(1),
        textposition='auto',
    ))
    
    fig.add_trace(go.Bar(
        name='Count',
        x=results_df['Chunk'],
        y=results_df['Count'],
        text=results_df['Count'],
        textposition='auto',
    ))

    fig.update_layout(
        title='Chunk Analysis',
        title_x=0.5,
        barmode='group',
        xaxis_title='Chunk',
        yaxis_title='Time (seconds) / Count',
        plot_bgcolor='white'
    )

    results['chunk_data'] = chunk_results
    return fig, results

def non_contingent_head_above_threshold(df):
    """
    Create non-contingent analysis plot
    Returns a tuple of (plotly figure, results dictionary)
    """
    total_rows = len(df)
    results = {}

    # Create status column
    status = ['On' if (i // 300) % 2 == 0 else 'Off' for i in range(total_rows)]
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(total_rows)),
        y=df['current height'],
        mode='lines',
        name='Current Height'
    ))
    
    # Add colored background for On/Off status
    for i in range(0, total_rows, 300):
        if (i // 300) % 2 == 0:
            fig.add_vrect(
                x0=i,
                x1=min(i + 300, total_rows),
                fillcolor="lightgreen",
                opacity=0.2,
                layer="below",
                line_width=0,
                name='On Period'
            )
        else:
            fig.add_vrect(
                x0=i,
                x1=min(i + 300, total_rows),
                fillcolor="lightpink",  # Changed from 'lightred' to 'lightpink'
                opacity=0.2,
                layer="below",
                line_width=0,
                name='Off Period'
            )
    
    fig.update_layout(
        title='Non-Contingent Analysis',
        title_x=0.5,
        xaxis_title='Time',
        yaxis_title='Current Height',
        plot_bgcolor='white',
        showlegend=True
    )

    results['total_rows'] = total_rows
    results['on_periods'] = status.count('On')
    results['off_periods'] = status.count('Off')
    
    return fig, results