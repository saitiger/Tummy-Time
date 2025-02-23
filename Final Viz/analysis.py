import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import os
import gc
import psutil
from datetime import datetime

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def read_config_file(folder_path):
    """Read configuration from Excel file in the folder"""
    excel_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]
    if not excel_files:
        raise Exception("No Excel configuration file found in the folder")
    
    config_path = os.path.join(folder_path, excel_files[0])
    config_df = pd.read_excel(config_path)
    
    # Validate required columns
    required_columns = ['filename', 'waketime', 'sleeptime', 'prone_tolerance']
    missing_columns = [col for col in required_columns if col not in config_df.columns]
    if missing_columns:
        raise Exception(f"Missing required columns in Excel file: {missing_columns}")
    
    # Convert to dictionary for easy lookup
    config_dict = {}
    for _, row in config_df.iterrows():
        config_dict[row['filename']] = {
            'waketime': row['waketime'] if pd.notna(row['waketime']) else None,
            'sleeptime': row['sleeptime'] if pd.notna(row['sleeptime']) else None,
            'prone_tolerance': str(row['prone_tolerance'])  # Ensure it's in string format
        }
    
    return config_dict

def process_files_sequentially(folder_path):
    """Process files using configuration from Excel"""
    # Get configuration from Excel
    try:
        config_dict = read_config_file(folder_path)
        print(f"Successfully read configuration for {len(config_dict)} files")
    except Exception as e:
        print(f"Error reading configuration file: {str(e)}")
        return None, None
    
    # Get global minimum size parameter (keeping this as user input)
    min_size = int(input("Enter minimum size for tummy time duration (in seconds, default 60): ") or 60)
    
    results_df = pd.DataFrame(columns=['File', 'Average_Duration_Seconds', 'Prone_Tolerance_Seconds'])
    all_durations_data = []
    
    # Process only CSV files that have configurations
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv') and f in config_dict]
    total_files = len(csv_files)
    
    if not csv_files:
        print("No CSV files found matching the configuration")
        return None, None
    
    for i, filename in enumerate(csv_files, 1):
        print(f"\nProcessing file {i} of {total_files}: {filename}")
        print(f"Current memory usage: {get_memory_usage():.2f} MB")
        
        file_path = os.path.join(folder_path, filename)
        config = config_dict[filename]
        
        try:
            # Process the dataset
            df = process_dataset(file_path)
            if isinstance(df, str):
                print(f"Error processing {filename}: {df}")
                continue
            
            # Get parameters from config
            wake_time = config['waketime']
            sleep_time = config['sleeptime']
            prone_tolerance = config['prone_tolerance']
            prone_tolerance_seconds = time_to_seconds(prone_tolerance)
            
            print(f"Using configuration - Wake: {wake_time}, Sleep: {sleep_time}, Prone Tolerance: {prone_tolerance}")
            
            # Calculate tummy time durations
            durations = tummy_time_duration(df, min_size, sleep_time, wake_time)
            
            # Explicitly delete the large DataFrame
            del df
            gc.collect()
            
            if not durations:
                print(f"No valid tummy time sessions found in {filename}")
                continue
            
            print(f"Found {len(durations)} tummy time sessions")
            
            # Create and display the plot
            fig = plot_exercise_durations(prone_tolerance, durations)
            fig.show()
            
            # Get bars to remove
            removed_bars = get_removed_bars(len(durations))
            
            # Collect individual durations
            file_durations = collect_individual_durations(durations, removed_bars, filename, prone_tolerance_seconds)
            all_durations_data.extend(file_durations)
            
            # Calculate average duration
            avg_duration = calculate_average_duration(durations, removed_bars)
            
            # Add to results DataFrame
            new_row = pd.DataFrame({
                'File': [filename],
                'Average_Duration_Seconds': [avg_duration],
                'Prone_Tolerance_Seconds': [prone_tolerance_seconds]
            })
            results_df = pd.concat([results_df, new_row], ignore_index=True)
            
            del durations
            gc.collect()
            
            print(f"Memory usage after processing: {get_memory_usage():.2f} MB")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    print("\nCreating final visualizations...")
    
    all_durations_df = pd.DataFrame(all_durations_data)
    
    # Regression plot
    plt.figure(figsize=(10, 6))
    sns.regplot(
        data=results_df,
        x='Prone_Tolerance_Seconds',
        y='Average_Duration_Seconds',
        scatter_kws={'alpha':0.5},
        line_kws={'color': 'red'}
    )
    plt.xlabel('Prone Tolerance (seconds)')
    plt.ylabel('Average Tummy Time Duration (seconds)')
    plt.title('Regression Plot: Prone Tolerance vs Average Tummy Time Duration')
    plt.grid(True, alpha=0.3)
    plt.savefig('regression_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Scatter plot by cluster 
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=all_durations_df,
        x='Prone_Tolerance_Seconds',
        y='Duration_Seconds',
        hue='File',
        alpha=0.6,
        s=100
    )
    plt.xlabel('Prone Tolerance (seconds)')
    plt.ylabel('Individual Duration (seconds)')
    plt.title('Scatter Plot: Individual Durations by File')
    plt.grid(True, alpha=0.3)
    
    if len(all_durations_df['File'].unique()) > 10:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    
    plt.tight_layout()
    plt.savefig('individual_durations_scatter.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results_df, all_durations_df
