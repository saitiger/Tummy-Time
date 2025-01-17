import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import ttest_rel

def difference_plot(df):
  cols = [d for d in df.columns if "RPM" in d]

  dplot = df[cols].mean().reset_index()
  dplot['Visit'] = "V" + dplot['index'].str.split("_").str[1]
  dplot['Scores'] = dplot['index'].str.split("_").str[3]
  dplot.drop('index',inplace = True,axis = 1)
  dplot.rename(columns = {0:'Values'},inplace = True)

  df_differences = (
    dplot.pivot(index='Visit', columns='Scores', values='Values')
    .assign(Difference=lambda x: x['RPM'] - x['RPMsub'])
    .reset_index()
  )

  sns.lineplot(data=dplot, x='Visit', y='Values', hue='Scores', marker='o')

  for index, row in df_differences.iterrows():
      x = row['Visit']
      y = (row['RPM'] + row['RPMsub']) / 2
      difference = row['Difference']
      plt.text(x, y, f'{difference:.2f}', color='red', ha='center', fontsize=10)

  sns.despine()
  plt.ylabel("Values")
  plt.show()

def withinvisit_ttest(dataset, visit_num):
    """
    Perform a paired t-test within a single visit for RPM and RPMsub scores.

    Parameters:
        dataset (DataFrame): The input dataset containing the scores.
        visit_num (int): The visit number (must be 2, 3, 4, or 5).

    Returns:
        tuple: A tuple containing the t-statistic, p-value, and a result interpretation message.
    """
    if visit_num not in [2, 3, 4, 5]:
        return None, None, "Enter a valid Visit Number (2, 3, 4, or 5)."

    try:
        filtered_data = dataset[
            (~dataset[f'C_{visit_num}_APS_RPM'].isna()) &
            (~dataset[f'C_{visit_num}_APS_RPMsub'].isna())
        ][[f'C_{visit_num}_APS_RPM', f'C_{visit_num}_APS_RPMsub']]
    except KeyError as e:
        return None, None, f"Column not found in dataset: {e}"

    if filtered_data.empty:
        return None, None, "No valid data for t-test after filtering."

    t_stat, p_value = ttest_rel(
        filtered_data[f'C_{visit_num}_APS_RPM'],
        filtered_data[f'C_{visit_num}_APS_RPMsub']
    )

    if p_value < 0.01:
        interpretation = (
            f"The difference within visit {visit_num} between RPM and RPMsub is highly statistically significant "
            f"(p-value = {p_value:.4f}). There is strong evidence to suggest a meaningful change."
        )
    elif p_value < 0.05:
        interpretation = (
            f"The difference within visit {visit_num} between RPM and RPMsub is statistically significant "
            f"(p-value = {p_value:.4f}). There is moderate evidence to suggest a meaningful change."
        )
    else:
        interpretation = (
            f"The difference within visit {visit_num} between RPM and RPMsub is not statistically significant "
            f"(p-value = {p_value:.4f}). There is insufficient evidence to suggest a meaningful change."
        )

    return t_stat, p_value, interpretation

def across_visit_ttest(dataset, start_visit, score_type):
    """
    Perform a paired t-test across visits for a given score type.

    Parameters:
        dataset (DataFrame): The input dataset containing the scores.
        start_visit (int): The starting visit number (must be between 2 and 4).
        score_type (str): The type of score ('RPM' or 'RPMsub').

    Returns:
        tuple: A tuple containing the t-statistic, p-value, and a result interpretation message.
    """

    if not (2 <= start_visit <= 4):
        return None, None, "Start visit must be between 2 and 4 (inclusive)."
    if score_type not in ['RPM', 'RPMsub']:
        return None, None, "Score type must be either 'RPM' or 'RPMsub'."

    end_visit = start_visit + 1

    try:
        filtered_data = dataset[
            (~dataset[f'C_{start_visit}_APS_{score_type}'].isna()) &
            (~dataset[f'C_{end_visit}_APS_{score_type}'].isna())
        ][[f'C_{start_visit}_APS_{score_type}', f'C_{end_visit}_APS_{score_type}']]
    except KeyError as e:
        return None, None, f"Column not found in dataset: {e}"

    if filtered_data.empty:
        return None, None, "No valid data for t-test after filtering."

    t_stat, p_value = ttest_rel(
        filtered_data[f'C_{start_visit}_APS_{score_type}'],
        filtered_data[f'C_{end_visit}_APS_{score_type}']
    )

    if p_value < 0.01:
        interpretation = (
            f"The difference between visits {start_visit} and {end_visit} for {score_type} is highly statistically significant "
            f"(p-value = {p_value:.4f}). There is strong evidence to suggest a meaningful change."
        )
    elif p_value < 0.05:
        interpretation = (
            f"The difference between visits {start_visit} and {end_visit} for {score_type} is statistically significant "
            f"(p-value = {p_value:.4f}). There is moderate evidence to suggest a meaningful change."
        )
    else:
        interpretation = (
            f"The difference between visits {start_visit} and {end_visit} for {score_type} is not statistically significant "
            f"(p-value = {p_value:.4f}). There is insufficient evidence to suggest a meaningful change."
        )

    return t_stat, p_value, interpretation

# Visit Visualization

def visit_visualization(dataset,visit_num):
  search_terms = ["LK", "ES", "FN", "EC", "SL"]
  check = [c for c in df.columns 
         if any(term in c for term in search_terms) 
         and (len(c.split("_")[3])==2 or len(c.split("_")[3])==5)]
  df_plot = df[check].mean().reset_index()
  split_values = df_plot['index'].str.split('_', expand=True)

  
  df_plot = df_plot.assign(
    Visit = 'V' + split_values[1],
    Score = split_values[3].str[:2],
    Value = df_plot[0],
    With_Popup = ~split_values[3].str[2:].str.contains('sub', case=False)
  ).drop(columns=['index', 0])
  ax = sns.barplot(data=df_plot[df_plot["Visit"]==f"V{visit_num}"],x="Score",y="Value",hue="With_Popup")
  for container in ax.containers:
      ax.bar_label(container, fmt='%.2f', padding=3, 
                 fontsize=10,  
                 fontweight='bold')  
      
  sns.despine(bottom = True,left = True)
  plt.yticks([])
  plt.ylabel("Frequency")
  plt.show()

def all_visits_behaviour_plot(df):
    search_terms = ["LK", "ES", "FN", "EC", "SL"]
    check = [c for c in df.columns 
             if any(term in c for term in search_terms) 
             and (len(c.split("_")[3])==2 or len(c.split("_")[3])==5)]
    
    df_plot = df[check].mean().reset_index()
    split_values = df_plot['index'].str.split('_', expand=True)
    
    df_processed = df_plot.assign(
        Visit = 'V' + split_values[1],
        Score = split_values[3].str[:2],
        Value = df_plot[0],
        With_Popup = ~split_values[3].str[2:].str.contains('sub', case=False) 
    ).drop(columns=['index', 0])[['Visit', 'Score', 'Value', 'With_Popup']]
    
    # With Popup
    plt.figure(figsize=(10,5))
    plt.title("With Popup")
    ax1 = sns.barplot(data=df_processed[df_processed['With_Popup']], 
                     x="Visit", y="Value", hue='Score')
    
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.1f', padding=3, 
                     fontsize=10, fontweight='bold')
    
    sns.despine(bottom=True, left=True)
    plt.yticks([])
    plt.ylabel("Frequency")
    plt.show()
    
    # Without Popup
    plt.figure(figsize=(10,5))
    plt.title("Without Popup")
    ax2 = sns.barplot(data=df_processed[~df_processed['With_Popup']], 
                     x="Visit", y="Value", hue='Score')
    
    for container in ax2.containers:  
        ax2.bar_label(container, fmt='%.1f', padding=3, 
                     fontsize=10, fontweight='bold')
    
    sns.despine(bottom=True, left=True)
    plt.yticks([])
    plt.ylabel("Frequency")
    plt.show()
    
    # Dataframe 
    popup_true = df_processed[df_processed['With_Popup'] == True]
    popup_false = df_processed[df_processed['With_Popup'] == False]

    # Set indices for alignment
    popup_true = popup_true.set_index(['Visit', 'Score'])
    popup_false = popup_false.set_index(['Visit', 'Score'])

    difference = popup_true['Value'] - popup_false['Value']

    difference_df = difference.reset_index()
    difference_df.columns = ['Visit', 'Score', 'Value_Difference']
    return difference_df
