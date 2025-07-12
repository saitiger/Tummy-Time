import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Utility Functions
def normalize_column_names(df):
    return df.rename(columns=lambda x: x.replace('CSu', 'Csu'))

def filter_desired_columns(df, substrings):
    return [col for col in df.columns if any(sub in col for sub in substrings)]

def extract_metric_from_col(col_name):
    return col_name.split('_')[3]

# 1. Correlation Matrix + P-Value Matrix
def calculate_metric_correlation(df, group_num=None):
    df = normalize_column_names(df)
    if group_num is not None:
        df = df[df["C_1_TRT"] == group_num]

    desired_cols = filter_desired_columns(df, ['MTc', 'APS_RPM', 'Csu']) + ['CID']
    df_corr = df[desired_cols].dropna()

    df_melt = df_corr.melt(id_vars='CID')
    df_melt['Metric'] = df_melt['variable'].apply(extract_metric_from_col)
    df_melt.rename(columns={'value': 'Score'}, inplace=True)

    df_plot = df_melt.groupby(['CID', 'Metric'])['Score'].mean().reset_index()
    df_pivot = df_plot.pivot(index='CID', columns='Metric', values='Score')

    correlation_matrix = df_pivot.corr(method='pearson')

    # P-value matrix
    p_value_matrix = pd.DataFrame(np.zeros_like(correlation_matrix), 
                                  columns=correlation_matrix.columns, 
                                  index=correlation_matrix.index)

    for col1 in df_pivot.columns:
        for col2 in df_pivot.columns:
            if col1 == col2:
                p_value_matrix.loc[col1, col2] = 0
            else:
                temp_df = df_pivot[[col1, col2]].dropna()
                if len(temp_df) > 1:
                    _, p = pearsonr(temp_df[col1], temp_df[col2])
                    p_value_matrix.loc[col1, col2] = p
                    p_value_matrix.loc[col2, col1] = p
                else:
                    p_value_matrix.loc[col1, col2] = np.nan
                    p_value_matrix.loc[col2, col1] = np.nan

    return correlation_matrix, p_value_matrix

# 2. Print Pearson r for APS-RPM with Motor and Cognitive
def corr_group(df, group_num):
    df = normalize_column_names(df)
    df = df[df["C_1_TRT"] == group_num]
    desired_cols = filter_desired_columns(df, ['MTc', 'APS_RPM', 'Csu']) + ['C_1_TRT']
    df_corr = df[desired_cols].dropna()

    for i in range(2, 6):
        r_motor, p_motor = pearsonr(df_corr[f'C_{i}_BS3_MTc'], df_corr[f'C_{i}_APS_RPM'])
        r_cog, p_cog = pearsonr(df_corr[f'C_{i}_BS3_Csu'], df_corr[f'C_{i}_APS_RPM'])
        print(f"Visit {i}")
        print(f"Motor/APSP: r = {r_motor:.2f}, p = {p_motor:.3f}")
        print(f"Cognitive/APSP: r = {r_cog:.2f}, p = {p_cog:.3f}")

# 3. Line Plot of Metric Score over Visits
def plot_score_diff(df, metric, group_num):
    metric_map = {"Cognitive": "Csu", "Motor": "MTc"}
    metric_col = metric_map[metric]

    df = normalize_column_names(df)
    df_corr = df[df["C_1_TRT"] == group_num]
    desired_cols = filter_desired_columns(df_corr, [metric_col, 'APS_RPM'])
    df_corr = df_corr[desired_cols].dropna()

    df_melt = df_corr.melt()
    df_melt["Visit"] = "V" + df_melt['variable'].str.split('_').str[1]
    df_melt["Metric"] = df_melt['variable'].apply(extract_metric_from_col)
    df_melt.rename(columns={'value': 'Score'}, inplace=True)

    df_plot = df_melt.groupby(['Visit', 'Metric'])['Score'].mean().reset_index()

    sns.lineplot(data=df_plot, x='Visit', y='Score', hue='Metric', linewidth=3, palette='bright')
    sns.despine()
    plt.tick_params(axis='x', length=0)

    for _, row in df_plot.iterrows():
        plt.text(row['Visit'], row['Score'] + 4, round(row['Score'], 2),
                 ha='left', fontsize='small', color='black', weight='semibold')
    plt.show()

# 4. Scatter Plot per Visit for a Group
def scatter_plot_group(df, metric, group_num):
    metric_map = {"Cognitive": "Csu", "Motor": "MTc"}
    metric_col = metric_map[metric]

    df = normalize_column_names(df)
    df_corr = df[df["C_1_TRT"] == group_num]
    desired_cols = filter_desired_columns(df_corr, [metric_col, 'APS_RPM'])
    df_corr = df_corr[desired_cols].dropna()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for i in range(2, 6):
        row, col = divmod(i - 2, 2)
        sns.regplot(
            data=df_corr,
            x=f'C_{i}_BS3_{metric_col}',
            y=f'C_{i}_APS_RPM',
            ci=None,
            ax=axes[row, col]
        )
        axes[row, col].set_title(f"Visit {i}")

    plt.tight_layout()
    plt.show()
