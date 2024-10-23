import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
import scipy.stats as stats
from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn
from statsmodels.stats.anova import AnovaRM

excel_file = pd.ExcelFile('Data Dictionary.xlsx')

sheet_names = excel_file.sheet_names

print(sheet_names)

data_dict = pd.read_excel('Data Dictionary.xlsx',sheet_name = 'Values')

df = df[[
    'C_2_APS_CLK','C_2_APS_CES','C_2_APS_CEC','C_2_APS_CFN','C_2_APS_CSL',
    'C_3_APS_CLK','C_3_APS_CES','C_3_APS_CEC','C_3_APS_CFN','C_3_APS_CSL',
    'C_4_APS_CLK','C_4_APS_CES','C_4_APS_CEC','C_4_APS_CFN','C_4_APS_CSL',
    'C_2_APS_PLK','C_2_APS_PES','C_2_APS_PEC','C_2_APS_PFN','C_2_APS_PSL',
    'C_3_APS_PLK','C_3_APS_PES','C_3_APS_PEC','C_3_APS_PFN','C_3_APS_PSL',
    'C_4_APS_PLK','C_4_APS_PES','C_4_APS_PEC','C_4_APS_PFN','C_4_APS_PSL',
    'C_2_APS_GLK','C_2_APS_GES','C_2_APS_GEC','C_2_APS_GFN','C_2_APS_GSL',              
    'C_3_APS_GLK','C_3_APS_GES','C_3_APS_GEC','C_3_APS_GFN','C_3_APS_GSL',              
    'C_4_APS_GLK','C_4_APS_GES','C_4_APS_GEC','C_4_APS_GFN','C_4_APS_GSL'              
        ]]

df_ANOVA = df.loc[df['C_1_TRT']==1,['CID',
    'C_2_APS_CLK','C_2_APS_CES','C_2_APS_CEC','C_2_APS_CFN','C_2_APS_CSL',
    'C_3_APS_CLK','C_3_APS_CES','C_3_APS_CEC','C_3_APS_CFN','C_3_APS_CSL',
    'C_4_APS_CLK','C_4_APS_CES','C_4_APS_CEC','C_4_APS_CFN','C_4_APS_CSL',
    'C_2_APS_PLK','C_2_APS_PES','C_2_APS_PEC','C_2_APS_PFN','C_2_APS_PSL',
    'C_3_APS_PLK','C_3_APS_PES','C_3_APS_PEC','C_3_APS_PFN','C_3_APS_PSL',
    'C_4_APS_PLK','C_4_APS_PES','C_4_APS_PEC','C_4_APS_PFN','C_4_APS_PSL',
    'C_2_APS_GLK','C_2_APS_GES','C_2_APS_GEC','C_2_APS_GFN','C_2_APS_GSL',              
    'C_3_APS_GLK','C_3_APS_GES','C_3_APS_GEC','C_3_APS_GFN','C_3_APS_GSL',              
    'C_4_APS_GLK','C_4_APS_GES','C_4_APS_GEC','C_4_APS_GFN','C_4_APS_GSL'              
        ]]

df = df.dropna(how = 'any')
df_grp = df.agg(['mean','std']).reset_index()
df_mean = df_grp[df_grp['index']=='mean'].drop('index',axis = 1)
df_std = df_grp[df_grp['index']=='std'].drop('index',axis = 1)

df_mean = df_mean.melt()
df_std = df_std.melt()
df_std['Visit'] = 'V'+df_std['variable'].str.split('_').str[1]
df_std['Toy'] = df_std['variable'].str.split('_').str[3].str[0]
df_std['Behavior'] = df_std['variable'].str.split('_').str[3].str[1:]
df_std = df_std.drop('variable',axis = 1)

df_mean['Visit'] = 'V'+df_mean['variable'].str.split('_').str[1]
df_mean['Toy'] = df_mean['variable'].str.split('_').str[3].str[0]
df_mean['Behavior'] = df_mean['variable'].str.split('_').str[3].str[1:]
df_mean = df_mean.drop('variable',axis = 1)

merged_df = pd.merge(df_mean, df_std, on=['Visit', 'Toy', 'Behavior'], suffixes=('_df_mean', '_df_std'))

def behavior_plot(df, behavior):
    df = df[[
        'C_2_APS_CLK', 'C_2_APS_CES', 'C_2_APS_CEC', 'C_2_APS_CFN', 'C_2_APS_CSL',
        'C_3_APS_CLK', 'C_3_APS_CES', 'C_3_APS_CEC', 'C_3_APS_CFN', 'C_3_APS_CSL',
        'C_4_APS_CLK', 'C_4_APS_CES', 'C_4_APS_CEC', 'C_4_APS_CFN', 'C_4_APS_CSL',
        'C_2_APS_PLK', 'C_2_APS_PES', 'C_2_APS_PEC', 'C_2_APS_PFN', 'C_2_APS_PSL',
        'C_3_APS_PLK', 'C_3_APS_PES', 'C_3_APS_PEC', 'C_3_APS_PFN', 'C_3_APS_PSL',
        'C_4_APS_PLK', 'C_4_APS_PES', 'C_4_APS_PEC', 'C_4_APS_PFN', 'C_4_APS_PSL',
        'C_2_APS_GLK', 'C_2_APS_GES', 'C_2_APS_GEC', 'C_2_APS_GFN', 'C_2_APS_GSL',
        'C_3_APS_GLK', 'C_3_APS_GES', 'C_3_APS_GEC', 'C_3_APS_GFN', 'C_3_APS_GSL',
        'C_4_APS_GLK', 'C_4_APS_GES', 'C_4_APS_GEC', 'C_4_APS_GFN', 'C_4_APS_GSL'
    ]]

    mean_df = df.mean().reset_index().rename(columns={'index': 'Variable', 0: 'Frequency Mean'})
    std_df = df.std().reset_index().rename(columns={'index': 'Variable', 0: 'Frequency Std'})

    df_plot = pd.merge(mean_df, std_df, on='Variable')

    df_plot['Visit'] = 'V' + df_plot['Variable'].str.split('_').str[1]
    df_plot['Toy'] = df_plot['Variable'].str.split('_').str[3].str[0]
    df_plot['Behavior'] = df_plot['Variable'].str.split('_').str[3].str[1:]

    toy_mapping = {'C': 'Cups', 'P': 'Popup', 'G': 'Gumball'}
    df_plot['Toy'] = df_plot['Toy'].replace(toy_mapping)

    df_plot = df_plot.drop('Variable', axis=1)
    df_plot = df_plot[df_plot['Behavior'] == behavior]

    df_plot['Frequency Mean'] = df_plot['Frequency Mean'].round(2)
    df_plot['Frequency Std'] = df_plot['Frequency Std'].round(2)

    df_plot['Mean ± Std'] = df_plot['Frequency Mean'].astype(str) + ' ± ' + df_plot['Frequency Std'].astype(str)

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_plot, x='Visit', y='Frequency Mean', hue='Toy', palette='bright', linewidth=3)

    plt.title(f'Behavior Scores for {behavior}')
    plt.xlabel('Visit')
    plt.ylabel('Score')
    sns.despine()
    plt.legend(title='Toy', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

    pivot_df = df_plot.pivot(index='Toy', columns='Visit', values='Mean ± Std')
    pivot_df.reset_index(inplace=True)

    return pivot_df
tbl = pd.DataFrame()

# Create the 'mean ± std' formatted string
merged_df['mean_std'] = merged_df['value_df_mean'].round(2).astype(str) + ' ± ' + merged_df['value_df_std'].round(2).astype(str)

# Group by 'Toy', 'Behavior', and 'Visit' to handle duplicates
agg_df = merged_df.groupby(['Toy', 'Behavior', 'Visit']).agg({
    'value_df_mean': 'mean',  # Taking average of mean in case of duplicates
    'value_df_std': 'mean',   # Taking average of std in case of duplicates
}).reset_index()

# Create the 'mean ± std' formatted string again after aggregation
agg_df['mean_std'] = agg_df['value_df_mean'].round(2).astype(str) + ' ± ' + agg_df['value_df_std'].round(2).astype(str)

# Now pivot the table with multi-level index (Toy, Behavior)
pivot_table = agg_df.pivot(index=['Toy', 'Behavior'], columns='Visit', values='mean_std')

pivot_table

df_ANOVA = df_ANOVA.dropna(how = 'all', subset = df.columns[1:])

def repeated_measures(df, behavior_type):
    
    df_behavior = df[['CID',
        f'C_2_APS_C{behavior_type}',f'C_3_APS_C{behavior_type}',f'C_4_APS_C{behavior_type}', 
        f'C_2_APS_P{behavior_type}',f'C_3_APS_P{behavior_type}',f'C_4_APS_P{behavior_type}', 
        f'C_2_APS_G{behavior_type}',f'C_3_APS_G{behavior_type}',f'C_4_APS_G{behavior_type}'
    ]]
    df_behavior = df_behavior.dropna(how='any')
    
    df_behavior = df_behavior.melt(id_vars = 'CID')

    df_behavior['Visit'] = 'V' + df_behavior['variable'].str.split('_').str[1]
    df_behavior['Toy'] = df_behavior['variable'].str.split('_').str[3].str[0]

    df_behavior = df_behavior.drop('variable',axis = 1)
    aov = AnovaRM(df_behavior, depvar='value', subject='CID', within=['Toy', 'Visit'])
    anova_results = aov.fit()
    print(anova_results)

# Usage 
# repeated_measures(df_ANOVA,'ES')

def create_df(df, behavior_type):
    
    df_behavior = df[['CID',
        f'C_2_APS_C{behavior_type}',f'C_3_APS_C{behavior_type}',f'C_4_APS_C{behavior_type}', 
        f'C_2_APS_P{behavior_type}',f'C_3_APS_P{behavior_type}',f'C_4_APS_P{behavior_type}', 
        f'C_2_APS_G{behavior_type}',f'C_3_APS_G{behavior_type}',f'C_4_APS_G{behavior_type}'
    ]]
    df_behavior = df_behavior.dropna(how='any')
    
    df_behavior = df_behavior.melt(id_vars = 'CID')

    df_behavior['Visit'] = 'V' + df_behavior['variable'].str.split('_').str[1]
    df_behavior['Toy'] = df_behavior['variable'].str.split('_').str[3].str[0]

    df_behavior = df_behavior.drop('variable',axis = 1)
    return df_behavior

# Summary
# Across the five tests, the main effect of Visit is significant. Toy effect is significant for case 1,3,5
# The interaction effect is significant in Case 2.

# How to Interpret Tukey's HSD Results

# Group Comparisons: A list of comparisons between each combination of Toy and Visit.
# Mean Difference: The difference in means between the groups being compared.
# P-Value: The significance of the comparison.
# Reject Null: Whether the null hypothesis can be rejected for that comparison (True or False).

def create_interaction_plot(data, x, trace, response, figsize=(10, 6)):
    means = data.groupby([x, trace])[response].mean().unstack()
    sems = data.groupby([x, trace])[response].sem().unstack()
    fig, ax = plt.subplots(figsize=figsize)
    markers = ['o', 's', '^', 'D']  # Different markers for different lines
    colors = sns.color_palette("husl", len(means.columns))
    
    for i, (col, color) in enumerate(zip(means.columns, colors)):
        ax.errorbar(means.index, means[col], yerr=sems[col],
                   marker=markers[i % len(markers)], color=color,
                   label=f'{trace}={col}', capsize=5)
    
    ax.set_xlabel(x)
    ax.set_ylabel(f'Mean {response}')
    ax.set_title(f'Interaction Plot: {x} × {trace} on {response}')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(title=trace)
    
    summary = analyze_interaction(data, x, trace, response)
    
    return fig, summary

def analyze_interaction(data, x, trace, response):
    means = data.groupby([x, trace])[response].mean().unstack()
    
    slopes = {}
    for col in means.columns:
        slope = np.polyfit(range(len(means.index)), means[col], 1)[0]
        slopes[col] = slope
    
    # Determine if lines are parallel (approximately)
    slope_values = list(slopes.values())
    slope_diff = max(slope_values) - min(slope_values)
    parallel_threshold = 0.1 * np.mean(abs(np.array(slope_values)))
    
    # Calculate if lines cross
    lines_cross = False
    for i in range(len(means.index)-1):
        current_diffs = means.iloc[i].diff().dropna()
        next_diffs = means.iloc[i+1].diff().dropna()
        if any(current_diffs * next_diffs < 0):
            lines_cross = True
            break
    
    # Prepare interpretation
    summary = {
        'slopes': slopes,
        'parallel_lines': slope_diff <= parallel_threshold,
        'lines_cross': lines_cross,
        'interpretation': []
    }
    
    # Add interpretation hints
    if summary['parallel_lines']:
        summary['interpretation'].append(
            "The lines appear to be roughly parallel, suggesting NO significant interaction "
            "between the factors. Each factor may have main effects, but they likely act independently."
        )
    else:
        summary['interpretation'].append(
            "The lines are not parallel, suggesting a potential interaction between the factors. "
            "The effect of one factor depends on the level of the other factor."
        )
    
    if summary['lines_cross']:
        summary['interpretation'].append(
            "The lines cross, indicating a strong interaction effect. "
            "The relationship between one factor and the response variable reverses "
            "at different levels of the other factor."
        )
    
    return summary

fig, summary = create_interaction_plot(df_interaction, 'Toy', 'Visit', 'value')
