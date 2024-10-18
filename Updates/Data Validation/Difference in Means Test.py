import pandas as pd 
import numpy as np
import scipy.stats as stats
from scipy.stats import levene, f_oneway
import matplotlib.pyplot as plt
import seaborn as sns 
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM
import warnings

df_stats2 = df.loc[((df['C_1_TRT']==1)|(df['C_1_TRT']==2)),
            ['C_1_TRT',
             'C_2_APS_CES','C_3_APS_CES','C_4_APS_CES',
             'C_2_APS_PES','C_3_APS_PES','C_4_APS_PES',
             'C_2_APS_GES','C_3_APS_GES','C_4_APS_GES',
            ]].dropna(how = 'any')

df_stats3 = df.loc[((df['C_1_TRT']==1)|(df['C_1_TRT']==2)),
            ['C_1_TRT',
             'C_2_APS_CES','C_3_APS_CES','C_4_APS_CES',
             'C_2_APS_PES','C_3_APS_PES','C_4_APS_PES',
             'C_2_APS_GES','C_3_APS_GES','C_4_APS_GES',
             'C_2_APS_CLK','C_3_APS_CLK','C_4_APS_CLK',
             'C_2_APS_PLK','C_3_APS_PLK','C_4_APS_PLK',
             'C_2_APS_GLK','C_3_APS_GLK','C_4_APS_GLK',
             'C_2_APS_CFN','C_3_APS_CFN','C_4_APS_CFN',
             'C_2_APS_PFN','C_3_APS_PFN','C_4_APS_PFN',
             'C_2_APS_GFN','C_3_APS_GFN','C_4_APS_GFN',
             'C_2_APS_CSL','C_3_APS_CSL','C_4_APS_CSL',
             'C_2_APS_PSL','C_3_APS_PSL','C_4_APS_PSL',
             'C_2_APS_GSL','C_3_APS_GSL','C_4_APS_GSL',
             'C_2_APS_CEC','C_3_APS_CEC','C_4_APS_CEC',
             'C_2_APS_PEC','C_3_APS_PEC','C_4_APS_PEC',
             'C_2_APS_GEC','C_3_APS_GEC','C_4_APS_GEC',
            ]].dropna(how = 'any')


df_stats3['CES'] = df_stats3[['C_2_APS_CES','C_3_APS_CES','C_4_APS_CES']].sum(axis=1)
df_stats3['GES'] = df_stats3[['C_2_APS_GES','C_3_APS_GES','C_4_APS_GES']].sum(axis=1)
df_stats3['PES'] = df_stats3[['C_2_APS_PES','C_3_APS_PES','C_4_APS_PES']].sum(axis=1)

df_stats3['CSL'] = df_stats3[['C_2_APS_CSL','C_3_APS_CSL','C_4_APS_CSL']].sum(axis=1)
df_stats3['GSL'] = df_stats3[['C_2_APS_GSL','C_3_APS_GSL','C_4_APS_GSL']].sum(axis=1)
df_stats3['PSL'] = df_stats3[['C_2_APS_GSL','C_3_APS_PSL','C_4_APS_PSL']].sum(axis=1)


df_stats3['CLK'] = df_stats3[['C_2_APS_CLK','C_3_APS_CLK','C_4_APS_CLK']].sum(axis=1)
df_stats3['GLK'] = df_stats3[['C_2_APS_GLK','C_3_APS_GLK','C_4_APS_GLK']].sum(axis=1)
df_stats3['PLK'] = df_stats3[['C_2_APS_PLK','C_3_APS_PLK','C_4_APS_PLK']].sum(axis=1)


df_stats3['CEC'] = df_stats3[['C_2_APS_CEC','C_3_APS_CEC','C_4_APS_CEC']].sum(axis=1)
df_stats3['GEC'] = df_stats3[['C_2_APS_GEC','C_3_APS_GEC','C_4_APS_GEC']].sum(axis=1)
df_stats3['PEC'] = df_stats3[['C_2_APS_PEC','C_3_APS_PEC','C_4_APS_PEC']].sum(axis=1)


df_stats3['CFN'] = df_stats3[['C_2_APS_CFN','C_3_APS_CFN','C_4_APS_CFN']].sum(axis=1)
df_stats3['GFN'] = df_stats3[['C_2_APS_GFN','C_3_APS_GFN','C_4_APS_GFN']].sum(axis=1)
df_stats3['PFN'] = df_stats3[['C_2_APS_PFN','C_3_APS_PFN','C_4_APS_PFN']].sum(axis=1)


df_stats3 = df_stats3[['C_1_TRT',
                      'CES','GES','PES',
                      'CFN','GFN','PFN',
                      'CEC','GEC','PEC',
                      'CLK','GLK','PLK',
                      'CSL','GSL','PSL'                     
                      ]]

groups = {
    'C': [df_stats3[df_stats3['C_1_TRT'] == 1]['CES'], df_stats3[df_stats3['C_1_TRT'] == 2]['CES']],
    'G': [df_stats3[df_stats3['C_1_TRT'] == 1]['GES'], df_stats3[df_stats3['C_1_TRT'] == 2]['GES']],
    'P': [df_stats3[df_stats3['C_1_TRT'] == 1]['PES'], df_stats3[df_stats3['C_1_TRT'] == 2]['PES']],
}

results = {}
for toy, group in groups.items():
    f_stat, p_value = f_oneway(*group)
    results[toy] = {'F-statistic': f_stat, 'p-value': p_value}

for toy, result in results.items():
    print(f'Toy {toy}: F-statistic = {result["F-statistic"]}, p-value = {result["p-value"]}')


groups = {
    'C': [df_stats3[df_stats3['C_1_TRT'] == 1]['CES'], df_stats3[df_stats3['C_1_TRT'] == 2]['CES']],
    'G': [df_stats3[df_stats3['C_1_TRT'] == 1]['GES'], df_stats3[df_stats3['C_1_TRT'] == 2]['GES']],
    'P': [df_stats3[df_stats3['C_1_TRT'] == 1]['PES'], df_stats3[df_stats3['C_1_TRT'] == 2]['PES']],
}

homogeneity_results = {}
for toy, group in groups.items():
    stat, p_value = levene(*group)
    homogeneity_results[toy] = {'Levene statistic': stat, 'p-value': p_value}

# Print Levene's Test results
for toy, result in homogeneity_results.items():
    print(f'Toy {toy}: Levene statistic = {result["Levene statistic"]}, p-value = {result["p-value"]}')


results = {}

for toy in toys:
    for metric in metrics:
        metric_col = f'{toy[0]}{metric}'
        
        group1 = df_stats3[df_stats3['C_1_TRT'] == 1][metric_col]
        group2 = df_stats3[df_stats3['C_1_TRT'] == 2][metric_col]
        
        if group1.nunique() > 1 and group2.nunique() > 1:
            f_stat, p_value = f_oneway(group1, group2)
            
            mean1 = group1.mean()
            mean2 = group2.mean()
            
            results[(toy, metric)] = {
                'Mean Group 1': mean1,
                'Mean Group 2': mean2,
                'Mean Difference': mean2 - mean1,  # Difference: Group 2 - Group 1
                'F-statistic': f_stat,
                'p-value': p_value
            }
        else:
            results[(toy, metric)] = {
                'Mean Group 1': group1.mean() if not group1.empty else None,
                'Mean Group 2': group2.mean() if not group2.empty else None,
                'Mean Difference': None,
                'F-statistic': None,
                'p-value': None,
                'Warning': 'One or both groups have constant values or insufficient variability.'
            }

for (toy, metric), result in results.items():
    print(f'Toy: {toy}, Metric: {metric}')
    print(f'  Mean Group 1: {result["Mean Group 1"]}')
    print(f'  Mean Group 2: {result["Mean Group 2"]}')
    print(f'  Mean Difference: {result["Mean Difference"]}')
    print(f'  F-statistic: {result["F-statistic"]}, p-value: {result["p-value"]}')
    if 'Warning' in result:
        print(f'  Warning: {result["Warning"]}\n')
    else:
        print()


# Define the toys and metrics

results = {}

# Iterate through each metric
for metric in metrics:
    # Prepare to collect data for both treatment groups
    for treatment in [1, 2]:
        # Get the metric column name
        metric_col = f'{toys[0][0]}{metric}'  # Assuming metric columns follow the naming convention
        
        # Collect data for each toy within the same treatment group
        groups = [df_stats3[(df_stats3['C_1_TRT'] == treatment)][f'{toy[0]}{metric}'] for toy in toys]
        
        # Check if there are valid groups to analyze
        if all(group.nunique() > 1 for group in groups):
            f_stat, p_value = f_oneway(*groups)
            
            # Calculate means for each group
            means = [group.mean() for group in groups]
            
            results[(treatment, metric)] = {
                'Means': means,
                'F-statistic': f_stat,
                'p-value': p_value
            }
        else:
            results[(treatment, metric)] = {
                'Means': [group.mean() if not group.empty else None for group in groups],
                'F-statistic': None,
                'p-value': None,
                'Warning': 'One or more groups have constant values or insufficient variability.'
            }

# Print the results
for (treatment, metric), result in results.items():
    print(f'Treatment Group: {treatment}, Metric: {metric}')
    print(f'  Means: {result["Means"]}')
    print(f'  F-statistic: {result["F-statistic"]}, p-value: {result["p-value"]}')
    if 'Warning' in result:
        print(f'  Warning: {result["Warning"]}\n')
    else:
        print()

metrics_columns = {
    'EC': ['CEC', 'GEC', 'PEC'],
    'FN': ['CFN', 'GFN', 'PFN']
}

results = {}

for metric, columns in metrics_columns.items():
    data = []
    
    for toy, col in zip(toys, columns):
        group = df_stats3[df_stats3['C_1_TRT'] == 1][col].dropna()  # Change to 2 if needed
        data.extend([(toy, value) for value in group])
    
    df_comparision = pd.DataFrame(data, columns=['Toy', 'Score'])

    model = ols('Score ~ Toy', data = df_comparision).fit()
    anova_results = sm.stats.anova_lm(model, typ=2)

    if anova_results['PR(>F)'][0] < 0.05:  # Check p-value
        tukey_result = pairwise_tukeyhsd(df_comparision['Score'], df_comparision['Toy'], alpha=0.05)
        results[metric] = tukey_result
    else:
        results[metric] = "No significant difference found."

for metric, result in results.items():
    print(f'Metric: {metric}')
    if isinstance(result, str):
        print(result)
    else:
        print(result.summary())


df_melt = pd.melt(df_stats3, id_vars=['C_1_TRT'], value_vars=df_stats3.columns[1:],
                  var_name='Metric', value_name='Value')

# Perform ANOVA
model = ols('Value ~ C(C_1_TRT) + C(Metric)', data=df_melt).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

print("ANOVA results:")
print(anova_table)

df_stats = df_stats.reset_index().drop(columns = 'index')
df_stats.groupby('C_1_TRT').mean().reset_index()

df_stats.groupby('C_1_TRT').agg(['min','max','mean']).reset_index()

sns.kdeplot(data = df_stats, x = 'C_2_APS_RPM', hue = 'C_1_TRT',palette = 'bright')
sns.despine()
plt.show()

def create_kdeplot(df, visit_columns, trt_column, palette='bright'):
    plt.figure(figsize=(12, 6))
    
    # Get unique treatment values
    trt_values = df[trt_column].unique()
    
    # Define line styles for each visit
    line_styles = ['-', ':', '--', '-.']  # Add more if needed
    
    for i, visit_col in enumerate(visit_columns):
        for trt in trt_values:
            sns.kdeplot(
                data=df[df[trt_column] == trt], 
                x=visit_col, 
                color=palette[trt] if isinstance(palette, dict) else None,
                linestyle=line_styles[i % len(line_styles)],
                label=f'Visit {i+2}: {trt}'
            )
    
    plt.legend(title='Visit: Treatment')
    sns.despine()
    plt.show()

visit_columns = ['C_2_APS_RPM', 'C_3_APS_RPM', 'C_4_APS_RPM']
create_kdeplot(df_stats, visit_columns, 'C_1_TRT', palette={1.0: 'blue', 2.0: 'red'})


group1 = df_stats[df_stats['C_1_TRT'] == 1.0]
group2 = df_stats[df_stats['C_1_TRT'] == 2.0]

anova_results = {}
for column in ['C_2_APS_RPM', 'C_3_APS_RPM', 'C_4_APS_RPM']:
    f_val, p_val = stats.f_oneway(group1[column], group2[column])
    anova_results[column] = {'F-value': f_val, 'P-value': p_val}

anova_results

t_test_results = {}
for column in ['C_2_APS_RPM', 'C_3_APS_RPM', 'C_4_APS_RPM']:
    t_stat, p_val = stats.ttest_ind(group1[column], group2[column], equal_var=False)  # Welch's t-test
    t_test_results[column] = {'T-statistic': t_stat, 'P-value': p_val}

t_test_results
