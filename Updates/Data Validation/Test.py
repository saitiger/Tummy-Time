df_grp = df.agg(['mean','std']).reset_index()

df_grp['C_T'] = df_grp[['X_2_CEN','X_2_CEN','X_2_CEN']].sum(axis=1)
df_grp['G_T'] = df_grp[['X_2_GEN','X_2_GEN','X_2_GEN']].sum(axis=1)
df_grp['P_T'] = df_grp[['X_2_PEN','X_2_PEN','X_2_PEN']].sum(axis=1)

groups = {
    'C': [df_grp[df_grp['Treatment Group'] == 1]['C_T'], df_stats3[df_stats3['Treatment Group'] == 5]['C_T']],
    'G': [df_grp[df_grp['Treatment Group'] == 1]['G_T'], df_stats3[df_stats3['Treatment Group'] == 5]['G_T']],
    'P': [df_grp[df_grp['Treatment Group'] == 1]['P_T'], df_stats3[df_stats3['Treatment Group'] == 5]['P_T']],
}

results = {}
for prod, group in groups.items():
    f_stat, p_value = f_oneway(*group)
    results[toy] = {'F-statistic': f_stat, 'p-value': p_value}

for prod, result in results.items():
    print(f'Prod {prod}: F-statistic = {result["F-statistic"]}, p-value = {result["p-value"]}')

homogeneity_results = {}
for prod, group in groups.items():
    stat, p_value = levene(*group)
    homogeneity_results[toy] = {'Levene statistic': stat, 'p-value': p_value}

# Print Levene's Test results
for prod, result in homogeneity_results.items():
    print(f'Prod {prod}: Levene statistic = {result["Levene statistic"]}, p-value = {result["p-value"]}')

metrics_columns = {
    'EF': ['CEF', 'GEF', 'PEF'],
    'XZ': ['CXZ', 'GXZ', 'PXZ']
}

results = {}

for metric, columns in metrics_columns.items():
    data = []
    
    for toy, col in zip(toys, columns):
        group = df_stats[df_stats['Treatment Group'] == 1][col].dropna()  
        data.extend([(toy, value) for value in group])
    
    df_comparision = pd.DataFrame(data, columns=['Product', 'Score'])

    model = ols('Score ~ Product', data = df_comparision).fit()
    anova_results = sm.stats.anova_lm(model, typ=2)

    if anova_results['PR(>F)'][0] < 0.05:  # Check p-value
        tukey_result = pairwise_tukeyhsd(df_comparision['Score'], df_comparision['Product'], alpha=0.05)
        results[metric] = tukey_result
    else:
        results[metric] = "No significant difference found."

for metric, result in results.items():
    print(f'Metric: {metric}')
    if isinstance(result, str):
        print(result)
    else:
        print(result.summary())

df_mean = df_grp[df_grp['index']=='mean'].drop('index',axis = 1)
df_std = df_grp[df_grp['index']=='std'].drop('index',axis = 1)

merged_df = pd.merge(df_mean, df_std, on=['Interaction_Number', 'Product', 'Metric'], suffixes=('_df_mean', '_df_std'))

tbl = pd.DataFrame()

# Create the 'mean ± std' formatted string
merged_df['mean_std'] = merged_df['value_df_mean'].round(2).astype(str) + ' ± ' + merged_df['value_df_std'].round(2).astype(str)

# Group by 'Toy', 'Behavior', and 'Visit' to handle duplicates
agg_df = merged_df.groupby(['Product', 'Metric', 'Interaction_Number']).agg({
    'value_df_mean': 'mean',  # Taking average of mean in case of duplicates
    'value_df_std': 'mean',   # Taking average of std in case of duplicates
}).reset_index()

# Create the 'mean ± std' formatted string again after aggregation
agg_df['mean_std'] = agg_df['value_df_mean'].round(2).astype(str) + ' ± ' + agg_df['value_df_std'].round(2).astype(str)

# Now pivot the table with multi-level index (Toy, Behavior)
pivot_table = agg_df.pivot(index=['Product', 'Behavior'], columns='Interaction_Number', values='mean_std')

pivot_table

df_interaction = create_df(df_ANOVA,'FN')

tukey_results = pairwise_tukeyhsd(df_interaction['Score'], 
                                    df_interaction['Product'].astype(str) + ' | ' + df_interaction['Interaction_Number'].astype(str))
print(tukey_results)

plt.figure(figsize=(10, 6))
sns.pointplot(data=df_interaction, x='Interaction_Number', y='Score', hue='Product', 
               markers=["o", "s", "D"], dodge=True, errorbar='sd')
plt.title('Interaction Plot: Product vs. Interaction_Number')
plt.xlabel('Interaction_Number')
plt.ylabel('Dependent Variable')
plt.legend(title='Product')
sns.despine()
plt.show()
