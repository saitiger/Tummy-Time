df_plot['Difference'] = df_plot['Values'] - df_plot['Values'].shift(-1)

df_difference = df_plot[df_plot['Scores'] == 'GA'][['Visit', 'Difference']]

df_difference
sns.lineplot(data = df_difference, x = 'Visit', y = 'Difference')
sns.despine()
plt.show()

sns.lineplot(data = df_plot, x = 'Visit', y = 'Values', hue = 'Scores')
sns.despine()
plt.show()

df_differences = (
    df_plot.pivot(index='Visit', columns='Scores', values='Values')
    .assign(Difference=lambda x: x['GA'] - x['UA'])
    .reset_index()
)

sns.lineplot(data=df_plot, x='Visit', y='Values', hue='Scores', marker='o')

for index, row in df_differences.iterrows():
    x = row['Visit']
    y = (row['GA'] + row['UA']) / 2
    difference = row['Difference']
    plt.text(x, y, f'{difference:.2f}', color='red', ha='center', fontsize=10)

sns.despine()
plt.ylabel("Values")
plt.show()

df[cols].isna().sum()
df[df['Month_1_GA'].isna()]
df[df['Month_2_GA'].isna()]
df[df['Month_1_UA'].isna()]
df[df['Month_1_UA'].isna()]

def withinvisit_ttest(dataset, visit_num):
    if visit_num not in [2, 3, 4, 5]:
        return None, None, "Enter a valid Visit Number (2, 3, 4, or 5)."

    try:
        filtered_data = dataset[
            (~dataset[f'C_{visit_num}_GA'].isna()) &
            (~dataset[f'C_{visit_num}_UA'].isna())
        ][[f'C_{visit_num}_GA', f'C_{visit_num}_UA']]
    except KeyError as e:
        return None, None, f"Column not found in dataset: {e}"

    if filtered_data.empty:
        return None, None, "No valid data for t-test after filtering."

    t_stat, p_value = ttest_rel(
        filtered_data[f'C_{visit_num}_GA'],
        filtered_data[f'C_{visit_num}_UA']
    )

    if p_value < 0.01:
        interpretation = (
            f"The difference within visit {visit_num} between GA and UA is highly statistically significant "
            f"(p-value = {p_value:.4f}). There is strong evidence to suggest a meaningful change."
        )
    elif p_value < 0.05:
        interpretation = (
            f"The difference within visit {visit_num} between GA and UA is statistically significant "
            f"(p-value = {p_value:.4f}). There is moderate evidence to suggest a meaningful change."
        )
    else:
        interpretation = (
            f"The difference within visit {visit_num} between GA and UA is not statistically significant "
            f"(p-value = {p_value:.4f}). There is insufficient evidence to suggest a meaningful change."
        )

    return t_stat, p_value, interpretation

def across_visit_ttest(dataset,initial_visit, score_type):

    if not (2 <= initial_visit <= 4):
        return None, None, "Start visit must be between 2 and 4 (inclusive)."
    if score_type not in ['GA', 'UA']:
        return None, None, "Score type must be either 'GA' or 'UA'."

    final_visit = initial_visit + 1

    try:
        filtered_data = dataset[
            (~dataset[f'C_{initial_visit}_{score_type}'].isna()) &
            (~dataset[f'C_{final_visit}_{score_type}'].isna())
        ][[f'C_{initial_visit}_{score_type}', f'C_{final_visit}_{score_type}']]
    except KeyError as e:
        return None, None, f"Column not found in dataset: {e}"

    if filtered_data.empty:
        return None, None, "No valid data for t-test after filtering."

    t_stat, p_value = ttest_rel(
        filtered_data[f'C_{initial_visit}_{score_type}'],
        filtered_data[f'C_{final_visit}_{score_type}']
    )

    if p_value < 0.01:
        interpretation = (
            f"The difference between visits {initial_visit} and {final_visit} for {score_type} is highly statistically significant "
            f"(p-value = {p_value:.4f}). There is strong evidence to suggest a meaningful change."
        )
    elif p_value < 0.05:
        interpretation = (
            f"The difference between visits {initial_visit} and {final_visit} for {score_type} is statistically significant "
            f"(p-value = {p_value:.4f}). There is moderate evidence to suggest a meaningful change."
        )
    else:
        interpretation = (
            f"The difference between visits {initial_visit} and {final_visit} for {score_type} is not statistically significant "
            f"(p-value = {p_value:.4f}). There is insufficient evidence to suggest a meaningful change."
        )

    return t_stat, p_value, interpretation
