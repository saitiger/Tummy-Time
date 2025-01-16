# Perform Kruskal-Wallis H-test
groups = [group for _, group in df_long.groupby('Visit')['APS_TWS']]
h_statistic, p_value = stats.kruskal(*groups)

print("Kruskal-Wallis Test Results:")
print(f"H-statistic: {h_statistic:.4f}")
print(f"p-value: {p_value:.4f}")

# Calculate medians for each visit
medians = df_long.groupby('Visit')['APS_TWS'].median()
print("\nMedians for each visit:")
print(medians)

# Calculate sample sizes for each visit
sample_sizes = df_long.groupby('Visit')['APS_TWS'].count()
print("\nSample sizes for each visit:")
print(sample_sizes)
