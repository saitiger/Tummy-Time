df_long = pd.melt(df, id_vars=['CID'], value_vars=['C_2_APS_TWS', 'C_3_APS_TWS', 'C_4_APS_TWS'], 
                  var_name='Visit', value_name='APS_TWS')
df_long['Visit'] = df_long['Visit'].map({'C_2_APS_TWS': 1, 'C_3_APS_TWS': 2, 'C_4_APS_TWS': 3})

# Remove NaN values
df_long = df_long.dropna()

# Test for normality
_, p_value = stats.shapiro(df_long['APS_TWS'])
print(f"Shapiro-Wilk test p-value: {p_value:.4f}")

# Visual check for normality
plt.figure(figsize=(10, 4))
plt.subplot(121)
stats.probplot(df_long['APS_TWS'], dist="norm", plot=plt)
plt.title("Q-Q Plot")
plt.subplot(122)
df_long['APS_TWS'].hist()
plt.title("Histogram")
plt.tight_layout()
plt.show()

# Perform one-way ANOVA
groups = [group for _, group in df_long.groupby('Visit')['APS_TWS']]
f_value, p_value = stats.f_oneway(*groups)
print(f"One-way ANOVA results:")
print(f"F-value: {f_value:.4f}")
print(f"p-value: {p_value:.4f}")

# Post-hoc test (Tukey's HSD)
tukey_results = pairwise_tukeyhsd(df_long['APS_TWS'], df_long['Visit'])
print("\nTukey's HSD Test Results:")
print(tukey_results)

# Calculate means for each visit
means = df_long.groupby('Visit')['APS_TWS'].mean()
print("\nMeans for each visit:")
print(means)
