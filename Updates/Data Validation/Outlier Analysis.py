search_terms = ["LK", "ES", "EC"]
check1 = [c for c in df.columns 
             if any(term in c for term in search_terms) 
             and (len(c.split("_")[3])==2 or (len(c.split("_")[3])==5))]

# Contribution of top 10 C_2_APS_LK
df['C_2_APS_LK'].nlargest(10).sum()/df['C_2_APS_LK'].sum()*100.0

mean = df['C_2_APS_EC'].mean()
std_dev = df['C_2_APS_EC'].std()
outliers = df[(df['C_2_APS_EC'] < mean - 2 * std_dev) | (df['C_2_APS_EC'] > mean + 2 * std_dev)][['C_2_APS_EC', 'C_2_APS_ECsub']]

outliers_count = outliers.shape[0]

print("Number of outliers:", outliers_count)
print("Outlier rows:")
outliers['C_2_APS_EC']

search_terms = ["LK", "ES", "EC"]
check = [c for c in df.columns if any(term in c for term in search_terms) and len(c.split("_")[3]) == 2]
res = []

for c in check:
    difference_in_behaviour_score = []
    mean = df[c].mean()
    std_dev = df[c].std()
    
    # Identify outliers
    outliers = df[(df[c] < mean - 2 * std_dev) | (df[c] > mean + 2 * std_dev)][[c, c + 'sub']]
    outlier_value = outliers[c].tolist()
    corresponding_outlier_sub = outliers[c + 'sub'].tolist()
    outliers_count = outliers.shape[0]
    outlier_idx = df[(df[c] < mean - 2 * std_dev) | (df[c] > mean + 2 * std_dev)].index.tolist()

    
    # Calculate top 10 contribution
    top_10_contribution = df[c].nlargest(10).sum() / df[c].sum() * 100.0
    
    # Compute difference in behaviour score for outliers
    # outlier_bool = (df[c] < mean - 2 * std_dev) | (df[c] > mean + 2 * std_dev)
    # difference_in_behaviour_score = (df[outlier_bool][c] - df[outlier_bool][c + 'sub']).tolist()
    
    res.append([c, outliers_count, top_10_contribution, outlier_value, corresponding_outlier_sub,outlier_idx])

outliers_df = pd.DataFrame(res,columns = ['Column','Number_of_Outliers','Top_10_Contribution(%)','Outlier_Value','Corresponding_Sub_Score','Outlier_Index_Array'])
outliers_df['Behaviour'] = outliers_df['Column'].str.split("_").str[3]
outliers_df['Visit'] = 'V'+outliers_df['Column'].str.split("_").str[1]
outliers_df.drop(columns = 'Column',axis = 1,inplace = True)
new_column_order = ['Visit', 'Behaviour'] + list(outliers_df.columns[0:-2])
outliers_df = outliers_df[new_column_order]
outliers_df.head()

# Flatten the Outlier_Index_Array column
outlier_indices = [item for sublist in outliers_df['Outlier_Index_Array'] for item in sublist]

# Count occurrences of each element
from collections import Counter

counts = Counter(outlier_indices)

# Find elements repeated at least once
repeated_elements = [key for key, value in counts.items() if value > 1]

print("Repeated elements:", repeated_elements)

check1.sort()
df_multiple_times_outlier = df[check1].iloc[repeated_elements]

sub_cols = [c for c in df_multiple_times_outlier.columns if 'sub' in c]
-df_multiple_times_outlier.diff(axis = 1)[sub_cols]

# df.iloc[df['C_2_APS_EC'].nlargest(10).reset_index()['index'].tolist()][['C_2_APS_EC','C_2_APS_ECsub']]

print(df['C_2_APS_EC'].mean()+2*df['C_2_APS_EC'].std())
print(df['C_2_APS_EC'].std())
df['C_2_APS_EC'].mean()

df.iloc[df['C_5_APS_ES'].nlargest(10).reset_index()['index'].tolist()][['C_5_APS_ES','C_5_APS_ESsub']].sort_index(axis=0, ascending=True, inplace=False)

df[check].pivot(index = 'V')

df_pivot[(df_pivot['V']=='V2')&(df_pivot['Behavior']=='LK')].T[:62]

df.iloc[4900:5700]

df_plot = df.groupby('A').agg({'Overall class':pd.Series.mode,'B':'mean','Acceleration':'mean'}).round(4).reset_index()

df.iloc[4900:5700]['Overall class'].value_counts()

df.iloc[5700::100]['Overall class'].value_counts()

df['A'] = pd.to_datetime(df['A'], format='%Y-%m-%d %H:%M:%S:%f').dt.floor('s')
cols = df.columns[1:].tolist()
cols
df_plot = df.groupby('A').agg({'Overall class':pd.Series.mode,'B':'mean','Acceleration':'mean'}).round(4).reset_index()
