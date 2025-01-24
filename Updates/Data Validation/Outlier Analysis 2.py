print(df[df['current height']>df['threshold']]['Time'].count()/100)
print(df['Toy status'].value_counts()[1]/100)

threshold_check = set(df[df['current height']<df['threshold']].index)
toy_status = set(df[df['Toy status']==1].index)
toy_status.intersection(threshold_check) # Empty set 

print(df[df['current height']>df['threshold']]['Toy status'].count())

# Checking if there are multiple "phase of data collection"  
print(df[df['current height']>df['threshold']].groupby(df['phase of data collection'])['Toy status'].value_counts())

# Problematic Rows
print("Number of Rows : ",df[(df['current height']>df['threshold'])&(df['Toy status']==0)].shape[0])
print(df[(df['current height']>df['threshold'])&(df['Toy status']==0)].head())

print(df[(df['current height']>df['threshold'])&(df['Toy status']==0)]['set_threshold_flag'].value_counts())
print(df[(df['current height']>df['threshold'])&(df['Toy status']==1)].head())

filt_df_1 = df[(df['current height']>df['threshold'])&(df['Toy status']==1)]
print("Total Number of Rows : " ,filt_df_1.shape[0])
print(filt_df_1[filt_df_1['filtered height']>filt_df_1['threshold']]['Time'].count())

filt_df_2 = df[(df['current height']>df['threshold'])&(df['Toy status']==0)]
print("Total Number of Rows : " ,filt_df_2.shape[0])
print(filt_df_2[filt_df_2['filtered height']>filt_df_2['threshold']]['Time'].count())

filt_df_1 = df[(df['average_height']>df['threshold'])&(df['Toy status']==1)]
print("Total Number of Rows : " ,filt_df_1.shape[0])
print(filt_df_1[filt_df_1['average_height']>filt_df_1['threshold']]['Time'].count())

filt_df_2 = df[(df['average_height']>df['threshold'])&(df['Toy status']==0)]
print("Total Number of Rows : " ,filt_df_2.shape[0])
print(filt_df_2[filt_df_2['average_height']>filt_df_2['threshold']]['Time'].count())

# Filtered Height, average_height,Head above threshold, set_threshold_flag,face_detection_flag,Time aren't the issue

print(df[(df['current height']>df['threshold'])&(df['Toy status']==1)].describe())
print(df['face_detection_flag'].unique())
print(df.groupby('Toy status')['face_detection_flag'].value_counts())
print(df[(df['current height']>df['threshold'])&(df['Toy status']==0)].groupby('face_detection_flag')['Toy status'].value_counts())
print(df.groupby('set_threshold_flag')['set_threshold_flag'].value_counts())

print(df['set_threshold_flag'].unique())

# Finding differences between the two datasets
diff_df = pd.merge(filt_df_1, filt_df_2, how='outer', indicator=True)
# Rows only in filt_df_1 or filt_df_2 (i.e., differences between the two sets)
print(diff_df[diff_df['_merge'] != 'both'])

print("Summary statistics for filt_df_1:")
print(filt_df_1.describe())
print("Summary statistics for filt_df_2:")
print(filt_df_2.describe())

# Checking for missing values in both DataFrames
print("Missing values in filt_df_1:")
print(filt_df_1.isnull().sum())

print("Missing values in filt_df_2:")
print(filt_df_2.isnull().sum())

# Comparing distributions of 'current height'
sns.histplot(filt_df_1['current height'], label='filt_df_1 - current height', kde=True)
sns.histplot(filt_df_2['current height'], label='filt_df_2 - current height', kde=True)
plt.legend()
plt.show()

# Inspecting categorical columns value counts
for col in filt_df_1.select_dtypes(include=['object']).columns:
    print(f"Value counts for {col} in filt_df_1:")
    print(filt_df_1[col].value_counts())

    print(f"Value counts for {col} in filt_df_2:")
    print(filt_df_2[col].value_counts())

