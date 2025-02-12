# Drop these values
false_flags = round(100.0*b1[b1['face_detection_flag']==False]['Time'].count()/b1['Time'].count(),2)
duration  = b1[b1['face_detection_flag']==False]['Time'].count()/20
total_duration = b1['Time'].count()/20
print("False Face Detection Flags")
print(f"Percentage of Total Data : {false_flags}%")
print(f"Duration of False Flags : {duration//60} Minutes {round(duration%60,2)} Seconds")
print(f"Total Duration of Video : {total_duration//60} Minutes {round(total_duration%60,2)} Seconds")

# Frequency Check
b1['Time'] = pd.to_datetime(b1['Time'],format = '%Y-%m-%d %H:%M:%S.%f')
std = b1['Time'].diff().dt.total_seconds().std()
b1['Time'].diff().dt.total_seconds().describe()
time_diffs = b1['Time'].diff().dt.total_seconds().dropna()

std = time_diffs.std()

lower_bound = 0.05 - std
upper_bound = 0.05 + std

above_count = time_diffs[time_diffs > upper_bound].count()
below_count = time_diffs[time_diffs < lower_bound].count()

print(f"Count above {upper_bound:.5f} seconds: {above_count}")
print(f"Count below {lower_bound:.5f} seconds: {below_count}")

sns.histplot(b1['Time'].diff().dt.total_seconds().dropna(), bins=50, kde=False)
plt.show()

# print(b1['current height'].mean())

# print(b1['threshold'])

# b1[b1['current height']!=b1['filtered height']]

chk = b1[(~b1['current height'].isna())&(~b1['filtered height'].isna())]

chk_2 = chk[~np.isclose(chk['current height'], chk['filtered height'], atol=1e-9)]

print("Number of Rows : ",chk_2.shape[0])

chk_2.head()

# b1[b1['current height']==b1['filtered height']]

# b1[['current height','filtered height']].dtypes

threshold1 = b1['threshold'].tail(1).values[0] 

threshold2 = b2['threshold'].tail(1).values[0] 

print("T1 :",threshold1,"T2 :",threshold2)

# Data Validation Contingent 
print("More than one values for Time :",cont['Time'].nunique()!=1)
print()
print(cont['face_detection_flag'].value_counts())
# print("total data points:", cont.shape[0])
print()
print("Values for Null current_height and False face_detection should be equal")
print("Null current_height:",cont[cont['current height'].isnull()].shape[0])
print("False face_detection:",cont[cont['face_detection_flag']==False].shape[0])

false_flags = round(100.0*cont[cont['face_detection_flag']==False]['Time'].count()/cont['Time'].count(),2)
duration  = cont[cont['face_detection_flag']==False]['Time'].count()//20
total_duration = cont['Time'].count()/20
print("False Face Detection Flags")
print(f"Percentage of Total Data : {false_flags}%")
print(f"Duration of False Flags : {round(duration//60,2)} Minutes {round(duration%60,2)} Seconds")
print(f"Total Duration of Video : {round(total_duration//60,2)} Minutes {round(total_duration%60,2)} Seconds")

print("More than one values for Time :",NC['Time'].nunique()!=1)
print()
print(NC['face_detection_flag'].value_counts())
# print("total data points:", cont.shape[0])
print()
print("Values for Null current_height and False face_detection should be equal")
print("Null current_height:",NC[NC['current height'].isnull()].shape[0])
print("False face_detection:",NC[NC['face_detection_flag']==False].shape[0])

false_flags = round(100.0*NC[NC['face_detection_flag']==False]['Time'].count()/cont['Time'].count(),2)
duration  = NC[NC['face_detection_flag']==False]['Time'].count()//20
total_duration = NC['Time'].count()/20
print("False Face Detection Flags")
print(f"Percentage of Total Data : {false_flags}%")
print(f"Duration of False Flags : {round(duration//60,2)} Minutes {round(duration%60,2)} Seconds")
print(f"Total Duration of Video : {round(total_duration//60,2)} Minutes {round(total_duration%60,2)} Seconds")
