import pandas as pd
import os 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px

# import chardet

# file_path = "50012_right hip_102373_2024-11-27 15-23-42.csv"

# # Read only the first 5000 bytes
# with open(file_path, "rb") as file:
#     raw_data = file.read(5000)

# # Detect the encoding
# result = chardet.detect(raw_data)

# # Extract encoding details
# detected_encoding = result['encoding']
# confidence = result['confidence']

# print(f"Detected encoding: {detected_encoding}")
# print(f"Confidence: {confidence}")

# # If you want to decode the data
# if detected_encoding:
#     try:
#         text = raw_data.decode(detected_encoding)
#         print(text)
#     except UnicodeDecodeError as e:
#         print(f"Decoding error: {e}")

df = df.iloc[:,:5]
columns = ['A','B','C','D','E']
df.columns = columns
# df.head()
plt.figure(figsize = (20,8))
sns.lineplot(data = df, x = 'A', y = 'B')
sns.despine()
# df2.shape[0]

sns.lineplot(data = df2, x = 'A', y = 'B')
sns.despine()

# d['A'] = d['A'].str.replace(r'\.(\d{3})$', r'.\1000', regex=True)  # Convert .000 to .000000
# d['A'] = pd.to_datetime(d['A'], errors='coerce')
# d[d['A'].isna()]

d_plot = d.groupby('A')[['B','C','D','E']].last().reset_index()
d_plot.shape[0]

# d['A'] = pd.to_datetime(d['A'], errors='coerce')  # Use 'coerce' to handle any errors gracefully
# print(df.dtypes)

start_time = pd.Timestamp("2023-05-12 20:21:00")
end_time = pd.Timestamp("2023-05-13 19:50:00")

filtered_df = d_plot[(d_plot['A'] >= start_time) & (d_plot['A'] <= end_time)]

filtered_df.shape[0]/3600

fig = px.scatter(dd_plot, x='A', y='B', color = 'Class',
    hover_data={
    "C": False,
    "D": False,
    "E angle": False,
    "Rotation": False,
    "F": False,
    "G": False,
    "Class": True
})

fig.update_layout(
    width=1500,  
    height=600  
)
fig.show()
