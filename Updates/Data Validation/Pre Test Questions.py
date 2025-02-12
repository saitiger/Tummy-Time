df_plot = NC[NC['face_detection_flag']==True][['Time','current height','head above the threshold','Toy status']].reset_index(drop=True)
df_plot['Time'] = pd.to_datetime(df_plot['Time'], format='%Y-%m-%d %H:%M:%S.%f')
df_plot['Toy status'] = (df_plot.index // 300 % 2).astype(int)
df_plot['row_number'] = range(1, len(df_plot) + 1)

# Time above 0 
total_duration = df_plot[df_plot['current height'] > 0]['Time'].count()/ 20 
print(f'Time above 0: {int(total_duration // 60)} Minutes {round(total_duration % 60, 2)} Seconds')

# Time based on Toy Status
toy_status_count = df_plot[df_plot['current height'] > 0].groupby('Toy status')['Time'].count().reset_index()

toy_status_count['Duration'] = toy_status_count['Time'].apply(lambda x: {
    'minutes': int((x/20) // 60),
    'seconds': round((x/20) % 60, 2)
})

toy_status_count['Time'] = toy_status_count['Duration'].apply(
    lambda x: f"{x['minutes']} Minutes {x['seconds']} Seconds"
)

toy_status_count = toy_status_count.drop('Duration', axis=1)

print(toy_status_count)

# Count transitions from below/equal to 0 to above 0
count = (
    (df_plot['current height'].shift(1) <= 0) & 
    (df_plot['current height'] > 0)
).sum()

print(f"Number of transitions above 0: {count}")

# Count transitions from below/equal to 0 to above 0, grouped by Toy status
transitions = (
    df_plot[
        (df_plot['current height'].shift(1) <= 0) & 
        (df_plot['current height'] > 0)
    ]
    .groupby('Toy status')
    .size()
    .reset_index(name='Number of transitions')
)

print("\nTransitions above 0 by toy status:")
print(transitions)
