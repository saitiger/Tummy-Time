df_plot = cont[cont['face_detection_flag']==True][['Time','current height','head above the threshold','Toy status']]

fig, ax = plt.subplots(2, 1, figsize=(20, 8), sharex=True)

sns.lineplot(data=df_plot, x='Time', y='current height', ax=ax[0])
ax[0].set_xticks([])  
sns.despine(ax=ax[0])  
sns.lineplot(data=df_plot, x='Time', y='Toy status', ax=ax[1],color = 'r')
plt.show()

df_plot = cont[cont['face_detection_flag'] == True][['Time', 'current height', 'head above the threshold', 'Toy status']]

# Ensure sorting before resampling
df_plot = df_plot.sort_values(by='Time')

# Resampling
height_resampled = df_plot.set_index('Time')[['current height']].resample('500ms').mean()
threshold_resampled = df_plot.set_index('Time')[['head above the threshold']].resample('500ms').last()

# Combine resampled data
df_resampled = pd.concat([height_resampled, threshold_resampled], axis=1).reset_index()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10), sharex=True)

sns.lineplot(
    data=df_resampled,
    x='Time',
    y='current height',
    ax=ax1,
    linewidth=1.5,
    color='blue'
)
ax1.set_ylabel('Current Height')
ax1.grid(False)
ax1.set_xticks([]) 
ax1.set_xlabel('')
sns.despine(ax=ax1, bottom=True)
ax1.axhline(y=0, color='red', linestyle='--', linewidth=1.5)

sns.lineplot(
    data=df_resampled,
    x='Time',
    y='head above the threshold',
    ax=ax2,
    linewidth=1.5,
    color='red',
    drawstyle='steps-post'
)
ax2.set_ylabel('Toy Status')
ax2.set_ylim(-0.1, 1.1)
ax2.grid(False)
ax2.set_yticks([0, 1])
ax2.set_yticklabels(['False', 'True'])
sns.despine(ax=ax2)

plt.tight_layout()
plt.show()

total_duration = cont['head above the threshold'].value_counts().values[0]
print(f"Total Duration : {total_duration//1200} Minutes {(total_duration//20)%60} Seconds")

df_plot = NC[NC['face_detection_flag']==True][['Time','current height','head above the threshold','Toy status']].reset_index(drop=True)

df_plot['Time'] = pd.to_datetime(df_plot['Time'], format='%Y-%m-%d %H:%M:%S.%f')

df_plot['Toy status'] = (df_plot.index // 300 % 2).astype(int)

df_plot['row_number'] = range(1, len(df_plot) + 1)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10), sharex=True)

sns.lineplot(
    data=df_plot,
    x='row_number',
    y='current height',
    ax=ax1,
    linewidth=1.5,
    color='blue'
)
ax1.set_ylabel('Current Height')
ax1.grid(False)  
ax1.set_xticks([])  
ax1.set_xlabel('')
sns.despine(ax=ax1, bottom=True)
ax1.axhline(y=0, color='red', linestyle='--', linewidth=1.5)

sns.lineplot(
    data=df_plot,
    x='row_number',
    y='Toy status',
    ax=ax2,
    linewidth=1.5,
    color='red',
    drawstyle='steps-post'  
)
ax2.set_ylabel('Toy Status')
ax2.set_ylim(-0.1, 1.1)
ax2.grid(False)  
ax2.set_yticks([0, 1])  
ax2.set_yticklabels(['False', 'True'])  
sns.despine(ax=ax2)

plt.tight_layout()
plt.show()

def contigent_head_above_threshold(df, chunk_time=120):
    """
    Chunk time is the duration of each chunk we want to visualize by default it is 2 Minutes (120 Seconds)
    Count denotes the number of times when head goes from below the threshold to above the threshold
    """

    df_plot = df[['head above the threshold', 'Toy status', 'Time']]
    chunk_size = chunk_time * 20
    chunk_results = []

    total_video_length = df_plot['Time'].count() * 50 / 1000
    head_above_threshold_total_time = df_plot[df_plot['head above the threshold'] == True]['Time'].count() * 50 / 1000
    print("Total Video Length : ", total_video_length // 60, 'minutes', round(total_video_length % 60, 2), 'seconds')
    print("Head Above Threshold Time : ", head_above_threshold_total_time // 60, 'minutes', round(head_above_threshold_total_time % 60, 2), 'seconds')

    for i, chunk_start in enumerate(range(0, len(df), chunk_size)):
        chunk = df.iloc[chunk_start:chunk_start + chunk_size]

        count = ((chunk['head above the threshold'].shift(1) == False) &
                 (chunk['head above the threshold'] == True)).sum()
        chunk_time = chunk['Time'].count() * 50 / 1000
        head_above_threshold_total_time = (
            chunk[chunk['head above the threshold'] == True]['Time'].count() * 50 / 1000
        )

        chunk_results.append({
            'Chunk': f'Chunk {i + 1}',
            'chunk_time': chunk_time,
            'head_above_threshold_total_time': head_above_threshold_total_time,
            'count': count
        })

    results_df = pd.DataFrame(chunk_results)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=results_df.melt(id_vars='Chunk'), x='Chunk', y='value', hue='variable', palette='bright')

    for p in ax.patches:
        if p.get_height() != 0:
            ax.annotate(f'{int(p.get_height())}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        xytext=(0, 10), textcoords='offset points')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=['Chunk Time', 'Head Above Threshold Total Time', 'Count'], title='Metrics')

    plt.ylabel('Time (seconds) / Count')
    plt.xlabel('Chunk')
    sns.despine()
    plt.show()


def contigent_head_above_threshold_2(df, chunk_time=120):
    """
    Count denotes the number of times when head goes from below the threshold to above the threshold
    This modifies the behaviour of the earlier function to only display the size of the chunk when it is not equal to the
    desired size as provided by the user {By default 120 seconds}
    """

    df_plot = df[['head above the threshold', 'Toy status', 'Time']]
    chunk_size = chunk_time * 20
    chunk_results = []

    total_video_length = df_plot['Time'].count() * 50 / 1000
    head_above_threshold_total_time = df_plot[df_plot['head above the threshold'] == True]['Time'].count() * 50 / 1000
    print("Total Video Length : ", total_video_length // 60, 'minutes', round(total_video_length % 60, 2), 'seconds')
    print("Head Above Threshold Time : ", head_above_threshold_total_time // 60, 'minutes', round(head_above_threshold_total_time % 60, 2), 'seconds')

    for i, chunk_start in enumerate(range(0, len(df), chunk_size)):
        chunk = df.iloc[chunk_start:chunk_start + chunk_size]

        count = ((chunk['head above the threshold'].shift(1) == False) &
                 (chunk['head above the threshold'] == True)).sum() # Need to rename this column
        chunk_time = chunk['Time'].count() * 50 / 1000
        head_above_threshold_total_time = (
            chunk[chunk['head above the threshold'] == True]['Time'].count() * 50 / 1000
        )

        # Check if the chunk size is less than the expected chunk size
        if len(chunk) < chunk_size:
            chunk_results.append({
                'Chunk': f'Chunk {i + 1}',
                'chunk_time': chunk_time,
                'head_above_threshold_total_time': head_above_threshold_total_time,
                'count': count
            })
        else:
            chunk_results.append({
                'Chunk': f'Chunk {i + 1}',
                'chunk_time': None,  # Set chunk_time to None for full-sized chunks
                'head_above_threshold_total_time': head_above_threshold_total_time,
                'count': count
            })

    results_df = pd.DataFrame(chunk_results)

    # Ensure the 'Chunk' column is treated as a categorical variable with the correct order
    results_df['Chunk'] = pd.Categorical(results_df['Chunk'],
                                         categories=sorted(results_df['Chunk'].unique(), key=lambda x: int(x.split()[1])),
                                         ordered=True)

    # Melt the DataFrame for plotting
    melted_df = results_df.melt(id_vars='Chunk', value_vars=['chunk_time', 'head_above_threshold_total_time', 'count'])

    # Filter out rows where chunk_time is None
    melted_df = melted_df[~((melted_df['variable'] == 'chunk_time') & (melted_df['value'].isna()))]

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=melted_df, x='Chunk', y='value', hue='variable', palette='bright', order=results_df['Chunk'].cat.categories)

    for p in ax.patches:
        if p.get_height() != 0:
            ax.annotate(f'{int(p.get_height())}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        xytext=(0, 10), textcoords='offset points')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=['Chunk Time', 'Head Above Threshold Total Time', 'Count'], title='Metrics')

    plt.ylabel('Time (seconds) / Count')
    plt.xlabel('Chunk')
    sns.despine()
    plt.show()
