# Behavior Source Plot
def behavior_source_plot(df, source):
    mean_df = df.mean().reset_index().rename(columns={'index': 'Variable', 0: 'Frequency Mean'})
    std_df = df.std().reset_index().rename(columns={'index': 'Variable', 0: 'Frequency Std'})

    df_plot = pd.merge(mean_df, std_df, on='Variable')

    df_plot['Visit'] = 'V' + df_plot['Variable'].str.split('_').str[1]
    df_plot['Source'] = df_plot['Variable'].str.split('_').str[3].str[0]
    df_plot['Behavior'] = df_plot['Variable'].str.split('_').str[3].str[1:]

    df_plot = df_plot.drop('Variable', axis=1)
    df_plot = df_plot[df_plot['Behavior'] == behavior]

    df_plot['Frequency Mean'] = df_plot['Frequency Mean'].round(2)
    df_plot['Frequency Std'] = df_plot['Frequency Std'].round(2)

    df_plot['Mean ± Std'] = df_plot['Frequency Mean'].astype(str) + ' ± ' + df_plot['Frequency Std'].astype(str)

    plt.figure(figsize=(10, 6))
    plt.rcParams['axes.labelsize'] = 16
    sns.lineplot(data=df_plot, x='Visit', y='Frequency Mean', hue='Source', palette='bright', linewidth=3)

    plt.ylim(bottom=0)
    plt.xlabel('Visit')
    plt.ylabel('Mean Frequency')
    sns.despine()
    legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize = 12)
    plt.tick_params(axis='x', labelsize=12)  
    plt.tick_params(axis='y', labelsize=12)  
    legend.set_title('')
    plt.show()

    pivot_df = df_plot.pivot(index='Source', columns='Visit', values='Mean ± Std')
    pivot_df.reset_index(inplace=True)

    return pivot_df

# Interaction Plot
plt.figure(figsize=(10, 6))
sns.pointplot(data=df_int, x='Visit', y='value', hue='Source',
               markers=["o", "s", "D"], dodge=True, errorbar='sd')
plt.title('Interaction Plot: Source vs. Visit',fontsize = 16)
plt.xlabel('Visit')
plt.ylabel('Mean Frequency')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize = 12)
plt.tick_params(axis='x', labelsize=12)  
plt.tick_params(axis='y', labelsize=12)  
sns.despine()
plt.show()