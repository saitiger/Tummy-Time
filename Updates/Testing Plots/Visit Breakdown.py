import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt

search_terms = ["GA", "UA", "FN", "MT", "IS"]
check = [c for c in df.columns 
         if any(term in c for term in search_terms) 
         and (len(c.split("_")[3])==2 or len(c.split("_")[3])==5)]
df_plot = df[check].mean().reset_index()

# Visit Visualization

def visit_visualization(dataset,visit_num):
  search_terms = ["GA", "UA", "FN", "MT", "IS"]
  check = [c for c in df.columns 
         if any(term in c for term in search_terms) 
         and (len(c.split("_")[3])==2 or len(c.split("_")[3])==5)]
  df_plot = df[check].mean().reset_index()

  split_values = df_plot['index'].str.split('_', expand=True)

  
  df_plot = df_plot.assign(
    Visit = 'M' + split_values[1],
    Score = split_values[3].str[:2],
    Value = df_plot[0],
    With_mon = ~split_values[3].str[2:].str.contains('sub', case=False)
  ).drop(columns=['index', 0])
  ax = sns.barplot(data=df_plot[df_plot["Visit"]==f"V{visit_num}"],x="Score",y="Value",hue="With_mon")
  for container in ax.containers:
      ax.bar_label(container, fmt='%.2f', padding=3, 
                 fontsize=10,  
                 fontweight='bold')  
      
  sns.despine(bottom = True,left = True)
  plt.yticks([])
  plt.ylabel("Frequency")
  plt.show()

# Visit_Channel

def all_visits_channel_plot(df):
  search_terms = ["GA", "UA", "FN", "MT", "IS"]
  check = [c for c in df.columns 
         if any(term in c for term in search_terms) 
         and (len(c.split("_")[3])==2 or len(c.split("_")[3])==5)]
  df_plot = df[check].mean().reset_index()

  split_values = df_plot['index'].str.split('_', expand=True)

    df_processed = df_plot.assign(
        Visit = 'V' + split_values[1],
        Score = split_values[3].str[:2],
        Value = df_plot[0],
        With_mon = ~split_values[3].str[2:].str.contains('sub', case=False) 
    ).drop(columns=['index', 0])[['Visit', 'Score', 'Value', 'With_mon']]
    
    plt.figure(figsize=(10,5))
    ax1 = sns.barplot(data=df_processed[df_processed['With_mon']], 
                     x="Visit", y="Value", hue='Score')
    
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.1f', padding=3, 
                     fontsize=10, fontweight='bold')
    
    sns.despine(bottom=True, left=True)
    plt.yticks([])
    plt.ylabel("Frequency")
    plt.show()
    
    plt.figure(figsize=(10,5))
    plt.title("Without Mon")
    ax2 = sns.barplot(data=df_processed[~df_processed['With_mon']], 
                     x="Visit", y="Value", hue='Score')
    
    for container in ax2.containers:  
        ax2.bar_label(container, fmt='%.1f', padding=3, 
                     fontsize=10, fontweight='bold')
    
    sns.despine(bottom=True, left=True)
    plt.yticks([])
    plt.ylabel("Frequency")
    plt.show()
    
    # Dataframe 
    mon_true = df_processed[df_processed['With_mon'] == True]
    mon_false = df_processed[df_processed['With_mon'] == False]

    # Set indices for alignment
    mon_true = mon_true.set_index(['Visit', 'Score'])
    mon_false = mon_false.set_index(['Visit', 'Score'])

    difference = mon_true['Value'] - mon_false['Value']

    difference_df = difference.reset_index()
    difference_df.columns = ['Visit', 'Score', 'Value_Difference']
    return difference_df
