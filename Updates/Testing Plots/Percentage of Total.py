import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

threshold_1 = df['threshold'].iloc[-1]
threshold_2 = df2['threshold'].iloc[-2]

results = pd.DataFrame(columns=['Total Duration', 'Head Above The Threshold Duration'])

total_time_seconds = df['Time'].count()/20
head_above_threshold_seconds = df[df['head above the threshold']==True]['Time'].count()/20
total_time_seconds_2 = df2['Time'].count()/20
head_above_threshold_seconds_2 = df2[df2['head above the threshold']==True]['Time'].count()/20

# Define row dictionaries
row = {'Total Duration': total_time_seconds, 'Head Above The Threshold Duration': head_above_threshold_seconds}
row2 = {'Total Duration': total_time_seconds_2, 'Head Above The Threshold Duration': head_above_threshold_seconds_2}

# Concatenate and assign back to results
results = pd.concat([results, pd.DataFrame([row, row2])], ignore_index=True)

# print(results)

results_nc = pd.DataFrame(columns=['Total Duration', 'Head Above The Threshold Duration'])

total_time_seconds_nc = df3['Time'].count()/20
head_above_threshold_seconds_nc = df3[df3['current height']>=threshold_1]['Time'].count()/20
total_time_seconds_2_nc = df4['Time'].count()/20
head_above_threshold_2_seconds_nc = df4[df4['current height']>=threshold_2]['Time'].count()/20

# Define row dictionaries
row = {'Total Duration': total_time_seconds_nc, 'Head Above The Threshold Duration': head_above_threshold_seconds_nc}
row2 = {'Total Duration': total_time_seconds_2_nc, 'Head Above The Threshold Duration': head_above_threshold_2_seconds_nc}

# Concatenate and assign back to results
results_nc = pd.concat([results_nc, pd.DataFrame([row, row2])], ignore_index=True)

# print(results_nc)

xtick_labels = ["50012", "50013"]

fig, ax = plt.subplots(figsize=(6, 4))

# Total Duration as full bar
bars = ax.bar(range(len(results)), results['Total Duration'], label='Total Duration', color="#00588e")

# Head Above The Threshold inside Total Duration
slashed_bars = ax.bar(
    range(len(results)), 
    results['Head Above The Threshold Duration'], 
    label='Head Above Threshold', 
    color="#78cafc"
)

# Add total duration label above each bar
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width()/2,
        height + 5,  # Position slightly above the bar
        f'{height:.1f}s',
        ha='center',
        va='bottom',
        fontweight='bold'
    )

# Add both duration and percentage label in the bottom bars
for i, bar in enumerate(slashed_bars):
    height = bar.get_height()
    total_height = results['Total Duration'].iloc[i]
    percentage = (height / total_height) * 100
    
    # Only add text if there's enough space
    if height > total_height * 0.15:  # If slashed portion is at least 15% of total
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_y() + height/2,  # Center of the bottom portion
            f'{height:.1f}s ({percentage:.1f}%)',
            ha='center',
            va='center',
            fontweight='bold',
            color='black'
        )

ax.set_xticks(range(len(results))) 
ax.set_xticklabels(xtick_labels)  

ax.set_ylabel('Duration (seconds)')
ax.legend()

plt.tight_layout()
sns.despine()
plt.show()

xtick_labels = ["50012", "50013"]

fig, ax = plt.subplots(figsize=(6, 4))

# Total Duration as full bar
bars = ax.bar(range(len(results_nc)), results_nc['Total Duration'], label='Total Duration', color="#00588e")

# Head Above The Threshold inside Total Duration
slashed_bars = ax.bar(
    range(len(results_nc)), 
    results_nc['Head Above The Threshold Duration'], 
    label='Head Above Threshold', 
    color="#78cafc"
)

# Add total duration label above each bar
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width()/2,
        height + 5,  # Position slightly above the bar
        f'{height:.1f}s',
        ha='center',
        va='bottom',
        fontweight='bold'
    )

# Add duration & percentage labels for slashed bars
for i, bar in enumerate(slashed_bars):
    height = bar.get_height()
    total_height = results_nc['Total Duration'].iloc[i]
    percentage = (height / total_height) * 100
    
    # Only add text if there's enough space
    text_label = f'{height:.1f}s ({percentage:.1f}%)' if height > total_height * 0.15 else f'{percentage:.1f}%'
    
    ax.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_y() + height / 2,  # Center of the bottom portion
        text_label,
        ha='center',
        va='center',
        fontweight='bold',
        color='black',
        fontsize=9 if height > total_height * 0.15 else 8  # Adjust font size for better fit
    )

ax.set_xticks(range(len(results_nc))) 
ax.set_xticklabels(xtick_labels)  

ax.set_ylabel('Duration (seconds)')
ax.legend()

plt.tight_layout()
sns.despine()
plt.show()
