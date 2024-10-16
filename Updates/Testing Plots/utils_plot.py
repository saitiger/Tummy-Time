import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

def data_dict_fill(data_dictionary):
    data_dictionary['Variable'].ffill(inplace = True)
    return data_dictionary[['Variable','Value','Label']]

# Usage 
# my_dict = data_dict_fill(data_dict2)

def behaviour_plot_for_toy(df, toy_name, group, data_dictionary=None):
    toy_type_title = toy_name  # Used for name of Plot 
    
    toy_mapping = {
        'Gumball': 'Gn', 'GUMBALL': 'Gn', 'GumBall': 'Gn',
        'Popup': 'Pn', 'POPUP': 'Pn', 'Pop-up': 'Pn', 'Pop Up': 'Pn', 'Pop up': 'Pn',
        'Cups': 'Cn', 'cups': 'Cn', 'CUPS': 'Cn'
    }
    
    toy_metric_mapping = {
        'Gumball': 'G', 'GUMBALL': 'G', 'GumBall': 'G',
        'Popup': 'P', 'POPUP': 'P', 'Pop-up': 'P', 'Pop Up': 'P', 'Pop up': 'P',
        'Cups': 'C', 'cups': 'C', 'CUPS': 'C'
    }
    
    behaviour_mapping = {
    'ES':'Simple Explore','EC':'Complex Explore',
    'LK':'Look','SL':'Solution','FN':'Function'
    }
    
    if toy_name not in toy_mapping:
        raise ValueError("toy_type must be one of 'Popup', 'Gumball' or 'Cups'")
        
    if data_dictionary is not None:
        label_mapping = dict(zip(
            data_dictionary[data_dictionary['Variable'] == 'C_1_TRT']['Value'],
            data_dictionary[data_dictionary['Variable'] == 'C_1_TRT']['Label']
        ))
        group_type = label_mapping.get(group, group)
    else:
        group_type = group
    
    metric_initial = toy_metric_mapping[toy_name]
        
    df_plot = df[['C_1_TRT',
                 f'C_2_APS_{metric_initial}LK',f'C_2_APS_{metric_initial}ES',f'C_2_APS_{metric_initial}EC',f'C_2_APS_{metric_initial}FN',f'C_2_APS_{metric_initial}SL',
                 f'C_3_APS_{metric_initial}LK',f'C_3_APS_{metric_initial}ES',f'C_3_APS_{metric_initial}EC',f'C_3_APS_{metric_initial}FN',f'C_3_APS_{metric_initial}SL',
                 f'C_4_APS_{metric_initial}LK',f'C_4_APS_{metric_initial}ES',f'C_4_APS_{metric_initial}EC',f'C_4_APS_{metric_initial}FN',f'C_4_APS_{metric_initial}SL'
                 ]]
    df_plot = df_plot.dropna(how='any')
    df_plot = df_plot.loc[df_plot['C_1_TRT']==group].drop(columns='C_1_TRT', axis=1)
    df_plot = df_plot.mean().reset_index()
    df_plot.columns = ['Visit', 'Score']
    df_plot['Visit'] = df_plot['Visit'].str.replace('C_2_APS','V2')
    df_plot['Visit'] = df_plot['Visit'].str.replace('C_3_APS','V3')
    df_plot['Visit'] = df_plot['Visit'].str.replace('C_4_APS','V4')
    df_plot['Score Name'] = df_plot['Visit'].str.split('_').str[1].str[1:3]
    df_plot['Score Name']
    df_plot['Visit'] = df_plot['Visit'].str.split('_').str[0]
    df_plot['Score Name'] = df_plot['Score Name'].map(behaviour_mapping)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Score Name', y='Score', hue='Visit', data=df_plot, palette='bright')
    plt.title(f'Behaviour Scores for Toy : {toy_type_title} & Group : {group_type}')
    plt.xlabel("")
    plt.ylabel('Frequency Mean for Group')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend(title='Visit')
    sns.despine()
    plt.show()
    return df_plot
# Usage
# behaviour_plot_for_toy(df,'Popup',2,my_dict)

def behaviour_count_greater_than_zero(df, toy_name):
    toy_mapping = {
        'Gumball': 'G', 'GUMBALL': 'G', 'GumBall': 'G',
        'Popup': 'P', 'POPUP': 'P', 'Pop-up': 'P', 'Pop Up': 'P', 'Pop up': 'P',
        'Cups': 'C', 'cups': 'C', 'CUPS': 'C'
    }
    
    if toy_name not in toy_mapping:
        raise ValueError("toy_name must be one of 'Popup', 'Gumball' or 'Cups'")
    
    toy_initial = toy_mapping[toy_name]
    
    metrics = ['n', 'LK', 'ES', 'EC', 'FN', 'SL']
    columns = ['C_1_TRT'] + [f'C_{i}_APS_{toy_initial}{m}' for i in range(2, 5) for m in metrics]
    
    df_cnt = df[columns].groupby('C_1_TRT').agg(lambda x: (x > 0).sum()).T.reset_index()
    df_cnt.columns = ['Metric', 'Control', 'SPEEDI-EARLY', 'SPEEDI-LATE']
    
    df_cnt['Visit'] = df_cnt['Metric'].str.split('_').str[1]
    df_cnt['Metric Type'] = df_cnt['Metric'].str.split('_').str[3].str[1:]
    
    metric_type_mapping = {'n': 'TWS', 'LK': 'Look', 'ES': 'Simple Explore', 'EC': 'Complex Explore', 'FN': 'Function', 'SL': 'Solution'}
    df_cnt['Metric Type'] = df_cnt['Metric Type'].map(metric_type_mapping)
    
    result = df_cnt[['Metric Type', 'Visit', 'Control', 'SPEEDI-EARLY', 'SPEEDI-LATE']]
    
    print("Number of Rows with value greater than zero")
    return result

# Usage 
# behaviour_count_greater_than_zero(df2, 'Gumball')

def behaviour_plot_combined(df,behaviour,group,data_dictionary=None):
    
    if data_dictionary is not None:
        label_mapping = dict(zip(
            data_dictionary[data_dictionary['Variable'] == 'C_1_TRT']['Value'],
            data_dictionary[data_dictionary['Variable'] == 'C_1_TRT']['Label']
        ))
        group_type = label_mapping.get(group, group)
    else:
        group_type = group
    
    behaviour_mapping = {
    'Simple Explore':'ES','simple explore':'ES', 'SIMPLE EXPLORE':'ES',
    'Complex Explore':'EC', 'complex explore':'EC', 'COMPLEX EXPLORE':'EC',
    'Look':'LK','LOOK':'LK','Looks':'LK','LOOKS':'LK','look':'LK',
    'Solution':'SL','solution':'SL','SOLUTION':'SL',
    'Function':'FN','function':'FN'
    }
    
    behaviour_title = behaviour.capitalize()
    
    if behaviour not in behaviour_mapping:
        raise ValueError("behaviour must be one of 'Simple Explore', 'Complex Explore','Look','Solution' or 'Function'")
    
    behaviour_shorthand = behaviour_mapping[behaviour]
    
    desired_cols = ['C_1_TRT',
         f'C_2_APS_G{behaviour_shorthand}',f'C_3_APS_G{behaviour_shorthand}',f'C_4_APS_G{behaviour_shorthand}',
         f'C_2_APS_C{behaviour_shorthand}',f'C_3_APS_C{behaviour_shorthand}',f'C_4_APS_C{behaviour_shorthand}',
         f'C_2_APS_P{behaviour_shorthand}',f'C_3_APS_P{behaviour_shorthand}',f'C_4_APS_P{behaviour_shorthand}'
        ]

    df_plot = df.loc[df['C_1_TRT']==group,desired_cols]

    df_plot = df_plot.dropna(how = 'any').drop(columns = 'C_1_TRT')
    df_plot = df_plot.mean().reset_index().rename(columns = {'index':'Visit',0:'Score'})
    df_plot['Toy'] = df_plot['Visit'].str.split('_').str[3].str[0]
    df_plot['Visit'] = df_plot['Visit'].str.split('_').str[1]
    df_plot['Toy'].replace({'G':'Gumball','C':'Cups','P':'Popup'},inplace = True)
    plt.figure(figsize=(10, 6))
    # custom_palette = ["#FF5733", "#33FF57", "#3357FF"]  
    sns.lineplot(data = df_plot, x = 'Visit', y = 'Score',hue = 'Toy',palette = 'bright' ,marker = 'o', linewidth = 3)
    sns.despine()
    plt.ylabel("Count")
    plt.title(f"{behaviour_title} Frequency for {group_type}")
    plt.show()
    return df_plot

# Usage
# behaviour_plot_combined(df2,'Complex Explore',3,my_dict)

def toy_score_comparison(df, toy_name, group, score_cognitive, score_gross_motor, data_dictionary=None):
    toy_mapping = {
        'Gumball': 'Gn', 'GUMBALL': 'Gn', 'GumBall': 'Gn',
        'Popup': 'Pn', 'POPUP': 'Pn', 'Pop-up': 'Pn', 'Pop Up': 'Pn', 'Pop up': 'Pn',
        'Cups': 'Cn', 'cups': 'Cn', 'CUPS': 'Cn'
    }
    
    if score_cognitive not in ['Cognitive Scaled', 'Cognitive Raw', 'Cognitive Composite']:
        raise ValueError('score_cognitive should be one of Cognitive Scaled, Cognitive Raw or Cognitive Composite')
    
    if score_gross_motor not in ['Gross Motor Raw', 'Gross Motor Scaled']:
        raise ValueError('score_gross_motor should be one of Gross Motor Raw or Gross Motor Scaled')
        
    if toy_name not in toy_mapping:
        raise ValueError("toy_name must be one of 'Popup', 'Gumball' or 'Cups'")
        
    if data_dictionary is not None:
        label_mapping = dict(zip(
            data_dictionary[data_dictionary['Variable'] == 'C_1_TRT']['Value'],
            data_dictionary[data_dictionary['Variable'] == 'C_1_TRT']['Label']
        ))
        group_type = label_mapping.get(group, group)
    else:
        group_type = group
        
    score_mapping = {
        'Cognitive Scaled': 'CSs', 'Cognitive Raw': 'CSu', 'Cognitive Composite': 'CSc',
        'Gross Motor Raw': 'GMu', 'Gross Motor Scaled': 'GMs'
    }
    
    metric1_shorthand = score_mapping[score_cognitive]
    metric2_shorthand = score_mapping[score_gross_motor]
    toy_metric = toy_mapping[toy_name]
        
    cols = ['C_1_TRT',
        f'C_2_APS_{toy_metric}', f'C_3_APS_{toy_metric}', f'C_4_APS_{toy_metric}',
        f'C_2_BS3_{metric1_shorthand}', f'C_3_BS3_{metric1_shorthand}', f'C_4_BS3_{metric1_shorthand}',
        f'C_2_BS3_{metric2_shorthand}', f'C_3_BS3_{metric2_shorthand}', f'C_4_BS3_{metric2_shorthand}'
    ]
    
    df_plot = df[cols]
    df_plot = df_plot.dropna(how='any')
    df_plot = df_plot[df_plot['C_1_TRT'] == group].drop(columns='C_1_TRT')
    df_plot = df_plot.mean().reset_index().rename(columns={'index': 'Visit', 0: 'Score'})
    df_plot['Score Type'] = df_plot['Visit'].str.split('_').str[3]
    df_plot['Visit'] = df_plot['Visit'].str.split('_').str[1]
    
    df_plot['Score Type'] = df_plot['Score Type'].replace({
        toy_metric: 'TWS',
        metric1_shorthand: score_cognitive,
        metric2_shorthand: score_gross_motor
    })

    plt.figure(figsize=(10, 6))
#     sns.lineplot(data=df_plot, x='Visit', y='Score', hue='Score Type', linewidth=3, marker='o')
    # custom_palette = ["#FF5733", "#33FF57", "#3357FF"]  
    sns.lineplot(data = df_plot, x = 'Visit', y = 'Score', hue = 'Score Type', linewidth = 3, palette = 'bright')
    plt.title(f'Scores for {toy_name}, Group : {group_type}')
    sns.despine()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    plt.show()
    return df_plot

# Usage 
# toy_score_comparison(df2,'Gumball',2,'Cognitive Raw','Gross Motor Raw',my_dict)

def demographic_plots(df, demographic_type, score_type, data_dictionary=None):
    """
    Creates plots of demographic data over different visits.
    
    Parameters:
    - df: The input dataframe.
    - demographic_type (str): Type of demographic ('Sex', 'Race', or 'Ethnicity')
    - score_type (str): Type of score ('RPM' or 'TWS').
    - data_dictionary (pandas.DataFrame): Data dictionary *Optional for label mapping
    """
    
    demographic_mapping = {
        'sex': 'SEX', 'Sex': 'SEX', 'SEX': 'SEX',
        'ethnicity': 'ETH', 'Ethnicity': 'ETH', 'eth': 'ETH', 'ETH': 'ETH',
        'race': 'Race', 'Race': 'Race', 'RACE': 'Race'
    }
    
    if demographic_type not in demographic_mapping:
        raise ValueError("demographic_type must be one of 'Sex', 'Race' or 'Ethnicity'")
        
    demographic_type = demographic_mapping[demographic_type]
    
    if score_type not in ['RPM', 'TWS']:
        raise ValueError("score_type must be RPM or TWS")
            
    plot_data = df.groupby(f'C_1_{demographic_type}')[[f'C_2_APS_{score_type}', f'C_3_APS_{score_type}', f'C_4_APS_{score_type}']].mean().reset_index().melt(id_vars=f'C_1_{demographic_type}')
    plot_data['Visit'] = plot_data['variable'].replace({
        f'C_2_APS_{score_type}': 'V2',
        f'C_3_APS_{score_type}': 'V3',
        f'C_4_APS_{score_type}': 'V4'
    })
    plot_data.rename(columns={'value': score_type}, inplace=True)
    plot_data.drop(columns='variable', inplace=True)
    
    if data_dictionary is not None:
        label_mapping = dict(zip(
            data_dictionary[data_dictionary['Variable'] == f'C_1_{demographic_type}']['Value'],
            data_dictionary[data_dictionary['Variable'] == f'C_1_{demographic_type}']['Label']
        ))
        plot_data[f'{demographic_type}'] = plot_data[f'C_1_{demographic_type}'].map(label_mapping)
    else:
        plot_data[f'{demographic_type}'] = plot_data[f'C_1_{demographic_type}']
    
    # Filter out groups with all zero values
    plot_data_filtered = plot_data.groupby(f'{demographic_type}').filter(lambda x: x[score_type].sum() > 0)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=plot_data_filtered, x='Visit', y=score_type, hue=f'{demographic_type}', palette='bright',marker = 
                 'o',linewidth = 3)
    sns.despine()
        
    plt.title(f'Demographic : {demographic_type} , Score : {score_type}')
    plt.xlabel('Visit')
    plt.ylabel(f'{score_type} Score')
    
    plt.show()
    return plot_data_filtered

# Usage 
# demographic_plots(df2, 'Sex', 'TWS',my_dict)

def individual_plots(df,score_type):
    """
    Creates individual plots for each subject in the study.

    # Parameters:
    - df (pandas.DataFrame): The input dataframe.
    - score_type (str): Type of score ('RPM' or 'TWS').

    """
    if score_type not in ['RPM','TWS']:
        raise ValueError("score_type must be RPM or TWS")
    
    for i in range(len(df)):
        CID = df.loc[i, 'CID']
        res = df.loc[i:i, [f'C_2_APS_{score_type}', f'C_3_APS_{score_type}', f'C_4_APS_{score_type}']].rename(
            columns={f"C_2_APS_{score_type}": "V2", f"C_3_APS_{score_type}": "V3", f"C_4_APS_{score_type}": "V4"}
        ).melt()
        res.rename(columns={"variable": "Visit", "value": score_type}, inplace=True)
        
        plt.figure()
        if res[score_type].isna().all():
            plt.title(f'Empty plot for CID {CID} (all values are NaN)')
            plt.xlabel('Visit')
            plt.ylabel(score_type)
        else:
            sns.lineplot(data = res, x = 'Visit', y = score_type, linewidth = 3)
            plt.title(f'Line plot for CID {CID}')
                    
        plt.savefig(f'plot_TWS_CID_{CID}.png')
        plt.close()
        
# Usage
# individual_plots(df2,'TWS')

def Bayley_vs_APSP_plot(df, bs3_type='CSs', trt_value=1, data_dictionary=None):
    
    """
    Creates a plot comparing Bayley scores with APSP scores.
    # Parameters:
    - df (pandas.DataFrame): The input dataframe.
    - bs3_type (str): Type of Bayley score ('CSs', 'CSu', 'CSc', 'GMu', or 'GMs'). Default value is CSs
    - trt_value (int): Treatment value (1, 2, or 3). Default value is 1 
    - data_dictionary (pandas.DataFrame): Data dictionary *Optional for label mapping. 
    
    """
    
    # Validate input parameters
    if bs3_type not in ['CSs', 'CSu', 'CSc', 'GMu', 'GMs']:
        raise ValueError("bs3_type must be one of 'CSs', 'CSu', 'CSc', 'GMu', or 'GMs'")
    if trt_value not in [1, 2, 3]:
        raise ValueError("trt_value must be 1, 2, or 3")
    
    # Extract last three letters for bs3_type
    metric_type = 'Gross Motor' if bs3_type[:2] == 'GM' else 'Cognitive'
    
    if bs3_type[-1] == 'c':
        metric_type += ' Composite'
    elif bs3_type[-1] == 's':
        metric_type += ' Scaled'
    elif bs3_type[-1] == 'u':
        metric_type += ' Raw'
    
    desired_cols = [
        'C_1_TRT',
        'C_2_APS_RPM', 'C_3_APS_RPM', 'C_4_APS_RPM',
        f'C_2_BS3_{bs3_type}', f'C_3_BS3_{bs3_type}', f'C_4_BS3_{bs3_type}'
    ]
    
    df_compare1 = df[desired_cols]
    
    df_compare = df_compare1.dropna(subset=[
        'C_2_APS_RPM', f'C_2_BS3_{bs3_type}',
        f'C_3_BS3_{bs3_type}', f'C_4_BS3_{bs3_type}'
    ], how='any')
    
    # Create label mapping if data dictionary is provided
    if data_dictionary is not None:
        label_mapping = dict(zip(
            data_dictionary[data_dictionary['Variable'] == 'C_1_TRT']['Value'],
            data_dictionary[data_dictionary['Variable'] == 'C_1_TRT']['Label']
        ))
    
    group_type = label_mapping.get(trt_value)
    
    # Filter data based on C_1_TRT value
    plot_data = df_compare[df_compare['C_1_TRT'] == trt_value].drop(columns='C_1_TRT')
    
    mean_data = plot_data.mean()
    plot_df = mean_data.reset_index()
    plot_df.columns = ['Metric', 'Value']
    
    # Extract visit number and metric name
    plot_df['Visit'] = plot_df['Metric'].str.extract('C_(\d+)_')
    plot_df['Metric'] = plot_df['Metric'].str.extract('C_\d+_(.+)')
    plot_df['Visit'] = 'V' + plot_df['Visit']
    
    # Replace APS_RPM and BS3 labels
    plot_df['Metric'] = plot_df['Metric'].replace({
        f'BS3_{bs3_type}': metric_type
    })
    
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=plot_df, x='Visit', y='Value', hue='Metric', marker='o', palette='bright',linewidth = 3)
    
    plt.title(f'Comparison of APSP and Bayley {metric_type} Scores for {group_type}')
    plt.xlabel('Visit')
    plt.ylabel('Value')
    plt.legend(title='Metric')
    plt.xticks(plot_df['Visit'].unique())
    sns.despine()
    plt.xlabel("")
    plt.ylabel("")
    plt.show()
    
    return mean_data

# Usage 
# Bayley_vs_APSP_plot(df2,'GMs',3,my_dict)

def create_toy_plot(df, toy_type = 'Gumball', data_dictionary = None ):
    
    """
    Creates a plot of toy scores over different visits and treatment groups.

    ### Parameters:
    - df (pandas.DataFrame): The input dataframe.
    - toy_type (str): Type of toy ('Popup', 'Gumball', or 'Cups').
    - data_dictionary (pandas.DataFrame): Data dictionary *Optional for label mapping.

    """
    
    toy_type_title = toy_type
    
    toy_mapping = {
        'Gumball': 'Gn', 'GUMBALL': 'Gn', 'GumBall': 'Gn',
        'Popup': 'Pn', 'POPUP': 'Pn', 'Pop-up': 'Pn', 'Pop Up': 'Pn', 'Pop up':'Pn',
        'Cups': 'Cn', 'cups': 'Cn', 'CUPS': 'Cn'
    }
    
    if toy_type not in toy_mapping:
        raise ValueError("toy_type must be one of 'Popup', 'Gumball' or 'Cups'")
    
    toy_type = toy_mapping[toy_type]
     
    cols_set = ['C_1_TRT', f'C_2_APS_{toy_type}', f'C_3_APS_{toy_type}', f'C_4_APS_{toy_type}']
    
    toy_data = df[cols_set].dropna(how='any')
    
    mean_data = (toy_data.groupby('C_1_TRT')[[f'C_2_APS_{toy_type}',f'C_3_APS_{toy_type}',f'C_4_APS_{toy_type}']].mean()
                 .reset_index().rename(columns={f'C_2_APS_{toy_type}':'V2',
                f'C_3_APS_{toy_type}':'V3',f'C_4_APS_{toy_type}':'V4'})
                )
    
    # Add full treatment names to mean_data
    label_mapping = dict(zip(
        data_dictionary[data_dictionary['Variable'] == 'C_1_TRT']['Value'],
        data_dictionary[data_dictionary['Variable'] == 'C_1_TRT']['Label']
    ))

    mean_data['Treatment Group'] = mean_data['C_1_TRT'].map(label_mapping)
    
    mean_data.drop(columns = 'C_1_TRT',inplace = True)
    
    toy_data_plot = (
        mean_data
        .melt(id_vars= 'Treatment Group')
        .rename(columns={'variable': 'Visit', 'value': 'Score'})
        .replace({f'C_2_APS_{toy_type}': 'V2', f'C_3_APS_{toy_type}': 'V3', f'C_4_APS_{toy_type}': 'V4'})
    )
    
    # custom_palette = ["#FF5733", "#33FF57", "#3357FF"]
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=toy_data_plot, x='Visit', y='Score', hue='Treatment Group', marker = 'o',linewidth = 3,palette='bright')
    sns.despine()
    plt.title(f'{toy_type_title} Scores')
    plt.xlabel('Visit')
    plt.ylabel('Score')
    plt.legend(title='Treatment')
    plt.show()
    
    return mean_data[['Treatment Group','V2','V3','V4']]

# Usage 
# create_toy_plot(df2,'Popup',my_dict)

def create_toy_plot_grp_wise(df,grp,data_dictionary=None):
    """
    Creates a plot of toy scores for a specific treatment group.

    # Parameters:
    - df (pandas.DataFrame): The input dataframe.
    - grp (int): Treatment group (1, 2, or 3).
    - data_dictionary (pandas.DataFrame): Data dictionary *Optional for label mapping.

    """
    
    if grp not in [1, 2, 3]:
        raise ValueError("grp must be 1, 2, or 3")

    cols_set =['C_1_TRT',
               'C_2_APS_Gn', 'C_3_APS_Gn', 'C_4_APS_Gn',
              'C_2_APS_Cn', 'C_3_APS_Cn', 'C_4_APS_Cn',
               'C_2_APS_Pn', 'C_3_APS_Pn', 'C_4_APS_Pn'
              ]
    
    toy_data = df.loc[df['C_1_TRT']==grp,cols_set].drop(columns = 'C_1_TRT')
    plot_df_toy = toy_data.mean().reset_index().rename(columns = {'index':'Toy Type',0:'Score'})
    
    if data_dictionary is not None:
        label_mapping = dict(zip(
            data_dictionary[data_dictionary['Variable'] == 'C_1_TRT']['Value'],
            data_dictionary[data_dictionary['Variable'] == 'C_1_TRT']['Label']
        ))
    
    group_type = label_mapping.get(grp)
    
    plot_df_toy['Visit'] = plot_df_toy['Toy Type'].str.extract('C_(\d+)_')
    plot_df_toy['Toy Type'] = plot_df_toy['Toy Type'].str.extract('APS_(\w+)')
    plot_df_toy['Visit'] = 'V' + plot_df_toy['Visit']

    plot_df_toy['Toy Type'].replace({'Gn':'Gumball','Cn':'Cups','Pn':'Popup'},inplace = True)
    # custom_palette = ["#FF5733", "#33FF57", "#3357FF"]
    plt.figure(figsize=(10, 6))
    plt.title(f"APSP Scores for {group_type}")
    sns.lineplot(data = plot_df_toy , x = 'Visit', y = 'Score', hue = 'Toy Type', linewidth = 3,marker = 'o', palette = 'bright')
    sns.despine()
    plt.show()
    
    return plot_df_toy

# Usage 
# create_toy_plot_grp_wise(df,2,my_dict)

def plot_aps_scores(df, score_type='TWS', plot_type='boxplot', data_dictionary=None):
    """
    Create a box or strip plot of APS scores (TWS or RPM) by treatment group to visualize distribution 
    and outliers.
    
    # Parameters:
    - df (pandas.DataFrame): The input dataframe
    - score_type (str): Score type ('TWS' or 'RPM'). Default is 'TWS'
    - plot_type (str): 'boxplot' or 'stripplot'. Deafault is 'boxplot'
    - data_dictionary (pandas.DataFrame): Data Dictionary * Optional for label mapping
    
    """
    
    # Validate inputs
    if score_type not in ['TWS', 'RPM']:
        raise ValueError("score_type must be either 'TWS' or 'RPM'")
    if plot_type not in ['boxplot', 'stripplot']:
        raise ValueError("plot_type must be either 'boxplot' or 'stripplot'")
    
    score_cols = [f'C_2_APS_{score_type}', f'C_3_APS_{score_type}', f'C_4_APS_{score_type}']
    
    df_clean = df.dropna(subset=score_cols, how='all')
    
    df_melted = df_clean.melt(id_vars=['C_1_TRT'], value_vars=score_cols,
                              var_name='Visit', value_name='APS_Value')
    
    # Create label mapping if data dictionary is provided
    if data_dictionary is not None:
        label_mapping = dict(zip(
            data_dictionary[data_dictionary['Variable'] == 'C_1_TRT']['Value'],
            data_dictionary[data_dictionary['Variable'] == 'C_1_TRT']['Label']
        ))
        df_melted['C_1_TRT'] = df_melted['C_1_TRT'].map(label_mapping)
    
    df_melted['Visit'].replace({'C_2_APS_TWS':'V2','C_3_APS_TWS':'V3','C_4_APS_TWS':'V4'},inplace = True)
    
    plt.figure(figsize=(12, 6))
    if plot_type == 'boxplot':
        sns.boxplot(x='C_1_TRT', y='APS_Value', hue='Visit', data=df_melted)
    else: 
        sns.stripplot(x='C_1_TRT', y='APS_Value', hue='Visit', data=df_melted, jitter=True, dodge=True)
    plt.xlabel("")
    plt.ylabel(f"{score_type}")
    sns.despine()
    plt.show()

# Usage
# plot_aps_scores(df2,'RPM','boxplot',my_dict)

def plot_bins(df, visit_num):
    """
    Creates a plot of binned APS TWS scores for a specific visit.

    # Parameters:
    - df (pandas.DataFrame): The input dataframe.
    - visit_num (str): Visit number ('Visit 2', 'Visit 3', or 'Visit 4').

    """
    
    visit_map = {
        'Visit 2': 'C_2_APS_TWS',
        'Visit 3': 'C_3_APS_TWS',
        'Visit 4': 'C_4_APS_TWS'
    }
    
    if visit_num not in visit_map:
        raise ValueError("Invalid visit number. Choose from: 'Visit 2', 'Visit 3', 'Visit 4'.")

    df_clean = df.dropna(subset=[visit_map[visit_num]], how='any')

    df_melted = df_clean.melt(id_vars=['C_1_TRT'], 
                               value_vars=[visit_map[visit_num]],
                               var_name='APS_TWS_Type', 
                               value_name='APS_TWS_Value')

    df_melted_filtered = df_melted[df_melted['APS_TWS_Value'] != 0]

    value_counts = (df_melted_filtered.groupby(['C_1_TRT', 'APS_TWS_Type', 'APS_TWS_Value'])
                    .size().reset_index(name='Count'))

    # Create bins
    bins = [0, 100, 150, 200, 250, 300, 350, 400, 450, 500, float('inf')]
    labels = ['0-100', '100-150', '150-200', '200-250', '250-300', '300-350', '350-400', '400-450', '450-500', '500+']
    
    value_counts['APS_TWS_Bin'] = pd.cut(value_counts['APS_TWS_Value'], bins=bins, labels=labels, right=False)

    filtered_result = value_counts[value_counts['APS_TWS_Type'] == visit_map[visit_num]][['C_1_TRT', 'APS_TWS_Value', 'Count', 'APS_TWS_Bin']]

    bin_counts = filtered_result['APS_TWS_Bin'].value_counts()

    # Filter bins with counts > 0
    non_zero_bins = bin_counts[bin_counts > 0].index
    sorted_bins = sorted(non_zero_bins, key=lambda x: bins[labels.index(x)])

    plt.figure(figsize=(12, 6))
    sns.countplot(data=filtered_result[filtered_result['APS_TWS_Bin'].isin(non_zero_bins)], 
                    x='APS_TWS_Bin', order=sorted_bins, palette = 'bright')

    plt.title(f'Bins for APS_TWS : {visit_num}')
    plt.xticks(rotation=45) 
    plt.xlabel("")
    plt.ylabel("")
    plt.show()

# Usage
# plot_bins(df, 'Visit 4')

def plot_for_grp_behaviour_toy(df, toy_name, behaviour, data_dictionary=None):
    toy_initial = toy_name[0].upper()
    behaviour_mapping = {
    'Simple Explore':'ES','Complex Explore':'EC',
    'Look':'LK','Solution':'SL','Function':'FN'
    }
    behaviour_type = behaviour_mapping[behaviour]
    
    label_mapping = dict(zip(
        data_dictionary[data_dictionary['Variable'] == 'C_1_TRT']['Value'],
        data_dictionary[data_dictionary['Variable'] == 'C_1_TRT']['Label']
    ))
    
    df_filt = df[[f'C_2_APS_{toy_initial}{behaviour_type}',f'C_3_APS_{toy_initial}{behaviour_type}',f'C_4_APS_{toy_initial}{behaviour_type}','C_1_TRT']]
    df_plot = df_filt.groupby('C_1_TRT').mean().reset_index().melt(id_vars = 'C_1_TRT')
    df_plot['Treatment Group'] = df_plot['C_1_TRT'].map(label_mapping)
    df_plot.rename(columns = {'variable':'Visit','value':'Frequency'},inplace = True)
    df_plot['Visit'] = 'V' + df_plot['Visit'].str.split('_').str[1]
    plt.figure(figsize=(10, 6))
    sns.lineplot(data = df_plot, x = 'Visit', y = 'Frequency', hue = 'Treatment Group',palette = 'bright',marker = 'o',linewidth = 3)
    plt.title(f'{behaviour} for {toy_name}')
    sns.despine()
# Usage 
# plot_for_grp_behaviour_toy(df,'Gumball','Complex Explore',my_dict)
