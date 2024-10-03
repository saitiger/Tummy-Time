import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

def data_dict_fill(data_dictionary):
    data_dictionary['Variable'].ffill(inplace = True)
    return data_dictionary[['Variable','Value','Label']]

# Usage 
# my_dict = data_dict_fill(data_dict2)

def demographic_plots(df, demographic_type, score_type, data_dictionary=None):
    """
    Creates plots of demographic data over different visits.

    # Parameters:
    - df: The input dataframe.
    - demographic_type (str) : Type of demographic ('Sex', 'Race', or 'Ethnicity')
    - score_type (str) : Type of score ('RPM' or 'TWS').
    - data_dictionary (pandas.DataFrame): Data dictionary *Optional for label mapping
    
    """
    demographic_mapping = {
        'sex': 'SEX', 'Sex': 'SEX', 'SEX': 'SEX',
        'ethnicity': 'ETH', 'Ethnicity': 'ETH', 'eth': 'ETH', 'ETH': 'ETH',
        'race': 'Race', 'Race': 'Race', 'RACE': 'Race'}
    
    if demographic_type not in demographic_mapping:
        raise ValueError("demographic_type must be one of 'Sex', 'Race' or 'Ethnicity'")
    
    demographic_type = demographic_mapping[demographic_type]
    
    if score_type not in ['RPM','TWS']:
        raise ValueError("score_type must be RPM or TWS")
            
    plot_data = df.groupby(f'C_1_{demographic_type}')[[f'C_2_APS_{score_type}',f'C_3_APS_{score_type}',f'C_4_APS_{score_type}']].mean().melt()
    plot_data[demographic_type] = plot_data.groupby('variable').cumcount() + 1.0
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
        plot_data[f'{demographic_type}'] = plot_data[demographic_type].map(label_mapping)
    else:
        plot_data[f'{demographic_type}'] = plot_data[demographic_type].astype(str)
    
    # Filter out groups with all zero values
    plot_data_filtered = plot_data.groupby(f'{demographic_type}').filter(lambda x: x[score_type].sum() > 0)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=plot_data_filtered, x='Visit', y=score_type, hue=f'{demographic_type}', palette='bright')
    sns.despine()
    
    handles, labels = plt.gca().get_legend_handles_labels()
    present_labels = plot_data_filtered[f'{demographic_type}'].unique()
    plt.legend([handle for handle, label in zip(handles, labels) if label in present_labels],
               [label for label in labels if label in present_labels],
               title=demographic_type)
    
    plt.show()
    return plot_data
    
# Usage 
# demographic_plots(df2,'Race','TWS',my_dict)

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
            sns.lineplot(data=res, x = 'Visit', y = score_type)
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
        group_type = label_mapping.get(trt_value, f"Group {trt_value}")
    else:
        group_type = f"Group {trt_value}"
    
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
    
    color_plt = ["#FF5733", "#3357FF"]  
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=plot_df, x='Visit', y='Value', hue='Metric', marker='o', palette=color_plt)
    
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
    if data_dictionary is not None:
        label_mapping = dict(zip(
            data_dictionary[data_dictionary['Variable'] == 'C_1_TRT']['Value'],
            data_dictionary[data_dictionary['Variable'] == 'C_1_TRT']['Label']
        ))
        mean_data['Treatment Group'] = mean_data['C_1_TRT'].map(label_mapping)
    else:
        mean_data['Treatment Group'] = 'Group ' + mean_data['C_1_TRT'].astype(str)
    
    mean_data.drop(columns = 'C_1_TRT',inplace = True)
    
    toy_data_plot = (
        mean_data
        .melt(id_vars= 'Treatment Group')
        .rename(columns={'variable': 'Visit', 'value': 'Score'})
        .replace({f'C_2_APS_{toy_type}': 'V2', f'C_3_APS_{toy_type}': 'V3', f'C_4_APS_{toy_type}': 'V4'})
    )
    
    custom_palette = ["#FF5733", "#33FF57", "#3357FF"]
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=toy_data_plot, x='Visit', y='Score', hue='Treatment Group', palette=custom_palette)
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
