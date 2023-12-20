#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 15:06:59 2023

@author: karinmorandell
"""

import mylib.figs as myfigs

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import shapiro, ttest_ind, ranksums, f_oneway, kruskal
import scikit_posthocs as sp


def format_pval(pval, verbose=None):

    if pval>0.15:
        p_value_txt = "p > 0.15"
    elif (pval<=0.15) & (pval>0.05):
        p_value_txt = f"p = {round(pval,3)}"
    elif (pval<=0.05) & (pval>0.01):
        p_value_txt = f"p = {round(pval,3)}"
    elif (pval<=0.01) & (pval>0.001):
        p_value_txt = f"p = {round(pval,3)}"
    elif pval<=0.001:
        p_value_txt = "p < 0.001" 
        
    if verbose==1:
        print(p_value_txt)
    
    return p_value_txt

def calculate_descriptive_statistics(df, column_name):
    """
    Calculate statistics (mean, median, and standard deviation) for a DataFrame column.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column for which statistics will be calculated.

    Returns:
        str: A string containing the statistics information.
    """
    column_data = df[column_name]
    mean_value = round(column_data.mean(), 2)  # Rounded mean
    median_value = round(column_data.median(), 2)  # Rounded median
    std_value = round(column_data.std(), 2)  # Rounded standard deviation
    sem_value = column_data.sem()

    statistics_info = f'Mean±SEM: {mean_value} ± {sem_value:.2f}, Std: {std_value}, Median: {median_value}'
    return statistics_info
    



def perform_comparative_statistics(data, group_column, variable_column,xticks = None, group_order=None,alpha=0.05,verbose=False):
    # debugging data=DF
    #group_column = 'Resp_Quartile'
    
    if not group_order: 
        groups = np.unique(data[group_column].values)
        groups = np.sort(groups)
    else:
        groups = group_order
    ngroups = len(groups)
    
    dataIsAlreadyGrouped = len(data)==ngroups# data is grouped in dataframe
    if dataIsAlreadyGrouped:
        data = data.explode(variable_column)
        data[variable_column] = pd.to_numeric(data[variable_column])

    grouped_data = [data[data[group_column] == group][variable_column].values for group in groups]

    if not xticks:
        L = np.arange(ngroups)
    else:
        L = np.array(xticks)  # Convert xticks to a NumPy array
            
    
    parametric = True  # Default to parametric test, change based on normality test
    # Check for statistical significance pair-wise comparison
    significant_combinations = []  # Initialize as an empty list
    # Check normality for each group
    normality_test_results = []

    for group in groups:
        group_data = data[data[group_column] == group][variable_column]
        stat, p = shapiro(group_data)
        normality_test_results.append((group, stat, p))

    # Perform Levene's test for homoscedasticity
    homoscedasticity_test_results = stats.levene(*grouped_data)
    
    # If any group fails normality test, switch to non-parametric
    if any(p > alpha for (_, _, p) in normality_test_results):
        parametric = False

    # If any group fails homoscedasticity test, switch to non-parametric
    if homoscedasticity_test_results.pvalue > alpha:
        parametric = False
    
    if ngroups <= 2:
        if parametric:
            _, p_value = ttest_ind(data[data[group_column] == groups[0]][variable_column],
                                   data[data[group_column] == groups[1]][variable_column])
            test_used = 'T-Test'
        else:
            _, p_value = ranksums(data[data[group_column] == groups[0]][variable_column],
                                  data[data[group_column] == groups[1]][variable_column])
            test_used = 'Rank Sum Test'

        if p_value < 0.05:
            significant_combinations.append({'grp1': 0, 'grp2': 1, 'pval': p_value})
        
    else:
        if parametric:
            _, p_value = f_oneway(*grouped_data)
            test_used = 'ANOVA Test'
        else:
            _, p_value = kruskal(*grouped_data)
            test_used = 'Kruskal-Wallis Test'

        # add multiple comparison pvalue Bonnferonni correction
        if not xticks:
            L = np.arange(ngroups)
        else:
            L = np.array(xticks)  # Convert xticks to a NumPy array
        combinations = [(L[x], L[x + y]) for y in reversed(range(1, len(L))) for x in range((len(L) - y))]
        combinations = sorted(combinations, key=lambda x: (x[0], x[1]))# important for plotting stars in an organized way
        
        n_combinations = len(combinations)
        if not group_order:
            data_sorted =  data.sort_values(by=group_column) 
        else:
            data_sorted = data.set_index(group_column).loc[group_order].reset_index()


        if p_value<=alpha:
            if parametric:
                # Perform Tukey's HSD post-hoc test (parametric)
                posthoc_result = sp.posthoc_tukey(data_sorted, val_col=variable_column, group_col=group_column)
            else:
                # Perform Dunn's test (non-parametric)
                posthoc_result = sp.posthoc_dunn(data_sorted, val_col=variable_column, group_col=group_column)
            
            # print results for paper
            posthoc_result_corrected = posthoc_result*n_combinations
            posthoc_result_corrected = np.round(posthoc_result_corrected,4)
            
            # Get the shape of the array
            n_rows, n_cols = posthoc_result_corrected.shape
            
            # Create a Boolean mask for the upper triangle (including the diagonal)
            upper_triangle = np.triu(np.ones((n_rows, n_cols), dtype=bool), k=0)

            posthoc_result_formatted = posthoc_result_corrected
            posthoc_result_formatted[upper_triangle] = np.nan
            posthoc_result_formatted[posthoc_result_corrected > alpha] = 'ns'
            
            posthoc_result_formatted = posthoc_result_formatted.fillna(' ')

            print(posthoc_result_formatted)
            
            for c in combinations:
                group1 = c[0]
                group2 = c[1]
                # Significance

                p = posthoc_result.iloc[group1, group2]

                # Apply Bonferroni correction to the p-value
                p = p * n_combinations
                
                if p <= 0.05:
                    # Append the significant combination to the list as a dictionary
                    significant_combinations.append({'grp1': group1, 'grp2': group2, 'pval': p})

    # Convert the list of dictionaries into a DataFrame
    significant_combinations_df = pd.DataFrame(significant_combinations)
    
    
    # Verbose output
    if verbose:
        # Print summary of the statistical tests used
        print(f"Statistical Test Used: {test_used}")
        print(f"P-Value: {format_pval(p_value)}")
    
        # Calculate mean and standard error for each group
        summary_stats = data.groupby(group_column)[variable_column].agg(['mean', 'sem'])
        print("\nGroup Statistics (Mean ± SE):")
        paper_string = ''
        for group in groups:
            mean = summary_stats.loc[group, 'mean']
            sem = summary_stats.loc[group, 'sem']
            print(f"Group {group}: {mean:.3f} ± {sem:.3f}")
            paper_string = paper_string+f"{group}: {mean:.2f} ± {sem:.2f}, "
        
        print(paper_string)
    
    
    return test_used, p_value ,significant_combinations_df
'''
# Create a sample DataFrame
data = pd.DataFrame({
    'Group': ['A', 'A','A','B', 'B','B', 'C', 'C','C', 'D', 'D','D'],
    'Value': [2, 1,3,       31, 36,39,    30, 35,38,    20,21,23]
})

test_used, p_value, significant_combinations = perform_comparative_statistics(data, 'Group', 'Value')
print(significant_combinations)
add_stats_annot(plt.gca(),significant_combinations)
'''    






    
def compute_and_display_stats_xGroups(ax=None, # destination axis defined as 
                                      DATA=None, # dataframe with 2 possible configurations : grouped or lists
                                      variable_name=None,#
                                      grouping_columns=None, 
                                      group1_options=None,
                                      group2_options=None, 
                                      colors1=['black', 'gray'],
                                      ystep = -0.1,
                                      xticks=None,
                                      explode_dataframe=False):

    """
    Compute and display statistics for grouped data in a Matplotlib axis.
    
    Parameters:
    ax (matplotlib axis, optional): The destination axis for plotting.
    DATA (pandas DataFrame, optional): The dataframe containing the data.
    variable_name (str, optional): The name of the variable to compute statistics for.
    grouping_columns (list, optional): A list of column names for grouping data.
    group1_options (list, optional): A list of options for the first grouping column.
    group2_options (list, optional): A list of options for the second grouping column.
    colors1 (list, optional): A list of colors for plotting.
    ystep (float, optional): The vertical step for text placement.
    xticks (list, optional): Custom x-axis tick positions.
    explode_dataframe (bool, optional): Whether to explode the DataFrame before processing.
    
    Returns:
    None
"""

    if not ax: # Check if is None or empty
        print('No target axis was provided.')
        return
    
    if DATA is None or DATA.empty:
        print('No data was provided')
        return
    
    if explode_dataframe:
        # Explode the 'values' column to separate rows
        DATA = DATA.explode(variable_name)
        # DATA[variable_name] = pd.to_numeric(DATA[variable_name])

    if variable_name  is None:
        print('No variable name was provided.')
        return

    if not grouping_columns:
        print('No grouping columns names were provided.')
        return
    else:
        group1_column = grouping_columns[0]

        if len(grouping_columns)>1:
            group2_column = grouping_columns[1]
        else:
            group2_column = ""

    
    if not group1_options:
        group1_options = DATA[group1_column].unique()
        group1_options = sorted(group1_options)


    if not group2_options:
        if group2_column !="": 
            group2_options = DATA[group2_column].unique()
            group2_options = sorted(group2_options)


    # Print statistics for group 1
    xpos = -0.5
    # Get info about y-axis
    bottom, top = ax.get_ylim()
    yrange = top - bottom
    yrange = 2.2
    ypos = -0.7*yrange
    ystep = ystep*yrange
    yrange2 = 2.2
    
    for idx,g1 in enumerate(group1_options):
        i4data = DATA[group1_column].isin([g1])
        data1 = DATA[i4data]

        thiscolor = colors1[idx]
        print(f"\ngroup1_column = {group1_column} - group1_option = {g1}")
        
        if group2_column=="":

            # print descriptive stats
            ypos = ypos + ystep
            stats_txt = calculate_descriptive_statistics(data1,variable_name)
            ax.text(xpos, ypos,g1+" : "+stats_txt, fontsize=10, color=thiscolor)

            
            
        else:    
            print(f"group2_column = {group2_column} group2_options = {group2_options}")
            for g2 in group2_options:
                i4data = data1[group2_column].isin([g2])
                data2 = data1[i4data]
    
                # print descriptive stats
                ypos = ypos + ystep
    
                stats_txt = calculate_descriptive_statistics(data2,variable_name)
                ax.text(xpos, ypos,g2+" : "+stats_txt, fontsize=10, color=thiscolor)

            
            # print comparative stats
            ypos = ypos + ystep
            
            test_used,p_value,significant_combinations= perform_comparative_statistics(data1, group2_column, variable_name,group_order=group2_options)
            
            # add text below plot
            p_value_txt = format_pval(p_value)
            stats_txt = f"{test_used}: {p_value_txt}"
            ax.text(xpos, ypos, stats_txt, fontsize=10, color=thiscolor,fontstyle='italic')

            # add stats stars and horizontal bars
            if idx==0:
                xshift=-0.1
            else:
                xshift=0.1
                
            for irow,row in significant_combinations.iterrows():
                significant_combinations.loc[irow,'grp1']= row['grp1']+ xshift
                significant_combinations.loc[irow,'grp2']= row['grp2']+ xshift
                
            #print(f"significant_combinations = {significant_combinations}")
            myfigs.add_stats_annot(ax,significant_combinations,yrange=yrange2)


    if group2_column=="":
        # print comparative stats
        ypos = ypos + ystep
        
        test_used,p_value,significant_combinations= perform_comparative_statistics(DATA, group1_column, variable_name)
        
        # add text below plot
        p_value_txt = format_pval(p_value)
        stats_txt = f"{test_used}: {p_value_txt}"
        ax.text(xpos, ypos, stats_txt, fontsize=10, color=colors1[0],fontstyle='italic')

        # add stats stars and horizontal bars
        myfigs.add_stats_annot(ax,significant_combinations,yrange=yrange2)


