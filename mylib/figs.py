#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 15:12:05 2023

@author: karinmorandell
"""

import mylib.stats as mystats
import mylib.files as myfiles

# Basic libraries
import numpy as np
import pandas as pd
#from IPython.display import display# for displaying dataframes


# Data visualization
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#from matplotlib.colors import ListedColormap

# File management
import os
#import platform
#from scipy.io import loadmat
#import scipy.io

#import h5py

# Strings
#import re

# ML
#from sklearn.cluster import KMeans



def format_combined_data(DF, DF_restCells,quartile_labels,pAreaName,quartile_variable='resp'):
    
    # Combine the data from DF and DF_restCells, adding a 'Dataset' column
    DF['Dataset'] = 'DF'
    DF_restCells['Dataset'] = 'DF_restCells'
    
    # Add quartiles labels as column
    
    DF['Resp_Quartile'] = pd.qcut(DF[quartile_variable], q=4, labels=quartile_labels)
    DF_restCells['Resp_Quartile'] = pd.qcut(DF_restCells[quartile_variable], q=4, labels=quartile_labels)

    combined_data = pd.concat([DF, DF_restCells])
    
    # Filter data to include only AC cells
    combined_data = combined_data[combined_data['AreaName'].isin(pAreaName)]


    return combined_data

def create_combined_boxplots(DF, DF_restCells, quartile_variable = 'resp_rest'):

    # Define colors
    color1 = 'gray'
    color2 = 'blue'
    yticks = np.arange(-1,1.1,0.5)
    pAreaName = ['AAF', 'A1', 'DP', 'A2']  # Sort areas as desired
    quartile_labels = ['Q1', 'Q2', 'Q3', 'Q4']
    ystep = -0.1

    combined_data = format_combined_data(DF, DF_restCells,quartile_labels,pAreaName,quartile_variable=quartile_variable)


    ## Create subplots for each grouping
    fig, axes = plt.subplots(1, 3, figsize=(15, 10), sharey=True)  # Create 1 rows, 3 columns
    plt.rcParams['svg.fonttype'] = 'none'

###
    ## Boxplot for the entire population (ax0)
    ax = axes[0]
    sns.boxplot(x='Dataset', y='MI', data=combined_data, ax=ax, notch=False, 
                palette={'DF': color1, 'DF_restCells': color2}, 
                width=0.4, medianprops={'color':'red', 'linewidth': 1.5},
               hue = 'Dataset')
    ax.set_title('Ungrouped')
    ax.set_xlabel('')
    ax.set_yticks(yticks)

    legend_elements = [
    mpatches.Patch(color=color1, label='All cells'),
    mpatches.Patch(color=color2, label='Rest cells')
    ]
    legend=ax.legend(handles= legend_elements)
    
    # Set xtick labels for ax0
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(['All cells', 'Rest cells'])
    compute_and_display_stats_xGroups(ax=ax,DATA=combined_data,
                                      variable_name="MI",grouping_columns=['Dataset'],
                                      colors1=[color1,color2],ystep=ystep,xticks=ax.get_xticks())

    
###
    
    ## AX1 Boxplot grouped by quartile response sizes
    ax = axes[1]

    sns.boxplot(x='Resp_Quartile', y='MI', hue='Dataset', data=combined_data, ax=ax, notch=False, 
                palette={'DF': color1, 'DF_restCells': color2}, width=0.5, 
                medianprops={'color':'red', 'linewidth': 1.5},
               legend=False)
    
    ax.set_title(f'Grouped by Quartile Response Sizes ({quartile_variable})')
    ax.set_xlabel('Quartile Response Sizes')
    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    ax.set_xticklabels(quartile_labels)

    # Display stats 
    compute_and_display_stats_xGroups(ax=ax,DATA=combined_data,
                                      variable_name="MI",grouping_columns=['Dataset', "Resp_Quartile"],
                                      colors1=[color1,color2],ystep = ystep,
                                      xticks=xticks)
    
###
    
    ## AX2 Boxplot grouped by areas from pAreaName
    ax = axes[2]
    sns.boxplot(x='AreaName', y='MI', hue='Dataset', data=combined_data, ax=ax, notch=False, 
                palette={'DF': color1, 'DF_restCells': color2}, width=0.5, order=pAreaName, 
                medianprops={'color':'red', 'linewidth': 1.5},legend=False)
    
    ax.set_title('Grouped by AC areas')
    ax.set_xlabel('AC areas')
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(pAreaName)

    # Display stats 
    compute_and_display_stats_xGroups(ax=ax,DATA=combined_data,
                                      variable_name="MI",grouping_columns=['Dataset', "AreaName"],
                                      group2_options=pAreaName,
                                      colors1=[color1,color2],ystep = ystep,
                                      xticks=ax.get_xticks())
    
    
    
###
    ## Format axes
    
    for iax,ax in enumerate(axes):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_yticks(yticks)
        ax.tick_params(axis='both', colors='black') 
        ax.set_box_aspect(1.7)  # Set the aspect ratio to 1 for the spines
        ax.axhline(y=0, color='gray', linestyle=':', linewidth=1)
    
    plt.tight_layout()
        
    # Show the plots
    plt.show()

    return fig,axes

# # Example usage:
# quartile_variable = 'resp_rest'
# fig,AX = create_combined_boxplots(DF, DF_restCells,quartile_variable = quartile_variable)

# url = os.path.join(fo_figs,f'combined_boxplots_with_quartiles_tests_quartile_variable__{quartile_variable}.svg')
# my_save_plot(fig, url)



def plot_grouped_histogram(data4stats, custom_bins):
    """
    Plots filled histograms for each group in the data4stats dataframe on the same plot,
    with bars of different groups side by side and the probability value on top.
    The histogram uses custom-defined bins.

    Parameters:
    data4stats (pd.DataFrame): DataFrame with 'group' and 'values' columns.
    custom_bins (list): List of bin edges.
    """
    # Get the order of groups from the original dataframe
    group_order = data4stats['group'].unique()
    
    # Explode the 'values' column to separate rows
    exploded_data = data4stats.explode('values')
    exploded_data['values'] = pd.to_numeric(exploded_data['values'])

    # Bin the data and create custom bin labels
    exploded_data['binned'] = pd.cut(exploded_data['values'], bins=custom_bins, right=False, labels=['1', '2', '3 or more'])

    # Calculate probabilities for each bin and group
    prob_data = (exploded_data.groupby(['group', 'binned'], group_keys=True)
                 .size()
                 .groupby(level=0)
                 .apply(lambda x: x / float(x.sum()))
                 .reset_index(name='probability'))

    # Set the style for the plot
    sns.set(style="whitegrid")

    # Initialize the matplotlib figure
    fig=plt.figure(figsize=(6, 6))

    # Create the bar plot
    sns.barplot(data=prob_data, x='binned', y='probability', hue='group', palette="viridis", hue_order=group_order)

    # Add labels, title and adjust x-ticks
    plt.xlabel('Values')
    plt.ylabel('Probability')
    plt.title('Probability Histogram by Group')

    # Adjust y-axis limits
    plt.ylim(0, 1)

    # Adding the probability values on top of the bars
    ax = plt.gca()
        
    for p in ax.patches:
        # Check if the height of the patch is in the 'probability' column of prob_data
        if p.get_height() in prob_data['probability'].values:
            # Annotate the bar patch
            ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=12, color='black', xytext=(0, 5),
                        textcoords='offset points')

    # Show the plot
    plt.show()
    return fig

if False:
    # Debugging Example
    
    # Define your custom bins
    custom_bins = [1, 2, 3, np.inf]
    
    # Assuming you have a DataFrame called 'data4stats' with 'group' and 'values' columns
    # (you should replace this with your actual data)
    data4stats = pd.DataFrame({'group': ['Group1', 'Group1', 'Group2', 'Group2'],
                               'values': [1, 2, 2, 3]})
    
    # Call the function to plot grouped histograms
    fig = plot_grouped_histogram(data4stats, custom_bins)
    
    # Define the path to save the plot (you should replace this with your actual path)
    path_fig = 'path/to/save/plot.png'
    
    # Save the plot (you can use your custom save_plot function here)
    plt.savefig(path_fig)
    

def mysavefig(figure, file_path, dpi=300,open=False):
    """
    Save a Matplotlib figure to a file based on the file format determined from the file extension.

    Parameters:
        figure (matplotlib.figure.Figure): The Matplotlib figure to save.
        file_path (str): The file path where the file will be saved, including the desired extension (e.g., ".png", ".svg").
    """
    try:
        # Set the figure's DPI for high-resolution output
        figure.set_dpi(dpi)

        # Remove the box around legends
        for ax in figure.get_axes():
            legend = ax.get_legend()
            if legend:
                legend.set_frame_on(False)
            # Remove the white background box for each axis
            ax.set_facecolor('none')
            
            # Remove the box around the plot
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            

        


        # Get the directory path from the file_path
        directory = os.path.dirname(file_path)

        # Create the directory if it does not exist
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Determine the file format from the file extension
        file_format = os.path.splitext(file_path)[1][1:].lower()  # Remove the leading dot and make lowercase


        # Save the figure using the determined format
        figure.savefig(file_path, 
                       format=file_format, 
                       bbox_inches="tight",
                       transparent = True,
                       facecolor = None,
                       pad_inches = 0,
                      )
        print(f"Plot saved as {file_path}")
        
        if open:
            myfiles.open_folder(file_path)

    except Exception as e:
        print(f"Error saving plot: {e}")

# Example usage:
# my_save_plot(my_figure, "my_plot.png")  # Saves as PNG
# my_save_plot(my_figure, "my_plot.svg")  # Saves as SVG
#my_save_plot(fig, path_fig) 

def compute_and_display_stats_xGroups(ax=None, DATA=None, variable_name="",
                                      grouping_columns=None, group1_options=None,
                                      group2_options=None,
                                      colors1=['black', 'gray'], ystep=-0.1, xticks=None):
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

    Returns:
    None
    """

    if ax is None:
        print('No target axis was provided.')
        return

    if DATA is None or DATA.empty:
        print('No data was provided. Should be a dataframe.')
        return

    if variable_name == "":
        print('No variable name was provided. Should be a string.')
        return

    if not grouping_columns:
        print('No grouping columns names were provided.')
        return
    else:
        group1_column = grouping_columns[0]

        if len(grouping_columns) > 1:
            group2_column = grouping_columns[1]
        else:
            group2_column = ""

    if group1_options is None:
        group1_options = DATA[group1_column].unique()
        group1_options = sorted(group1_options)

    if group2_column and group2_options is None:
        if group2_column in DATA.columns:
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
            stats_txt = mystats.calculate_descriptive_statistics(data1,variable_name)
            ax.text(xpos, ypos,g1+" : "+stats_txt, fontsize=10, color=thiscolor)

            
            
        else:    
            print(f"group2_column = {group2_column} group2_options = {group2_options}")
            for g2 in group2_options:
                i4data = data1[group2_column].isin([g2])
                data2 = data1[i4data]
    
                # print descriptive stats
                ypos = ypos + ystep
    
                stats_txt = mystats.calculate_descriptive_statistics(data2,variable_name)
                ax.text(xpos, ypos,g2+" : "+stats_txt, fontsize=10, color=thiscolor)

            
            # print comparative stats
            ypos = ypos + ystep
            
            test_used,p_value,significant_combinations= mystats.perform_comparative_statistics(data1, group2_column, variable_name,group_order=group2_options)
            
            # add text below plot
            p_value_txt = mystats.format_pval(p_value)
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
            mystats.add_stats_annot(ax,significant_combinations,yrange=yrange2)


    if group2_column=="":
        # print comparative stats
        ypos = ypos + ystep
        
        test_used,p_value,significant_combinations= mystats.perform_comparative_statistics(DATA, group1_column, variable_name)
        
        # add text below plot
        p_value_txt = mystats.format_pval(p_value)
        stats_txt = f"{test_used}: {p_value_txt}"
        ax.text(xpos, ypos, stats_txt, fontsize=10, color=colors1[0],fontstyle='italic')

        # add stats stars and horizontal bars
        mystats.add_stats_annot(ax,significant_combinations,yrange=yrange2)


def colormap(input):
    cmap_name = 'Set2'
    if isinstance(input,int):
        n=input
    elif isinstance(input,pd.DataFrame):
        n = len(input)
    elif isinstance(input,str):
        cmap_name = input
        n = 100
    
    cmap = plt.colormaps[cmap_name]
    cluster_colors = [cmap(i) for i in np.linspace(0, 1, n)]
    
    return cmap,cluster_colors


def add_stats_annot(ax,significant_combinations,yrange=None):

    """
    Add statistical significance annotations to a barplot.

    Parameters:
        ax (matplotlib.axes.Axes): The matplotlib axes where the barplot is created.
        significant_combinations (pandas.DataFrame): DataFrame containing information about significant combinations.
        yrange (float, optional): The range of the y-axis. If not provided, it is calculated from the plot.

    Returns:
        None

    Explanation of the logic:
        This function adds statistical significance annotations to a barplot by creating significance bars and labels
        above the bars. It works as follows:

        1. It calculates the range of the y-axis if 'yrange' is not provided.
        2. It assigns significance labels ('*', '**', '***', or 'ns') to each significant combination based on 'pval'.
        3. It identifies groups for horizontal joined forked significance bars.
        4. It creates horizontal bars and labels for significant groups.
        5. It creates individual significance bars and labels for non-grouped combinations.
        6. It adjusts the y-axis limits to accommodate the added annotations.

    Example usage:
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np

        # Sample significant_combinations DataFrame
        significant_combinations = pd.DataFrame({
            'grp1': ['A', 'B', 'A', 'C'],
            'grp2': ['B', 'C', 'D', 'D'],
            'pval': [0.03, 0.001, 0.02, 0.1]
        })

        # Create a barplot
        fig, ax = plt.subplots()
        ax.bar(['A', 'B', 'C', 'D'], [10, 15, 12, 8])

        # Add statistical significance annotations
        add_stats_annot(ax, significant_combinations)

        # Show the plot
        plt.show()
    """
    
    if significant_combinations.empty:
        return

    if significant_combinations.shape[0]==0:
        return
    
    if not yrange:
        yrange,bottom,top = get_yrange(ax)
    else:
        bottom,top = plt.get_ylim(ax)

    # First make new column text
    for i, row in significant_combinations.iterrows():
        # Significance level
        p = row['pval']
        if p < 0.001:
            sig_symbol = '***'
        elif p < 0.01:
            sig_symbol = '**'
        elif p <= 0.05:
            sig_symbol = '*'
        else:
            sig_symbol = 'ns'            
        
        significant_combinations.at[i,'text']=sig_symbol
        
    # Find groups for horizontal joined forked significance bar
    barGrpId = -1.0  # Initialize barGrpId as an integer

    if significant_combinations.shape[0]==1:
        skipList = [0]
    else:
        skipList = []
    
    # Assign NaN to the 'group' column
    significant_combinations['group']= assign_groupID(significant_combinations)

                
    print(f"significant_combinations in add_stats_annot={significant_combinations}")
    
    # define heights of vertical tips
    voffset =  yrange * 0.05
    h_tips_big = yrange * 0.05
    h_tips_small = yrange * 0.025
    
    # Plot grouped significance bars
    if significant_combinations.shape[0]!=1:
        bar_groups = significant_combinations['group'].unique()
        bar_groups = bar_groups[~np.isnan(bar_groups)]
        level = 1
        
        for b in bar_groups:
            if b==np.nan:
                continue
                
            rows = significant_combinations[significant_combinations['group']==b]
            if rows.shape[0]==0:
                continue
                
            sig_symbol =  rows['text'].values[0]
        
            x1s = rows['grp1'].values
            x2s = rows['grp2'].values
    
            x1 = np.unique(x1s)
            x2 = np.unique(x2s)
            if len(x1)==1:
                xTopTip = np.mean(x2)
                x1 = np.mean(x1)
                xTxt = np.mean([x1 ,xTopTip])
                xAlone = x1
                xGroup = x2s
            else:
                xTopTip = np.mean(x1)
                x2 = np.mean(x2)
                xTxt = np.mean([x2 ,xTopTip])
                xAlone = x2
                xGroup = x1s
                
            MIN= np.min(xGroup)
            MAX= np.max(xGroup)
            
            # Plot the top bar
            
            h_bar = yrange * 0.10
            h0 = (h_bar+voffset) * level
            y_bar = top + h0 + h_bar
            h_tips = yrange * 0.025
            y_tips = y_bar-h_bar 
            ax.plot(
                [xAlone, xAlone, xTopTip, xTopTip],
                [y_tips, y_bar, y_bar, y_tips+h_tips], lw=1, c='k')
            
            h_text = (yrange * -0.01)
            y_text = y_bar + h_text
            ax.text(xTxt, y_text, sig_symbol, ha='center', c='k',fontsize=16)
            
            # Plot fork horizontal line
            ax.plot([MIN, MAX],[y_tips, y_tips]+h_tips, lw=1, c='k')  
            
            # Plot little fork
            for xg in xGroup:
                ax.plot(
                    [xg, xg],
                    [y_tips, y_tips+h_tips], lw=1, c='k')
       
            # What level is this bar among the bars above the plot?
            level +=1

    otherBars = significant_combinations[significant_combinations['group'].isna()]

    # Significance other bars
    for i, row in otherBars.iterrows():

        x1 = row['grp1']
        x2 = row['grp2']
        
        # What level is this bar among the bars above the plot?
        level = len(significant_combinations) - i
        
        # Plot the bar
        bar_height = (yrange * 0.05 * level) + top
        bar_tips = bar_height - (yrange * 0.02)
        ax.plot(
            [x1, x1, x2, x2],
            [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k')
        
        h_text = (yrange * -0.01)
        text_height = bar_height + h_text
        ax.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', c='k',fontsize=16)

    # Adjust y-axis
    #ax.set_ylim(bottom , text_height+yrange * 0.08)
    ax.set_ylim(auto=True)
    
    
def assign_groupID(s):
    #s = significant_combinations
    s['group'] = np.nan
    s['group'] = s['group'].astype(float)
    # Group the bars
    s['groupID1'] = s.groupby(['text', 'grp1']).ngroup()  
    s['groupID2'] = s.groupby(['text', 'grp2']).ngroup()  
    
    p_groupID1 = np.unique(s['groupID1'])
    p_groupID2 = np.unique(s['groupID2'])
    
    if p_groupID1!=s['groupID1']:
        for irow,row in s.iterrows():
            row['groupID1']==s['groupID1']
            
            
    
    return s


# def add_stats_annot2(ax, significant_combinations, yrange=None):
#     """
#     Add statistical significance annotations to a barplot.
    
#     [Function description remains unchanged]
#     """
    
#     if significant_combinations.empty or significant_combinations.shape[0] == 0:
#         return
    
#     if not yrange:
#         yrange, bottom, top = get_yrange(ax)
#     else:
#         bottom, top = plt.get_ylim(ax)

#     # Assign significance symbols
#     significant_combinations['text'] = significant_combinations['pval'].apply(assign_sig_symbol)

#     # Group the bars
#     significant_combinations['group'] = significant_combinations.groupby(['text', 'grp1']).ngroup()

#     grouped = significant_combinations.groupby('group')
#     bar_height = top + yrange * 0.1
#     tip_height = yrange * 0.02

#     for name, group in grouped:
#         if len(group) > 1:
#             # Draw a horizontal line for grouped bars
#             grp1 = group['grp1'].values
#             grp2 = group['grp2'].values
#             x_values = np.concatenate([grp1, grp2])
#             x_min, x_max = np.min(x_values), np.max(x_values)

#             ax.plot([x_min, x_max], [bar_height, bar_height], color='black', lw=1)
#             draw_significance_tips(ax, x_values, bar_height, tip_height)

#             # Draw text
#             ax.text((x_min + x_max) / 2, bar_height + tip_height, group['text'].iloc[0], ha='center', fontsize=12)

#         else:
#             # Draw individual bars
#             x1 = group['grp1'].values[0]
#             x2 = group['grp2'].values[0]
#             ax.plot([x1, x1, x2, x2], [bar_height - tip_height, bar_height, bar_height, bar_height - tip_height], color='black', lw=1)
#             ax.text((x1 + x2) / 2, bar_height + tip_height, group['text'].iloc[0], ha='center', fontsize=12)

#         bar_height += yrange * 0.1  # Increase height for the next group

#     ax.set_ylim(bottom, bar_height + yrange * 0.2)  # Adjust y-axis limits
#     ax.set_ylim(auto=True)

# def assign_sig_symbol(pval):
#     """Assign significance symbols based on p-value."""
#     if pval < 0.001:
#         return '***'
#     elif pval < 0.01:
#         return '**'
#     elif pval <= 0.05:
#         return '*'
#     else:
#         return 'ns'

# def draw_significance_fork(ax, x_values, bar_height, tip_height):
#     """Draw tips for significance bars."""
#     for x in x_values:
#         ax.plot([x, x], [bar_height, bar_height + tip_height], color='black', lw=1)
  

# def draw_significance_grouped_fork(ax, x_values, bar_height, tip_height):
#     """Draw tips for significance bars."""
#     for x in x_values:
#         ax.plot([x, x], [bar_height, bar_height + tip_height], color='black', lw=1)


# def draw_significance_tips(ax, x_values, bar_height, tip_height):
#     """Draw tips for significance bars."""
#     for x in x_values:
#         ax.plot([x, x], [bar_height, bar_height + tip_height], color='black', lw=1)

    
def get_yrange(ax):
    """
    Calculate the range of y-axis values in a matplotlib Axes.
    
    [Function description remains unchanged]
    """    
    # Initialize min and max values
    ymin = float('inf')
    ymax = float('-inf')

    # Iterate through all artists on the plot
    for artist in ax.get_children():
        #print(type(artist))
        
        if isinstance(artist, plt.Line2D):
            # Handle Line2D objects (lines or error bars)
            xdata, ydata = artist.get_data()
            if len(ydata) > 0:
                ymin = min(ymin, np.min(ydata))
                ymax = max(ymax, np.max(ydata))
                
        elif isinstance(artist, mpatches.Rectangle):
            # Handle Rectangle objects (bars in barplots)
            thisy_min = artist.get_y() # Assuming the bar height represents the y-value
            thisy_max = artist._height
            #print(ydata)
            if isinstance(thisy_min, np.ndarray):
                ymin = min(ymin, np.min(thisy_min))
                ymax = max(ymax, np.max(thisy_max))
            else:
                ymin = min(ymin, thisy_min)
                ymax = max(ymax, thisy_max)
                
        elif isinstance(artist, plt.Text):
            # Handle Text objects (annotations)
            position = artist.get_position()  # Get the position in data coordinates
            y_position = position[1]  # Get the y-coordinate of the position
            ymin = min(ymin, y_position)
            ymax = max(ymax, y_position)
        yrange = ymax - ymin
        #print(yrange)


    yrange = ymax - ymin
    yrange = np.float64(yrange)

    return yrange,ymin,ymax

