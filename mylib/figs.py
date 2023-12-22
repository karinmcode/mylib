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

    if grouping_columns is None:
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
                significant_combinations.loc[irow,'condition1']= row['condition1']+ xshift
                significant_combinations.loc[irow,'condition2']= row['condition2']+ xshift
                
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

def plot_stats_annot(ax,significant_combinations,yrange=None):
    add_stats_annot(ax,significant_combinations,yrange=yrange)

def plot_sig_annot(ax,significant_combinations,yrange=None):
    add_stats_annot(ax,significant_combinations,yrange=yrange)

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
            'condition1': ['A', 'B', 'A', 'C'],
            'condition2': ['B', 'C', 'D', 'D'],
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
    
    ## CHECK INPUTS
    if significant_combinations.empty:
        print('No significant comparison found.')
        return

    if significant_combinations.shape[0]==0:
        print('No significant comparison found.')
        return
    
    # replace old version of significan_combinations
    if all(item in significant_combinations.columns for item in ['grp1', 'grp2']):
        significant_combinations=significant_combinations.rename({'grp1':'condition1','grp2':'condition2'})
    
    significant_combinations = add_xtick_values(ax,significant_combinations)
    
    if yrange is None:
        yrange,bottom,top = get_yrange(ax)
    else:
        bottom,top = plt.get_ylim(ax)

    ## GET SIGNIFICANCE STARS
    # First make new column text
    for i, row in significant_combinations.iterrows():
        # Significance level
        sig_stars = pval2stars(row['pval'])
        significant_combinations.at[i,'text']=sig_stars
        
    ## DEFINE GROUPS FOR JOINT SIGNIFICANCE BARS
    # Find groups for horizontal joined forked significance bar
    significant_combinations= assign_groupID(significant_combinations)
                    
    # Define offsets, heights of vertical tips
    voffset =  yrange * 0.05
    tip_height = yrange * 0.03
    voffset_text = yrange * 0.11
    voffset2 = voffset+tip_height*1.5
    top = top+tip_height
    level = 1

    ## PLOT SIGNIFICANCE COMPARISONS BETWEEN 2 CONDITIONS
    bars_2 = significant_combinations[significant_combinations['group_size']==1]
    previous_x1 = np.nan
    previous_x2 = np.nan
    for i, row in bars_2.iterrows():

        x1 = row['x1']
        x2 = row['x2']
        sig_stars = row['text']
        
        # give more space between bars if they are intersecting
        intersecting = are_bars_intersecting(x1, x2, previous_x1, previous_x2)
        if intersecting:
            level +=1
        
        # Compute bar y value
        ybar = top + (voffset * level)
             
        
        # Plot the bar
        plot_sig_bar_2_condition(ax,x1,x2,ybar,sig_stars,tip_height,voffset_text = voffset_text)
        
        # What level is this bar among the bars above the plot?
        level +=1
        
        previous_x1 = x1
        previous_x2 = x2
        
    ## PLOT GROUPED SIGNIFICANCE COMPARISONS
    level +=2
    i4groupedSig = significant_combinations['group_size']>1
    bar_groups = significant_combinations[i4groupedSig]['group'].unique()
    previous_x1 = np.nan
    previous_x2 = np.nan    
    top = top + voffset * level - voffset2
    voffset = voffset2 
    level = 1
    for b in bar_groups:
            
        rows = significant_combinations[significant_combinations['group']==b]
                        
        
        # give more space between bars if they are intersecting
        x1 = np.min(rows['x1'].values)
        x2 = np.max(rows['x2'].values)
        intersecting = are_bars_intersecting(x1, x2, previous_x1, previous_x2)
        if intersecting:
            level +=1

            
        # Compute top bar y value
        ybar = top + voffset * level  
        
        # Plot significance of grouped bar
        plot_sig_bar_grouped(ax,rows,ybar,tip_height,voffset_text = voffset_text)
        
        # What level is this bar among the bars above the plot?
        level +=1
        
        previous_x1 = x1
        previous_x2 = x2

        
    # Adjust y-axis
    yrange,bottom,newtop = get_yrange(ax)
    newtop = newtop+3*tip_height
    ax.set_ylim(bottom , newtop)

def are_bars_intersecting(current_x1, current_x2, previous_x1, previous_x2):
    """
    Check if the current bar intersects with the previous bar.
    
    :param current_x1: The starting x value of the current bar.
    :param current_x2: The ending x value of the current bar.
    :param previous_x1: The starting x value of the previous bar.
    :param previous_x2: The ending x value of the previous bar.
    :return: True if the bars intersect without just touching, False otherwise.
    """
    included = (previous_x1 >= current_x1 and previous_x2 <= current_x2)
    current_smaller = (previous_x1 <= current_x1 and previous_x2 >= current_x2)
    intersect_left = (previous_x1 > current_x1 and previous_x1 < current_x2)
    intersect_right = (previous_x2 < current_x2 and previous_x1 < current_x2)
    
    # Check if either end of one bar is strictly within the range of the other bar
    return  intersect_left or \
            intersect_right or \
            included or \
            current_smaller
               
               



def plot_sig_bar_2_condition(ax,x1,x2,ybar,sig_stars,tip_height,voffset_text=None,color=None,fontsize=12):
    
    if color is None:
        color = [c + 0.001 for c in [0,0,0]]# set color that is slightly different from black for easier Illustrator selection
    
    if voffset_text is None:
        voffset_text = tip_height
        
        
    # Adjust vertical offset for text to be less if the text contains stars
    if '*' in sig_stars:
        voffset_text = voffset_text*0.8
            
    ytips = ybar-tip_height
    ax.plot(
        [x1, x1, x2, x2],
        [ytips, ybar, ybar, ytips], lw=1, c=color)
    
    ytext = ybar+voffset_text
    ax.text((x1 + x2) * 0.5, ytext, sig_stars, ha='center',va='top', c=color,fontsize=fontsize)


def plot_sig_bar_grouped(ax,rows,ybar,tip_height,color=None,voffset_text=None,fontsize=12):
    
    if color is None:
        color = [c + 0.001 for c in [0,0,0]]# set color that is slightly different from black for easier Illustrator selection
        
    if voffset_text is None:
        voffset_text = tip_height
        
        
    sig_stars =  rows['text'].values[0]
    
    # Adjust vertical offset for text to be less if the text contains stars
    if '*' in sig_stars:
        voffset_text = voffset_text*0.8 
        

    x1s = rows['x1'].values
    x2s = rows['x2'].values

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
        
    # Compute x extent of bar
    MIN= np.min(xGroup)
    MAX= np.max(xGroup)
    
    # Plot top bar
    ytips = ybar-tip_height 
    ax.plot(
        [xAlone, xAlone, xTopTip, xTopTip],
        [ytips-1.5*tip_height, ybar, ybar,  ytips], lw=1, c=color)
    
    # Plot stars
    ytext = ybar + voffset_text
    ax.text(xTxt, ytext, sig_stars, ha='center',va='top', c=color,fontsize=fontsize)
    
    # Plot fork horizontal line
    ax.plot([MIN, MAX],[ytips, ytips], lw=1, c=color)  
    
    # Plot little vertical tips
    for xg in xGroup:
        ax.plot(
            [xg, xg],
            [ytips, ytips-tip_height], lw=1, c=color)

    


def  add_xtick_values(ax,s):
    # s = significant_combinations
    s['x1']= np.nan
    s['x2']= np.nan
    
    # Get x-axis tick labels
    xtick_labels = [xtick_label.get_text() for xtick_label in ax.get_xticklabels()]
    xtick = ax.get_xticks()
    
        
    for idx,row in s.iterrows():
        c1 = row['condition1']
        c2 = row['condition2']
        if isinstance(s['condition1'][0],str):
          
            s.at[idx,'x1']=[x for x, label in zip(xtick, xtick_labels) if label==c1][0]
            s.at[idx,'x2']=[x for x, label in zip(xtick, xtick_labels) if label==c2][0]
        
        else:
            
            s.at[idx,'x1']=c1
            s.at[idx,'x2']=c2
    
    return s


def assign_groupID(s,debug=False):
    """s is significant combinations dataframe 
    
    # Sample significant_combinations DataFrame
    s = pd.DataFrame({
        'condition1': ['A', 'B', 'A', 'C'],
        'condition2': ['B', 'C', 'D', 'D'],
        'text': ['**','***','**','ns']
    })
    
    # Sample significant_combinations DataFrame with group by condition2 better
    s = pd.DataFrame({
        'condition1': ['A', 'B', 'A', 'C'],
        'condition2': ['B', 'C', 'D', 'D'],
        'text': ['ns','***','**','**']
    })    
    
    # DEBUG      
    s = significant_combinations
    """
    
    
    # Sort by condition1 to start
    s = s.sort_values(by=['condition1','text'], ascending=[True, True])
    
    if debug:
        print(s)
    
    # Group the bars
    s['group1'] = s.groupby(['text', 'condition1']).ngroup()  
    s['group2'] = s.groupby(['text', 'condition2']).ngroup()  
    
    if debug:
        print(s)  
        
    ## Check if it is better to group by condition 1 or 2 according to the size of the group
    # First compute the group size for all combinations
    # Initialize the group column with integers
    s['group1_size'] =  np.nan
    s['group2_size'] =  np.nan    
    for idx,row in s.iterrows():
        groupID = row['group1']
        ncomp = np.sum(s['group1']==groupID)
        s.at[idx,'group1_size']=ncomp
        
        groupID = row['group2']
        ncomp = np.sum(s['group2']==groupID)
        s.at[idx,'group2_size']=ncomp  
        
        
    if debug:
        print(s)
        
    # Second, decide which grouping is best
    group2_better_than_group1 = np.any(s['group1_size']<s['group2_size'])
    if group2_better_than_group1:
        # Sort by group2 and text for clarity
        s = s.sort_values(by=['group2','group2_size','text'], ascending=[True, False,True])
        if debug:
            print(s)        
        s['group']=s['group2']
        s['group_size']=s['group2_size']
    else:
        s = s.sort_values(by=['group1','group1_size','text'], ascending=[True, False,True])
        s['group']=s['group1']
        s['group_size']=s['group1_size']
    
    return s


def pval2stars(p):

    if p < 0.001:
        sig_stars = '***'
    elif p < 0.01:
        sig_stars = '**'
    elif p <= 0.05:
        sig_stars = '*'
    else:
        sig_stars = 'ns'   
        
    return sig_stars
    
    
def get_yrange(ax):
    """
    Calculate the range of y-axis values in a matplotlib Axes.
    """    
    # Initialize min and max values
    ymin = float('inf')
    ymax = float('-inf')

    # Iterate through all artists on the plot
    for artist in ax.get_children():
        
        if isinstance(artist, plt.Line2D):
            
            # Check if the artist is a whisker or cap in a boxplot
            if 'whisker' in artist.get_label() or 'cap' in artist.get_label():
                ydata = artist.get_ydata()
                ymin = min(ymin, np.min(ydata))
                ymax = max(ymax, np.max(ydata))

            else:
                # Handle regular Line2D objects
                xdata, ydata = artist.get_data()
                if len(ydata) > 0:
                    ymin = min(ymin, np.min(ydata))
                    ymax = max(ymax, np.max(ydata))

                
        # elif isinstance(artist, mpatches.Rectangle):
        #     # Handle Rectangle objects (bars in barplots)
        #     thisy_min = artist.get_y() # Assuming the bar height represents the y-value
        #     thisy_max = artist._height
        #     print(f"thisy_min={thisy_min}")
        #     print(f"artist={artist}")
        #     print(f"artist.color={artist.get_facecolor()}")
        #     color = artist.get_facecolor()
        #     if artist.get_alpha() is None:
        #         continue           
        #     if artist.get_facecolor() is None:
        #         continue                  
        #     print(f"thisy_min={artist.get}")
        #     if isinstance(thisy_min, np.ndarray):
        #         ymin = min(ymin, np.min(thisy_min))
        #         ymax = max(ymax, np.max(thisy_max))
        #     else:
        #         ymin = min(ymin, thisy_min)
        #         ymax = max(ymax, thisy_max)
                
        elif isinstance(artist, plt.Text):
            # Handle Text objects (annotations)
            position = artist.get_position()  # Get the position in data coordinates
            #print(f"position={position}")
            alpha = artist.get_alpha()
            if alpha is None:
                continue
            y_position = position[1]  # Get the y-coordinate of the position
            ymin = min(ymin, y_position)
            ymax = max(ymax, y_position)
        #print(f"ymin({type(artist)})={ymin}")


    yrange = ymax - ymin
    yrange = np.float64(yrange)

    return yrange,ymin,ymax

# Function to check if an artist is part of a boxplot
def is_boxplot_artist(artist, boxplot_elements):
    for line_collections in ['whiskers', 'caps', 'boxes', 'medians', 'fliers']:
        if artist in boxplot_elements[line_collections]:
            return True
    return False

DEBUG=False
if DEBUG:
    import random
    

    # Generating random y-values for the bar plot
    y_values = [random.randint(5, 20) for _ in range(6)]
    
    # Generating random p-values for the significant_combinations DataFrame
    p_values = [round(random.uniform(0.001, 0.1), 3) for _ in range(10)]
    
    # Sample significant_combinations DataFrame with random p-values
    significant_combinations = pd.DataFrame({
        'condition1': ['A', 'B', 'A', 'C', 'D', 'E', 'F', 'A', 'B', 'C'],
        'condition2': ['B', 'C', 'D', 'D', 'E', 'F', 'A', 'F', 'E', 'A'],
        'pval': p_values
    })
    
    # Sample significant_combinations DataFrame with random p-values
    significant_combinations = pd.DataFrame({
        'condition1': [1, 2, 1, 3, 4, 5, 6, 1, 2, 3],
        'condition2': [2,3,4,4,5,6,1,6,5,1],
        'pval': p_values
    })    
    
    # Create a barplot with random y-values
    xvalues= list(significant_combinations['condition1'].unique())
    fig, ax = plt.subplots()
    ax.bar(xvalues, y_values)
    
    # Add statistical significance annotations
    add_stats_annot(ax, significant_combinations)
    
    # Show the plot
    plt.show()
