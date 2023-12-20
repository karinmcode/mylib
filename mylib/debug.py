#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 15:30:06 2023

@author: karinmorandell
"""

# Basic libraries
import numpy as np
import pandas as pd
from IPython.display import display# for displaying dataframes

def myvar(variable_or_name, max_elements=5, max_chars=1000):
    
    # If variable_or_name is a string, assume it's the name of the variable
    if isinstance(variable_or_name, str):
        var_name = variable_or_name
        variable = globals().get(var_name, None)
        if variable is None:
            print(f"Variable '{var_name}' not found in the current namespace.")
            return
    else:
        # If it's not a string, assume it's the variable itself
        variable = variable_or_name
        # Attempt to get the variable name from the caller's namespace
        callers_local_vars = locals()
        for var in callers_local_vars:
            if callers_local_vars[var] is variable:
                var_name = var
                break
        else:
            # If not found, use a default name
            var_name = "Variable"

    # Check if the variable is an int
    if isinstance(variable, int):
        var_type = type(variable).__name__
        print(f"{var_name}: {var_type} (Value: {variable})")
    
    # Check if the variable is a NumPy array
    elif isinstance(variable, np.ndarray):
        var_shape = variable.shape
        var_type = type(variable).__name__
        
        print(f"{var_name}: {var_type} (Shape: {var_shape})")
        
        # Convert the variable slice to a list before printing
        values_slice = variable[:max_elements].tolist()
        values_repr = str(values_slice)
        if len(values_repr) > max_chars:
            values_repr = values_repr[:max_chars] + '...'
        print(f"Sneak peek of values: {values_repr}")
    
    # Check if the variable is a dictionary
    elif isinstance(variable, dict):
        var_type = type(variable).__name__
        print(f"{var_name}: {var_type}")
        for key, value in variable.items():
            key_type = type(value).__name__
            key_shape = ()
            if isinstance(value, np.ndarray):
                key_shape = value.shape
            print(f"  Key: '{key}' ({key_type}, Shape: {key_shape})")
            
    # Check if the variable is a list
    elif isinstance(variable, list):
        max_elements = 5
        var_type = type(variable).__name__
        print(f"{var_name}: {var_type} with {len(variable)} items")

        if len(variable)>(max_elements*2):
            str_repr = str(variable[0:max_elements])+ '...'+str(variable[-max_elements:-1])
            
        if len(str_repr) > max_chars:
            str_repr = str_repr[:max_chars] + '...'
        print(f"Sneak peek of values: {str_repr}")

    else:
        var_type = type(variable).__name__
        print(f"{var_name}: {var_type}")
    
    print("")
    
    
    
    


def printD(D, var_names=None):
    info_list = []
    if var_names!=None:
        D = {k: D[k] for k in var_names if k in D}
    
    for field_name, data in D.items():
        info = {"Name": field_name}

        if isinstance(data, list):
            info["Size"] = len(data)
            info["Type"] = "list"
            info["Snapshot"] = str(data[:5])  # Showing first 5 elements
        elif isinstance(data, pd.DataFrame):
            info["Size"] = data.shape
            info["Type"] = "pd.DataFrame"
            info["Snapshot"] = str(data.head())
        elif isinstance(data, dict):
            info["Size"] = len(data)
            info["Type"] = "dict"
            info["Snapshot"] = str({k: data[k] for k in list(data)[:5]})  # First 5 key-value pairs
        else:
            info["Size"] = data.shape if hasattr(data, 'shape') else "N/A"
            info["Type"] = type(data).__name__
            info["Snapshot"] = str(data)[:100]  # First 100 characters

        info_list.append(info)

    info_df = pd.DataFrame(info_list)
    display(info_df)


if False:
    # Example usage
    D = {
        'list_example': [1, 2, 3, 4, 5, 6],
        'dataframe_example': pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]}),
        'dict_example': {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'},
        'number': 42
    }
    printD(D)

if False:
    printD(D)
