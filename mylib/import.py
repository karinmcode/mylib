#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 15:33:46 2023

@author: karinmorandell
"""

import scipy.io
import h5py
import numpy as np
import pandas as pd


def import_mat_small(file_path, fields_to_import):
    try:
        # Load the MATLAB .mat file
        mat_file = scipy.io.loadmat(file_path)

        # Create a dictionary to store the imported fields
        imported_data = {}

        # Access and store the specified fields from the loaded .mat file
        for field in fields_to_import:
            if field in mat_file:
                imported_data[field] = mat_file[field]
            else:
                print(f"Field '{field}' not found in the .mat file.")

        # Perform operations with the imported data as needed
        # ...

        return imported_data

    except FileNotFoundError:
        print(f"The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None



def import_mat_data(file_path, fields_to_import):
    """
    Import specific fields from a .mat file saved in HDF5 format.

    Args:
        file_path (str): Path to the .mat file.
        fields_to_import (list): List of field names to import from the 'DATA' structure.

    Returns:
        dict: A dictionary containing the imported fields.
    """
    data_dict = {}

    try:
        with h5py.File(file_path, 'r') as mat_file:
            data_group = mat_file['DATA']
            print(data_group.keys())

            for field_name in fields_to_import:
                field_data = data_group[field_name]
                
                if isinstance(field_data, h5py.Dataset):
                    # Handle Matlab vectors
                    data_dict[field_name] = np.array(field_data)
                    
                elif isinstance(field_data, h5py.Group):
                    # Handle cell arrays, tables, or structures
                    if field_data.get('cell_contents') is not None:
                        cell_contents = field_data['cell_contents']
                        
                        if cell_contents.shape[0] == 1:
                            # If it's 1 row and n columns, convert it to a list of strings
                            data_dict[field_name] = [str(item[0]) for item in cell_contents[0]]
                        else:
                            # If it's n rows and 1 column, convert it to a list of strings
                            data_dict[field_name] = [str(item[0]) for item in cell_contents]
                    else:
                        data_dict[field_name] = {}
                        for subfield_name in field_data.keys():
                            subfield_data = field_data[subfield_name]
                            if isinstance(subfield_data, h5py.Dataset):
                                data_dict[field_name][subfield_name] = np.array(subfield_data)
                            elif isinstance(subfield_data, h5py.Group):
                                # Handle nested structures if needed
                                data_dict[field_name][subfield_name] = {}
                                for nested_subfield_name in subfield_data.keys():
                                    nested_subfield_data = subfield_data[nested_subfield_name]
                                    data_dict[field_name][subfield_name][nested_subfield_name] = np.array(nested_subfield_data)

    except Exception as e:
        print(f"Error: {e}")
    
    return data_dict