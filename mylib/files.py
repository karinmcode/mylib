#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 10:00:55 2023

@author: karinmorandell
"""

import os
import subprocess


def open_folder(file_path):
    
    if os.path.exists(file_path)==False:
        
        print("File does not exist. Tring opening parent folder.")
        folder_path=get_folder(file_path)
        if os.path.exists(file_path)==False:
            print("Parent folder does not exist either.")
            return


    # Extract the folder (directory) path from the file path
    folder_path = get_folder(file_path)
    
    subprocess.run(['open', folder_path])

def get_folder(file_path):
    
    # Normalize the file path (convert backslashes to forward slashes if on Windows)
    normalized_file_path = os.path.normpath(file_path)
    
    # Extract the folder (directory) path from the normalized file path
    folder_path = os.path.dirname(normalized_file_path)
    
    return folder_path