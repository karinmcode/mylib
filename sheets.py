#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 15:28:47 2023

@author: karinmorandell
"""
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import platform
import pandas as pd


# Example usage:
# df = read_google_spreadsheet(url='https://docs.google.com/spreadsheets/d/YOUR_SPREADSHEET_KEY/edit', sheet_name='Your Sheet Name')

isPC = platform.system() == 'Windows'

if isPC:
    credential_path = 'H:\My Drive\Coding\Google Cloud\myjupyterlabsheetaccess.json'
else:
    credential_path = '/Users/karinmorandell/Library/CloudStorage/GoogleDrive-karinmorandell.pro@gmail.com/My Drive/Coding/Google Cloud/myjupyterlabsheetaccess.json'


def read_google_spreadsheet(spreadsheet_name=None, url=None, key=None, sheet_name=None, use_headers=True, credential_path=None):
    
    
    # Set up the credentials
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(credential_path, scope)
    client = gspread.authorize(creds)

    # Open the spreadsheet
    if spreadsheet_name:
        spreadsheet = client.open(spreadsheet_name)
    elif url:
        spreadsheet = client.open_by_url(url)
    elif key:
        spreadsheet = client.open_by_key(key)
    else:
        raise ValueError("You must provide either a spreadsheet_name, url, or key.")

    # Open the specified sheet or default to the first sheet
    if sheet_name:
        worksheet = spreadsheet.worksheet(sheet_name)
    else:
        worksheet = spreadsheet.get_worksheet(0)

    # Get data from the sheet
    rows = worksheet.get_all_values()

    # Convert to a pandas DataFrame
    df = pd.DataFrame(rows)

    # Optionally set the first row as headers
    if use_headers:
        df.columns = df.iloc[0]
        df = df.iloc[1:]
    
    df.head()
    return df