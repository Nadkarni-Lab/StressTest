# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 07:58:15 2024

@author: klange01
"""


import sys
import_folder_path = r"C:\Users\klange01\Desktop\Email Response\code"
sys.path.append (import_folder_path)

from call_GPT_research import GPTAgent

gptagent = GPTAgent ()

import pandas as pd
import os

import json

import numpy as np

from datasets import load_dataset

ds = load_dataset("openlifescienceai/medmcqa")

df_qa = pd.DataFrame (ds ["train"])

df_qa = df_qa [df_qa ["choice_type"] == "single"].reset_index (drop = True)

df_qa ["raw_question"] = df_qa ["question"]

df_qa ["question"] = df_qa.apply (lambda row:f"""
{row ["raw_question"]}
A: {row ["opa"]}
B: {row ["opb"]}
C: {row ["opc"]}
D: {row ["opd"]}""", axis = 1)

df_qa ["answer"] = df_qa ["cop"].map ({0:"A", 1:"B", 2:"C", 3:"D"})






# Define the path to the folder
folder_path = '17.8.24'

# List all pickle files in the folder
pickle_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]

# Load each pickle file into a dataframe and store in a list
dataframes = [pd.read_pickle(os.path.join(folder_path, file)) for file in pickle_files]

# Concatenate all dataframes into one
df = pd.concat(dataframes, ignore_index=True)

# Display the concatenated dataframe
print(df.shape)





#### fix model names

df ["model"] = df ["model"].apply (
    lambda x: x.replace ("GPT-3.5", "GPT-3.5-turbo-16k"))




#### scores


def func_json (x):
    
    try:
        
        x = gptagent.flex_json (x)
        
        if str (x)[0] == "[" and str (x)[-1] == "]":
            
            return x
        
        else:
            
            return "FAIL"
    
    except:
        
        return "FAIL"


df ["json"] = df ["raw_json"].apply (lambda x: func_json (x))

df ["len"] = df ["json"].apply (lambda x: len (x))



def check_answers (df_qa, row):
    
    if row ["json"] == "FAIL":
        
        return np.nan
        
    s = 0
    
    for item in row ["json"]:
        
        try:
            
            qst_num = int (item ["question"])
            
            if item ["answer"] == df_qa ["answer"].iloc [qst_num]:
                
               s = s + 1
                
        except:
            
            pass
            
    return s / row ["test_size"]
        
    
df ["score"] = df.apply (
    lambda row: check_answers (df_qa, row), axis = 1)






#### Scores


from scipy import stats

# Assuming df is your DataFrame
# Calculate the mean and count per group
grouped = df [df ["len"] == df ["test_size"]].groupby(['model', 'test_size'])['score']
mean_scores = grouped.mean()
counts = grouped.count()

# Calculate standard error of the mean (SEM)
std_devs = grouped.std()
sem = std_devs / counts**0.5

# Calculate the 95% confidence interval
confidence_interval = sem * stats.t.ppf((1 + 0.95) / 2., counts - 1)

# Create a DataFrame with the results
results_df = pd.DataFrame({
    'mean_score': mean_scores,
    'lower_bound': mean_scores - confidence_interval,
    'upper_bound': mean_scores + confidence_interval
}).reset_index()



# Format the mean score and CI into one string per entry, converting to percentage and keeping one decimal place
results_df['score_ci'] = results_df.apply(
    lambda row: f"{row['mean_score'] * 100:.1f} ({row['lower_bound'] * 100:.1f}-{row['upper_bound'] * 100:.1f})", 
    axis=1
)

# Pivot the table to have models as columns and test_size as rows
pivot_df = results_df.pivot(index='test_size', columns='model', values='score_ci').reset_index()

# Clean up the column names if needed
pivot_df.columns.name = None



# Function to replace 'nan' in strings
def replace_nan(val):
    if type(val) == str and 'nan' in val:
        return '-'
    return val

# Apply the function to each element in the DataFrame
pivot_df = pivot_df.applymap(replace_nan)

# For actual NaN values (not the string 'nan'), replace with '-'
pivot_df = pivot_df.fillna('-')

pivot_df.to_excel ("Accuracy.xlsx")




#### JSON FAIL


# Assuming df is your DataFrame
# Calculate the mean and count per group

df ["fail"] = df ["json"] == "FAIL"

grouped = df.groupby(['model', 'test_size'])['fail']
mean_scores = grouped.mean()
counts = grouped.count()

# Calculate standard error of the mean (SEM)
std_devs = grouped.std()
sem = std_devs / counts**0.5

# Calculate the 95% confidence interval
confidence_interval = sem * stats.t.ppf((1 + 0.95) / 2., counts - 1)

# Create a DataFrame with the results
results_df = pd.DataFrame({
    'mean_score': mean_scores,
    'lower_bound': mean_scores - confidence_interval,
    'upper_bound': mean_scores + confidence_interval
}).reset_index()



# Format the mean score and CI into one string per entry, converting to percentage and keeping one decimal place
results_df['score_ci'] = results_df.apply(
    lambda row: f"{row['mean_score'] * 100:.1f} ({row['lower_bound'] * 100:.1f}-{row['upper_bound'] * 100:.1f})", 
    axis=1
)

# Pivot the table to have models as columns and test_size as rows
pivot_df = results_df.pivot(index='test_size', columns='model', values='score_ci').reset_index()

# Clean up the column names if needed
pivot_df.columns.name = None


pivot_df.to_excel ("fail.xlsx")





# Function to replace 'nan' in strings
def replace_nan(val):
    if type(val) == str and 'nan' in val:
        return '-'
    return val

# Apply the function to each element in the DataFrame
pivot_df = pivot_df.applymap(replace_nan)

# For actual NaN values (not the string 'nan'), replace with '-'
pivot_df = pivot_df.fillna('-')





#### JSON OMISSION


# Assuming df is your DataFrame
# Calculate the mean and count per group

df ["omit"] = (df ["test_size"] != df ["len"]) & (df ["fail"] == False)

grouped = df.groupby(['model', 'test_size'])['omit']
mean_scores = grouped.mean()
counts = grouped.count()

# Calculate standard error of the mean (SEM)
std_devs = grouped.std()
sem = std_devs / counts**0.5

# Calculate the 95% confidence interval
confidence_interval = sem * stats.t.ppf((1 + 0.95) / 2., counts - 1)

# Create a DataFrame with the results
results_df = pd.DataFrame({
    'mean_score': mean_scores,
    'lower_bound': mean_scores - confidence_interval,
    'upper_bound': mean_scores + confidence_interval
}).reset_index()



# Format the mean score and CI into one string per entry, converting to percentage and keeping one decimal place
results_df['score_ci'] = results_df.apply(
    lambda row: f"{row['mean_score'] * 100:.1f} ({row['lower_bound'] * 100:.1f}-{row['upper_bound'] * 100:.1f})", 
    axis=1
)

# Pivot the table to have models as columns and test_size as rows
pivot_df = results_df.pivot(index='test_size', columns='model', values='score_ci').reset_index()

# Clean up the column names if needed
pivot_df.columns.name = None



# Function to replace 'nan' in strings
def replace_nan(val):
    if type(val) == str and 'nan' in val:
        return '-'
    return val

# Apply the function to each element in the DataFrame
pivot_df = pivot_df.applymap(replace_nan)

# For actual NaN values (not the string 'nan'), replace with '-'
pivot_df = pivot_df.fillna('-')


pivot_df.to_excel ("omit.xlsx")
