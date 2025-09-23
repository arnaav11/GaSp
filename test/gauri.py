import pandas as pd
import tensorflow as tf
import numpy as np
from tf.keras.preprocessing.text.Tokenizer import Tokenizer
from tf.keras.layers.Embedding import Embedding
from tf.keras.layers.Normalization import Normalization
from tf.keras.layers.Dense import Dense
from tf.keras.layers.LSTM import LSTM
from tf.keras.layers.Concatenate import Concatenate
import os

# merging the client profiles from mock portfolios
# list directory method python

folder_path = "test/databases/mock_portfolios copy"     # Path to the folder containing CSV filesn

contents = os.listdir(folder_path)
print("Current directory contents:", contents)

profiles= []
for folder in contents:
    try:
        
        df = pd.read_csv(os.path.join(folder_path, folder, "client_profile.csv"))
        df = df.drop(columns=["address"])
        profiles.append(df)
       
        print(df.head())
    except Exception as e:
        print(f"Error reading {folder}: {e}")
    
merged_df = pd.concat(profiles, ignore_index=True)
print(merged_df.head())
merged_df.to_csv(os.path.join(folder_path, 'all_profiles.csv'), index=False)

#merging bank_statement.csv of every file with a unique client id
statements = []
for folder in contents:
    try:
        df = pd.read_csv(os.path.join(folder_path, folder, "bank_statement.csv"))
        id = pd.read_csv(os.path.join(folder_path, folder, "client_profile.csv"))
        
        df["client_id"] = int(id["client_id"])
        statements.append(df)
        print(df.head())    
    except Exception as e:
        pass
    
merged_df = pd.concat(statements, ignore_index=True)
print(merged_df.head())
merged_df.to_csv(os.path.join(folder_path, 'all_statements.csv'), index=False)  

# merged dep_report of every file witgh a unique client id
debt_reports = []
for folder in contents:
    try:
        df = pd.read_csv(os.path.join(folder_path, folder, "debt_report.csv"))
        id = pd.read_csv(os.path.join(folder_path, folder, "client_profile.csv"))
        
        df["client_id"] = int(id["client_id"])
        debt_reports.append(df)
        print(df.head())    
    except Exception as e:
        pass    
merged_df = pd.concat(debt_reports, ignore_index=True)
print(merged_df.head()) 
merged_df.to_csv(os.path.join(folder_path, 'all_debt_reports.csv'), index=False)  







  
        
 

###### 
# sentiment_df = pd.read_csv('test/databases/sentiment_analysis_gasp.csv', encoding='ISO-8859-1')


# print(sentiment_df.head())  

# train







# to train the model? creating the output table i have the input table which if all profiles right now







