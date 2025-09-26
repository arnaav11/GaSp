import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler   #to split data into x and y x is numerical and y is target value????
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout  
df=pd.read_csv('databases/LC_loans_granting_model_dataset.csv')
df=pd.read_csv('databases/train_lending_club.csv')
df=pd.read_csv('test/databases/sentiment_analysis_gasp.csv')
