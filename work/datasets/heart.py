#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(".."))
from common import common


# In[2]:


base_path = common.base_path


# In[3]:


def get_heart_df():
    config = {
        'NORMAL_TARGET': 0,
        'TARGET_COLUMN': 'num',
        'TARGET_DICT': {
            0: 'Absense',
            1: 'Slight Presence',
            2: 'Presence',
            3: 'Moderate Presence',
            4: 'High Presence'
        },
        'INV_TARGET_DICT': {
            'Absense': 0,
            'Slight Presence': 1,
            'Presence': 2,
            'Moderate Presence': 3,
            'High Presence': 4
        },
        
        # List of numerical columns (these are to be standardized)
        'NUMERICAL_COLUMNS': ['age', 'trestbps', 'chol', 'thalch', 'oldpeak'],
        # List of categorical columns (these are to be one hot encoded)
        'CATEGORICAL_COLUMNS': ['sex','cp', 'fbs', 'restecg', 'exang'],
        # List of ordinal columns (these are to be label encoded)
        'ORDINAL_COLUMNS': [],
    }
    target_column = config['TARGET_COLUMN']
    all_df = pd.read_csv(f'{base_path}/datasources/heart/heart_disease_uci.csv')
    
    # Headers of column
    main_labels = all_df.columns
    
    print('Normal class: ', all_df[target_column].mode())
    return (all_df, main_labels, config)


# In[4]:


def get_processed_heart_df():
    all_df, main_labels, config = get_heart_df()
    # print('main_labels', main_labels)

    # Drop columns
    all_df = all_df.drop(['id','dataset'], axis=1)
    
    # Fill values
    chol_median = all_df.loc[all_df['chol'] != 0, 'chol'].median()
    all_df = all_df.fillna(value={'chol': chol_median})
    all_df.loc[all_df['chol'] == 0, 'chol'] = chol_median 

    mean_peak = all_df.oldpeak.mean()
    all_df = all_df.fillna(value={'oldpeak': mean_peak})
    all_df.loc[all_df['oldpeak'] == 0, 'oldpeak'] = mean_peak
    
    mean_bp = all_df.loc[all_df['trestbps'] != 0, 'trestbps'].mean()
    all_df = all_df.fillna(value={'trestbps': mean_bp})
    all_df.loc[all_df['trestbps'] == 0, 'trestbps'] = mean_bp

    mean_hr = all_df.loc[all_df['thalch'] != 0, 'thalch'].mean()
    all_df = all_df.fillna(value={'thalch': mean_hr})
    all_df.loc[all_df['thalch'] == 0, 'thalch'] = mean_hr

    all_df.drop(labels=['ca','thal','slope'], axis=1, inplace=True)
    all_df = all_df.astype({'sex':'category', 'cp':'category', 'fbs':'bool', 'restecg':'category', 'exang':'bool'})
    all_df.dropna(inplace=True)

    # One Hot Encoder
    ohe, all_df = common.one_hot_encode(all_df, config['CATEGORICAL_COLUMNS'])

    main_labels = all_df.columns
    # print('main_labels', main_labels)
    
    return (all_df, main_labels, config)


# In[ ]:




