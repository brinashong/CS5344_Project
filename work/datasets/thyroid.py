#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sys
import os
from fancyimpute import KNN

sys.path.append(os.path.abspath(".."))
from common import common


# In[2]:


base_path = common.base_path


# In[3]:


def get_thyroid_df():
    config = {
        'TARGET_COLUMN': 'target',
        
        # List of numerical columns (these are to be standardized)
        'NUMERICAL_COLUMNS': ['age', 'TT4', 'T3', 'T4U', 'FTI', 'TSH'],
        # List of categorical columns (these are to be one hot encoded)
        'CATEGORICAL_COLUMNS': [],
        # List of ordinal columns (these are to be label encoded)
        'ORDINAL_COLUMNS': [],
    }
    target_column = config['TARGET_COLUMN']
    all_df = pd.read_csv(f'{base_path}/datasources/thyroid/thyroidDF.csv')
    
    # Headers of column
    main_labels = all_df.columns
    
    print('Normal class: ', all_df[target_column].mode())
    return (all_df, main_labels, config)


# In[4]:


def fill_values(df, column, val1, val2):
    threshold = df[column].value_counts()[val1] / (df.shape[0] - df[column].isnull().sum())

    for i in df.index:
        if pd.isna(df.loc[i, column]) or pd.isnull(df.loc[i, column]):
            rand_num = np.random.rand()
            if rand_num > threshold:
                df.loc[i, column] = val2
            else:
                df.loc[i, column] = val1
    return df
    
def get_processed_thyroid_df():
    mapping = {'-':"Negative", 'A':'Hyperthyroid', 'AK':"Hyperthyroid",
               'B':"Hyperthyroid", 'C':"Hyperthyroid", 'C|I': 'Hyperthyroid', 
               'D':"Hyperthyroid", 'D|R':"Hyperthyroid", 'E': "Hypothyroid", 
               'F': "Hypothyroid", 'FK': "Hypothyroid", 'G': "Hypothyroid", 
               'GK': "Hypothyroid", 'GI': "Hypothyroid", 'GKJ': 'Hypothyroid', 'H|K': 'Hypothyroid'}
           
    all_df, main_labels, config = get_thyroid_df()
    # print('main_labels', main_labels)
    
    # Drop column
    all_df = all_df[all_df['target'].isin(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'AK', 'C|I', 'H|K', 'GK', 'FK', 'GI', 'GKJ', 'D|R', '-'])]
    all_df.drop(['TSH_measured','T3_measured','TT4_measured','T4U_measured','FTI_measured','TBG_measured','referral_source','patient_id'],axis=1 ,inplace=True)
    all_df = all_df.drop('TBG', axis=1)
    
    # Map and filter
    all_df['target'] = all_df['target'].map(mapping)
    all_df = all_df[all_df['age'] < 100]
    all_df = all_df.dropna(thresh=21)
    target_to_num = {
        'Negative': 0,
        'Hypothyroid':1,
        'Hyperthyroid':2,
    }
    all_df['target'] = all_df['target'].map(target_to_num)
    all_df['pregnant'] = all_df['pregnant'].replace({'t': 1, 'f': 0})
    
    # Fill values
    all_df = fill_values(all_df.copy(), 'sex', 'F', 'M')
    columns = ['sex', 'age', 'TT4', 'T3', 'T4U', 'FTI', 'TSH']
    fill_df = all_df.loc[:, columns]
    fill_df = fill_df.fillna(np.nan)
    sex_to_num = {
        'F':0,
        'M':1,
    }
    fill_df['sex'] = fill_df['sex'].map(sex_to_num)
    knn = KNN(k=13)
    knn_imputed_df = knn.fit_transform(fill_df)
    knn_imputed_df = pd.DataFrame(knn_imputed_df, index=fill_df.index)
    knn_imputed_df = knn_imputed_df.rename(columns=dict(zip(knn_imputed_df.columns,columns)))
    all_df.update(knn_imputed_df)

    # Create df to be used
    columns = ['age', 'TT4', 'T3', 'T4U', 'FTI', 'TSH', 'pregnant', 'target']
    model_df = all_df.loc[:, columns]

    config['INV_TARGET_DICT'] = target_to_num
    config['TARGET_DICT'] = {v: k for k, v in config['INV_TARGET_DICT'].items()}
    print('TARGET_DICT', config['TARGET_DICT'])
    
    config['NORMAL_TARGET'] = target_to_num['Negative']
    print('NORMAL_TARGET', config['NORMAL_TARGET'])
    
    main_labels = model_df.columns
    # print('main_labels', main_labels)
    
    return (model_df, main_labels, config)


# In[ ]:




