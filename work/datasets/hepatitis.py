#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(".."))
from common import common


# In[3]:


base_path = common.base_path


# In[4]:


def get_hepatitis_df():
    config = {
        'TARGET_COLUMN': 'Category',
        
        # List of numerical columns (these are to be standardized)
        'NUMERICAL_COLUMNS': ['Age', 'ALB', 'ALP', 'ALT', 'AST','BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT'],
        # List of categorical columns (these are to be one hot encoded)
        'CATEGORICAL_COLUMNS': ['Sex'],
        # List of ordinal columns (these are to be label encoded)
        'ORDINAL_COLUMNS': [],
    }
    target_column = config['TARGET_COLUMN']
    all_df = pd.read_csv(f'{base_path}/datasources/hepatitis/HepatitisCdata.csv')
    
    # Headers of column
    main_labels = all_df.columns
    
    print('Normal class: ', all_df[target_column].mode())
    return (all_df, main_labels, config)


# In[5]:


def get_processed_hepatitis_df():
    all_df, main_labels, config = get_hepatitis_df()
    # print('main_labels', main_labels)
    target_column = config['TARGET_COLUMN']

    # Drop column
    all_df.drop(['Unnamed: 0'],axis=1,inplace=True)

    # Fix missing values
    numerical_columns = config['NUMERICAL_COLUMNS']
    median = all_df[numerical_columns].median()
    all_df[numerical_columns] = all_df[numerical_columns].fillna(median)
    
    # Label Encoder
    le, all_df = common.label_encode(all_df, [target_column])

    config['TARGET_DICT'] = {index: label for index, label in enumerate(le.classes_)}
    config['INV_TARGET_DICT'] = {v: k for k, v in config['TARGET_DICT'].items()}
    print('TARGET_DICT', config['TARGET_DICT'])
    
    config['NORMAL_TARGET'] = config['INV_TARGET_DICT']['0=Blood Donor']
    print('NORMAL_TARGET', config['NORMAL_TARGET'])

    # One Hot Encoder
    ohe, all_df = common.one_hot_encode(all_df, config['CATEGORICAL_COLUMNS'])
    
    main_labels = all_df.columns
    print('main_labels', main_labels)
    
    return (all_df, main_labels, config)


# In[ ]:




