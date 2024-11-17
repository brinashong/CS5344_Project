#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(".."))
from common import common


# In[38]:


base_path = common.base_path


# In[39]:


def get_cirrhosis_df():
    config = {
        'TARGET_COLUMN': 'Status',
        # List of numerical columns (these are to be standardized)
        'NUMERICAL_COLUMNS': ['N_Days', 'Age', 'Bilirubin', 'Cholesterol', 'Albumin', 'Copper','Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin'],
        # List of categorical columns (these are to be one hot encoded)
        'CATEGORICAL_COLUMNS': ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema'],
        # List of ordinal columns (these are to be label encoded)
        'ORDINAL_COLUMNS': [],
    }
    target_column = config['TARGET_COLUMN']
    all_df = pd.read_csv(f'{base_path}/datasources/cirrhosis/train.csv')
    
    # Headers of column
    main_labels = all_df.columns
    
    print('Normal class: ', all_df[target_column].mode())
    return (all_df, main_labels, config)


# In[40]:


def get_processed_cirrhosis_df():
    all_df, main_labels, config = get_cirrhosis_df()
    # print('main_labels', main_labels)
    target_column = config['TARGET_COLUMN']

    # Preprocess
    all_df = all_df.drop(['id'], axis = 1)
    columns_to_fill = ['Cholesterol', 'Copper', 'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin', 'Stage']
    all_df[columns_to_fill] = all_df[columns_to_fill].fillna(all_df[columns_to_fill].median())
    col_to_fill = ['Drug','Ascites','Hepatomegaly','Spiders']     
    all_df[col_to_fill] = all_df[col_to_fill].fillna('unknown')
    
    # Label Encoder
    le, all_df = common.label_encode(all_df, [target_column])
    
    config['TARGET_DICT'] = {index: label for index, label in enumerate(le.classes_)}
    config['INV_TARGET_DICT'] = {v: k for k, v in config['TARGET_DICT'].items()}
    print('TARGET_DICT', config['TARGET_DICT'])
    le, all_df = common.label_encode(all_df, config['ORDINAL_COLUMNS'])
    
    config['NORMAL_TARGET'] = config['INV_TARGET_DICT']['C']
    print('NORMAL_TARGET', config['NORMAL_TARGET']) # status of the patient C (censored), CL (censored due to liver tx), or D (death)

    # One Hot Encoder
    ohe, all_df = common.one_hot_encode(all_df, config['CATEGORICAL_COLUMNS'])
    
    main_labels = list(all_df.columns)
    print('main_labels', main_labels)
    
    return (all_df, main_labels, config)


# In[ ]:




