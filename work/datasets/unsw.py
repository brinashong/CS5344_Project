#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(".."))
from common import common


# In[2]:


base_path = common.base_path


# In[3]:


def get_unsw_df():
    config = {
        'TARGET_COLUMN': 'attack_cat',
        
        # List of numerical columns (these are to be standardized)
        'NUMERICAL_COLUMNS': ['dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sload', 'dload', 'sloss', 'dloss', 
                              'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 
                              'ackdat', 'smean', 'dmean', 'trans_depth', 'response_body_len', 
                              'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_ftp_cmd', 'ct_flw_http_mthd'],
        # List of categorical columns (these are to be one hot encoded)
        'CATEGORICAL_COLUMNS': ['proto', 'service', 'state'],
        # List of ordinal columns (these are to be label encoded)
        'ORDINAL_COLUMNS': [],
    }
    target_column = config['TARGET_COLUMN']
    all_df = pd.read_parquet(f'{base_path}/datasources/unsw/UNSW_NB15_training-set.parquet')
    
    # Headers of column
    main_labels = all_df.columns
    
    print('Normal class: ', all_df[target_column].mode())
    return (all_df, main_labels, config)


# In[4]:


def get_processed_unsw_df():
    all_df, main_labels, config = get_unsw_df()
    # print('main_labels', main_labels)
    target_column = config['TARGET_COLUMN']

    # Drop column
    all_df = all_df.drop(columns='label')

    # Fix wrong values
    all_df['is_ftp_login'] = np.where(all_df['is_ftp_login']>1, 1, all_df['is_ftp_login']) # Should be binary value
    all_df['attack_cat'] = all_df['attack_cat'].replace('backdoors','backdoor', regex=True) # Fix typo
    all_df['service'] = all_df['service'].apply(lambda x:"None" if x == "-" else x) # Remove "-" and replacing those with "None"
    
    # Label Encoder
    le, all_df = common.label_encode(all_df, [target_column])

    config['TARGET_DICT'] = {index: label for index, label in enumerate(le.classes_)}
    config['INV_TARGET_DICT'] = {v: k for k, v in config['TARGET_DICT'].items()}
    print('TARGET_DICT', config['TARGET_DICT'])
    
    config['NORMAL_TARGET'] = config['INV_TARGET_DICT']['Normal']
    print('NORMAL_TARGET', config['NORMAL_TARGET'])

    # One Hot Encoder
    ohe, all_df = common.one_hot_encode(all_df, config['CATEGORICAL_COLUMNS'])
    
    main_labels = all_df.columns
    print('main_labels', main_labels)
    
    return (all_df, main_labels, config)


# In[ ]:




