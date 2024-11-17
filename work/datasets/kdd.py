#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(".."))
from common import common


# In[10]:


base_path = common.base_path


# In[11]:


def get_kdd_df():
    config = {
        'TARGET_COLUMN': 'attack',
    
        # List of numerical columns (these are to be standardized)
        'NUMERICAL_COLUMNS': ['duration', 'src_bytes', 'dst_bytes',
                             'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'num_compromised', 
                             'root_shell', 'su_attempted', 'num_file_creations', 'num_shells', 'num_access_files', 
                             'num_outbound_cmds', 'count', 'srv_count', 'serror_rate', 
                             'rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 
                             'dst_host_srv_count', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate'],
        # List of categorical columns (these are to be one hot encoded)
        'CATEGORICAL_COLUMNS': ['protocol_type', 'service', 'flag'],
        # List of ordinal columns (these are to be label encoded)
        'ORDINAL_COLUMNS': [],
    }
    target_column = config['TARGET_COLUMN']
    
    cols = open(f"{base_path}/datasources/kdd/kddcup.names",'r').read()
    cols = [c[:c.index(':')] for c in cols.split('\n')[1:-1]]
    cols.append('attack')
    
    all_df = pd.read_csv(f"{base_path}/datasources/kdd/corrected", names = cols)
    
    main_labels = cols
    
    print('Normal class: ', all_df[target_column].mode())
    return (all_df, main_labels, config)


# In[27]:


def get_processed_kdd_df():
    all_df, main_labels, config = get_kdd_df()
    # print('main_labels', main_labels)
    target_column = config['TARGET_COLUMN']
    
    # Label Encoder
    all_df[target_column] = all_df[target_column].str[:-1]
    le, all_df = common.label_encode(all_df, [target_column])
    
    config['TARGET_DICT'] = {index: label for index, label in enumerate(le.classes_)}
    config['INV_TARGET_DICT'] = {v: k for k, v in config['TARGET_DICT'].items()}
    print('TARGET_DICT', config['TARGET_DICT'])
    
    config['NORMAL_TARGET'] = config['INV_TARGET_DICT']['normal']
    print('NORMAL_TARGET', config['NORMAL_TARGET'])

    # One Hot Encoder
    ohe, all_df = common.one_hot_encode(all_df, config['CATEGORICAL_COLUMNS'])
    
    main_labels = list(all_df.columns)
    print('main_labels', main_labels)
    
    return (all_df, main_labels, config)


# In[ ]:




