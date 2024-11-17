#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(".."))
from common import common


# In[3]:


base_path = common.base_path


# In[4]:


def get_cicids_df():
    config = {
        'TARGET_COLUMN': 'ClassLabel',
        
        # List of numerical columns (these are to be standardized)
        'NUMERICAL_COLUMNS': ['Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
               'Fwd Packets Length Total', 'Bwd Packets Length Total',
               'Fwd Packet Length Max', 'Fwd Packet Length Mean',
               'Fwd Packet Length Std', 'Bwd Packet Length Max',
               'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s',
               'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
               'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std',
               'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean',
               'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags',
               'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s',
               'Bwd Packets/s', 'Packet Length Max', 'Packet Length Mean',
               'Packet Length Std', 'Packet Length Variance', 'Avg Packet Size', 'Avg Fwd Segment Size',
               'Avg Bwd Segment Size', 'Subflow Fwd Packets', 'Subflow Fwd Bytes',
               'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'Init Fwd Win Bytes',
               'Init Bwd Win Bytes', 'Fwd Act Data Packets', 'Fwd Seg Size Min',
               'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean',
               'Idle Std', 'Idle Max', 'Idle Min'],
        # List of categorical columns (these are to be one hot encoded)
        'CATEGORICAL_COLUMNS': [],
        # List of ordinal columns (these are to be label encoded)
        'ORDINAL_COLUMNS': [],
    }
    target_column = config['TARGET_COLUMN']
    all_df = pd.read_csv(f'{base_path}/datasources/cicids/cic-collection-sample.csv')
    
    # Headers of column
    main_labels = all_df.columns
    
    print('Normal class: ', all_df[target_column].mode())
    return (all_df, main_labels, config)


# In[5]:


def get_processed_cicids_df():
    all_df, main_labels, config = get_cicids_df()
    # print('main_labels', main_labels)
    target_column = config['TARGET_COLUMN']

    # Drop columns
    all_df = all_df.drop(columns='Label')
    all_df = all_df.loc[:, ~all_df.columns.str.contains('^Unnamed')]
    
    # Label Encoder
    le, all_df = common.label_encode(all_df, [target_column])
    
    config['TARGET_DICT'] = {index: label for index, label in enumerate(le.classes_)}
    config['INV_TARGET_DICT'] = {v: k for k, v in config['TARGET_DICT'].items()}
    print('TARGET_DICT', config['TARGET_DICT'])
    
    config['NORMAL_TARGET'] = config['INV_TARGET_DICT']['Benign']
    print('NORMAL_TARGET', config['NORMAL_TARGET'])

    main_labels = all_df.columns
    print('main_labels', main_labels)
    
    return (all_df, main_labels, config)


# In[ ]:




