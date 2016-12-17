# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 17:10:11 2016
@author: i-MaTh
"""

import pandas as pd
import numpy as np

def merger_train_users(train_set, users_set):
    ### combine train_set and users_set ### 
    train_set['n_null'] = (train_set<0).sum(axis=1)
    train_set = train_set[train_set.n_null < train_set.shape[1] - 2]
    #print train_set.shape #(26789, 30)
    train_set = train_set.merge(users_set, how='inner', on='uploader')
    train_set.to_csv('merge_train.csv', index = None)

def dic_write(dic, _file):
    with open(_file, 'wb') as outfile:
        for key in dic:
            data = []
            data.append(key)
            _str = '{d[0]},'
            for i in range(len(dic[key])):    
                _str += '{d[%d]}'%(i+1)
                if i < len(dic[key])-1:
                    _str +=','
                data.append(dic[key][i])
            
            _str += '\n'
            outfile.write(_str.format(d = data))    
    outfile.close()    

def dic_read(_file):
    data = {}
    with open(_file, 'r') as infile:
        for line in infile:
            ins = line.strip().split(',')
            data[ins[0]] = ins[1:]
           
    return data
    
if __name__ == '__main__':
    #train_set = pd.read_csv('train_data.csv', dtype = str)
    #train_set.fillna(-1, inplace=True)
    #print train_set.shape #(26901, 29)

    test_set = pd.read_csv('test_data.csv', dtype = str)
    test_set.fillna(-1, inplace=True)
    #print test_set.shape #(100000, 3)
    #users_set = pd.read_csv('users_metadata.csv')
    #print users_set.shape #(1062324, 4)
    #videos_set = pd.read_csv('videos_metadata.csv')
    #print videos_set.shape #(197317, 3)
    
    m_train = pd.read_csv('merge_train.csv')
    
    '''
    sub_features = ['video_id','1','2','3','4','5','6','7','8','9',\
        '10','11','12','13','14','15','16','17','18','19','20']
    
    dic_uploader = {}
    for key, member in m_train.groupby('uploader'):
        tmp = member[sub_features].values
        (m,n) = tmp.shape
        dic_uploader[key] = tmp.reshape(m*n)

    dic_write(dic_uploader, 'dic_uploader.csv')

    dic_category = {}
    for key, member in m_train.groupby('category'):
        tmp = member[sub_features].values
        (m,n) = tmp.shape
        dic_category[key] = tmp.reshape(m*n)

    dic_write(dic_category, 'dic_category1.csv')
    '''
    

    gby_uploader = dic_read('dic_category.csv')
    labels = []
    for i in range(test_set.shape[0]):
        for key in gby_uploader:
            if test_set['target'][i] == -1 or \
            (test_set['source'][i] in gby_uploader[key] and test_set['target'][i] in gby_uploader[key]):
                labels.append(1)
            else:
                labels.append(0)

    submission = pd.DataFrame()
    submission['edge_id'] = test_set.edge_id
    submission['edge_present'] = labels
    submission.to_csv('sub426.csv', index = None)



