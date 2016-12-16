# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 15:18:33 2016

@author: Administrator
"""

import pandas as pd
import numpy as np

path = '.\\bytecup2016data\\'

train = pd.read_table(path + 'invited_info_train.txt', names = ['qid', 'uid', 'label'])
#print train.shape # (245752, 3)
validate = pd.read_csv(path + 'validate_nolabel.txt')
#print validate.shape # (30466, 3)
train_val = pd.concat([train, validate])
#print train_val.shape # (276218, 3)

unique_qid = train_val['qid'].unique()
#print unique_qid.shape #(7778L,)
unique_uid = train_val['uid'].unique()
#print unique_uid.shape #(27937L,)
unique_qid_uid = np.hstack((unique_qid, unique_uid))
#print unique_qid_uid.shape #(35715L,)

dic = {}
for index, item in enumerate(unique_qid_uid):
    dic[item] = '%d:1'%(index + 1)
    
#print len(dic) #(35715L,)
fm_train = train.replace(dic)
temp = pd.DataFrame()
temp['label'] = fm_train.label
temp['qid'] = fm_train.qid
temp['uid'] = fm_train.uid
temp.to_csv('fm_train.csv', index = None, sep = ' ', header = False)

fm_val = validate.replace(dic)
temp = pd.DataFrame()
temp['label'] = fm_val.label
temp['qid'] = fm_val.qid
temp['uid'] = fm_val.uid
temp.fillna(0).to_csv('fm_validate.csv', index = None, sep = ' ', header = False)










