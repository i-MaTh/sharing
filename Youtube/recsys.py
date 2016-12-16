# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 17:10:11 2016

@author: i-MaTh
"""

import pandas as pd
import numpy as np

train_set = pd.read_csv('train_data.csv', dtype = str)
#print train.columns
'''
print train_set.iloc[5].tolist()
for i in range(20): 
    t = str(train_set.iloc[5][str(i+1)])
    if t != 'nan':
        print train_set[train_set.video_id == t].category
'''

test_set = pd.read_csv('test_data.csv', dtype = str).fillna('NULL')
'''
#print test_set.columns #['source', 'target', 'edge_id']
t_v_id = train_set['video_id'].tolist()
print "generate source video category..."
s_category = []
for s in test_set['source']:
    if s in t_v_id:
        s_category.append(train_set[train_set['video_id'] == s].category.tolist()[0])
    else:
        s_category.append('null')
print "generate target video category..."
t_category = []     
for t in test_set['target']:
    if (t != 'NULL') and (t in t_v_id):
        t_category.append(train_set[train_set['video_id'] == t].category.tolist()[0])
    else:
        t_category.append('null')


test_set['s_category'] = s_category
test_set['t_category'] = t_category
test_set.to_csv('tmp_test.csv', index = None)

#print test_set[test_set['target'].isnull() == True].index.tolist()
'''
tmp_test = pd.read_csv('tmp_test.csv')
label  = (tmp_test.s_category == tmp_test.t_category).tolist()


comm = pd.read_csv('solutionn1.csv')
comm['common'] = comm['common'] / max(comm['common'])

preds = []
candidates = test_set[test_set['target'] == 'NULL'].index.tolist()

for idx, t in enumerate(comm['common']):
    if (idx in candidates) or label[idx]:
        preds.append(1.0)
    else:
        preds.append(t)



submission = pd.DataFrame()
submission['edge_id'] = comm.edge_id
submission['edge_present'] = preds
submission.to_csv('sub1.csv', index = None)
#print comm






'''
sub =pd.read_csv('.//random_solution-2.csv')
target = np.ones(sub.shape[0], dtype = int)
sub['edge_present'] = target
sub.to_csv('random_solution-3.csv', index = None)
print sub.head(5)
'''