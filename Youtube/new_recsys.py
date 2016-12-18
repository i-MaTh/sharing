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
    
def toy_test(train, test):
    cols_target = ['video_id','1','2','3','4','5','6','7','8','9',\
        '10','11','12','13','14','15','16','17','18','19','20']
    train = train[cols_target]
    train['o_degree'] = 20 - (train<0).sum(axis=1)
    train.rename(columns = {'video_id':'source'}, inplace=True)
    print test.shape        
    
    sub = test.merge(train, how='inner', on='source')
    sub = sub[['edge_id', 'o_degree']].astype(int)
    sub = sub.sort(columns = ['edge_id'], ascending=True)
    sub['proba'] = sub['o_degree'] / float(max(sub['o_degree']))
    
    submission = pd.DataFrame()
    submission['edge_id'] = sub.edge_id
    submission['edge_present'] = 1-sub.proba
    submission.to_csv('toy_test3.csv', index=None)
    
def ensemble():
    test_set = pd.read_csv('test_data.csv', dtype=str)
    test_set.fillna(-1, inplace=True)
    in_proba = pd.read_csv('non_out_degree.csv', dtype=str)
    co_cate = pd.read_csv('gby_category.csv', dtype=str)
    co_up = pd.read_csv('gby_uploader.csv', dtype=str)
    test = test_set.merge(in_proba, how='inner', on='edge_id')
    test = test.merge(co_cate, how='inner', on='edge_id')
    test = test.merge(co_up, how='inner', on='edge_id')
    
    # first revise
    labels = []    
    for i in range(test.shape[0]):
        flag = (test['co_category'][i] == '1') and (test['co_uploader'][i] == '1')
        if flag:
            labels.append(1)
        else:
            labels.append(test['in_proba'][i])
    
    # revise the probabilities
    test['in_proba'] = labels


    sub_test = test.groupby('source')
    c = 0
    df = pd.DataFrame()
    for key, values in sub_test:
        values['revise'] = (values['co_category'] == '1') & (values['co_uploader'] == '1')
        df = pd.concat([df, values])
        c += 1
        print "processing at %d" % c        
        
    df.to_csv('level_1.csv', index=None)  

    
    
def level2():
    test = pd.read_csv('level_1.csv')
    df = pd.DataFrame()
    c = 0
    labels = []
    for key, values in test.groupby('source'):
        m = values.shape[0]
        values.reset_index(inplace=True) #reset index of sub_dataframe
        for i in range(m):
            if values['revise'].sum() >= (values['in_proba'][0])*20:
                if values['revise'][i]:
                    labels.append(1)
                else :
                    labels.append(0)
            else:
                labels.append(values['in_proba'][0])
    
        df = pd.concat([df, values])
        c += 1
        print "processing at %d" % c
    df['edge_present'] = labels
    df = df.sort(columns = ['edge_id'], ascending=True)
    df[['edge_id', 'edge_present']].to_csv('revise_o_degree_1.csv', index=None)            
    
def level3():
    test = pd.read_csv('level_1.csv')
    df = pd.DataFrame()
    c = 0
    labels = []
    for key, values in test.groupby('source'):
        m = values.shape[0]
        values.reset_index(inplace=True) #reset index of sub_dataframe
        c_n = 0
        if values['in_proba'][0] != 0.0:
            for i in range(m):
                if values['revise'].sum() >= (values['in_proba'][0])*20:
                    if values['revise'][i]:
                        labels.append(1)
                    else :
                        labels.append(0)
                else:
                    if values['co_uploader'][i] == 1:
                        labels.append(1)
                        c_n += 1
                    else:
                        labels.append(values['in_proba'][0] - float(c_n)/20)
                        #labels.append(0)
        else:
            for i in range(m):
                labels.append(0.0)
                
        df = pd.concat([df, values])
        c += 1
        print "processing at %d" % c
    df['edge_present'] = labels
    df = df.sort(columns = ['edge_id'], ascending=True)
    df[['edge_id', 'edge_present']].to_csv('revise_o_degree_10.csv', index=None) 

def target_filter():
    test = pd.read_csv('test_data.csv')
    test.fillna(-1, inplace=True)
    ans = pd.read_csv('revise_o_degree_10.csv')
    labels = []
    for i in range(test.shape[0]):
        if test['target'][i] == -1:
            labels.append(1)
        else:
            labels.append(ans['edge_present'][i])
            print "processing at %d" % i
            
    ans['edge_present'] = labels
    ans.to_csv('revise_o_degree_11.csv', index=None)
    
    
if __name__ == '__main__':
    #start = datetime.now()
    #train_set = pd.read_csv('train_data.csv', dtype = str)
    #train_set.fillna(-1, inplace=True)
    #print train_set.shape #(26901, 29)
    #test_set = pd.read_csv('test_data.csv', dtype = str)
    #test_set.fillna(-1, inplace=True)
    #print test_set.shape #(100000, 3)
    #users_set = pd.read_csv('users_metadata.csv')
    #print users_set.shape #(1062324, 4)
    #videos_set = pd.read_csv('videos_metadata.csv')
    #print videos_set.shape #(197317, 3)
    #m_train = pd.read_csv('merge_train.csv')    
    
    # toy testing
    #toy_test(train_set, test_set)
    
    #ensemble()
    #level2()
    #level3()    
    target_filter()
    
    '''
    d1 = pd.read_csv('gby_uploader.csv')
    d2 = pd.read_csv('revise_o_degree_1.csv')
    submission = pd.DataFrame()
    submission['edge_id'] = d1.edge_id
    submission['edge_present'] = 0.5*d1.co_uploader + 0.5*d2.edge_present
    submission.to_csv('revise_o_degree_5.csv', index=None)
    '''
    
    
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
    '''
    # Counting number of videos in train set
    gby_category = dic_read('dic_category.csv')
    tr_videos = set()
    for key in gby_category:
        for v in gby_category[key]:
            if v != '-1':
                tr_videos.add(v)
    tmp = pd.DataFrame()
    tmp['video_id'] = list(tr_videos)        
    tmp.to_csv('tr_video_id.csv', index = None) #218340
    '''
    
    # descriptions of raw data
    '''
    #v_meta = pd.read_csv('videos_metadata.csv', dtype = str)
    #vm = v_meta['video_id'].unique() #(197317,)
    #print vm.shape 
    tr_v = train_set['video_id'].unique()
    print tr_v.shape
    #v1 = pd.read_csv('tr_video_id.csv', dtype = str)    
    #v1 = v1['video_id'].unique() #(218340,)
    #print v1.shape
    v2 = test_set['source'].unique() #(25602,)
    v3 = test_set['target'].unique() #(43049,)
    print v2.shape
    print v3.shape
    # number of source in tr_v is 25602
    # number of target in tr_v is 12404    
    
    # number of source in v1 is 25601
    # number of target in v1 is 26940

    # number of source in vm is 3067
    # number of target in vm is 4288 

    c = 0
    for v in v3:
        if v in tr_v:
            c += 1
    print c 
    '''
    
    
    '''
    gby_uploader = dic_read('dic_category.csv')
    labels = []
    for i in range(test_set.shape[0]):
        for key in gby_uploader:
            if test_set['target'][i] == -1 or \
            (test_set['source'][i] in gby_uploader[key] and test_set['target'][i] in gby_uploader[key]):
                label = 1
				break
            else:
                label = 0
		labels.append(label)

	epoch_end = datetime.now()
	duration = epoch_end - start
	print "time: %s" % duratution
    submission = pd.DataFrame()
    submission['edge_id'] = test_set.edge_id
    submission['edge_present'] = labels
    submission.to_csv('sub1804.csv', index = None)
	'''


