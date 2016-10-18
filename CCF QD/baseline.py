# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 12:54:43 2016

@author: Administrator
"""

import pandas as pd
import numpy as np
from numpy import *
import xgboost as xgb

def preprocessing(data):
    record = []
    for item in data.YD_ID:
        temp = []
        for i in range(1, 6):
            for j in range(11-i):
                temp.append(item[j:j + i])
        record.append(temp)
    return record
    
def leak(path):
    train = pd.read_csv(path + 'train.csv', header = None, names = ['YD_ID', 'QD_Flag'], dtype = {'YD_ID':str})
    test = pd.read_csv(path + 'test.csv', header = None, names = ['YD_ID'], dtype = {'YD_ID':str})
    
    # 构造特征，处理训练集
    _train = preprocessing(train)
    # print len(_train[0])
    features = ['YD_ID'+str(i) for i in range(len(_train[0]))]
    train_x = pd.DataFrame(_train, columns = features, dtype = float)
    train_y = train.QD_Flag
    # 构造特征，处理测试集
    _test = preprocessing(test)
    test_x = pd.DataFrame(_test, columns = features, dtype = float)
	
    dtrain = xgb.DMatrix(train_x, label = train_y)
    dtest = xgb.DMatrix(test_x)

    # 使用xgboost训练模型
    param = {'objective':'binary:logistic', 'eta':0.025, 'max_depth':6, 'silent':1, 'eval_metric':'map'}
    param['scale_pos_weight'] = 3
    param['nthread'] = 16
    param['lambda'] = 4
    param['subsample'] = 0.8
    param['min_child_weight'] = 3
    param['seed'] = 109
    num_round = 812
    watchlist = [(dtrain, 'train')]    
    bst = xgb.train(param, dtrain, num_round, evals = watchlist)
    #xgb.cv(param, dtrain, num_round, nfold = 5, metrics = {'map'}, seed = 23, \
    #callbacks={xgb.callback.print_evaluation(show_stdv = True)})
    pred = bst.predict(dtest)
    test_result = pd.DataFrame(test['YD_ID'].tolist(),columns=["Idx"])
    test_result['score'] = pred
    # 输出结果
    rs = test_result.sort(columns = ['score'], ascending = False)
    rs['Idx'].to_csv('prediction.csv', index = None)

if __name__ == "__main__":
    path = 'F:\\competition\\'
    leak(path)














