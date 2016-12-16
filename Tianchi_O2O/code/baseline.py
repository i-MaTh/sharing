# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 20:46:23 2016
@author: i-MaTh
Description:CCF-O2O

"""

import pandas as pd
import numpy as np
import xgboost as xgb

seed = 1024
np.random.seed(seed)


if __name__ == '__main__':
    path = '.\\CCF-O2O\\'
    train1 = pd.read_csv(path + 'train_fea_0515.csv')
    train2 = pd.read_csv(path + 'train_fea_0630.csv')
    train = pd.concat([train1, train2])
    features = ['distance', 'minus', 'upper', 'day', 'positive15']
    train_numerical = train[features]
    train_category = pd.get_dummies(train[['l_discount', 'l_minus']].fillna('other'))
    train = pd.concat([train_category, train_numerical], axis = 1)
    
    train_y = train.positive15
    train_x = train.drop('positive15', axis = 1).fillna(-1) # shape:(390680, 10)
    
    test = pd.read_csv(path + 'test_fea_0815.csv')
    features = ['distance', 'minus', 'upper', 'day']
    test_numerical = test[features]
    test_category = pd.get_dummies(test[['l_discount', 'l_minus']].fillna('other'))
    test_x = pd.concat([test_category, test_numerical], axis = 1) .fillna(-1) # shape:(113640, 10)
    
    xgboost = xgb.XGBClassifier(
            n_estimators = 1009,
            learning_rate = 0.03,
            max_depth = 6,
            subsample = 0.7,
            colsample_bytree = 0.8,
            reg_lambda = 3,
            seed = seed,
            scale_pos_weight =  2,
            )
    
    xgboost.fit(
        train_x,
        train_y,
        eval_metric = 'auc',
        eval_set = [(train_x, train_y)],
        early_stopping_rounds = 100,
        )
    
    y_preds = xgboost.predict_proba(test_x)
    
    print y_preds[:,0]
    submission = pd.DataFrame()
    submission['User_id'] = test.user
    submission['Coupon_id'] = test.coupon
    submission['Date_received'] = test.date
    submission['Probability'] = y_preds[:,0]
    
    submission.to_csv('sub.csv', index = None)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

