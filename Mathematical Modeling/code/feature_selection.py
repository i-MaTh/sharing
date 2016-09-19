# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 00:27:58 2016

@author: wjz
"""

from numpy import *
import numpy as np
import pandas as pd
import csv
from sklearn import cross_validation as cv
from sklearn import metrics
from sklearn.feature_selection import chi2, f_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import LinearSVC 
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.feature_selection import SelectFromModel

def selectBychi2(path):
    data = pd.read_csv(path)
    data_y = data.label
    data_x = data.drop('label', axis = 1)
    c, p = chi2(data_x, data_y)
    print p, sum(p<0.05),min(p)
    print c,sum(c<0.05),min(c)
    print type(p)
    
def selectByvariance(path):
    data = pd.read_csv(path)
    data_y = data.label
    data_x = data.drop('label', axis = 1)
    _var = np.var(data_x)
    print np.sort(_var)
    print sum(_var < 0.01)
    
def selectByf_classif(path, n):
    data = pd.read_csv(path)
    #print data.shape
    data_y = data.label
    data_x = data.drop('label', axis = 1)
    #print data_x.columns

    f, p = f_classif(data_x, data_y)
    p_df = pd.DataFrame(columns=data_x.columns.tolist())
    p_df.loc[0] = p
    df = p_df.T.sort(columns = 0, ascending = True)
    #df.head(n).to_csv('top_p.csv')
    #print df.head(10)
    return df.head(n).index.tolist()

def selectBylasso(path, col_list):
    data = pd.read_csv(path)
    data_y = data.label
    data_x = data.drop('label', axis = 1)
    data_x = data_x[col_list]
    
    llr = LogisticRegression(penalty="l1",C = 0.9, class_weight="balanced", max_iter = 500, random_state =9)\
        .fit(data_x, data_y)
        
    df = pd.DataFrame(columns=data_x.columns.tolist())
    df.loc[0] = llr.coef_[0]
    df.to_csv('sort_selectionBylasso.csv')
    '''
    model = SelectFromModel(llr, prefit=True)
    X_new = model.transform(data_x)
    print X_new
    print X_new.shape
    
    train_x, test_x, train_y, test_y =  cv.train_test_split(data_x, data_y, test_size = 0.2, \
		random_state = 33)
    lsvc = LinearSVC(C=0.01, penalty="l2", dual=False).fit(train_x, train_y)
    #llr = LogisticRegression(penalty="l1",C = 0.8, class_weight="balanced", max_iter = 500, random_state =9)\
    #    .fit(train_x, train_y)
    pred = lsvc.predict(test_x)
    acc = metrics.accuracy_score(test_y, pred)	#精确度指标
    print("Accuracy is %f" % acc)
    print sum(test_y == 1), sum(test_y == 0)
    '''
    
def selectByxgb(path):
	data = pd.read_csv(path + 'ohg_genotype.csv')
	f_name = pd.read_csv(path + 'lasso_selection.csv')
	data_y = data.label
	data_x = data.drop('label',axis=1)
	data_x = data_x[f_name.feature_name.tolist()]
	#print data_x.shape (1000, 470)
	train_x, test_x, train_y, test_y =  cv.train_test_split(data_x, data_y, test_size = 0.2, \
		random_state = 33)
	xg_train = xgb.DMatrix(train_x, label = train_y)
	xg_test = xgb.DMatrix(test_x, label = test_y)
	dtest = xgb.DMatrix(test_x)

	dtrain = xgb.DMatrix(data_x, label = data_y)
	# setup parameters for xgboost
	params = {'objective': 'binary:logistic', 'eval_metric':'auc', 'eta': 0.015, 'gamma':0.1}
	params['max_depth'] = 5
	params['subsample'] = 0.65
	params['colsample_bytree'] = 0.35
	params['seed'] = 1009
	params['nthread'] = 16
	params['labmda'] = 5

	#d = xgb.cv(params,dtrain,num_boost_round = 1500, nfold = 10, metrics='auc', seed=1009)#733
	#print d
	#watchlist = [ (xg_train, 'train'), (xg_test, 'test') ]
	num_round = 1300
	bst = xgb.train(params, xg_train, num_round)
 
	dic = bst.get_score(importance_type='gain')
	f_out = open('xgb_selection.csv', 'wb')
	writer = csv.writer(f_out)
	for field in dic:
		t = []
		t.append(field)
		t.append(dic[field])
		writer.writerow(t)
	f_out.close()
    
def select_top_k(path):
    data = pd.read_csv(path)
    data = data.sort(['gain_value'], ascending = False)
    data.head(430).to_csv('ans1.csv', index = None)
    return data.head(430).feature_name.tolist()

def evaluationBysvm(path, col_list):
    data = pd.read_csv(path)
    data_y = data.label
    data_x = data.drop('label',axis=1)
    data_x = data_x[col_list]
    print data_x.shape
    train_x, test_x, train_y, test_y =  cv.train_test_split(data_x, data_y, test_size = 0.2, \
		random_state = 63)
    lsvc = LinearSVC(C=0.01, penalty="l2", dual = False).fit(train_x, train_y)
    #llr = LogisticRegression(penalty="l1",C = 0.8, class_weight="balanced", max_iter = 500, random_state =9)\
    #    .fit(train_x, train_y)
    pred = lsvc.predict(test_x)
    acc = metrics.accuracy_score(test_y, pred)	#精确度指标
    print("Accuracy is %f" % acc)

def evaluate_gene_candidate(path):
    data_y = pd.read_csv(path + 'phenotype.txt', header = None)
    gene_cand = pd.read_csv(path + 'gene_candidate.csv').head(8)
    col_list = []
    for i in gene_cand['index']:
        col_list += pd.read_csv(path + 'gene_info/gene_%d.dat' % i, header = None)[0].tolist()
    
    genotype = pd.read_csv(path + 'genotype.csv')[col_list]
    col_name = genotype.columns.tolist()
    ohe_data = pd.get_dummies(genotype, prefix = col_name)
    ohe_data['label'] = data_y.values
    #data = ohe_data.reindex(np.random.permutation(ohe_data.index))    
    data_x = ohe_data.drop('label',axis=1)

    train_x, test_x, train_y, test_y =  cv.train_test_split(data_x, data_y, test_size = 0.2, \
		random_state = 53)
    lsvc = LinearSVC(C=0.01, penalty="l2", dual=False).fit(train_x, train_y)
    #llr = LogisticRegression(penalty="l1",C = 0.8, class_weight="balanced", max_iter = 500, random_state =9)\
    #    .fit(train_x, train_y)
    pred = lsvc.predict(test_x)
    acc = metrics.accuracy_score(test_y, pred)	#精确度指标
    print("Accuracy is %f" % acc)
    #return col_list


if __name__ == '__main__':
    import sys
    reload(sys)
    sys.setdefaultencoding('utf8')
    path = '/home/wjz/Desktop/math/'
    data = pd.read_csv(path + 'ohg_genotype.csv')
    #print data.shape
    #selectBychi2(path + 'ohg_genotype.csv')
    #selectByvariance(path + 'ohg_genotype.csv')
    col_list = selectByf_classif(path + 'm_ohe_genotype.csv', 20005)
    #col_list = pd.read_csv(path + 'lasso_selection.csv').feature_name.tolist()
    #selectBylasso(path + 'ohg_genotype.csv', col_list)
    #col_list = select_top_k(path + 'xgb_selection.csv')
    #evaluate_gene_candidate(path)
    #print col_list
    evaluationBysvm(path + 'm_ohe_genotype.csv', col_list)



















