# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 19:00:00 2016
Description:Mathematical Modeling(B)
@author: wjz

"""

import numpy as np
import pandas as pd
import csv
from sklearn import cross_validation as cv
from sklearn import metrics
import xgboost as xgb
import matplotlib.pyplot as plt

def dat2csv(path):
	f_in = open(path, 'r')
	contents = f_in.readlines()
	f_in.close()
	f_out = open('genotype.csv', 'wb')
	writer = csv.writer(f_out)
	for field in contents:
		field = field.strip('\n').split(' ')
		writer.writerow(field)
	f_out.close()

def convert(path):
	data = pd.read_csv(path)
	for col in data.columns:
		uni = data[col].unique() #获得每一列的唯一元素
		data[col] = data[col].replace(uni[0], 0).replace(uni[1], 1).replace(uni[2], 2)
	data.to_csv('convert_genotype.csv', index = None)

def one_hot_encoding(path):
	data = pd.read_csv(path)
	col_name = data.columns.tolist()
	ohe = pd.get_dummies(data, prefix = col_name)
	ohe.to_csv('one_hot_genotype.csv', index = None)

def merge_X_Y(path):
	data_x = pd.read_csv(path + 'one_hot_genotype.csv')
	data_y = pd.read_csv(path + 'phenotype.txt', header = None)
	data_x['label'] = data_y.values
	data_x.to_csv('one_hot_genotype.csv', index = None)

def shuffler(path):
	#shuffle data
	df = pd.read_csv(path + 'one_hot_genotype.csv')
	df.reindex(np.random.permutation(df.index)).to_csv('ohg_genotype.csv', index = None)

def multi_phenos2label(path):
	data = pd.read_csv(path + 'multi_phenos.txt')
	uni = data['label'].unique() #获得每一列的唯一元素
	data['label'] = data['label'].replace(uni, range(len(uni)))
	data.to_csv('multi_label.csv', index = None)

def training_xgb(path):
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
	#get prediction

	dic = bst.get_score(importance_type='gain')
	f_out = open('xgb_selection.csv', 'wb')
	writer = csv.writer(f_out)
	for field in dic:
		t = []
		t.append(field)
		t.append(dic[field])
		writer.writerow(t)
	f_out.close()
	#xgb.plot_importance(bst, height=0.3, importance_type='gain') #可视化出前20个importance feature,
	#plt.show()
	#pred = bst.predict(dtest)


if __name__ == '__main__':
	import sys
	reload(sys)
	sys.setdefaultencoding('utf8')
	path = '/home/wjz/Desktop/math/'
	#dat2csv(path + 'genotype.dat')
	#convert(path + 'genotype.csv')
	#one_hot_encoding(path + 'genotype.csv')
	#merge_X_Y(path)
	#shuffler(path)
	#training_xgb(path)
	#multi_phenos2label(path)
	data = pd.read_csv(path + 'one_hot_genotype.csv')
	data_y = pd.read_csv(path + 'multi_label.csv')
	data['label'] = data_y.label
	data.reindex(np.random.permutation(data.index)).to_csv('m_ohe_genotype.csv', index = None)
	print data_x.shape
	















