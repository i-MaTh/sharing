# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 19:00:00 2016

@author: wjz
"""

import numpy as np
import pandas as pd
import csv
from collections import Counter

def cal_pQAndpU(path):
	# 分别计算每个专家回答问题的概率和每个问题被专家回答的概率
	#data_1[data_1['qid'].isin(data_2['qid'])] #isin()函数可方便地过滤DataFrame中的数据
	data = pd.read_csv(path)
	prob_q = 1.0 * data[data.label == 1]['qid'].value_counts() / data['qid'].value_counts()
	prob_u = 1.0 * data[data.label == 1]['uid'].value_counts() / data['uid'].value_counts()

	prob_q.fillna(0).to_csv('prob_q.csv', header = ['prob_q'])
	prob_u.fillna(0).to_csv('prob_u.csv', header = ['prob_u'])

def cal_val(path):
	# 计算验证集结果
	val_data = pd.read_csv(path + 'validate.csv')
	prob_q = pd.read_csv(path + 'prob_q.csv')
	prob_u = pd.read_csv(path + 'prob_u.csv')
	combine = pd.merge(val_data, prob_q, on='qid',how='left').fillna(prob_q.mean()[0])
	combine = pd.merge(combine, prob_u, on='uid', how='left').fillna(prob_u.mean()[0])  #均值填补由于中位数填补
	combine['label'] = combine['prob_q'] * combine['prob_u']
	combine[['qid','uid','label']].to_csv('temp.csv', index = None)

def cal_merge(path):
	data1 = pd.read_csv(path + 'temp091521.csv')
	data2 = pd.read_csv(path + 'temp091535.csv')
	combine = data1
	combine['label'] = 0.5 * (data1['label'] + data2['label'])
	combine.to_csv('temp.csv', index = None)

if __name__ == '__main__':
	path = 'C:\\Users\\Administrator\\Desktop\\ByteCup\\code\\'
	#cal_pQAndpU(path + 'train.csv')
	'''
	train = pd.read_csv(path + 'train.csv')
	#print(len(train['qid'].unique()), len(train['uid'].unique()))
	validate = pd.read_csv(path + 'validate.csv').fillna(-1)
	temp = train.append(validate)
	print(len(temp['qid'].unique()), len(temp['uid'].unique())) #(7778, 27937)分别对应问题的个数和专家的个数
	'''
	cal_val(path)
	#cal_merge(path)


