# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 18:09:19 2016

@author: Administrator
"""

import csv
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

def find_gene(path):
    candidate = pd.read_csv(path + 'snps_candidate.csv', header = None)[0].tolist()
    s = []
    for i in range(300):
        f_name = '\\gene_info\\gene_%d.dat' % (i + 1)
        gene = pd.read_csv(path + f_name, header = None)[0].tolist()
        n = 0
        for g in gene:
            if g in candidate:
                n += 1
        s.append([i+1, n, len(gene), n*1.0/len(gene)])
    
    pd.DataFrame(s).to_csv('gene_candidate.csv', index = None, header = ['index', 'num1', 'num2', 'ratio'])
    
        
def gen_candidate(path):
    data = pd.read_csv(path)
    psnp = []
    for t in data['features'].tolist():
        if t.split('_')[0] not in psnp:
            psnp.append(t.split('_')[0])
        
    pd.DataFrame(psnp).to_csv('ans1.csv', index = None)    

def visualize(path):
    data = pd.read_csv(path + 'gene_candidate.csv')
    t = data.ratio
    x = range(len(t))
    plt.scatter(x,t,c='b')
    #plt.ylim(0,170) 
    plt.xlabel('Order Number(sort decreasingly) of Gene', fontsize = 14)
    plt.ylabel(' Ratio of SNPs Candidate', fontsize = 14)
    plt.show()
    
if __name__ == '__main__':
    path = 'C:\\Users\\Administrator\\Desktop\\math\\'
    #gen_candidate(path + 'xgb_selection.csv') 
    #find_gene(path)
    #visualize(path)
    data = pd.read_csv(path + 'gene_candidate.csv')
    data.sort(['ratio'], ascending = False).to_csv('gene_candidate.csv', index = None)







