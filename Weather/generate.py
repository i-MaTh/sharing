# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 18:52:01 2016

@author: i-MaTh
"""

import numpy as np
import pandas as pd
from random import random,uniform,seed,randint
from datetime import datetime,timedelta
import matplotlib.pyplot as plt

seed(109)

def dateRange(start, end):
    days = (datetime.strptime(end, "%Y-%m-%d") - datetime.strptime(start, "%Y-%m-%d")).days + 1
    return [datetime.strftime(datetime.strptime(start, "%Y-%m-%d") + timedelta(i), "%Y-%m-%d") for i in xrange(days)] 

def date_time_generate(date):
    _time = []             
    for i in range(7,22):
        _str1 = ''
        _str1 += '%d:'%i
        for j in range(60):
            _str2 = ''
            if j < 10:
                _str2 += '0%d:'%j
            else:
                _str2 += '%d:'%j
            for k in range(0,60,5):
                _str3 = ''                
                if k < 10:
                    _str3 += '0%d'%k
                else:
                    _str3 += '%d'%k
                _time.append(_str1 + _str2 + _str3)
    #print len(_time) #10800
    _date = []
    for d in date:
        for _ in range(len(_time)):
            _date.append(d)
    #print len(_date) #324000      
    return _date, _time
    
def w_generate(_date, _time, n_days):
    #合成气象数据
    w_temperature = [round(uniform(3,20), 2) for _ in range(n_days*len(_time))] #气象温度
    humidity = [round(uniform(0.4,1.0), 2) for _ in range(n_days*len(_time))] #相对湿度
    wind_velocity = [round(uniform(2,6), 2) for _ in range(n_days*len(_time))] #风速
    _type = ['fine', 'cloudy', 'overcast', 'rain'] #天气类型
    w_type = []
    for _ in range(n_days):
        rand = randint(0,3)
        for i in range(len(_time)):
            w_type.append(_type[rand]) 
    
    w_data = pd.DataFrame()
    #w_data['weather_temperature'] = w_temperature
    w_data['w_temperature'] = w_temperature
    w_data['humidity'] = humidity
    w_data['velocity'] = wind_velocity
    w_data['weather_type'] = w_type
    w_data['time'] = _time*n_days
    w_data['date'] = _date
    #w_data.to_csv('info_weather.csv', index=None)
    print w_data.head(10)
    

def b_generate(_date, _time, n_days):
    #合成个人体征数据
    b_temperature = [round(uniform(36,37.5), 2) for _ in range(n_days*len(_time))]
    heart_rate = [randint(60,100) for _ in range(n_days*len(_time))]
       
    b_data = pd.DataFrame()
    b_data['body_temperature'] = b_temperature
    b_data['heart_rate'] = heart_rate
    b_data['time'] = _time*n_days
    b_data['date'] = _date
    #b_data.to_csv('info_body.csv', index=None)
    print b_data.head(10)
    
def u_generate(k):
    #合成用户信息    
    u_names = []
    chars = 'AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz0123456789'
    for i in range(k):   
        _str = ''
        for _ in range(10):
            _str += chars[randint(0, len(chars)-1)]
        u_names.append(_str)
    
    gender = ['male','female']
    u_genders = []
    for _ in range(k):
        u_genders.append(gender[randint(0,len(gender)-1)])
        
    u_ages = [round(uniform(35,75), 2) for _ in range(k)]
    
    heights = [round(uniform(1.5,1.85), 2) for _ in range(k)]       
    
    weights = [round(uniform(45,70), 2) for _ in range(k)]
    sickness_type = ['Hyperlipidemia','Hypertension','Heart_disease','Diabetes','Coronary_disease','Cerebral_thrombosis']
    u_sickness = []
    for _ in range(k):
        u_sickness.append(sickness_type[randint(0,len(sickness_type)-1)])
    
    u_data = pd.DataFrame()
    u_data['name'] = u_names
    u_data['gender'] = u_genders
    u_data['age'] = u_ages
    u_data['height'] = heights
    u_data['weight'] = weights
    u_data['info_sickness'] = u_sickness
    #u_data.to_csv('info_user.csv', index=None)
    print u_data.head(10)

def plot_importance():
    title = 'Feature importance'
    xlabel = 'F-score'
    ylabel = 'Features'
    values = [50,97,107,110,178,195,267,288]
    labels = ['time','weather_type','body_temperature','age', 'weight','heart_rate','weather_temperature','info_sickness']
    
    _, ax = plt.subplots(1, 1)
    ylocs = np.arange(len(values))
    ax.barh(ylocs, values, align = 'center', height = 0.2)
    for x, y in zip(values, ylocs):
        ax.text(x + 2, y, x, va = 'center')
    
    ax.set_yticks(ylocs)
    ax.set_yticklabels(labels)
    xlim = (0, max(values) * 1.1)
    ax.set_xlim(xlim)
    ylim = (-1, len(values))
    ax.set_ylim(ylim)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)

if __name__ == '__main__':
    #d = dateRange('2016-11-1', '2016-11-30')
    #_date, _time = date_time_generate(d)
    #w_generate(_date, _time, 30)
    #b_generate(_date, _time, 30)
    #u_generate(100)
    plot_importance()














