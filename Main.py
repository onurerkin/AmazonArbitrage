#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 22:28:42 2018

@author: Erkin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# T VALUE FUNCTION

def t_ind(quotes, tgt_margin=0.2, n_days=30):
    quotes=quotes[['date','lowest_newprice']]
    quotes=quotes.reset_index(drop=True)

    t_matrix=pd.DataFrame(quotes.date).iloc[0:len(quotes)-n_days,]
    t_matrix['T']=[0]*(len(quotes)-n_days)
    for i in range(len(quotes)-n_days):
        a=quotes.iloc[i:i+n_days,:]
        a['first_price']=quotes.iloc[i,1]
        a['variation']=(a.lowest_newprice-a.first_price)/a.first_price
        t_value=a[(a.variation>tgt_margin) | (a.variation<-tgt_margin)].variation.sum()
        t_matrix.iloc[i,1]=t_value
        
    plt.subplot(2, 1, 1)        
    dates = matplotlib.dates.date2num(t_matrix.date)
    plt.plot_date(dates, t_matrix['T'],linestyle='solid', marker='None')
    plt.title(' T vs time ')
    plt.xlabel('Time')
    plt.ylabel('T value')
        
         
    plt.subplot(2, 1, 2)
    dates = matplotlib.dates.date2num(quotes.iloc[0:len(quotes)-n_days,].date)
    plt.plot_date(dates, quotes.iloc[0:len(quotes)-n_days,]['lowest_newprice'],linestyle='solid', marker='None')
    plt.xlabel('Time')
    plt.ylabel('price')
    plt.show()
    plt.show()
    return t_matrix
        

# FUNCTION ENDS

# importing necessary datasets.
product_info=pd.read_csv('/Users/Erkin/Desktop/McGill/personal project/data/product_info.csv')
product_info_head=product_info.head(1000)

product=pd.read_csv('/Users/Erkin/Desktop/McGill/personal project/data/product.csv')
product_head=product.head(1000)

#product names
product_names=product.iloc[:,1:3]
merged=pd.merge(product_names,product_info,how='right',on='asin')
merged_head=merged.head(1000)

#lowest price na replacement
merged=merged.drop('Unnamed: 0',axis=1)
merged['lowest_newprice']=merged['lowest_newprice'].fillna(merged['list_price'])
merged_head=merged.head(1000)
merged.isna().sum()


#removing values with less than 200 observations.
asd=merged.groupby(['asin']).count()
asd=asd[asd.date > 200]
asd.reset_index(level=0, inplace=True)
merged=merged[merged.asin.isin(asd.asin)]
merged=merged.reset_index(drop=True)
merged['date'] =  pd.to_datetime(merged['date']).dt.date


asins=merged.asin.unique().tolist()

product0=t_ind(quotes=merged[merged.asin==asins[0]])
product1=t_ind(quotes=merged[merged.asin==asins[1]])
product2=t_ind(quotes=merged[merged.asin==asins[2]])
product3=t_ind(quotes=merged[merged.asin==asins[3]])




        
    