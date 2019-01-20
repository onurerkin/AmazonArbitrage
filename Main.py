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

#def t_ind(quotes, tgt_margin=0.2, n_days=30):
#    quotes=quotes[['date','lowest_newprice']]
#    quotes=quotes.reset_index(drop=True)
#
#    t_matrix=pd.DataFrame(quotes.date).iloc[0:len(quotes)-n_days,]
#    t_matrix['T']=0
#    t_matrix['decision']='hold'
#    for i in range(len(quotes)-n_days):
#        a=quotes.iloc[i:i+n_days,:]
#        a['first_price']=quotes.iloc[i,1]
#        a['variation']=(a.lowest_newprice-a.first_price)/a.first_price
#        t_value=len(a[(a.variation>tgt_margin)]) - len(a[(a.variation<-tgt_margin)])
#        t_matrix.iloc[i,1]=t_value
#        if (t_value > 10):
#            t_matrix.iloc[i,2]='buy'
#        elif(t_value < -10):
#            t_matrix.iloc[i,2]='sell'
#        
#    plt.subplot(2, 1, 1)        
#    dates = matplotlib.dates.date2num(t_matrix.date)
#    plt.plot_date(dates, t_matrix['T'],linestyle='solid', marker='None')
#    plt.title(' T vs time ')
#    plt.xlabel('Time')
#    plt.ylabel('T value')
#        
#         
#    plt.subplot(2, 1, 2)
#    dates = matplotlib.dates.date2num(quotes.iloc[0:len(quotes)-n_days,].date)
#    plt.plot_date(dates, quotes.iloc[0:len(quotes)-n_days,]['lowest_newprice'],linestyle='solid', marker='None')
#    plt.xlabel('Time')
#    plt.ylabel('price')
#    plt.show()
#    plt.show()
#    return t_matrix
#        

# FUNCTION ENDS

# importing necessary datasets.
product_info=pd.read_csv('/Users/Erkin/Desktop/McGill/personal project/data/product_info.csv')
product_info=product_info.drop('Unnamed: 0',axis=1)
product_info_head=product_info.head(1000)

product=pd.read_csv('/Users/Erkin/Desktop/McGill/personal project/data/product.csv')
product=product.drop('Unnamed: 0',axis=1)
product_head=product.head(1000)

map_product=pd.read_csv('/Users/Erkin/Desktop/McGill/personal project/data/map_product.csv')
map_product=map_product.drop('Unnamed: 0',axis=1)
map_product_head=map_product.head(1000)

product_answer=pd.read_csv('/Users/Erkin/Desktop/McGill/personal project/data/product_answer.csv')
product_answer=product_answer.drop('Unnamed: 0',axis=1)
product_answer_head=product_answer.head(1000)

product_question=pd.read_csv('/Users/Erkin/Desktop/McGill/personal project/data/product_question.csv')
product_question=product_question.drop('Unnamed: 0',axis=1)
product_question_head=product_question.head(1000)

product_review=pd.read_csv('/Users/Erkin/Desktop/McGill/personal project/data/product_review.csv')
product_review=product_review.drop('Unnamed: 0',axis=1)
product_review_head=product_review.head(1000)

merged=pd.read_csv('/Users/Erkin/Desktop/McGill/personal project/data/merged.csv')
merged=merged.drop('Unnamed: 0',axis=1)
merged_head=merged.head(1000)

#product names
#product_names=product.iloc[:,1:3]
#merged=pd.merge(product_names,product_info,how='right',on='asin')
#merged_head=merged.head(1000)
#
##lowest price na replacement
#merged=merged.drop('Unnamed: 0',axis=1)
#merged['lowest_newprice']=merged['lowest_newprice'].fillna(merged['list_price'])
#merged_head=merged.head(1000)
#merged.isna().sum()




#removing values with less than 200 observations.
#asd=merged.groupby(['asin']).count()
#asd=asd[asd.date > 200]
#asd.reset_index(level=0, inplace=True)
#merged=merged[merged.asin.isin(asd.asin)]
#merged=merged.reset_index(drop=True)
#merged['date'] =  pd.to_datetime(merged['date']).dt.date
#
#unique_asins=merged.asin.unique()
#merged['T']=99
#for asin in unique_asins:
#    print(asin)
#    quotes=merged[merged.asin==asin]
#    iterable=quotes.iloc[0:len(quotes)-n_days,]
#    for i, row in iterable.iterrows():
#        a=quotes.loc[i:i+n_days,:]
#        a['first_price']=quotes.loc[i,'lowest_newprice']
#        a['variation']=(a.lowest_newprice-a.first_price)/a.first_price
#        t_value=len(a[(a.variation>tgt_margin)]) - len(a[(a.variation<-tgt_margin)])
#        merged.loc[i,'T']=t_value



#asins=merged.asin.unique().tolist()
#product0=t_ind(quotes=merged[merged.asin==asins[0]])
#product1=t_ind(quotes=merged[merged.asin==asins[1]])
#product2=t_ind(quotes=merged[merged.asin==asins[2]])
#product3=t_ind(quotes=merged[merged.asin==asins[3]])
#product4=t_ind(quotes=merged[merged.asin==asins[4]])
#product5=t_ind(quotes=merged[merged.asin==asins[5]])
#product6=t_ind(quotes=merged[merged.asin==asins[6]])
#




## Create the time index
#product6.set_index('date', inplace=True)
#ts=product6.drop('decision',axis=1)
#
#
## Verify the time index
#product6.head()
#product6.info()
#product6.index


## Run the AutoRegressive model
#from statsmodels.tsa.ar_model import AR
#ar1=AR(ts)
#model1=ar1.fit()
## View the results
#print('Lag: %s' % model1.k_ar)
#print('Coefficients: %s' % model1.params)
#
## Separate the data into training and test
#split_size = round(len(ts)*0.3)
#ts_train,ts_test = ts[0:len(ts)-split_size], ts[len(ts)-split_size:]
#
## Run the model again on the training data
#ar2=AR(ts_train)
#model2=ar2.fit()
#
## Predicting the outcome based on the test data
#ts_test_pred_ar = model2.predict(start=len(ts_train),end=len(ts_train)+len(ts_test)-1,dynamic=False)
#ts_test_pred_ar.index=ts_test.index
#
## Calculating the mean squared error of the model    
#from sklearn.metrics import mean_squared_error
#error = mean_squared_error(ts_test,ts_test_pred_ar)
#print('Test MSE: %.3f' %error)
#
## Plot the graph comparing the real value and predicted value
#from matplotlib import pyplot
#fig = plt.figure(dpi=100)
#pyplot.plot(ts_test)
#pyplot.plot(ts_test_pred_ar)


#df_dateasin=merged[['date','asin']]
#
#
#
#reviews_sorted=product_review.sort_values('review_date')
#reviews_sorted['number_of_reviews']=reviews_sorted.groupby(['asin','review_date']).cumcount()+1
#reviews_sorted['star_tot']=reviews_sorted.groupby('asin').star.cumsum()
#reviews_sorted = reviews_sorted.drop_duplicates(['asin','review_date'], keep='last')
#df_dateasin = df_dateasin.drop_duplicates(['asin','date'], keep='last')
#df_dateasin.columns=['review_date','asin']
#reviews_sorted = pd.merge(df_dateasin,reviews_sorted,how='left')
#reviews_sorted_head=reviews_sorted.head(1000)
#
#
#
#
#
#reviews_sorted['reviews_total']=reviews_sorted.groupby('asin').number_of_reviews.cumsum()
#reviews_sorted['star_avg']=reviews_sorted.star_tot/reviews_sorted.number_of_reviews
#reviews_sorted = reviews_sorted.drop_duplicates(['asin','review_date'], keep='last')
#t1=reviews_sorted[['review_date','asin','number_of_reviews','star_avg']]
#t1.columns=['date','asin','number_of_reviews','star_avg']
#merged2 = pd.merge(merged,t1,how='left')
#
#t2 = pd.merge(df_dateasin,t1,how='left')
#
#nul = merged2['number_of_reviews'].isnull()
#nul.groupby((nul.diff() == 1).cumsum()).cumsum()*3 + merged2['number_of_reviews'].ffill()
#


# FEATURE ENGINEERING

# aggregation of number of reviews and average star rating
reviews_sorted=product_review.set_index('review_date').sort_index()
reviews_sorted['number_of_reviews']=reviews_sorted.groupby('asin').cumcount()+1
reviews_sorted['star_tot']=reviews_sorted.groupby('asin').star.cumsum()
reviews_sorted=reviews_sorted.reset_index()
reviews_sorted['star_avg']=reviews_sorted.star_tot/reviews_sorted.number_of_reviews
reviews_sorted = reviews_sorted.drop_duplicates(['asin','review_date'], keep='last')
t1=reviews_sorted[['review_date','asin','number_of_reviews','star_avg']]
t1.columns=['date','asin','number_of_reviews','star_avg']
merged2 = pd.merge(merged,t1,how='left')
merged2['number_of_reviews']=merged2.groupby('asin').number_of_reviews.fillna(method='ffill')
merged2['number_of_reviews']=merged2.groupby('asin').number_of_reviews.fillna(method='bfill')
merged2['star_avg']=merged2.groupby('asin').star_avg.fillna(method='ffill')
merged2['star_avg']=merged2.groupby('asin').star_avg.fillna(method='bfill')
merged2_head=merged2.head(10000)
df_pred = merged2[merged2['T'] < 40]
df_pred_head=df_pred.head(10000)
df_pred['decision']=0 #don't buy
df_pred.loc[(df_pred['T']>7),'decision']=1 #buy
df_pred_head=df_pred.head(10000)

#removing products less than 150 datapoints
asd=df_pred.groupby(['asin']).count()
asd=asd[asd.date > 150]
asd.reset_index(level=0, inplace=True)
df_pred=df_pred[df_pred.asin.isin(asd.asin)]
df_pred=df_pred.reset_index(drop=True)
df_pred=df_pred.dropna(subset=['sales_rank'])




# BENCHMARK MODEL
asins=df_pred.asin.unique().tolist()
from sklearn.ensemble import RandomForestClassifier

products=[]
accuracies=[]
precisions=[]
recalls=[]
fscores=[]
supports=[]

d = {}
for i in range(len(asins)):
    d["product" + str(i)] = df_pred[df_pred.asin==asins[i]]
    
    
benchmark_model={}
benchmark_ytest={}
for key, value in d.items():
    X=value[['lowest_newprice','total_new','total_used','sales_rank']]
    y=value.decision


    split_size = round(len(X)*0.3)
    X_train,X_test = X[0:len(X)-split_size], X[len(X)-split_size:]
    y_train, y_test = y[0:len(y)-split_size], y[len(y)-split_size:]
    y_test=y_test.reset_index(drop=True)
#   from sklearn.model_selection import train_test_split
#   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 5)
    randomforest = RandomForestClassifier(random_state=0,n_estimators=100,max_depth=10)
    model = randomforest.fit(X_train, y_train)
    
    
    y_test_pred=pd.DataFrame(model.predict(X_test))
    test_pred=pd.concat([y_test,y_test_pred],axis=1)
    benchmark_ytest[str(key)]=test_pred


    from sklearn.metrics import accuracy_score
    
    benchmark_model[str(key)]=accuracy_score(y_test,y_test_pred)


    from sklearn.metrics import precision_recall_fscore_support as score
    
    precision, recall, fscore, support = score(y_test, y_test_pred)
    
    products.append(key)
    accuracies.append(accuracy_score(y_test,y_test_pred))
    precisions.append(precision)
    recalls.append(recall)
    fscores.append(fscore)
    supports.append(support)

products_df=pd.DataFrame({'products':products})
accuracies_df=pd.DataFrame({'accuracy':accuracies})
precisions_df=pd.DataFrame(precisions, columns=['precision_hold','precision_buy'])
recalls_df=pd.DataFrame(recalls, columns=['recall_hold','recall_buy'])
fscores_df=pd.DataFrame(fscores, columns=['fscore_hold','fscore_buy'])
supports_df=pd.DataFrame(supports, columns=['support_hold','support_buy'])
benchmark_scores=pd.concat([products_df,accuracies_df,precisions_df,recalls_df,fscores_df,supports_df],axis=1)
benchmark_scores=benchmark_scores.dropna()
benchmark_scores=benchmark_scores[benchmark_scores['support_buy']!=0]




# IMPROVED MODEL

asins=df_pred.asin.unique().tolist()

#Accuracy for product 0
product0=df_pred[df_pred.asin==asins[0]]
X=product0[['lowest_newprice','total_new','total_used','sales_rank','number_of_reviews','star_avg']]
y=product0.decision


from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier(random_state=0)
model = randomforest.fit(X, y)

from sklearn.feature_selection import SelectFromModel
sfm = SelectFromModel(model, threshold=0.05)
sfm.fit(X, y)
for feature_list_index in sfm.get_support(indices=True):
    print(X.columns[feature_list_index])
    
pd.DataFrame(list(zip(X.columns,model.feature_importances_)), columns = ['predictor','Gini coefficient'])


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 5)

model = randomforest.fit(X_train, y_train)


y_test_pred=pd.DataFrame(model.predict(X_test))

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_test_pred)


#Accuracy for product 1

product2=df_pred[df_pred.asin==asins[2]]
X=product2[['lowest_newprice','total_new','total_used','sales_rank','number_of_reviews','star_avg']]
y=product2.decision


from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier(random_state=0)
model = randomforest.fit(X, y)

from sklearn.feature_selection import SelectFromModel
sfm = SelectFromModel(model, threshold=0.05)
sfm.fit(X, y)
for feature_list_index in sfm.get_support(indices=True):
    print(X.columns[feature_list_index])
    
pd.DataFrame(list(zip(X.columns,model.feature_importances_)), columns = ['predictor','Gini coefficient'])


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 5)

model = randomforest.fit(X_train, y_train)


y_test_pred=pd.DataFrame(model.predict(X_test))

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_test_pred)


#Accuracy for product 7

product7=df_pred[df_pred.asin==asins[7]]
X=product7[['lowest_newprice','total_new','total_used','sales_rank','number_of_reviews','star_avg']]
y=product7.decision


from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier(random_state=0)
model = randomforest.fit(X, y)

from sklearn.feature_selection import SelectFromModel
sfm = SelectFromModel(model, threshold=0.05)
sfm.fit(X, y)
for feature_list_index in sfm.get_support(indices=True):
    print(X.columns[feature_list_index])
    
pd.DataFrame(list(zip(X.columns,model.feature_importances_)), columns = ['predictor','Gini coefficient'])


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 5)

model = randomforest.fit(X_train, y_train)


y_test_pred=pd.DataFrame(model.predict(X_test))

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_test_pred)





#Accuracy for product 9

product9=df_pred[df_pred.asin==asins[9]]
X=product9[['lowest_newprice','total_new','total_used','sales_rank','number_of_reviews','star_avg']]
y=product9.decision


from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier(random_state=0)
model = randomforest.fit(X, y)

from sklearn.feature_selection import SelectFromModel
sfm = SelectFromModel(model, threshold=0.05)
sfm.fit(X, y)
for feature_list_index in sfm.get_support(indices=True):
    print(X.columns[feature_list_index])
    
pd.DataFrame(list(zip(X.columns,model.feature_importances_)), columns = ['predictor','Gini coefficient'])


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 5)

model = randomforest.fit(X_train, y_train)


y_test_pred=pd.DataFrame(model.predict(X_test))

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_test_pred))
y_test_pred['actual']=y_test.reset_index(drop=True)

