#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 22:28:42 2018

@author: Erkin
"""

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

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




# hold and buy
merged2_head=merged2.head(10000)
df_pred = merged2[merged2['T'] < 40]
df_pred_head=df_pred.head(10000)
df_pred['decision']=0 #don't buy
df_pred.loc[(df_pred['T']>5),'decision']=1 #buy
df_pred_head=df_pred.head(10000)

#
## price diff
#price_diff=[]
#df_pred['price_diff']=0
#for game in df_pred.asin.unique():
#    price_diff.append(0)
#    for row in range(1,len(df_pred[df_pred.asin==game])):
#        price_diff.append((df_pred.iloc[row,4]-df_pred.iloc[row-1,4])/df_pred.iloc[row-1,4]) 
#df_pred['price_diff']=price_diff
#df_pred_head=df_pred.head(10000)
#
#


#removing products less than 150 datapoints
asd=df_pred.groupby(['asin']).count()
asd=asd[asd.date > 150]
asd.reset_index(level=0, inplace=True)
df_pred=df_pred[df_pred.asin.isin(asd.asin)]
df_pred=df_pred.reset_index(drop=True)
df_pred=df_pred.dropna(subset=['sales_rank'])


#%%

# BENCHMARK MODEL


#random forest
asins=df_pred.asin.unique().tolist()
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


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
    
#    sfm = SelectFromModel(model, threshold=0.03)
#    sfm.fit(X_train, y_train)
#    for feature_list_index in sfm.get_support(indices=True):
#        print(X_train.columns[feature_list_index])

    
    
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
benchmark_scores=benchmark_scores[benchmark_scores['support_buy']>10]

benchmark_scores.precision_buy.mean()  #precision is 52.7%
benchmark_scores.recall_buy.mean()  #recall is  38%
benchmark_scores.accuracy.mean()   #accuracy is 70%

#%%




#regression
asins=df_pred.asin.unique().tolist()
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score


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
    y=value['T']
    dec=value.decision


    split_size = round(len(X)*0.3)
    X_train,X_test = X[0:len(X)-split_size], X[len(X)-split_size:]
    y_train, y_test = y[0:len(y)-split_size], y[len(y)-split_size:]
    y_test=y_test.reset_index(drop=True)
    dec_train,dec_test=dec[0:len(dec)-split_size], dec[len(dec)-split_size:]
#   from sklearn.model_selection import train_test_split
#   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 5)
    randomforest = RandomForestRegressor(random_state=0,n_estimators=100,max_depth=10)
    model = randomforest.fit(X_train, y_train)
    
#    sfm = SelectFromModel(model, threshold=0.03)
#    sfm.fit(X_train, y_train)
#    for feature_list_index in sfm.get_support(indices=True):
#        print(X_train.columns[feature_list_index])

    
    
    y_test_pred=pd.DataFrame(model.predict(X_test))
    y_test_pred['decision']=0
    y_test_pred.loc[y_test_pred[0]>5,'decision']=1
    y_test_pred=y_test_pred.drop([0],axis=1)
    test_pred=pd.concat([dec_test,y_test_pred],axis=1)
    benchmark_ytest[str(key)]=test_pred


    
    benchmark_model[str(key)]=accuracy_score(dec_test,y_test_pred)

    
    precision, recall, fscore, support = score(dec_test, y_test_pred)
    
    products.append(key)
    accuracies.append(accuracy_score(dec_test,y_test_pred))
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
benchmark_scores=benchmark_scores[benchmark_scores['support_buy']>10]

benchmark_scores.precision_buy.mean()  #precision is 51% - 57
benchmark_scores.recall_buy.mean()  #recall is  56% - 46
benchmark_scores.accuracy.mean()   #accuracy is 78% - 71


#%%



# all products (# just a random trial)
#
#X=df_pred[['total_new','total_used','sales_rank','price_diff']]
#y=df_pred.decision
#
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 5)
#randomforest = RandomForestClassifier(random_state=0,n_estimators=100,max_depth=10)
#model = randomforest.fit(X_train, y_train)
#    
#sfm = SelectFromModel(model, threshold=0.03)
#sfm.fit(X_train, y_train)
#for feature_list_index in sfm.get_support(indices=True):
#    print(X_train.columns[feature_list_index])
#y_test_pred=pd.DataFrame(model.predict(X_test))
#
#
#from sklearn import metrics
#from sklearn.metrics import classification_report
#
#print("MODEL B1: All Products \n")
#
#print ('The precision for this classifier is ' + str(metrics.precision_score(y_test, y_test_pred)))
#print ('The recall for this classifier is ' + str(metrics.recall_score(y_test, y_test_pred)))
#print ('The f1 for this classifier is ' + str(metrics.f1_score(y_test, y_test_pred)))
#print ('The accuracy for this classifier is ' + str(metrics.accuracy_score(y_test, y_test_pred)))
#
#print ('\nHere is the classification report:')
#print (classification_report(y_test, y_test_pred))
#
#from sklearn.metrics import confusion_matrix
#print(pd.DataFrame(confusion_matrix(y_test, y_test_pred, labels=[1, 0]), index=['true:1', 'true:0'], columns=['pred:1', 'pred:0']))
#




# IMPROVED MODEL
#asins=df_pred.asin.unique().tolist()
#
##Accuracy for product 0
#product0=df_pred[df_pred.asin==asins[0]]
#X=product0[['lowest_newprice','total_new','total_used','sales_rank','number_of_reviews','star_avg']]
#y=product0.decision
#
#
#from sklearn.ensemble import RandomForestClassifier
#randomforest = RandomForestClassifier(random_state=0)
#model = randomforest.fit(X, y)
#
#from sklearn.feature_selection import SelectFromModel
#sfm = SelectFromModel(model, threshold=0.05)
#sfm.fit(X, y)
#for feature_list_index in sfm.get_support(indices=True):
#    print(X.columns[feature_list_index])
#    
#pd.DataFrame(list(zip(X.columns,model.feature_importances_)), columns = ['predictor','Gini coefficient'])
#
#
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 5)
#
#model = randomforest.fit(X_train, y_train)
#
#
#y_test_pred=pd.DataFrame(model.predict(X_test))
#
#from sklearn.metrics import accuracy_score
#accuracy_score(y_test,y_test_pred)
#
#
##Accuracy for product 1
#
#product2=df_pred[df_pred.asin==asins[2]]
#X=product2[['lowest_newprice','total_new','total_used','sales_rank','number_of_reviews','star_avg']]
#y=product2.decision
#
#
#from sklearn.ensemble import RandomForestClassifier
#randomforest = RandomForestClassifier(random_state=0)
#model = randomforest.fit(X, y)
#
#from sklearn.feature_selection import SelectFromModel
#sfm = SelectFromModel(model, threshold=0.05)
#sfm.fit(X, y)
#for feature_list_index in sfm.get_support(indices=True):
#    print(X.columns[feature_list_index])
#    
#pd.DataFrame(list(zip(X.columns,model.feature_importances_)), columns = ['predictor','Gini coefficient'])
#
#
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 5)
#
#model = randomforest.fit(X_train, y_train)
#
#
#y_test_pred=pd.DataFrame(model.predict(X_test))
#
#from sklearn.metrics import accuracy_score
#accuracy_score(y_test,y_test_pred)
#
#
##Accuracy for product 7
#
#product7=df_pred[df_pred.asin==asins[7]]
#X=product7[['lowest_newprice','total_new','total_used','sales_rank','number_of_reviews','star_avg']]
#y=product7.decision
#
#
#from sklearn.ensemble import RandomForestClassifier
#randomforest = RandomForestClassifier(random_state=0)
#model = randomforest.fit(X, y)
#
#from sklearn.feature_selection import SelectFromModel
#sfm = SelectFromModel(model, threshold=0.05)
#sfm.fit(X, y)
#for feature_list_index in sfm.get_support(indices=True):
#    print(X.columns[feature_list_index])
#    
#pd.DataFrame(list(zip(X.columns,model.feature_importances_)), columns = ['predictor','Gini coefficient'])
#
#
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 5)
#
#model = randomforest.fit(X_train, y_train)
#
#
#y_test_pred=pd.DataFrame(model.predict(X_test))
#
#from sklearn.metrics import accuracy_score
#accuracy_score(y_test,y_test_pred)
#
#
#
#
#
##Accuracy for product 9
#
#product9=df_pred[df_pred.asin==asins[9]]
#X=product9[['lowest_newprice','total_new','total_used','sales_rank','number_of_reviews','star_avg']]
#y=product9.decision
#
#
#from sklearn.ensemble import RandomForestClassifier
#randomforest = RandomForestClassifier(random_state=0)
#model = randomforest.fit(X, y)
#
#from sklearn.feature_selection import SelectFromModel
#sfm = SelectFromModel(model, threshold=0.05)
#sfm.fit(X, y)
#for feature_list_index in sfm.get_support(indices=True):
#    print(X.columns[feature_list_index])
#    
#pd.DataFrame(list(zip(X.columns,model.feature_importances_)), columns = ['predictor','Gini coefficient'])
#
#
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 5)
#
#model = randomforest.fit(X_train, y_train)
#
#
#y_test_pred=pd.DataFrame(model.predict(X_test))
#
#from sklearn.metrics import accuracy_score
#print(accuracy_score(y_test,y_test_pred))
#y_test_pred['actual']=y_test.reset_index(drop=True)
#

#%%




# MATCHING QUESTION AND ANSWERS WITH PRODUCTS

matching=product[['asin','forum_id']]

product_question_asin = pd.merge(product_question,matching, on=['forum_id'])
matching=product_question_asin[['asin','forum_id','question_id']]
product_answer_asin=pd.merge(product_answer,matching, on=['question_id'])


# FEATURE ENGINEERING FOR QUESTIONS AND ANSWERS

# for questions
questions_sorted=product_question_asin.set_index('question_date').sort_index()
questions_sorted['sentiment_total']=questions_sorted.groupby('asin').sentiment.cumsum()
questions_sorted['number_of_questions']=questions_sorted.groupby('asin').sentiment.cumcount()+1
questions_sorted['sentiment_avg']=questions_sorted.sentiment_total/questions_sorted.number_of_questions

questions_sorted['polarity_total']=questions_sorted.groupby('asin').polarity.cumsum()
questions_sorted['polarity_avg']=questions_sorted.polarity_total/questions_sorted.number_of_questions

questions_sorted['subjectivity_total']=questions_sorted.groupby('asin').subjectivity.cumsum()
questions_sorted['subjectivity_avg']=questions_sorted.subjectivity_total/questions_sorted.number_of_questions

questions_sorted['len_question']=questions_sorted.question.apply(len)
questions_sorted['len_question_total']=questions_sorted.groupby('asin').len_question.cumsum()
questions_sorted['question_lenght_avg']=questions_sorted['len_question_total']/questions_sorted.number_of_questions

questions_sorted=questions_sorted.reset_index()
questions_sorted = questions_sorted.drop_duplicates(['asin','question_date'], keep='last')


questions_useful=questions_sorted[['question_date','asin', 'sentiment_total', 'number_of_questions', 'sentiment_avg',
       'polarity_total', 'polarity_avg', 'subjectivity_total',
       'subjectivity_avg', 'len_question_total',
       'question_lenght_avg']]

questions_useful.columns=['date','asin', 'sentiment_total_question', 'number_of_questions', 'sentiment_avg_question',
       'polarity_total_question', 'polarity_avg_question', 'subjectivity_total_question',
       'subjectivity_avg_question', 'len_question_total',
       'question_lenght_avg']


merged_ques = pd.merge(df_pred,questions_useful,how='left')
merged_ques[['sentiment_total_question', 'number_of_questions', 'sentiment_avg_question',
       'polarity_total_question', 'polarity_avg_question', 'subjectivity_total_question',
       'subjectivity_avg_question', 'len_question_total',
       'question_lenght_avg']]=merged_ques.groupby('asin')[['sentiment_total_question', 'number_of_questions', 'sentiment_avg_question',
       'polarity_total_question', 'polarity_avg_question', 'subjectivity_total_question',
       'subjectivity_avg_question', 'len_question_total',
       'question_lenght_avg']].fillna(method='ffill')
merged_ques[['sentiment_total_question', 'number_of_questions', 'sentiment_avg_question',
       'polarity_total_question', 'polarity_avg_question', 'subjectivity_total_question',
       'subjectivity_avg_question', 'len_question_total',
       'question_lenght_avg']]=merged_ques.groupby('asin')[['sentiment_total_question', 'number_of_questions', 'sentiment_avg_question',
       'polarity_total_question', 'polarity_avg_question', 'subjectivity_total_question',
       'subjectivity_avg_question', 'len_question_total',
       'question_lenght_avg']].fillna(method='bfill')
merged_ques_head=merged_ques.head(10000)


#for answers
product_answer_sorted=product_answer_asin.set_index('answer_date').sort_index()
product_answer_sorted['number_of_answers']=product_answer_sorted.groupby('asin').cumcount()+1
product_answer_sorted['sentiment_total']=product_answer_sorted.groupby('asin').sentiment.cumsum()
product_answer_sorted['sentiment_avg']=product_answer_sorted.sentiment_total/product_answer_sorted.number_of_answers


product_answer_sorted['polarity_total']=product_answer_sorted.groupby('asin').polarity.cumsum()
product_answer_sorted['polarity_avg']=product_answer_sorted.polarity_total/product_answer_sorted.number_of_answers

product_answer_sorted['subjectivity_total']=product_answer_sorted.groupby('asin').subjectivity.cumsum()
product_answer_sorted['subjectivity_avg']=product_answer_sorted.subjectivity_total/product_answer_sorted.number_of_answers

product_answer_sorted['len_answer']=product_answer_sorted.answer.apply(len)
product_answer_sorted['len_answer_total']=product_answer_sorted.groupby('asin').len_answer.cumsum()
product_answer_sorted['answer_lenght_avg']=product_answer_sorted['len_answer_total']/product_answer_sorted.number_of_answers

product_answer_sorted=product_answer_sorted.reset_index()


product_answer_useful=product_answer_sorted[['answer_date','asin',
        'number_of_answers', 'sentiment_total', 'sentiment_avg',
       'polarity_total', 'polarity_avg', 'subjectivity_total',
       'subjectivity_avg', 'len_answer_total',
       'answer_lenght_avg']]

product_answer_useful.columns=['date','asin',
        'number_of_answers', 'sentiment_total_answer', 'sentiment_avg_answer',
       'polarity_total_answer', 'polarity_avg_answer', 'subjectivity_total_answer',
       'subjectivity_avg_answer', 'len_answer_total',
       'answer_lenght_avg']



merged_ans = pd.merge(merged_ques,product_answer_useful,how='left')
merged_ans[ ['number_of_answers', 'sentiment_total_answer', 'sentiment_avg_answer',
       'polarity_total_answer', 'polarity_avg_answer', 'subjectivity_total_answer',
       'subjectivity_avg_answer', 'len_answer_total',
       'answer_lenght_avg']]=merged_ans.groupby('asin')[['number_of_answers', 'sentiment_total_answer', 'sentiment_avg_answer',
       'polarity_total_answer', 'polarity_avg_answer', 'subjectivity_total_answer',
       'subjectivity_avg_answer', 'len_answer_total',
       'answer_lenght_avg']].fillna(method='ffill')
merged_ans[ ['number_of_answers', 'sentiment_total_answer', 'sentiment_avg_answer',
       'polarity_total_answer', 'polarity_avg_answer', 'subjectivity_total_answer',
       'subjectivity_avg_answer', 'len_answer_total',
       'answer_lenght_avg']]=merged_ans.groupby('asin')[['number_of_answers', 'sentiment_total_answer', 'sentiment_avg_answer',
       'polarity_total_answer', 'polarity_avg_answer', 'subjectivity_total_answer',
       'subjectivity_avg_answer', 'len_answer_total',
       'answer_lenght_avg']].fillna(method='bfill')
merged_ans_head=merged_ans.head(20000)
merged_ans.len_answer_total.isna().sum()
merged_ans_dropedna=merged_ans.dropna()

len(df_pred[df_pred.decision==0])/len(df_pred)

#%%




# #### IMPROVED MODEL ####
#random forest classificiation


## keeping products with more than 150 data points



asd=merged_ans_dropedna.groupby(['asin']).count()
asd=asd[asd.date > 150]
asd.reset_index(level=0, inplace=True)
merged_ans_dropedna=merged_ans_dropedna[merged_ans_dropedna.asin.isin(asd.asin)]
merged_ans_dropedna=merged_ans_dropedna.reset_index(drop=True)

asins=merged_ans_dropedna.asin.unique().tolist()
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score

#
products=[]
accuracies=[]
precisions=[]
recalls=[]
fscores=[]
supports=[]

d = {}
for i in range(len(asins)):
    d["product" + str(i)] = merged_ans_dropedna[merged_ans_dropedna.asin==asins[i]]
    
importance = pd.DataFrame()

    
improved_model={}
improved_ytest={}
for key, value in d.items():
    print(key)
    
    X=value[['lowest_newprice','total_new','total_used','sales_rank','number_of_reviews','star_avg']]
#    X=value.drop(['asin', 'name', 'date', 'list_price','lowest_usedprice','tradein_value','T','decision'],axis=1)
    y=value.decision
    
    ## feature selection
    randomforest = RandomForestClassifier(random_state=0)
    model = randomforest.fit(X, y)

    sfm = SelectFromModel(model, threshold=0.01)
    sfm.fit(X, y)
    for feature_list_index in sfm.get_support(indices=True):
        print(X.columns[feature_list_index])
    
    
    feature_idx = sfm.get_support()
    feature_name = X.columns[feature_idx]
    print(pd.DataFrame(list(zip(X.columns,model.feature_importances_)), columns = ['predictor','Gini coefficient']).sort_values('Gini coefficient',ascending=False))
    temp_importance=pd.DataFrame([list(model.feature_importances_)],columns=X.columns)
    key_index=[key]
    temp_importance.index = key_index
    importance=importance.append(temp_importance)
    X_important = pd.DataFrame(sfm.transform(X))
    X_important.columns = feature_name


# model
    split_size = round(len(X_important)*0.3)
    X_train,X_test = X[0:len(X)-split_size], X[len(X)-split_size:]
#    X_train,X_test = X_important[0:len(X_important)-split_size], X_important[len(X_important)-split_size:]
    y_train, y_test = y[0:len(y)-split_size], y[len(y)-split_size:]
    y_test=y_test.reset_index(drop=True)
#   from sklearn.model_selection import train_test_split
#   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 5)
    randomforest = RandomForestClassifier(random_state=0,n_estimators=100,max_depth=10)
    model = randomforest.fit(X_train, y_train)
    
    # prediction
    y_test_pred=pd.DataFrame(model.predict(X_test))
    test_pred=pd.concat([y_test,y_test_pred],axis=1)
    improved_ytest[str(key)]=test_pred


    improved_model[str(key)]=accuracy_score(y_test,y_test_pred)
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
improved_scores=pd.concat([products_df,accuracies_df,precisions_df,recalls_df,fscores_df,supports_df],axis=1)
improved_scores=improved_scores.dropna()
improved_scores=improved_scores[improved_scores['support_buy']!=0]

print('precision:',improved_scores.precision_buy.mean())  #precision is 40%
print('recall:',improved_scores.recall_buy.mean())  #recall is  43%
print('accuracy:',improved_scores.accuracy.mean())   #accuracy is 75.3%

# importance dataframe removing zeros

importance=importance[importance.lowest_newprice!=0]
print(importance.mean().sort_values(ascending=False))

#%%







#regression
asd=merged_ans_dropedna.groupby(['asin']).count()
asd=asd[asd.date > 150]
asd.reset_index(level=0, inplace=True)
merged_ans_dropedna=merged_ans_dropedna[merged_ans_dropedna.asin.isin(asd.asin)]
merged_ans_dropedna=merged_ans_dropedna.reset_index(drop=True)

asins=merged_ans_dropedna.asin.unique().tolist()
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score

#
products=[]
accuracies=[]
precisions=[]
recalls=[]
fscores=[]
supports=[]

d = {}
for i in range(len(asins)):
    d["product" + str(i)] = merged_ans_dropedna[merged_ans_dropedna.asin==asins[i]]
    
importance = pd.DataFrame()

    
improved_model={}
improved_ytest={}
for key, value in d.items():
    print(key)
    
    X=value[['lowest_newprice','total_new','total_used','sales_rank','number_of_reviews','star_avg']]
#    X=value.drop(['asin', 'name', 'date', 'list_price','lowest_usedprice','tradein_value','T','decision'],axis=1)
    y=value['T']
    dec=value.decision

    ## feature selection
    randomforest = RandomForestRegressor(random_state=0)
    model = randomforest.fit(X, y)

    sfm = SelectFromModel(model, threshold=0.01)
    sfm.fit(X, y)
    for feature_list_index in sfm.get_support(indices=True):
        print(X.columns[feature_list_index])
    
    
    feature_idx = sfm.get_support()
    feature_name = X.columns[feature_idx]
    print(pd.DataFrame(list(zip(X.columns,model.feature_importances_)), columns = ['predictor','Gini coefficient']).sort_values('Gini coefficient',ascending=False))
    temp_importance=pd.DataFrame([list(model.feature_importances_)],columns=X.columns)
    key_index=[key]
    temp_importance.index = key_index
    importance=importance.append(temp_importance)
    X_important = pd.DataFrame(sfm.transform(X))
    X_important.columns = feature_name


# model
    split_size = round(len(X_important)*0.3)
    X_train,X_test = X[0:len(X)-split_size], X[len(X)-split_size:]
#    X_train,X_test = X_important[0:len(X_important)-split_size], X_important[len(X_important)-split_size:]
    y_train, y_test = y[0:len(y)-split_size], y[len(y)-split_size:]
    y_test=y_test.reset_index(drop=True)
    dec_train,dec_test=dec[0:len(dec)-split_size], dec[len(dec)-split_size:]

#   from sklearn.model_selection import train_test_split
#   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 5)
    randomforest = RandomForestRegressor(random_state=0,n_estimators=100,max_depth=10)
    model = randomforest.fit(X_train, y_train)
    
    # prediction
    y_test_pred=pd.DataFrame(model.predict(X_test))
    y_test_pred['decision']=0
    y_test_pred.loc[y_test_pred[0]>5,'decision']=1
    y_test_pred=y_test_pred.drop([0],axis=1)
    test_pred=pd.concat([dec_test,y_test_pred],axis=1)
    improved_ytest[str(key)]=test_pred
    




    improved_model[str(key)]=accuracy_score(dec_test,y_test_pred)
    precision, recall, fscore, support = score(dec_test, y_test_pred)
    products.append(key)
    accuracies.append(accuracy_score(dec_test,y_test_pred))
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
improved_scores=pd.concat([products_df,accuracies_df,precisions_df,recalls_df,fscores_df,supports_df],axis=1)
improved_scores=improved_scores.dropna()
improved_scores=improved_scores[improved_scores['support_buy']>10]

print('RandomForest Regression Improved Precison',improved_scores.precision_buy.mean())  #precision is 55%
print('RandomForest Regression Improved Recall',improved_scores.recall_buy.mean())  #recall is  47.5%
print('RandomForest Regression Improved Accuracy',improved_scores.accuracy.mean())   #accuracy is 70%

#baseline accuracy is 63%
improved_scores.support_hold.sum()/(improved_scores.support_buy.sum()+improved_scores.support_hold.sum())

# importance dataframe removing zeros

importance=importance[importance.lowest_newprice!=0]
print(importance.mean().sort_values(ascending=False))



#%%


## KNN


#regression
asd=merged_ans_dropedna.groupby(['asin']).count()
asd=asd[asd.date > 150]
asd.reset_index(level=0, inplace=True)
merged_ans_dropedna=merged_ans_dropedna[merged_ans_dropedna.asin.isin(asd.asin)]
merged_ans_dropedna=merged_ans_dropedna.reset_index(drop=True)

asins=merged_ans_dropedna.asin.unique().tolist()
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


products=[]
accuracies=[]
precisions=[]
recalls=[]
fscores=[]
supports=[]

d = {}
for i in range(len(asins)):
    d["product" + str(i)] = merged_ans_dropedna[merged_ans_dropedna.asin==asins[i]]
    
importance = pd.DataFrame()

    
improved_model={}
improved_ytest={}
for key, value in d.items():
    print(key)
    
    X=value[['lowest_newprice','total_new','total_used','sales_rank','number_of_reviews','star_avg']]
#    X=value.drop(['asin', 'name', 'date', 'list_price','lowest_usedprice','tradein_value','T','decision'],axis=1)
    y=value['T']
    dec=value.decision
    

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

#    ## feature selection
#    randomforest = RandomForestRegressor(random_state=0)
#    model = randomforest.fit(X, y)
#
#    sfm = SelectFromModel(model, threshold=0.01)
#    sfm.fit(X, y)
#    for feature_list_index in sfm.get_support(indices=True):
#        print(X.columns[feature_list_index])
#    
#    
#    feature_idx = sfm.get_support()
#    feature_name = X.columns[feature_idx]
#    print(pd.DataFrame(list(zip(X.columns,model.feature_importances_)), columns = ['predictor','Gini coefficient']).sort_values('Gini coefficient',ascending=False))
#    temp_importance=pd.DataFrame([list(model.feature_importances_)],columns=X.columns)
#    key_index=[key]
#    temp_importance.index = key_index
#    importance=importance.append(temp_importance)
#    X_important = pd.DataFrame(sfm.transform(X))
#    X_important.columns = feature_name


# model
    split_size = round(len(X)*0.3)
    X_train,X_test = X[0:len(X)-split_size], X[len(X)-split_size:]
#    X_train,X_test = X_important[0:len(X_important)-split_size], X_important[len(X_important)-split_size:]
    y_train, y_test = y[0:len(y)-split_size], y[len(y)-split_size:]
    y_test=y_test.reset_index(drop=True)
    dec_train,dec_test=dec[0:len(dec)-split_size], dec[len(dec)-split_size:]




#   from sklearn.model_selection import train_test_split
#   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 5)
    knn_reg=KNeighborsRegressor(n_neighbors=2)
#    params = {'n_neighbors':[5,6,7,8,9,10],
#          'leaf_size':[1,2,3,5],
#          'weights':['uniform', 'distance'],
#          'algorithm':['auto', 'ball_tree','kd_tree','brute'],
#          'n_jobs':[-1]}
#    model1 = GridSearchCV(model, param_grid=params, n_jobs=1)
    model=knn_reg.fit(X_train,y_train)
#    print("Best Hyper Parameters:\n",model1.best_params_)

    
#    randomforest = RandomForestRegressor(random_state=0,n_estimators=100,max_depth=10)
#    model = randomforest.fit(X_train, y_train)
    
    # prediction
    y_test_pred=pd.DataFrame(model.predict(X_test))
    y_test_pred['decision']=0
    y_test_pred.loc[y_test_pred[0]>5,'decision']=1
    y_test_pred=y_test_pred.drop([0],axis=1)
    test_pred=pd.concat([dec_test,y_test_pred],axis=1)
    improved_ytest[str(key)]=test_pred
    




    improved_model[str(key)]=accuracy_score(dec_test,y_test_pred)
    precision, recall, fscore, support = score(dec_test, y_test_pred)
    products.append(key)
    accuracies.append(accuracy_score(dec_test,y_test_pred))
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
improved_scores_knn=pd.concat([products_df,accuracies_df,precisions_df,recalls_df,fscores_df,supports_df],axis=1)
improved_scores_knn=improved_scores_knn.dropna()
improved_scores_knn=improved_scores_knn[improved_scores_knn['support_buy']>10]

print('KNN Regression Improved Precision',improved_scores_knn.precision_buy.mean())  #precision is 50%
print('KNN Regression Improved Recall',improved_scores_knn.recall_buy.mean())  #recall is  34%
print('KNN Regression Improved Accuracy',improved_scores_knn.accuracy.mean())   #accuracy is 67%

#baseline accuracy is 63%
improved_scores.support_hold.sum()/(improved_scores.support_buy.sum()+improved_scores.support_hold.sum())

# importance dataframe removing zeros

#importance=importance[importance.lowest_newprice!=0]
#print(importance.mean().sort_values(ascending=False))

#%%



#### KNN classification


asd=merged_ans_dropedna.groupby(['asin']).count()
asd=asd[asd.date > 150]
asd.reset_index(level=0, inplace=True)
merged_ans_dropedna=merged_ans_dropedna[merged_ans_dropedna.asin.isin(asd.asin)]
merged_ans_dropedna=merged_ans_dropedna.reset_index(drop=True)

asins=merged_ans_dropedna.asin.unique().tolist()
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
#
products=[]
accuracies=[]
precisions=[]
recalls=[]
fscores=[]
supports=[]

d = {}
for i in range(len(asins)):
    d["product" + str(i)] = merged_ans_dropedna[merged_ans_dropedna.asin==asins[i]]
    
importance = pd.DataFrame()

    
improved_model={}
improved_ytest={}
for key, value in d.items():
    print(key)
    
    X=value[['lowest_newprice','total_new','total_used','sales_rank','number_of_reviews','star_avg']]
#    X=value.drop(['asin', 'name', 'date', 'list_price','lowest_usedprice','tradein_value','T','decision'],axis=1)
    y=value.decision
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
#    ## feature selection
#    randomforest = RandomForestClassifier(random_state=0)
#    model = randomforest.fit(X, y)
#
#    sfm = SelectFromModel(model, threshold=0.01)
#    sfm.fit(X, y)
#    for feature_list_index in sfm.get_support(indices=True):
#        print(X.columns[feature_list_index])
#    
#    
#    feature_idx = sfm.get_support()
#    feature_name = X.columns[feature_idx]
#    print(pd.DataFrame(list(zip(X.columns,model.feature_importances_)), columns = ['predictor','Gini coefficient']).sort_values('Gini coefficient',ascending=False))
#    temp_importance=pd.DataFrame([list(model.feature_importances_)],columns=X.columns)
#    key_index=[key]
#    temp_importance.index = key_index
#    importance=importance.append(temp_importance)
#    X_important = pd.DataFrame(sfm.transform(X))
#    X_important.columns = feature_name


# model
    split_size = round(len(X)*0.3)
    X_train,X_test = X[0:len(X)-split_size], X[len(X)-split_size:]
#    X_train,X_test = X_important[0:len(X_important)-split_size], X_important[len(X_important)-split_size:]
    y_train, y_test = y[0:len(y)-split_size], y[len(y)-split_size:]
    y_test=y_test.reset_index(drop=True)
#   from sklearn.model_selection import train_test_split
#   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 5)
    knn_class=KNeighborsClassifier(n_neighbors=3)
    model = knn_class.fit(X_train, y_train)
    
    # prediction
    y_test_pred=pd.DataFrame(model.predict(X_test))
    test_pred=pd.concat([y_test,y_test_pred],axis=1)
    improved_ytest[str(key)]=test_pred


    
    improved_model[str(key)]=accuracy_score(y_test,y_test_pred)
    
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
improved_scores_knn_clas=pd.concat([products_df,accuracies_df,precisions_df,recalls_df,fscores_df,supports_df],axis=1)
improved_scores_knn_clas=improved_scores_knn_clas.dropna()
improved_scores_knn_clas=improved_scores_knn_clas[improved_scores_knn_clas['support_buy']>10]

print('KNN Classification Improved Precision:',improved_scores_knn_clas.precision_buy.mean())  #precision is 42%
print('KNN Classification Improved Recall:',improved_scores_knn_clas.recall_buy.mean())  #recall is  27%
print('KNN Classification Improved Accuracy:',improved_scores_knn_clas.accuracy.mean())   #accuracy is 67%


#%%


# ANN Classification
from sklearn.neural_network import MLPClassifier

asd=merged_ans_dropedna.groupby(['asin']).count()
asd=asd[asd.date > 150]
asd.reset_index(level=0, inplace=True)
merged_ans_dropedna=merged_ans_dropedna[merged_ans_dropedna.asin.isin(asd.asin)]
merged_ans_dropedna=merged_ans_dropedna.reset_index(drop=True)

asins=merged_ans_dropedna.asin.unique().tolist()
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, make_scorer
custom_scorer = make_scorer(precision_score, greater_is_better=True,  pos_label=0)

#
products=[]
accuracies=[]
precisions=[]
recalls=[]
fscores=[]
supports=[]

d = {}
for i in range(len(asins)):
    d["product" + str(i)] = merged_ans_dropedna[merged_ans_dropedna.asin==asins[i]]
    
importance = pd.DataFrame()

    
improved_model={}
improved_ytest={}
for key, value in d.items():
    print(key)
    
    X=value[['lowest_newprice','total_new','total_used','sales_rank','number_of_reviews','star_avg']]
#    X=value.drop(['asin', 'name', 'date', 'list_price','lowest_usedprice','tradein_value','T','decision'],axis=1)
    y=value.decision
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
#    ## feature selection
#    randomforest = RandomForestClassifier(random_state=0)
#    model = randomforest.fit(X, y)
#
#    sfm = SelectFromModel(model, threshold=0.01)
#    sfm.fit(X, y)
#    for feature_list_index in sfm.get_support(indices=True):
#        print(X.columns[feature_list_index])
#    
#    
#    feature_idx = sfm.get_support()
#    feature_name = X.columns[feature_idx]
#    print(pd.DataFrame(list(zip(X.columns,model.feature_importances_)), columns = ['predictor','Gini coefficient']).sort_values('Gini coefficient',ascending=False))
#    temp_importance=pd.DataFrame([list(model.feature_importances_)],columns=X.columns)
#    key_index=[key]
#    temp_importance.index = key_index
#    importance=importance.append(temp_importance)
#    X_important = pd.DataFrame(sfm.transform(X))
#    X_important.columns = feature_name


# model
    split_size = round(len(X)*0.3)
    X_train,X_test = X[0:len(X)-split_size], X[len(X)-split_size:]
#    X_train,X_test = X_important[0:len(X_important)-split_size], X_important[len(X_important)-split_size:]
    y_train, y_test = y[0:len(y)-split_size], y[len(y)-split_size:]
    y_test=y_test.reset_index(drop=True)
#   from sklearn.model_selection import train_test_split
#   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 5)
#    knn_class=KNeighborsClassifier(n_neighbors=6)
#    model = knn_class.fit(X_train, y_train)
    
    mlp = MLPClassifier(max_iter=500, hidden_layer_sizes=(10), activation='tanh', learning_rate='adaptive', alpha=0.05)

#    parameter_space = {
#        'hidden_layer_sizes': [(5,5,5), (5,10,5), (5,5), (5,10), (5), (10)],
#        'activation': ['tanh', 'relu'],
#        'solver': ['sgd', 'adam'],
#        'alpha': [0.0001, 0.05],
#        'learning_rate': ['constant','adaptive'],
#    }
        
    

#    clf = GridSearchCV(estimator=mlp,scoring=custom_scorer,param_grid=parameter_space, n_jobs=-1)
    mlp.fit(X_train, y_train)
#    print("Best Hyper Parameters:\n",clf.best_params_)
    
    # prediction
    y_test_pred=pd.DataFrame(mlp.predict(X_test))
    test_pred=pd.concat([y_test,y_test_pred],axis=1)
    improved_ytest[str(key)]=test_pred


    from sklearn.metrics import accuracy_score
    improved_model[str(key)]=accuracy_score(y_test,y_test_pred)
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
improved_scores_ann_clas=pd.concat([products_df,accuracies_df,precisions_df,recalls_df,fscores_df,supports_df],axis=1)
improved_scores_ann_clas=improved_scores_ann_clas.dropna()
improved_scores_ann_clas=improved_scores_ann_clas[improved_scores_ann_clas['support_buy']>10]

print('ANN Classification Improved Precision:',improved_scores_ann_clas.precision_buy.mean())  #precision is 46%
print('ANN Classification Improved Recall:',improved_scores_ann_clas.recall_buy.mean())  #recall is  52%
print('ANN Classification Improved Accuracy:',improved_scores_ann_clas.accuracy.mean())   #accuracy is 62%


#%%




#regression ANN
asd=merged_ans_dropedna.groupby(['asin']).count()
asd=asd[asd.date > 150]
asd.reset_index(level=0, inplace=True)
merged_ans_dropedna=merged_ans_dropedna[merged_ans_dropedna.asin.isin(asd.asin)]
merged_ans_dropedna=merged_ans_dropedna.reset_index(drop=True)

asins=merged_ans_dropedna.asin.unique().tolist()
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor



products=[]
accuracies=[]
precisions=[]
recalls=[]
fscores=[]
supports=[]

d = {}
for i in range(len(asins)):
    d["product" + str(i)] = merged_ans_dropedna[merged_ans_dropedna.asin==asins[i]]
    
importance = pd.DataFrame()

    
improved_model={}
improved_ytest={}
for key, value in d.items():
    print(key)
    
    X=value[['lowest_newprice','total_new','total_used','sales_rank','number_of_reviews','star_avg']]
#    X=value.drop(['asin', 'name', 'date', 'list_price','lowest_usedprice','tradein_value','T','decision'],axis=1)
    y=value['T']
    dec=value.decision
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

#    ## feature selection
#    randomforest = RandomForestRegressor(random_state=0)
#    model = randomforest.fit(X, y)
#
#    sfm = SelectFromModel(model, threshold=0.01)
#    sfm.fit(X, y)
#    for feature_list_index in sfm.get_support(indices=True):
#        print(X.columns[feature_list_index])
#    
#    
#    feature_idx = sfm.get_support()
#    feature_name = X.columns[feature_idx]
#    print(pd.DataFrame(list(zip(X.columns,model.feature_importances_)), columns = ['predictor','Gini coefficient']).sort_values('Gini coefficient',ascending=False))
#    temp_importance=pd.DataFrame([list(model.feature_importances_)],columns=X.columns)
#    key_index=[key]
#    temp_importance.index = key_index
#    importance=importance.append(temp_importance)
#    X_important = pd.DataFrame(sfm.transform(X))
#    X_important.columns = feature_name


# model
    split_size = round(len(X)*0.3)
    X_train,X_test = X[0:len(X)-split_size], X[len(X)-split_size:]
#    X_train,X_test = X_important[0:len(X_important)-split_size], X_important[len(X_important)-split_size:]
    y_train, y_test = y[0:len(y)-split_size], y[len(y)-split_size:]
    y_test=y_test.reset_index(drop=True)
    dec_train,dec_test=dec[0:len(dec)-split_size], dec[len(dec)-split_size:]

#   from sklearn.model_selection import train_test_split
#   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 5)
    mlp = MLPRegressor(max_iter=500, hidden_layer_sizes=(5,5,5), activation='tanh', learning_rate='adaptive', alpha=0.05)
#    params = {'n_neighbors':[5,6,7,8,9,10],
#          'leaf_size':[1,2,3,5],
#          'weights':['uniform', 'distance'],
#          'algorithm':['auto', 'ball_tree','kd_tree','brute'],
#          'n_jobs':[-1]}
#    model1 = GridSearchCV(model, param_grid=params, n_jobs=1)
    model=mlp.fit(X_train,y_train)
#    print("Best Hyper Parameters:\n",model1.best_params_)

    
#    randomforest = RandomForestRegressor(random_state=0,n_estimators=100,max_depth=10)
#    model = randomforest.fit(X_train, y_train)
    
    # prediction
    y_test_pred=pd.DataFrame(model.predict(X_test))
    print(y_test_pred)
    y_test_pred['decision']=0
    y_test_pred.loc[y_test_pred[0]>5,'decision']=1
    y_test_pred=y_test_pred.drop([0],axis=1)
    test_pred=pd.concat([dec_test,y_test_pred],axis=1)
    improved_ytest[str(key)]=test_pred
    
    
    
    improved_model[str(key)]=accuracy_score(dec_test,y_test_pred)
    precision, recall, fscore, support = score(dec_test, y_test_pred)
    products.append(key)
    accuracies.append(accuracy_score(dec_test,y_test_pred))
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
improved_scores_ann_reg=pd.concat([products_df,accuracies_df,precisions_df,recalls_df,fscores_df,supports_df],axis=1)
improved_scores_ann_reg=improved_scores_ann_reg.dropna()
improved_scores_ann_reg=improved_scores_ann_reg[improved_scores_ann_reg['support_buy']>10]

print('ANN Regression Improved Precision:',improved_scores_ann_reg.precision_buy.mean())  #precision is 25%
print('ANN Regression Improved Recall:',improved_scores_ann_reg.recall_buy.mean())  #recall is  %15
print('ANN Regression Improved Accuracy:',improved_scores_ann_reg.accuracy.mean())   #accuracy is %66

#baseline accuracy is 63%
improved_scores_ann_reg.support_hold.sum()/(improved_scores_ann_reg.support_buy.sum()+improved_scores_ann_reg.support_hold.sum())

# importance dataframe removing zeros

#importance=importance[importance.lowest_newprice!=0]
#print(importance.mean().sort_values(ascending=False))

#%%

### Boosting regression

#import xgboost as xgb
import pandas as pd
