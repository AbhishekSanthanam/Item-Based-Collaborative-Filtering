'''
Algorithms implemented by Abhishek Sekar (EE18B067) and Abhishek Santhanam (CS18B049)
The ensemble model roughly takes <5 hrs to produce the output.
'''







#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import scipy as sp
from tqdm import tqdm #to keep track of iterations
from collections import defaultdict # for labelling every unique customer
from sklearn.model_selection import train_test_split #splitting eval and train set
#from sklearn.metrics import mean_squared_error  #mse
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm
import catboost
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import explained_variance_score


# In[ ]:


train_data=pd.read_csv('/kaggle/input/inputdata/train.csv')
#train_data.sort_values(by=["song_id"],inplace=True)
test_data=pd.read_csv('/kaggle/input/inputdata/test.csv')
#save_for_later=pd.read_csv('/kaggle/input/inputdata/save_for_later.csv')
#save_for_later["score"]=[5 for i in range(len(save_for_later))]
#train_data=train_data.append(save_for_later)
train_data["idx"]=[i for i in range(1,len(train_data)+1)]
test_data["idx"]=[i for i in range(1,len(test_data)+1)]
#test_data.sort_values(by=["song_id"],inplace=True)
songs=pd.read_csv('/kaggle/input/inputdata/songs.csv')
songs_t=songs.copy(deep=True)
songs_t=songs_t[["released_year","language","number_of_comments"]]
songs_t.dropna()
means=songs_t.median(axis=0).astype('int')
for i in range(1,10001):
    if i in list(songs["song_id"]):
        i=i
    else:
        df_t=pd.DataFrame({"song_id":i,"released_year":[means["released_year"]],"language":["eng"],"number_of_comments":[means["number_of_comments"]]})
        songs=songs.append(df_t,ignore_index=True)
fin_tr_data=pd.merge(train_data,songs,on="song_id")
#print(fin_tr_data)
fin_tst_data=pd.merge(test_data,songs,on="song_id")
song_label=pd.read_csv('/kaggle/input/inputdata//song_labels.csv')
song_label.sort_values(by=["platform_id"],inplace=True)
song_label=song_label[["platform_id","count","label_id"]]
count_song_label=song_label.groupby(["platform_id"]).sum()
grouped_song_label=(song_label.groupby(["platform_id"]).max())
platform_id=grouped_song_label.index
platform_id_cnt=count_song_label.index
grouped_song_label["platform_id"]=platform_id
count_song_label["platform_id"]=platform_id_cnt
grouped_song_label.index=[i for i in range(1,len(platform_id)+1)]
count_song_label.index=[i for i in range(1,len(platform_id)+1)]
count_song_label["count_pid"]=count_song_label["count"]
count_song_label=count_song_label[["platform_id","count_pid"]]
#print(count_song_label)
grouped_song_label=grouped_song_label[["platform_id","label_id","count"]]
gsl_means=grouped_song_label.median(axis=0).astype('int')
#print(grouped_song_label.loc[grouped_song_label["count"].idxmax(),"platform_id"])
fin_tr_data["platform_id"]=fin_tr_data["platform_id"].replace(np.nan,"U865")
fin_tst_data["platform_id"]=fin_tst_data["platform_id"].replace(np.nan,"U865")
fin_tr_data=pd.merge(fin_tr_data,grouped_song_label,on="platform_id")
fin_tr_data=pd.merge(fin_tr_data,count_song_label,on="platform_id")
fin_tr_data.sort_values(by=['idx'],inplace=True)
#print(fin_tr_data)
fin_tr_data=fin_tr_data[["customer_id","song_id","released_year","language","number_of_comments","label_id","count","score","count_pid"]]
fin_tst_data=pd.merge(fin_tst_data,grouped_song_label,on="platform_id")
fin_tst_data=pd.merge(fin_tst_data,count_song_label,on="platform_id")
fin_tst_data.sort_values(by=['idx'],inplace=True)
fin_tst_data=fin_tst_data[["customer_id","song_id","released_year","language","number_of_comments","label_id","count","count_pid"]]
fin_tr_data["language"]=fin_tr_data["language"].replace(np.nan,"no-lang")
fin_tst_data["language"]=fin_tst_data["language"].replace(np.nan,"no-lang")
tr_means=fin_tr_data.median(axis=0).astype('int')
tst_means=fin_tst_data.median(axis=0).astype('int')
fin_tr_data["released_year"]=fin_tr_data["released_year"].replace(np.nan,tr_means["released_year"])
fin_tst_data["released_year"]=fin_tst_data["released_year"].replace(np.nan,tst_means["released_year"])
save_for_later=pd.read_csv('/kaggle/input/inputdata//save_for_later.csv')
customer_song_count=save_for_later.groupby(["customer_id"]).count()
customer_id=customer_song_count.index
customer_song_count["customer_id"]=customer_id
customer_song_count.index=[i for i in range(1,len(customer_song_count)+1)]
#print(customer_song_count)
customer_song_count_dict={}
for i in range(len(customer_song_count)):
    customer_song_count_dict[customer_song_count["customer_id"].values[i]]=customer_song_count["song_id"].values[i]
song_count=[]
for i in range(len(fin_tr_data)):
    c_id=fin_tr_data["customer_id"].values[i]
    if c_id in customer_song_count_dict.keys():
        song_count.append(customer_song_count_dict[c_id])
    else:
        song_count.append(0)
fin_tr_data["song_count"]=song_count
song_count=[]
for i in range(len(fin_tst_data)):
    c_id=fin_tst_data["customer_id"].values[i]
    if c_id in customer_song_count_dict.keys():
        song_count.append(customer_song_count_dict[c_id])
    else:
        song_count.append(0)
fin_tst_data["song_count"]=song_count        
#print(fin_tr_data,fin_tst_data)
customer_song_count=save_for_later.groupby(["song_id"]).count()
song_id=customer_song_count.index
customer_song_count["song_id"]=song_id
customer_song_count.index=[i for i in range(1,len(customer_song_count)+1)]
#print(customer_song_count)
customer_song_count_dict={}
for i in range(len(customer_song_count)):
    customer_song_count_dict[customer_song_count["song_id"].values[i]]=customer_song_count["customer_id"].values[i]
customer_count=[]
for i in range(len(fin_tr_data)):
    s_id=fin_tr_data["song_id"].values[i]
    if s_id in customer_song_count_dict.keys():
        customer_count.append(customer_song_count_dict[s_id])
    else:
        customer_count.append(0)
fin_tr_data["customer_count"]=customer_count
customer_count=[]
for i in range(len(fin_tst_data)):
    s_id=fin_tst_data["song_id"].values[i]
    if s_id in customer_song_count_dict.keys():
        customer_count.append(customer_song_count_dict[s_id])
    else:
        customer_count.append(0)
fin_tst_data["customer_count"]=customer_count
encoder=OrdinalEncoder()
encoder.fit(fin_tr_data.loc[:,["language"]])
train_lang=encoder.fit_transform(fin_tr_data.loc[:,["language"]])
test_lang=encoder.fit_transform(fin_tst_data.loc[:,["language"]])
fin_tr_data["language"]=train_lang
fin_tst_data["language"]=test_lang
train_cid=encoder.fit_transform(fin_tr_data.loc[:,["customer_id"]])
test_cid=encoder.fit_transform(fin_tst_data.loc[:,["customer_id"]])
fin_tr_data["customer_id"]=train_cid
fin_tst_data["customer_id"]=test_cid


# In[ ]:


fin_tr_data.to_csv('train_mod_ordered.csv',index=False)
fin_tst_data.to_csv('test_mod_ordered.csv',index=False)


# In[ ]:


X_train=fin_tr_data[["customer_id","song_id","released_year","language","number_of_comments","label_id","count","song_count","customer_count"]]
Y_train=fin_tr_data["score"]
#X_train,X_test,Y_train,Y_test=train_test_split(X_train,Y_train,test_size=0.8,random_state=42)
#X_val,X_test,Y_val,Y_test=train_test_split(X_test,Y_test,test_size=0.5,random_state=42)
X_test=fin_tst_data[["customer_id","song_id","released_year","language","number_of_comments","label_id","count","song_count","customer_count"]]
#model=LinearRegression()
#model = xgb.XGBRegressor(objective ='reg:squarederror',max_depth=28,min_child_weight=1.5,colsample_bytree=0.4,eta=0.4)
#model=CatBoostRegressor(vdepth=16,learning_rate=0.1,l2_leaf_reg=2,loss_function='RMSE')
model=LGBMRegressor(max_depth=12,n_estimators=1000,lambda_l2=2,bagging_fraction=0.8,feature_fraction=0.7)
model.fit(X_train,Y_train,categorical_feature=[0,2,3])
Y_pred=model.predict(X_test)
#print(mean_squared_error(Y_test,Y_pred))
answer=pd.DataFrame()
index=[i for i in range(len(Y_pred))]
answer["test_row_id"]=index
answer["score"]=Y_pred
answer.to_csv('submission.csv',index=False)


# In[ ]:


#ordering the data, arranging it

#reading the data
train_data=pd.read_csv('train.csv')
df = pd.DataFrame(train_data)
#print(df.nunique()) #~ 14053 customers, 10000 songs

test_data = pd.read_csv('test.csv')
test_df   = pd.DataFrame(test_data)

#train_set and eval_set split , assuming tr has all the customers
tr_data,eval_data = train_test_split(train_data,test_size =0.20, random_state =42) #train:eval :: 0.8 : 0.2
tr_data = train_data
eval_data = test_data
#eval_data = train_data  #for finding overfitting
#manipulations for tr_data
tr_df = pd.DataFrame(tr_data)
#print(tr_df.nunique()) 
tr_df.sort_values(by=['customer_id'],inplace = True) # this sorts the customers in ascending order
#indexing the customers as it aids in computation and readability
temp = defaultdict(lambda: len(temp))
ind = [temp[ele] for ele in tr_df['customer_id']] # array of indices wrt customers
tr_df['customer_index']= ind

#manipulations for eval_data
eval_df = pd.DataFrame(eval_data)
#print(eval_df.nunique()) 
#eval_df.sort_values(by=['customer_id'],inplace = True) # this sorts the customers in ascending order
#print(len(eval_df['customer_id']))
# do this only if eval is test else don't, clean up explicitly
ind = [temp[ele] for ele in eval_df['customer_id']] # array of indices wrt customers
eval_df['customer_index']= ind

print(eval_df)
#print(res[:1000])
customer_id = tr_df['customer_id'].unique() #contains all customer ids, position of customer id represents index values


# In[ ]:


# generating the rating matrix A from the tr set
no_customers = tr_df.nunique()[0] # number of distinct customers
no_songs     = tr_df.nunique()[1] # number of distinct songs

song_rating_count = np.zeros((no_songs), dtype = int)
user_rating_count = np.zeros((no_customers), dtype  = int)

# initialize
A = np.zeros((no_songs,no_customers),dtype = int) #rating matrix, contains the rating vectors
S = np.zeros((no_songs,no_songs))   #scoring matrix, contains the similarity between item i and item j

#Filling up A matrix : which is just another repn of the dataframe

#converting dataframe elements into numpy arrays
cust_ind_tr = tr_df['customer_index'].to_numpy()
song_id_tr  = tr_df['song_id'].to_numpy()
score_tr    = tr_df['score'].to_numpy()

#print(cust_ind_tr,song_id_tr,score_tr,tr_df)

for k in tqdm(range(len(cust_ind_tr))):
    i = int(song_id_tr[k])-1  #song id
    j = int(cust_ind_tr[k]) #customer index
    A[i][j] = int(score_tr[k]) #corresponding score is stored in the array
    
#print(A)
'''
#finding statistics on how many ratings each song has
for i in tqdm(range(no_songs)):
    for j in range(no_customers):
        if(A[i][j]):
            song_rating_count[i] += 1
            
#print(song_rating_count)

pop_song = 0   #flag for popular songs to optimize algo
rating_thresh = 50 #min 50 ratings for a song to be rated popular
pop_song_ind = [] #indices of popular songs

for i in tqdm(range(no_songs)):
    if(song_rating_count[i] >= rating_thresh):
        pop_song += 1
        pop_song_ind.append(i)
#Filling up S matrix where S[i][j] = similarity between song i and song j

for i in tqdm(pop_song_ind):
    for j in pop_song_ind:
              S[i][j] = sp.stats.pearsonr(A[i,:],A[j,:])[0] #pearson similarity between song i and song j

'''

for i in tqdm(range(no_songs)):
    for j in range(i,no_songs):
            if (i == j):
                  S[i][j] = 1
            else:
                  S[i][j] = sp.stats.pearsonr(A[i,:],A[j,:])[0] #pearson similarity between song i and song j

    for j in range(i):
              S[i][j] = S[j][i]    
  
  
        


# In[ ]:


#baseline estimate values

b_songs = np.zeros(no_songs)
b_customers = np.zeros(no_customers)
b_avg = np.true_divide(A.sum(),(A != 0).sum()) #mean rating value
B       = np.zeros(A.shape)  #baseline scores matrix

for i in tqdm(range(no_songs)):
        b_songs[i] = np.true_divide(A[i,:].sum(),(A[i,:] != 0).sum()) - b_avg   #average rating deviation given to song i

for j in tqdm(range(no_customers)):
        b_customers[j] = np.true_divide(A[:,j].sum(),(A[:,j] != 0).sum()) - b_avg #average rating given by customer j
        
for i in tqdm(range(no_songs)):
    for j in range(no_customers):
        B[i][j] = b_avg + b_songs[i] + b_customers[j]


# In[ ]:


# prediction part KNN with baseline

#converting dataframe elements into numpy arrays
present_cust_ind = eval_df['customer_index'].to_numpy() #customer index
song_id_eval  = eval_df['song_id'].to_numpy()   #song_id
#score_eval    = eval_df['score'].to_numpy()

# evaluating performance for k =20

k_NN = 20
pred_score = np.zeros(len(song_id_eval))
#sort the S matrix while retaining old indices
#S_sorted = S
S_indices = np.zeros(S.shape)
for i in tqdm(range(no_songs)):
     #sort the S matrix 
    #S_sorted[i,:].sort(reverse = True)
    
    S_indices[i] = [b[0] for b in sorted(enumerate(S[i,:]),key=lambda i:i[1], reverse = True)]
    
    
for k in tqdm(range(len(present_cust_ind))):
    i = song_id_eval[k] - 1 #song_id
    j = present_cust_ind[k]  #customer_ind
    
    #finding nearest neighbour vectors
    score_NN = []  #pearson scores
    #song_NN  = []  #most similar songs
    n = 1
    for j1 in S_indices[i]:

        if(A[int(j1)][j]): #if song j1 is rated by the customer j
            n += 1
            if(S[i][int(j1)] > 0):    #there is a positive similarity between song j1 and i
                score_NN.append(S[i][int(j1)])
                #song_NN.append(j1)
                pred_score[k] += S[i][int(j1)]*(A[int(j1)][j] - B[int(j1)][j]) 
            
            
        if(n >= k_NN):
            break
    if(pred_score[k]):        
        pred_score[k] /= sum(score_NN)
        pred_score[k] += B[i][j]
        


# In[ ]:


#assigning lgbm predictions to scores that weren't predicted by C.F
lgbm_pred = pd.read_csv('submission.csv ')
lgbm_df   = pd.DataFrame(lgbm_pred)
lgbm_scores = lgbm_df['score'].to_numpy()
#print(lgbm_df)
for i in range(len(pred_score)):
    if(pred_score[i] == 0):
        pred_score[i] = lgbm_scores[i]


# In[ ]:


lgbm_pred = pd.read_csv('submission.csv ')
lgbm_df   = pd.DataFrame(lgbm_pred)
#cf_pred   = pd.read_csv('submission_baseline_K20.csv ')
#cf_df   = pd.DataFrame(cf_pred)
lgbm_scores = lgbm_df['score'].to_numpy()
#cf_scores   = cf_df['score'].to_numpy()
cf_scores = pred_score
lambda_val = 0.40 #importance to be given to lgbm pred
pred_scores = np.zeros(len(pred_score))
pred_scores = lambda_val*(lgbm_scores) + (1-lambda_val)*cf_scores
'''
submission=pd.DataFrame()
index=[i for i in range(len(pred_scores))]
submission["test_row_id"]=index
submission["score"]=pred_scores
submission.to_csv('submission_mix_lambda_0.40_baseline.csv',index=False)
print(submission)
'''

#score clipping
for i in range(len(pred_scores)):
    if(pred_scores[i] > 4.95):
        pred_scores[i] = 5
    elif((pred_scores[i] >= 3.95)&(pred_scores[i] <= 4.05)):
        pred_scores[i] = 4
    elif((pred_scores[i] >= 2.95)&(pred_scores[i] <= 3.05)):
        pred_scores[i] = 3
    elif((pred_scores[i] >= 1.95)&(pred_scores[i] <= 2.05)):
        pred_scores[i] = 2
    elif((pred_scores[i] >= 0.95)&(pred_scores[i] <= 1.05)):
        pred_scores[i] = 1
        
submission=pd.DataFrame()
index=[i for i in range(len(pred_scores))]
submission["test_row_id"]=index
submission["score"]=pred_scores
submission.to_csv('submission_mix_lambda_0.4125_baseline_clipping.csv',index=False)
print(submission)

