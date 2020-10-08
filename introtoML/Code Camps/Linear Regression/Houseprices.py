#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.impute import KNNImputer
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mserr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# In[2]:


#Reading the datasets
train_x = pd.read_csv("//home//koushik//Desktop//Machine Learning//train.csv",index_col=0)
train_x


# In[3]:


train_y = train_x['SalePrice']
train_x.drop('SalePrice',axis=1,inplace=True)
train_x


# In[4]:


test_x = pd.read_csv("//home//koushik//Desktop//Machine Learning//test.csv",index_col=0)
test_x


# In[5]:


# As a first step of pre-processing remove columns with null value ratio greater than provided limit
sample_size = len(train_x)
sample_size


# In[6]:


sample_size = len(test_x)
sample_size


# In[7]:


#Train dataset
train_col_with_nullvalues=[[col,float(train_x[col].isnull().sum())/float(sample_size)] for col in train_x.columns if train_x[col].isnull().sum()]
train_col_with_nullvalues


# In[8]:


#Test dataset
test_col_with_nullvalues=[[col,float(test_x[col].isnull().sum())/float(sample_size)] for col in test_x.columns if test_x[col].isnull().sum()]
test_col_with_nullvalues


# In[9]:


print(len(train_col_with_nullvalues))
print(len(test_col_with_nullvalues))


# In[10]:


train_col_to_drop=[x for (x,y) in train_col_with_nullvalues if y>0.5]
train_col_to_drop


# In[11]:


test_col_to_drop=[x for (x,y) in test_col_with_nullvalues if y>0.5]
test_col_to_drop


# In[12]:


train_x.drop(train_col_to_drop,axis=1,inplace=True)
# test_x.drop(col_to_drop,axis=1,inplace=True)
print(train_x)
# print(test_x)


# In[13]:


test_x.drop(test_col_to_drop,axis=1,inplace=True)
# test_x.drop(col_to_drop,axis=1,inplace=True)
print(test_x)


# In[14]:


# As a second pre-processing step find all categorical columns and one hot  encode them. 
# Before one hot encode fill all null values with dummy in those columns.  
# Some categorical columns in train_x may not have null values in train_x but have null values in test_x.
# To overcome this problem we will add a row to the train_x with all dummy values for categorical values. 
# Once one hot encoding is complete drop the added dummy column


# In[15]:


# Train dataset
train_categorical_columns=[col for col in train_x.columns if train_x[col].dtype==object]
train_categorical_columns
print(len(train_categorical_columns))
train_ordinal_columns=[col for col in train_x.columns if col not in train_categorical_columns]
train_ordinal_columns
print(len(train_ordinal_columns))


# In[16]:


#Test datset
test_categorical_columns=[col for col in test_x.columns if test_x[col].dtype==object]
test_categorical_columns
print(len(test_categorical_columns))
test_ordinal_columns=[col for col in test_x.columns if col not in test_categorical_columns]
test_ordinal_columns
print(len(test_ordinal_columns))


# In[17]:


train_unique = []
for col in train_categorical_columns:
    train_unique.append(len(train_x[col].unique()))
print(train_unique)
# train_x['SaleCondition']


# In[18]:


test_unique = []
for col in test_categorical_columns:
    test_unique.append(len(test_x[col].unique()))
print(test_unique)


# In[19]:


dummy_row=list()
for col in train_x.columns:
    if col in train_categorical_columns:
        dummy_row.append("dummy")
    else:
        dummy_row.append("")
new_row = pd.DataFrame([dummy_row],columns=train_x.columns)
train_x = pd.concat([train_x,new_row],axis=0, ignore_index=True)
train_x


# In[20]:


for col in train_categorical_columns:
    train_x[col].fillna(value="dummy",inplace=True)
    test_x[col].fillna(value="dummy",inplace=True)
    
enc = OneHotEncoder(drop='first',sparse=False)
enc.fit(train_x[train_categorical_columns])
trainx_enc=pd.DataFrame(enc.transform(train_x[train_categorical_columns]))
trainx_enc.columns=enc.get_feature_names(train_categorical_columns)

testx_enc=pd.DataFrame(enc.transform(test_x[train_categorical_columns]))
testx_enc.columns=enc.get_feature_names(train_categorical_columns)

train_x = pd.concat([train_x[train_ordinal_columns],trainx_enc],axis=1,ignore_index=True)
test_x = pd.concat([test_x[train_ordinal_columns],testx_enc],axis=1,ignore_index=True)

train_x.drop(train_x.tail(1).index,inplace=True)
train_x


# In[21]:


test_x.shape


# In[22]:


imputer = KNNImputer(n_neighbors=2)
imputer.fit(train_x)
trainx_filled = imputer.transform(train_x)
trainx_filled=pd.DataFrame(trainx_filled,columns=train_x.columns)
trainx_filled


# In[23]:


testx_filled = imputer.transform(test_x)
testx_filled=pd.DataFrame(trainx_filled,columns=test_x.columns)
testx_filled.reset_index(drop=True,inplace=True)
testx_filled


# In[24]:


# Standardization

scaler = preprocessing.StandardScaler().fit(train_x)
train_x=scaler.transform(trainx_filled)
test_x=scaler.transform(testx_filled)
print(test_x)
print(train_x)


# In[25]:


test_x = pd.DataFrame(test_x)
train_x = pd.DataFrame(train_x)


# In[26]:


test_x.drop(test_x.tail(1).index,inplace=True)


# In[27]:


type(train_x)


# In[29]:


train_x.shape


# In[30]:


test_x.shape


# In[31]:


X_train, X_test, y_train, y_test = train_test_split(train_x, train_y.values.ravel(), test_size=0.5, random_state=42)


# In[32]:


# Applying regression techniques(Linear, Ridge, Lasso, Elasticnet)


reg = LinearRegression()
reg = LinearRegression().fit(X_train, y_train)
print(reg.score(X_train,y_train))
print(reg.score(X_test,y_test))


# In[33]:


from math import sqrt
from sklearn.metrics import r2_score, mean_squared_error
lin = LinearRegression()
lin.fit(X_train, y_train)
# print(lin.coef_)
lin.intercept_
predictions = lin.predict(X_test)
print(sqrt(mean_squared_error(y_test, predictions)))
import sklearn
from sklearn.metrics import r2_score
sklearn.metrics.r2_score(y_test, predictions, sample_weight=None, multioutput='uniform_average')


# In[34]:


import statsmodels.api as sm
from statsmodels.api import OLS
model = sm.OLS(y_train,X_train)
results = model.fit()
print(results.summary())


# In[35]:


import statsmodels.api as sm
from statsmodels.api import OLS
model = sm.OLS(y_test,X_test)
results = model.fit()
print(results.summary())


# In[36]:


from math import sqrt
from sklearn.metrics import r2_score, mean_squared_error
Ridgereg = Ridge(alpha = 0.2,tol = 0.1)
Ridgereg = Ridgereg.fit(X_train,y_train)
# print(Ridgereg.score(X_train,y_train))
# print(Ridgereg.score(X_test,y_test))

print(sqrt(mean_squared_error(y_test, Ridgereg.predict(X_test))))
print('R2 Value/Coefficient of Determination: {}'.format(Ridgereg.score(X_test,y_test)))
                                                                       


# In[37]:


from sklearn.linear_model import Lasso
from math import sqrt
from sklearn.metrics import r2_score, mean_squared_error


lassoreg = Lasso(alpha=0.001, normalize=True)
lassoreg.fit(X_train, y_train)
# lassoreg.predict(X_train)
print(sqrt(mean_squared_error(y_test, lassoreg.predict(X_test))))
print('R2 Value/Coefficient of Determination: {}'.format(lassoreg.score(X_test, y_test)))



# In[38]:


from sklearn.linear_model import ElasticNet
Elas = ElasticNet(alpha=0.001, normalize=True)
Elas.fit(X_train, y_train)

# print(sqrt(mean_squared_error(ytrain, Elas.predict(xtrain))))
print(sqrt(mean_squared_error(y_test, Elas.predict(X_test))))
print('R2 Value/Coefficient of Determination: {}'.format(Elas.score(X_test, y_test)))


# In[39]:


# test_x = np.array(test_x)
type(test_x)


# In[40]:


test_x


# In[41]:


score_train=[]
score_test=[]
mse_train=[]
mse_test=[]
alpha=[]
for sigma in np.linspace(0.1, 10,10):
    alpha.append(sigma)
    Ridgereg = Ridge(alpha = 0.2,tol = 0.0001)
    Ridgereg = Ridgereg.fit(X_train,y_train)
    score_train.append(round(Ridgereg.score(X_train, y_train),10))
    score_test.append(round(Ridgereg.score(X_test, y_test),10))
#     print("score_train = ",score_train)
#     print("score_test = ", score_test)
    mse_train.append(sqrt(mean_squared_error(y_train, Ridgereg.predict(X_train))))
    mse_test.append(sqrt(mean_squared_error(y_test, Ridgereg.predict(X_test))))
print(alpha,'\n',"Score train = ", score_train, '\n',"Score test = ", score_test,'\n',"MSE train =", mse_train, '\n',mse_test) 


# In[42]:


plt.figure(1)
plt.plot(alpha, score_train, 'g--',label="train_score")
plt.plot(alpha, score_test, 'r-o',label="test_score")
plt.xlabel='Alpha'
plt.legend()
plt.figure(2)
plt.plot(alpha, mse_train, 'y--',label="train_mse")
plt.plot(alpha, mse_test, 'c-o',label="test_mse")
plt.xlabel='Alpha'
plt.legend()
plt.show()


# In[44]:


testpred=pd.DataFrame(Elas.predict(test_x),columns=['SalePrice'])
testpred.index.name = 'Id'
testpred.to_csv("test_pred_1.csv")

