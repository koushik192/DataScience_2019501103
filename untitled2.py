# -- coding: utf-8 --
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from scipy.stats import skew
from scipy.stats import uniform
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import make_scorer

path = "//home//koushik//Desktop//"
train = path+"train.csv"
test = path+"test.csv"
trainx_df = pd.read_csv(train,index_col='Id')
trainy_df = trainx_df['SalePrice']
trainx_df.drop('SalePrice',axis=1,inplace=True)
testx_df = pd.read_csv(test,index_col='Id')
sample_size = len(trainx_df)
columns_with_null_values=[]
columns_with_null_values=[[col,float(trainx_df[col].isnull().sum())/float(sample_size)]
                          for col in trainx_df.columns if
                          trainx_df[col].isnull().sum()]
columns_to_drop = [x for (x,y) in columns_with_null_values if y>.3]
trainx_df.drop(columns_to_drop,axis=1,inplace=True)
testx_df.drop(columns_to_drop,axis=1,inplace=True)
categorical_columns=[col for col in trainx_df.columns if 
                     trainx_df[col].dtype==object]
#categorical_columns.append('MSSubClass')
ordinal_columns = [col for col in trainx_df.columns if col not in categorical_columns]
dummy_row=list()
for col in trainx_df.columns:
    if col in categorical_columns:
        dummy_row.append("dummy")
    else:
        dummy_row.append("")
#print(dummy_row)

new_row=pd.DataFrame([dummy_row],columns=trainx_df.columns)
trainx_df=pd.concat([trainx_df,new_row],axis = 0,ignore_index=True)
testx_df=pd.concat([testx_df],axis=0,ignore_index=True)
for col in categorical_columns:
    trainx_df[col].fillna(value="dummy",inplace=True)
    testx_df[col].fillna(value="dummy",inplace=True)
enc = OneHotEncoder(drop='first',sparse=False)
enc.fit(trainx_df[categorical_columns])
#print(enc.get_feature_names(categorical_columns))
trainx_enc=pd.DataFrame(enc.transform(trainx_df[categorical_columns]))
testx_enc = pd.DataFrame(enc.transform(testx_df[categorical_columns]))
trainx_enc.columns=enc.get_feature_names(categorical_columns)
testx_enc.columns=enc.get_feature_names(categorical_columns)
trainx_df=pd.concat([trainx_df[ordinal_columns],trainx_enc],axis=1,ignore_index=
                    True)
testx_df=pd.concat([testx_df[ordinal_columns],testx_enc],axis=1,ignore_index=
                    True)
trainx_df.drop(trainx_df.tail(1).index,inplace=True)
imputer = KNNImputer(n_neighbors=2)
imputer.fit(trainx_df)
trainx_df_filled=imputer.transform(trainx_df)
trainx_df_filled = pd.DataFrame(trainx_df_filled,columns=trainx_df.columns)
testx_df_filled=imputer.transform(testx_df)
testx_df_filled = pd.DataFrame(testx_df_filled,columns=testx_df.columns)
testx_df_filled.reset_index(drop=True,inplace=True)
print(trainx_df_filled.isnull().sum())
scaler = preprocessing.StandardScaler()