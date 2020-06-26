#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

path1 = 'C://Users//yu.gan//Desktop//Churn_Modelling.csv'
df = pd.read_csv(path1, engine = 'python')

def country(x):
    if x == 'France':
        return 1
    if x == 'Germany':
        return 2
    if x == 'Spain':
        return 3
df['country'] = df['Geography'].apply(lambda x: country(x))


def gender(x):
    if x == 'Female':
        return 0
    if x == 'Male':
        return 1
df['sex'] = df['Gender'].apply(lambda x: gender(x))


def bal(x):
    if x != 0:
        return math.log(x)
    else:
        return 0
df['bal'] = df['Balance'].apply(lambda x: bal(x))
df['est_sal'] = df['EstimatedSalary'].apply(lambda x: bal(x))

df1 = df[['CustomerId', 'CreditScore','Age','Tenure','NumOfProducts','HasCrCard','IsActiveMember', 'country', 'sex','bal', 'est_sal', 'Exited' ]]
test = df1.iloc[8000:10000,]

train = df1.iloc[0:8000,:]
plt.figure(figsize=(12,10), dpi = 100)
sns.heatmap(train[['CreditScore','Age','Tenure','NumOfProducts','HasCrCard','IsActiveMember', 'country', 'sex','bal', 'est_sal', 'Exited']].corr(),xticklabels=True, yticklabels=True, cmap='RdYlGn', center=0, annot=True)
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
plt.figure(figsize=(13, 26))
box = train[['Age','Tenure','NumOfProducts','HasCrCard','IsActiveMember', 'country', 'sex','bal', 'est_sal']].boxplot()
# est_sal outliers较多
train.est_sal.describe()

plt.figure(figsize=(30, 30))
plt.hist(train['Age'], bins=40, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
plt.subplot(3,3,1)
plt.hist(train['Tenure'], bins=40, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
plt.subplot(3,3,2)
plt.hist(train['NumOfProducts'], bins=40, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
plt.subplot(3,3,3)
plt.hist(train['HasCrCard'], bins=40, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
plt.subplot(3,3,4)
plt.hist(train['IsActiveMember'], bins=40, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
plt.subplot(3,3,5)
plt.hist(train['country'], bins=40, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
plt.subplot(3,3,6)
plt.hist(train['sex'], bins=40, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
plt.subplot(3,3,7)
plt.hist(train['bal'], bins=10, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
plt.subplot(3,3,8)
plt.hist(train['est_sal'], bins=10, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
plt.subplot(3,3,9)
plt.show()

# find outliers
def percentile_swift(x):
    percentile_x_output = []
    percentile_x = np.percentile(x,(25,50,75), interpolation = 'midpoint')
    Q1 = percentile_x[0]# 上四分位
    Q3 = percentile_x[2]# 下四分位
    IQR = Q3 - Q1# 四分位距
    llim = Q1 - 1.5 * IQR
    percentile_x_output.append(llim)
    ulim = Q3 + 1.5 * IQR
    percentile_x_output.append(ulim)
    return percentile_x_output

def outliers(llim,ulim,x):
    if llim < x and x < ulim:
        return 1
    else: 
        return 0

a = percentile_swift(train.est_sal)
train['est_sal_corr'] = train['est_sal'].apply(lambda x: outliers(a[0],a[1],x))

aa = percentile_swift(train.Age)
train['Age_corr'] = train['Age'].apply(lambda x: outliers(aa[0], aa[1],x))
