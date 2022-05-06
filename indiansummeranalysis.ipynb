#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[21]:


import pandas as pd


# In[22]:


import seaborn as sns


# In[11]:


import matplotlib.pyplot as plt


# In[12]:


import warnings


# In[13]:


warnings.filterwarnings('ignore')


# In[90]:


summer_data = pd.read_csv('C:\\Users\\Administrator\\Downloads\\indian_summer.csv') 


# In[91]:


summer_data.head()


# In[10]:


summer_data.tail() 


# In[11]:


summer_data.shape


# In[12]:


summer_data.columns


# In[13]:


summer_data.info()


# In[14]:


summer_data.describe()


# In[15]:


summer_data.isnull().sum()


# In[17]:


summer_data = summer_data.drop(['sealevelpressure'], axis = 1) 


# In[18]:


summer_data.columns


# In[20]:


summer_data.dropna(inplace = True)


# In[21]:


summer_data.shape


# In[22]:


summer_data['conditions'].unique() 


# In[23]:


summer_data['conditions'].value_counts() 


# In[24]:


plt.figure(figsize=(15,6)) 
sns.countplot('conditions', data = summer_data, palette='hls') 
plt.xticks(rotation = 90) 
plt.show()


# In[27]:


count_clear=len(summer_data[summer_data.conditions=="Clear"]) 
count_pcloudy=len(summer_data[summer_data.conditions=="Partially cloudy"]) 
count_rpcloudy=len(summer_data[summer_data.conditions=="Rain, Partially cloudy"]) 
count_ro=len(summer_data[summer_data.conditions=="Rain, Overcast"]) 
count_overcast=len(summer_data[summer_data.conditions=="Overcast"]) 
count_rain=len(summer_data[summer_data.conditions=="Rain"])


# In[28]:


print("Percent of Clear:{:2f}%".format((count_clear/(len(summer_data.conditions))*100)))
print("Percent of Partial Cloudy:{:2f}%".format((count_pcloudy/(len(summer_data.conditions))*100)))
print("Percent of Rain Partial Cloudy:{:2f}%".format((count_rpcloudy/(len(summer_data.conditions))*100)))
print("Percent of Rain Overcast:{:2f}%".format((count_ro/(len(summer_data.conditions))*100)))
print("Percent of Overcast:{:2f}%".format((count_overcast/(len(summer_data.conditions))*100)))
print("Percent of Rain:{:2f}%".format((count_rain/(len(summer_data.conditions))*100)))


# In[29]:


summer_data[["humidity","tempmax","tempmin","windspeed"]].describe()


# In[30]:


sns.set(style="darkgrid") 
fig,axs=plt.subplots(2,2,figsize=(10,8)) 
sns.histplot(data=summer_data,x="humidity",kde=True,ax=axs[0,0],color='green') 
sns.histplot(data=summer_data,x="tempmax",kde=True,ax=axs[0,1],color='red') 
sns.histplot(data=summer_data,x="tempmin",kde=True,ax=axs[1,0],color='skyblue') 
sns.histplot(data=summer_data,x="windspeed",kde=True,ax=axs[1,1],color='orange')


# In[31]:


sns.set(style="darkgrid") 
fig,axs=plt.subplots(2,2,figsize=(10,8)) 
sns.violinplot(data=summer_data,x="humidity",kde=True,ax=axs[0,0],color='green') 
sns.violinplot(data=summer_data,x="tempmax",kde=True,ax=axs[0,1],color='red') 
sns.violinplot(data=summer_data,x="tempmin",kde=True,ax=axs[1,0],color='skyblue') 
sns.violinplot(data=summer_data,x="windspeed",kde=True,ax=axs[1,1],color='yellow')


# In[32]:


plt.figure(figsize=(12,6)) 
sns.boxplot("humidity","conditions",data=summer_data,palette="YlOrBr")


# In[33]:


plt.figure(figsize=(12,6)) 
sns.boxplot("windspeed","conditions",data=summer_data,palette="inferno")


# In[34]:


plt.figure(figsize=(12,6)) 
sns.boxplot("tempmin","conditions",data=summer_data,palette="inferno")


# In[35]:


plt.figure(figsize=(12,7)) 
sns.heatmap(summer_data.corr(),annot=True,cmap='coolwarm')


# In[36]:


from scipy import stats


# In[37]:


summer_data.plot("humidity","tempmax",style='o') 
print("Pearson correlation:",summer_data["humidity"].corr(summer_data["tempmax"])) 
print("T Test and P value:",stats.ttest_ind(summer_data["humidity"],summer_data["tempmax"]))


# In[38]:


summer_data.plot("windspeed","tempmax",style='o') 
print("Pearson correlation:",summer_data["windspeed"].corr(summer_data["tempmax"])) 
print("T Test and P value:",stats.ttest_ind(summer_data["windspeed"],summer_data["tempmax"]))


# In[40]:


summer_data.plot("tempmin","tempmax",style='o')
print("Pearson correlation:",summer_data["tempmin"].corr(summer_data["tempmax"])) 
print("T Test and P value:",stats.ttest_ind(summer_data["tempmin"],summer_data["tempmax"]))


# In[48]:


df=summer_data.drop(['Date', 'sunrise', 'sunset', 'description'],axis=1)


# In[54]:


Q1=df.quantile(0.25) 
Q3=df.quantile(0.75) 
IQR=Q3-Q1 
df=df[~((df<(Q1-1.5*IQR))|(df>(Q3+1.5*IQR))).any(axis=1)] 


# In[56]:



sns.set(style="darkgrid") 
fig,axs=plt.subplots(2,2,figsize=(10,8)) 
sns.histplot(data=df,x="humidity",kde=True,ax=axs[0,0],color='green') 
sns.histplot(data=df,x="tempmax",kde=True,ax=axs[0,1],color='red') 
sns.histplot(data=df,x="tempmin",kde=True,ax=axs[1,0],color='skyblue') 
sns.histplot(data=df,x="windspeed",kde=True,ax=axs[1,1],color='orange')


# In[57]:


df.head()


# In[58]:


df1 = df.drop(['City'], axis = 1) 


# In[72]:


df1.head()


# In[130]:


from sklearn.preprocessing import StandardScaler,LabelEncoder 
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC 
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[131]:


lc=LabelEncoder() 
df1["conditions"]=lc.fit_transform(df1["conditions"]) 


# In[129]:


df1.head()


# In[ ]:





# In[147]:





# In[ ]:





# In[ ]:





# In[ ]:




