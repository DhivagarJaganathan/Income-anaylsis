#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data=pd.read_csv("C:\\Users\\DHIVAGAR\\Desktop\\Data files\\adult_data.csv")


# In[3]:


data.head()


# In[4]:


data.tail()


# In[5]:


data.shape


# In[6]:


data.dtypes


# In[7]:


data.info()


# In[8]:


data.isnull().sum()


# In[9]:


data.corr()


# In[10]:


sns.heatmap(data.corr(),annot=True)


# In[11]:


data.drop([" fnlwgt"],axis=1,inplace=True)


# In[ ]:





# In[12]:


data.head()


# In[13]:


sns.distplot(data.age)


# In[14]:


data.columns.values.tolist()


# In[15]:


data.columns=["age","workclass","education","eduNum","maritalsts","occupation","relationship","race","sex","capitalgain","capitalloss","hour_per_week","country","salary"]


# In[16]:


import sklearn


# In[17]:


from sklearn.preprocessing import LabelEncoder


# In[18]:


le=LabelEncoder()


# In[19]:


data["salary"]=le.fit_transform(data.salary.values)


# In[20]:


data.head()


# In[117]:


plt.figure(figsize=(60,30))
sns.countplot(data["age"],hue=data["salary"])
plt.yticks(size=50)
plt.xticks(size=30)
plt.show()


# In[22]:


sns.distplot(data.hour_per_week)


# In[23]:


data.workclass.unique()


# In[24]:


plt.figure(figsize=(20,5))
sns.countplot(data["workclass"],hue=data["salary"])


# In[25]:


plt.figure(figsize=(20,5))
sns.countplot(data["education"],hue=data["salary"])


# In[26]:


data.occupation.unique()


# In[27]:


plt.figure(figsize=(40,10))
sns.countplot(data["occupation"],hue=data["salary"])
plt.yticks(size=20)
plt.xticks(size=20)
plt.xlabel("occupation",fontsize=20)
plt.ylabel("count",fontsize=20)


# In[28]:


plt.figure(figsize=(10,5))
sns.countplot(data["sex"],hue=data["salary"])


# In[29]:


plt.figure(figsize=(20,5))
sns.countplot(data["hour_per_week"],hue=data["salary"])


# In[30]:


import sklearn


# In[31]:


from sklearn.preprocessing import LabelEncoder


# In[32]:


le=LabelEncoder()


# In[33]:


for i in range(0,data.shape[1]):
    if data.dtypes[i]=="object":
        data[data.columns[i]]=le.fit_transform(data[data.columns[i]])


# In[34]:


data.head()


# In[35]:


x=data.drop(["salary"],axis=1)


# In[36]:


y=data["salary"]


# In[37]:


from sklearn.preprocessing import StandardScaler


# In[38]:


sc=StandardScaler()


# In[39]:


x=sc.fit_transform(x)


# In[40]:


from sklearn.model_selection import train_test_split


# In[41]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2)


# In[42]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[43]:


from sklearn.preprocessing import StandardScaler


# In[44]:


sc=StandardScaler()


# In[45]:


x=sc.fit_transform(x)


# In[46]:


from sklearn.tree import DecisionTreeClassifier


# In[47]:


dc=DecisionTreeClassifier(criterion='gini',max_depth=100)


# In[48]:


dc.fit(x_train,y_train)


# In[49]:


pred=dc.predict(x_test)


# In[50]:


pred[0:5]


# In[51]:


y_test[0:5]


# In[52]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[53]:


print(confusion_matrix(pred,y_test))


# In[54]:


print(classification_report(pred,y_test))


# In[55]:


accuracy_score(pred,y_test)


# In[56]:


from sklearn.ensemble import RandomForestClassifier


# In[57]:


rf=RandomForestClassifier(n_estimators=100,criterion='entropy',)


# In[58]:


rf.fit(x_train,y_train)


# In[59]:


pred3=rf.predict(x_test)


# In[60]:


pred3[0:5]


# In[61]:


y_test[0:5]


# In[62]:


print(confusion_matrix(pred3,y_test))
print(classification_report(pred3,y_test))
print(accuracy_score(pred3,y_test))


# In[63]:


from sklearn.ensemble import AdaBoostClassifier


# In[64]:


ad=AdaBoostClassifier(n_estimators=200)


# In[65]:


ad.fit(x_train,y_train)


# In[66]:


pred4=ad.predict(x_test)


# In[67]:


print(confusion_matrix(pred4,y_test))
print(classification_report(pred4,y_test))
print(accuracy_score(pred4,y_test))


# # Testing the test dataset with machine learning algorithms

# In[68]:


data1=pd.read_csv("C:\\Users\\DHIVAGAR\\Desktop\\Data files\\adult_test.csv")


# In[69]:


data1.head()


# In[70]:


data1.isnull().sum()


# In[71]:


data1.tail()


# In[ ]:





# In[72]:


data1.head()


# In[73]:


print(data1.columns)


# In[74]:


data1[' workclass']=data1[' workclass'].str.replace("?"," ")


# In[75]:


data1.head()


# In[76]:


data1[' occupation']=data1[' occupation'].str.replace("?"," ")


# In[77]:


data1.head()


# In[78]:


data1.isnull().sum()


# In[79]:


data1.corr()


# In[80]:


data1.drop([" fnlwgt"],axis=1,inplace=True)


# In[81]:


from sklearn.preprocessing import LabelEncoder


# In[82]:


le=LabelEncoder()


# In[83]:


for i in range(0,data1.shape[1]):
    if data1.dtypes[i]=="object":
        data1[data1.columns[i]]=le.fit_transform(data1[data1.columns[i]])


# In[84]:


data1.head()


# In[85]:


x=data1.drop([" salary"],axis=1)


# In[86]:


y=data1[" salary"]


# In[87]:


from sklearn.model_selection import train_test_split


# In[88]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2)


# In[89]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[90]:


from sklearn.preprocessing import StandardScaler


# In[91]:


sc=StandardScaler()


# In[92]:


x=sc.fit_transform(x)


# In[93]:


from sklearn.tree import DecisionTreeClassifier


# In[94]:


dc=DecisionTreeClassifier(criterion='gini',max_depth=100)


# In[95]:


dc.fit(x_train,y_train)


# In[96]:


pred=dc.predict(x_test)


# In[97]:


pred[0:5]


# In[98]:


y_test[0:5]


# In[99]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[100]:


print(confusion_matrix(pred,y_test))
print(classification_report(pred,y_test))
print(accuracy_score(pred,y_test))


# In[101]:


from sklearn.ensemble import RandomForestClassifier


# In[102]:


rf=RandomForestClassifier(n_estimators=100,criterion='entropy',)


# In[103]:


rf.fit(x_train,y_train)


# In[104]:


pred1=rf.predict(x_test)


# In[105]:


print(confusion_matrix(pred1,y_test))
print(classification_report(pred1,y_test))
print(accuracy_score(pred1,y_test))


# In[106]:


from sklearn.ensemble import AdaBoostClassifier


# In[107]:


ad=AdaBoostClassifier(n_estimators=200)


# In[108]:


ad.fit(x_train,y_train)


# In[109]:


pred2=ad.predict(x_test)


# In[110]:


print(confusion_matrix(pred2,y_test))
print(classification_report(pred2,y_test))
print(accuracy_score(pred2,y_test))

