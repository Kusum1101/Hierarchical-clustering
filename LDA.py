#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import Lib 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


#Import dataset 
data= pd.read_csv("C:\\Users\\dell\\Downloads\\Wine-211105-185251.csv")


# In[3]:


data


# In[4]:


x= data.iloc[:,:-1]


# In[5]:


x


# In[6]:


y= data.iloc[:,-1]


# In[7]:


y


# In[8]:


#split the dataset into training and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=42)


# In[9]:


x_train


# In[10]:


y_train


# In[11]:


x_test


# In[12]:


y_test


# In[13]:


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[14]:


x_train


# In[15]:


x_test


# In[16]:


#Apply LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
x_train = lda.fit_transform(x_train, y_train)
x_test = lda.transform(x_test)


# In[17]:


#Logistic Regression
from sklearn.linear_model import LogisticRegression 
lr=  LogisticRegression()
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)


# In[18]:


y_pred


# In[19]:


y_test


# In[22]:


#Confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)


# In[23]:


#accuracy score 
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# In[41]:


#Visualising Training Set Results
from matplotlib.colors import ListedColormap
x_set, y_set = x_train, y_train
x1, x2 = np.meshgrid(
    np.arange(start = x_set[: ,0].min()-1, stop = x_set[: , 0].max() +1 , step = 0.25),
    np.arange(start = x_set[: ,1].min()-1, stop = x_set[: , 1].max() +1 , step = 0.25),
     )

plt.contourf(x1,x2, lr.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
            alpha = 0.75 , cmap= ListedColormap(('red','blue','green')))

plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())

for i,j  in  enumerate(np.unique(y_set)):
    
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j,1],
               c =ListedColormap(('red','blue','green'))(i), label = j)
    
   
    plt.title("Training Set")
    plt.xlabel("LD1")
    plt.ylabel("LD2")
    plt.legend()
    
            


# In[43]:



#Visualising Test Set Results
from matplotlib.colors import ListedColormap
x_set, y_set = x_test, y_test
x1, x2 = np.meshgrid(
    np.arange(start = x_set[: ,0].min()-1, stop = x_set[: , 0].max()+1 , step = 0.25),
    np.arange(start = x_set[: ,1].min()-1, stop = x_set[: , 1].max()+1 , step = 0.25),
)

plt.contourf(x1,x2, lr.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
            alpha =0.75 , cmap= ListedColormap(('red','blue','green')))

plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())

for i,  j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j,1],
               c =ListedColormap(('red','blue','green'))(i), label = j)
    plt.title("Test Set")
    plt.xlabel("LD1")
    plt.ylabel("LD2")
    plt.legend()
      


# In[ ]:




