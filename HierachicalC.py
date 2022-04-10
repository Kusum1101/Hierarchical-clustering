#!/usr/bin/env python
# coding: utf-8

# In[37]:


#Hierarchical Clustering 


# In[38]:


#Import the libraries
import pandas as pd
import matplotlib.pyplot as plt


# In[39]:


#import dataset
data= pd.read_csv("C:\\Users\\dell\\Downloads\\Mall_Customers-211105-191711.csv")
X = data.iloc[:, :].values


# In[40]:


X


# In[41]:


#how many clusters we want by using dendogram to find the optimal no. of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title("Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Euclidean distance")
plt.show()


# In[42]:


#Train the model 
from sklearn.cluster import AgglomerativeClustering
clustering = AgglomerativeClustering(n_clusters = 5)
y_hc = clustering.fit_predict(X)


# In[43]:


y_hc


# In[44]:


#visualising the clusters 
plt.scatter(X[y_hc== 0 , 0], X[y_hc==0 , 1], c = 'red' , label = 'cluster 1')
plt.scatter(X[y_hc== 1 , 0], X[y_hc==1 , 1], c = 'green' , label = 'cluster 2')
plt.scatter(X[y_hc== 2 , 0], X[y_hc==2 , 1], c = 'pink' , label = 'cluster 3')
plt.scatter(X[y_hc== 3 , 0], X[y_hc==3 , 1], c = 'orange' , label = 'cluster 4')
plt.scatter(X[y_hc== 4 , 0], X[y_hc==4 , 1], c = 'blue' , label = 'cluster 5')
plt.title("Cluster of Customers")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score(1-100)")
plt.legend()
plt.show()


# In[45]:


X[ y_hc == 0, 1]
#Annual income of all the points belonging to cluster 0


# In[46]:


X[y_hc == 2 , 1]


# In[ ]:




