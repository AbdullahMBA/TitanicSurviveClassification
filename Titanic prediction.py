
# coding: utf-8

# In[30]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.preprocessing import Imputer


# In[31]:


train = pd.read_csv('train.csv')


# In[32]:


train.head()


# In[33]:


train.isnull()


# In[34]:


sns.heatmap(train.isnull() ,yticklabels=False , cbar=False , cmap='viridis')


# In[35]:


sns.set_style('whitegrid')


# In[36]:


sns.countplot(x='Survived' ,hue='Pclass', data = train)


# In[37]:


sns.distplot(train['Age'].dropna(),kde=False,bins=30)


# In[38]:


train.info()


# In[39]:


sns.countplot(x='SibSp',data=train)


# In[40]:


sns.distplot(train['Fare'],bins=40)


# In[41]:


train['Age'].dropna()


# In[42]:


train


# In[43]:


sns.heatmap(train.isnull(),yticklabels=False , cbar = 'viridis')


# In[85]:


X = train.iloc[:,:-1].values


# In[45]:




# In[46]:




# In[47]:


plt.figure(figsize=(10,7))
sns.boxplot(x='Pclass' , y = 'Age' , data=train)


# In[48]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 28
        else:
            return 23
    else:
        return Age
    


# In[49]:


train['Age'] = train[['Age' , 'Pclass']].apply(impute_age , axis = 1)


# In[50]:


sns.heatmap(train.isnull() , cmap='viridis'  , yticklabels=False , cbar=False)


# In[51]:


train.drop('Cabin' , axis= 1 )


# In[52]:


train.dropna(inplace=True)


# In[53]:


sns.heatmap(train.isnull() , cmap='viridis'  , yticklabels=False , cbar=False)


# In[54]:


train.Sex[train.Sex == 'male'] = 0
train.Sex[train.Sex == 'female'] = 1


# In[55]:


train.head()


# In[58]:


sex = pd.get_dummies(train['Sex'], drop_first=True)


# In[60]:


embark = pd.get_dummies(train['Embarked'], drop_first=True)


# In[61]:


embark.head()


# In[62]:


train = pd.concat([train , sex, embark],axis=1)


# In[63]:


train.head(2)


# In[64]:


train.drop(['Sex' , 'Embarked' , 'Name' , 'Ticket'],axis=1  ,inplace=True)


# In[70]:


train


# In[71]:


train.drop('PassengerId' , axis=1 , inplace=True)


# In[72]:


train.drop('Cabin' , axis= 1 , inplace=True)


# In[73]:


train


# In[74]:


X = train.drop('Survived' , axis=1)
y = train['Survived']


# In[75]:


from sklearn.cross_validation import train_test_split


# In[76]:


X_test ,y_train , X_train , y_test = train_test_split(X,y , test_size = 0.3 , random_state = 101)


# In[77]:


from sklearn.linear_model import LogisticRegression


# In[93]:


logmodel = LogisticRegression()


# In[94]:


logmodel.fit(X_train , y_train)

print(train)

