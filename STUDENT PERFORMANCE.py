#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import necessary libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[2]:


#load the dataset
StudPer = pd.read_csv('student_exam_data.csv')


# In[3]:


#view the first 5 rows of the dataframe
StudPer.head()


# In[4]:


#view the properties of the dataframe
StudPer.info()


# In[5]:


#view the statistical description of the columns in the dataframe
StudPer.describe()


# In[6]:


#view the size of the dataframe
StudPer.shape


# In[36]:


# Compute the correlation matrix
corr_matrix = StudPer.corr()

# Plot the correlation matrix using a heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')

# Display the plot
plt.show()


# In[37]:


# Create a scatterplot between 'Study Hours' and 'Pass/Fail'
plt.scatter(StudPer['Study Hours'], StudPer['Pass/Fail'])
plt.xlabel('Study Hours')
plt.ylabel('Pass/Fail')
plt.title('Scatterplot of Study Hours vs Pass/Fail')
plt.show()


# In[11]:


#divide the dataframe into features(X) and class label(y)
X = StudPer.drop('Pass/Fail', axis = 1)
y = StudPer['Pass/Fail']


# In[12]:


X


# In[13]:


y


# In[14]:


#split the dataframe into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3)


# In[15]:


#standardize the train set
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[16]:


X_train


# In[17]:


X_test


# In[19]:


#view the distribution of class labels in the dataframe
y_train.value_counts()


# In[21]:


#apply SMOTE on the train set to deal with the class imbalance
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_train, y_train = smote.fit_resample(X_train, y_train)


# In[22]:


y_train.value_counts()


# In[24]:


#initiate the classifier
DTmodel = DecisionTreeClassifier()


# In[25]:


DTmodel


# In[26]:


#train the model
DTmodel.fit(X_train, y_train)


# In[27]:


#view the predicted outcomes of the model
y_predDT = DTmodel.predict(X_test)
y_predDT


# In[28]:


from sklearn import metrics


# In[29]:


accDT = metrics.accuracy_score(y_test, y_predDT)
accDT


# In[30]:


precDT = metrics.precision_score(y_test, y_predDT)
precDT


# In[31]:


recDT = metrics.recall_score(y_test, y_predDT)
recDT


# In[32]:


rocDT = metrics.roc_auc_score(y_test, y_predDT)
rocDT


# In[33]:


f1_scoreDT = metrics.f1_score(y_test, y_predDT)
f1_scoreDT


# In[34]:


cmDT = metrics.confusion_matrix(y_test, y_predDT)
cmDT


# In[35]:


#view the classification report
result_DT = metrics.classification_report(y_test, y_predDT)
print('Classification Report:\n')
print(result_DT)


# In[38]:


#visualize the confusion matrix
ax = sns.heatmap(cmDT, cmap = 'flare', annot= True, fmt = 'd')
plt.xlabel('Predicted Class', fontsize = 12)
plt.ylabel('True Class', fontsize = 12)
plt.title('Confusion Matrix', fontsize = 12)
plt.show()


# In[39]:


#Make predictions based on the training set
y_train_predDT = DTmodel.predict(X_train)

#Make predictions based on the test set
y_test_predDT = DTmodel.predict(X_test)

#Calculate the accuracy of the training set
acc_trainDT = metrics.accuracy_score(y_train, y_train_predDT)

#Calculate the accuracy of the test set
acc_testDT = metrics.accuracy_score(y_test, y_test_predDT)

print('Training Accuracy:', acc_trainDT)
print('Test Accuracy:', acc_testDT)


# In[40]:


#initiate the random forest classifier
RFmodel = RandomForestClassifier()


# In[41]:


#train the random forest classifier
RFmodel.fit(X_train, y_train)


# In[42]:


#view the predicted outcomes of the model
y_predRF = RFmodel.predict(X_test)
y_predRF


# In[43]:


accRF = metrics.accuracy_score(y_test, y_predRF)
accRF


# In[44]:


precRF = metrics.precision_score(y_test, y_predRF)
precRF


# In[45]:


recRF = metrics.recall_score(y_test, y_predRF)
recRF


# In[46]:


ROCRF = metrics.roc_auc_score(y_test, y_predRF)
ROCRF


# In[ ]:




