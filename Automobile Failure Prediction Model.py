#!/usr/bin/env python
# coding: utf-8

# # AUTOMOBILE/MACHINE FAILURE PREDICTION

# # Importing libraries and CSV file

# In[2]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[3]:


df = pd.read_csv("train.csv")
df.head()


# In[57]:


df.columns


# Type: A categorical variable that represents the machine type. This variable can be used to specify different machine types or models.
# 
# Air temperature [K]: A continuous variable that measures the air temperature in Kelvin in the environment where the process takes place (for example, a factory or workplace).
# 
# 

# # Exploratory Data Analysis
# 
# Describe,
# Null value check,
# Correlation heatmap,
# Scatter plot,
# Histplot,

# In[4]:


df.describe()


# In[13]:


df['Type'].value_counts()


# In[12]:


df.isnull().sum()


# In[4]:


df.info()


# In[5]:


fig, ax = plt.subplots(figsize=(15,10))  
sns.heatmap(df.corr(),annot=True,lw=2,linecolor='white',cmap='coolwarm')


# In[10]:


sns.pairplot(data = df)


# In[15]:


sns.histplot(df['Air temperature [K]'])
plt.xlabel('Air temperature [K]')
plt.ylabel('Count')
plt.title('Distribution of Air Temperature')
plt.show()


# In[17]:


sns.barplot(x='Type', y='Machine failure', data=df)
plt.xlabel('Type')
plt.ylabel('Machine failure')
plt.title('Machine Failure by Type')
plt.show()


# In[18]:


sns.scatterplot(x='Torque [Nm]', y='Rotational speed [rpm]', hue='Machine failure', data=df)
plt.xlabel('Torque [Nm]')
plt.ylabel('Rotational speed [rpm]')
plt.title('Machine Failure by Torque and Rotational Speed')
plt.legend()
plt.show()


# In[19]:


# Target Variable Distribution
sns.countplot(x='Machine failure', data=df)
plt.xlabel('Machine failure')
plt.ylabel('Count')
plt.title('Distribution of Machine Failure')
plt.show()


# # Encoding of categorical cols

# In[23]:


from sklearn.preprocessing import OrdinalEncoder

categorical_cols = ["Type"]

# Low, medium, high, in order (0, 1, 2)
encoder = OrdinalEncoder(categories=[['L','M','H']])
df[categorical_cols] = encoder.fit_transform(df[categorical_cols])
df.describe()


# # Feature Engineering 

# Combining errors in one coulumn

# In[24]:


df.columns = df.columns.str.replace('[\[\]]', '', regex=True)
df


# In[25]:


df['Faliures'] = df['TWF'] + df['PWF'] + df['OSF'] + df['RNF'] + df['HDF']


# In[26]:


df["Power"] = df["Torque Nm"] * df["Rotational speed rpm"]
df["Power"].head()


# In[27]:


df["temp_ratio"] = df["Process temperature K"] / df["Air temperature K"]
df["temp_ratio"].head()


# In[28]:


df["Process temperature C"] = df["Process temperature K"] - 273.15
df["Process temperature C"].head()


# In[29]:


df["Air temperature C"] = df["Air temperature K"] - 273.15
df["Air temperature C"].head()


# In[30]:


df["temp_C_ratio"] = df["Process temperature C"] / df["Air temperature C"]
df["temp_C_ratio"].head()


# In[31]:


df["tool_wear_speed"] = df["Tool wear min"] * df["Rotational speed rpm"]
df["tool_wear_speed"].head()


# In[32]:


df["torque wear ratio"] = df["Torque Nm"] / (df["Tool wear min"] + 0.0001)
df["torque times wear"] = df["Torque Nm"] * df["Tool wear min"]
df.head()


# In[33]:


df["torque wear ratio"] = df["Torque Nm"] / (df["Tool wear min"] + 0.0001)
df["torque times wear"] = df["Torque Nm"] * df["Tool wear min"]
df.head()


# In[35]:


df


# GETTING PRODUCT ID NUMERIC 

# In[36]:


df["product_id_num"] = pd.to_numeric(df["Product ID"].str.slice(start=1))
df[["Product ID", "product_id_num"]].head()


# # SPLITING INTO TRAIN AND VALIDATION
# 
# 
# Using Random Forest 
# 

# In[37]:


X = df.drop(["id", "Product ID",'Machine failure'], axis=1)
Y = df['Machine failure']


# In[38]:


from sklearn.model_selection import train_test_split


# In[39]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.75, random_state = 101)


# # RandomForest

# In[40]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 5)
rf.fit(X_train,Y_train)
predict = rf.predict(X_test)


# 

# In[41]:


from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
print(classification_report(Y_test,predict))
print(roc_auc_score(Y_test, [x[1] for x in rf.predict_proba(X_test)]))


# In[42]:


# Feature importance
feature_importance = rf.feature_importances_

# Sort feature importance in descending order
sorted_idx = np.argsort(feature_importance)[::-1]

# Sort feature names accordingly
feature_names = X_train.columns[sorted_idx]

# Visualize feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance[sorted_idx], y=feature_names)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Random Forest Feature Importance')
plt.show()


# # Logistic Regression

# In[43]:


from sklearn.linear_model import LogisticRegression


# In[44]:


lr = LogisticRegression()
lr.fit(X_train,Y_train)
predict = lr.predict(X_test)


# In[45]:


print(classification_report(Y_test,predict))
print(roc_auc_score(Y_test, [x[1] for x in rf.predict_proba(X_test)]))


# # GradientBoostingClassifier

# In[46]:


from sklearn.ensemble import GradientBoostingClassifier


# In[49]:


gbr = GradientBoostingClassifier()
gbr.fit(X_train,Y_train)
predict = gbr.predict(X_test)


# In[50]:


print(classification_report(Y_test,predict))
print(roc_auc_score(Y_test, [x[1] for x in rf.predict_proba(X_test)]))


# 

# In[ ]:





# # Ensembleing Decision Tree and Gradient Boost

# In[51]:


from sklearn.ensemble import VotingClassifier

model = VotingClassifier(estimators=[('gbr', gbr),
                                     ('rf', rf)],
                        voting='soft')

model.fit(X_train, Y_train)

print(roc_auc_score(Y_test, [x[1] for x in model.predict_proba(X_test)]))


# # PROSSESSING TEST DATA

# In[52]:


test_df = pd.read_csv("test.csv")
test_df[categorical_cols] = encoder.transform(test_df[categorical_cols])
test_df.columns = test_df.columns.str.replace('[\[\]]', '', regex=True)
test_df["Power"] = test_df["Torque Nm"] * test_df["Rotational speed rpm"]
test_df["temp_ratio"] = test_df["Process temperature K"] / test_df["Air temperature K"]
test_df["Process temperature C"] = test_df["Process temperature K"] - 273.15
test_df["Air temperature C"] = test_df["Air temperature K"] - 273.15
test_df["temp_C_ratio"] = test_df["Process temperature C"] / test_df["Air temperature C"]
test_df["Failure Sum"] = (test_df["TWF"] +
                            test_df["HDF"] +
                            test_df["PWF"] +
                            test_df["OSF"] +
                            test_df["RNF"])

test_df["tool_wear_speed"] = test_df["Tool wear min"] * test_df["Rotational speed rpm"]
test_df["torque wear ratio"] = test_df["Torque Nm"] / (test_df["Tool wear min"] + 0.0001)
test_df["torque times wear"] = test_df["Torque Nm"] * test_df["Tool wear min"]
test_df["product_id_num"] = pd.to_numeric(test_df["Product ID"].str.slice(start=1))

ids = test_df["id"]
test_X = test_df.drop(["id", "Product ID"], axis=1)


# 
# # Prediction and Submission

# In[53]:


ensemble_preds = model.predict_proba(test_X)
predicted_probs_ensemble = np.array([pred[1] for pred in ensemble_preds])


# In[54]:


submission_df = pd.DataFrame({
"id" : ids,
"Machine failure": predicted_probs_ensemble
})
submission_df.shape


# In[55]:


submission_df.tail(10)


# In[49]:


submission_df.to_csv("submission.csv", index=False)


# In[ ]:





# # Thank you
# 

# In[ ]:




