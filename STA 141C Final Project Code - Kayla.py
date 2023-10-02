#!/usr/bin/env python
# coding: utf-8

# Predictors: Year, Status, Mileage, Price

# In[141]:


import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
from sklearn.impute import SimpleImputer

os.chdir('/Users/kaylapham/Downloads')

# Load the dataset
data = pd.read_csv('car_data.csv')

# Remove entries that don't have information on Price, MSRP, and/or Mileage
data = data[data['Price'] != 'Not Priced']
data = data[data['MSRP'] != 'Not specified']
data = data[data['Mileage'] != 'Not available']
data = data.iloc[:,1:]

# Convert the "price" column to numeric
data['Price'] = pd.to_numeric(data['Price'].str.replace('$', '').str.replace(',', ''))
# Convert the "Mileage" column to numeric
data['Mileage'] = pd.to_numeric(data['Mileage'].str.replace(' mi.', '').str.replace(',', ''))
# Convert the "Status" column to binary 
# 0 if Used; 1 if anything else
mask = data['Status'] == 'Used'
data.loc[mask, 'Status'] = 0
data.loc[~mask, 'Status'] = 1
# Convert the "MSRP" column to numeric
# For the entries that say price drop, 
# remove the 'price drop' string
# and subtract the price drop amount from the entry in 'Price' column
mask = data['MSRP'].str.contains('price drop')
data.loc[mask, 'MSRP'] = pd.to_numeric(data.loc[mask, 'Price']) - pd.to_numeric(data.loc[mask, 'MSRP'].str.extract(r'\$(\d+)', expand=False))
for i, row in data.iterrows():
    msrp_val = row['MSRP']
    if isinstance(msrp_val, str):
        if 'MSRP' in msrp_val:
            if '$' in msrp_val:
                data.at[i, 'MSRP'] = float(msrp_val.replace('MSRP', '').replace('$', '').replace(',', ''))
        elif 'Price Drop' in msrp_val:
            data.at[i, 'MSRP'] = pd.to_numeric(row['Price']) - pd.to_numeric(msrp_val.replace('price drop ', ''))
    

data


# In[151]:


x = imputer.fit_transform(data.drop(['Model'], axis=1))
y = data['MSRP']

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize the KNN regressor and fit the model on the training data
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(x_train, y_train)

# Predict the values for the testing set
y_pred = knn.predict(x_test)

# Calculate the mean squared error of the predictions
mse = mean_squared_error(y_test, y_pred)
print('Mean squared error:', mse)


# In[147]:


import matplotlib.pyplot as plt

# Plot the predicted vs. actual values
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([0, 400000], [0, 400000], 'r--') 
plt.xlabel('Actual MSRP')
plt.ylabel('Predicted MSRP')
plt.title('KNN Regression Results')
plt.show()


# In[171]:


x = imputer.fit_transform(data.drop(['Model', 'Mileage'], axis=1))
y = data['MSRP']

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize the KNN regressor and fit the model on the training data
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(x_train, y_train)

# Predict the values for the testing set
y_pred = knn.predict(x_test)

# Calculate the mean squared error of the predictions
mse = mean_squared_error(y_test, y_pred)
print('Mean squared error:', mse)


# Removing the outlier

# In[191]:


df = pd.DataFrame({'Y Test': y_test, 'Y Pred': y_pred})
# Find the index of the outlier in the dataframe
outlier_index = df[(df['Y Test'] > 500000)].index


# Drop the outlier row from the dataframe
df = df.drop(outlier_index)
df


# In[194]:


# Calculate the mean squared error of the predictions if we remove the outlier
mse = mean_squared_error(df['Y Test'], df['Y Pred'])
print('Mean squared error:', mse)


# The MSE signficantly decreases when we remove the outlier.

# In[195]:


# Plot the predicted vs. actual values
plt.scatter(df['Y Test'], df['Y Pred'], alpha=0.5)
plt.plot([0, 400000], [0, 400000], 'r--') 
plt.xlabel('Actual MSRP')
plt.ylabel('Predicted MSRP')
plt.title('KNN Regression Results')
plt.show()


# Predictors: Status, Price
# 

# In[153]:


x = imputer.fit_transform(data.drop(['Model', 'Year','Mileage'], axis=1))
y = data['MSRP']

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize the KNN regressor and fit the model on the training data
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(x_train, y_train)

# Predict the values for the testing set
y_pred = knn.predict(x_test)

# Calculate the mean squared error of the predictions
mse = mean_squared_error(y_test, y_pred)
print('Mean squared error:', mse)


# Predictors: Status

# In[164]:


x = imputer.fit_transform(data.drop(['Model', 'Year','Price','Mileage'], axis=1))
y = data['MSRP']

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize the KNN regressor and fit the model on the training data
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(x_train, y_train)

# Predict the values for the testing set
y_pred = knn.predict(x_test)

# Calculate the mean squared error of the predictions
mse = mean_squared_error(y_test, y_pred)
print('Mean squared error:', mse)

