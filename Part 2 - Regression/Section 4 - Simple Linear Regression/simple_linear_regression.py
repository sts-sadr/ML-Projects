# Simple Linear Regression
# Data Preprocessing

# Importing the libraries
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values # Independent variable vector
y = dataset.iloc[:,1].values # Dependent variable vector

#Taking care of Missing Data
"""from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0) #info about Imputer can be accessed using CTRL + I
imputer = imputer.fit(X[:,1:3]) # Fit Imputer object to the matrix set X, Imputer object is fitted on matrix X
X[:, 1:3] = imputer.transform(X[:, 1:3]) # Replace missing data of X by mean of the column"""

# Encoding Categorical Data, We need encoded categorical data (numerical values) because equations need numeric values and to use the categorical data in the equations we will have to transform them to numerical values
"""from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder() # LabelEncoder Transforms Non-numerical data to Numerical data
X[:, 0] = labelencoder_X.fit_transform(X[:, 0]) #LabelEncoder transforms all the values of column that contains country names to numerical encoded values
#Dummy Encoding: It is required because Machine Learning equations might think that the value corresponding to say France (0) is smaller than Spain (2) and use it in ordering the values; Basically according to the ML equations Spain would be larger than France on that particular basis. Dummy Encoding divides each value of the categorical data into a separate column, adds one to the column when a particular categorical data is being accesed (other categorical data have 0's in that column)
onehotencoder = OneHotEncoder(categorical_features = [0]) #CTRL + I to check more about OneHotEncoder, 0 is the index of categorical column consisting of the country names
X = onehotencoder.fit_transform(X).toarray()
 # Since the last column Purchased of data set is a dependent variable the ML Model will know that it is a category therefore we dont need to use OneHotEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)"""
 
 # Splitting the Dataset into the Training set and Test set
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test Set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary VS Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary VS Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()



