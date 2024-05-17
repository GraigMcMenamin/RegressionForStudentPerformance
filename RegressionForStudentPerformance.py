"""

Original file is located at
    https://colab.research.google.com/drive/1qYZ6n2ILTgaM6SXaTZvDGhCIwB0uJJ-z

**OVERVIEW**
<p>Group 1 - Graig McMenamin
<p>Regression Task
<p>Target Column: G3 (final grades on 0-20 scale)
<p>Dataset: https://archive.ics.uci.edu/dataset/320/student+performance

# Description of Dataset Attributes
"""

# @title
# DESCRIPTION OF DATASET ATTRIBUTES
# 1 school - student's school (binary: "GP" - Gabriel Pereira or "MS" - Mousinho da Silveira)
# 2 sex - student's sex (binary: "F" - female or "M" - male)
# 3 age - student's age (numeric: from 15 to 22)
# 4 address - student's home address type (binary: "U" - urban or "R" - rural)
# 5 famsize - family size (binary: "LE3" - less or equal to 3 or "GT3" - greater than 3)
# 6 Pstatus - parent's cohabitation status (binary: "T" - living together or "A" - apart)
# 7 Medu - mother's education (numeric: 0 - none,  1 - primary education (4th grade), 2 – 5th to 9th grade, 3 – secondary education or 4 – higher education)
# 8 Fedu - father's education (numeric: 0 - none,  1 - primary education (4th grade), 2 – 5th to 9th grade, 3 – secondary education or 4 – higher education)
# 9 Mjob - mother's job (nominal: "teacher", "health" care related, civil "services" (e.g. administrative or police), "at_home" or "other")
# 10 Fjob - father's job (nominal: "teacher", "health" care related, civil "services" (e.g. administrative or police), "at_home" or "other")
# 11 reason - reason to choose this school (nominal: close to "home", school "reputation", "course" preference or "other")
# 12 guardian - student's guardian (nominal: "mother", "father" or "other")
# 13 traveltime - home to school travel time (numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour)
# 14 studytime - weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)
# 15 failures - number of past class failures (numeric: n if 1<=n<3, else 4)
# 16 schoolsup - extra educational support (binary: yes or no)
# 17 famsup - family educational support (binary: yes or no)
# 18 paid - extra paid classes within the course subject (Math or Portuguese) (binary: yes or no)
# 19 activities - extra-curricular activities (binary: yes or no)
# 20 nursery - attended nursery school (binary: yes or no)
# 21 higher - wants to take higher education (binary: yes or no)
# 22 internet - Internet access at home (binary: yes or no)
# 23 romantic - with a romantic relationship (binary: yes or no)
# 24 famrel - quality of family relationships (numeric: from 1 - very bad to 5 - excellent)
# 25 freetime - free time after school (numeric: from 1 - very low to 5 - very high)
# 26 goout - going out with friends (numeric: from 1 - very low to 5 - very high)
# 27 Dalc - workday alcohol consumption (numeric: from 1 - very low to 5 - very high)
# 28 Walc - weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)
# 29 health - current health status (numeric: from 1 - very bad to 5 - very good)
# 30 absences - number of school absences (numeric: from 0 to 93)

# # these grades are related with the course subject, Math or Portuguese:
# 31 G1 - first period grade (numeric: from 0 to 20)
# 31 G2 - second period grade (numeric: from 0 to 20)
# 32 G3 - final grade (numeric: from 0 to 20, output target)

# Additional note: there are several (382) students that belong to both datasets .
# These students can be identified by searching for identical attributes
# that characterize each student, as shown in the annexed R file.

"""# Step 1 Read Dataset

"""

import csv
import os

# THIS FUNCTION CONVERTS ALL SEMICOLONS IN THE DATASET TO COMMAS
# It outputs a new "student-mat_converted.csv" file that is then used in the models.

def convert_csv(input_file):
    output_file = os.path.splitext(input_file)[0] + "_converted.csv"
    with open(input_file, 'r', newline='') as infile:
        reader = csv.reader(infile, delimiter=';')
        data = list(reader)

    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        for row in data:
            writer.writerow(row)

    print(f"CSV file '{input_file}' has been converted and saved as '{output_file}'.")

convert_csv('student-mat.csv')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('student-mat_converted.csv')

df.shape

df.describe()

df.head()

df.info()

df.isnull().sum()

sns.boxplot(x="G3", y="school", data=df)
plt.xlabel("Final Grade (0-20)")
plt.ylabel("School")
# school names are "GP" Gabriel Pereira or "MS" - Mousinho da Silveira
# this plot show how students grades compare between the two schools that are in the dataset
plt.show()

"""# Step 2: K Nearest Neighbors Regressor"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import timeit
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
seed = 123

# Instantiate LabelEncoder
label_encoder = LabelEncoder()

# List of columns to encode
# This encodes all categorical values
columns_to_encode = ['sex', 'famsize', 'Fjob', 'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'nursery', 'higher', 'internet', 'romantic']

# Loop through each column and encode
for column in columns_to_encode:
    # Encode the categorical column
    df[column] = label_encoder.fit_transform(df[column])

# Define input and output features
# see DESCRIPTION OF DATASET ATTRIBUTES section for detailed descriptions of values
X = df[['sex', 'age', 'famsize', 'Medu', 'Fedu', 'Fjob', 'reason', 'guardian', 'failures', 'schoolsup', 'famsup', 'paid', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc',  'health', 'absences']]
y = df[['G3']] # G3 (the target) is the students final grade

# Create training and testing data
X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.2, random_state=seed
)

# Scale the input features

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
result = []

import timeit
from sklearn import neighbors

knn = neighbors.KNeighborsRegressor(n_neighbors = 5)
tic=timeit.default_timer() # start timer
knn.fit(X_train_scaled, np.ravel(y_train))
toc=timeit.default_timer() - tic # end timer
print("The train time in seconds: ", toc)

pred=knn.predict(X_test_scaled) #X_test
result.append(["KNNs", round(mean_absolute_error(y_test,pred), 2), round(np.sqrt(mean_squared_error(y_test, pred)), 2), round(r2_score(y_test, pred), 2)])
print(result[-1])

# Residual analysis
residuals = np.ravel(y_test) - pred
plt.figure(figsize=(10, 6))
sns.scatterplot(x=np.ravel(y_test), y=residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs. Actual Values')
plt.xlabel('Actual Values')
plt.ylabel('Residuals')
plt.show()

from sklearn.model_selection import learning_curve, validation_curve

# Learning curve
train_sizes, train_scores, test_scores = learning_curve(knn, X_train_scaled, np.ravel(y_train), cv=5)
avg_train_scores = np.mean(train_scores, axis=1)
avg_test_scores = np.mean(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, avg_train_scores, label='Training Score')
plt.plot(train_sizes, avg_test_scores, label='Cross-Validation Score')
plt.title('Learning Curve')
plt.xlabel('Training Examples')
plt.ylabel('Score')
plt.legend()
plt.show()

# Validation curve
param_range = np.arange(1, 20)
train_scores, test_scores = validation_curve(knn, X_train_scaled, np.ravel(y_train), param_name="n_neighbors", param_range=param_range, cv=5)
avg_train_scores = np.mean(train_scores, axis=1)
avg_test_scores = np.mean(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(param_range, avg_train_scores, label='Training Score')
plt.plot(param_range, avg_test_scores, label='Cross-Validation Score')
plt.title('Validation Curve')
plt.xlabel('Number of Neighbors')
plt.ylabel('Score')
plt.legend()
plt.show()

"""# Step 2: SGD Regressor"""

from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# Define input and output features
# see DESCRIPTION OF DATASET ATTRIBUTES section for detailed descriptions of values
X = df[['sex', 'age', 'famsize', 'Medu', 'Fedu', 'Fjob', 'reason', 'guardian', 'failures', 'schoolsup', 'famsup', 'paid', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc',  'health', 'absences']]
y = df['G3']  # Assuming 'G3' is the target variable

# Create training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# Scale the input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the SGDRegressor
sgd_regressor = SGDRegressor()

# Fit the model
tic=timeit.default_timer() # start timer
sgd_regressor.fit(X_train_scaled, y_train)
toc=timeit.default_timer() - tic # end timer after training
print("The train time in seconds: ", toc)

# Predictions
pred = sgd_regressor.predict(X_test_scaled)

# Evaluate the model
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)
print("R-squared Score:", r2)

import matplotlib.pyplot as plt

# Fit the model first
sgd_regressor.fit(X_train_scaled, y_train)

# Visualize feature coefficients
coefficients = sgd_regressor.coef_
plt.figure(figsize=(10, 6))
plt.bar(range(len(coefficients)), coefficients)
plt.xticks(range(len(coefficients)), X.columns, rotation=45)  # Use column names of X as feature names
plt.xlabel('Feature')
plt.ylabel('Coefficient')
plt.title('Feature Coefficients')
plt.show()

residuals = y_test - pred
plt.figure(figsize=(10, 6))
plt.scatter(y_test, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Actual Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(sgd_regressor, X_train_scaled, y_train, cv=5)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, color='blue', marker='o', label='Training Score')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', label='Validation Score')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.xlabel('Number of Training Samples')
plt.ylabel('Score')
plt.title('Learning Curve')
plt.legend(loc='lower right')
plt.show()

"""# Step 3: Decision Tree Regressor"""

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Define input features (X) and target variable (y)
# see DESCRIPTION OF DATASET ATTRIBUTES section for detailed descriptions of values
X = df[['sex', 'age', 'famsize', 'Medu', 'Fedu', 'Fjob', 'reason', 'guardian', 'failures', 'schoolsup', 'famsup', 'paid', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc',  'health', 'absences']]
y = df['G3']  # Assuming 'G3' is the final grade column

# Partition the dataset into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# Construct a decision tree regressor from the training data
regressor = DecisionTreeRegressor(min_samples_split=100)

# fit and time model
tic=timeit.default_timer() # start timer before training
model = regressor.fit(X_train, y_train)
toc=timeit.default_timer() - tic # end timer after training
print("The train time in seconds: ", toc)

# Calculate predictions for the testing set
pred = regressor.predict(X_test)

# Calculate evaluation metrics
rmse = mean_squared_error(y_test, pred, squared=False)
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2) Score:", r2)

# Feature importances
feature_importances = model.feature_importances_

# Plot feature importances
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importances)), feature_importances)
plt.xticks(range(len(feature_importances)), X.columns, rotation=45)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importances')
plt.show()

residuals = y_test - pred
plt.figure(figsize=(10, 6))
plt.scatter(y_test, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Actual Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

from sklearn import tree
plt.figure(figsize=(20, 10))
tree.plot_tree(model, feature_names=X.columns, filled=True)
plt.title('Decision Tree Visualization')
plt.show()

"""# Step 4: XGB Regressor"""

import xgboost as xgb

# Define input features (X) and target variable (y)
# see DESCRIPTION OF DATASET ATTRIBUTES section for detailed descriptions of values
X = df[['sex', 'age', 'famsize', 'Medu', 'Fedu', 'Fjob', 'reason', 'guardian', 'failures', 'schoolsup', 'famsup', 'paid', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc',  'health', 'absences']]
y = df['G3']  # Assuming 'G3' is the final grade column

# Partition the dataset into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

tic=timeit.default_timer() # start timer before training
dtrain = xgb.DMatrix(X_train, label=y_train)
toc=timeit.default_timer() - tic # end timer after training
print("The train time in seconds: ", toc)

param = {'max_depth': 3, 'eta': 0.1, 'objective': 'reg:squarederror'}

num_round = 100
bst = xgb.train(param, dtrain, num_round)

dtest = xgb.DMatrix(X_test)
predictions = bst.predict(dtest)

rmse = mean_squared_error(y_test, predictions, squared=False)
print("Root Mean Squared Error (RMSE):", rmse)

mae = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error (MAE):", mae)

r2 = r2_score(y_test, predictions)
print("R-squared (R2) Score:", r2)

xgb.plot_importance(bst)

# Calculate residuals
residuals = y_test - predictions

# Plot residuals
plt.scatter(predictions, residuals)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.axhline(y=0, color='r', linestyle='-')  # Add horizontal line at y=0
plt.title('Residual Plot')
plt.show()

plt.scatter(y_test, predictions)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # Add diagonal line for reference
plt.show()

"""# Step 5: Adaboost Regressor"""

from sklearn.ensemble import AdaBoostRegressor

# Define input features (X) and target variable (y)
# see DESCRIPTION OF DATASET ATTRIBUTES section for detailed descriptions of values
X = df[['sex', 'age', 'famsize', 'Medu', 'Fedu', 'Fjob', 'reason', 'guardian', 'failures', 'schoolsup', 'famsup', 'paid', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc',  'health', 'absences']]
y = df['G3']  # Assuming 'G3' is the final grade column

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# Define the base estimator
base_estimator = DecisionTreeRegressor(max_depth=3)

# Instantiate the AdaBoostRegressor with the base estimator
ada_model = AdaBoostRegressor(base_estimator=base_estimator, n_estimators=50, learning_rate=1.0, loss='linear', random_state=None)

# Fit the model
tic=timeit.default_timer() # start timer before training
ada_model.fit(X_train, y_train)
toc=timeit.default_timer() - tic # end timer after training
print("The train time in seconds: ", toc)

# Make predictions
predictions = ada_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2) Score:", r2)

# Get feature importances
feature_importances = ada_model.feature_importances_

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importances)), feature_importances, tick_label=X.columns)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importance Plot')
plt.xticks(rotation=45)
plt.show()

# Get training and validation errors for each boosting round
train_errors = []
test_errors = []
for pred_train, pred_test in zip(ada_model.staged_predict(X_train), ada_model.staged_predict(X_test)):
    train_errors.append(mean_squared_error(y_train, pred_train))
    test_errors.append(mean_squared_error(y_test, pred_test))

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_errors, label='Train MSE')
plt.plot(test_errors, label='Test MSE')
plt.xlabel('Number of Boosting Rounds')
plt.ylabel('Mean Squared Error')
plt.title('Learning Curve')
plt.legend()
plt.show()

# Calculate residuals
residuals = y_test - predictions

# Plot residual plot
plt.figure(figsize=(10, 6))
plt.scatter(predictions, residuals)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()

"""# Step 5 (con't) Bagging Regressor"""

from sklearn.ensemble import BaggingRegressor
from sklearn.datasets import make_regression

# Define input features (X) and target variable (y)
# see DESCRIPTION OF DATASET ATTRIBUTES section for detailed descriptions of values
X = df[['sex', 'age', 'famsize', 'Medu', 'Fedu', 'Fjob', 'reason', 'guardian', 'failures', 'schoolsup', 'famsup', 'paid', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc',  'health', 'absences']]
y = df['G3']  # Assuming 'G3' is the final grade column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train BaggingRegressor
bagging_model = BaggingRegressor(n_estimators=10, random_state=42)
tic=timeit.default_timer() # start timer before training
bagging_model.fit(X_train, y_train)
toc=timeit.default_timer() - tic # end timer after training
print("The train time in seconds: ", toc)

# Predict on the test set
y_pred = bagging_model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Calculate R^2 score
print("R^2 Score:", r2)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Calculate mean absolute error
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)

# Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='-')
plt.title('Residual Plot')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()

"""# Testing with other data

Just to make sure I wasn't doing something wrong that was causing me to get low r-squared values, I decided to do a test with one of the models I used. The test (below) shows the Bagging regressor model used in the exact same way I used it with my data, but training with synthetic data instead.

As shown in the output, I got an r-squared value of 0.897. Then I noticed I was using 10000 samples when my dataset only had 400. To Show the affect the amount of data can have on the training on a model, I ran the same test with only 100 samples. This drop the r-squared value to just 0.60.

This could hint that if I had a massive amount of the same quality of data that I did, maybe my models would've performed significantly better.
"""

# Generate synthetic data
X, y = make_regression(n_samples=10000, n_features=10, noise=0.1, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train BaggingRegressor
bagging_model = BaggingRegressor(n_estimators=10, random_state=42)
tic=timeit.default_timer() # start timer before training
bagging_model.fit(X_train, y_train)
toc=timeit.default_timer() - tic # end timer after training

# Predict on the test set
y_pred = bagging_model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Test 1 results (10000 samples)")
print("\nR^2 Score:", r2)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)
print("The train time in seconds: ", toc)


# RUN THE TEST AGAIN BUT WITH LESS SAMPLES

# Generate synthetic data
X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train BaggingRegressor
bagging_model = BaggingRegressor(n_estimators=10, random_state=42)
tic=timeit.default_timer() # start timer before training
bagging_model.fit(X_train, y_train)
toc=timeit.default_timer() - tic # end timer after training

# Predict on the test set
y_pred = bagging_model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nTest 2 results (100 samples)")
print("\nR^2 Score:", r2)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)
print("The train time in seconds: ", toc)

"""# Conclusion

Overall, I tried virtually every combination of input values to make these models peform their best. I studied the results of many approaches of input to get the highest accuracy and r squared values possible. Yet, most of the values were not very high. This shows that in this dataset, with this quality of data, either there was not enough of the data to get a high accuracy, or the data just might not be correlated enough to the target.

- My dataset had no missing values. This was checked with df.info() and df.isnull().sum() in Step 1.
- I couldn't implement the models due to about half of the values being numeric. These values were converted to equivalent numeric values using label encoding in Step 2.
- My dataset did not have any columns that were read as object type that should be datatime object.
- Some columns were not included as input values when training the models. These columns were decided to be excluded during the process of testing each column and determining whether it had a positive impact on the prediction accuracy of the models.
- I did not identify any duplicates in my dataset.
- The K Nearest Neighbors regressor and the XGB regressor both had an r quared value of 0.36. While this is not very high, these were the models that performed the best.
"""