import numpy as np
import pandas as pd
from scipy.stats import randint as sp_randint, uniform
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_percentage_error, classification_report, accuracy_score, r2_score

dataset = pd.DataFrame(pd.read_csv("House Price India.csv"))
# print(dataset.columns)

dataset.drop(['id', 'Date', 'Renovation Year', 'Lattitude',
              'Longitude', 'living_area_renov', 'lot_area_renov',
              'Number of schools nearby', 'waterfront present', 'number of views'],
             axis=1,
             inplace=True)

X = dataset.drop(['Price'], axis=1)
Y = dataset['Price']

# Split the training set into
# training and validation set
X_train, X_valid, Y_train, Y_valid = train_test_split(
    X, Y, train_size=0.7, test_size=0.3, random_state=0)

model_RFR = RandomForestRegressor(n_estimators=10)
model_RFR.fit(X_train, Y_train)
Y_pred = model_RFR.predict(X_valid)

print(mean_absolute_percentage_error(Y_valid, Y_pred))

# plt.figure(figsize=(12, 6))
# sns.heatmap(dataset.corr(),
#             cmap = 'BrBG',
#             fmt = '.2f',
#             linewidths = 2,
#             annot = True)
# plt.show()

# Define the parameter distributions
param_dist = {
    'n_estimators': sp_randint(100, 1000),

    'max_depth': [None] + list(range(5, 30, 5)),
    'min_samples_split': sp_randint(2, 20),
    'min_samples_leaf': sp_randint(1, 20),
    'bootstrap': [True, False]
}

# Create a Random Forest Regressor model
model = RandomForestRegressor(random_state=42)

# Perform random search with cross-validation
random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10, cv=5, random_state=42)

# Fit the random search to the training data
random_search.fit(X_train, Y_train)

# Get the best hyperparameters and the best model
best_params = random_search.best_params_
best_model = random_search.best_estimator_

print(best_params)
print(best_model)
# Evaluate the best model on the test data
y_pred = best_model.predict(X_valid)
