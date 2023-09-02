import pickle

import pandas as pd
from scipy.stats import randint as sp_randint, uniform
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error

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
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, train_size=0.8, test_size=0.2, random_state=0)

RFR = RandomForestRegressor(max_depth=25, min_samples_leaf=2, min_samples_split=13,
                            n_estimators=513, random_state=42)
RFR.fit(X_train, Y_train)
Y_pred = RFR.predict(X_test)
print(type(X_test))
print(X_test)
print(type(Y_pred))
print(Y_pred)

mae = mean_absolute_error(Y_test, Y_pred)
print(mean_absolute_percentage_error(Y_test, Y_pred))

with open('random_forest_model.pickle', 'wb') as f:
    pickle.dump(RFR, f)