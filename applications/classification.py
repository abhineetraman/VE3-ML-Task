import pandas as pd
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.model_selection import RandomizedSearchCV

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

housing_data, housing_target = fetch_california_housing(as_frame=True, return_X_y=True)

housing_data

# creating pipeline

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehotencoder', OneHotEncoder())
])

num_attribs = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude'] # Include Latitude and Longitude in numerical
cat_attribs = []


full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', cat_pipeline, cat_attribs)
])

#creating pipeline for data splitting, model with pipeline usage, then use metrices for validation set and then use it in test set

X_train, X_test, y_train, y_test = train_test_split(housing_data, housing_target, test_size=0.1, random_state=42)

model = Pipeline([
    ('preprocessing', full_pipeline),
    ('model', RandomForestRegressor())
])

#randomized search CV for hyper-parameter Tuning
param_grid = {
    'model__n_estimators': [100, 200, 50],
    'model__max_depth': [None, 10, 20, 30],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [2, 5, 10],
}

random_search = RandomizedSearchCV(
    model,
    param_distributions=param_grid,
    n_iter=10,
    scoring='neg_mean_squared_error',
    cv=5)

random_search.fit(X_train, y_train)

print(f"Best parameters: {random_search.best_params_}")
print(f"Best score: {random_search.best_score_}")

model.set_params(**random_search.best_params_)
model.fit(X_train, y_train)

print("Training set score: {:.2f}".format(model.score(X_train, y_train)))
print("Test set score: {:.2f}".format(model.score(X_test, y_test)))

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print("testing_score")
print(f"R2 score: {r2}")
print(f"MSE: {mse}")

import joblib

joblib.dump(model, 'model.pkl')