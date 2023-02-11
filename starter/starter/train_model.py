# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from ml.data import process_data, eliminate_space_and_dash, separate_cat_int_var
from ml.model import train_model, inference, compute_model_metrics, save_model
import pandas as pd
import numpy as np

# Add code to load in the data.
data = pd.read_csv("starter\data\census.csv")

# Eliminate spaces and dashes in column names 
data.columns = eliminate_space_and_dash(data, " ", "")
data.columns = eliminate_space_and_dash(data, "-", "_")

# Get categorical and integer features 
categorical_features, integer_features = separate_cat_int_var(data)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)


X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features,integer_features=integer_features, label="salary", training=True, encoder=None, lb=None)


# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process(
    test, categorical_features=cat_features,integer_features=integer_features, label="salary", training=False, encoder=encoder, lb=lb)

#Eliminate spaces and dash in new columns names
X_train.columns = eliminate_space_and_dash(X_train, " ", "")
X_train.columns = eliminate_space_and_dash(X_train, "-", "_")
X_test.columns = eliminate_space_and_dash(X_test, " ", "")
X_test.columns = eliminate_space_and_dash(X_test, "-", "_")

# Train and save a model.
model = train_model(X_train, y_train)
save_model(model)

#Make some predictions 
preds = inferences(model, X_test)

# Get Metrics 
precision, recall, fbeta = compute_model_metrics(y_test, preds)

