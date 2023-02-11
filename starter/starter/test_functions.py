from ml.data import separate_cat_int_var
from sklearn.model_selection import train_test_split
from ml.data import process_data
import pandas as pd
import pytest

logging.basicConfig(
    filename='./logs/model.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

@pytest.fixture
def data():
    """ Simple function to read the data."""
    df = pd.read_csv('starter\data\census.csv')
    return df

def test_data_shape(data):
    """ If the data has no null values """
    assert data.shape == data.dropna().shape, "Dropping null changes shape."

def test_categorical_features(data):
    cat_features = [
        'workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country'
        ]

    categorical, _= separate_cat_int_var(data)

    try:
        assert categorical == cat_features
        logging.info("Categorical features are correct: SUCCESS")

    except AssertionError as err:
        logging.error("The categorical features doesn't match: ERROR")
        raise err



def test_process_data():
	'''
	test the process data function
	'''
	train, test = train_test_split(data, test_size=0.20)

    X_train, y_train, encoder, lb = process(
    train, categorical_features=cat_features,integer_features=integer_features, label="salary", training=True, encoder=None, lb=None)
    X_test, y_test, encoder, lb = process(
    test, categorical_features=cat_features,integer_features=integer_features, label="salary", training=False, encoder=encoder, lb=lb)
	
    try:
		assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        logging.info("Testing process data function: SUCCESS")
	except AssertionError as err:
		logging.error("Testing function process data: the sizes are not similar ")
        raise err

    return X_train, X_test, y_train, y_test
