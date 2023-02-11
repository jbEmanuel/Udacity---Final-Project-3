import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelBinarizer
from feature_engine.encoding import OneHotEncoder


def eliminate_space_and_dash(df, x, y):
  """ This function has the purpose of eliminating space and undesirable signs in the columns names 
 
 Inputs 
 -----------
 df: pd.DataFrame 
 dataframe with the columns
 x: [str] 
 character or spaces to be eliminated, e.g, ("-")
 y: [str] 
 new charater or absence of space that will replace x 
 
 Returns
 ----------
 df.columns: dataframe columns transformed

 """  

  columns = [col.replace(x, y) for col in list(df.columns)]
  df.columns = columns
  return df.columns 
 
def separate_cat_int_var(df):
    """ This function receives a dataframe and returns the list of all categoricall and integert features separated 
 
 Inputs 
 -----------
 df: pd.DataFrame 

 Returns
 ----------
 categorical_features: list
 list containing all categorical features
 integer_features: list
  list containing all integer features

    """
    categorical_features = []
    integer_features = []
    for var in list(df.columns):
        if df[var].dtype == "int64":
            integer_features.append(var)
        else:
            categorical_features.append(var)
  
    return categorical_features.remove('salary'), integer_features

def replace_spaces_in_categ_column(df, categorical_features):
  """ 
  This function receives a dataframe and returns the list of all categoricall and integert features separated 
 
 Inputs 
 -----------
 df: pd.DataFrame 
 categorical_features: list 
 list of categorical features
 
 Returns
 ----------
 df: pd.DataFrame 
 
  """
  for var in categorical_features:
    df[var] = df[var].str.replace(" ", "")
    df[var] = df[var].str.replace("-", "_")
  return df 


def process_data(
    X, categorical_features=[],integer_features=[], label=None, training=True, encoder=None, lb=None
):
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    integer_features: list[str]
        List containing the names of the integer features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : pd.Dataframe
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : feature_engine.encoding.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])
    
    X_categorical = X[categorical_features]
    X_continuous = X[integer_features]

    if training is True:
        encoder = OneHotEncoder(
                    top_categories=None,
                    variables=categorical_features,drop_last=True)  # to return k-1, false to return k
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass
    
    X = pd.concat([X_categorical.reset_index(drop=True), X_continuous.reset_index(drop=True)], axis=1, ignore_index=True)
    X.columns = list(X_categorical.columns) + list(X_continuous.columns)

    
    return X, y, encoder, lb
