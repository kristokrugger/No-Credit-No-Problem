import pathlib

import logistic_regression_model

import pandas as pd

RANDOM_STATE = 0

pd.options.display.max_rows = 10
pd.options.display.max_columns = 10

PACKAGE_ROOT = pathlib.Path(logistic_regression_model.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
DATASET_DIR = PACKAGE_ROOT / "datasets"

# data
TRAINING_DATA_FILE = "train.csv"
TESTING_DATA_FILE = "test.csv"
TARGET = "income"

# variables
FEATURES = [
     'age',
     'capital_gain',
     'capital_loss',
     'hours_per_week',
     'workclass',
     'marital_status',
     'occupation',
     'relationship',
     'race',
     'gender',
     'native_country',
     'education'
]

# variables with NA in train set
VARS_WITH_NA = [
    'workclass', 
    'occupation', 
    'native_country'
]

# variables with infrequent values in train set
VARS_WITH_RARE = [
     'workclass',
     'education',
     'marital_status',
     'occupation',
     'race',
     'native_country'
]



# numerical variables
NUMERICAL_VARS = [
    'age', 
    'capital_gain', 
    'capital_loss', 
    'hours_per_week'
]

# ordinal variables to encode with target-guided encoding
ORDINAL_VARS = ['education']

# nominal variables to encode with one-hot encoding
NOMINAL_VARS = [
     'workclass',
     'marital_status',
     'occupation',
     'relationship',
     'race',
     'gender',
     'native_country'
]


VARS_NA_NOT_ALLOWED = [
    feature
    for feature in FEATURES
    if feature not in VARS_WITH_NA
]


PIPELINE_NAME = "log_regression"
PIPELINE_SAVE_FILE = f"{PIPELINE_NAME}_output_v"
