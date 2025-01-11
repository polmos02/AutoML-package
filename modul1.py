import pandas as pd
import numpy as np
from skopt.space import Real, Integer
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier,  GradientBoostingClassifier

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.feature_selection import SelectFromModel
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score
from scipy.stats import uniform, randint
import random


def numerical_preprocess(data):
    num_columns = data.select_dtypes(include=['float64', 'int64']).columns

    if len(num_columns) == 0:
        return data

    imputer = SimpleImputer(strategy='mean')
    data[num_columns] = imputer.fit_transform(data[num_columns])

    scaler = MinMaxScaler(feature_range=(0, 1))
    data[num_columns] = scaler.fit_transform(data[num_columns])

    return data

def categorical_preprocess(data):
    cat_columns = data.select_dtypes(include=['object']).columns

    if len(cat_columns) == 0:
        return data

    imputer = SimpleImputer(strategy='most_frequent')
    data[cat_columns] = imputer.fit_transform(data[cat_columns])


    return data


def feature_select(data, target):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(data, target)
    selector = SelectFromModel(model, threshold="mean")  # Można także ustawić próg np. 0.01, lub "mean"
    data_selected = selector.transform(data)
    selected_features = data.columns[selector.get_support()]
    data_selected = pd.DataFrame(data_selected, columns=selected_features)
    return data_selected, selected_features

def prep(data, target=None, mode='train', features = None):
    data = numerical_preprocess(data)
    data = categorical_preprocess(data)
    data = pd.get_dummies(data, drop_first=True, dtype=int)
    if mode == 'train':
        data, selected_features = feature_select(data, target)
        return data, selected_features
    if mode == 'test':
        data = data[features]
        return data