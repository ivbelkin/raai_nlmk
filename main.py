from nlmk.dataset import Dataset
from nlmk.feature_selection import non_constant_features, variative_features, add_del, add_noisy_features

import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from collections import Counter
from tqdm import tqdm
import os

np.warnings.filterwarnings("ignore")
os.environ["PYTHONUNBUFFERED"] = "1"

RULONI_PATH = "data/Ruloni.csv"
VALKI_PATH = "data/Valki.csv"
ZAVALKI_PATH = "data/Zavalki.csv"
TEST_PATH = "data/Test.csv"

N_WORKERS = 4

params = dict(
    max_depth=3,
    learning_rate=0.01,
    n_estimators=2000,
    verbosity=0,
    objective="reg:squarederror",
    nthread=2,
    tree_method="gpu_exact"
)


def main():
    ds = Dataset(RULONI_PATH, VALKI_PATH, ZAVALKI_PATH, TEST_PATH)

    train_X, train_y, valid_X, valid_y = ds.get_train_valid_data()
    train_all_X, train_all_y, test_X = ds.get_train_test_data()

    features = variative_features(train_all_X)

    train_X = add_noisy_features(train_X[features])
    valid_X = add_noisy_features(valid_X[features])

    features = set()

    features = add_del(
        params,
        train_X, train_y,
        valid_X, valid_y,
        features_init=set(features),
        n_workers=N_WORKERS
    )

    print(features)


if __name__ == "__main__":
    main()
