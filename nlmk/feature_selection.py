import numpy as np
import os

from tqdm import tqdm_notebook as tqdm
from collections import Counter
from multiprocessing import Pool
from sys import stderr, stdout
from copy import deepcopy
from xgboost import XGBRegressor


def add_del(params, train_X, train_y, valid_X, valid_y, features_init=None, n_workers=1):
    features_all = set(train_X.columns)
    features = features_init or set()
    feature_sets = []

    last_train_score = np.inf
    last_valid_score = np.inf

    while True:
        print("ADD")
        feature, train_score, valid_score = add_one(params, train_X, train_y, valid_X, valid_y, features, n_workers)
        if valid_score < last_valid_score and not feature.startswith("rnd_"):
            features = features.union({feature})
            feature_sets.append(features)
            last_valid_score = valid_score
            print("\t", feature, "TRAIN SCORE", train_score, "VALID SCORE", valid_score)
        else:
            break

    while True:
        print("DEL")
        feature, train_score, valid_score = del_one(params, train_X, train_y, valid_X, valid_y, features, n_workers)
        if valid_score < last_valid_score and not feature.startswith("rnd_"):
            features = features - {feature}
            feature_sets.append(features)
            last_valid_score = valid_score
            print("\t", feature, "TRAIN SCORE", train_score, "VALID SCORE", valid_score)
        else:
            break

    return features


def evaluate(params, train_X, train_y, valid_X, valid_y):
    model = XGBRegressor(**params)
    model.fit(
        train_X, train_y,
        eval_set=[(train_X, train_y), (valid_X, valid_y)],
        verbose=False
    )
    idx = np.argmin(model.evals_result()["validation_1"]["rmse"])
    train_score = model.evals_result()["validation_0"]["rmse"][idx]
    valid_score = model.evals_result()["validation_1"]["rmse"][idx]
    return train_score, valid_score


def add_one(params, train_X, train_y, valid_X, valid_y, features, n_workers=1):
    features_all = set(train_X.columns)
    features_unused = features_all - features

    with Pool(n_workers) as p:
        seq = [(f, params, train_X, train_y, valid_X, valid_y, features, n_workers) for f in features_unused]
        scores = dict(p.map(add, seq))

    best_feature, (train_score, valid_score) = min(scores.items(), key=lambda x: x[1][1])
    return best_feature, train_score, valid_score


def add(args):
    feature, params, train_X, train_y, valid_X, valid_y, features, n_workers = args
    params["gpu_id"] = os.getpid() % n_workers
    features_new = list(features.union({feature}))
    print("pid", os.getpid(), "gpu", params["gpu_id"], feature); stdout.flush()
    return feature, evaluate(
        params,
        train_X[features_new], train_y,
        valid_X[features_new], valid_y
    )


def del_one(params, train_X, train_y, valid_X, valid_y, features, n_workers=1):
    with Pool(n_workers) as p:
        seq = [(f, params, train_X, train_y, valid_X, valid_y, features, n_workers) for f in features]
        scores = dict(p.map(del_, seq))

    best_feature, (train_score, valid_score) = min(scores.items(), key=lambda x: x[1][1])
    return best_feature, train_score, valid_score


def del_(args):
    feature, params, train_X, train_y, valid_X, valid_y, features, n_workers = args
    params["gpu_id"] = os.getpid() % n_workers
    features_new = list(features - {feature})
    print("pid", os.getpid(), "gpu", params["gpu_id"], feature); stdout.flush()
    return feature, evaluate(
        params,
        train_X[features_new], train_y,
        valid_X[features_new], valid_y
    )

def add_noisy_features(X, factor=1, features=None):
    features = features or list(X.columns)
    idx = np.arange(len(X))
    for _ in range(factor):
        for feature in features:
            np.random.shuffle(idx)
            X["rnd_" + feature] = X[feature].values[idx]
    return X


def non_constant_features(train_X, features=None):
    features = features or list(train_X.columns)
    train_nu = train_X[features].nunique()
    return list(train_nu[train_nu > 1].index)


def variative_features(train_X, features=None, q=0.99):
    features = features or list(train_X.columns)
    N = len(train_X)
    features_new = []
    for feature in features:
        cnt = Counter(train_X[feature])
        if cnt.most_common()[0][1] / N < q:
            features_new.append(feature)
    return features_new
