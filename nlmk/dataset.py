import numpy as np
import pandas as pd


class Dataset:

    def __init__(self, ruloni_path, valki_path, zavalki_path, test_path):
        self.ruloni_path = ruloni_path
        self.valki_path = valki_path
        self.zavalki_path = zavalki_path
        self.test_path = test_path

        self.ruloni_df = pd.read_csv(ruloni_path)
        self.valki_df = pd.read_csv(valki_path)
        self.zavalki_df = pd.read_csv(zavalki_path)
        self.test_df = pd.read_csv(test_path)

        self.train_valid_split = "2018-07-01 00:00:00"
        self.train_test_split = "2018-04-01 00:00:00"

        self.cat_to_int = {
            "положение_в_клети": {"низ": 0, "верх": 1}
        }
        self.categorical = ["номер_клетки", "номер_валка", "положение_в_клети"]

    def get_train_valid_data(self):
        train_df = self.zavalki_df.query("дата_завалки < '{}'".format(self.train_valid_split))
        valid_df = self.zavalki_df.query("дата_завалки >= '{}'".format(self.train_valid_split))
        return self.prepare_X_y(train_df, valid_df)

    def get_train_test_data(self):
        train_df = self.zavalki_df.query("дата_завалки >= '{}'".format(self.train_test_split))
        return self.prepare_X_y(train_df, self.test_df)

    def prepare_X_y(self, train_df, valid_df):
        train_df["положение_в_клети"] = train_df["положение_в_клети"].map(self.cat_to_int["положение_в_клети"])
        valid_df["положение_в_клети"] = valid_df["положение_в_клети"].map(self.cat_to_int["положение_в_клети"])

        train_X = train_df.drop(["дата_завалки", "дата_вывалки", "износ"], axis=1)
        train_y = train_df["износ"]

        if "износ" in valid_df.columns:
            valid_X = valid_df.drop(["дата_завалки", "дата_вывалки", "износ"], axis=1)
            valid_y = valid_df["износ"]
        else:
            valid_X = valid_df.drop(["дата_завалки", "дата_вывалки"], axis=1)
            valid_y = None

        if "id" in valid_df.columns:
            valid_X = valid_X.drop("id", axis=1)

        train_X, valid_X = self.mean_target(train_X, train_y, valid_X)

        if valid_y is not None:
            return train_X, train_y, valid_X, valid_y
        else:
            return train_X, train_y, valid_X

    def save_submission(self, path, iznos):
        df = pd.DataFrame({"id": self.test_df["id"], "iznos": iznos})
        df.to_csv(path, index=False)

    def one_hot_categorical(self, df):
        return pd.get_dummies(df, columns=self.categorical)

    def mean_target(self, train_X, train_y, valid_X):
        for fname in self.categorical:
            mapping = train_y.groupby(train_X[fname]).apply(np.mean)
            train_X[fname] = train_X[fname].map(mapping)
            valid_X[fname] = valid_X[fname].map(mapping)
        return train_X, valid_X
