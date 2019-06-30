import numpy as np
import pandas as pd
from tqdm import tqdm


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
        self.categorical = ["номер_клетки", "номер_валка", "положение_в_клети", "материал_валка"]
        self.alphas = [0, 0, 0, 0]

        self.add_zavalka_number()
        self.handle_datetime()

    def get_train_valid_data(self):
        train_df = self.zavalki_df.query("дата_завалки < '{}'".format(self.train_valid_split)).drop("дата_завалки", axis=1)
        valid_df = self.zavalki_df.query("дата_завалки >= '{}'".format(self.train_valid_split)).drop("дата_завалки", axis=1)
        return self.prepare_X_y(train_df, valid_df)

    def get_train_test_data(self):
        train_df = self.zavalki_df.query("дата_завалки >= '{}'".format(self.train_test_split)).drop("дата_завалки", axis=1)
        test_df = self.test_df.drop("дата_завалки", axis=1)
        return self.prepare_X_y(train_df, test_df)

    def prepare_X_y(self, train_df, valid_df):
        train_df["положение_в_клети"] = train_df["положение_в_клети"].map(self.cat_to_int["положение_в_клети"])
        valid_df["положение_в_клети"] = valid_df["положение_в_клети"].map(self.cat_to_int["положение_в_клети"])

        train_X = train_df.drop(["износ"], axis=1)
        train_y = train_df["износ"]

        if "износ" in valid_df.columns:
            valid_X = valid_df.drop(["износ"], axis=1)
            valid_y = valid_df["износ"]
        else:
            valid_X = valid_df.copy()
            valid_y = None

        if "id" in valid_df.columns:
            valid_X = valid_X.drop("id", axis=1)

        train_X = train_X.merge(self.valki_df, on="номер_валка", how="left")
        valid_X = valid_X.merge(self.valki_df, on="номер_валка", how="left")

        mapping = self.ruloni_df.groupby("номер_завалки")["Масса"].apply(sum)
        train_X["суммарная_масса"] = train_X["номер_завалки"].map(mapping)
        valid_X["суммарная_масса"] = valid_X["номер_завалки"].map(mapping)

        # mapping = self.ruloni_df.groupby("номер_завалки")["Масса"].apply(max)
        # train_X["максимальная_масса"] = train_X["номер_завалки"].map(mapping)
        # valid_X["максимальная_масса"] = valid_X["номер_завалки"].map(mapping)

        # mapping = self.ruloni_df.groupby("номер_завалки")["Масса"].apply(min)
        # train_X["минимальная_масса"] = train_X["номер_завалки"].map(mapping)
        # valid_X["минимальная_масса"] = valid_X["номер_завалки"].map(mapping)

        # mapping = self.ruloni_df.groupby("номер_завалки")["Масса"].apply(np.median)
        # train_X["медианная_масса"] = train_X["номер_завалки"].map(mapping)
        # valid_X["медианная_масса"] = valid_X["номер_завалки"].map(mapping)

        train_X, valid_X = self.mean_target(train_X, train_y, valid_X)

        train_X = train_X.drop("номер_завалки", axis=1)
        valid_X = valid_X.drop("номер_завалки", axis=1)

        if valid_y is not None:
            return train_X, train_y, valid_X, valid_y
        else:
            return train_X, train_y, valid_X

    def save_submission(self, path, iznos):
        df = pd.DataFrame({"id": self.test_df["id"], "iznos": iznos})
        df.to_csv(path, index=False)

    def save_submission_pair(self, path, iznos):
        self.save_submission(path.replace(".csv", "_p.csv"), iznos + 1.0)
        self.save_submission(path.replace(".csv", "_n.csv"), iznos - 1.0)

    @staticmethod
    def calc_score(p, n):
        e = np.sqrt((p ** 2 + n ** 2) / 2 - 1)
        np_err = 5e-6
        e_err = np_err * np.sqrt(p ** 2 + n ** 2) / (2 * e)
        return e - e_err, e + e_err

    def one_hot_categorical(self, df):
        return pd.get_dummies(df, columns=self.categorical)

    def mean_target(self, train_X, train_y, valid_X):
        global_mean = np.mean(train_y)
        N = len(train_y)
        for alpha, fname in zip(self.alphas, self.categorical):
            mapping = train_y.groupby(train_X[fname]).apply(np.mean)
            counts = train_y.groupby(train_X[fname]).apply(len)

            n = train_X[fname].map(counts)
            train_X[fname] = (train_X[fname].map(mapping) * n + alpha * global_mean * N) / (n + alpha * N)

            n = valid_X[fname].map(counts)
            valid_X[fname] = (valid_X[fname].map(mapping) * n + alpha * global_mean * N) / (n + alpha * N)
        return train_X, valid_X

    def add_zavalka_number(self):
        train_zavalki_dates = sorted(set(self.zavalki_df["дата_завалки"]))
        test_zavalki_dates = sorted(set(self.test_df["дата_завалки"]))
        zavalki_dates = train_zavalki_dates + test_zavalki_dates
        mapping = {z: i for i, z in enumerate(zavalki_dates)}

        self.zavalki_df["номер_завалки"] = self.zavalki_df["дата_завалки"].map(mapping)
        self.test_df["номер_завалки"] = self.test_df["дата_завалки"].map(mapping)

        zn = 0
        self.ruloni_df["номер_завалки"] = -1
        N = len(zavalki_dates)
        zavalki_number = []
        for i in tqdm(range(len(self.ruloni_df))):
            if zn < N - 1 and self.ruloni_df["Время_обработки"].iloc[i] >= zavalki_dates[zn + 1]:
                zn += 1
            zavalki_number.append(zn)
        self.ruloni_df["номер_завалки"] = zavalki_number

    def handle_datetime(self):
        t1 = self.zavalki_df["дата_завалки"].map(pd.to_datetime)
        t2 = self.zavalki_df["дата_вывалки"].map(pd.to_datetime)
        self.zavalki_df["продолжительность_завалки"] = list(map(lambda x: x.seconds, t2 - t1))
        self.zavalki_df = self.zavalki_df.drop(["дата_вывалки"], axis=1)

        t1 = self.test_df["дата_завалки"].map(pd.to_datetime)
        t2 = self.test_df["дата_вывалки"].map(pd.to_datetime)
        self.test_df["продолжительность_завалки"] = list(map(lambda x: x.seconds, t2 - t1))
        self.test_df = self.test_df.drop(["дата_вывалки"], axis=1)
