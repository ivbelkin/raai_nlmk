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

        self.calk_on_days = 90
        self.pred_on_days = 10

        self.train_lags = list(range(0, 91, 10))
        self.train_step = 10

        self.marki = None
        self.general_preparations()

    def get_train_valid_data(self):
        train_df = self.zavalki_df.query("дата_завалки < '{}'".format(self.train_valid_split)).drop("дата_завалки", axis=1)
        valid_df = self.zavalki_df.query("дата_завалки >= '{}'".format(self.train_valid_split)).drop("дата_завалки", axis=1)
        return self.prepare_X_y(train_df, valid_df)

    def get_train_test_data(self):
        train_df = self.zavalki_df.query("дата_завалки >= '{}'".format(self.train_test_split)).drop("дата_завалки", axis=1)
        test_df = self.test_df.drop(["id", "дата_завалки"], axis=1)
        return self.prepare_X_y(train_df, test_df)

    def prepare_X_y(self, train_df, valid_df):
        train_X = train_df.drop(["износ"], axis=1)
        train_y = train_df["износ"]

        if "износ" in valid_df.columns:
            valid_X = valid_df.drop(["износ"], axis=1)
            valid_y = valid_df["износ"]
        else:
            valid_X = valid_df.copy()
            valid_y = None

        train_X = self.basic_features(train_X)
        valid_X = self.basic_features(valid_X)

        # train_X_list, train_y_list = [], []
        # total = len([0 for _ in self.train_splits(train_X["день"])])
        # for src, dst, lag in tqdm(self.train_splits(train_X["день"]), total=total):
        #     train_src_X, train_src_y = train_X[src].copy(), train_y[src].copy()
        #     train_dst_X = train_X[dst].copy()
        #
        #     train_dst_X["лаг"] = lag
        #     train_dst_X = self.time_features(train_src_X, train_src_y, train_dst_X)
        #
        #     train_X_list.append(train_dst_X)
        #
        #     train_dst_y = train_y.iloc[dst]
        #     train_y_list.append(train_dst_y)
        #
        # valid_X_list, valid_y_list = [], []
        # total = len([0 for _ in self.train_test_splits(train_X["день"], valid_X["день"])])
        # for src, dst, lag in tqdm(self.train_test_splits(train_X["день"], valid_X["день"]), total=total):
        #     train_src_X, train_src_y = train_X[src].copy(), train_y[src].copy()
        #     valid_dst_X = valid_X[dst].copy()
        #
        #     valid_dst_X["лаг"] = lag
        #     valid_dst_X = self.time_features(train_src_X, train_src_y, valid_dst_X)
        #
        #     valid_X_list.append(valid_dst_X)
        #
        #     if valid_y is not None:
        #         valid_dst_y = valid_y.iloc[dst]
        #         valid_y_list.append(valid_dst_y)
        #
        # train_X = pd.concat(train_X_list, ignore_index=True)
        # train_y = pd.concat(train_y_list, ignore_index=True)
        #
        # valid_X = pd.concat(valid_X_list, ignore_index=True)
        # if valid_y is not None:
        #     valid_y = pd.concat(valid_y_list, ignore_index=True)

        cols_to_drop = ["номер_завалки", "день"]
        train_X = train_X.drop(cols_to_drop, axis=1)
        valid_X = valid_X.drop(cols_to_drop, axis=1)

        train_X, train_y = self.filter_outliers(train_X, train_y)

        if valid_y is not None:
            return train_X, train_y, valid_X, valid_y
        else:
            return train_X, train_y, valid_X

    def train_test_splits(self, train_on, test_on):
        src_end = np.max(train_on) + 1
        src = (train_on >= src_end - self.calk_on_days).values

        test_min = np.min(test_on)
        test_max = np.max(test_on)
        i = 0
        while test_min + i * self.pred_on_days <= test_max:
            dst_start = test_min + i * self.pred_on_days
            dst_end = dst_start + self.pred_on_days
            dst = (test_on >= dst_start).values & (test_on < dst_end).values
            lag = dst_start - src_end
            i += 1
            if src.any() and dst.any():
                yield src, dst, lag

    def train_splits(self, train_on):
        train_max = np.max(train_on)
        train_min = np.min(train_on)

        i = 0
        while train_min + i * self.train_step + self.calk_on_days + self.train_lags[0] <= train_max:
            src_start = train_min + i * self.train_step
            src_end = src_start + self.calk_on_days
            src = (train_on >= src_start).values & (train_on < src_end).values
            i += 1

            for lag in self.train_lags:
                dst_start = src_end + lag
                dst_end = dst_start + self.pred_on_days
                dst = (train_on >= dst_start).values & (train_on < dst_end).values

                if src.any() and dst.any():
                    yield src, dst, lag

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

    def one_hot_categorical(self, train_X, valid_X):
        return pd.get_dummies(train_X, columns=self.categorical), pd.get_dummies(valid_X, columns=self.categorical)

    def mean_target(self, src_X, src_y, dst_X):
        for alpha, fname in zip(self.alphas, self.categorical):
            N = len(src_y)
            global_mean = np.mean(src_y)

            mapping = src_y.groupby(src_X[fname]).apply(np.mean)
            counts = src_y.groupby(src_X[fname]).apply(len)

            n = dst_X[fname].map(counts)
            mean = dst_X[fname].map(mapping)

            dst_X[fname] = (mean * n + alpha * global_mean * N) / (n + alpha * N)
            dst_X[fname] = dst_X[fname].map(lambda x: np.round(x, decimals=2))

        return dst_X

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
        self.zavalki_df["продолжительность_завалки_мин"] = (t2 - t1).map(lambda x: x.seconds // 60)
        self.zavalki_df["день"] = t1.map(lambda x: x.dayofyear)
        self.zavalki_df["день_недели"] = t1.map(lambda x: x.dayofweek)
        self.zavalki_df["час"] = t1.map(lambda x: x.hour)
        self.zavalki_df = self.zavalki_df.drop(["дата_вывалки"], axis=1)

        t1 = self.test_df["дата_завалки"].map(pd.to_datetime)
        t2 = self.test_df["дата_вывалки"].map(pd.to_datetime)
        self.test_df["продолжительность_завалки_мин"] = (t2 - t1).map(lambda x: x.seconds // 60)
        self.test_df["день"] = t1.map(lambda x: x.dayofyear)
        self.test_df["день_недели"] = t1.map(lambda x: x.dayofweek)
        self.test_df["час"] = t1.map(lambda x: x.hour)
        self.test_df = self.test_df.drop(["дата_вывалки"], axis=1)

    def general_preparations(self):
        self.add_zavalka_number()
        self.handle_datetime()

        self.zavalki_df["положение_в_клети"] = self.zavalki_df["положение_в_клети"].map(self.cat_to_int["положение_в_клети"])
        self.test_df["положение_в_клети"] = self.test_df["положение_в_клети"].map(self.cat_to_int["положение_в_клети"])

        self.zavalki_df = self.zavalki_df.merge(self.valki_df, on="номер_валка", how="left")
        self.test_df = self.test_df.merge(self.valki_df, on="номер_валка", how="left")

        self.zavalki_df["материал_валка"] = self.zavalki_df["материал_валка"].map(lambda x: int(x.split(" ")[-1]))
        self.test_df["материал_валка"] = self.test_df["материал_валка"].map(lambda x: int(x.split(" ")[-1]))

        self.ruloni_df["Марка"] = self.ruloni_df["Марка"].map(lambda x: int(x.split(" ")[-1]))
        self.marki = list(set(self.ruloni_df["Марка"]))

    def time_features(self, src_X, src_y, dst_X):
        # dst_X = self.mean_target(src_X, src_y, dst_X)
        return dst_X

    def basic_features(self, X):
        agg_fs = {
            "сум": np.sum, "мин": np.min, "макс": np.max, "мед": np.median, "ср": np.mean
        }
        for fname in ["Масса", "Толщина", "Ширина"]:
            for prefix, f in agg_fs.items():
                mapping = self.ruloni_df.groupby("номер_завалки")[fname].apply(f)
                X[prefix + "_" + fname] = X["номер_завалки"].map(mapping)

                mapping = self.ruloni_df.groupby(["номер_завалки", "Марка"])[fname].apply(f).to_dict()
                for marka in self.marki:
                    X[prefix + "_" + fname + "_марки_" + str(marka)] = X["номер_завалки"].map(
                        lambda x: mapping.get((x, marka), 0)
                    )

        mapping = self.ruloni_df.groupby("номер_завалки")["Масса"].apply(len)
        X["число_рулонов"] = X["номер_завалки"].map(mapping)

        return X

    def filter_outliers(self, X, y):
        q1 = np.percentile(y, 1)
        q99 = np.percentile(y, 99)

        mask = (q1 < y).values & (y < q99).values

        return X[mask], y[mask]
