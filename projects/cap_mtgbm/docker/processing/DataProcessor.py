import pandas as pd
import numpy as np
import glob
import os
import time
import pickle

from src.utils.util import *


class DataProcessor:
    """
    Clean and process the data

    Parameter
    ---------
    train_path: The directory to read the train data from

    test_path: The directory to read the test data from
    """

    def __init__(
        self, train_path="/data/final_samp", test_path="/data/test_reduce_with_tag"
    ):
        self.train_path = train_path
        self.test_path = test_path

    def data_processing(self):
        """
        Clean and separate train and test data
        """
        raw_train_data = pd.read_pickle(self.train_path)
        raw_train_label = raw_train_data[["isFraud"]]
        clean_train_data, clean_train_label = self.clean_sample(
            raw_train_data, raw_train_label
        )

        raw_test_data = pd.read_pickle(self.test_path)
        raw_test_label = raw_test_data[["isFraud"]]
        clean_test_data, clean_test_label = self.clean_sample(
            raw_test_data, raw_test_label
        )

        return clean_train_data, clean_train_label, clean_test_data, clean_test_label

    def data_loader(self, src_path=None, out_path=None):
        """
        load train/test data and tag given corresponding path

        Parameters:
            src_path: path where downloaded data are stored

            out_path: path where concat data will be saved
        """
        data_paths = [os.path.join(src_path, f) for f in os.listdir(src_path)]
        data_paths = [i for i in data_paths if os.path.isfile(i)]
        l = []
        data_paths.sort()
        for filename in data_paths:
            df = pd.read_csv(filename, delimiter=",", header=None)
            l.append(df)
        df = pd.concat(l, axis=0, ignore_index=True)
        df.to_pickle(out_path)
        return df

    def reduce_mem_usage(self, df=None):
        """
        iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
        """
        start_mem = df.memory_usage().sum() / 1024**2
        print("Memory usage of dataframe is {:.2f} MB".format(start_mem))
        for col in df.columns:
            col_type = df[col].dtype
            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == "int":
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif (
                        c_min > np.iinfo(np.int16).min
                        and c_max < np.iinfo(np.int16).max
                    ):
                        df[col] = df[col].astype(np.int16)
                    elif (
                        c_min > np.iinfo(np.int32).min
                        and c_max < np.iinfo(np.int32).max
                    ):
                        df[col] = df[col].astype(np.int32)
                    elif (
                        c_min > np.iinfo(np.int64).min
                        and c_max < np.iinfo(np.int64).max
                    ):
                        df[col] = df[col].astype(np.int64)
                else:
                    if (
                        c_min > np.finfo(np.float16).min
                        and c_max < np.finfo(np.float16).max
                    ):
                        df[col] = df[col].astype(np.float16)
                    elif (
                        c_min > np.finfo(np.float32).min
                        and c_max < np.finfo(np.float32).max
                    ):
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
            else:
                df[col] = df[col].astype("category")

        end_mem = df.memory_usage().sum() / 1024**2
        print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
        print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

        return df

    def create_paymeth_label(self, df):
        """
        create subtasks fraud label using major payment types
        classify payment types with multiple classes including GC as GC only
        """
        if (df.paymeth == "CC") and (df.isFraud == 0):
            df.isCCfrd = 0
        if (df.paymeth == "DD") and (df.isFraud == 0):
            df.isDDfrd = 0
        if (df.paymeth == "GC") and (df.isFraud == 0):
            df.isGCfrd = 0
        if (df.paymeth == "LineOfCredit") and (df.isFraud == 0):
            df.isLOCfrd = 0
        if (df.paymeth == "Cimarron") and (df.isFraud == 0):
            df.isCimfrd = 0
        if (
            (df.paymeth == "CC,GC")
            or (df.paymeth == "DD,GC")
            or (df.paymeth == "GC,LineOfCredit")
        ):
            df.paymeth = "GC"
            if df.isFraud == 0:
                df.isGCfrd = 0
        return df

    def clean_sample(self, data, label):
        """
        A function for cleaning data based on major payment types.

        Parameters
        ----------
            data: training or test dataset

            label: training or test fraud label
        """
        if "isFraud" in data.columns:
            df_X = self.reduce_mem_usage(data.drop(["paymeth", "isFraud"], axis=1))
        else:
            df_X = self.reduce_mem_usage(data.drop(["paymeth"], axis=1))
        #         label.columns = ['isFraud']
        #         df_Y = label

        paymeth = data["paymeth"]
        cond1 = (
            (paymeth == "CC")
            | (paymeth == "DD")
            | (paymeth == "LineOfCredit")
            | (paymeth == "GC")
            | (paymeth == "Cimarron")
            | (paymeth == "CC,GC")
            | (paymeth == "DD,GC")
            | (paymeth == "GC,LineOfCredit")
        )
        ptr = paymeth[cond1].dropna()
        clean_data = pd.concat([ptr, label, df_X], axis=1, join="inner").reset_index(
            drop=True
        )
        print(clean_data.shape)

        labels = clean_data[["paymeth", "isFraud"]]
        a = np.empty(labels.shape[0])
        a.fill(1)  # 1 -> good orders
        subtasks = ["isCCfrd", "isDDfrd", "isGCfrd", "isLOCfrd", "isCimfrd"]
        for j in range(len(subtasks)):
            labels[subtasks[j]] = a.astype(int)

        start_time = time.time()
        print(start_time)
        clean_labels = labels.apply(lambda x: self.create_paymeth_label(x), axis=1)
        print("--- %.2f mins ---" % ((time.time() - start_time) / 60))
        print(clean_labels.shape)
        return clean_data, clean_labels
