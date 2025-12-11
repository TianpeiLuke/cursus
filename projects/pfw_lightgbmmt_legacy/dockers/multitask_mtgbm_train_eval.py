# load libraries
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import time

import csv
from pathlib import Path
import os
from random import shuffle
import pickle
from tqdm import tqdm
import json

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing

from config.utils import *
from utils.processing import *
import mtgbm.lightgbmmt as lgb

parser = argparse.ArgumentParser()
parser.add_argument("--exp_id", type=str, help="unique experiment id")
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument(
    "--config_data",
    type=str,
    default="./config/config_data.yaml",
    help="path to config file for data",
)
parser.add_argument(
    "--config_mtgbm",
    type=str,
    default="./config/config_mtgbm.yaml",
    help="path to config file for mmoe model",
)
args = parser.parse_args()
if __name__ == "__main__":
    # load config
    config_mtgbm = load_config(args.config_mtgbm)
    config_data = load_config(args.config_data)
    num_label = config_data["num_tasks"]
    config_mtgbm["num_labels"] = num_label

    # define save path
    save_path = f"/home/ec2-user/SageMaker/models/trial_{args.exp_id}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(os.path.join(save_path, "log_train.txt"), "a") as f:
        pprint("Model parameters", f)
        pprint(config_mtgbm, f)
        pprint("------------------", f)
        pprint("Data parameters", f)
        pprint(config_data, f)
        pprint("------------------", f)
    # load data
    dataloader_train, dataloader_val, dataloader_oot = load_data(
        batch_size=args.batch_size,
        eval_flag=False,
        save_path=save_path,
        model_type="tree",
        **config_data,
    )

    # create df for saving output
    df = dataloader_oot.dataset.y
    df.columns = [f"{tag}_true" for tag in config_data["target_tags"]]
    if config_data["id_col"] is not None:
        df["id"] = dataloader_oot.dataset.order_id
    if config_data["value_col"] is not None:
        df["value"] = dataloader_oot.dataset.value
    if config_data["aux_tags"] is not None:
        for i, aux_tag in enumerate(config_data["aux_tags"]):
            df[f"{aux_tag}_true"] = dataloader_oot.dataset.y_aux.iloc[:, i]
    if config_data["abusive_tags"] is not None:
        for i, abusive_tag in enumerate(config_data["abusive_tags"]):
            df[f"{abusive_tag}_true"] = dataloader_oot.dataset.y_abusive.iloc[:, i]

    # extract data matrices
    X_train = dataloader_train.dataset.x.astype("float32").values
    y_train = dataloader_train.dataset.y.astype("float32").values
    X_oot = dataloader_oot.dataset.x.astype("float32").values

    # define functions for training
    def self_metric(preds, train_data):
        labels = train_data.get_label()
        labels2 = labels.reshape((num_label, -1)).transpose()[:, 0]
        preds2 = preds.reshape((num_label, -1)).transpose()[:, 0]
        preds2 = 1.0 / (1.0 + np.exp(-preds2))
        score = roc_auc_score(labels2, preds2)
        return "self_metric", score, False

    def mymse3(preds, train_data, ep):
        labels = train_data.get_label()  # a large concat vector, length = # of rows in train_data * number of total labels
        labels2 = labels.reshape((num_label, -1)).transpose()
        preds2 = preds.reshape((num_label, -1)).transpose()

        preds3 = 1.0 / (1.0 + np.exp(-preds2))
        grad2 = preds3 - labels2  # gradients: G_i
        hess2 = preds3 * (1.0 - preds3)  # hessians: H_i

        w = np.ones((num_label))  # can adjust weights here
        grad = np.sum(grad2 * w, axis=1)  # G_e in Algorithm 2
        hess = np.sum(hess2 * w, axis=1)  # H_e in Algorithm 2

        w2 = np.ones((num_label))
        grad2 = grad2 * w2  # G_u in Algorithm 3
        hess2[:, -1] = hess2[:, -1] * 0.1 + 1  # H_u in Algorithm 3
        return grad, hess, grad2, hess2

    # train model
    dataset_train = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(
        config_mtgbm,
        train_set=dataset_train,
        num_boost_round=500,
        verbose_eval=70,
        fobj=mymse3,
        feval=self_metric,
    )
    model.set_num_labels(num_label)

    # prediction
    yhat_oot = model.predict(X_oot)
    for i, tag in enumerate(config_data["target_tags"]):
        df[f"{tag}_pred"] = yhat_oot[:, i]
    df["isdnr_pred"] = 1.0 / (
        1.0 + np.exp(-df["isdnr_pred"].values)
    )  # convert ot prob. from log-prob
    df["isflr_pred"] = 1.0 / (1.0 + np.exp(-df["isflr_pred"].values))
    df["isrr_pred"] = 1.0 / (1.0 + np.exp(-df["isrr_pred"].values))

    # save
    df.to_feather(f"{save_path}/predictions.f", compression="zstd")
