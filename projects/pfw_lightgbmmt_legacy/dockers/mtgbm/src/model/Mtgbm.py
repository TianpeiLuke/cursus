import pandas as pd
import numpy as np
import glob
import os
import time

import matplotlib.pyplot as plt
import math
import lightgbm as lgb
import lightgbmmt as lgbm
import random
import seaborn as sns
import pickle

from math import log2, log10
from sklearn import preprocessing
from sklearn.metrics import (
    log_loss,
    roc_auc_score,
    accuracy_score,
    f1_score,
    roc_curve,
    precision_score,
    recall_score,
)
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    auc,
)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, TimeSeriesSplit, train_test_split
from scipy.special import expit, rel_entr

from src.lossFunction.customLossKDswap import custom_loss_KDswap
from src.lossFunction.baseLoss import base_loss
from src.lossFunction.customLossNoKD import custom_loss_noKD

seed = 10
np.random.seed(seed)
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
from collections import Counter
from src.utils.util import *


class MtGbm:
    """
    MTGBM with our proposed weighting method and KD loss implementation

    Parameters
    ----------
    loss_type: choose different loss types.
               Please choose "auto_weight" for our proposed weight,
               or "auto_weight_KD" with KD loss included

    config: a configuration file containing self-defined parameters

    X_train: training dataset

    train_labels: training label matrix with all tasks

    X_test: test dataset

    test_labels: test label matrix with all tasks

    sub_tasks_list: subtasks name list

    y_train, y_train_s: main task, sub tasks in training set

    y_test, y_test_s: main task, sub tasks in test set

    y_pred: a dict for storing predicted scores under each lgb model

    idx_test_dic: a dict of indices under each task for test set

    model: the MTGBM training model

    df_pred: predicted value in a dataframe

    y_lgbmt: predicted main task

    y_lgbmtsub: predicted subtasks
    """

    def __init__(
        self,
        config,
        X_train,
        train_labels,
        X_test,
        test_labels,
        main_task,
        sub_tasks_list,
        loss_type=None,
    ):
        self.params = from_json(config)
        self.loss_type = loss_type
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.X_train = X_train
        self.X_test = X_test
        self.main_task = main_task
        self.y_train = train_labels[main_task].copy().reset_index(drop=True)
        self.y_test = test_labels[main_task].copy().reset_index(drop=True)
        self.targets = sub_tasks_list
        self.y_train_s = train_labels[self.targets].copy().reset_index(drop=True)
        self.y_test_s = test_labels[self.targets].copy().reset_index(drop=True)
        #        self.idx_test_dic = filter_idx(self.test_labels)

        self.model = None
        self.df_pred = None
        self.y_lgbmt = np.zeros(self.X_test.shape[0])
        self.y_lgbmtsub = np.zeros((self.X_test.shape[0], len(self.targets)))

    def train(self):
        """
        Model training and validation process
        """
        print("Training set size: ", self.X_train.shape)
        print(
            "Training main task shape: ",
            len(self.y_train),
            " Training sub tasks shape: ",
            self.y_train_s.shape,
        )

        # --- train validation split
        X_tr, X_vl, Y_tr, Y_vl = train_test_split(
            self.X_train, self.y_train, test_size=0.1, random_state=seed
        )
        tr_idx, val_idx = Y_tr.index, Y_vl.index
        # sub tasks split
        arr = np.array(range(len(self.y_train_s.columns)))
        Y_tr2, Y_vl2 = (
            self.y_train_s.iloc[tr_idx, arr],
            self.y_train_s.iloc[val_idx, arr],
        )

        trn_labels = self.train_labels.iloc[tr_idx].reset_index()
        val_labels = self.train_labels.iloc[val_idx].reset_index()
        # Assume all sub-task trainings share the same training set

        idx_trn_dic = {}
        idx_val_dic = {}
        # idx_trn_dic[0] is for main task
        idx_trn_dic[0] = trn_labels.index
        idx_val_dic[0] = val_labels.index
        # The rest is for sub tasks
        for i in range(len(self.targets)):
            idx_trn_dic[i + 1] = trn_labels.index
            idx_val_dic[i + 1] = val_labels.index

        #        idx_trn_dic = {0: trn_labels.index,
        #           1: trn_labels.index,
        #           2: trn_labels.index,
        #           3: trn_labels.index,
        #           4: trn_labels.index,
        #           5: trn_labels.index}
        #        idx_val_dic = {0: val_labels.index,
        #           1: val_labels.index,
        #           2: val_labels.index,
        #           3: val_labels.index,
        #           4: val_labels.index,
        #           5: val_labels.index}

        cate_feature = []
        debug = True
        num_label = 1 + len(self.targets)
        mt_params = {
            "objective": "custom",
            "num_labels": num_label,
            "tree_learner": "serial2",
            "boosting": "gbdt",
            "max_depth": self.params.max_depth,
            "learning_rate": self.params.learning_rate,  # 0.03
            "bagging_fraction": self.params.bagging_fraction,
            "feature_fraction": self.params.feature_fraction,
            "verbosity": self.params.verbosity,
            "lambda_l1": self.params.lambda_l1,
            "lambda_l2": self.params.lambda_l2,
            "num_leaves": self.params.num_leaves,  # 750,
            "min_child_weight": self.params.min_child_weight,
            "min_data_in_leaf": self.params.min_data_in_leaf,  # 100,
            "num_threads": self.params.num_threads,
            "metric_freq": self.params.metric_freq,
            "data_random_seed": self.params.data_random_seed,
        }
        verbose_eval = self.params.verbose_eval
        num_rounds = self.params.num_rounds
        early_stopping_rounds = self.params.early_stopping_rounds

        d_train = lgbm.Dataset(
            X_tr,
            label=np.concatenate([Y_tr.values.reshape((-1, 1)), Y_tr2.values], axis=1),
        )
        d_valid = lgbm.Dataset(
            X_vl,
            label=np.concatenate([Y_vl.values.reshape((-1, 1)), Y_vl2.values], axis=1),
        )

        start_time = time.time()
        #         trn_label_mat = np.array(pd.concat([Y_tr, Y_tr2], axis=1))
        #         val_label_mat = np.array(pd.concat([Y_vl, Y_vl2], axis=1))
        if early_stopping_rounds == -1:
            if self.loss_type == "auto_weight":
                cl = custom_loss_noKD(num_label, idx_val_dic, idx_trn_dic)
                self.model = lgbm.train(
                    mt_params,
                    train_set=d_train,
                    num_boost_round=num_rounds,
                    valid_sets=d_valid,
                    verbose_eval=verbose_eval,
                    fobj=cl.self_obj,
                    feval=cl.self_eval,
                )

            elif self.loss_type == "auto_weight_KD":
                cl = custom_loss_KDswap(num_label, idx_val_dic, idx_trn_dic, 100)
                self.model = lgbm.train(
                    mt_params,
                    train_set=d_train,
                    num_boost_round=num_rounds,
                    valid_sets=d_valid,
                    verbose_eval=verbose_eval,
                    fobj=cl.self_obj,
                    feval=cl.self_eval,
                )

            else:
                # default setting from original MTGBM implementation with fixed weight vector
                cl = base_loss(idx_val_dic)
                self.model = lgbm.train(
                    mt_params,
                    train_set=d_train,
                    num_boost_round=num_rounds,
                    valid_sets=d_valid,
                    verbose_eval=verbose_eval,
                    fobj=cl.base_obj,
                    feval=cl.base_eval,
                )
        else:
            if self.loss_type == "auto_weight":
                cl = custom_loss_noKD(num_label, idx_val_dic, idx_trn_dic)
                self.model = lgbm.train(
                    mt_params,
                    train_set=d_train,
                    num_boost_round=num_rounds,
                    valid_sets=d_valid,
                    early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=verbose_eval,
                    fobj=cl.self_obj,
                    feval=cl.self_eval,
                )

            elif self.loss_type == "auto_weight_KD":
                cl = custom_loss_KDswap(num_label, idx_val_dic, idx_trn_dic, 100)
                self.model = lgbm.train(
                    mt_params,
                    train_set=d_train,
                    num_boost_round=num_rounds,
                    valid_sets=d_valid,
                    early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=verbose_eval,
                    fobj=cl.self_obj,
                    feval=cl.self_eval,
                )

            else:
                # default setting from original MTGBM implementation with fixed weight vector
                cl = base_loss(idx_val_dic)
                self.model = lgbm.train(
                    mt_params,
                    train_set=d_train,
                    num_boost_round=num_rounds,
                    valid_sets=d_valid,
                    verbose_eval=verbose_eval,
                    early_stopping_rounds=early_stopping_rounds,
                    fobj=cl.base_obj,
                    feval=cl.base_eval,
                )
        self.model.set_num_labels(num_label)
        self.model.save_model("model.txt")
        print("--- training time: %.2f mins ---" % ((time.time() - start_time) / 60))

        # --- Check evaluation results from model training
        plt.style.use("ggplot")
        # -- plot evaluation curve
        eval_score = np.array(cl.eval_mat)
        subtask_name = self.targets
        task_name = np.insert(subtask_name, 0, "main")
        for j in range(eval_score.shape[1]):
            plt.plot(eval_score[:, j], label=task_name[j])
        plt.legend(ncol=2)
        plt.ylim(0.6, 1)
        plt.title("Evaluation Results")
        plt.savefig("mtg.png")
        plt.show()
        # -- plot subtask weight changing
        weight = np.array(cl.w_trn_mat)
        for j in range(1, weight.shape[1]):
            plt.plot(weight[:, j], label=task_name[j])
        plt.legend(ncol=2)
        plt.title("Weights Changing Trend")
        plt.savefig("weight_change.png")
        plt.show()

    def predict(self):
        """
        Model prediction
        """
        print("Test set size: ", self.X_test.shape)
        print(
            "Test main task shape: ",
            len(self.y_test),
            " Test sub tasks shape: ",
            self.y_test_s.shape,
        )
        temp = self.model.predict(self.X_test)
        self.y_lgbmt = expit(temp[:, 0])
        self.y_lgbmtsub = expit(temp[:, 1:])

        print(
            "main task test metrics:",
            " AUC ",
            roc_auc_score(self.y_test, self.y_lgbmt),
            " logloss ",
            log_loss(self.y_test, self.y_lgbmt),
            " f1 score ",
            f1_score(self.y_test, self.y_lgbmt.round(0)),
        )

        df_pred = pd.DataFrame()
        df_pred[self.main_task] = self.y_lgbmt
        for i in range(len(self.targets)):
            df_pred[self.targets[i]] = self.y_lgbmtsub[:, i]

        self.df_pred = df_pred

    #        self.df_pred = pd.DataFrame({'Overall': self.y_lgbmt,
    #                                     'is_abusive_dnr': self.y_lgbmtsub[:, 0],
    #                                     'is_abusive_pda': self.y_lgbmtsub[:, 1],
    #                                     'is_abusive_rr': self.y_lgbmtsub[:, 2],
    #                                     'is_abusive_flr': self.y_lgbmtsub[:, 3],
    #                                     'is_abusive_mdr': self.y_lgbmtsub[:, 4]})

    def evaluate(self):
        """
        Evaluate final results
        """
        true_cc, pred_cc = (
            self.y_test_s.iloc[self.idx_test_dic[1]]["is_abusive_dnr"],
            self.df_pred.iloc[self.idx_test_dic[1]]["CC"],
        )
        true_dd, pred_dd = (
            self.y_test_s.iloc[self.idx_test_dic[2]]["is_abusive_pda"],
            self.df_pred.iloc[self.idx_test_dic[2]]["DD"],
        )
        true_gc, pred_gc = (
            self.y_test_s.iloc[self.idx_test_dic[3]]["is_abusive_rr"],
            self.df_pred.iloc[self.idx_test_dic[3]]["GC"],
        )
        true_loc, pred_loc = (
            self.y_test_s.iloc[self.idx_test_dic[4]]["is_abusive_flr"],
            self.df_pred.iloc[self.idx_test_dic[4]]["LineOfCredit"],
        )
        true_cim, pred_cim = (
            self.y_test_s.iloc[self.idx_test_dic[5]]["is_abusive_mdr"],
            self.df_pred.iloc[self.idx_test_dic[5]]["Cimarron"],
        )

        # --- plot feature importance
        train_columns = self.X_train.columns
        feature_importances = (
            self.model.feature_importance() / sum(self.model.feature_importance())
        ) * 100
        results = pd.DataFrame(
            {"Features": train_columns, "Importances": feature_importances}
        )

        sns.set(font_scale=0.75)
        sns.barplot(
            x="Importances",
            y="Features",
            data=results.sort_values(by="Importances", ascending=False)[0:20],
        )
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.show()

        # --- plot ROC curves
        plt.style.use("ggplot")
        fpr, tpr, thres = roc_curve(self.y_test, self.y_lgbmt)
        fpr_m1, tpr_m1, thresholds1 = roc_curve(true_cc, pred_cc)
        fpr_m2, tpr_m2, thresholds2 = roc_curve(true_dd, pred_dd)
        fpr_m3, tpr_m3, thresholds3 = roc_curve(true_gc, pred_gc)
        fpr_m4, tpr_m4, thresholds4 = roc_curve(true_loc, pred_loc)
        fpr_m5, tpr_m5, thresholds5 = roc_curve(true_cim, pred_cim)

        plt.plot([0, 1], [0, 1], "k--")
        plt.plot(
            fpr,
            tpr,
            label="main task, AUC=%0.4f" % roc_auc_score(self.y_test, self.y_lgbmt),
        )
        plt.plot(
            fpr_m1,
            tpr_m1,
            label="CreditCard, AUC = %0.4f" % roc_auc_score(true_cc, pred_cc),
        )
        plt.plot(
            fpr_m2,
            tpr_m2,
            label="DirectDeposit, AUC = %0.4f" % roc_auc_score(true_dd, pred_dd),
        )
        plt.plot(
            fpr_m3,
            tpr_m3,
            label="GiftCard, AUC = %0.4f" % roc_auc_score(true_gc, pred_gc),
        )
        plt.plot(
            fpr_m4,
            tpr_m4,
            label="LineOfCredit, AUC = %0.4f" % roc_auc_score(true_loc, pred_loc),
        )
        plt.plot(
            fpr_m5,
            tpr_m5,
            label="Cimarron, AUC = %0.4f" % roc_auc_score(true_cim, pred_cim),
        )
        plt.legend()
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("MTGBM ROC")
        # plt.savefig('ROC_mt.png')
        plt.show()

        # --- plot PRAUC curves
        pr, rec, pr_thres = precision_recall_curve(self.y_test, self.y_lgbmt)
        p_m1, r_m1, thres11 = precision_recall_curve(true_cc, pred_cc)
        p_m2, r_m2, thres22 = precision_recall_curve(true_dd, pred_dd)
        p_m3, r_m3, thres33 = precision_recall_curve(true_gc, pred_gc)
        p_m4, r_m4, thres44 = precision_recall_curve(true_loc, pred_loc)
        p_m5, r_m5, thres55 = precision_recall_curve(true_cim, pred_cim)

        plt.plot(rec, pr, label="main task, prAUC=%0.4f" % auc(rec, pr))
        plt.plot(r_m1, p_m1, label="CreditCard, AUC = %0.4f" % auc(r_m1, p_m1))
        plt.plot(r_m2, p_m2, label="DirectDeposit, AUC = %0.4f" % auc(r_m2, p_m2))
        plt.plot(r_m3, p_m3, label="GiftCard, AUC = %0.4f" % auc(r_m3, p_m3))
        plt.plot(r_m4, p_m4, label="LineOfCredit, AUC = %0.4f" % auc(r_m4, p_m4))
        plt.plot(r_m5, p_m5, label="Cimarron, AUC = %0.4f" % auc(r_m5, p_m5))
        plt.legend()
        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.title("MTGBM PRcurves")
        # plt.savefig('PR_mt.png')
        plt.show()
