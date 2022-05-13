#!/usr/bin/env python3
"""Train classification models on the given pull requests data."""

# !!! DISCLAIMER !!!
#
# I (Ondřej Kuhejda) did not create this script.
# This script was created by Lenarduzzi et al.
# and is distributed with the CC BY 4.0 license.
#
# https://creativecommons.org/licenses/by/4.0/
#
# The script was slightly modified to better suit
# my needs.
#
# The original version is available here:
# https://figshare.com/s/d47b6f238b5c92430dd7?file=14949029

import click
import json
import pickle
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy import interp
from multiprocessing import Manager
import multiprocessing
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas
import os
import matplotlib
matplotlib.use("PDF")


def cross_validate_and_plot(clf, X, y, column_names, name, number_of_cores):
    """Train the model using the given regression algorithm.

    It uses given dataset to train and evaluate the model.
    """
    num_folds = 5

    cv = StratifiedKFold(n_splits=num_folds)
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []

    N, P = X.shape

    # Aggregate the importances over folds here:
    importances_random = np.zeros(P)
    importances_drop = np.zeros(P)

    # Loop over crossvalidation folds:

    scores = []  # Collect accuracies here

    tnList = []
    fpList = []
    fnList = []
    tpList = []
    precisionList = []
    recallList = []
    f1List = []
    mccList = []

    i = 1
    count = 0
    for train, test in cv.split(X, y):
        count += 1

 #       print("Fitting model on fold %d/%d..." % (i, num_folds))

        X_train = X[train, :]
        y_train = y[train]
        X_test = X[test, :]
        y_test = y[test]

        # Train the model
        clf.fit(X_train, y_train)

        # Predict for validation data:
        probas_ = clf.predict_proba(X_test)
        y_pred = clf.predict(X_test)

        # Compute ROC curve and area under the curve
        fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])

        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        # calculate confusion matrix, precision, f1 and Matthews Correlation Coefficient

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        tnList.append(tn / (tn + fp))
        tpList.append(tp / (fn + tp))
        fpList.append(fp / (tn + fp))
        fnList.append(fn / (fn + tp))

        precisionList.append(precision)
        recallList.append(recall)
        f1List.append(f1)
        mccList.append(mcc)
#        print(classification_report(y_test, y_pred))

        # Finally: measure feature importances for each column at a time

        baseline = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
        scores.append(baseline)

        # Importance calculated random shuffling column's elements
        for col in range(P):

            #            print("Assessing feature %d/%d..." % (col+1, P), end = "\r")

            # Store column for restoring it later
            save = X_test[:, col].copy()

            # Randomly shuffle all values in this column
            X_test[:, col] = np.random.permutation(X_test[:, col])

            # Compute AUC score after distortion
            m = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

            # Restore the original data
            X_test[:, col] = save

            # Importance is incremented by the drop in accuracy:
            importances_random[col] += (baseline - m)

        # save feature importance values in .csv
        idx = np.argsort(importances_random)
        sorted_column_names = list(np.array(column_names)[idx])

        importance_random = pandas.DataFrame(
            {'Variables': sorted_column_names, 'Importance': importances_random[idx]})
        target_file = "Importances_random_folds/%s-%s" % (name, count)
        make_sure_folder_exists(target_file)
        importance_random.to_csv(
            target_file + ".csv", columns=['Variables', 'Importance'], sep=',', index=False, float_format='%.4f')

        # Importance calculated dropping columns

        # create list shared by processes
        manager = Manager()
        n = manager.list()
#        n=[]

        # define drop-column function
        def drop_column(col):
            # Drop a column at the time and fit the model
            X_drop = np.delete(X, np.s_[col], axis=1)
            X_train_drop = X_drop[train, :]
            X_test_drop = X_drop[test, :]
            clf.fit(X_train_drop, y_train)

            # Compute AUC score after distortion
            m = roc_auc_score(y_test, clf.predict_proba(X_test_drop)[:, 1])

            #            # Restore the original data
            #            X_test[:, col] = save

            n.append(m)

        # run drop-column function in parallel
        Parallel(n_jobs=number_of_cores, verbose=10)(
            delayed(drop_column)(col) for col in range(P))

        # Importance is incremented by the drop in accuracy:
        for col in range(P):
            importances_drop[col] += (baseline - n[col])

        # save feature importance values in .csv
        idx = np.argsort(importances_drop)
        sorted_column_names = list(np.array(column_names)[idx])

        importance_drop = pandas.DataFrame(
            {'Variables': sorted_column_names, 'Importance': importances_drop[idx]})
        target_file = "Importances_drop_folds/%s-%s" % (name, count)
        make_sure_folder_exists(target_file)
        importance_drop.to_csv(
            target_file + ".csv", columns=['Variables', 'Importance'], sep=';', index=False, float_format='%.4f')

        i += 1
        print("\n")

    # Average the metrics over folds

    print("confusion matrix " + str(name))
    tnList = 100 * np.array(tnList)
    tpList = 100 * np.array(tpList)
    fnList = 100 * np.array(fnList)
    fpList = 100 * np.array(fpList)
    precisionList = 100 * np.array(precisionList)
    recallList = 100 * np.array(recallList)
    f1List = 100 * np.array(f1List)
    mccList = 100 * np.array(mccList)

    # show metrics

    # print("TN: %.02f %% ± %.02f %% - FN: %.02f %% ± %.02f %%" % (np.mean(tnList),
    #                                                              np.std(
    #                                                                  tnList),
    #                                                              np.mean(
    #                                                                  fnList),
    #                                                              np.std(fnList)))
    # print("FP: %.02f %% ± %.02f %% - TP: %.02f %% ± %.02f %%" % (np.mean(fpList),
    #                                                              np.std(
    #                                                                  fpList),
    #                                                              np.mean(
    #                                                                  tpList),
    #                                                              np.std(tpList)))
    # print("Precision: %.02f %% ± %.02f %%  Recall: %.02f %% ± %.02f %%" % (np.mean(precisionList),
    #                                                                        np.std(
    #                                                                            precisionList),
    #                                                                        np.mean(
    #                                                                            recallList),
    #                                                                        np.std(recallList)))
    # print("Precision: %.02f %% ± %.02f %% - F1: %.02f %% ± %.02f %% - MCC: %.02f %% ± %.02f %%" % (np.mean(precisionList),
    #                                                                                                np.std(
    #                                                                                                    precisionList),
    #                                                                                                np.mean(
    #                                                                                                    f1List),
    #                                                                                                np.std(
    #                                                                                                    f1List),
    #                                                                                                np.mean(
    #                                                                                                    mccList),
    #                                                                                                np.std(mccList)))
    # save metrics as .csv
    # Average the TPR over folds

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    metrics_tosave = pandas.DataFrame(
        {'CM_TNR_mean': np.mean(tnList), 'CM_TNR_std': np.std(tnList),
         'CM_FNR_mean': np.mean(fnList), 'CM_FNR_std': np.std(fnList),
         'CM_FPR_mean': np.mean(fpList), 'CM_FPR_std': np.std(fpList),
         'CM_TPR_mean': np.mean(tpList), 'CM_TPR_std': np.std(tpList),
         'Precision_mean': np.mean(precisionList), 'Precision_std': np.std(precisionList),
         'Recall_mean': np.mean(recallList), 'Recall_std': np.std(recallList),
         'AUC_mean': 100 * mean_auc, 'AUC_std': std_auc,
         'F1_mean': np.mean(f1List), 'F1_std': np.std(f1List),
         'MCC_mean': np.mean(mccList), 'MCC_std': np.std(mccList)}, index=[0])

    target_file = "Metrics/%s" % name
    make_sure_folder_exists(target_file)
    metrics_tosave.to_csv(target_file + ".csv", columns=['CM_TNR_mean', 'CM_TNR_std',
                                                         'CM_FNR_mean', 'CM_FNR_std',
                                                         'CM_FPR_mean', 'CM_FPR_std',
                                                         'CM_TPR_mean', 'CM_TPR_std',
                                                         'Precision_mean', 'Precision_std',
                                                         'Recall_mean', 'Recall_std',
                                                         'AUC_mean', 'AUC_std',
                                                         'F1_mean', 'F1_std',
                                                         'MCC_mean', 'MCC_std'], sep=',', index=False)

    # Plot AUC curve
    plt.figure(1)
    plt.clf()

    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.4f $\pm$ %0.4f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")

    target_file = "AUCs/plots/%s" % name
    make_sure_folder_exists(target_file)
    plt.savefig(target_file + ".pdf", bbox_inches="tight")

    # Save AUC curve data
    auc_curves_data = pandas.DataFrame({"mean_fpr": mean_fpr,
                                        "mean_tpr": mean_tpr,
                                        "tprs_lower": tprs_lower,
                                        "tprs_upper": tprs_upper})
    target_file = "AUCs/values/%s" % name
    make_sure_folder_exists(target_file)
    auc_curves_data.to_csv(target_file + ".csv",
                           columns=["mean_fpr", "mean_tpr",
                                    "tprs_lower", "tprs_upper"],
                           sep=',', index=False)

    # Plot importances:
    plt.figure(2)
    plt.clf()

    # Divide importances by num folds to get the average

    importances_average_random = importances_random / num_folds

    idx = np.argsort(importances_average_random)
    sorted_column_names = list(np.array(column_names)[idx])

    importance_average_random = pandas.DataFrame(
        {'Variables': sorted_column_names, 'Importance': importances_average_random[idx]})
    target_file = "Importances_random/values/%s" % name
    make_sure_folder_exists(target_file)
    importance_average_random.to_csv(
        target_file + ".csv", columns=['Variables', 'Importance'], sep=',', index=False, float_format='%.4f')

    fontsize = 2 if P > 100 else 8

    plt.barh(np.arange(P), 100 *
             importances_average_random[idx], align='center')
    plt.yticks(np.arange(P), sorted_column_names, fontsize=fontsize)
    plt.xlabel("Feature importance (drop in score [%])")
    plt.title("Feature importances (baseline AUC = %.4f %%)" %
              (100 * np.mean(scores)))

    plt.ylabel("<-- Less important     More important -->")

    target_file = "Importances_random/plots/%s" % name
    make_sure_folder_exists(target_file)
    plt.savefig(target_file + ".pdf", bbox_inches="tight")

    # Plot importance with dropping:
    # Plot importances:
    plt.figure(2)
    plt.clf()

    # Divide importances by num folds to get the average

    importances_average_drop = importances_drop / num_folds

    idx = np.argsort(importances_average_drop)
    sorted_column_names = list(np.array(column_names)[idx])

    importance_average_drop = pandas.DataFrame(
        {'Variables': sorted_column_names, 'Importance': importances_average_drop[idx]})
    target_file = "Importances_drop/values/%s" % name
    make_sure_folder_exists(target_file)
    importance_average_drop.to_csv(
        target_file + ".csv", columns=['Variables', 'Importance'], sep=',', index=False, float_format='%.4f')

    fontsize = 2 if P > 100 else 8

    plt.barh(np.arange(P), 100 * importances_average_drop[idx], align='center')
    plt.yticks(np.arange(P), sorted_column_names, fontsize=fontsize)
    plt.xlabel("Feature importance (drop in score [%])")
    plt.title("Feature importances - Drop-column (baseline AUC = %.4f %%)" %
              (100 * np.mean(scores)))

    plt.ylabel("<-- Less important     More important -->")

    target_file = "Importances_drop/plots/%s" % name
    make_sure_folder_exists(target_file)
    plt.savefig(target_file + ".pdf", bbox_inches="tight")

    return mean_fpr, mean_tpr


def make_sure_folder_exists(path):
    """Create folder if it not exists."""
    folder = os.path.dirname(path)
    if not os.path.isdir(folder):
        os.makedirs(folder)


@click.command()
@click.option('--only-introduced-issues', "-i", is_flag=True)
@click.option('--only-fixed-issues', "-f", is_flag=True)
@click.option('--number-of-cores', "-c", type=int, default=3)
@click.argument("csv_file_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_folder_path", type=click.Path(exists=True,
                                                      file_okay=False))
def cli(csv_file_path: str, output_folder_path: str,
        only_introduced_issues: bool, only_fixed_issues: bool,
        number_of_cores: int):
    """Train classification models on the given pull requests data."""
    if only_introduced_issues and only_fixed_issues:
        raise click.UsageError("Flags '--only_introduced_issues' and "
                               "'--only_fixed_issues' are mutually exclusive.")

    start_time = time.time()

    # Read in data
    df = pandas.read_csv(csv_file_path)
    df = df.drop(columns=[column for column in list(df) if not
                          (column == "accepted" or
                           column.startswith("results_"))])

    # Set the output directory as working directory.
    os.chdir(output_folder_path)

    # df = create_groups_cv(df, splits)

    # remove infinite values and NaN values
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    y = df["accepted"]
    cols = [c for c in df.columns if "results_" in c]
    # cols.append("Groups")
    if len(cols) == 0:
        quit()
    X = df[cols]

    rulecount = open("rulecount.txt", "w+")
    rulecount.write("Rule count: %d\r\n" % (len(cols)))
    rulecount.close()

    X = np.array(X)

    if only_introduced_issues:
        X = X.clip(min=0)
    if only_fixed_issues:
        X = X.clip(max=0)

    y = np.array(y)

    # Define classifiers to try: (clf, name) pairs
    classifiers = [
        (LogisticRegression(C=1, penalty="l1", solver="liblinear"),
         "LogisticRegression"),
        (RandomForestClassifier(n_estimators=100, n_jobs=number_of_cores,
                                random_state=0), "RandomForest"),
        (GradientBoostingClassifier(n_estimators=number_of_cores,
                                    random_state=0), "GradientBoost"),
        (ExtraTreesClassifier(n_estimators=100,
                              random_state=0), "ExtraTrees"),
        (DecisionTreeClassifier(random_state=0), "DecisionTrees"),
        (BaggingClassifier(n_estimators=100, n_jobs=number_of_cores,
                           random_state=0), "Bagging"),
        (AdaBoostClassifier(n_estimators=100, random_state=0), "AdaBoost"),
        (XGBClassifier(n_estimators=100, n_jobs=number_of_cores,
                       randomstate=0), "XGBoost")
    ]

    # Loop over each and crossvalidate
    # RULEID Prediction

    # Store all ROC curves here:

    for clf, name in classifiers:
        print("Evaluating %s classifier (ruleid)" % name)
        fpr, tpr = cross_validate_and_plot(clf, X, y, cols, name + "_ruleid",
                                           number_of_cores)

    end_time = time.time()
    execution_time = end_time - start_time
    print("Total execution time: %s seconds" % execution_time)


if __name__ == "__main__":
    cli()
