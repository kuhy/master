#!/usr/bin/env python3

import click
import datetime
import numpy as np
import os
import pandas
import pickle

from sklearn.ensemble import (AdaBoostRegressor,
                              BaggingRegressor,
                              GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import (ElasticNet, LinearRegression)
from sklearn.metrics import (explained_variance_score,
                             mean_absolute_error,
                             mean_squared_error,
                             r2_score)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


def train_and_evaluate_model(X_train, X_test, y_train, y_test, column_names,
                             model, model_name):
    # Train the model.
    model.fit(X_train, y_train)

    # Save the model.
    with open(model_name + ".pkl", 'wb') as f:
        pickle.dump(model, f)

    # Compute feature importance using permutation.
    importance_results = permutation_importance(model, X_test, y_test,
                                                n_repeats=30, random_state=0,
                                                scoring="r2")
    importance = importance_results.importances_mean

    # Save the feature importance into the file.
    feature_importance = pandas.DataFrame({"Issue": column_names,
                                           "Importance": importance})
    feature_importance.to_csv(model_name + "_importance.csv",
                              columns=["Issue", "Importance"],
                              sep=',', index=False, float_format="%.8f")

    # Predict time_opened bases on the found issues.
    y_predicted = model.predict(X_test)

    # Save the expected and predicted values into the file.
    predicted_data = pandas.DataFrame({"Predicted": y_predicted,
                                       "Actual": y_test})
    predicted_data.to_csv(model_name + "_predicted.csv",
                          columns=["Predicted", "Actual"],
                          sep=',', index=False, float_format="%.4f")

    # Compute the metrics about model performance.
    mae = mean_absolute_error(y_test, y_predicted)
    mse = mean_squared_error(y_test, y_predicted)
    r2 = r2_score(y_test, y_predicted)
    ev = explained_variance_score(y_test, y_predicted)

    # Save the data about model performance into the file.
    model_metrics = pandas.DataFrame({"MAE": mae, "MSE": mse, "R2": r2,
                                      "EV": ev}, index=[0])
    model_metrics.to_csv(model_name + "_metrics.csv",
                         columns=["MAE", "MSE", "R2", "EV"],
                         sep=',', index=False, float_format="%.4f")


@click.command()
@click.argument("csv_file_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_folder_path", type=click.Path(exists=True,
                                                      file_okay=False))
def cli(csv_file_path: str, output_folder_path: str):
    # Import the dataset.
    df = pandas.read_csv(csv_file_path)
    df = df.drop(columns=[column for column in list(df) if not
                          (column == "time_opened" or
                           column.startswith("results_"))])

    cols = [c for c in df.columns if "results_" in c]
    X = df[cols]
    X = np.array(X)

    y = df["time_opened"]
    y = np.array(y)

    # Split the dataset into training and test sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=0)
    # Define the regressors that will be used.
    regressors = [
        (LinearRegression(n_jobs=-1), "LinearRegression"),
        (ElasticNet(random_state=0), "ElasticNet"),
        (DecisionTreeRegressor(random_state=0), "DecisionTree"),
        (RandomForestRegressor(random_state=0, n_estimators=100, n_jobs=-1),
         "RandomForest"),
        (AdaBoostRegressor(random_state=0, n_estimators=100), "AdaBoost"),
        (BaggingRegressor(random_state=0, n_estimators=100, n_jobs=-1),
         "Bagging"),
        (GradientBoostingRegressor(random_state=0, n_estimators=100),
         "GradientBoost")
    ]

    # Set the output directory as working directory.
    os.chdir(output_folder_path)

    # Train and evaluate defined regression models.
    for (model, model_name) in regressors:
        print(str(datetime.datetime.now()) +
              f": Starting to train {model_name} regressor.")
        train_and_evaluate_model(X_train, X_test, y_train, y_test, cols,
                                 model, model_name)


if __name__ == "__main__":
    cli()
