import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats
import scipy.signal as sg
from sklearn.multioutput import MultiOutputRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import utils_gn, utils_sig
import importlib

importlib.reload(utils_gn)
importlib.reload(utils_sig)
xgb.set_config(verbosity=0)


def knee_elbow_detection(
    x_data,
    y_data,
    type,
    want_clean_data=False,
    p0=None,
    p0_db=None,
    p0_exp=None,
    plot=False,
    ylabel=None,
    ylim=None,
    title=None,
    point_name1=None,
    point_name2=None,
):
    """
    Function that detect knees and elbows by fitting Bacon-Watts and Double Bacon-Watts to a given data.

    Args:
    ----
        -x_data:      an array of independent variable values
        -y_data:      an array of dependent variable values
        -type:        specifies which to detect: "knee" or "elbow"
        -p0:          an array of initial values for Bacon-Watts model
        -p0_db:       an array of initial values for Double Bacon-Watts model
        -p0_exp:      an array of initial values for exponential model
        -plot:        a boolean, either to plot the results or not
        -ylabel:      y-axis label
        -ylim:        y_axis limit
        -title:       figure title
        -point_name1: name of the marked point in Bacon-Watt
        -point_name2: name of the marked point in Double Bacon-Watt

    Returns:
    -------
           cleaned data/knees/elbows.
    """

    # Define the Bacon-Watts and Double Bacon-Watts models
    def bacon_watts_model(x, alpha0, alpha1, alpha2, x1):
        return alpha0 + alpha1 * (x - x1) + alpha2 * (x - x1) * np.tanh((x - x1) / 1e-8)

    def double_bacon_watts_model(x, alpha0, alpha1, alpha2, alpha3, x0, x2):
        return (
            alpha0
            + alpha1 * (x - x0)
            + alpha2 * (x - x0) * np.tanh((x - x0) / 1e-8)
            + alpha3 * (x - x2) * np.tanh((x - x2) / 1e-8)
        )

    # Define the exponential model for data transformation
    def exponential_model(x, a, b, c, d, e):
        return a * np.exp(b * x - c) + d * x + e

    # Remove outliers from y_data
    clean_data = sg.medfilt(y_data, 5)

    # Get the length of clean data
    cl = len(clean_data)

    # Fit isotonic regression to data to obtain monotonic data
    if type == "knee":
        isotonic_reg = IsotonicRegression(increasing=False)
    elif type == "elbow":
        isotonic_reg = IsotonicRegression()
    clean_data = isotonic_reg.fit_transform(x_data, clean_data)

    # Force convexity on the cleaned y_data to prevent early detection of onset
    if (p0_exp is None) and type == "knee":
        p0_exp = [-4, 5e-3, 10, 0, clean_data[0]]
        bounds = ([-np.inf, 0, 0, -1, 0], [0, np.inf, np.inf, 0, np.inf])
    elif (p0_exp is None) and type == "elbow":
        p0_exp = [4, 0.03, 22, 0, clean_data[0]]
        bounds = (0, np.inf)
    popt_exp, _ = curve_fit(
        exponential_model, x_data, clean_data, p0=p0_exp, bounds=bounds
    )
    clean_data = exponential_model(x_data, *popt_exp)

    if want_clean_data:
        return clean_data

    # Fit the Bacon-Watts model
    if (p0 is None) and type == "knee":
        p0 = [1, -1e-4, -1e-4, cl * 0.7]
        bw_bounds = ([-np.inf, -np.inf, -np.inf, cl / 4], [np.inf, np.inf, np.inf, cl])
    elif (p0 is None) and type == "elbow":
        p0 = [1, 1, 1, cl / 1.5 + 1]
        bw_bounds = (
            [-np.inf, -np.inf, -np.inf, cl / 1.5],
            [np.inf, np.inf, np.inf, cl],
        )
    popt, pcov = curve_fit(
        bacon_watts_model, x_data, clean_data, p0=p0, maxfev=50000, bounds=bw_bounds
    )
    confint = [popt[3] - 1.96 * np.diag(pcov)[3], popt[3] + 1.96 * np.diag(pcov)[3]]

    # Fit the Double Bacon-Watts
    if (p0_db is None) and type == "knee":
        p0_db = [
            popt[0],
            popt[1] + popt[2] / 2,
            popt[2],
            popt[2] / 2,
            0.8 * popt[3],
            1.1 * popt[3],
        ]
        dbw_bounds = (
            [-np.inf, -np.inf, -np.inf, -np.inf, cl / 4, cl / 2],
            [np.inf, np.inf, np.inf, np.inf, cl, cl],
        )
    elif (p0_db is None) and type == "elbow":
        p0_db = [1, 1, 1, 1, cl / 1.5 + 1, cl / 1.5 + 3]
        dbw_bounds = (
            [-np.inf, -np.inf, -np.inf, -np.inf, cl / 4, cl / 4],
            [np.inf, np.inf, np.inf, np.inf, cl, cl],
        )
    popt_db, pcov_db = curve_fit(
        double_bacon_watts_model,
        x_data,
        clean_data,
        p0=p0_db,
        maxfev=50000,
        bounds=dbw_bounds,
    )
    confint_db = [
        popt_db[4] - 1.96 * np.diag(pcov_db)[4],
        popt_db[4] + 1.96 * np.diag(pcov_db)[4],
    ]

    if plot:
        # Plot results
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(x_data, y_data, "b--", label="True data", alpha=0.7)
        ax[0].plot(x_data, clean_data, "g-", label="Cleaned data")
        ax[0].plot(
            x_data,
            bacon_watts_model(x_data, *popt),
            "r-",
            linewidth=2,
            label="Bacon-Watts",
        )
        ax[0].plot(
            [popt[3]],
            [bacon_watts_model(popt[3], *popt)],
            marker="o",
            markersize=5,
            markeredgecolor="black",
            markerfacecolor="black",
            label=point_name1,
        )
        ax[0].axvline(x=popt[3], color="black", linestyle="--")
        ax[0].fill_betweenx(
            ylim, x1=confint[0], x2=confint[1], color="k", alpha=0.3, label="95% C.I"
        )
        ax[0].set_xlabel("Cycle number", fontsize=16)
        ax[0].set_ylabel(ylabel, fontsize=16)
        ax[0].grid(alpha=0.3)
        ax[0].set_ylim(ylim)
        ax[0].set_title(title, fontsize=16)
        ax[0].legend()

        ax[1].plot(x_data, y_data, "b--", label="True data", alpha=0.7)
        ax[1].plot(x_data, clean_data, "g-", label="Cleaned data")
        ax[1].plot(
            x_data,
            double_bacon_watts_model(x_data, *popt_db),
            "r-",
            label="Double Bacon-Watts",
        )
        ax[1].plot(
            [popt_db[4]],
            [double_bacon_watts_model(popt_db[4], *popt_db)],
            marker="o",
            markersize=5,
            markeredgecolor="black",
            markerfacecolor="black",
            label=point_name2,
        )
        ax[1].axvline(x=popt_db[4], color="black", linestyle="--")
        ax[1].fill_betweenx(
            ylim,
            x1=confint_db[0],
            x2=confint_db[1],
            color="k",
            alpha=0.3,
            label="95% C.I",
        )
        ax[1].set_xlabel("Cycle number", fontsize=16)
        ax[1].set_ylabel(ylabel, fontsize=16)
        ax[1].grid(alpha=0.3)
        ax[1].set_ylim(ylim)
        ax[1].set_title(title, fontsize=16)
        ax[1].legend()

        plt.tight_layout()
        plt.show()

    if type == "knee":
        # Calculate values at knee-point and knee-onset
        k_o = popt_db[4]  # knee-onset
        k_p = popt[3]  # knee-point
        q_at_k_o = double_bacon_watts_model(
            popt_db[4], *popt_db
        )  # capacity at knee-onset
        q_at_k_p = bacon_watts_model(popt[3], *popt)  # capacity at knee-point

        return k_o, k_p, q_at_k_o, q_at_k_p

    if type == "elbow":
        # Calculate values at knee-point and knee-onset
        e_o = popt_db[4]  # elbow-onset
        e_p = popt[3]  # elbow-point
        ir_at_e_o = double_bacon_watts_model(
            popt_db[4], *popt_db
        )  # resistance at elbow-onset
        ir_at_e_p = bacon_watts_model(popt[3], *popt)  # resistance at elbow-point

        return e_o, e_p, ir_at_e_o, ir_at_e_p


def metrics_calculator(y_true, y_pred, multi=False):
    """
    A function that calculates regression metrics.

    Args:
    ----
              y_true (array):  an array containing the true values of y
              y_pred (array):  an array containing the predicted values of y
              multi (bool):    a boolean to specify multi-output option
    Returns:
    -------
            dictionary of MAE, MAPE, RMSE.
    """

    if multi:
        return {
            "MAE": mean_absolute_error(y_true, y_pred, multioutput="raw_values"),
            "MAPE": mean_absolute_percentage_error(
                y_true, y_pred, multioutput="raw_values"
            ),
            "RMSE": np.sqrt(
                mean_squared_error(y_true, y_pred, multioutput="raw_values")
            ),
        }

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
    }


def axis_to_fig(axis):
    """
    Converts axis to fig object.

    Args:
    ----
         axis (object): axis object

    Returns:
    -------
            transformed axis oject.
    """

    fig = axis.figure

    def transform(coord):
        return fig.transFigure.inverted().transform(axis.transAxes.transform(coord))

    return transform


def add_sub_axes(axis, rect):
    """
    Adds sub-axis to existing axis object.

    Args:
    ----
         axis (object):        axis object
         rect (list or tuple): list or tuple specifying axis dimension

    Returns:
    -------
           fig object with added axis.
    """
    fig = axis.figure
    left, bottom, width, height = rect
    trans = axis_to_fig(axis)
    figleft, figbottom = trans((left, bottom))
    figwidth, figheight = trans([width, height]) - trans([0, 0])
    return fig.add_axes([figleft, figbottom, figwidth, figheight])


def kfold_cross_validation(X, y, model, cv):
    """
    A function that performs k-fold cross validation/repeated
    k-fold cross validation.

    Ars:
    ---
              X, y (array):          training set
              model (object):        model to be validated
              cv (int or cv object): int or cv object like RepeatedKFold
    Returns:
    -------
            a dictionary with key as test score and value as (score value, std of score value).
    """

    # define metrics to be used
    metrics = {
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error",
        "MAPE": "neg_mean_absolute_percentage_error",
    }
    # metrics = {'MAE': 'neg_mean_absolute_error', 'RMSE': 'neg_root_mean_squared_error'}

    # calculate scores
    scores = cross_validate(model, X, y, scoring=metrics, cv=cv, n_jobs=-1)
    scores_summary = {
        key: abs(val).mean()
        for key, val in scores.items()
        if key in ["test_" + metric for metric in metrics]
    }
    scores_raw = {
        key: abs(val)
        for key, val in scores.items()
        if key in ["test_" + metric for metric in metrics]
    }

    return scores_summary, scores_raw


def create_time_steps():
    """
    Creates a dictionary of time step codes;
    format: {1: 0.05, 2: 0.1, ...}, where the
    key is the code and the value is the
    time in mins.
    """

    i = 0.05
    time_steps = []

    while i < 4:
        time_steps.append(i)
        i += 0.05

    return dict(zip(np.arange(len(time_steps)) + 1, time_steps))


def model_feature_selection(
    train_raw, test_raw, y_test_df, target_list, k_list, params, step_size=1
):
    """
    Function that performs feature selection through rrct; build
    model on selected features and return metrics
    for each selection threshold.

    Args:
    ----
        train_raw (dict):           raw train data from which  train features will be generated
        test_raw (dict):            raw test data from which test features will be generated
        y_test_df (DataFrame):      data frame of test target values
        target_list (list):         list of target names for prediction
        k_list (list):              list of feature selection thresholds
        params (dict):              parameter space for the XGBoost model
        step_size (int):            code for subsampling time steps; check the output of create_time_steps()


    Returns:
    -------
            pandas data frame of model metrics for each selection threshold.
    """

    model = TransformedTargetRegressor(
        XGBRegressor(**params), func=np.log10, inverse_func=antilog
    )

    track_metrics = pd.DataFrame(
        columns=[
            "MAE_train",
            "MAPE_train",
            "RMSE_train",
            "MAE_test",
            "MAPE_test",
            "RMSE_test",
        ],
        index=k_list,
    )

    for k in k_list:
        print("k: ", k)

        # Transforms raw data to training data
        tr = utils_gn.FeatureTransformation(
            n=100, feature_selection=True, k=k, step_size=step_size
        )
        X_train, y_train = tr.fit_transform(
            data=train_raw,
            targets=target_list,
            with_eol=True,
            sig_level=2,
            multi_cycle=False,
        )
        X_test = tr.transform(test_raw, sig_level=2, multi_cycle=False)
        y_test = y_test_df[target_list].values

        model = model.fit(X_train, y_train)
        train_scores = metrics_calculator(y_train, model.predict(X_train), multi=False)
        test_scores = metrics_calculator(y_test, model.predict(X_test), multi=False)

        track_metrics.loc[k, ["MAE_train", "MAPE_train", "RMSE_train"]] = (
            train_scores["MAE"],
            train_scores["MAPE"],
            train_scores["RMSE"],
        )
        track_metrics.loc[k, ["MAE_test", "MAPE_test", "RMSE_test"]] = (
            test_scores["MAE"],
            test_scores["MAPE"],
            test_scores["RMSE"],
        )

    return track_metrics


def model_feature_selection_robustness(
    train_raw, test_raw, y_test_df, target_list, params, step_size_dict, times_needed, k
):
    """
    This function tests the robustness of model and
    rrct to change in frequency of data subsampling.

    Args:
    ----
        train_raw (dict):           raw train data from which  train features will be generated
        test_raw (dict):            raw test data from which test features will be generated
        y_test_df (DataFrame):      data frame of test target values
        target_list (list):         list of target names for prediction
        params (dict):              parameter space for the XGBoost model
        step_size_dict (dict):      dictionary of sub-sampling time steps codes; check the output of create_time_steps()
        times_needed (list):        a list of time step codes (subset of keys of step_size_dict) needed for robustness
                                    investigation
        k (float):                  feature selection threshold


    Returns:
    -------
            pandas data frame of selected feature names and model metrics for each selection threshold.
    """

    # Initialize model
    model = TransformedTargetRegressor(
        XGBRegressor(**params), func=np.log10, inverse_func=antilog
    )

    # Initialize model tracker data frame
    track_metrics = pd.DataFrame(
        columns=[
            "Selected features",
            "MAE_train",
            "MAPE_train",
            "RMSE_train",
            "MAE_test",
            "MAPE_test",
            "RMSE_test",
        ],
        index=[step_size_dict[key] for key in times_needed],
    )

    for h in times_needed:
        print("h: ", step_size_dict[h])

        # Transforms raw data to training data
        tr = utils_gn.FeatureTransformation(
            n=100, feature_selection=True, k=k, step_size=h
        )
        X_train, y_train = tr.fit_transform(
            data=train_raw,
            targets=target_list,
            with_eol=True,
            sig_level=2,
            multi_cycle=False,
        )
        X_test, y_test = (
            tr.transform(test_raw, sig_level=2, multi_cycle=False),
            y_test_df[target_list].values,
        )

        # Fit the model
        model = model.fit(X_train, y_train)

        # Calculate metrics
        train_scores = metrics_calculator(y_train, model.predict(X_train), multi=False)
        test_scores = metrics_calculator(y_test, model.predict(X_test), multi=False)

        # Update tracker
        track_metrics.loc[
            step_size_dict[h],
            [
                "Selected features",
                "MAE_train",
                "MAPE_train",
                "RMSE_train",
                "MAE_test",
                "MAPE_test",
                "RMSE_test",
            ],
        ] = (
            tr.selected_feature_names[0],
            train_scores["MAE"],
            train_scores["MAPE"],
            train_scores["RMSE"],
            test_scores["MAE"],
            test_scores["MAPE"],
            test_scores["RMSE"],
        )

    return track_metrics


def test_of_robustness(
    model, model_tr, time_steps, X_test_data, y_test_data, targets, step_size_dict
):
    """
    A function to test the robustness of the model built under signature method. The test
    is carried out by using the model to predict targets using features generated under
    a list of time steps provided.

    Args:
    ----
         model:                 test model object
         model_tr:              model transformation object
         time_steps:            a list of sub-sampling time steps
         X_test_data:           raw test data from which  features will be generated
         y_test_data:           raw test data from which targets will be generated
         targets:               a list of targets for prediction
         step_size_dict:        a dictionary of time steps codes

    Returns:
    -------
            a list of time steps used, an array of errors
    """

    # Get the y_test from raw data
    y_test = y_test_data[targets].values

    mae = []
    # rmse = []

    time_used_in_mins = []

    for t in time_steps:

        print("h: ", step_size_dict[t])

        # Transfrom the test set
        X_test = utils_sig.ccv_signature_features(
            data_dict=X_test_data, step_size=t, n=100, sig_level=2, multi_cycle=False
        ).values
        X_test = model_tr.sc.transform(X_test)

        # Get the predictions from the model
        predictions = model.predict(X_test)
        test_scores = metrics_calculator(y_test, predictions, multi=True)

        # Append errors to the error list
        mae.append(test_scores["MAE"])
        # rmse.append(test_scores['RMSE'])

        # Update times in minutes
        time_used_in_mins.append(step_size_dict[t])

    return time_used_in_mins, np.array(mae)  # np.array(mae), np.array(rmse)


def antilog(x):
    """
    Calculates common antilogarithm of x (float).
    """
    return 10**x


class ModelPipeline:
    """
    Class that fit a multi-output XGBoost regression model
    to a given training data X (array of features), y (array
    of targets).

    Methods:
    -------
            fit: fit model to a given training data X (array of features), y (array
                 of targets)
    """

    model_type = "Extreme Gradient Boost Regressor"

    def __init__(self, params, transform_target):
        self.params = params  # model parameters (dict)
        self.best_model = None  # best model (object)
        self.transform_target = (
            transform_target  # bool; whether to log-transform target or not
        )

    def fit(self, X, y):
        if self.transform_target:
            self.best_model = TransformedTargetRegressor(
                MultiOutputRegressor(XGBRegressor(**self.params)),
                func=np.log10,
                inverse_func=antilog,
            )

            self.best_model.fit(X, y)
            return self.best_model

        self.best_model = MultiOutputRegressor(XGBRegressor(**self.params))
        self.best_model.fit(X, y)
        return self.best_model


class ModifiedQuadraticSpline:
    """
    Class that implements modified quadratic spline described in the
    Method section of the paper: https://www.sciencedirect.com/science/article/pii/S0378775322014549

    Methods:
    -------
            fit:       fit spline to a given pair x (array of independent
                       variable values), y (array of dependent variable values).

            evaluate:  evaluates the spline at given points x (array).
    """

    def __init__(self):
        self.sol = None
        self.points = None

    def fit(self, x, y):

        A = np.zeros((9, 9))
        A[0:2, 0:3] = np.array([[1, x[0], x[0] ** 2], [1, x[1], x[1] ** 2]])
        A[2:4, 3:6] = np.array([[1, x[1], x[1] ** 2], [1, x[2], x[2] ** 2]])
        A[4:6, 6:9] = np.array([[1, x[2], x[2] ** 2], [1, x[3], x[3] ** 2]])
        A[6, 1], A[6, 2], A[6, 4], A[6, 5] = 1, 2 * x[1], -1, -2 * x[1]
        A[7, 4], A[7, 5], A[7, 7], A[7, 8] = 1, 2 * x[2], -1, -2 * x[2]
        A[8, 2] = 1

        b = np.array([y[0], y[1], y[1], y[2], y[2], y[3], 0.0, 0.0, 0.0])

        self.sol = np.linalg.solve(A, b)
        self.points = x

        return self

    def evaluate(self, x):

        if x[0] < self.points[0] or x[-1] > self.points[-1]:
            return "Out of range of interpolation"
        res = []
        for el in x:
            if self.points[0] <= el < self.points[1]:
                res.append(self.sol[0] + self.sol[1] * el + self.sol[2] * el**2)
            elif self.points[1] <= el < self.points[2]:
                res.append(self.sol[3] + self.sol[4] * el + self.sol[5] * el**2)
            elif self.points[2] <= el <= self.points[3]:
                res.append(self.sol[6] + self.sol[7] * el + self.sol[8] * el**2)
        return res


def modified_spline_evaluation(x, y, eval_points):
    """
    Function that fits and evaluate spline at given points.

    Args:
    ----
         x, y (array):         arrays of points to be used to fit the spline
         eval_points (array):  points of evaluation

    Returns:
    -------
            array of evaluations.
    """
    spl = ModifiedQuadraticSpline()
    spl.fit(x, y)

    return spl.evaluate(eval_points)


def confidence_interval_estimate(prediction, variance, confidence_level=0.95):
    """
    Function that estimates a confidence interval for a point prediction.

    Args:
    ----
         prediction (array):        predicted value
         variance (float):          estimated variance
         confidence_level (float):  level of certainty

    Returns:
    -------
           confidence interval for a given prediction.
    """
    tail_prob = (1 - confidence_level) / 2

    upper_z = stats.norm.ppf(1 - tail_prob)
    lower_z = stats.norm.ppf(tail_prob)

    return np.sqrt(variance) * prediction * np.array([lower_z, upper_z]) + prediction


def prediction_interval(
    X,
    y,
    model,
    n_bootstraps,
    target_list,
    predictions,
    confidence_level=0.95,
    plot_dist=False,
):
    """
    Function that calculates prediction interval for given predictions using the idea of bootstrapping.

    Args:
    ----
        X, y (array):              training set
        model (object):            unfitted model
        n_bootstraps (int):        number of bootstraps
        target_list (list):        list of target variables
        predictions (array):       predicted values
        confidence_level (float):  level of certainty
        plot_dist (bool):          specify whether to plot distribution of residuals or not

    Returns:
    -------
            prediction intervals, variances of residuals
    """
    residuals = []

    for _ in range(n_bootstraps):
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2)
        md = model.fit(X_tr, y_tr)
        pred = md.predict(X_val)
        residuals.append(((y_val - pred) / y_val).tolist())  # weighted residuals

    residuals = np.array(residuals)

    temp = []
    var_list = []

    if plot_dist:
        fig, ax = plt.subplots(1, len(target_list), figsize=(20, 4))

    for j in range(len(target_list)):
        for i in range(n_bootstraps):
            temp.append(residuals[i, :, j].tolist())
        temp = np.array(temp)
        if plot_dist:
            ax[j].set_title(target_list[j], fontsize=16)
            ax[j].grid()
            sns.distplot(temp.ravel(), kde=True, ax=ax[j])
            if j > 0:
                ax[j].set_ylabel("")
            else:
                ax[j].set_ylabel("Density of prediction errors", fontsize=14)
        var_list.append(np.var(temp.ravel()))
        temp = []

    if plot_dist:
        plt.show()

    return [
        [
            confidence_interval_estimate(el, var_list[j], confidence_level)
            for el in predictions[:, j]
        ]
        for j in range(len(target_list))
    ], var_list


def confidence_interval_metrics(
    actual, predictions, n_bootstraps, target_list, metric_type, alpha=0.05
):
    """
    Function that sets up a confidence interval for model metrics.

    Args:
    ----
        actual (array):      actual values
        predictions (array): predicted values
        n_bootstraps (int):  number of bootstraps
        target_list (list):  list of target variables
        metric_type (str):   type of metric
        alpha (float):       confidence level

    Returns:
    -------
            metric's confidence interval (list) for the targets.
    """
    target_metric_ci = []
    errors = actual - predictions
    alpha_tail = alpha / 2
    for i in range(len(target_list)):
        metric_estimates = []

        for _ in range(n_bootstraps):
            re_sample_idx = np.random.randint(0, len(errors[:, i]), errors[:, i].shape)

            if metric_type == "mae":
                metric_estimates.append(np.mean(np.abs(errors[:, i][re_sample_idx])))
            elif metric_type == "rmse":
                metric_estimates.append(
                    np.sqrt(np.mean((errors[:, i][re_sample_idx]) ** 2))
                )
            elif metric_type == "mape":
                metric_estimates.append(
                    np.mean(
                        abs((errors[:, i][re_sample_idx]) / actual[:, i][re_sample_idx])
                    )
                )

        sorted_estimates = np.sort(np.array(metric_estimates))
        conf_interval = [
            np.round(sorted_estimates[int(alpha_tail * n_bootstraps)], 6),
            np.round(sorted_estimates[int((1 - alpha_tail) * n_bootstraps)], 6),
        ]

        target_metric_ci.append(np.array(conf_interval))

    return target_metric_ci


def confidence_interval_any(values, n_bootstraps, metric_type=None, alpha=0.05):
    """
    Estimate condidence interval for any list of realizations of mae, mape,
    or rmse through bootsrapping.

    Args:
    ----
        values (array):      array of realizations of metric
        n_bootstraps (int):  number of bootstraps
        metric_type (str):   type of metric
        alpha (float):       confidence level

    Returns:
    -------
            confidence inteval (list).

    """
    alpha_tail = alpha / 2
    metric_estimates = []
    values = np.array(values)

    for _ in range(n_bootstraps):
        re_sample_idx = np.random.randint(0, len(values), values.shape)

        if metric_type == "rmse":
            metric_estimates.append(np.sqrt(np.mean((values[re_sample_idx]) ** 2)))
        else:
            metric_estimates.append(np.mean(values[re_sample_idx]))

    sorted_estimates = np.sort(np.array(metric_estimates))
    return [
        np.round(sorted_estimates[int(alpha_tail * n_bootstraps)], 6),
        np.round(sorted_estimates[int((1 - alpha_tail) * n_bootstraps)], 6),
    ]
