import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy import stats
from rrct.algorithm import RRCTFeatureSelection
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import train_test_split, cross_validate


def metric_calculator(y_true: np.ndarray, y_pred: np.ndarray, multioutput: str):
    return {
        "MAE": mean_absolute_error(y_true, y_pred, multioutput=multioutput),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred, multioutput=multioutput),
        "RMSE": mean_squared_error(
            y_true, y_pred, multioutput=multioutput, squared=False
        ),
    }


def prediction_interval_estimate(
    prediction: np.ndarray, variance: float, confidence_level: float = 0.95
) -> np.ndarray:
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
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray,
    model: TransformedTargetRegressor,
    n_bootstraps: int,
    target_list: list[str],
    predictions: np.ndarray,
    confidence_level: float = 0.95,
) -> tuple[list, list]:
    """
    Function that calculates prediction interval for given predictions using the idea of bootstrapping.

    Args:
    ----
        X, y (array):             training set
        model (object):           unfitted model
        n_bootstraps (int):       number of bootstraps
        target_list (list):       list of target variables
        predictions (array):      predicted values
        confidence_level (float): level of certainty

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

    for j in range(len(target_list)):
        for i in range(n_bootstraps):
            temp.append(residuals[i, :, j].tolist())
        temp = np.array(temp)
        var_list.append(np.var(temp.ravel()))
        temp = []

    return [
        [
            prediction_interval_estimate(el, var_list[j], confidence_level)
            for el in predictions[:, j]
        ]
        for j in range(len(target_list))
    ], var_list


def metric_confidence_interval(
    actual: np.ndarray,
    predictions: np.ndarray,
    n_bootstraps: int,
    target_list: list[str],
    metric_type: str,
    alpha: float = 0.05,
) -> list[np.ndarray]:
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


def confidence_interval_any(
    values: list[float] | np.ndarray,
    n_bootstraps: int,
    metric_type: str = "mae",
    alpha: float = 0.05,
) -> list[float]:
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


def kfold_cross_validation(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray,
    model: TransformedTargetRegressor,
    cv: int,
) -> tuple[dict, dict]:

    metrics = {
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error",
        "MAPE": "neg_mean_absolute_percentage_error",
    }

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


def format_prediction_and_interval(
    predictions: dict[str, list | np.ndarray],
    prediction_intervals: dict[str, list | np.ndarray],
    target_list: list[str],
    sample_cells: list[str],
) -> pd.DataFrame:
    """Create dataframe of predicted values
    and their prediction intervals"""

    prediction_df = pd.DataFrame(index=sample_cells, columns=target_list)

    for i, cell in enumerate(sample_cells):
        prediction_df.loc[cell, ["k-o", "k-p", "e-o", "e-p", "EOL"]] = predictions[
            "cycle_model"
        ][i]
        prediction_df.loc[
            cell, ["Qatk-o", "Qatk-p", "IRate-o", "IRate-p", "IRatEOL"]
        ] = predictions["capacity_ir_model"][i]

    for i, target in enumerate(target_list[:5]):
        prediction_df[f"{target} CI"] = prediction_intervals["cycle_model"][i]

    for i, target in enumerate(target_list[5:]):
        prediction_df[f"{target} CI"] = prediction_intervals["capacity_ir_model"][i]

    columns_rearranged = []
    for target in target_list:
        columns_rearranged.extend((target, f"{target} CI"))
    prediction_df = prediction_df[columns_rearranged]

    return prediction_df


class ModifiedQuadraticSpline:
    """
    Class that implements modified quadratic spline described in the
    Method section of the paper:
    https://www.sciencedirect.com/science/article/pii/S0378775322014549

    Methods:
    -------
            fit:       fit spline to a given pair x (array of independent
                       variable values), y (array of dependent variable values).

            evaluate:  evaluates the spline at given points x (array).
    """

    def __init__(self):
        self.sol: None | np.ndarray = None
        self.points: np.ndarray | list[float] = None

    def fit(self, x: np.ndarray | list[float], y: np.ndarray | list[float]):

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

    def evaluate(self, x: np.ndarray | list[float]):

        if x[0] < self.points[0] or x[-1] > self.points[-1]:
            raise ValueError("x boundary is out of range of interpolation")
        res: list[float] = []
        for el in x:
            if self.points[0] <= el < self.points[1]:
                res.append(self.sol[0] + self.sol[1] * el + self.sol[2] * el**2)
            elif self.points[1] <= el < self.points[2]:
                res.append(self.sol[3] + self.sol[4] * el + self.sol[5] * el**2)
            elif self.points[2] <= el <= self.points[3]:
                res.append(self.sol[6] + self.sol[7] * el + self.sol[8] * el**2)
        return np.array(res)


def modified_spline_evaluation(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    eval_points: np.ndarray | list[float],
) -> np.ndarray:
    """
    Function that fits and evaluate spline at given points.

    Args:
    ----
         x, y :       arrays of points to be used to fit the spline
         eval_points: points of evaluation

    Returns:
    -------
            array of evaluations.
    """
    spl = ModifiedQuadraticSpline()
    spl.fit(x, y)

    return spl.evaluate(eval_points)


@dataclass(frozen=True)
class PredictedFullCurve:
    actual_cycle: np.ndarray | list[int | float]
    actual_curve: np.ndarray | list[float]
    predicted_cycle: np.ndarray
    predicted_curve: np.ndarray
    predicted_cycle_lb: np.ndarray
    predicted_cycle_ub: np.ndarray
    predicted_curve_lb: np.ndarray
    predicted_curve_ub: np.ndarray


def predict_full_curve(
    actual_cycle: np.ndarray | list[int | float],
    actual_curve: np.ndarray | list[float],
    predicted_cycle_points: list[int | float],
    predicted_curve_points: list[float],
    predicted_cycle_points_lb: list[int | float],
    predicted_cycle_points_ub: list[int | float],
    predicted_curve_points_lb: list[float],
    predicted_curve_points_ub: list[float],
) -> PredictedFullCurve:

    all_predicted_cycles = {
        k: np.arange(1, int(v[-1]) + 1)
        for k, v in zip(
            ["values", "lb", "ub"],
            [
                predicted_cycle_points,
                predicted_cycle_points_lb,
                predicted_cycle_points_ub,
            ],
        )
    }

    all_predicted_curves = {}
    for k, x, y in zip(
        ["values", "lb", "ub"],
        [
            predicted_cycle_points,
            predicted_cycle_points_lb,
            predicted_cycle_points_ub,
        ],
        [
            predicted_curve_points,
            predicted_curve_points_lb,
            predicted_curve_points_ub,
        ],
    ):

        all_predicted_curves[k] = modified_spline_evaluation(
            x=x,
            y=y,
            eval_points=all_predicted_cycles[k],
        )

    return PredictedFullCurve(
        actual_cycle=actual_cycle,
        actual_curve=actual_curve,
        predicted_cycle=all_predicted_cycles["values"],
        predicted_cycle_lb=all_predicted_cycles["lb"],
        predicted_cycle_ub=all_predicted_cycles["ub"],
        predicted_curve=all_predicted_curves["values"],
        predicted_curve_lb=all_predicted_curves["lb"],
        predicted_curve_ub=all_predicted_curves["ub"],
    )


def feature_importance_analysis(
    model: TransformedTargetRegressor, feature_names: list[str], target_list: list[str]
) -> pd.DataFrame:
    """
    Function that calculates feature importance for a fitted model.

    Args:
    ----
         model:         model object
         feature_names: name of the features
         target_list:   list of targets

    Returns:
    -------
            data frame of feature importance.
    """

    # Create a lambda function to scale importance values to the interval [0, 1]
    scaler = lambda x: (x - x.min()) / (x.max() - x.min())

    # Get the importance list
    feature_importance = [
        scaler(model.regressor_.estimators_[i].feature_importances_)
        for i in range(len(target_list))
    ]
    # Cast feature importance list to a 2D numpy array
    feature_importance = np.array(feature_importance)

    return pd.DataFrame(
        data=feature_importance.T, columns=target_list, index=feature_names
    )


def get_rrct_selected_features(
    num_features: int,
    X: pd.DataFrame,
    y: np.ndarray,
) -> list[str]:

    selector = RRCTFeatureSelection(
        K=num_features,
        scale_feature=False,
    )
    selector = selector.apply(X.values, y)

    feature_names = np.array(X.columns)

    return feature_names[selector.selected_features_indices_].tolist()
