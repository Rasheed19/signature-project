import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor

from utils.analyser import metric_calculator, metric_confidence_interval


def model_evaluator(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray,
    model: TransformedTargetRegressor,
    target_list: list[str],
) -> pd.DataFrame:

    predictions = model.predict(X)
    scores = metric_calculator(y, predictions, multioutput="raw_values")
    scores = pd.DataFrame.from_dict(scores)

    for col, met in zip(["MAE CI", "MAPE CI", "RMSE CI"], ["mae", "mape", "rmse"]):
        scores[col] = metric_confidence_interval(
            actual=y,
            predictions=predictions,
            n_bootstraps=10_000,
            target_list=target_list,
            metric_type=met,
            alpha=0.05,
        )

    scores.index = target_list

    return scores
