import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV

from utils.helper import antilog

xgb.set_config(verbosity=0)


def hyperparameter_tuner(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    params: dict,
) -> dict:

    model = TransformedTargetRegressor(
        regressor=XGBRegressor(),
        func=np.log10,
        inverse_func=antilog,
    )
    gs = GridSearchCV(
        estimator=model, param_grid=params, scoring="neg_mean_absolute_error", cv=3
    ).fit(
        X_train, y_train[:, -1] if len(y_train[0]) > 1 else y_train
    )  # model finetuned for the last target if y_train is a 2D array
    best_params = {k.split("__")[-1]: v for k, v in gs.best_params_.items()}

    return best_params
