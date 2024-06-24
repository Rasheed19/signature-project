import logging
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.compose import TransformedTargetRegressor
from sklearn.multioutput import MultiOutputRegressor

from utils.helper import antilog

xgb.set_config(verbosity=0)


def model_trainer(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    params: dict,
) -> TransformedTargetRegressor:

    regressor = (
        MultiOutputRegressor(XGBRegressor(**params))
        if len(y_train[0]) > 1
        else XGBRegressor(**params)
    )  # don't wrap estimator in multi-output if y_train is 1D

    model = TransformedTargetRegressor(
        regressor=regressor,
        func=np.log10,
        inverse_func=antilog,
    )

    model.fit(X_train, y_train)

    return model
