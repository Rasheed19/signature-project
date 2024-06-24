import numpy as np
import pandas as pd
import logging
import warnings

from steps import (
    data_loader,
    data_splitter,
    experimental_data_preprocessor,
    hyperparameter_tuner,
    model_trainer,
)
from utils.analyser import get_rrct_selected_features, metric_calculator
from utils.data_wrangler import create_knee_elbow_data
from utils.helper import config_logger
from utils.plotter import plot_rrct_driven_modelling_history

warnings.filterwarnings("ignore")


def rrct_driven_modelling_pipeline(
    not_loaded: bool,
    no_proposed_split: bool,
    sig_level: int,
    test_size: float,
    model_type: str,
    param_space: dict,
) -> None:
    """
    Pipeline for investigating the effect of
    varying the percetage of features selected by the RRCT
    feature selection algorithm on the proposed models.
    """

    config_logger()
    logger = logging.getLogger(__name__)

    loaded_data = data_loader(not_loaded=not_loaded)

    train_data, test_data = data_splitter(
        loaded_data=loaded_data,
        test_size=test_size,
        no_proposed_split=no_proposed_split,
    )
    targets = {
        k: create_knee_elbow_data(data_dict=data)
        for k, data in zip(["train", "test"], [train_data, test_data])
    }

    preprocessor = experimental_data_preprocessor(
        train_data=train_data,
        test_data=test_data,
        train_targets=targets["train"],
        test_targets=targets["test"],
        target_list=[
            "EOL" if model_type == "cycle_model" else "IRatEOL"
        ],  # only EOL and IRatEOL are considered for this experiment
        num_cycles=100,
        step_size=1,
        sig_level=sig_level,
    )

    percentages = np.linspace(0.1, 0.9, 9)
    metric_tracker = pd.DataFrame(
        columns=[
            "MAE_train",
            "MAPE_train",
            "RMSE_train",
            "MAE_test",
            "MAPE_test",
            "RMSE_test",
        ],
        index=percentages,
    )

    logger.info("Running experiment...")

    for p in percentages:

        print(f"selecting top {int(p * 100)}% of features for modelling.")

        selected_features = get_rrct_selected_features(
            num_features=int(
                p * preprocessor.X_train.shape[1]
            ),  # select p * 100 of the whole features
            X=preprocessor.X_train,
            y=preprocessor.y_train,
        )

        params = hyperparameter_tuner(
            X_train=preprocessor.X_train[selected_features],
            y_train=preprocessor.y_train,
            params=param_space[model_type],
        )

        model = model_trainer(
            X_train=preprocessor.X_train[selected_features],
            y_train=preprocessor.y_train,
            params=params,
        )

        train_scores = metric_calculator(
            y_true=preprocessor.y_train,
            y_pred=model.predict(preprocessor.X_train[selected_features]),
            multioutput="uniform_average",
        )
        test_scores = metric_calculator(
            y_true=preprocessor.y_test,
            y_pred=model.predict(preprocessor.X_test[selected_features]),
            multioutput="uniform_average",
        )

        metric_tracker.loc[p, ["MAE_train", "MAPE_train", "RMSE_train"]] = (
            train_scores["MAE"],
            train_scores["MAPE"],
            train_scores["RMSE"],
        )
        metric_tracker.loc[p, ["MAE_test", "MAPE_test", "RMSE_test"]] = (
            test_scores["MAE"],
            test_scores["MAPE"],
            test_scores["RMSE"],
        )

    metric_tracker = metric_tracker.apply(pd.to_numeric)
    metric_tracker.index = (metric_tracker.index * 100).astype(
        "int"
    )  # change index to percentages
    metric_tracker["MAPE_train"] = (
        metric_tracker["MAPE_train"] * 100
    )  # change MAPE to percentages
    metric_tracker["MAPE_test"] = (
        metric_tracker["MAPE_test"] * 100
    )  # change MAPE to percentages

    plot_rrct_driven_modelling_history(
        metric_tracker=metric_tracker,
        model_type=model_type,
    )

    logger.info(
        "RRCT-driven modelling pipeline has finished successfully. "
        "See the 'plots' folder for the generated plots"
    )

    return None
