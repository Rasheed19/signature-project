import numpy as np
import logging

from steps import (
    data_loader,
    data_splitter,
    experimental_data_preprocessor,
    hyperparameter_tuner,
    model_trainer,
)
from utils.data_wrangler import create_knee_elbow_data
from utils.analyser import (
    kfold_cross_validation,
    confidence_interval_any,
)
from utils.helper import config_logger, create_time_steps
from utils.plotter import plot_subsampling_time_effect_history


def subsampling_time_step_effect_pipeline(
    not_loaded: bool,
    no_proposed_split: bool,
    sig_level: int,
    test_size: float,
    param_space: dict,
) -> None:
    """
    Pipeline for investigating the effect of the
    subsampling time steps on the proposed
    models.
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

    time_step_map = create_time_steps()

    crossval_mae, crossval_rmse = [], []
    crossval_mae_ci, crossval_rmse_ci = [], []
    time_step_list: list[float] = []

    logger.info("Running experiment...")

    for time_step_code, time_step in zip(
        list(time_step_map.keys())[::2],
        list(time_step_map.values())[
            ::2
        ],  # only time steps with codes [1, 3, 5, ...] are considered for simplicity
    ):

        print(
            f"iteration at time step: {time_step:.2f} mins with code: {time_step_code}"
        )

        preprocessor = experimental_data_preprocessor(
            train_data=train_data,
            test_data=test_data,
            train_targets=targets["train"],
            test_targets=targets["test"],
            target_list=["EOL"],  # only EOL is considered for this experiment
            num_cycles=100,
            step_size=time_step_code,
            sig_level=sig_level,
        )

        params = hyperparameter_tuner(
            X_train=preprocessor.X_train,
            y_train=preprocessor.y_train,
            params=param_space["cycle_model"],
        )

        model = model_trainer(
            X_train=preprocessor.X_train,
            y_train=preprocessor.y_train,
            params=params,
        )

        val_scores_summary, val_scores_raw = kfold_cross_validation(
            X=preprocessor.X_train,
            y=preprocessor.y_train,
            model=model,
            cv=3,
        )
        crossval_mae.append(val_scores_summary["test_MAE"])
        crossval_rmse.append(val_scores_summary["test_RMSE"])
        crossval_mae_ci.append(
            confidence_interval_any(
                values=val_scores_raw["test_MAE"], n_bootstraps=1000, alpha=0.1
            )
        )
        crossval_rmse_ci.append(
            confidence_interval_any(
                values=val_scores_raw["test_RMSE"], n_bootstraps=1000, alpha=0.1
            )
        )
        time_step_list.append(time_step)

    history = dict(
        time_steps=time_step_list,
        crossval_mae=crossval_mae,
        crossval_rmse=crossval_rmse,
        crossval_mae_ci=np.array(crossval_mae_ci),
        crossval_rmse_ci=np.array(crossval_rmse_ci),
    )
    plot_subsampling_time_effect_history(history=history)

    logger.info(
        "Subsampling time step effect experiment has finished successfully. "
        "Check the 'plots' folder for the generated figures."
    )

    return None
