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
    metric_calculator,
)
from utils.helper import config_logger
from utils.plotter import plot_cycle_number_effect_history


def cycle_number_effect_pipeline(
    not_loaded: bool,
    no_proposed_split: bool,
    sig_level: int,
    test_size: float,
    model_type: str,
    param_space: dict,
) -> None:
    """
    Pipeline for investigating the effect of the
    number of cycles of data on the proposed
    models.
    """

    config_logger()
    logger = logging.getLogger(__name__)

    list_of_cycles = np.arange(10, 101, 2)

    if model_type == "cycle_model":
        target_list = ["k-o", "k-p", "EOL", "e-o", "e-p"]
        split_list = [target_list[:3], target_list[3:]]
    elif model_type == "capacity_ir_model":
        target_list = ["Qatk-o", "Qatk-p", "IRate-o", "IRate-p", "IRatEOL"]
        split_list = [target_list[:2], target_list[2:]]
    else:
        raise ValueError(
            "'model_type' must be 'cycle_model' or 'capacity_ir_model', "
            f"but {model_type} is given."
        )

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

    history: dict[str, dict] = {}

    logger.info("Running cycle number effect experiment...")

    for split in split_list:
        crossval_mae, crossval_rmse = [], []
        test_mae, test_rmse = [], []
        crossval_mae_ci, crossval_rmse_ci = [], []
        test_mae_ci, test_rmse_ci = [], []

        logger.info(f"Running experiment with {', '.join(split)} as targets...")

        for n in list_of_cycles:
            print(f"running experiment with: {n} cycles")

            preprocessor = experimental_data_preprocessor(
                train_data=train_data,
                test_data=test_data,
                train_targets=targets["train"],
                test_targets=targets["test"],
                target_list=split,
                num_cycles=n,
                step_size=1,
                sig_level=sig_level,
            )

            params = hyperparameter_tuner(
                X_train=preprocessor.X_train,
                y_train=preprocessor.y_train,
                params=param_space[model_type],
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
                cv=5,
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

            test_scores = metric_calculator(
                preprocessor.y_test,
                model.predict(preprocessor.X_test),
                multioutput="uniform_average",
            )
            test_mae.append(test_scores["MAE"])
            test_rmse.append(test_scores["RMSE"])

            test_error = np.ravel(
                (preprocessor.y_test - model.predict(preprocessor.X_test))
            )
            test_mae_ci.append(
                confidence_interval_any(
                    values=abs(test_error),
                    n_bootstraps=1000,
                    metric_type="mae",
                    alpha=0.1,
                )
            )
            test_rmse_ci.append(
                confidence_interval_any(
                    values=test_error, n_bootstraps=1000, metric_type="rmse", alpha=0.1
                )
            )

        history[", ".join(split)] = dict(
            crossval_mae=crossval_mae,
            crossval_rmse=crossval_rmse,
            crossval_mae_ci=np.array(crossval_mae_ci),
            crossval_rmse_ci=np.array(crossval_rmse_ci),
            test_mae=test_mae,
            test_rmse=test_rmse,
            test_mae_ci=np.array(test_mae_ci),
            test_rmse_ci=np.array(test_rmse_ci),
        )

    for eval_type in ["crossval", "test"]:
        plot_cycle_number_effect_history(
            list_of_cycles=list_of_cycles,
            history=history,
            model_type=model_type,
            evaluation_type=eval_type,
        )

    logger.info(
        "Cycle number effect experiment finished successfully. "
        "Check the 'plots' folder for the generated figures."
    )

    return None
