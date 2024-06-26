import logging
import numpy as np

from steps import (
    data_loader,
    data_splitter,
    data_preprocessor,
    experimental_data_preprocessor,
    hyperparameter_tuner,
    model_trainer,
)
from utils.data_wrangler import create_knee_elbow_data
from utils.analyser import metric_calculator
from utils.helper import config_logger, create_time_steps
from utils.plotter import plot_high_freq_model_robustness_history


def high_freq_model_robustness_pipeline(
    not_loaded: bool,
    no_proposed_split: bool,
    num_cyles: int,
    sig_level: int,
    test_size: float,
    param_space: dict,
) -> None:
    """
    Checking robustness through trainig a model
    on high frequency data and then testing it on
    data generated under low frequency data.
    """

    config_logger()
    logger = logging.getLogger(__name__)

    MODEL_NAMES: list[str] = ["cycle_model", "capacity_ir_model"]

    loaded_data = data_loader(not_loaded=not_loaded)

    train_data, test_data = data_splitter(
        loaded_data=loaded_data,
        test_size=test_size,
        no_proposed_split=no_proposed_split,
    )

    preprocessor = data_preprocessor(
        train_data=train_data,
        test_data=test_data,
        num_cycles=num_cyles,
        sig_level=sig_level,
        multi_cycle=False,
        step_size=1,  # high frequency model with subsampling step size of 0.05 mins and code 1
    )

    # hyparameter tuning for both models
    logger.info("Training high frequency model...")
    params = {
        k: hyperparameter_tuner(
            X_train=preprocessor.X_train,
            y_train=y,
            params=param_space[k],
        )
        for k, y in zip(
            MODEL_NAMES,
            [preprocessor.y_train_cycle, preprocessor.y_train_cap_ir],
        )
    }

    print("best hyperparameters: ")
    print(params)

    # train both models with the best hyperparameters
    models = {
        k: model_trainer(
            X_train=preprocessor.X_train,
            y_train=y,
            params=params[k],
        )
        for k, y in zip(
            MODEL_NAMES,
            [preprocessor.y_train_cycle, preprocessor.y_train_cap_ir],
        )
    }

    cycle_model_test_scores: list[list[float]] = []
    cap_ir_model_test_scores: list[list[float]] = []
    time_step_list: list[float] = []

    time_step_map = create_time_steps()
    time_step_code = np.arange(
        10, 90, step=10
    )  # only time steps with codes [10, 20, 30,  ...] are considered for brevity

    targets = {
        k: create_knee_elbow_data(data_dict=data)
        for k, data in zip(["train", "test"], [train_data, test_data])
    }

    logger.info("Running experiment...")

    for c in time_step_code:

        print(f"iteration at time step: {time_step_map[c]:.2f} mins with code: {c}")

        experimental_preprocessor = experimental_data_preprocessor(
            train_data=train_data,
            test_data=test_data,
            train_targets=targets["train"],
            test_targets=targets["test"],
            target_list=preprocessor.cycle_target_list
            + preprocessor.cap_ir_target_list,  # all targets are considered, will be separted later
            num_cycles=num_cyles,
            step_size=c,
            sig_level=sig_level,
            no_scaled_features=True,
        )

        temp_X_test = preprocessor.preprocess_pipeline.transform(
            experimental_preprocessor.X_test
        )  # use the high frequency pipeline for transformation

        cycle_model_test_scores.append(
            metric_calculator(
                y_true=experimental_preprocessor.y_test[
                    :, :5
                ],  # the first 5 labels are for cycle model
                y_pred=models["cycle_model"].predict(temp_X_test),
                multioutput="raw_values",
            )[
                "MAE"
            ].tolist()  # consider only MAE for brevity
        )
        cap_ir_model_test_scores.append(
            metric_calculator(
                y_true=experimental_preprocessor.y_test[
                    :, 5:
                ],  # the rest labels are for capacity-ir model
                y_pred=models["capacity_ir_model"].predict(temp_X_test),
                multioutput="raw_values",
            )["MAE"].tolist()
        )
        time_step_list.append(time_step_map[c])

    plot_high_freq_model_robustness_history(
        cycle_model_test_scores=np.array(cycle_model_test_scores),
        cap_ir_model_test_scores=np.array(cap_ir_model_test_scores),
        cycle_target_list=preprocessor.cycle_target_list,
        cap_ir_target_list=preprocessor.cap_ir_target_list,
        time_step_list=time_step_list,
    )

    logger.info(
        "High frequency model robustness experiment finished successfully. "
        "Check the 'plots' folder for the generated figures."
    )

    return None
