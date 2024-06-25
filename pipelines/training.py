import logging
import numpy as np
from scipy.signal import medfilt

from steps import (
    data_loader,
    data_splitter,
    data_preprocessor,
    hyperparameter_tuner,
    model_trainer,
    model_evaluator,
)
from utils.helper import config_logger
from utils.analyser import (
    prediction_interval,
    format_prediction_and_interval,
    predict_full_curve,
    feature_importance_analysis,
)
from utils.plotter import (
    plot_predicted_full_curve,
    plot_parity_history,
    plot_feature_importance_analysis_history,
)


def training_pipeline(
    not_loaded: bool,
    no_proposed_split: bool,
    num_cycles: int,
    multi_cycle: bool,
    step_size: int,
    sig_level: int,
    test_size: float,
    param_space: dict,
    exclude_b8: bool,
    include_analysis: bool = False,
    include_curve_prediction: bool = False,
    sample_cells: list[str] | None = None,
) -> None:

    config_logger()
    logger = logging.getLogger(__name__)

    MODEL_NAMES: list[str] = ["cycle_model", "capacity_ir_model"]

    loaded_data = data_loader(not_loaded=not_loaded)

    if exclude_b8:
        loaded_data = {
            cell: loaded_data[cell] for cell in loaded_data if not cell.startswith("b8")
        }

    train_data, test_data = data_splitter(
        loaded_data=loaded_data,
        test_size=test_size,
        no_proposed_split=no_proposed_split,
    )

    preprocessor = data_preprocessor(
        train_data=train_data,
        test_data=test_data,
        num_cycles=num_cycles,
        sig_level=sig_level,
        multi_cycle=multi_cycle,
        step_size=step_size,
    )

    # TODO: rrct feature importance??

    # hyparameter tuning for both models
    logger.info("Training model...")
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

    # evaluate both models
    logger.info("Evaluating model..")
    for k, v in models.items():

        print("-" * 20)
        print(k)
        print("-" * 20)

        print("train metrics: ")
        train_metrics = model_evaluator(
            X=preprocessor.X_train,
            y=(
                preprocessor.y_train_cycle
                if k == "cycle_model"
                else preprocessor.y_train_cap_ir
            ),
            model=v,
            target_list=(
                preprocessor.cycle_target_list
                if k == "cycle_model"
                else preprocessor.cap_ir_target_list
            ),
        )
        print(train_metrics)

        print("test metrics: ")
        test_metrics = model_evaluator(
            X=preprocessor.X_test,
            y=(
                preprocessor.y_test_cycle
                if k == "cycle_model"
                else preprocessor.y_test_cap_ir
            ),
            model=v,
            target_list=(
                preprocessor.cycle_target_list
                if k == "cycle_model"
                else preprocessor.cap_ir_target_list
            ),
        )
        print(test_metrics)

    if include_analysis:

        logger.info("Feature importance and parity analysis in progress...")

        # parity plot
        cycle_train_predictions = models["cycle_model"].predict(preprocessor.X_train)
        cycle_test_predictions = models["cycle_model"].predict(preprocessor.X_test)

        capir_train_predictions = models["capacity_ir_model"].predict(
            preprocessor.X_train
        )
        capir_test_predictions = models["capacity_ir_model"].predict(
            preprocessor.X_test
        )

        parity_history = {}
        for i, l in enumerate(preprocessor.cycle_target_list):
            parity_history[l] = {
                "y_train": preprocessor.y_train_cycle[:, i],
                "y_test": preprocessor.y_test_cycle[:, i],
                "y_train_pred": cycle_train_predictions[:, i],
                "y_test_pred": cycle_test_predictions[:, i],
            }

        for i, l in enumerate(preprocessor.cap_ir_target_list):
            parity_history[l] = {
                "y_train": preprocessor.y_train_cap_ir[:, i],
                "y_test": preprocessor.y_test_cap_ir[:, i],
                "y_train_pred": capir_train_predictions[:, i],
                "y_test_pred": capir_test_predictions[:, i],
            }
        plot_parity_history(parity_history=parity_history)

        # feature importance analysis
        feature_importance_analysis_history = {}
        for m, l in zip(
            MODEL_NAMES,
            [preprocessor.cycle_target_list, preprocessor.cap_ir_target_list],
        ):
            feature_importance_analysis_history[m] = feature_importance_analysis(
                model=models[m],
                feature_names=list(preprocessor.X_train.columns),
                target_list=l,
            )

        for m, h in feature_importance_analysis_history.items():
            plot_feature_importance_analysis_history(analysis_df=h, figure_save_name=m)

    if include_curve_prediction and sample_cells is not None:

        logger.info(
            "Prediction capacity and internal resistance curves for sample cells..."
        )

        X_inf = preprocessor.X_test.copy()
        X_inf.index = test_data.keys()
        X_inf = X_inf.loc[sample_cells]  # do curve predictions for sample cells only

        predictions = {k: v.predict(X_inf.values) for k, v in models.items()}

        prediction_intervals = {
            k: prediction_interval(
                X=preprocessor.X_train,
                y=y,
                model=models[k],
                n_bootstraps=100,
                target_list=l,
                predictions=predictions[k],
                confidence_level=0.90,
            )[0]
            for k, l, y in zip(
                predictions.keys(),
                [preprocessor.cycle_target_list, preprocessor.cap_ir_target_list],
                [preprocessor.y_train_cycle, preprocessor.y_train_cap_ir],
            )
        }
        inf_history = format_prediction_and_interval(
            predictions=predictions,
            prediction_intervals=prediction_intervals,
            target_list=preprocessor.cycle_target_list
            + preprocessor.cap_ir_target_list,
            sample_cells=sample_cells,
        )

        predicted_curve_history = {}
        for cell in sample_cells:

            temp_history = {}

            for curve in ["QDischarge", "IR"]:

                actual_curve = medfilt(test_data[cell]["summary"][curve])
                actual_cycle = np.arange(1, actual_curve.shape[0] + 1)

                if curve == "QDischarge":
                    cycle_point_names = ("k-o", "k-p")
                    curve_point_names = ("Qatk-o", "Qatk-p")
                else:
                    cycle_point_names = ("e-o", "e-p")
                    curve_point_names = ("IRate-o", "IRate-p")

                predicted_curve_points = [
                    actual_curve[0],
                    inf_history.loc[cell][curve_point_names[0]],
                    inf_history.loc[cell][curve_point_names[1]],
                    (
                        actual_curve[-1]
                        if curve == "QDischarge"
                        else inf_history.loc[cell]["IRatEOL"]
                    ),
                ]

                predicted_curve = predict_full_curve(
                    actual_cycle=actual_cycle,
                    actual_curve=actual_curve,
                    predicted_cycle_points=[
                        1,
                        inf_history.loc[cell][cycle_point_names[0]],
                        inf_history.loc[cell][cycle_point_names[1]],
                        inf_history.loc[cell]["EOL"],
                    ],
                    predicted_cycle_points_lb=[
                        1,
                        inf_history.loc[cell][f"{cycle_point_names[0]} CI"][0],
                        inf_history.loc[cell][f"{cycle_point_names[1]} CI"][0],
                        inf_history.loc[cell]["EOL CI"][0],
                    ],
                    predicted_cycle_points_ub=[
                        1,
                        inf_history.loc[cell][f"{cycle_point_names[0]} CI"][1],
                        inf_history.loc[cell][f"{cycle_point_names[1]} CI"][1],
                        inf_history.loc[cell]["EOL CI"][1],
                    ],
                    # note that the CI in the plotted curves are generated from CI in cycle numbers;
                    # although that of CI of IR or capacities can also be used
                    predicted_curve_points_ub=predicted_curve_points,
                    predicted_curve_points=predicted_curve_points,
                    predicted_curve_points_lb=predicted_curve_points,
                )

                temp_history[curve] = predicted_curve

            predicted_curve_history[cell] = temp_history

        for curve in ["QDischarge", "IR"]:
            plot_predicted_full_curve(
                predicted_curve_history=predicted_curve_history,
                curve_name=curve,
            )

    logger.info(
        "Training pipeline has finished successfully. "
        "See the 'plots' folder for the generated plots"
    )

    return None
