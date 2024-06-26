import numpy as np
import logging
import warnings

from steps import (
    data_loader,
    data_splitter,
    experimental_data_preprocessor,
)
from utils.analyser import get_rrct_selected_features
from utils.data_wrangler import create_knee_elbow_data
from utils.helper import config_logger, create_time_steps
from utils.plotter import (
    plot_rrct_robustness_heatmap,
    plot_top_10p_rrct_selected_features,
)

warnings.filterwarnings("ignore")


def rrct_robustness_pipeline(
    not_loaded: bool,
    no_proposed_split: bool,
    num_cycles: int,
    sig_level: int,
    test_size: float,
    model_type: str,
) -> None:
    """
    Pipeline for investigating the effect of the
    subsampling time steps on the nature of features
    selected by the RRCT feature selection algorithm.
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
    TARGET_NAME = (
        "EOL" if model_type == "cycle_model" else "IRatEOL"
    )  # only EOL and IRatEOL is considered for this experiment

    time_step_map = create_time_steps()
    time_step_code = np.arange(0, 90, step=10)
    time_step_code[0] = (
        1  # only time steps with codes [1, 10, 20 , ...] are considered for brevity
    )
    time_step_list: list[float] = []

    selected_features_list: list[list[str]] = []
    NO_OF_SELECTED_FEATURES: int = 10  # maximum number of selected features

    logger.info("Running experiment...")

    for c in time_step_code:

        print(f"iteration at time step: {time_step_map[c]:.2f} mins with code: {c}")

        preprocessor = experimental_data_preprocessor(
            train_data=train_data,
            test_data=test_data,
            train_targets=targets["train"],
            test_targets=targets["test"],
            target_list=[TARGET_NAME],
            num_cycles=num_cycles,
            step_size=c,
            sig_level=sig_level,
        )

        selected_features = get_rrct_selected_features(
            num_features=NO_OF_SELECTED_FEATURES,
            X=preprocessor.X_train,
            y=preprocessor.y_train,
        )
        selected_features_list.append(selected_features)
        time_step_list.append(time_step_map[c])

    # calculate similarity scores
    similarity_scores = [
        [
            len(np.intersect1d(features, others, assume_unique=False))
            / NO_OF_SELECTED_FEATURES
            for others in selected_features_list
        ]
        for features in selected_features_list
    ]
    similarity_scores = np.array(similarity_scores)
    plot_rrct_robustness_heatmap(
        similarity_scores=similarity_scores, time_step_list=time_step_list
    )

    plot_top_10p_rrct_selected_features(
        selected_features=selected_features_list[-1][
            :6
        ],  # we only have interest in the rank of top 10% (first 6) features generated under 4 min (code 80) subsampling rate
        figure_tag=TARGET_NAME,
    )

    logger.info(
        "RRCT robustness to time step subsampling experiment has finished successfully. "
        "Check the 'plots' folder for the generated figures."
    )

    return None
