import click

from pipelines import (
    cycle_number_effect_pipeline,
    subsampling_time_step_effect_pipeline,
    rrct_robustness_pipeline,
    high_freq_model_robustness_pipeline,
    rrct_driven_modelling_pipeline,
)
from utils.helper import load_yaml_file
from utils.definitions import ROOT_DIR


@click.command(
    help="""
    Entry point for running pipelines that deal with
    experimental effects of various factors on models.
    """
)
@click.option(
    "--not-loaded",
    is_flag=True,
    default=False,
    help="""If given, raw data will be loaded and 
    cleaned.
        """,
)
@click.option(
    "--no-proposed-split",
    is_flag=True,
    default=False,
    help="""If given, train-test split used 
    in this study will not be used for modelling.
        """,
)
@click.option(
    "--model-type",
    default="cycle_model",
    help="""State which model to run experiment
    for. Valid options are 'cycle_model' for cycle model
    and 'capacity_ir_model' for capacity-ir model. Default 
    to 'cycle_model'.
        """,
)
@click.option(
    "--experiment-type",
    default="cycle-number-effect",
    help="""State which type of experiment to run.
    Valid options are 'cycle-number-effect',
    'time-step-effect', 'rrct-robustness', 'high-freq-model-robustness',
    and 'rrct-driven-modelling'. Default to 'cycle-number-effect'.
        """,
)
def main(
    not_loaded: bool = False,
    no_proposed_split: bool = False,
    model_type: str = "cycle",
    experiment_type: str = "cycle-number-effect",
) -> None:

    MODEL_CONFIG = load_yaml_file(path=f"{ROOT_DIR}/configs/model_config.yaml")
    DATA_CONFIG = load_yaml_file(path=f"{ROOT_DIR}/configs/data_config.yaml")

    if experiment_type == "cycle-number-effect":
        cycle_number_effect_pipeline(
            not_loaded=not_loaded,
            no_proposed_split=no_proposed_split,
            sig_level=DATA_CONFIG["sig_level"],
            test_size=MODEL_CONFIG["test_size"],
            model_type=model_type,
            param_space=MODEL_CONFIG["param_space"],
        )

    elif experiment_type == "time-step-effect":
        subsampling_time_step_effect_pipeline(
            not_loaded=not_loaded,
            no_proposed_split=no_proposed_split,
            sig_level=DATA_CONFIG["sig_level"],
            test_size=MODEL_CONFIG["test_size"],
            param_space=MODEL_CONFIG["param_space"],
        )

    elif experiment_type == "rrct-robustness":
        rrct_robustness_pipeline(
            not_loaded=not_loaded,
            no_proposed_split=no_proposed_split,
            sig_level=DATA_CONFIG["sig_level"],
            test_size=MODEL_CONFIG["test_size"],
            model_type=model_type,
        )

    elif experiment_type == "high-freq-model-robustness":
        high_freq_model_robustness_pipeline(
            not_loaded=not_loaded,
            no_proposed_split=no_proposed_split,
            sig_level=DATA_CONFIG["sig_level"],
            test_size=MODEL_CONFIG["test_size"],
            param_space=MODEL_CONFIG["param_space"],
        )

    elif experiment_type == "rrct-driven-modelling":
        rrct_driven_modelling_pipeline(
            not_loaded=not_loaded,
            no_proposed_split=no_proposed_split,
            sig_level=DATA_CONFIG["sig_level"],
            test_size=MODEL_CONFIG["test_size"],
            model_type=model_type,
            param_space=MODEL_CONFIG["param_space"],
        )

    else:
        raise ValueError(
            "experiment-type must be one of 'cycle-number-effect', "
            "'time-step-effect', 'rrct-robustness', 'high-freq-model-robustness', "
            f"and 'rrct-driven-modelling' but {experiment_type} is given."
        )

    return None


if __name__ == "__main__":
    main()
