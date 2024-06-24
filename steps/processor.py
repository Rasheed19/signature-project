import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
import logging

from utils.data_wrangler import (
    create_knee_elbow_data,
    ccv_signature_features,
    DataFrameCaster,
)
from utils.helper import config_logger


def imputer_scaler_caster_pipeline(feature_names: list):

    return Pipeline(
        [
            ("imputer", SimpleImputer()),
            ("scaler", StandardScaler()),
            ("cast", DataFrameCaster(columns=feature_names)),
        ]
    )


@dataclass(frozen=True)
class DataPreprocessorOutput:

    X_train: pd.DataFrame
    y_train_cycle: np.ndarray
    y_train_cap_ir: np.ndarray
    X_test: pd.DataFrame
    y_test_cycle: np.ndarray
    y_test_cap_ir: np.ndarray
    preprocess_pipeline: Pipeline
    cycle_target_list: list[str]
    cap_ir_target_list: list[str]


@dataclass(frozen=True)
class ExperimentalDataPreprocessorOutput:

    X_train: pd.DataFrame
    y_train: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray
    preprocess_pipeline: Pipeline | None
    target_list: list[str]


def data_preprocessor(
    train_data: dict[str, dict],
    test_data: dict[str, dict],
    num_cycles: int,
    sig_level: int,
    multi_cycle: bool,
    step_size: int,
) -> DataPreprocessorOutput:

    config_logger()
    logger = logging.getLogger(__name__)

    logger.info("Preprocessing data for modelling...")

    features = {
        k: ccv_signature_features(
            data_dict=data,
            step_size=step_size,
            num_cycles=num_cycles,
            sig_level=sig_level,
            multi_cycle=multi_cycle,
            return_ccv=False,
            return_sig=False,
        )
        for k, data in zip(["train", "test"], [train_data, test_data])
    }

    targets = {
        k: create_knee_elbow_data(data_dict=data)
        for k, data in zip(["train", "test"], [train_data, test_data])
    }

    preprocess_pipeline = imputer_scaler_caster_pipeline(
        feature_names=features["train"].columns
    )
    X_train = preprocess_pipeline.fit_transform(features["train"])
    X_test = preprocess_pipeline.transform(features["test"])

    cycle_target_list = ["k-o", "k-p", "e-o", "e-p", "EOL"]
    cap_ir_target_list = ["Qatk-o", "Qatk-p", "IRate-o", "IRate-p", "IRatEOL"]

    return DataPreprocessorOutput(
        X_train=X_train,
        y_train_cycle=targets["train"][cycle_target_list].values,
        y_train_cap_ir=targets["train"][cap_ir_target_list].values,
        X_test=X_test,
        y_test_cycle=targets["test"][cycle_target_list].values,
        y_test_cap_ir=targets["test"][cap_ir_target_list].values,
        preprocess_pipeline=preprocess_pipeline,
        cycle_target_list=cycle_target_list,
        cap_ir_target_list=cap_ir_target_list,
    )


def experimental_data_preprocessor(
    train_data: dict[str, dict],
    test_data: dict[str, dict],
    train_targets: pd.DataFrame,
    test_targets: pd.DataFrame,
    target_list: list[str],
    num_cycles: int,
    step_size: int,
    sig_level: int,
    no_scaled_features: bool = False,
):
    features = {
        k: ccv_signature_features(
            data_dict=data,
            step_size=step_size,
            num_cycles=num_cycles,
            sig_level=sig_level,
            multi_cycle=False,
            return_ccv=False,
            return_sig=False,
        )
        for k, data in zip(["train", "test"], [train_data, test_data])
    }

    if no_scaled_features:
        return ExperimentalDataPreprocessorOutput(
            X_train=features["train"],
            y_train=train_targets[target_list].values,
            X_test=features["test"],
            y_test=test_targets[target_list].values,
            preprocess_pipeline=None,
            target_list=target_list,
        )

    preprocess_pipeline = imputer_scaler_caster_pipeline(
        feature_names=features["train"].columns
    )
    X_train = preprocess_pipeline.fit_transform(features["train"])
    X_test = preprocess_pipeline.transform(features["test"])

    return ExperimentalDataPreprocessorOutput(
        X_train=X_train,
        y_train=train_targets[target_list].values,
        X_test=X_test,
        y_test=test_targets[target_list].values,
        preprocess_pipeline=preprocess_pipeline,
        target_list=target_list,
    )
