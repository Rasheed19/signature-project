import logging
import pandas as pd

from utils.helper import config_logger
from utils.data_wrangler import split_train_test_by_id
from utils.definitions import ROOT_DIR


def data_splitter(
    loaded_data: dict[str, dict],
    test_size: float,
    no_proposed_split: bool,
) -> tuple[dict[str, dict], dict[str, dict]]:

    config_logger()
    logger = logging.getLogger(__name__)

    logger.info("Splitting data for modelling...")

    if no_proposed_split:

        train_data, test_data = split_train_test_by_id(
            data=loaded_data, test_size=test_size
        )

        return train_data, test_data

    train_cells = pd.read_csv(f"{ROOT_DIR}/train_test_cells/train_cells.csv")
    test_cells = pd.read_csv(f"{ROOT_DIR}/train_test_cells/test_cells.csv")

    train_data = {cell: loaded_data[cell] for cell in train_cells["train_cells"].values}
    test_data = {cell: loaded_data[cell] for cell in test_cells["test_cells"].values}

    return train_data, test_data
