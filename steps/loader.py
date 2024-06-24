import logging

from utils.severson_attia import dump_severson_attia_structured_data
from utils.definitions import ROOT_DIR
from utils.helper import read_data, config_logger


def data_loader(not_loaded: bool) -> dict[str, dict]:

    config_logger()
    logger = logging.getLogger(__name__)
    logger.info("Loading data..")

    if not_loaded:
        dump_severson_attia_structured_data()

    data = read_data(path=ROOT_DIR, fname="data/structured_severson_attia_data.pkl")

    return data
