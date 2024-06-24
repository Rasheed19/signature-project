import os
import requests
import pickle
import yaml
from typing import Any
import logging
from itertools import islice
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import curve_fit
from scipy.signal import medfilt

from .definitions import ROOT_DIR


def read_data(fname: str, path: str) -> Any:
    """
    Function that reads .pkl file from a
    a given folder.

    Args:
    ----
        fname (str):  file name
        path (str): path to folder

    Returns:
    -------
            loaded file.
    """
    # Load pickle data
    with open(os.path.join(path, fname), "rb") as fp:
        loaded_file = pickle.load(fp)

    return loaded_file


def dump_data(data: Any, fname: str, path: str) -> None:
    """
    Function that dumps a pickled data into
    a specified path

    Args:
    ----
        data (Any): data to be pickled
        fname (str):  file name
        path (str): path to folder

    Returns:
    -------
            None
    """
    with open(os.path.join(path, fname), "wb") as fp:
        pickle.dump(data, fp)

    return None


def load_yaml_file(path: str) -> dict[Any, Any]:
    with open(path, "r") as file:
        data = yaml.safe_load(file)

    return data


class CustomFormatter(logging.Formatter):

    green = "\x1b[0;32m"
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(levelname)s: [%(name)s] %(message)s"
    # "[%(asctime)s] (%(name)s) %(levelname)s: %(message)s"
    # "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def config_logger() -> None:

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(CustomFormatter())

    logging.basicConfig(
        level=logging.INFO,
        handlers=[console_handler],
    )


def get_dict_subset(dictionary: dict, length: int) -> dict:
    return dict(islice(dictionary.items(), length))


def knee_elbow_detection(
    x_data: np.ndarray,
    y_data: np.ndarray,
    type: str,
    want_clean_data: bool = False,
    p0: np.ndarray | None = None,
    p0_db: np.ndarray | None = None,
    p0_exp: np.ndarray | None = None,
    plot: bool = False,
    ylabel: str | None = None,
    ylim: list[float] | None = None,
    title: str | None = None,
    point_name1: str | None = None,
    point_name2: str | None = None,
) -> tuple[float]:
    """
    Function that detect knees and elbows by fitting Bacon-Watts and Double Bacon-Watts to a given data.

    Args:
    ----
        -x_data:      an array of independent variable values
        -y_data:      an array of dependent variable values
        -type:        specifies which to detect: "knee" or "elbow"
        -p0:          an array of initial values for Bacon-Watts model
        -p0_db:       an array of initial values for Double Bacon-Watts model
        -p0_exp:      an array of initial values for exponential model
        -plot:        a boolean, either to plot the results or not
        -ylabel:      y-axis label
        -ylim:        y_axis limit
        -title:       figure title
        -point_name1: name of the marked point in Bacon-Watt
        -point_name2: name of the marked point in Double Bacon-Watt

    Returns:
    -------
           cleaned data/knees/elbows.
    """

    # Define the Bacon-Watts and Double Bacon-Watts models
    def bacon_watts_model(x, alpha0, alpha1, alpha2, x1):
        return alpha0 + alpha1 * (x - x1) + alpha2 * (x - x1) * np.tanh((x - x1) / 1e-8)

    def double_bacon_watts_model(x, alpha0, alpha1, alpha2, alpha3, x0, x2):
        return (
            alpha0
            + alpha1 * (x - x0)
            + alpha2 * (x - x0) * np.tanh((x - x0) / 1e-8)
            + alpha3 * (x - x2) * np.tanh((x - x2) / 1e-8)
        )

    # Define the exponential model for data transformation
    def exponential_model(x, a, b, c, d, e):
        return a * np.exp(b * x - c) + d * x + e

    # Remove outliers from y_data
    clean_data = medfilt(y_data, 5)

    # Get the length of clean data
    cl = len(clean_data)

    # Fit isotonic regression to data to obtain monotonic data
    if type == "knee":
        isotonic_reg = IsotonicRegression(increasing=False)
    elif type == "elbow":
        isotonic_reg = IsotonicRegression()
    clean_data = isotonic_reg.fit_transform(x_data, clean_data)

    # Force convexity on the cleaned y_data to prevent early detection of onset
    if (p0_exp is None) and type == "knee":
        p0_exp = [-4, 5e-3, 10, 0, clean_data[0]]
        bounds = ([-np.inf, 0, 0, -1, 0], [0, np.inf, np.inf, 0, np.inf])
    elif (p0_exp is None) and type == "elbow":
        p0_exp = [4, 0.03, 22, 0, clean_data[0]]
        bounds = (0, np.inf)
    popt_exp, _ = curve_fit(
        exponential_model, x_data, clean_data, p0=p0_exp, bounds=bounds
    )
    clean_data = exponential_model(x_data, *popt_exp)

    if want_clean_data:
        return clean_data

    # Fit the Bacon-Watts model
    if (p0 is None) and type == "knee":
        p0 = [1, -1e-4, -1e-4, cl * 0.7]
        bw_bounds = ([-np.inf, -np.inf, -np.inf, cl / 4], [np.inf, np.inf, np.inf, cl])
    elif (p0 is None) and type == "elbow":
        p0 = [1, 1, 1, cl / 1.5 + 1]
        bw_bounds = (
            [-np.inf, -np.inf, -np.inf, cl / 1.5],
            [np.inf, np.inf, np.inf, cl],
        )
    popt, pcov = curve_fit(
        bacon_watts_model, x_data, clean_data, p0=p0, maxfev=50000, bounds=bw_bounds
    )
    confint = [popt[3] - 1.96 * np.diag(pcov)[3], popt[3] + 1.96 * np.diag(pcov)[3]]

    # Fit the Double Bacon-Watts
    if (p0_db is None) and type == "knee":
        p0_db = [
            popt[0],
            popt[1] + popt[2] / 2,
            popt[2],
            popt[2] / 2,
            0.8 * popt[3],
            1.1 * popt[3],
        ]
        dbw_bounds = (
            [-np.inf, -np.inf, -np.inf, -np.inf, cl / 4, cl / 2],
            [np.inf, np.inf, np.inf, np.inf, cl, cl],
        )
    elif (p0_db is None) and type == "elbow":
        p0_db = [1, 1, 1, 1, cl / 1.5 + 1, cl / 1.5 + 3]
        dbw_bounds = (
            [-np.inf, -np.inf, -np.inf, -np.inf, cl / 4, cl / 4],
            [np.inf, np.inf, np.inf, np.inf, cl, cl],
        )
    popt_db, pcov_db = curve_fit(
        double_bacon_watts_model,
        x_data,
        clean_data,
        p0=p0_db,
        maxfev=50000,
        bounds=dbw_bounds,
    )
    confint_db = [
        popt_db[4] - 1.96 * np.diag(pcov_db)[4],
        popt_db[4] + 1.96 * np.diag(pcov_db)[4],
    ]

    if plot:
        # Plot results
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(x_data, y_data, "b--", label="True data", alpha=0.7)
        ax[0].plot(x_data, clean_data, "g-", label="Cleaned data")
        ax[0].plot(
            x_data,
            bacon_watts_model(x_data, *popt),
            "r-",
            linewidth=2,
            label="Bacon-Watts",
        )
        ax[0].plot(
            [popt[3]],
            [bacon_watts_model(popt[3], *popt)],
            marker="o",
            markersize=5,
            markeredgecolor="black",
            markerfacecolor="black",
            label=point_name1,
        )
        ax[0].axvline(x=popt[3], color="black", linestyle="--")
        ax[0].fill_betweenx(
            ylim, x1=confint[0], x2=confint[1], color="k", alpha=0.3, label="95% C.I"
        )
        ax[0].set_xlabel("Cycle number", fontsize=16)
        ax[0].set_ylabel(ylabel, fontsize=16)
        ax[0].grid(alpha=0.3)
        ax[0].set_ylim(ylim)
        ax[0].set_title(title, fontsize=16)
        ax[0].legend()

        ax[1].plot(x_data, y_data, "b--", label="True data", alpha=0.7)
        ax[1].plot(x_data, clean_data, "g-", label="Cleaned data")
        ax[1].plot(
            x_data,
            double_bacon_watts_model(x_data, *popt_db),
            "r-",
            label="Double Bacon-Watts",
        )
        ax[1].plot(
            [popt_db[4]],
            [double_bacon_watts_model(popt_db[4], *popt_db)],
            marker="o",
            markersize=5,
            markeredgecolor="black",
            markerfacecolor="black",
            label=point_name2,
        )
        ax[1].axvline(x=popt_db[4], color="black", linestyle="--")
        ax[1].fill_betweenx(
            ylim,
            x1=confint_db[0],
            x2=confint_db[1],
            color="k",
            alpha=0.3,
            label="95% C.I",
        )
        ax[1].set_xlabel("Cycle number", fontsize=16)
        ax[1].set_ylabel(ylabel, fontsize=16)
        ax[1].grid(alpha=0.3)
        ax[1].set_ylim(ylim)
        ax[1].set_title(title, fontsize=16)
        ax[1].legend()

        plt.tight_layout()
        plt.show()

    if type == "knee":
        # Calculate values at knee-point and knee-onset
        k_o = popt_db[4]  # knee-onset
        k_p = popt[3]  # knee-point
        q_at_k_o = double_bacon_watts_model(
            popt_db[4], *popt_db
        )  # capacity at knee-onset
        q_at_k_p = bacon_watts_model(popt[3], *popt)  # capacity at knee-point

        return k_o, k_p, q_at_k_o, q_at_k_p

    if type == "elbow":
        # Calculate values at knee-point and knee-onset
        e_o = popt_db[4]  # elbow-onset
        e_p = popt[3]  # elbow-point
        ir_at_e_o = double_bacon_watts_model(
            popt_db[4], *popt_db
        )  # resistance at elbow-onset
        ir_at_e_p = bacon_watts_model(popt[3], *popt)  # resistance at elbow-point

        return e_o, e_p, ir_at_e_o, ir_at_e_p


def save_cells_as_csv(cells: list[str], save_name: str) -> pd.DataFrame:

    csv = pd.DataFrame()
    csv[save_name] = cells

    csv.to_csv(f"{ROOT_DIR}/train_test_cells/{save_name}.csv")

    return csv


def antilog(x: float) -> float:
    """
    Calculates common antilogarithm of x (float).
    """
    return 10**x


def create_time_steps() -> dict[int, float]:
    """
    Creates a dictionary of time step codes;
    format: {1: 0.05, 2: 0.1, ...}, where the
    key is the code and the value is the
    time in mins.
    """

    start_time_in_mins = 0.05
    time_steps = []

    while start_time_in_mins < 4:
        time_steps.append(start_time_in_mins)
        start_time_in_mins += 0.05

    return dict(zip(np.arange(len(time_steps)) + 1, time_steps))


def download_file(
    url: str, file_name: str, folder: str = "data/severson_attia"
) -> None:
    os.makedirs(folder, exist_ok=True)

    response = requests.get(url)
    with open(f"{ROOT_DIR}/{folder}/{file_name}", "wb") as file:
        file.write(response.content)
