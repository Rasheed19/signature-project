import hashlib
import numpy as np
import pandas as pd
import iisignature as isig
from scipy.interpolate import interp1d
from scipy.stats import skew, kurtosis
import pandas as pd

from .helper import knee_elbow_detection


def test_set_check(identifier: str, test_ratio: float) -> bool:
    """
    Function that checks if a sample belongs to a test set.

    Args:
    ----
        identifier:  identifier in the dataset
        test_ratio:  fraction of test set

    Returns:
    -------
            boolean corresponding to whether the hash of the identify <= test_ratio * 256
    """
    return hashlib.md5(identifier.encode()).digest()[-1] < 256 * test_ratio


def split_train_test_by_id(
    data: dict[str, dict], test_size: float
) -> tuple[dict, dict]:
    """
    Function to split data (cells) into train and test set.

    Args:
    ----
         data:        data to be split
         test_size:  fraction of test set

    Returns:
    -------
            train, test splits
    """
    np.random.seed(42)

    ids = np.array(list(data.keys()))

    # Shuffle the ids
    shuffled_indices = np.random.permutation(len(ids))
    ids = ids[shuffled_indices]

    in_test_set = [test_set_check(id_, test_size) for id_ in ids]
    ids_test = np.asarray(list(data.keys()))[in_test_set]

    return {k: data[k] for k in ids if k not in ids_test}, {
        k: data[k] for k in ids_test
    }


def evolution_fn(x: np.ndarray) -> list[float]:
    return [
        x.min(),
        x.max(),
        x.mean(),
        x.var(),
        skew(x),
        kurtosis(x, fisher=False),
    ]


def get_charge_discharge_values(
    data_dict: dict[str, dict], col_name: str, cell: str, cycle: str, option: str
) -> np.ndarray:
    """
    Function that extract only charge/discharge values of a given observed quantities.

    Args:
    ----
        data_dict (dict): a dictionary of battery cycling data
        col_name (str):   a string denoting name of observed quantity; e.g, 'I' for current
        cell (str):       a string denoting name of cell
        cycle (str):      a string denoting cycle number; e.g, '2'
        option (str):     a string specifying either pulling up "charge" or "discharge" values;

    Returns:
    -------
           returns extracted charge/discharge values
    """

    # An outlier in b1c2 at cycle 2176, measurement is in seconds and thus divide it by 60
    if cell == "b1c2" and cycle == "2176":
        summary_charge_time = (
            data_dict[cell]["summary"]["chargetime"][int(cycle) - 2] / 60
        )
    else:
        summary_charge_time = data_dict[cell]["summary"]["chargetime"][int(cycle) - 2]

    values = data_dict[cell]["cycle_dict"][cycle][col_name]

    if option == "charge":
        return np.array(
            values[
                data_dict[cell]["cycle_dict"][cycle]["t"] - summary_charge_time <= 1e-10
            ]
        )
    elif option == "discharge":
        return np.array(
            values[
                data_dict[cell]["cycle_dict"][cycle]["t"] - summary_charge_time > 1e-10
            ]
        )
    else:
        raise ValueError(f"option must be 'charge' or 'discharge but {option} given.")


def get_constant_indices(
    feature: list[float] | np.ndarray, option: str
) -> tuple[int, int]:
    """
    This function generates indices corresponding to the start
    and the end of constant values of a given feature.

    Args:
    ----
             feature (list/array): a list of considered feature, e.g. current, voltage
             option (str):         "charge" or "discharge" option

    Returns:
    -------
            tuple; start, end indices constant values for a given feature.
    """

    constant_feature_list = []
    constant_feature_index = []

    for i in range(1, len(feature)):
        if abs(feature[i - 1] - feature[i]) <= 1e-2:
            constant_feature_list.append(feature[i - 1])
            constant_feature_index.append(i - 1)

    if option == "charge":
        det_value = np.max(constant_feature_list)
        opt_list = [
            i
            for i, element in zip(constant_feature_index, constant_feature_list)
            if np.round(det_value - element, 2) <= 0.5
        ]

        return opt_list[0], opt_list[-1]

    elif option == "discharge":
        det_value = np.min(constant_feature_list)
        opt_list = [
            i
            for i, element in zip(constant_feature_index, constant_feature_list)
            if np.round(element - det_value, 2) <= 0.5
        ]
        return opt_list[0], opt_list[-1]

    else:
        raise ValueError(f"option must be 'charge' or 'discharge but {option} given.")


def strings_multi_cycle_features(num_cyles: int = 50) -> tuple[str]:
    """
    Function that returns names of generated features for a
    given cycle number.

    Args:
    ----
        num_cycles (int): a positive integer representing cycle number.

    Returns:
        tuple of feature names.
    """

    return (
        "Sig-1",
        f"Sig-{int(num_cyles / 2)}",
        f"Sig-{str(num_cyles)}",
        f"Sig-{str(num_cyles)}m1",
        "Sig-diff",
    )


def multi_cycle_features(feature_values_list: list[float], num_cycles: int = 50):
    """
    Function to generate cross-cycle features.

    Args:
    ----
         feature_values_list (list/array): list/array of feature values
         num_cycles (int):                          a positive integer representing cycle number

    Returns:
    -------
           list of cross-cycle feature values.
    """
    try:

        # Take 10% of num_cycles
        i = int(0.1 * num_cycles)

        # Create features corresponding to n
        y_0 = np.median(feature_values_list[:i])
        y_med = np.median(
            feature_values_list[int((num_cycles / 2) - i) : int((num_cycles / 2) + i)]
        )
        y_end = np.median(feature_values_list[-i:])
        y_endm0 = y_end - y_0
        y_diff = (y_end - y_med) - (y_med - y_0)

        return [y_0, y_med, y_end, y_endm0, y_diff]

    except TypeError:
        raise ValueError("num_cyles must be integer and >= 10")


def get_ccv_profile(
    data_dict: dict[str, dict], cell: str, cycle: str
) -> tuple[np.ndarray, np.ndarray]:
    discharge_values = {
        k: get_charge_discharge_values(data_dict, k, cell, cycle, "discharge")
        for k in ["I", "V", "t"]
    }

    # get the indices of the start and end of CC
    start_i, end_i = get_constant_indices(discharge_values["I"], "discharge")

    # get the corresponding voltages
    ccv = discharge_values["V"][start_i : end_i + 1]

    # get the corresponding time
    cct = discharge_values["t"][start_i : end_i + 1]
    cct = cct - min(cct)

    return cct, ccv


def ccv_signature_features(
    data_dict: dict[str, dict],
    step_size: int = 1,
    num_cycles: int = 50,
    sig_level: int = 2,
    multi_cycle: bool = True,
    return_ccv: bool = False,
    return_sig: bool = False,
) -> pd.DataFrame | dict:
    """
    Function that extracts features from battery cycling data using signature method.

    Args:
    ----
        data_dict:   dictionary containing battery cycling data
        step_size:   code for subsampling time steps
        num_cycles:  a positive integer indicating the number of cycles to use for feature extraction
        sig_level:   a positive integer indicating the number signature levels
        multi_cycle: a boolean either to return only multi-cycle features or not
        return_ccv:  a boolean either to return raw constant-current voltages only or not
        return_sig:  a boolean either to return signatures of ccv per cycle per cell only or not

    Returns:
    -------
            data frame of generated features.
    """

    sig_multi_features = []
    sig_evolution_features = []
    dict_for_signature = {}  # a dictionary to store signatures (an array) of each cell
    ccv_dict = {}  # initialize a dict to store ccv for each cycle for all cells

    for cell in data_dict.keys():
        signature_bucket = []

        # initialize a dictionary to store CCV for each cycle
        temp_cycle = {}

        for cycle in list(data_dict[cell]["cycle_dict"].keys())[:num_cycles]:

            cct, ccv = get_ccv_profile(data_dict=data_dict, cell=cell, cycle=cycle)
            temp_cycle[cycle] = [cct, ccv]

            # interpolation of voltage curve
            actual_length = len(cct)
            interested_length = int((1 / step_size) * actual_length)
            ccv_intp = interp1d(cct, ccv)
            a, b = min(cct), max(cct)
            t_interp = np.linspace(a, b, interested_length)
            ccv = ccv_intp(t_interp)

            # calculate the signature of the path
            path = np.stack((t_interp, ccv), axis=-1)
            signature = isig.sig(path, sig_level)
            signature_bucket.append(signature.tolist())

        ccv_dict[cell] = temp_cycle

        # get the multi cycle and evolution features
        signature_bucket = np.array(signature_bucket)
        sig_multi_union = []
        sig_evolution_union = []

        # Append the signature to the dictionary
        dict_for_signature[cell] = signature_bucket

        for i, _ in enumerate(signature_bucket[0]):
            sig_multi_union += multi_cycle_features(signature_bucket[:, i], num_cycles)
            sig_evolution_union += evolution_fn(signature_bucket[:, i])

        sig_multi_features.append(sig_multi_union)
        sig_evolution_features.append(sig_evolution_union)

    if return_ccv:
        return ccv_dict

    if return_sig:
        return dict_for_signature

    sig_multi_df = pd.DataFrame(
        data=np.array(sig_multi_features),
        columns=[
            ft + item
            for ft in ["S1-", "S2-", "S11-", "S12-", "S21-", "S22-"]
            for item in strings_multi_cycle_features(num_cycles)
        ],
        index=data_dict.keys(),
    )

    if multi_cycle:
        return sig_multi_df

    return sig_multi_df.join(
        pd.DataFrame(
            data=np.array(sig_evolution_features),
            columns=[
                stat + comp
                for comp in ["S1", "S2", "S11", "S12", "S21", "S22"]
                for stat in ["Min-", "Max-", "Mean-", "Var-", "Skew-", "Kurt-"]
            ],
            index=data_dict.keys(),
        )
    )


def create_knee_elbow_data(data_dict: dict[str, dict]) -> pd.DataFrame:
    """
    Function to create a dataframe with knee and elbow of cells.

    Args:
    ----
        data_dict (dict):    dictionary of battery cycling data

    Returns:
    -------
           pandas dataframe of knee and elbow of cells.
    """

    knee_elbow_data = pd.DataFrame(
        index=data_dict.keys(),
        columns=[
            "k-o",
            "k-p",
            "Qatk-o",
            "Qatk-p",
            "e-o",
            "e-p",
            "IRate-o",
            "IRate-p",
            "IRatEOL",
            "EOL",
        ],
    )
    for cell in data_dict.keys():
        qd = data_dict[cell]["summary"]["QDischarge"]
        qd_eol = (
            qd >= 0.88
        )  # we focus on the definition of eol: cycle number at 80% of nominal capacity
        qd = qd[qd_eol]
        ir = data_dict[cell]["summary"]["IR"]
        ir = ir[
            qd_eol[: len(ir)]
        ]  # we use the definition of eol to filter internal resistance
        knee_elbow_data.loc[cell, ["k-o", "k-p", "Qatk-o", "Qatk-p"]] = (
            knee_elbow_detection(x_data=np.arange(len(qd)) + 1, y_data=qd, type="knee")
        )
        knee_elbow_data.loc[cell, ["e-o", "e-p", "IRate-o", "IRate-p"]] = (
            knee_elbow_detection(x_data=np.arange(len(ir)) + 1, y_data=ir, type="elbow")
        )
        cleaned_ir = knee_elbow_detection(
            x_data=np.arange(len(ir)) + 1,
            y_data=ir,
            type="elbow",
            want_clean_data=True,
        )
        knee_elbow_data.loc[cell, ["IRatEOL", "EOL"]] = cleaned_ir[-1], len(
            qd
        )  # eol as the length of QDischarge vector

    # Just in case of wrong data type, change each entry to numerical data
    knee_elbow_data = knee_elbow_data.apply(pd.to_numeric)

    return knee_elbow_data


def cycle_life(data_dict: dict[str, dict]) -> pd.DataFrame:

    cycle_life = []
    for cell in data_dict.keys():
        qd = data_dict[cell]["summary"]["QDischarge"]
        qd_eol = (
            qd >= 0.88
        )  # we focus on the definition of eol: cycle number at 80% of nominal capacity
        qd = qd[qd_eol]

        cycle_life.append(len(qd))

    return pd.DataFrame(data=cycle_life, columns=["cycle_life"], index=data_dict.keys())


class DataFrameCaster:
    """Support class to cast type back to pd.DataFrame in sklearn Pipeline."""

    def __init__(self, columns: list):
        self.columns = columns

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X):
        return pd.DataFrame(X, columns=self.columns)
