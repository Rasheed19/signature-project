import numpy as np
import pandas as pd
from utils import utils_models
import importlib

importlib.reload(utils_models)


def create_knee_elbow_data(data_dict, with_eol=True):
    """
    Function to create a dataframe with knee and elbow of cells.

    Args:
    ----
        data_dict (dict):    dictionary of battery cycling data

    Returns:
    -------
           pandas dataframe of knee and elbow of cells.
    """

    if with_eol:
        knee_elbow_data = pd.DataFrame(index=data_dict.keys(),
                                       columns=['k-o', 'k-p', 'Qatk-o', 'Qatk-p',
                                                'e-o', 'e-p', 'IRate-o', 'IRate-p', 'IRatEOL', 'EOL'])
        for cell in data_dict.keys():
            qd = data_dict[cell]['summary']['QDischarge']
            qd_eol = qd >= 0.88  # we focus on the definition of eol: cycle number at 80% of nominal capacity
            qd = qd[qd_eol]
            ir = data_dict[cell]['summary']['IR']
            ir = ir[qd_eol[:len(ir)]]  # we use the definition of eol to filter internal resistance
            knee_elbow_data.loc[cell, ['k-o', 'k-p', 'Qatk-o', 'Qatk-p']] = utils_models.knee_elbow_detection(
                x_data=np.arange(len(qd)) + 1,
                y_data=qd,
                type='knee')
            knee_elbow_data.loc[cell, ['e-o', 'e-p', 'IRate-o', 'IRate-p']] = utils_models.knee_elbow_detection(
                x_data=np.arange(len(ir)) + 1,
                y_data=ir,
                type='elbow')
            cleaned_ir = utils_models.knee_elbow_detection(x_data=np.arange(len(ir)) + 1,
                                                           y_data=ir,
                                                           type='elbow',
                                                           want_clean_data=True
                                                           )
            knee_elbow_data.loc[cell, ['IRatEOL', 'EOL']] = cleaned_ir[-1], len(
                qd)  # eol as the length of QDischarge vector

    else:
        knee_elbow_data = pd.DataFrame(index=data_dict.keys(),
                                       columns=['k-o', 'k-p', 'Qatk-o', 'Qatk-p', 'e-o', 'e-p', 'IRate-o', 'IRate-p'])
        for cell in data_dict.keys():
            qd = data_dict[cell]['summary']['QDischarge']
            ir = data_dict[cell]['summary']['IR']
            ttk_o, ttk_p, _, q_at_k_o, q_at_k_p = utils_models.knee_elbow_detection(x_data=np.arange(len(qd)) + 1,
                                                                                    y_data=qd, type='knee')
            knee_elbow_data.loc[cell, ['k-o', 'k-p', 'Qatk-o', 'Qatk-p']] = ttk_o, ttk_p, q_at_k_o, q_at_k_p
            knee_elbow_data.loc[cell, ['e-o', 'e-p', 'IRate-o', 'IRate-p']] = utils_models.knee_elbow_detection(
                x_data=np.arange(len(ir)) + 1,
                y_data=ir,
                type='elbow')
    # Just in case of wrong data type, change each entry to numerical data
    knee_elbow_data = knee_elbow_data.apply(pd.to_numeric)

    return knee_elbow_data
