import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scipy.stats
import iisignature as isig
from utils import utils_gn
import importlib

importlib.reload(utils_gn)


def ccv_signature_features(
    data_dict,
    step_size=1,
    n=50,
    sig_level=2,
    multi_cycle=True,
    return_ccv=False,
    return_sig=False,
):
    """
    Function that extracts features from battery cycling data using signature method.

    Args:
    ----
        data_dict:   dictionary containing battery cycling data
        step_size:   code for subsampling time steps; check the output of create_time_steps() in utils_models
        n:           a positive integer indicating the number of cycles to use for feature extraction
        sig_level:   a positive integer indicating the number signature levels
        multi_cycle: a boolean either to return only multi-cycle features or not
        return_ccv:  a boolean either to return raw constant-current voltages only or not
        return_sig:  a boolean either to return signatures of ccv per cycle per cell only or not

    Returns:
    -------
            data frame of generated features.
    """

    def evolution_fn(x):
        return [
            x.min(),
            x.max(),
            x.mean(),
            x.var(),
            scipy.stats.skew(x),
            scipy.stats.kurtosis(x, fisher=False),
        ]

    sig_multi_features = []
    sig_evolution_features = []
    dict_for_signature = {}  # a dictionary to store signatures (an array) of each cell
    ccv_dict = {}  # initialize a dict to store ccv for each cycle for all cells

    for cell in data_dict.keys():
        signature_bucket = []

        # initialize a dictionary to store CCV for each cycle
        this_cycle = {}

        for cycle in list(data_dict[cell]["cycle_dict"].keys())[:n]:
            # get the discharge values
            i_values = utils_gn.get_charge_discharge_values(
                data_dict, "I", cell, cycle, "di"
            )
            v_values = utils_gn.get_charge_discharge_values(
                data_dict, "V", cell, cycle, "di"
            )
            t_values = utils_gn.get_charge_discharge_values(
                data_dict, "t", cell, cycle, "di"
            )

            # get the indices of the start and end of CC
            start_i, end_i = utils_gn.get_constant_indices(i_values, "di")

            # get the corresponding voltages
            ccv = v_values[start_i : end_i + 1]

            # get the corresponding time
            cct = t_values[start_i : end_i + 1]
            cct = cct - min(cct)

            this_cycle[cycle] = [cct, ccv]

            # interpolation of voltage curve
            actual_length = len(cct)
            interested_length = int((1 / step_size) * actual_length)

            ccv_intp = interp1d(cct, ccv)
            a, b = min(cct), max(cct)
            t_interp = np.linspace(a, b, interested_length)
            ccv = ccv_intp(t_interp)

            # create a two-dimensional path with time as the first component and ccv as the second component
            path = np.stack((t_interp, ccv), axis=-1)

            # calculate the signature of the path
            signature = isig.sig(path, sig_level)  # for the signature of the path

            signature_bucket.append(signature.tolist())

        ccv_dict[cell] = this_cycle

        # get the multi cycle and evolution features
        signature_bucket = np.array(signature_bucket)
        sig_multi_union = []
        sig_evolution_union = []

        # Append the signature to the dictionary
        dict_for_signature[cell] = signature_bucket

        for i, _ in enumerate(signature_bucket[0]):
            sig_multi_union += utils_gn.multi_cycle_features(signature_bucket[:, i], n)
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
            for item in utils_gn.strings_multi_cycle_features(n)
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


# return dict_for_signature


def plot_ccv_features(
    data_dict, fname, ylabel=None, ylim=None, sample_cells=None, option=1
):
    """
    Plot the evolution of constant-current voltage for sample cells.
    """
    if option == 1:
        # get the cells belonging to the same batch
        b1 = [cell for cell in data_dict.keys() if cell[:2] == "b1"]
        b2 = [cell for cell in data_dict.keys() if cell[:2] == "b2"]
        b3 = [cell for cell in data_dict.keys() if cell[:2] == "b3"]

        x_labels = dict(
            zip(
                data_dict["b1c0"].keys(),
                [
                    "Cycles",
                    r"Internal resistance ($\Omega$)",
                    "Min of CCV (V)",
                    "Max of CCV (V)",
                    "Mean of CCV (V)",
                    "Variance of CCV (V)",
                    "Skewness of CCV",
                    "Kurtosis of CCV",
                    "Area under CC Voltage Curve",
                    "Capacity (Ah)",
                ],
            )
        )

        for batch in [b1, b2, b3]:
            fig, ax = plt.subplots(3, 3, figsize=(20, 15))
            i = 0
            for feature in data_dict["b1c0"].keys():
                if feature not in [ylabel]:
                    for cell in batch:
                        ax[i // 3, i % 3].plot(
                            data_dict[cell][ylabel],
                            data_dict[cell][feature],
                            "o",
                            linewidth=1,
                            markersize=2,
                        )
                        ax[i // 3, i % 3].set_ylabel(x_labels[feature], fontsize=14)
                        ax[i // 3, i % 3].set_xlabel(x_labels[ylabel], fontsize=14)
                        ax[i // 3, i % 3].set_ylim(ylim)
                    i += 1
            i = 0

            # handles, labels = ax.get_legend_handles_labels()
            # fig.legend(handles, labels)

            plt.show()

    elif option == 2:
        fig = plt.figure(figsize=(16, 3.5))
        for i, cell in enumerate(sample_cells):
            ax = fig.add_subplot(1, 4, i + 1)
            # ax.text(0.05, 0.1, cell, transform=ax.transAxes,
            #       fontsize=16, fontweight='bold', va='top')

            if i == 0:
                ax.set_ylabel("CC discharge voltage (V)", fontsize=16)

            if i != 0:
                ax.set_yticklabels([])

            ax.set_xlabel("Time (minutes)", fontsize=16)

            cycles = [int(cyc) for cyc in data_dict[cell].keys()]
            cmap = plt.get_cmap("gist_heat", len(cycles))

            for cycle in data_dict[cell].keys():
                # x_axis = np.arange(len(data_dict[cell][cycle][1])) + 1 ax.plot(data_dict[cell][cycle][0],
                # data_dict[cell][cycle][1], c=cmap(int(cycle)), linewidth=1, alpha=0.5)
                ax.plot(
                    data_dict[cell][cycle][0] - min(data_dict[cell][cycle][0]),
                    data_dict[cell][cycle][1],
                    c=cmap(int(cycle)),
                    linewidth=1,
                    alpha=0.5,
                )
            ax.set_xlim(0, 16)

            """
            # Normalizer
            vmin, vmax = 1, len(cycles)
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

            # creating ScalarMappable
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])

            cbar = fig.colorbar(sm, ax=ax, orientation="horizontal",
                                ticks=range(1, len(cycles) + int((vmax - vmin) / 2), int((vmax - vmin) / 2)))
            cbar.set_label('Cycles', fontsize=16)
            """

            # Add text with an arrow pointing to a specific point on the plot
            ax.annotate(
                "Decreasing",
                xy=(10, 2.6),
                xytext=(8, 3.3),
                arrowprops=dict(facecolor="black", linewidth=0.1),
                size=16,
            )

            # ax.text(0.02, 0.1, cell, transform=ax.transAxes,
            #                       fontsize=16, fontweight='bold', va='top')

        plt.savefig(fname=fname, bbox_inches="tight")
