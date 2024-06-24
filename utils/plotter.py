import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns
from typing import Any, Callable

from .definitions import ROOT_DIR
from .data_wrangler import (
    cycle_life,
    ccv_signature_features,
    get_ccv_profile,
    create_knee_elbow_data,
)
from .analyser import PredictedFullCurve


def plot_ccv_evolution(
    data_dict: dict[str, dict],
    sample_cells: list[str],
) -> None:
    """
    Plot the evolution of constant-current voltage for sample cells.
    """

    data_dict = ccv_signature_features(
        data_dict=data_dict, num_cycles=-1, return_ccv=True
    )  # get constant-current voltage at discharge for all cycles (except the last)

    fig = plt.figure(figsize=(16, 3.5))
    for i, cell in enumerate(sample_cells):
        ax = fig.add_subplot(1, 4, i + 1)
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
        ax.annotate(
            "Decreasing",
            xy=(10, 2.6),
            xytext=(8, 3.3),
            arrowprops=dict(facecolor="black", linewidth=0.1),
            size=16,
        )
        ax.text(
            0.02,
            0.1,
            cell,
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            va="top",
        )

    plt.savefig(
        fname=f"{ROOT_DIR}/plots/sig-new-ccv-evol-with-arrows.pdf", bbox_inches="tight"
    )

    return None


def plot_target_distribution(data_dict: dict[str, dict]) -> None:
    _, ax = plt.subplots(1, 3, figsize=(16, 4))
    cmap = plt.get_cmap("gist_heat", len(data_dict))

    for i, cell in enumerate(data_dict.keys()):

        capacity = data_dict[cell]["summary"]["QDischarge"]
        ir = data_dict[cell]["summary"]["IR"]

        ax[1].plot(capacity, "o", linewidth=1, markersize=1, c=cmap(i))
        ax[1].set_xlabel("Cycles", fontsize=16)
        ax[1].set_ylabel("Capacity (Ah)", fontsize=16)
        ax[1].set_ylim([0.8, 1.1])

        ax[2].plot(ir, "o", linewidth=1, markersize=1, c=cmap(i))
        ax[2].set_xlabel("Cycles", fontsize=16)
        ax[2].set_ylabel(r"IR ($\Omega$)", fontsize=16)
        ax[2].set_ylim([0.010, 0.025])

    ax[0].hist(cycle_life(data_dict=data_dict).values, color="brown", ec="black")
    ax[0].set_xlabel("EOL of cells", fontsize=16)
    ax[0].set_ylabel("Frequency", fontsize=16)

    plt.tight_layout()
    plt.savefig(
        fname=f"{ROOT_DIR}/plots/sig-eolhist-cap-ir-explore.pdf", bbox_inches="tight"
    )

    return None


def plot_signature_geometry(
    data_dict: dict[str, dict], sample_cell: str, sample_cycle: str
) -> None:
    cct, ccv = get_ccv_profile(
        data_dict=data_dict,
        cell=sample_cell,
        cycle=sample_cycle,
    )

    _, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].plot(cct, ccv, color="brown")
    ax[0].hlines(
        y=min(ccv),
        xmin=min(cct),
        xmax=max(cct),
        linestyles="dashed",
        color="brown",
        linewidth=2.0,
    )
    ax[0].vlines(
        x=min(cct),
        ymin=min(ccv),
        ymax=max(ccv),
        linestyles="dashed",
        color="brown",
        linewidth=2.0,
    )
    ax[0].fill_between(cct, y1=min(ccv), y2=ccv, alpha=0.2, color="brown")
    ax[0].text(x=6.5, y=2.6, s=r"$-S^{1,2}$", fontsize=16)
    ax[0].set_ylabel(r"CC Voltage at discharge, $V(t)$", fontsize=16)
    ax[0].set_xlabel(r"time, $t$", fontsize=16)
    ax[0].text(x=6.0, y=2.05, s=r"$V=V_{final}$", fontsize=16)
    ax[0].text(x=0.02, y=2.6, s=r"$t=t_{initial}$", fontsize=16, rotation=90)
    ax[0].text(x=6, y=3.15, s=r"$V=V(t)$", fontsize=16)

    ax[1].plot(cct, ccv, color="brown")
    ax[1].hlines(
        y=max(ccv),
        xmin=min(cct),
        xmax=max(cct),
        linestyles="dashed",
        color="brown",
        linewidth=2.0,
    )
    ax[1].vlines(
        x=max(cct),
        ymin=min(ccv),
        ymax=max(ccv),
        linestyles="dashed",
        color="brown",
        linewidth=2.0,
    )
    ax[1].fill_between(cct, y1=max(ccv), y2=ccv, alpha=0.2, color="brown")
    ax[1].text(x=7.5, y=3.2, s=r"$-S^{2,1}$", fontsize=16)
    # ax[1].set_ylabel(r'CC Voltage at discharge, $V(t)$', fontsize=14)
    ax[1].set_xlabel(r"time, $t$", fontsize=16)
    ax[1].text(x=6.0, y=3.4, s=r"$V=V_{initial}$", fontsize=16)
    ax[1].text(x=13.5, y=3.0, s=r"$t=t_{final}$", fontsize=16, rotation=90)
    ax[1].text(x=6, y=3.0, s=r"$V=V(t)$", fontsize=16)

    plt.savefig(
        fname=f"{ROOT_DIR}/plots/sig_level2_geometry_inter.pdf", bbox_inches="tight"
    )

    return None


def plot_signature_geometry_change(
    data_dict: dict[str, dict], sample_cell: str
) -> None:

    _, ax = plt.subplots(1, 2, figsize=(16, 6))
    for cycle in ("2", "50", "100", "300", "400"):

        cct, ccv = get_ccv_profile(data_dict=data_dict, cell=sample_cell, cycle=cycle)

        ax[0].plot(cct, ccv, alpha=0.3)
        ax[0].fill_between(cct, y1=min(ccv), y2=ccv, alpha=0.2, label=f"Cycle {cycle}")

        ax[1].plot(cct, ccv, alpha=0.3)
        ax[1].fill_between(cct, y1=max(ccv), y2=ccv, alpha=0.2, label=f"Cycle {cycle}")

    ax[0].set_ylabel(r"CC Voltage at discharge, $V(t)$", fontsize=16)
    ax[0].set_xlabel(r"time, $t$", fontsize=16)

    ax[1].set_xlabel(r"time, $t$", fontsize=16)

    ax[0].set_title(r"$-S^{1,2}$", fontsize=16)
    ax[1].set_title(r"$-S^{2,1}$", fontsize=16)

    ax[0].legend()
    ax[1].legend(loc="lower left")

    plt.savefig(
        fname=f"{ROOT_DIR}/plots/sig_level2_geometry_inter2.pdf",
        bbox_inches="tight",
    )

    return None


def plot_feature_target_correlation(data_dict: dict[str, dict]) -> None:

    features = ccv_signature_features(
        data_dict=data_dict,
        step_size=1,
        num_cycles=100,
        sig_level=2,
        multi_cycle=False,
        return_ccv=False,
        return_sig=False,
    )
    targets = create_knee_elbow_data(data_dict=data_dict)

    merged_df = features.join(targets)
    merged_df = merged_df.drop(
        ["k-o", "k-p", "Qatk-o", "Qatk-p", "e-o", "e-p", "IRate-o", "IRate-p"], axis=1
    )

    target_cols = ["EOL", "IRatEOL"]
    feature_cols = merged_df.columns[:-2]

    corr_matrix = merged_df.corr()

    fig = plt.figure(figsize=(7, 11))
    for i, col in enumerate(target_cols):

        ax = fig.add_subplot(1, 2, i + 1)
        ax.text(
            0.05,
            0.99,
            col,
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            va="top",
        )

        if i == 1:
            ax.set_yticklabels([])

        corr_for_col = corr_matrix.loc[feature_cols, col]
        ax.barh(feature_cols, corr_for_col, color="brown", ec="black", alpha=0.78)
        ax.set_xlim([-1, 1])
        ax.set_xlabel(r"Correlation coefficent ($\rho$)", fontsize=12)
        ax.axvline(x=0.5, color="black", linestyle="--", alpha=0.5)
        ax.axvline(x=-0.5, color="black", linestyle="--", alpha=0.5)
        ax.axvline(x=-0.0, color="black", linestyle="-", alpha=0.5)
        # ax.tick_params(axis='y', labelsize=16)

    plt.tight_layout()
    plt.savefig(
        fname=f"{ROOT_DIR}/plots/sig-table1&2-features-corr-eol-ir.pdf",
        bbox_inches="tight",
    )

    return None


def plot_cycle_signature_correlation(
    data_dict: dict[str, dict], sample_cell: str
) -> None:

    signature_labels = [
        r"$S^1$",
        r"$S^2$",
        r"$S^{1,1}$",
        r"$S^{1,2}$",
        r"$S^{2,1}$",
        r"$S^{2,2}$",
    ]
    signatures = ccv_signature_features(
        data_dict={sample_cell: data_dict[sample_cell]},
        step_size=1,
        num_cycles=100,
        sig_level=2,
        return_sig=True,
    )
    cycle_numbers = np.arange(1, 101)

    fig = plt.figure(figsize=(16, 8))
    for i, label in enumerate(signature_labels):
        ax = fig.add_subplot(3, 2, i + 1)

        # Get the Pearson's correlation for the current signature component
        corr_for_sig = pearsonr(cycle_numbers, signatures[sample_cell][:, i])[0]

        ax.scatter(
            cycle_numbers, signatures[sample_cell][:, i], color="brown", s=20, alpha=0.5
        )
        ax.text(
            0.02,
            0.97 if i % 2 != 0 else 0.2,
            r"$\rho = {}$".format(np.round(corr_for_sig, 2)),
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            va="top",
        )
        ax.set_ylabel(label, fontsize=16)

        if i in [4, 5]:
            ax.set_xlabel("Cycles", fontsize=16)

    plt.tight_layout()
    plt.savefig(
        fname=f"{ROOT_DIR}/plots/sig-scatter-plot-sigcomponent-cyclenumber.pdf",
        bbox_inches="tight",
    )

    return None


def plot_cycle_number_effect_history(
    list_of_cycles: list[int] | np.ndarray,
    history: dict[str, dict],
    model_type: str,
    evaluation_type: str,
) -> None:

    if evaluation_type == "crossval":
        ylabels = (
            ["Cross-validated errors (cycles)", "Cross-validated errors (cycles)"]
            if model_type == "cycle_model"
            else ["Cross-validated errors (Ah)", r"Cross-validated errors ($\Omega$)"]
        )

    elif evaluation_type == "test":
        ylabels = (
            ["Average test errors (cycles)", "Average test errors (cycles)"]
            if model_type == "cycle_model"
            else ["Average test errors (Ah)", r"Average test errors ($\Omega$)"]
        )

    _, ax = plt.subplots(1, 2, figsize=(16, 4))

    for i, (split, data) in enumerate(history.items()):
        ax[i].plot(
            list_of_cycles,
            data[f"{evaluation_type}_mae"],
            "D--",
            label="MAE",
            color="blue",
            markersize=5,
        )
        ax[i].fill_between(
            list_of_cycles,
            data[f"{evaluation_type}_mae_ci"][:, 0],
            data[f"{evaluation_type}_mae_ci"][:, 1],
            color="blue",
            alpha=0.1,
            label="MAE: 90% CI",
        )

        ax[i].set_xlabel("Input cycle number", fontsize=16)

        ax[i].set_ylabel(ylabels[i], fontsize=16)
        ax[i].set_title(split, fontsize=18)

        ax[i].plot(
            list_of_cycles,
            data[f"{evaluation_type}_rmse"],
            "s-",
            color="crimson",
            label="RMSE",
            markersize=5,
        )
        ax[i].fill_between(
            list_of_cycles,
            data[f"{evaluation_type}_rmse_ci"][:, 0],
            data[f"{evaluation_type}_rmse_ci"][:, 1],
            color="crimson",
            alpha=0.1,
            label="RMSE: 90% CI",
        )

    handles, labels = ax[0].get_legend_handles_labels()
    ax[0].legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        fontsize=16,
        bbox_to_anchor=(1.0, -0.15),
    )

    if model_type == "cycle_model":
        plt.savefig(
            fname=f"{ROOT_DIR}/plots/sig_level2_{evaluation_type}_n_effect_cycles_tabs12.pdf",
            bbox_inches="tight",
        )
    else:
        plt.savefig(
            fname=f"{ROOT_DIR}/plots/sig_level2_{evaluation_type}_n_effect_capir_tabs12.pdf",
            bbox_inches="tight",
        )

    return None


def plot_predicted_full_curve(
    predicted_curve_history: dict[str, dict[str, PredictedFullCurve]], curve_name: str
) -> None:

    if curve_name == "QDischarge":
        x, y = (0.02, 0.2)
        figure_save_name = "capacity-fade"
        ylabel = "Capacity (Ah)"
        ylim = [0.85, 1.1]
    elif curve_name == "IR":
        x, y = (0.05, 0.95)
        figure_save_name = "ir-rise"
        ylabel = r"Internal Resistance ($\Omega$)"
        ylim = [0.014, 0.022]

    else:
        raise ValueError(
            "curve_name must be either 'QDischarge' or 'IR', "
            f"but {curve_name} is privided"
        )

    fig = plt.figure(figsize=(18, 7))

    for i, (cell, history) in enumerate(predicted_curve_history.items()):
        ax = fig.add_subplot(3, 3, i + 1)
        ax.text(
            x,
            y,
            cell,
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            va="top",
        )

        ax.plot(
            history[curve_name].actual_cycle,
            history[curve_name].actual_curve,
            "k--",
            label="Actual curve",
            linewidth=1.0,
        )
        ax.plot(
            history[curve_name].predicted_cycle,
            history[curve_name].predicted_curve,
            color="brown",
            label="Predicted curve",
            linewidth=2.0,
        )
        ax.fill(
            np.append(
                history[curve_name].predicted_cycle_lb,
                history[curve_name].predicted_cycle_ub[::-1],
            ),
            np.append(
                history[curve_name].predicted_curve_lb,
                history[curve_name].predicted_curve_ub[::-1],
            ),
            color="brown",
            label=r"90% CI",
            alpha=0.13,
        )

        if i == 7:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(
                handles,
                labels,
                loc="upper center",
                ncol=3,
                fontsize=16,
                bbox_to_anchor=(0.5, -0.4),
            )

        if i % 3 != 0:
            ax.set_yticklabels([])

        if i in [6, 7, 8]:
            ax.set_xlabel("Cycle", fontsize=16)

        ax.set_ylim(ylim)

    fig.text(
        0.08,
        0.5,
        ylabel,
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=16,
    )
    plt.savefig(
        fname=f"{ROOT_DIR}/plots/sig_level2_{figure_save_name}-curve.pdf",
        bbox_inches="tight",
    )

    return None


def axis_to_fig(axis: Any) -> Callable[[tuple], Any]:
    """
    Converts axis to fig object.

    Args:
    ----
         axis (object): axis object

    Returns:
    -------
            transformed axis oject.
    """

    fig = axis.figure

    def transform(coord: tuple | list):
        return fig.transFigure.inverted().transform(axis.transAxes.transform(coord))

    return transform


def add_sub_axes(axis: Any, rect: tuple | list) -> Any:
    """
    Adds sub-axis to existing axis object.

    Args:
    ----
         axis (object):        axis object
         rect (list or tuple): list or tuple specifying axis dimension

    Returns:
    -------
           fig object with added axis.
    """
    fig = axis.figure
    left, bottom, width, height = rect
    trans = axis_to_fig(axis)
    figleft, figbottom = trans((left, bottom))
    figwidth, figheight = trans([width, height]) - trans([0, 0])

    return fig.add_axes([figleft, figbottom, figwidth, figheight])


def plot_parity_history(parity_history: dict[str, np.ndarray]) -> None:
    fig = plt.figure(figsize=(14, 20))

    for i, (target, history) in enumerate(parity_history.items()):

        ax = fig.add_subplot(5, 2, i + 1)
        ax.text(
            0.05,
            0.95,
            target,
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            va="top",
        )

        ax.scatter(
            history["y_train"],
            history["y_train_pred"],
            s=50,
            color="royalblue",
            alpha=0.5,
            label="Train",
        )
        ax.scatter(
            history["y_test"],
            history["y_test_pred"],
            s=50,
            color="brown",
            alpha=0.5,
            label="Test",
            marker="D",
        )
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]

        # now plot both limits against each other
        ax.plot(lims, lims, "k--", alpha=0.75, zorder=100)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        if i % 2 == 0:
            ax.set_ylabel("Predicted values", fontsize=16)

        if i in [8, 9]:
            ax.set_xlabel("Measured values", fontsize=16)

        # embed histogram of residuals
        res_train = history["y_train"] - history["y_train_pred"]
        res_test = history["y_test"] - history["y_test_pred"]
        res = np.concatenate((res_train, res_test), casting="unsafe", dtype=float)

        subaxis = add_sub_axes(ax, [0.62, 0.17, 0.35, 0.2])
        subaxis.hist(res, bins=20, color="black", alpha=0.75, ec="black")
        subaxis.set_xlim(res.min(), -res.min())
        subaxis.set_xlabel("Residuals", fontsize=10)
        subaxis.set_ylabel("Frequency", fontsize=10)

        if i == 8:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(
                handles,
                labels,
                loc="upper center",
                ncol=3,
                fontsize=16,
                bbox_to_anchor=(1.0, -0.2),
            )

    plt.savefig(
        fname=f"{ROOT_DIR}/plots/sig_level2_parity-plot.pdf", bbox_inches="tight"
    )

    return None


def plot_feature_importance_analysis_history(
    analysis_df: pd.DataFrame, figure_save_name: str
) -> None:

    fig = plt.figure(figsize=(18, 3))

    df_index = np.array(analysis_df.index)

    for i, col in enumerate(analysis_df.columns):

        ax = fig.add_subplot(1, 5, i + 1)
        ax.text(
            0.6,
            0.95,
            col,
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            va="top",
        )

        col_importance = analysis_df[col].values
        sorted_index = np.argsort(col_importance)

        col_importance = col_importance[sorted_index]
        temp_index = df_index[sorted_index]

        ax.bar(
            temp_index[::-1][:10],
            col_importance[::-1][:10],
            color="brown",
            ec="black",
            alpha=0.78,
        )
        ax.tick_params(axis="x", rotation=90, labelsize=14)
        ax.tick_params(axis="y", labelsize=14)

        if i != 0:
            ax.set_yticklabels([])

        if i == 0:
            ax.set_ylabel("Feature importance", size=16)

    plt.savefig(
        fname=f"{ROOT_DIR}/plots/sig-feature-importance-{figure_save_name}-bar.pdf",
        bbox_inches="tight",
    )

    return None


def plot_subsampling_time_effect_history(
    history: dict[str, list[str] | np.ndarray]
) -> None:

    _, ax = plt.subplots(1, 2, figsize=(16, 4.5))

    ax[0].plot(
        history["time_steps"],
        history["crossval_mae"],
        "D--",
        label="EOL: MAE",
        color="blue",
    )
    ax[0].fill_between(
        history["time_steps"],
        history["crossval_mae_ci"][:, 0],
        history["crossval_mae_ci"][:, 1],
        color="blue",
        alpha=0.15,
        label="MAE: 90% CI",
    )

    ax[1].plot(
        history["time_steps"],
        history["crossval_rmse"],
        "s-",
        label="EOL: RMSE",
        color="crimson",
    )
    ax[1].fill_between(
        history["time_steps"],
        history["crossval_rmse_ci"][:, 0],
        history["crossval_rmse_ci"][:, 1],
        color="crimson",
        alpha=0.15,
        label="RMSE: 90% CI",
    )

    ax[0].legend(loc="lower right")
    ax[1].legend(loc="lower right")

    ax[0].set_xlabel("Sub-sampling time steps (mins)", fontsize=16)
    ax[0].set_ylabel("Cross-validation errors (cycles)", fontsize=16)
    ax[1].set_xlabel("Sub-sampling time steps (mins)", fontsize=16)

    plt.savefig(
        fname=f"{ROOT_DIR}/plots/sig_level2_subsample_tabs12.pdf", bbox_inches="tight"
    )

    return None


def plot_rrct_robustness_heatmap(
    similarity_scores: np.ndarray, time_step_list: list[float]
) -> None:

    _, ax = plt.subplots(figsize=(5, 4.5))
    axis_labels = np.round(time_step_list, 2)
    ax.set_xticklabels(axis_labels)
    ax.set_yticklabels(axis_labels)
    sns.heatmap(
        similarity_scores,
        vmin=0,
        vmax=1,
        xticklabels=axis_labels,
        yticklabels=axis_labels,
        linewidth=0.5,
        linecolor="black",
        ax=ax,
        cbar_kws={"label": "Similarity scores"},
        annot=True,
    )

    ax.set_xlabel("Sub-sampling time steps (mins)", fontsize=16)
    ax.set_ylabel("Sub-sampling time steps (mins)", fontsize=16)
    ax.figure.axes[-1].yaxis.label.set_size(16)
    plt.yticks(rotation=0)

    plt.savefig(
        fname=f"{ROOT_DIR}/plots/sig-similarity-scores.pdf", bbox_inches="tight"
    )

    return None


def plot_high_freq_model_robustness_history(
    cycle_model_test_scores: np.ndarray,
    cap_ir_model_test_scores: np.ndarray,
    cycle_target_list: list[str],
    cap_ir_target_list: list[str],
    time_step_list: list[float],
) -> None:

    list_of_markers = ["s-", "o-", "<-", ">-", "*-"]
    _, ax = plt.subplots(1, 3, figsize=(15, 4))

    for i in range(5):
        ax[0].plot(
            time_step_list,
            cycle_model_test_scores[:, i],
            list_of_markers[i],
            label=cycle_target_list[i],
        )
        ax[0].set_xlabel("Sub-sampling time steps (mins)", fontsize=16)
        ax[0].set_ylabel("MAE (Cycles)", fontsize=16)
    ax[0].legend()

    for i in range(2):
        ax[1].plot(
            time_step_list,
            cap_ir_model_test_scores[:, i],
            list_of_markers[i],
            label=cap_ir_target_list[i],
        )
        ax[1].set_xlabel("Sub-sampling time steps (mins)", fontsize=16)
        ax[1].set_ylabel("MAE (Ah)", fontsize=16)
    ax[1].legend()

    for i in range(2, 5):
        ax[2].plot(
            time_step_list,
            cap_ir_model_test_scores[:, i],
            list_of_markers[i],
            label=cap_ir_target_list[i],
        )
        ax[2].set_xlabel("Sub-sampling time steps (mins)", fontsize=16)
        ax[2].set_ylabel(r"MAE ($\Omega$)", fontsize=16)
    ax[2].legend()

    plt.tight_layout()
    plt.savefig(
        fname=f"{ROOT_DIR}/plots/sig-high-frequency-model-robustness.pdf",
        bbox_inches="tight",
    )

    return None


def plot_rrct_driven_modelling_history(
    metric_tracker: pd.DataFrame, model_type: str
) -> None:

    percetages = [str(i) for i in metric_tracker.index]

    fig = plt.figure(figsize=(10, 5))

    for i, item in enumerate(metric_tracker.columns):
        ax = fig.add_subplot(2, 3, i + 1)

        if i in [0, 2]:
            ax.set_title(
                (
                    item.split("_")[0] + " (cycles)"
                    if model_type == "cycle_model"
                    else item.split("_")[0] + r" ($\Omega$)"
                ),
                fontsize=16,
            )

        if i == 1:
            ax.set_title(item.split("_")[0] + r" ($\%$)", fontsize=16)

        ax.bar(
            metric_tracker.index,
            metric_tracker[item].values,
            width=5.0,
            color="brown",
            ec="black",
            alpha=0.78,
        )
        ax.set_xticks(metric_tracker.index, percetages)
        ax.tick_params(axis="x", rotation=90, labelsize=14)
        ax.tick_params(axis="y", labelsize=14)

        if i not in [3, 4, 5]:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Feature percentage (%)", fontsize=14)

        if i == 0:
            ax.set_ylabel("Train errors", size=16)

        if i == 3:
            ax.set_ylabel("Test errors", size=16)

    plt.tight_layout()
    plt.savefig(
        fname=f"{ROOT_DIR}/plots/sig_{model_type}_rrct_driven_modelling_history.pdf",
        bbox_inches="tight",
    )

    return None


def plot_top_10p_rrct_selected_features(
    selected_features: list[str], figure_tag: str
) -> None:

    # create a list of rank;
    # selected features are ranked from most important (1)
    # to least important (len(selected_features))
    rank_list = [i for i in range(1, len(selected_features) + 1)]
    _, ax = plt.subplots(figsize=(5, 5))

    ax.text(
        0.6,
        0.95,
        figure_tag,
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
    )

    ax.bar(selected_features, rank_list[::-1], color="brown", ec="black", alpha=0.78)

    for j, p in enumerate(ax.patches):
        ax.annotate(
            rank_list[j],
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="left",
            va="center",
            xytext=(0, 10),
            textcoords="offset points",
            size=14,
        )

    ax.set_xlabel("Top 10% selected features", fontsize=16)
    ax.tick_params(axis="x", rotation=90, labelsize=14)
    ax.set_ylabel("Rankings")
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.savefig(
        fname=f"{ROOT_DIR}/plots/sig-rank-10p-4min-{figure_tag}.pdf",
        bbox_inches="tight",
    )

    return None
