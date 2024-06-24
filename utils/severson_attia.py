import numpy as np
import os
import h5py
from datetime import datetime

from .definitions import ROOT_DIR
from .helper import read_data, load_yaml_file, dump_data


def time_monitor(initial_time: datetime | None = None) -> str | datetime:
    """
    This function monitors time from the start of a
    process to the end of the process.
    """
    if not initial_time:
        initial_time = datetime.now()
        return initial_time
    else:
        thour, temp_sec = divmod((datetime.now() - initial_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)

        return "%ih %imin and %ss." % (thour, tmin, round(tsec, 2))


def load_data(filename: str, batch_num: int, num_cycles: int | None = None) -> dict:
    """
    This function loads the downloaded matlab file into a dictionary.

    Args:
    ----
        filename:     string with the path of the data file
        batch_num:    index of this batch
        num_cycles:   number of cycles to be loaded

    Returns a dictionary with data for each cell in the batch.
    """

    # read the matlab file
    f = h5py.File(filename, "r")
    batch = f["batch"]

    # get the number of cells in this batch
    num_cells = batch["summary"].shape[0]

    # initialize a dictionary to store the result
    batch_dict = {}

    summary_features = [
        "IR",
        "QCharge",
        "QDischarge",
        "Tavg",
        "Tmin",
        "Tmax",
        "chargetime",
        "cycle",
    ]
    cycle_features = [
        "I",
        "Qc",
        "Qd",
        "Qdlin",
        "T",
        "Tdlin",
        "V",
        "discharge_dQdV",
        "t",
    ]

    for i in range(num_cells):

        # decide how many cycles will be loaded
        if num_cycles is None:
            loaded_cycles = f[batch["cycles"][i, 0]]["I"].shape[0]
        else:
            loaded_cycles = min(num_cycles, f[batch["cycles"][i, 0]]["I"].shape[0])

        if i % 10 == 0:
            print(f"* {i} cells loaded ({loaded_cycles} cycles)")

        # initialise a dictionary for this cell
        cell_dict = {
            "cycle_life": (
                f[batch["cycle_life"][i, 0]][()]
                if batch_num != 3
                else f[batch["cycle_life"][i, 0]][()] + 1
            ),
            "charge_policy": f[batch["policy_readable"][i, 0]][()]
            .tobytes()[::2]
            .decode(),
            "summary": {},
        }

        for feature in summary_features:
            cell_dict["summary"][feature] = np.hstack(
                f[batch["summary"][i, 0]][feature][0, :].tolist()
            )

        # for the cycle data
        cell_dict["cycle_dict"] = {}

        for j in range(loaded_cycles):
            cell_dict["cycle_dict"][str(j + 1)] = {}
            for feature in cycle_features:
                cell_dict["cycle_dict"][str(j + 1)][feature] = np.hstack(
                    (f[f[batch["cycles"][i, 0]][feature][j, 0]][()])
                )

        # converge into the batch dictionary
        batch_dict[f"b{batch_num}c{i}"] = cell_dict

    return batch_dict


def load_all_batches_to_dict(num_cycles: int | None = None):
    """
    This function load and save downloaded matlab files as pickle files.
    Note that the battery data (downloaded from https://data.matr.io/1/) must be
    put in "data" folder. After calling this function, extracted files
    in .pkl format will be stored in "data" folder.

    Args:
    ----
         num_cycles:  number of cycles to load
    """

    # paths for data file with each batch of cells
    mat_filenames = mat_filenames = {
        f"batch{b}": os.path.join(f"{ROOT_DIR}", "data/severson_attia", mat_file)
        for b, mat_file in zip(
            range(1, 9),
            [
                "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
                "2017-06-30_batchdata_updated_struct_errorcorrect.mat",
                "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
                "2018-08-28_batchdata_updated_struct_errorcorrect.mat",
                "2018-09-02_batchdata_updated_struct_errorcorrect.mat",
                "2018-09-06_batchdata_updated_struct_errorcorrect.mat",
                "2018-09-10_batchdata_updated_struct_errorcorrect.mat",
                "2019-01-24_batchdata_updated_struct_errorcorrect.mat",
            ],
        )
    }

    start = time_monitor()
    print("Loading batch 1 data...")
    batch1 = load_data(mat_filenames["batch1"], 1, num_cycles=num_cycles)
    print(time_monitor(start))

    start = time_monitor()
    print("\nLoading batch 2 data...")
    batch2 = load_data(mat_filenames["batch2"], 2, num_cycles=num_cycles)
    print(time_monitor(start))

    start = time_monitor()
    print("\nLoading batch 3 data...")
    batch3 = load_data(mat_filenames["batch3"], 3, num_cycles=num_cycles)
    print(time_monitor(start))

    start = time_monitor()
    print("\nLoading batch 4 data...")
    batch4 = load_data(mat_filenames["batch4"], 4, num_cycles=num_cycles)
    print(time_monitor(start))

    start = time_monitor()
    print("\nLoading batch 5 data...")
    batch5 = load_data(mat_filenames["batch5"], 5, num_cycles=num_cycles)
    print(time_monitor(start))

    start = time_monitor()
    print("\nLoading batch 6 data...")
    batch6 = load_data(mat_filenames["batch6"], 6, num_cycles=num_cycles)
    print(time_monitor(start))

    start = time_monitor()
    print("\nLoading batch 7 data...")
    batch7 = load_data(mat_filenames["batch7"], 7, num_cycles=num_cycles)
    print(time_monitor(start))

    start = time_monitor()
    print("\nLoading batch 8 data...")
    batch8 = load_data(mat_filenames["batch8"], 8, num_cycles=num_cycles)
    print(time_monitor(start))

    print(f"* {len(batch1.keys())} cells loaded in batch 1")
    print(f"* {len(batch2.keys())} cells loaded in batch 2")
    print(f"* {len(batch3.keys())} cells loaded in batch 3")
    print(f"* {len(batch4.keys())} cells loaded in batch 4")
    print(f"* {len(batch5.keys())} cells loaded in batch 5")
    print(f"* {len(batch6.keys())} cells loaded in batch 6")
    print(f"* {len(batch7.keys())} cells loaded in batch 7")
    print(f"* {len(batch8.keys())} cells loaded in batch 8")

    # there are four cells from batch1 that carried into batch2, we'll remove the data from batch2 and put it with
    # the correct cell from batch1
    b2_keys = ["b2c7", "b2c8", "b2c9", "b2c15", "b2c16"]
    b1_keys = ["b1c0", "b1c1", "b1c2", "b1c3", "b1c4"]
    add_len = [662, 981, 1060, 208, 482]

    # append data to batch 1
    for i, bk in enumerate(b1_keys):
        batch1[bk]["cycle_life"] = batch1[bk]["cycle_life"] + add_len[i]

        for j in batch1[bk]["summary"].keys():
            if j == "cycle":
                batch1[bk]["summary"][j] = np.hstack(
                    (
                        batch1[bk]["summary"][j],
                        batch2[b2_keys[i]]["summary"][j]
                        + len(batch1[bk]["summary"][j]),
                    )
                )
            else:
                batch1[bk]["summary"][j] = np.hstack(
                    (batch1[bk]["summary"][j], batch2[b2_keys[i]]["summary"][j])
                )

        last_cycle = len(batch1[bk]["cycle_dict"].keys())

        # useful when all cycles loaded
        if num_cycles is None:
            for j, jk in enumerate(batch2[b2_keys[i]]["cycle_dict"].keys()):
                batch1[bk]["cycle_dict"][str(last_cycle + j)] = batch2[b2_keys[i]][
                    "cycle_dict"
                ][jk]
    """
    The authors exclude cells that:
        * do not reach 80% capacity (batch 1)
        * were carried into batch2 but belonged to batch 1 (batch 2)
        * noisy channels (batch 3)
    """

    exc_cells = {
        "batch1": ["b1c8", "b1c10", "b1c12", "b1c13", "b1c22"],
        "batch2": ["b2c7", "b2c8", "b2c9", "b2c15", "b2c16"],
        "batch3": ["b3c37", "b3c2", "b3c23", "b3c32", "b3c38", "b3c39"],
    }

    for c in exc_cells["batch1"]:
        del batch1[c]

    for c in exc_cells["batch2"]:
        del batch2[c]

    for c in exc_cells["batch3"]:
        del batch3[c]

    # exclude the first cycle from all cells because this data was not part of the first batch of cells
    batches = [batch1, batch2, batch3, batch4, batch5, batch6, batch7, batch8]
    for batch in batches:
        for cell in batch.keys():
            del batch[cell]["cycle_dict"]["1"]

    for batch in batches:
        for cell in batch.keys():
            assert "1" not in batch[cell]["cycle_dict"].keys()

    for batch in batches:
        for cell in batch.keys():
            for feat in batch[cell]["summary"].keys():
                batch[cell]["summary"][feat] = np.delete(
                    batch[cell]["summary"][feat], 0
                )

    # combine all batches in one dictionary
    data_dict = {
        **batch1,
        **batch2,
        **batch3,
        **batch4,
        **batch5,
        **batch6,
        **batch7,
        **batch8,
    }

    return data_dict


def dump_severson_attia_structured_data() -> None:

    all_batches: dict = load_all_batches_to_dict(num_cycles=None)

    # Get cells without ir
    cells_without_ir = [
        cell
        for cell in all_batches.keys()
        if cell[:2] in ("b4", "b5", "b6", "b7", "b8")
    ]

    # Read the ir data
    ir_data = read_data(
        fname="predicted_ir.pkl", path=f"{ROOT_DIR}/data/severson_attia"
    )

    for cell in cells_without_ir:
        all_batches[cell]["summary"]["IR"] = ir_data[cell][0]

    # Remove batteries with more than 1200 cycle life
    #  and those belong to batches 4 to 7
    odd_cells = []
    for cell in all_batches.keys():
        discharge_capacity = all_batches[cell]["summary"]["QDischarge"]
        eol_bool = (
            discharge_capacity >= 0.88
        )  # eol is defined as cycle number at 80% of nominal capacity (which is 1.1 Ah)
        discharge_capacity = discharge_capacity[eol_bool]

        if np.any(
            [len(discharge_capacity) > 1200, cell[:2] in ("b4", "b5", "b6", "b7")]
        ):
            odd_cells.append(cell)

    all_batches = {k: all_batches[k] for k in all_batches.keys() if k not in odd_cells}

    # Dump the new data
    dump_data(
        data=all_batches,
        path=f"{ROOT_DIR}/data",
        fname="structured_severson_attia_data.pkl",
    )

    return None
