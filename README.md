# signature-project
This repository contains the codes for all the experiments performed in the paper [Early prediction of Remaining Useful Life for Lithium-ion cells using only signatures of voltage curves at 4 minute sampling rates.](https://www.sciencedirect.com/science/article/pii/S0306261923013387?via%3Dihub#fig2)

## Folder analysis

   - **config**: configuration files
   - **experiments**: jupyter notebook files for all experiments 
   - **utils**: custom modules used in the project

## Set up for running locally
1. Clone the repository by running
    ```
    git clone https://github.com/Rasheed19/signature-project.git
    ```
1. Navigate to the root folder, i.e., `signature-project` and create a python virtual environment by running
    ```
    python3 -m venv .venv
    ``` 
1. Activate the virtual environment by running
    ```
    source .venv/bin/activate
    ```
1. Prepare all modules by running
    ```
    pip install -e .
    ```
1. Create a folder named **data** in the root directory **signature-project**. Download the following data and put them in this folder:
    - all the batches of data in this link https://data.matr.io/1/ which are the data for the papers [Data driven prediciton of battery cycle life before capacity degradation by K.A. Severson, P.M. Attia, et al](https://www.nature.com/articles/s41560-019-0356-8) and [Attia, P.M., Grover, A., Jin, N. et al. Closed-loop optimization of fast-charging protocols for batteries with machine learning. Nature 578, 397–402 (2020).](https://doi.org/10.1038/s41586-020-1994-5)
    - the internal resistance data used to complement batch 8 can be downloaded from https://doi.org/10.7488/ds/2957 which is published in the paper [Strange, C.; Li, S.; Gilchrist, R.; dos Reis, G. Elbows of Internal Resistance Rise Curves in Li-Ion Cells. Energies 2021, 14, 1206.](https://doi.org/10.3390/en14041206)

1. Create folders named **plots** and **models** in the root directory **signature-project** to store the generated figures and models respectively.

1. Start running jupyter notebooks in the **experiments** folder.

1. When you are done experimenting, deactivate the virtual environment by running
    ```
    deactivate
    ```

_Licence: [CC BY 4.0.](https://creativecommons.org/licenses/by/4.0/legalcode)_
