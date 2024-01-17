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
If you use this work in your project, please reference:
    @article{IBRAHEEM2023121974,
        title = {Early prediction of Lithium-ion cell degradation trajectories using signatures of voltage curves up to 4-minute sub-sampling rates},
        journal = {Applied Energy},
        volume = {352},
        pages = {121974},
        year = {2023},
        issn = {0306-2619},
        doi = {https://doi.org/10.1016/j.apenergy.2023.121974},
        url = {https://www.sciencedirect.com/science/article/pii/S0306261923013387},
        author = {Rasheed Ibraheem and Yue Wu and Terry Lyons and Gonçalo {dos Reis}},
        keywords = {Capacity degradation, Path signature methodology, Voltage response under constant current at discharge, Lithium-ion cells, Machine learning, Remaining useful life},
        abstract = {Feature-based machine learning models for capacity and internal resistance (IR) curve prediction have been researched extensively in literature due to their high accuracy and generalization power. Most such models work within the high frequency of data availability regime, e.g., voltage response recorded every 1–4 s. Outside premium fee cloud monitoring solutions, data may be recorded once every 3, 5 or 10 min. In this low-data regime, there are little to no models available. This literature gap is addressed here via a novel methodology, underpinned by strong mathematical guarantees, called ‘path signature’. This work presents a feature-based predictive model for capacity fade and IR rise curves from only constant-current (CC) discharge voltage corresponding to the first 100 cycles. Included is a comprehensive feature analysis for the model via a relevance, redundancy, and complementarity feature trade-off mechanism. The ability to predict from subsampled ‘CC voltage at discharge’ data is investigated using different time steps ranging from 4 s to 4 min. It was discovered that voltage measurements taken at the end of every 4 min are enough to generate features for curve prediction with End of Life (EOL) and its corresponding IR values predicted with a mean absolute percentage error (MAPE) of approximately 13.2% and 2.1%, respectively. Our model under higher frequency (4 s) produces an improved accuracy with EOL predicted with an MAPE of 10%. Full implementation code publicly available.}
    }
   

_Licence: [CC BY 4.0.](https://creativecommons.org/licenses/by/4.0/legalcode)_
