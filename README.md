# signature-project
This repository contains the codes for all the experiments performed in the paper _'Early prediction of Remaining Useful Life for Lithium-ion cells using only signatures of voltage curves at 4 minute sampling rates.'_

To run the codes, create a folder named **data**. Download the following data and put them in this folder:

 - all the batches of data in this link https://data.matr.io/1/ which are the data for the papers [Data driven prediciton of battery cycle life before capacity degradation by K.A. Severson, P.M. Attia, et al](https://www.nature.com/articles/s41560-019-0356-8) and [Attia, P.M., Grover, A., Jin, N. et al. Closed-loop optimization of fast-charging protocols for batteries with machine learning. Nature 578, 397–402 (2020).](https://doi.org/10.1038/s41586-020-1994-5)
 
 - the internal resistance data used to complement batch 8 can be downloaded from https://doi.org/10.7488/ds/2957 which is published in the paper [Strange, C.; Li, S.; Gilchrist, R.; dos Reis, G. Elbows of Internal Resistance Rise Curves in Li-Ion Cells. Energies 2021, 14, 1206.](https://doi.org/10.3390/en14041206)

Create folders named **plots** and **models** to store the generated figures and models respectively.

The **utils** folder contains all the helper functions used in the **Jupyter notebooks**.

_Licence: [CC BY 4.0.](https://creativecommons.org/licenses/by/4.0/legalcode)_
