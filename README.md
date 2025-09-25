# IRRIGATION BUFFERS MASS BALANCES IN HMA

** Table of contents ** 
General 

Requirements

Data layout

Execution order

Reproducibility notes

Tips & troubleshooting

Citation

License


** General **

This file contains all the required code for running the simulation on impact of irrigation on the mass balances in High Mountain Asia


** Requirements **

Python ≥ 3.9

Core scientific stack: numpy, pandas, xarray

Plotting: matplotlib

IO & utilities: pathlib, tqdm

Parallelization: standard library multiprocessing (used by the scripts) 
    Multiprocessign code is written for mac (on windows this might be more straighforward as OGGM multiprocessing is not working directly on mac )

OGGM and dependencies (for glacier model context and data conventions): oggm


** Execution order **

It is build up out of the following sections

1. Climate Data Processing to define the monthly, seasonal and yearly 30-yr average perturbations 
2. Glacier simulations & Plotting

Each of the subsctions has a separate readme file with step by step description of how to run the code

The code is run in an OGGM environemnt


** Data layout **

Data is stored in one working directory. 
In my case called " 04. Modelled perturbation-glacier interactions - R13-15 A+1km2"
The working directory contains the following subfolders which are automaticly created by OGGM: log, per_glacier, pkls, summary
Additionally a directory is created to save master dataset: masters

Figures are saved in a separate folder. 

Please adjust all working directories and paths to your preferences


** Reproducability notes **

Code has been run on a macbook Pro 2023, with M3 chip.
Multiprocessing sections on glacier modelling can take up quite some time. 
Especially volume evolution can take up to a few days if run for all the model and member combinations.
Plotting sections will run very fast. The map plot can take a bit longer (few minutes)

** Tips and troubleshooting ** 

In case of returning errros, troubleshooting guidelines are included in the text.
Apart from that the code should be error free. 

** Citation ** 

Please include citation in line with publication data

** Licence ** 

Not needed


    