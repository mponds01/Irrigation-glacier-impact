# IRRIGATION BUFFERS MASS BALANCES IN HMA

## **Table of contents**

- General  
- Requirements  
- Data layout  
- Execution order  
- Reproducibility notes  
- Tips & troubleshooting  
- Citation  
- License  

---

## **General**

This repository contains all required code to run simulations assessing the impact of irrigation on glacier mass balances in High Mountain Asia (HMA).

---

## **Requirements**

- Python ≥ 3.9  
- Core scientific stack: `numpy`, `pandas`, `xarray`  
- Plotting: `matplotlib`  
- I/O & utilities: `pathlib`, `tqdm`  
- Parallelization: standard library `multiprocessing`  
  - Multiprocessing code is written for macOS  
  - On Windows this may be more straightforward, as OGGM multiprocessing does not work natively on macOS  
- OGGM and dependencies (for glacier model context and data conventions): `oggm`

---

## **Execution order**

The workflow is structured into the following sections:

1. Climate data processing to define monthly, seasonal, and yearly 30-year average perturbations  
2. Glacier simulations and plotting  

Each subsection contains a separate README file with step-by-step instructions on how to run the code.

All code is executed within an OGGM environment.

---

## **Data layout**

All data are stored in a single working directory.  
In my case, this directory is named:

4. Modelled perturbation-glacier interactions - R13-15 A+1km2


The working directory contains the following subfolders, which are automatically created by OGGM:

- `log`  
- `per_glacier`  
- `pkls`  
- `summary`  

Additionally, a directory named `masters` is created to store master datasets.

Figures are saved in a separate folder.

Please adjust all working directories and file paths to your own system and preferences.

---

## **Reproducibility notes**

The code was run on a MacBook Pro (2023) with an M3 chip.

Multiprocessing sections related to glacier modelling can be computationally expensive.  
In particular, volume evolution simulations can take several days when run for all model and ensemble member combinations.

Plotting routines run quickly, although map plotting may take a few minutes.

---

## **Tips & troubleshooting**

In case of errors, troubleshooting guidelines are included directly in the code comments and accompanying text.  
Aside from this, the code should run without issues when the environment is set up correctly.

---

## **Citation**

Please include a citation consistent with the associated publication.

