# IRRIGATION BUFFERS MASS BALANCES IN HMA

## **Table of contents**

- General  
- System Requirements
- Installation Guide
- Demo: Test Run
- Repository Structure & Data layout  
- Instructions for use
- Reproducibility notes  
- Tips & troubleshooting  
- Citation  
- License
- Compliance Statement

---

## **General**

This repository contains all code and data required to run glacier model simulations assessing the impact of **irrigation-induced climate perturbations on glacier mass balances in High Mountain Asia (HMA)**.

The simulations are performed using the **Open Global Glacier Model (OGGM)** and include:
- Climate data processing to define perturbations
- Glacier mass balance and volume evolution simulations
- Plotting and analysis routines

All code is provided to support **peer review, reproducibility, and reuse**.

---

## ** System Requirements**

### Operating System
- Tested on **macOS 14+ (Apple Silicon, M3 chip)**
- Expected to work on **Linux**
- Windows may be more straightforward for multiprocessing, as OGGM multiprocessing does not work natively on macOS

### Python
- **Python ≥ 3.9**

### Required Software & Libraries
- Core scientific stack: `numpy`, `pandas`, `xarray`  
- Plotting: `matplotlib`  
- I/O & utilities: `pathlib`, `tqdm`  
- Parallelization: standard library `multiprocessing`  
  - Multiprocessing code is written for macOS  
  - On Windows this may be more straightforward, as OGGM multiprocessing does not work natively on macOS  
- OGGM and dependencies (for glacier model context and data conventions): `oggm`

### Non-standard Hardware
- No non-standard hardware required  
- A multi-core CPU is recommended due to computational cost

---

## Installation Guide

This project relies entirely on **OGGM’s official installation procedure**.

Please follow the instructions provided on the OGGM documentation website:

> **OGGM Installation Guide**  
> https://docs.oggm.org/en/stable/installation.html

This includes:
- Creating a suitable Python environment
- Installing OGGM and all required dependencies
- Verifying the installation using OGGM’s test scripts

### Typical installation time
- **10–20 minutes** on a standard desktop or laptop computer

--- 

## Demo: Test Run

To verify that the environment and OGGM installation are functioning correctly, users are encouraged to run a **standard OGGM demo simulation**.

Please follow the official OGGM quick-start example:

> **OGGM Getting Started / Demo**  
> https://docs.oggm.org/en/stable/getting_started.html

This demo:
- Runs a small number of glaciers
- Downloads required test data automatically
- Produces example mass balance and volume outputs

### Expected runtime
- **5–15 minutes** on a standard desktop computer

Successful execution of the OGGM demo indicates that this repository can be run as intended.

---

## **Repository structure & Data layout**

All data are stored in a single working directory.  
For this manuscript, the directory is named:

"4. Modelled perturbation-glacier interactions - R13-15 A+1km2"

### Automatically created OGGM subdirectories
- `log/`
- `per_glacier/`
- `pkls/`
- `summary/`

### Additional directories
- `masters/`  
  Stores master datasets used across simulations
- `figures/`  
  Stores all generated plots and maps

> **Important:**  
> All working directories and file paths should be adjusted to your local system.

---

## Instructions for Use

### Workflow overview
The workflow is structured into the following sections:

1. **Climate data processing**
   - Definition of monthly, seasonal, and annual **30-year average climate perturbations**
2. **Glacier simulations**
   - OGGM-based mass balance and volume evolution simulations
3. **Plotting and analysis**
   - Time series and spatial diagnostics

Each section is contained in a **dedicated subdirectory** with a separate README file providing **step-by-step instructions**.

All scripts are designed to be executed **within an OGGM environment**.

---

## Reproducibility Notes

- Simulations were run on a **MacBook Pro (2023, Apple M3 chip)**.
- Glacier volume evolution simulations are computationally expensive:
  - Full ensemble simulations may take **several days**
- Plotting routines run efficiently:
  - Time series plots: seconds
  - Spatial plots: several minutes

Reproducing all manuscript figures requires executing the full workflow, including climate perturbation generation and ensemble glacier simulations.

---

## Tips & Troubleshooting

- Troubleshooting guidance is provided:
  - In code comments
  - In section-specific README files
- Common issues include:
  - Incorrect file paths
  - Missing OGGM dependencies
  - Multiprocessing limitations on macOS

When the environment is correctly configured, the code should run without modification.

---

## Citation

If you use this code or data, please cite the associated publication:

> *Citation to be added upon publication.*

---

## License

GNU AFFERO GENERAL PUBLIC LICENSE

--- 

### Compliance statement
This repository includes:
- Source code
- A clear installation pathway via OGGM documentation
- Demo instructions
- Usage and reproducibility guidance

It fulfills **Nature’s software and code availability requirements** for peer review and publication.

