#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 15:53:56 2024

@author: magaliponds
"""

from multiprocessing import Pool, set_start_method, get_context


# from multiprocessing import Pool
from oggm import cfg, utils, workflow
from oggm.workflow import execute_entity_task
from oggm.tasks import run_random_climate, init_present_time_glacier
import os
import pickle
from joblib import Parallel, delayed

#
# %% Cell 1: Initialize OGGM with the preferred model parameter set up
folder_path = '/Users/magaliponds/Documents/00. Programming'
wd_path = f'{folder_path}/03. Modelled perturbation-glacier interactions - R13-15 A+5km2/'
os.makedirs(wd_path, exist_ok=True)
cfg.initialize()
cfg.PATHS['working_dir'] = utils.mkdir(wd_path, reset=False)

# make a sum dir
sum_dir = os.path.join(wd_path, 'summary')
os.makedirs(sum_dir, exist_ok=True)

# make a logging directory
log_dir = os.path.join(wd_path, "log")
os.makedirs(log_dir, exist_ok=True)

# Make a pkl directory
pkls = os.path.join(wd_path, "pkls")
os.makedirs(pkls, exist_ok=True)

cfg.initialize(logging_level='WARNING')
cfg.PARAMS['baseline_climate'] = "GSWP3-W5E5"


cfg.PARAMS['store_model_geometry'] = True


# %% Cell 3c: Load gdirs_3r_a5 from pkl (fastest way to get started)

wd_path_pkls = f'{wd_path}/pkls_subset/'

gdirs_3r_a5 = []
for filename in os.listdir(wd_path_pkls):
    if filename.endswith('.pkl'):
        # f'{gdir.rgi_id}.pkl')
        file_path = os.path.join(wd_path_pkls, filename)
        with open(file_path, 'rb') as f:
            gdir = pickle.load(f)
            gdirs_3r_a5.append(gdir)

# %%        OGGM Configuration
cfg.PARAMS['border'] = 250
cfg.PARAMS['continue_on_error'] = True
cfg.PARAMS['use_multiprocessing'] = True
cfg.PARAMS['mp_processes'] = 4

# Constants
y0_comitted = 2015
halfsize = 14.5
members = [1, 3, 4, 6, 4]
models = ["IPSL-CM6", "E3SM", "CESM2", "CNRM", "NorESM"]

# Subset of glaciers
subset_gdirs = gdirs_3r_a5[:1]
for gdir in subset_gdirs:
    print(gdir.rgi_id)


def process_model_member(model, member, subset_gdirs):
    """Worker function for processing a single model and member."""
    sample_id = f"{model}.00{member}"
    print(f"Processing: {sample_id}")

    # Initialize glaciers
    execute_entity_task(init_present_time_glacier, subset_gdirs)

    # File paths
    out_id = f'_perturbed_{sample_id}_committed'
    out_id_climate_run = f'_perturbed_{sample_id}'
    opath = os.path.join(
        sum_dir, f'climate_run_output_perturbed_{sample_id}_comitted_test.nc')

    # Run climate task
    execute_entity_task(
        run_random_climate,
        subset_gdirs,
        nyears=1000,
        ys=y0_comitted - 1,
        halfsize=halfsize,
        y0=y0_comitted - halfsize - 1,
        ye=None,
        bias=0,
        seed=2,
        precipitation_factor=None,
        store_monthly_step=False,
        store_model_geometry=None,
        store_fl_diagnostics=True,
        climate_filename='gcm_data',
        climate_input_filesuffix=f'_perturbed_{sample_id}',
        output_filesuffix=out_id,
        init_model_filesuffix=out_id_climate_run,
        init_model_yr=y0_comitted - 1,
    )

    # Compile output
    ds_ptb = utils.compile_run_output(
        subset_gdirs, path=opath, input_filesuffix=out_id)
    print(f"Completed: {sample_id}")


# Pool-based parallel processing
if __name__ == "__main__":
    ctx = get_context("spawn")
    tasks = [(model, member) for model in ["IPSL", "CESM2"]
             for member in range(3)]
    # for task in tasks:
    #     process_model_member(task)
    # print(tasks)
    # with Pool(cfg.PARAMS['mp_processes']) as pool:
    #     pool.map(process_model_member, tasks)
    results = Parallel(n_jobs=4)(delayed(process_model_member)(
        model, member, None) for model, member in tasks)
    print(results)

# Baseline run for committed scenario
out_id = f'_baseline_W5E5.000_committed'
out_id_climate_run = '_baseline_W5E5.000'

try:
    execute_entity_task(
        run_random_climate,
        subset_gdirs,
        nyears=1000,
        ys=y0_comitted - 1,
        halfsize=halfsize,
        y0=y0_comitted - halfsize - 1,
        ye=None,
        bias=0,
        seed=2,
        precipitation_factor=None,
        store_monthly_step=False,
        store_model_geometry=None,
        store_fl_diagnostics=True,
        climate_filename='climate_historical',
        output_filesuffix=out_id,
        init_model_filesuffix=out_id_climate_run,
        init_model_yr=y0_comitted - 1,
    )
except Exception as e:
    print(f"Error during baseline committed scenario: {e}")

# Compile baseline outputs
opath_base = os.path.join(
    sum_dir, 'climate_run_output_baseline_W5E5.000_comitted.nc')
ds_base = utils.compile_run_output(
    subset_gdirs, input_filesuffix=out_id, path=opath_base)
