#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 16 14:38:41 2025

@author: magaliponds
"""

# %% Cell 0: Load data packages
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 15:36:52 2024

@author: magaliponds

This code runs perturbs the climate data and adds this to the baseline climate in OGGM and runs the climate & MB model with this data"""


# -*- coding: utf-8 -*-import oggm


import multiprocessing
from multiprocessing import Pool, set_start_method, get_context
from multiprocessing import Process
from multiprocessing import Pool, set_start_method, get_context
from multiprocessing import Process
import concurrent.futures
from matplotlib.lines import Line2D
import oggm
from oggm import utils, cfg, workflow, tasks, DEFAULT_BASE_URL, graphics, global_tasks
from oggm.core import massbalance, flowline
from oggm.utils import floatyear_to_date, hydrodate_to_calendardate
from oggm.sandbox import distribute_2d
from oggm.sandbox.edu import run_constant_climate_with_bias
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as clrs
import xarray as xr
import os
import seaborn as sns
import salem
from matplotlib.ticker import FuncFormatter
import pandas as pd
import numpy as np
from matplotlib import animation
from IPython.display import HTML, display
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.optimize import curve_fit
from tqdm import tqdm
import pickle
import textwrap
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import sys
from concurrent.futures import ProcessPoolExecutor
import time
from joblib import Parallel, delayed
#%%
# function_directory = "/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/src/03. Glacier simulations"
# sys.path.append(function_directory)
# from OGGM_data_processing import process_perturbation_data


#%% Cell 0: Initialize OGGM with the preferred model parameter set up

folder_path = '/Users/magaliponds/Documents/00. Programming'
wd_path = f'{folder_path}/04. Modelled perturbation-glacier interactions - R13-15 A+1km2/'
os.makedirs(wd_path, exist_ok=True)
cfg.initialize(logging_level='WARNING')
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
pkls_subset = os.path.join(wd_path, "pkls_subset_success")


cfg.PARAMS['baseline_climate'] = "GSWP3_W5E5"

cfg.PARAMS['store_model_geometry'] = True

cfg.PARAMS['use_multiprocessing'] = True
cfg.PARAMS['core'] = 9 # 🔧 set number of cores

# %% Cell 0: Set base parameters

colors_models = {
    "W5E5": ["#000000"],  # "#000000"],  # Black
    # Darker to lighter shades of purple
    "E3SM": ["#785EF0", "#8F7BF1", "#A6A8F2"],
    # Darker to lighter shades of pink
    "CESM2": ["#DC267F", "#E58A9E", "#F0A2B6", "#F7BCC4"],
    # Darker to lighter shades of orange
    "CNRM": ["#FE6100", "#FE7D33", "#FE9A66", "#FEB799", "#FECDB5", "#FEF1E1"],
    "IPSL-CM6": ["#FFB000"]  # Dark purple to lighter shades
}

members = [1, 3, 4, 6, 4, 1]
members_averages = [1, 2, 3, 5, 3]
models = ["IPSL-CM6", "E3SM", "CESM2", "CNRM", "NorESM", "W5E5"]
models_shortlist = ["IPSL-CM6", "E3SM", "CESM2", "CNRM", "NorESM"]
timeframe = "monthly"

y0_clim = 1985
ye_clim = 2014
y0_cf = 1901
ye_cf = 1985

#%% DOWNLOAD ONCE: Cell 1: Use the RGI to filter glaciers with Area>1

# # For example, RGI version 6, regions 13–15
# rgi_ids = []
# for reg in [13, 14, 15]:
#     rgi_fp = utils.get_rgi_region_file(f"{reg:02d}", version='6')
#     rgi_df = gpd.read_file(rgi_fp)
#     rgi_df = rgi_df[rgi_df['Area'] > 1]
#     rgi_ids.append(rgi_df)

# rgi_all = gpd.GeoDataFrame(pd.concat(rgi_ids))

# rgi_all.to_csv(os.path.join(wd_path,"masters","rgi_all_list.csv"))

#%% Cell 1b: Open list of all rgi_ids to initialize
rgi_all=pd.read_csv(os.path.join(wd_path,"masters","rgi_all_list.csv"), index_col=0)['RGIId']


    
#%% PROCESS ONCE: Cell 2: Initialize GDIRs (Empty) Without Redownloading


# def process_chunk(rgi_ids_chunk):
#     cfg.PARAMS['use_multiprocessing'] = False  # Disable OGGM parallelism here, to avoid children processes in multi processing
#     gdirs = workflow.init_glacier_directories(rgi_ids_chunk, 
#                                               reset=False, 
#                                               prepro_base_url=DEFAULT_BASE_URL,
#                                               from_prepro_level=4,
#                                               prepro_border=160)
#     for gdir in gdirs:
#         gdir_path = os.path.join(pkls, f'{gdir.rgi_id}.pkl')
#         with open(gdir_path, 'wb') as f:
#             pickle.dump(gdir, f)
#     print(f"Processed {len(gdirs)} glaciers")

# def split_into_chunks(lst, chunk_size):
#     for i in range(0, len(lst), chunk_size):
#         yield lst[i:i + chunk_size]

# def main():
#     chunks = list(split_into_chunks(rgi_all, 1000))
#     with multiprocessing.Pool(processes=cfg.PARAMS['core']) as pool:
#         pool.map(process_chunk, chunks)

# if __name__ == '__main__':
#     multiprocessing.set_start_method('spawn', force=True)
#     main()  

#%% Cell 2b: Open all pkls

# def load_single_pkl(filepath):
#     with open(filepath, 'rb') as f:
#         return pickle.load(f)

# def main():
#     # Collect all .pkl file paths
#     pkl_files = [os.path.join(pkls, f) for f in os.listdir(pkls) if f.endswith('.pkl')]
    
#     # Use all available cores (or limit it manually)
#     with ProcessPoolExecutor(max_workers=9) as executor:
#         gdirs = list(executor.map(load_single_pkl, pkl_files))

#     print(f"Loaded {len(gdirs)} glacier directories.")

#     # Optional: create dictionary
#     gdirs_dict = {gdir.rgi_id: gdir for gdir in gdirs}

#     return gdirs, gdirs_dict

# if __name__ == '__main__':
#     gdirs, gdirs_dict = main()
    
#%% Cell 3: Filter out all glacier ids with errors

# def main():
#     # Compile stats
#     stats = utils.compile_glacier_statistics(gdirs, path=os.path.join(
#         sum_dir, "prepro_stats.csv"), inversion_only=False, apply_func=None)

#     # Load statistics
#     statistics = pd.read_csv(os.path.join(sum_dir, "prepro_stats.csv"))
#     failed = statistics[statistics.run_dynamic_spinup_success == False]
#     failed_rgi_ids = set(failed['rgi_id'])

#     # Filter successful glaciers
#     gdirs_filtered = [gdir for gdir in gdirs if gdir.rgi_id not in failed_rgi_ids]
#     gdirs_dict = {gdir.rgi_id: gdir for gdir in gdirs_filtered}

#     # Save successful glaciers
#     pkls_subset = os.path.join(wd_path, "pkls_subset_success")
#     os.makedirs(pkls_subset, exist_ok=True)

#     for gdir in gdirs_filtered:
#         gdir_path = os.path.join(pkls_subset, f'{gdir.rgi_id}.pkl')
#         with open(gdir_path, 'wb') as f:
#             pickle.dump(gdir, f)

#     print(f"Saved {len(gdirs_filtered)} successful glaciers to {pkls_subset}")
#     return gdirs_filtered, gdirs_dict

# if __name__ == '__main__':
#     gdirs_filtered, gdirs_dict = main()

#%% Cell 3b: Open filtered subset



def load_single_pkl(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def main():
    # Collect all .pkl file paths
    pkl_files = [os.path.join(pkls_subset, f) for f in os.listdir(pkls_subset) if f.endswith('.pkl')]
    
    # Use all available cores (or limit it manually)
    with ProcessPoolExecutor(max_workers=9) as executor:
        gdirs = list(executor.map(load_single_pkl, pkl_files))

    print(f"Loaded {len(gdirs)} glacier directories.")

    # Optional: create dictionary
    gdirs_dict = {gdir.rgi_id: gdir for gdir in gdirs}
   

    return gdirs, gdirs_dict


if __name__ == '__main__':
    gdirs, gdirs_dict = main()
    print(f"✅ Loaded all gdirs")
    
#%% Cell 4: Process perturbations

# members = [1, 3, 4, 6, 4]
# models = ["IPSL-CM6", "E3SM", "CESM2", "CNRM", "NorESM"]
# timeframe = "monthly"


# # if you get a long error log saying that "columns" can not be renamed it is often related to multiprocessing

# def run_task():
#     start = time.time()
#     for m, model in enumerate(models):
#             for member in range(members[m]):
#                 if model =="IPSL-CM6" and member==0:
#                     # Provide the path to the perturbation dataset
#                     # if error with lon.min or ds['time.year'] check if lon>0 in creating input dataframe
#                     i_folder_ptb = f"/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/03. Output files/01. Climate data/07. OGGM Perturbation input files/{timeframe}/{model}/{member}"
#                     ds_path = f"{i_folder_ptb}/{model}.00{member}.{y0_clim}_{ye_clim}.{timeframe}.perturbation.input.%.degC.nc"
    
#                     # Provide the sample ID to provide the processed pertrubations with the correct output suffix
#                     sample_id = f"{model}.00{member}"
    
#                     workflow.execute_entity_task(process_perturbation_data, gdirs,
#                                                  ds_path=ds_path,
#                                                  # y0=1985, y1=2014,
#                                                  y0=None, y1=None,
#                                                  output_filesuffix=f'_perturbation_{sample_id}')
    
#                     opath_perturbations = os.path.join(
#                         sum_dir, f'climate_historical_perturbation_{sample_id}.nc')
    
#                     utils.compile_climate_input(gdirs, path=opath_perturbations, filename='climate_historical',
#                                                 input_filesuffix=f'_perturbation_{sample_id}',
#                                                 use_compression=True)
#     end = time.time()
#     print(f"✅ Finished in {(end - start)/60:.2f} minutes")
        
# if __name__ == '__main__':
#     multiprocessing.set_start_method('spawn', force=True)  # macOS-specific
#     run_task()
#%%   Cell 5: Perturb climate data  
# members = [1, 3, 4, 6, 4]
# models = ["IPSL-CM6", "E3SM", "CESM2", "CNRM", "NorESM"]
# timeframe = "monthly"


# # if you get a long error log saying that "columns" can not be renamed it is often related to multiprocessing
# def process_gdir(gdir, sample_id):
#     # Load original dataset
#     with xr.open_dataset(gdir.get_filepath('climate_historical')) as ds:
#         ds = ds.load()

#     clim_ptb = ds.copy().sel(time=slice('1985-01-01', '2014-12-31'))

#     # Load perturbation
#     with xr.open_dataset(gdir.get_filepath('climate_historical', filesuffix=f'_perturbation_{sample_id}')) as ds_ptb:
#         ds_ptb = ds_ptb.load()

#     # Apply perturbation
#     clim_ptb['temp'] = clim_ptb.temp - ds_ptb.temp
#     clim_ptb['prcp'] = clim_ptb.prcp - clim_ptb.prcp * ds_ptb.prcp

#     # Save result
#     clim_ptb.to_netcdf(gdir.get_filepath('gcm_data', filesuffix=f'_perturbed_{sample_id}'))

# def run_task():
#     start = time.time()
#     for m, model in enumerate(models):
#         for member in range(members[m]):
#             if member >= 1 or m == 0:
#                 sample_id = f"{model}.00{member}"
#                 print(f"▶️ Starting {sample_id}")
#                 Parallel(n_jobs=9)(delayed(process_gdir)(gdir, sample_id) for gdir in gdirs)

#                 # Do this once per sample, not per glacier
#                 df_stats = utils.compile_glacier_statistics(gdirs, filesuffix=f"_perturbed_{sample_id}")
#                 print(f"✅ Finished {sample_id}")
#     end = time.time()
#     print(f"⏱️ Total duration: {(end - start)/60:.2f} minutes")

# if __name__ == '__main__':
#     run_task()
    
#%%   Cell 6: Run MB model

models = ["IPSL-CM6"]#"W5E5", "IPSL-CM6", "E3SM", "CESM2", "CNRM", "NorESM"]
members = [1]#[1, 1, 3, 4, 6, 4]

# Per-glacier processing logic
def process_single_glacier(args):
    gdir, model, member, sample_id = args
    try:
        fls = gdir.read_pickle('model_flowlines')
        years = np.arange(1985, 2015)

        if model == "W5E5":
            years_ext = np.arange(2000, 2020)
            mbmod = massbalance.MonthlyTIModel(gdir, filename='climate_historical')
            mb_ts = mbmod.get_specific_mb(fls=fls, year=years)
            mb_ts_ext = mbmod.get_specific_mb(fls=fls, year=years_ext)
            return {
                "sample_id": sample_id,
                "rgi_id": gdir.rgi_id,
                "mb_ts": list(zip(years, mb_ts)),
                "mb_ts_ext": list(zip(years_ext, mb_ts_ext)),
                "mean_mb": np.mean(mb_ts),
                "mean_mb_ext": np.mean(mb_ts_ext),
                "error": None
            }
        else:
            mbmod = massbalance.MonthlyTIModel(
                gdir, filename='gcm_data', input_filesuffix=f'_perturbed_{sample_id}')
            mb_ts = mbmod.get_specific_mb(fls=fls, year=years)
            return {
                "sample_id": sample_id,
                "rgi_id": gdir.rgi_id,
                "mb_ts": list(zip(years, mb_ts)),
                "mean_mb": np.mean(mb_ts),
                "error": None
            }

    except Exception as e:
        return {
            "sample_id": sample_id,
            "rgi_id": gdir.rgi_id,
            "model": model,
            "member": member,
            "error": str(e)
        }

# Task list generation
def prepare_parallel_tasks(gdirs, models, members):
    task_list = []
    for m, model in enumerate(models):
        for member in range(members[m]):
            if member >= 1 or m == 0:
                sample_id = f"{model}.00{member}"
                for gdir in gdirs:
                    task_list.append((gdir, model, member, sample_id))
    return task_list

# Parallel execution
def run_parallel_mb_model(gdirs):
    start = time.time()
    task_args = prepare_parallel_tasks(gdirs, models, members)

    results = []
    with ProcessPoolExecutor(max_workers=9) as executor:
        for res in executor.map(process_single_glacier, task_args):
            results.append(res)

    # Organize results by sample_id
    grouped = {}
    for res in results:
        sid = res['sample_id']
        grouped.setdefault(sid, {
            "mean": [], "ts": [], "mean_ext": [], "ts_ext": [], "errors": []
        })
        if res['error']:
            grouped[sid]["errors"].append((res['rgi_id'], res.get('model', ''), res.get('member', ''), res['error']))
        else:
            grouped[sid]["mean"].append((res['rgi_id'], res['mean_mb']))
            grouped[sid]["ts"].extend((res['rgi_id'], y, mb) for y, mb in res['mb_ts'])
            if 'mb_ts_ext' in res:
                grouped[sid]["mean_ext"].append((res['rgi_id'], res['mean_mb_ext']))
                grouped[sid]["ts_ext"].extend((res['rgi_id'], y, mb) for y, mb in res['mb_ts_ext'])

    # Save outputs
    for sid, data in grouped.items():
        pd.DataFrame(data["mean"], columns=['rgi_id', 'B']).to_csv(
            os.path.join(sum_dir, f'specific_massbalance_mean_{sid}.csv'), index=False)
        pd.DataFrame(data["ts"], columns=['rgi_id', 'Year', 'Mass_Balance']).to_csv(
            os.path.join(sum_dir, f'specific_massbalance_timeseries_{sid}.csv'), index=False)
        if data["mean_ext"]:
            pd.DataFrame(data["mean_ext"], columns=['rgi_id', 'B']).to_csv(
                os.path.join(sum_dir, f'specific_massbalance_mean_extended_{sid}.csv'), index=False)
            pd.DataFrame(data["ts_ext"], columns=['rgi_id', 'Year', 'Mass_Balance']).to_csv(
                os.path.join(sum_dir, f'specific_massbalance_timeseries_extended_{sid}.csv'), index=False)
        if data["errors"]:
            pd.DataFrame(data["errors"], columns=['rgi_id', 'Model', 'Member', 'Error']).to_csv(
                os.path.join(log_dir, f'Error_Log_NoIrr_{sid}.csv'), index=False)

    end = time.time()
    print(f"✅ Completed in {(end - start)/60:.2f} minutes")

# Entry point
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    cfg.initialize()
    cfg.PARAMS['use_multiprocessing'] = False  # prevent nested multiprocessing

    run_parallel_mb_model(gdirs)
    
    
    
#%% Cell 7: Run from climate data

# # Constants
# y0_clim = 1985
# ye_clim = 2014
# models = ["IPSL-CM6", "E3SM", "CESM2", "CNRM", "NorESM"]
# members = [1, 3, 4, 6, 4]

# def load_gdirs(pkl_dir):
#     return [
#         pickle.load(open(os.path.join(pkl_dir, f), 'rb'))
#         for f in os.listdir(pkl_dir) if f.endswith('.pkl')
#     ]

# def run_climate_simulation(model,member,gdirs):
#     # model, member, gdirs = args

#     cfg.initialize()
#     cfg.PARAMS['continue_on_error'] = True
#     cfg.PARAMS['use_multiprocessing'] = False
#     cfg.PARAMS['border'] = 240

#     if model != "IPSL-CM6" and member == 0:
#         return

#     sample_id = f"{model}.00{member}"
#     print(f"▶️ Starting: {sample_id}")

#     out_id = f'_perturbed_{sample_id}'

#     workflow.execute_entity_task(tasks.init_present_time_glacier, gdirs)

#     # Run climate model (OGGM will use all available cores)
#     workflow.execute_entity_task(tasks.run_from_climate_data, gdirs,
#                                  ys=y0_clim, ye=ye_clim,
#                                  max_ys=None, fixed_geometry_spinup_yr=None,
#                                  store_monthly_step=False, store_model_geometry=True,
#                                  store_fl_diagnostics=True,
#                                  climate_filename='gcm_data',
#                                  climate_input_filesuffix=f'_perturbed_{sample_id}',
#                                  output_filesuffix=out_id,
#                                  zero_initial_glacier=False, bias=0,
#                                  init_model_filesuffix='_spinup_historical',
#                                  init_model_yr=y0_clim)

#     opath = os.path.join(sum_dir, f'climate_run_output_perturbed_{sample_id}.nc')
#     utils.compile_run_output(gdirs, input_filesuffix=out_id, path=opath)

#     log_path = os.path.join(log_dir, f'stats_perturbed_{sample_id}_climate_run.nc')
#     utils.compile_glacier_statistics(gdirs, path=log_path)

#     print(f"✅ Finished: {sample_id}")

# def run_baseline(gdirs):
#     sample_id = 'W5E5.000'

#     workflow.execute_entity_task(tasks.run_from_climate_data, gdirs,
#                                  ys=y0_clim, ye=ye_clim,
#                                  output_filesuffix=f'_baseline_{sample_id}',
#                                  init_model_filesuffix='_spinup_historical',
#                                  init_model_yr=y0_clim,
#                                  store_fl_diagnostics=True)

#     opath = os.path.join(sum_dir, f'climate_run_output_baseline_{sample_id}.nc')
#     utils.compile_run_output(gdirs, input_filesuffix=f'_baseline_{sample_id}', path=opath)

#     log_path = os.path.join(log_dir, f'stats_perturbed_{sample_id}_climate_run.csv')
#     utils.compile_glacier_statistics(gdirs, path=log_path)

# def main():
#     cfg.initialize()
#     cfg.PARAMS['continue_on_error'] = True
#     cfg.PARAMS['use_multiprocessing'] = False
#     cfg.PARAMS['border'] = 240

#     # Load gdirs
#     gdirs = load_gdirs(pkls_subset)
#     subset_gdirs = gdirs 

#     # Prepare model-member task list
#     # task_args = [(models[m], member, subset_gdirs) for m in range(len(models)) for member in range(members[m])]
#     # Sequentially run each model-member pair
#     for m, model in enumerate(models):
#         for member in range(members[m]):
#             run_climate_simulation(model, member, gdirs)


#     # Run parallel model simulations
#     # with ProcessPoolExecutor(max_workers=9) as executor:
#     #     executor.map(run_climate_simulation, task_args)

#     # Run baseline
#     run_baseline(subset_gdirs)

# if __name__ == '__main__':
#     multiprocessing.set_start_method('spawn', force=True)
#     main()





    