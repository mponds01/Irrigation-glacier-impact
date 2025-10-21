# %% Cell 0: Load data packages
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 15:36:52 2024

@author: magaliponds

This code runs perturbs the climate data for historic simulations (1985-2014)
 and adds this per variable (P only and T only) to the baseline climate in OGGM 
 and runs the climate & MB model with this data"""


# -*- coding: utf-8 -*-import oggm
import sys

function_directory = "/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/src/03. Glacier simulations"
sys.path.append(function_directory)
#%%
from OGGM_data_processing import process_perturbation_data
import multiprocessing
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

# %%

# %% Cell 0: Set base parameters
print("Cell 0: Set base parameters")
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


fig_path = '/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/04. Figures/02. OGGM simulations/01. Modelled output/3r_a1/'

# %% Cell 1: Initialize OGGM with the preferred model parameter set up
print("Cell 1: Initialize OGGM with the preferred model parameter set up")
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


cfg.PARAMS['baseline_climate'] = "GSWP3-W5E5"


cfg.PARAMS['store_model_geometry'] = True
cfg.PARAMS['use_multiprocessing'] = True
cfg.PARAMS['core'] = 9 # 🔧 set number of cores


# %% Cell 1: Load successful gdirs

wd_path_pkls = f'{wd_path}/pkls_subset_success/'

gdirs_3r_a1 = []
for filename in os.listdir(wd_path_pkls):
    if filename.endswith('.pkl'):
        # f'{gdir.rgi_id}.pkl')
        file_path = os.path.join(wd_path_pkls, filename)
        with open(file_path, 'rb') as f:
            gdir = pickle.load(f)
            gdirs_3r_a1.append(gdir)

# # print(gdirs)


# %% Cell 3: Perturb the climate historical with the processed Irr-perturbations, output is gcm file

print("Cell 3: Perturb the climate historical with the processed Irr-perturbations, output is gcm file")
def main():
    count = 0
    members = [4, 1, 3, 4, 6]
    models = [ "NorESM", "IPSL-CM6", "E3SM", "CESM2", "CNRM"]
    # gdir = gdirs_3r_a1[0]
    
    for m, model in enumerate(models):  # models_shortlist
        for member in range(members[m]):
            if model=="IPSL-CM6" or member>=1:
                for var in ["P","T"]:
                    for gdir in gdirs_3r_a1:
                        count += 1
                        print(round((count*100)/len(gdirs_3r_a1), 2))
                        # tasks.init_present_time_glacier(gdir)
                    # if m == 0:
                        with xr.open_dataset(gdir.get_filepath('climate_historical')) as ds:
                            ds = ds.load()
                        sample_id = f"{model}.00{member}"
                            # print(sample_id)
                            # make a copy of the historical climate
                        clim_ptb = ds.copy().sel(time=slice('1985-01-01', '2014-12-31'))
            
                        # open the perturbation dataset and add the perturbations
                        # with xr.open_dataset(gdir.get_filepath('climate_historical', filesuffix='_perturbation_{}_counterfactual'.format(sample_id))) as ds_ptb:
                        with xr.open_dataset(gdir.get_filepath('climate_historical', filesuffix=f'_perturbation_{sample_id}')) as ds_ptb:
                            ds_ptb = ds_ptb.load()
                        if var=="T":
                            clim_ptb['temp'] = clim_ptb.temp - ds_ptb.temp
                        if var=="P":
                            clim_ptb['prcp'] = clim_ptb.prcp - clim_ptb.prcp * ds_ptb.prcp
                        #
                        clim_ptb.to_netcdf(gdir.get_filepath(
                            # 'gcm_data', filesuffix='_perturbed_{}_counterfactual'.format(sample_id)))
                            'gcm_data', filesuffix=f'_perturbed_{sample_id}_{var}_only'))
                        # df_stats = utils.compile_glacier_statistics(
                        #     gdirs_3r_a1, filesuffix=f"_perturbed_{sample_id}")

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()     
    

# %% Cell 4: Run Mass Balance model for the glaciers (V = A*B)

print("Cell 4: Run Mass Balance model for the glaciers (V = A*B)")

def main():
    
    members = [4, 1, 3, 4, 6]
    models = [ "NorESM", "IPSL-CM6", "E3SM", "CESM2", "CNRM"]
    
    
    for m, model in enumerate(models):
        count = 0
        for member in range(members[m]):
            if model=="IPSL-CM6" or member>=1:
                for var in ["P","T"]:
        
                    # create a sample id for all the model x member combinations
                    sample_id = f"{model}.00{member}"
                    print(sample_id)
            
                    # create lists to store the model output
                    mb_ts_mean = []
                    mb_ts_all = []
                    mb_ts_all_ext = []
                    mb_ts_mean_ext = []
                    mb_ts_all_ext = []
                    error_ids = []
            
                    # load the gdirs_3r_a1
                    for (g, gdir) in enumerate(gdirs_3r_a1):
                        count += 1
            
                        try:
                            # provide the model flowlines and years for the mbmod
                            fls = gdir.read_pickle('model_flowlines')
                            years = np.arange(1985, 2015)
            
                            if model == "W5E5":
                                # extend range for w5e5, to see match w. geodetic mass balance
                                years_ext = np.arange(2000, 2020)
                                # Get the calibrated mass-balance model - the default is to use OGGM's "MonthlyTIModel" and compute specific mb
                                mbmod = massbalance.MonthlyTIModel(
                                    gdir, filename='climate_historical')
                                mb_ts = mbmod.get_specific_mb(fls=fls, year=years)
                                # also run the model for the extended years and save
                                mb_ts_ext = mbmod.get_specific_mb(fls=fls, year=years_ext)
                                for year, mb in zip(years, mb_ts):
                                    mb_ts_all.append((gdir.rgi_id, years, mb_ts))
                                    mb_ts_all_ext.append(
                                        (gdir.rgi_id, years_ext, mb_ts_ext))
                            else:
                                # print(gdir.rgi_id)
                                mbmod = massbalance.MonthlyTIModel(
                                    # gdir, filename='gcm_data', input_filesuffix='_perturbed_{}_counterfactual'.format(sample_id))
                                    gdir, filename='gcm_data', input_filesuffix=f'_perturbed_{sample_id}_{var}_only')
                                mb_ts = mbmod.get_specific_mb(fls=fls, year=years)
                            # Append all time series data to mb_ts_all
                            for year, mb in zip(years, mb_ts):
                                mb_ts_all.append((gdir.rgi_id, year, mb_ts))
            
                        # include an exception so the model will continue running on error and provide the error
                        except Exception as e:
                            # Handle the error and continue
                            print(
                                f"Error processing {gdir.rgi_id} with model {model} and member {member}: {e}")
                            # found error: RGI60-13.36875 no flowlines --> 542 to 541 glaciers in selected gdirs_3r_a1
                            error_ids.append((gdir.rgi_id, model, member, e))
                            continue
                        mean_mb = np.mean(mb_ts)
                        mb_ts_mean.append((gdir.rgi_id, mean_mb))
            
                        if model == "W5E5":
                            mean_mb_ext = np.mean(mb_ts_ext)
                            mb_ts_mean_ext.append((gdir.rgi_id, mean_mb_ext))
                        count += 1
            
                    # create a dataframe with the mass balance data of all gdirs_3r_a1
                    mb_df_mean = pd.DataFrame(mb_ts_mean, columns=['rgi_id', 'B'])
                    mb_df_mean.to_csv(os.path.join(
                        # sum_dir, f'specific_massbalance_mean_{sample_id}_counterfactual.csv'), index=False)
                    sum_dir, f'specific_massbalance_mean_{sample_id}_{var}_only.csv'), index=False)
                    mb_ts_df = pd.DataFrame(
                        mb_ts_all, columns=['rgi_id', 'Year', 'Mass_Balance'])
                    mb_ts_df.to_csv(os.path.join(
                        # sum_dir, f'specific_massbalance_timeseries_{sample_id}_counterfactual.csv'), index=False)
                    sum_dir, f'specific_massbalance_timeseries_{sample_id}_{var}_only.csv'), index=False)
            
                    # if model == "W5E5":  # only for W5E5 also create a dataframe for the extended timeseries
                    #     mb_df_mean_ext = pd.DataFrame(
                    #         mb_ts_mean_ext, columns=['rgi_id', 'B'])
                    #     mb_df_mean_ext.to_csv(os.path.join(
                    #         sum_dir, f'specific_massbalance_mean_extended_{sample_id}.csv'), index=False)
                    #     mb_ts_df_ext = pd.DataFrame(mb_ts_all_ext, columns=[
                    #                                 'rgi_id', 'Year', 'Mass_Balance'])
                    #     mb_ts_df_ext.to_csv(os.path.join(
                    #         sum_dir, f'specific_massbalance_timeseries_extended_{sample_id}.csv'), index=False)
            
                    # Optionally save the list of error cases to a CSV for later review
                    if error_ids:
                        error_df = pd.DataFrame(
                            error_ids, columns=['rgi_id', 'Model', 'Member', 'error'])
                        error_df.to_csv(os.path.join(
                            # log_dir, f'Error_Log_counterfactual_{sample_id}.csv'), index=False)
                            log_dir, f'Error_Log_NoIrr_{sample_id}_{var}_only.csv'), index=False)

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()     

# %% Cell 5: Run the climate model baseline and perturbations - Save pkl after running is done , as running takes quite a while

print("Cell 5: Run the climate model baseline and perturbations - Save pkl after running is done , as running takes quite a while")
def main():
    cfg.PARAMS['continue_on_error'] = True
    cfg.PARAMS['use_multiprocessing'] = True
    cfg.PARAMS['border'] = 240
    
    # gdir = gdirs_3r_a1[0]
    y0_clim = 1985
    ye_clim = 2014
    
    subset_gdirs = gdirs_3r_a1
    members = [1, 3, 4, 6, 4]
    models = ["IPSL-CM6", "E3SM", "CESM2", "CNRM", "NorESM"]
    for m, model in enumerate(models):
        for member in range(members[m]):
            if model=="IPSL-CM6" or member>=1:
                for var in ["P","T"]:
                    sample_id = f"{model}.00{member}"
                    print(sample_id)
                    workflow.execute_entity_task(
                        tasks.init_present_time_glacier, gdirs_3r_a1)
                    # out_id = f'_perturbed_{sample_id}_counterfactual'
                    out_id = f'_perturbed_{sample_id}'
            
                    workflow.execute_entity_task(
                        tasks.init_present_time_glacier, gdirs_3r_a1)
            
                    workflow.execute_entity_task(tasks.run_from_climate_data, subset_gdirs,
                                                 ys=y0_clim, ye=ye_clim,  # min_ys=None,
                                                 max_ys=None, fixed_geometry_spinup_yr=None,
                                                 store_monthly_step=False, store_model_geometry=True,
                                                 store_fl_diagnostics=True, climate_filename='gcm_data',
                                                 # climate_input_filesuffix='_perturbed_{}_counterfactual'.format(
                                                 #     sample_id),
                                                 climate_input_filesuffix=f'_perturbed_{sample_id}_{var}_only',
                                                 output_filesuffix=out_id,
                                                 zero_initial_glacier=False, bias=0,
                                                 temperature_bias=None, precipitation_factor=None,
                                                 init_model_filesuffix='_spinup_historical',
                                                 init_model_yr=y0_clim)
            
                    opath = os.path.join(
                        # sum_dir, f'climate_run_output_perturbed_{sample_id}_counterfactual.nc')
                        sum_dir, f'climate_run_output_perturbed_{sample_id}_{var}_only.nc')
                    ds_ptb = utils.compile_run_output(
                        subset_gdirs, input_filesuffix=out_id, path=opath)  # compile the run output
            
                    log_path = os.path.join(
                        log_dir, f'stats_perturbed_{sample_id}_{var}_only_climate_run.nc')
                    df_stats = utils.compile_glacier_statistics(
                        subset_gdirs, path=log_path)

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()     

