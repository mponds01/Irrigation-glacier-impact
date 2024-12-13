# %% Cell 0: Load data packages
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 15:36:52 2024

@author: magaliponds

This code runs perturbs the climate data and adds this to the baseline climate in OGGM and runs the climate & MB model with this data"""


# -*- coding: utf-8 -*-import oggm

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
from OGGM_data_processing import process_perturbation_data
import sys
function_directory = "/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/src/03. Glacier simulations"
sys.path.append(function_directory)


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

members = [1, 3, 4, 6, 4]
members_averages = [1, 2, 3, 5]
models = ["IPSL-CM6", "E3SM", "CESM2", "CNRM", "W5E5"]
models_shortlist = ["IPSL-CM6", "E3SM", "CESM2", "CNRM"]
timeframe = "monthly"

y0_clim = 1985
ye_clim = 2014
y0_cf = 1901
ye_cf = 1985


fig_path = '/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/04. Figures/02. OGGM simulations/01. Modelled output/3r_a5/'

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


# %% Cell 2a: Download all the regional shapefiles & flatten output list

regions = [13, 14, 15]  # The RGI regions you want to process

rgi_files = []
for region in regions:
    rgi_file = utils.get_rgi_region_file(region=region)
    rgi_files.append(rgi_file)

rgi_ids = []
for fr in rgi_files:
    gdf = gpd.read_file(fr)
    rgi_ids.append(gdf['RGIId'].values)

# Flatten the list of arrays and convert all elements to strings
rgi_ids_flattened = [item for sublist in rgi_ids for item in sublist]


# %% Cell 2b: ONLY DOWNLOAD ONCE (takes a long time) Get OGGM gdirs_3r

download_gdirs = False

if download_gdirs == True:

    # OGGM options
    oggm.cfg.PATHS['working_dir'] = utils.mkdir(wd_path, reset=False)
    oggm.cfg.PARAMS['store_model_geometry'] = True
    # cfg.PARAMS['use_multiprocessing'] = True

    gdirs = []
    with tqdm(total=len(rgi_ids_flattened), desc="Initializing Glacier Directories", mininterval=1.0) as pbar:
        for r, rgi_id in enumerate(rgi_ids_flattened):
            # Print RGI ID to check its correctness
            print(f"Processing RGI ID: {rgi_id}")
            # Perform your processing here
            new_gdirs = workflow.init_glacier_directories(
                [rgi_id],
                prepro_base_url=DEFAULT_BASE_URL,
                from_prepro_level=4,
                prepro_border=80
            )
            gdirs.extend(new_gdirs)

            # Update tqdm progress bar
            pbar.update(1)

gdirs_3r = gdirs  # rename gdir file to distinguish from other scripts
# %% Cell 2c: Save gdirs_3r from OGGM as a pkl file (Save at the end of every working session)

# Save each gdir individually
for gdir in gdirs:
    gdir_path = os.path.join(pkls, f'{gdir.rgi_id}.pkl')
    with open(gdir_path, 'wb') as f:
        pickle.dump(gdir, f)

# %% Cell 2d: Load gdirs_3r from pkl (fastest way to get started)

wd_path_pkls = f'{wd_path}/pkls/'

gdirs_3r = []
for filename in os.listdir(wd_path_pkls):
    if filename.endswith('.pkl'):
        # f'{gdir.rgi_id}.pkl')
        file_path = os.path.join(wd_path_pkls, filename)
        with open(file_path, 'rb') as f:
            gdir = pickle.load(f)
            gdirs_3r.append(gdir)

# # print(gdirs)
# %% Cell 3a: Filter the gdirs list where the area is greater than 5 km^2

# Pre-load all areas into memory first with a progress bar
areas = [gdir.rgi_area_km2 for gdir in tqdm(
    gdirs_3r, desc="Loading glacier areas")]

# Filter glaciers with area > 5 km², with a progress bar and total set
gdirs_3r_a5 = [gdir for gdir, area in tqdm(zip(gdirs_3r, areas), total=len(
    gdirs_3r), desc="Filtering glaciers") if area > 5]
# %% Cell 3b: Save the subset of gdirs with area larger than 5km2
pkls_subset = os.path.join(wd_path, "pkls_subset")
os.makedirs(pkls_subset, exist_ok=True)

# Save each gdir individually
for gdir in gdirs_3r_a5:
    gdir_path = os.path.join(pkls_subset, f'{gdir.rgi_id}.pkl')
    with open(gdir_path, 'wb') as f:
        pickle.dump(gdir, f)


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

# # print(gdirs)

# %% create overview of all glacier ids (for Rodrigo)

glacier_ids = []
for gdir in gdirs_3r_a5:
    glacier_ids.append((gdir.rgi_id))

print(len(glacier_ids))
glacier_ids = pd.DataFrame(
    glacier_ids, columns=['rgi_id'])
glacier_ids.to_csv(os.path.join(
    wd_path, "masters", "Overview_rgi_ids_all.csv"))


# %% Cell 3d: Filter out ids with errors

stats = utils.compile_glacier_statistics(gdirs_3r_a5, path=os.path.join(
    sum_dir, "prepro_stats.csv"), inversion_only=False, apply_func=None)

statistics = pd.read_csv(os.path.join(sum_dir, "prepro_stats.csv"))
failed = statistics[statistics.run_dynamic_spinup_success == False]
failed_rgi_ids = set(failed['rgi_id'])
failed_rgi_ids.add("RGI60-13.36875")
gdirs_test_filtered = [
    gdir for gdir in gdirs_3r_a5 if gdir.rgi_id not in failed_rgi_ids]

# %% Cell 3e: Save only successful filtered glaciers to new pkls

pkls_subset = os.path.join(wd_path, "pkls_subset_success")
os.makedirs(pkls_subset, exist_ok=True)

# Save each gdir individually
for gdir in gdirs_test_filtered:
    gdir_path = os.path.join(pkls_subset, f'{gdir.rgi_id}.pkl')
    with open(gdir_path, 'wb') as f:
        pickle.dump(gdir, f)

# %% Cell 3f: Load successful gdirs

wd_path_pkls = f'{wd_path}/pkls_subset_success/'

gdirs_3r_a5 = []
for filename in os.listdir(wd_path_pkls):
    if filename.endswith('.pkl'):
        # f'{gdir.rgi_id}.pkl')
        file_path = os.path.join(wd_path_pkls, filename)
        with open(file_path, 'rb') as f:
            gdir = pickle.load(f)
            gdirs_3r_a5.append(gdir)

# # print(gdirs)

# %% Cell 4: Process the Irr climate perturbations for all the gdirs and compile
cfg.initialize()
cfg.PATHS['working_dir'] = utils.mkdir(wd_path, reset=False)

members = [1, 3, 4, 6, 4]
models = ["IPSL-CM6", "E3SM", "CESM2", "CNRM", "NorESM"]
timeframe = "monthly"

# opath_climate = os.path.join(sum_dir, 'climate_historical.nc')
# utils.compile_climate_input(
#     gdirs_3r_a5, path=opath_climate, filename='climate_historical')

# if you get a long error log saying that "columns" can not be renamed it is often related to multiprocessing
# cfg.PARAMS['use_multiprocessing'] = False
for m, model in enumerate(models):
    if model == "NorESM":
        for member in range(members[m]):
            if member >= 1:
                # Provide the path to the perturbation dataset
                # if error with lon.min or ds['time.year'] check if lon>0 in creating input dataframe
                i_folder_ptb = f"/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/03. Output files/01. Climate data/07. OGGM Perturbation input files/{timeframe}/{model}/{member}"
                ds_path = f"{i_folder_ptb}/{model}.00{member}.{y0_clim}_{ye_clim}.{timeframe}.perturbation.input.%.degC.nc"
                # ds_path = f"{i_folder_ptb}/{model}.00{member}.{y0_cf}_{ye_cf}.{timeframe}.perturbation.input.%.degC.counterfactual.nc"

                # Provide the sample ID to provide the processed pertrubations with the correct output suffix
                sample_id = f"{model}.00{member}"

                workflow.execute_entity_task(process_perturbation_data, gdirs_3r_a5,
                                             ds_path=ds_path,
                                             # y0=1985, y1=2014,
                                             y0=None, y1=None,
                                             # output_filesuffix=f'_perturbation_{sample_id}_counterfactual')
                                             output_filesuffix=f'_perturbation_{sample_id}')

                opath_perturbations = os.path.join(
                    sum_dir, f'climate_historical_perturbation_{sample_id}.nc')
                # opath_perturbations = os.path.join(
                #     sum_dir, f'climate_historical_perturbation_{sample_id}_counterfactual.nc')

                utils.compile_climate_input(gdirs_3r_a5, path=opath_perturbations, filename='climate_historical',
                                            # input_filesuffix=f'_perturbation_{sample_id}_counterfactual',
                                            input_filesuffix=f'_perturbation_{sample_id}',
                                            use_compression=True)

# %% Cell 4b: Process the Irr climate perturbations for all the gdirs and compile - future
cfg.initialize()
cfg.PATHS['working_dir'] = utils.mkdir(wd_path, reset=False)

members = [4]
models = ["CESM2"]
scenarios = ["370"]  # 126",
timeframe = "monthly"

# opath_climate = os.path.join(sum_dir, 'climate_historical.nc')
# utils.compile_climate_input(
#     gdirs_3r_a5, path=opath_climate, filename='climate_historical')
y0_clim = 2015
ye_clim = 2074
# if you get a long error log saying that "columns" can not be renamed it is often related to multiprocessing
# cfg.PARAMS['use_multiprocessing'] = False
for m, model in enumerate(models):
    for member in range(members[m]):
        for s, scenario in enumerate(scenarios):
            if member != 0 and member < 3:
                # Provide the path to the perturbation dataset
                # if error with lon.min or ds['time.year'] check if lon>0 in creating input dataframe
                i_folder_ptb = f"/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/03. Output files/01. Climate data/07. OGGM Perturbation input files/{timeframe}/{model}/{member}/SSP{scenario}"
                ds_path = f"{i_folder_ptb}/{model}.00{member}.SSP{scenario}.{y0_clim}_{ye_clim}.{timeframe}.perturbation.input.%.degC.nc"
                # ds_path = f"{i_folder_ptb}/{model}.00{member}.{y0_cf}_{ye_cf}.{timeframe}.perturbation.input.%.degC.counterfactual.nc"

                # Provide the sample ID to provide the processed pertrubations with the correct output suffix
                sample_id = f"{model}.00{member}"

                workflow.execute_entity_task(process_perturbation_data, gdirs_3r_a5,
                                             ds_path=ds_path,
                                             # y0=y0_clim, y1=ye_clim,
                                             # y0=None, y1=None,
                                             output_filesuffix=f'_perturbation_{sample_id}_SSP{scenario}')

                opath_perturbations = os.path.join(
                    sum_dir, f'climate_historical_perturbation_{sample_id}_SSP{scenario}.nc')

                utils.compile_climate_input(gdirs_3r_a5, path=opath_perturbations, filename='climate_historical',
                                            input_filesuffix=f'_perturbation_{sample_id}_SSP{scenario}',
                                            use_compression=True)
# %% Cell 5: Perturb the climate historical with the processed Irr-perturbations, output is gcm file
# cfg.PARAMS['use_multiprocessing'] = True
count = 0
members = [1, 3, 4, 6, 1, 4]
models = ["IPSL-CM6", "E3SM", "CESM2", "CNRM", "NorESM"]
# gdir = gdirs_3r_a5[0]

for m, model in enumerate(models):  # models_shortlist
    for member in range(members[m]):
        for gdir in gdirs_3r_a5:
            count += 1
            print(round((count*100)/len(gdirs_3r_a5), 2))
            # tasks.init_present_time_glacier(gdir)
            if m == 0:
                with xr.open_dataset(gdir.get_filepath('climate_historical')) as ds:
                    ds = ds.load()
                sample_id = f"{model}.00{member}"
                # print(sample_id)
                # make a copy of the historical climate
            clim_ptb = ds.copy().sel(time=slice('1985-01-01', '2014-12-31'))

            # open the perturbation dataset and add the perturbations
            # with xr.open_dataset(gdir.get_filepath('climate_historical', filesuffix='_perturbation_{}_counterfactual'.format(sample_id))) as ds_ptb:
            with xr.open_dataset(gdir.get_filepath('climate_historical', filesuffix='_perturbation_{}'.format(sample_id))) as ds_ptb:
                ds_ptb = ds_ptb.load()

            clim_ptb['temp'] = clim_ptb.temp - ds_ptb.temp
            clim_ptb['prcp'] = clim_ptb.prcp - clim_ptb.prcp * ds_ptb.prcp
            #
            clim_ptb.to_netcdf(gdir.get_filepath(
                # 'gcm_data', filesuffix='_perturbed_{}_counterfactual'.format(sample_id)))
                'gcm_data', filesuffix='_perturbed_{}'.format(sample_id)))
            # df_stats = utils.compile_glacier_statistics(
            #     gdirs_3r_a5, filesuffix=f"_perturbed_{sample_id}")


members = [1, 3, 4, 6, 1, 4]
models = ["IPSL-CM6", "E3SM", "CESM2", "CNRM", "NorESM"]

for m, model in enumerate(models):
    for member in range(members[m]):

        # create a sample id for all the model x member combinations
        sample_id = f"{model}.00{member}"
        print(sample_id)
        # load the gdirs_3r_a5
        for (g, gdir) in enumerate(gdirs_3r_a5):
            if gdir.rgi_id == "RGI60-13.49226":
                # , filesuffix='_perturbation_{}_counterfactual'.format(sample_id))) as ds:
                with xr.open_dataset(gdir.get_filepath('climate_historical')) as ds:
                    ds = ds.load()
                clim_ptb = ds.copy()
                with xr.open_dataset(gdir.get_filepath('gcm_data', filesuffix='_perturbed_{}'.format(sample_id))) as ds_ptb:
                    ds_ptb = ds_ptb.load()
                clim_ptb['temp'] = clim_ptb.temp - ds_ptb.temp
                clim_ptb['prcp'] = clim_ptb.prcp - clim_ptb.prcp * ds_ptb.prcp

                clim_ptb = clim_ptb.dropna('time')
                ds_ptb = ds_ptb.dropna('time')

                plt.plot(ds_ptb.time, ds_ptb.temp, label="ds perturbation")
                print(ds_ptb.time)
                plt.plot(ds.time, ds.temp, label="ds og")
                print(ds.time)
                plt.plot(clim_ptb.time, clim_ptb.temp,
                         label="perturbed climate")
                print(clim_ptb.time)
                plt.legend()
# %% Perturb the climate data - future

count = 0
members = [4]
models = ["CESM2"]
# gdir = gdirs_3r_a5[0]

for m, model in enumerate(models):  # models_shortlist
    for member in range(members[m]):
        # for s,scenario in enumerate(["370"]):#126",
        for s, scenario in enumerate(["126"]):  # 126",
            for gdir in gdirs_3r_a5:
                if member != 0:
                    count += 1
                    print(round((count*100)/len(gdirs_3r_a5), 2))
                    # tasks.init_present_time_glacier(gdir)
                    sample_id = f"{model}.00{member}"

                    if m == 0:
                        with xr.open_dataset(gdir.get_filepath('climate_historical')) as ds:
                            ds = ds.load()
                        # print(sample_id)
                        # make a copy of the historical climate
                    clim_ptb = ds.copy().sel(time=slice('2015-01-01', '2074-12-31'))

                    # open the perturbation dataset and add the perturbations
                    with xr.open_dataset(gdir.get_filepath('climate_historical', filesuffix=f'_perturbation_{sample_id}_SSP{scenario}')) as ds_ptb:
                        ds_ptb = ds_ptb.load()

                    clim_ptb['temp'] = clim_ptb.temp - ds_ptb.temp
                    clim_ptb['prcp'] = clim_ptb.prcp - \
                        clim_ptb.prcp * ds_ptb.prcp
                    #
                    clim_ptb.to_netcdf(gdir.get_filepath(
                        'gcm_data', filesuffix=f'_perturbation_{sample_id}_SSP{scenario}'))
                    # clim_ptb.to_netcdf(gdir.get_filepath('gcm_data'), filesuffix='_perturbation_{}_SSP126'.format(sample_id))
                    # df_stats = utils.compile_glacier_statistics(
                    #     gdirs_3r_a5, filesuffix=f"_perturbed_{sample_id}")


# %% Cell 6: Run Mass Balance model for the glaciers (V = A*B)
members = [1, 3, 4, 6, 4]
models = ["IPSL-CM6", "E3SM", "CESM2", "CNRM", "NorESM"]


for m, model in enumerate(models):
    count = 0
    for member in range(members[m]):

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

        # load the gdirs_3r_a5
        for (g, gdir) in enumerate(gdirs_3r_a5):
            count += 1
            print(round((count*100)/(len(gdirs_3r_a5)*members[m]), 2))

            try:
                # provide the model flowlines and years for the mbmod
                fls = gdir.read_pickle('model_flowlines')
                years = np.arange(1985, 2014)

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
                    print(gdir.rgi_id)
                    mbmod = massbalance.MonthlyTIModel(
                        # gdir, filename='gcm_data', input_filesuffix='_perturbed_{}_counterfactual'.format(sample_id))
                        gdir, filename='gcm_data', input_filesuffix='_perturbed_{}'.format(sample_id))
                    mb_ts = mbmod.get_specific_mb(fls=fls, year=years)
                # Append all time series data to mb_ts_all
                for year, mb in zip(years, mb_ts):
                    mb_ts_all.append((gdir.rgi_id, year, mb_ts))

            # include an exception so the model will continue running on error and provide the error
            except Exception as e:
                # Handle the error and continue
                print(
                    f"Error processing {gdir.rgi_id} with model {model} and member {member}: {e}")
                # found error: RGI60-13.36875 no flowlines --> 542 to 541 glaciers in selected gdirs_3r_a5
                error_ids.append((gdir.rgi_id, model, member, e))
                continue
            mean_mb = np.mean(mb_ts)
            mb_ts_mean.append((gdir.rgi_id, mean_mb))

            if model == "W5E5":
                mean_mb_ext = np.mean(mb_ts_ext)
                mb_ts_mean_ext.append((gdir.rgi_id, mean_mb_ext))
            count += 1

        # create a dataframe with the mass balance data of all gdirs_3r_a5
        mb_df_mean = pd.DataFrame(mb_ts_mean, columns=['rgi_id', 'B'])
        mb_df_mean.to_csv(os.path.join(
            sum_dir, f'specific_massbalance_mean_{sample_id}_counterfactual.csv'), index=False)
        # sum_dir, f'specific_massbalance_mean_{sample_id}.csv'), index=False)
        mb_ts_df = pd.DataFrame(
            mb_ts_all, columns=['rgi_id', 'Year', 'Mass_Balance'])
        mb_ts_df.to_csv(os.path.join(
            sum_dir, f'specific_massbalance_timeseries_{sample_id}_counterfactual.csv'), index=False)
        # sum_dir, f'specific_massbalance_timeseries_{sample_id}.csv'), index=False)

        if model == "W5E5":  # only for W5E5 also create a dataframe for the extended timeseries
            mb_df_mean_ext = pd.DataFrame(
                mb_ts_mean_ext, columns=['rgi_id', 'B'])
            mb_df_mean_ext.to_csv(os.path.join(
                sum_dir, f'specific_massbalance_mean_extended_{sample_id}.csv'), index=False)
            mb_ts_df_ext = pd.DataFrame(mb_ts_all_ext, columns=[
                                        'rgi_id', 'Year', 'Mass_Balance'])
            mb_ts_df_ext.to_csv(os.path.join(
                sum_dir, f'specific_massbalance_timeseries_extended_{sample_id}.csv'), index=False)

        # Optionally save the list of error cases to a CSV for later review
        if error_ids:
            error_df = pd.DataFrame(
                error_ids, columns=['rgi_id', 'Model', 'Member', 'error'])
            error_df.to_csv(os.path.join(
                # log_dir, f'Error_Log_counterfactual_{sample_id}.csv'), index=False)
                log_dir, f'Error_Log_NoIrr_{sample_id}.csv'), index=False)

# %% Cell 7: Run the climate model baseline and perturbations - Save pkl after running is done , as running takes quite a while

cfg.PARAMS['continue_on_error'] = True
cfg.PARAMS['use_multiprocessing'] = False
cfg.PARAMS['border'] = 240

# gdir = gdirs_3r_a5[0]
y0_clim = 1985
ye_clim = 2014

subset_gdirs = gdirs_3r_a5
members = [1, 3, 4, 6, 4]
models = ["IPSL-CM6", "E3SM", "CESM2", "CNRM", "NorESM"]
for m, model in enumerate(models):
    for member in range(members[m]):
        sample_id = f"{model}.00{member}"
        print(sample_id)
        workflow.execute_entity_task(
            tasks.init_present_time_glacier, gdirs_3r_a5)
        # out_id = f'_perturbed_{sample_id}_counterfactual'
        out_id = f'_perturbed_{sample_id}'

        workflow.execute_entity_task(
            tasks.init_present_time_glacier, gdirs_3r_a5)

        workflow.execute_entity_task(tasks.run_from_climate_data, subset_gdirs,
                                     ys=y0_clim, ye=ye_clim,  # min_ys=None,
                                     max_ys=None, fixed_geometry_spinup_yr=None,
                                     store_monthly_step=False, store_model_geometry=True,
                                     store_fl_diagnostics=True, climate_filename='gcm_data',
                                     # climate_input_filesuffix='_perturbed_{}_counterfactual'.format(
                                     #     sample_id),
                                     climate_input_filesuffix='_perturbed_{}'.format(
                                         sample_id),
                                     output_filesuffix=out_id,
                                     zero_initial_glacier=False, bias=0,
                                     temperature_bias=None, precipitation_factor=None,
                                     init_model_filesuffix='_spinup_historical',
                                     init_model_yr=y0_clim)

        opath = os.path.join(
            # sum_dir, f'climate_run_output_perturbed_{sample_id}_counterfactual.nc')
            sum_dir, f'climate_run_output_perturbed_{sample_id}.nc')
        ds_ptb = utils.compile_run_output(
            subset_gdirs, input_filesuffix=out_id, path=opath)  # compile the run output

        log_path = os.path.join(
            log_dir, f'stats_perturbed_{sample_id}_climate_run.nc')
        df_stats = utils.compile_glacier_statistics(
            subset_gdirs, path=log_path)


# And run the climate model with reference data
workflow.execute_entity_task(tasks.run_from_climate_data, subset_gdirs,
                             ys=y0_clim, ye=ye_clim,
                             output_filesuffix='_baseline_W5E5.000',
                             init_model_filesuffix='_spinup_historical',
                             init_model_yr=y0_clim, store_fl_diagnostics=True)

opath_base = os.path.join(sum_dir, 'climate_run_output_baseline_W5E5.000.nc')
ds_base = utils.compile_run_output(
    subset_gdirs, input_filesuffix='_baseline_W5E5.000', path=opath_base)

log_path_base = os.path.join(
    log_dir, f'stats_perturbed_W5E5.000_climate_run.csv')
df_stats = utils.compile_glacier_statistics(
    subset_gdirs, path=log_path_base)

# %% Cell 8: Set up run with hydro

cfg.PARAMS['continue_on_error'] = True
cfg.PARAMS['use_multiprocessing'] = False
cfg.PARAMS['border'] = 240

# gdir = gdirs_3r_a5[0]
y0_clim = 1985
ye_clim = 2014

subset_gdirs = gdirs_3r_a5[:10]
members = [1, 3, 4, 6, 4]
models = ["IPSL-CM6", "E3SM", "CESM2", "CNRM", "NorESM"]
for m, model in enumerate(models):
    for member in range(members[m]):
        sample_id = f"{model}.00{member}"
        print(sample_id)
        out_id = f'_hydro_perturbed_{sample_id}'

        workflow.execute_entity_task(tasks.run_with_hydro, subset_gdirs,
                                     run_task=tasks.run_from_climate_data,
                                     ys=y0_clim, ye=ye_clim,
                                     climate_filename='gcm_data',
                                     # climate_input_filesuffix='_perturbed_{}_counterfactual'.format(
                                     #     sample_id),
                                     climate_input_filesuffix='_perturbed_{}'.format(
                                         sample_id),
                                     output_filesuffix=out_id,
                                     init_model_filesuffix='_spinup_historical',
                                     init_model_yr=y0_clim,
                                     store_fl_diagnostics=True,
                                     store_monthly_hydro=True)

        opath = os.path.join(
            # sum_dir, f'hydro_run_output_perturbed_{sample_id}_counterfactual.nc')
            sum_dir, f'hydro_run_output_perturbed_{sample_id}.nc')
        ds_ptb = utils.compile_run_output(
            subset_gdirs, input_filesuffix=out_id, path=opath)  # compile the run output

        log_path = os.path.join(
            log_dir, f'stats_perturbed_{sample_id}_hydro_run.nc')
        df_stats = utils.compile_glacier_statistics(
            subset_gdirs, path=log_path)

# # And run the climate model with reference data
# workflow.execute_entity_task(tasks.run_with_hydro, subset_gdirs,
#                              run_task=tasks.run_from_climate_data,
#                              ys=y0_clim, ye=ye_clim,
#                              climate_filename='climate_historical',
#                              # climate_input_filesuffix=
#                              output_filesuffix='_hydro_baseline_W5E5.000',
#                              init_model_filesuffix='_spinup_historical',
#                              init_model_yr=y0_clim, store_fl_diagnostics=True,
#                              store_monthly_hydro=True)

# opath_base = os.path.join(sum_dir, 'hydro_run_output_baseline_W5E5.000.nc')
# ds_base = utils.compile_run_output(
#     subset_gdirs, input_filesuffix='_hydro_baseline_W5E5.000', path=opath_base)

# log_path_base = os.path.join(
#     log_dir, f'stats_perturbed_W5E5.000_hydro_run.csv')
# df_stats = utils.compile_glacier_statistics(
#     subset_gdirs, path=log_path_base)

# %% Cell 9: Run comitted mass loss (random climate data)

cfg.PARAMS['border'] = 240
cfg.PARAMS['continue_on_error'] = True
cfg.PARAMS['use_multiprocessing'] = False


subset_gdirs = gdirs_3r_a5  # [:100]

members = [1, 6, 3, 4, 6, 4]
models = ["IPSL-CM6", "CNRM", "E3SM", "CESM2", "CNRM", "NorESM"]

y0_comitted = 2014
halfsize = 14.5
for m, model in enumerate(models):
    for member in range(members[m]):
        # try:
        sample_id = f"{model}.00{member}"
        print(sample_id)
        workflow.execute_entity_task(
            tasks.init_present_time_glacier, subset_gdirs)  # gdirs_3r_a5)

        # out_id = f'_perturbed_{sample_id}_counterfactual'
        out_id = f'_perturbed_{sample_id}_committed_random'
        out_id_climate_run = f'_perturbed_{sample_id}'

        workflow.execute_entity_task(tasks.run_random_climate, subset_gdirs,  # gdirs_3r_a5,
                                     nyears=250,  # nr of years to simulate
                                     ys=y0_comitted,  # start year of simulation
                                     halfsize=halfsize,  # half size applied to random climate distriubtion
                                     y0=y0_comitted-halfsize,
                                     ye=None, bias=0,  # bias to correction of climate data set to 0
                                     seed=2,  # initializing the random nr generator with number, so every time the same number applies
                                     precipitation_factor=None, store_monthly_step=False,
                                     store_model_geometry=None, store_fl_diagnostics=True,
                                     climate_filename='gcm_data',  # using the perturbed gcm data

                                     # climate_input_filesuffix='_perturbed_{}_counterfactual'.format(sample_id),
                                     climate_input_filesuffix='_perturbed_{}'.format(
                                         sample_id),

                                     output_filesuffix=out_id,  # adding file suffix
                                     init_model_filesuffix=out_id_climate_run,
                                     init_model_yr=y0_comitted,
                                     continue_on_error=True  # the year of the initial run you wnat to start from
                                     )
        opath = os.path.join(
            # sum_dir, f'climate_run_output_perturbed_{sample_id}_counterfactual.nc')
            sum_dir, f'climate_run_output_perturbed_{sample_id}_comitted_random.nc')

        log_path = os.path.join(
            log_dir, f'stats_perturbed_{sample_id}_comitted_random.csv')

        ds_ptb = utils.compile_run_output(
            subset_gdirs, path=opath, input_filesuffix=out_id)  # gdirs_3r_a5, compile the run output
        df_stats = utils.compile_glacier_statistics(
            subset_gdirs, path=log_path)

out_id = f'_baseline_W5E5.000_committed_random'
out_id_climate_run = '_baseline_W5E5.000'
workflow.execute_entity_task(tasks.run_random_climate, subset_gdirs,  # gdirs_3r_a5,
                             nyears=250,  # nr of years to simulate
                             ys=y0_comitted,  # start year of simulation
                             halfsize=halfsize,  # half size applied to random climate distriubtion
                             y0=y0_comitted-halfsize,
                             ye=None, bias=0,  # bias to correction of climate data set to 0
                             seed=2,  # initializing the random nr generator with number, so every time the same number applies
                             precipitation_factor=None, store_monthly_step=False,
                             store_model_geometry=None, store_fl_diagnostics=True,
                             climate_filename='climate_historical',  # using the perturbed gcm data
                             output_filesuffix=out_id,  # adding file suffix
                             init_model_filesuffix=out_id_climate_run,
                             init_model_yr=y0_comitted,
                             continue_on_error=True
                             )

log_path_base = os.path.join(
    log_dir, f'stats_perturbed_W5E5.000_comitted_random.csv')

opath_base = os.path.join(
    sum_dir, 'climate_run_output_baseline_W5E5.000_comitted_random.nc')

df_stats = utils.compile_glacier_statistics(
    subset_gdirs, path=log_path_base)

ds_base = utils.compile_run_output(
    subset_gdirs, input_filesuffix=out_id, path=opath_base)  # gdirs_3r_a5,


# %% Cell 10: Run comitted mass loss (constant climate data)

cfg.PARAMS['border'] = 240
cfg.PARAMS['continue_on_error'] = True
cfg.PARAMS['use_multiprocessing'] = False


subset_gdirs = gdirs_3r_a5  # [:100]
for gdir in subset_gdirs:
    print(gdir.rgi_id)

members = [1, 6, 3, 4, 6, 4]
models = ["IPSL-CM6", "CNRM", "E3SM", "CESM2", "CNRM", "NorESM"]

# y0_comitted = 2014
# halfsize = 14.5
# for m, model in enumerate(models):
#     for member in range(members[m]):
#         if model in ["CNRM", "E3SM", "CESM2", "NorESM"]:
#             if model == "CNRM" and member <= 3:
#                 continue  # Skip members <= 4 for CNRM
#             if member==0:
#                 continue
#             # try:
#             sample_id = f"{model}.00{member}"
#             print(sample_id)
#             # workflow.execute_entity_task(
#             #     tasks.init_present_time_glacier, subset_gdirs)  # gdirs_3r_a5)

#             # out_id = f'_perturbed_{sample_id}_counterfactual'
#             out_id = f'_perturbed_{sample_id}_committed_cst'
#             out_id_climate_run = f'_perturbed_{sample_id}'

#             workflow.execute_entity_task(tasks.run_constant_climate, subset_gdirs,  # gdirs_3r_a5,
#                                          nyears=250,  # nr of years to simulate
#                                          ys=y0_comitted,  # start year of simulation
#                                          halfsize=halfsize,  # half size applied to random climate distriubtion
#                                          y0=y0_comitted-halfsize,
#                                          ye=None, bias=0,  # bias to correction of climate data set to 0
#                                          precipitation_factor=None, store_monthly_step=False,
#                                          store_model_geometry=None, store_fl_diagnostics=True,
#                                          climate_filename='gcm_data',  # using the perturbed gcm data

#                                          # climate_input_filesuffix='_perturbed_{}_counterfactual'.format(sample_id),
#                                          climate_input_filesuffix='_perturbed_{}'.format(
#                                              sample_id),

#                                          output_filesuffix=out_id,  # adding file suffix
#                                          init_model_filesuffix=out_id_climate_run,
#                                          init_model_yr=y0_comitted,
#                                          continue_on_error=True  # the year of the initial run you wnat to start from
#                                          )
#             opath = os.path.join(
#                 # sum_dir, f'climate_run_output_perturbed_{sample_id}_counterfactual.nc')
#                 sum_dir, f'climate_run_output_perturbed_{sample_id}_comitted_cst.nc')

#             log_path = os.path.join(
#                 log_dir, f'stats_perturbed_{sample_id}_comitted_cst.csv')

#             ds_ptb = utils.compile_run_output(
#                 subset_gdirs, path=opath, input_filesuffix=out_id)  # gdirs_3r_a5, compile the run output
#             df_stats = utils.compile_glacier_statistics(
#                 subset_gdirs, path=log_path)

out_id = f'_baseline_W5E5.000_committed_cst'
out_id_climate_run = '_baseline_W5E5.000'
workflow.execute_entity_task(tasks.run_constant_climate, subset_gdirs,  # gdirs_3r_a5,
                             nyears=250,  # nr of years to simulate
                             ys=y0_comitted,  # start year of simulation
                             halfsize=halfsize,  # half size applied to random climate distriubtion
                             y0=y0_comitted-halfsize,
                             ye=None, bias=0,  # bias to correction of climate data set to 0
                             precipitation_factor=None, store_monthly_step=False,
                             store_model_geometry=None, store_fl_diagnostics=True,
                             climate_filename='climate_historical',  # using the perturbed gcm data
                             output_filesuffix=out_id,  # adding file suffix
                             init_model_filesuffix=out_id_climate_run,
                             init_model_yr=y0_comitted,
                             continue_on_error=True
                             )

log_path_base = os.path.join(
    log_dir, f'stats_perturbed_W5E5.000_comitted_cst.csv')

df_stats = utils.compile_glacier_statistics(
    subset_gdirs, path=log_path_base)

opath_base = os.path.join(
    sum_dir, 'climate_run_output_baseline_W5E5.000_comitted_cst.nc')


ds_base = utils.compile_run_output(
    subset_gdirs, input_filesuffix=out_id, path=opath_base)  # gdirs_3r_a5,
