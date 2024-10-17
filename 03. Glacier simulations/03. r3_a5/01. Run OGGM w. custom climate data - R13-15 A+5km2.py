# %% Cell 0: Load data packages
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 15:36:52 2024

@author: magaliponds

This code runs perturbs the climate data and adds this to the baseline climate in OGGM and runs the climate & MB model with this data"""


# -*- coding: utf-8 -*-import oggm
# from OGGM_data_processing import process_perturbation_data
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

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import pandas as pd
from scipy.optimize import curve_fit

from tqdm import tqdm
import pickle
import sys
import textwrap
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
function_directory = "/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/src/03. Glacier simulations"
sys.path.append(function_directory)

# %% Cell 0: Set base parameters

colors = {
    "W5E5": ["#000000"],  # "#000000"],  # Black
    # Darker to lighter shades of purple
    "E3SM": ["#785EF0", "#8F7BF1", "#A6A8F2"],
    # Darker to lighter shades of pink
    "CESM2": ["#DC267F", "#E58A9E", "#F0A2B6", "#F7BCC4"],
    # Darker to lighter shades of orange
    "CNRM": ["#FE6100", "#FE7D33", "#FE9A66", "#FEB799", "#FECDB5", "#FEF1E1"],
    "IPSL-CM6": ["#FFB000"]  # Dark purple to lighter shades
}

members = [1, 3, 4, 6, 1]
members_averages = [1, 2, 3, 5]
models = ["IPSL-CM6", "E3SM", "CESM2", "CNRM", "W5E5"]
models_shortlist = ["IPSL-CM6", "E3SM", "CESM2", "CNRM"]
timeframe = "monthly"

y0_clim = 1985
ye_clim = 2014

fig_path = '/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/04. Figures/02. OGGM simulations/01. Modelled output/3r_a5/'

# %% Cell 1: Initialize OGGM with the preferred model parameter set up
folder_path = '/Users/magaliponds/Documents/00. Programming'
wd_path = f'{folder_path}/03. Modelled perturbation-glacier interactions - R13-15 A+5km2/'
os.makedirs(wd_path, exist_ok=True)
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


cfg.PARAMS['use_multiprocessing'] = False
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
    cfg.PARAMS['use_multiprocessing'] = True

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


# %% Cell 4: Process the Irr climate perturbations for all the gdirs and compile


members = [1, 3, 4, 6, 1]
models = ["IPSL-CM6", "E3SM", "CESM2", "CNRM"]
timeframe = "monthly"

opath_climate = os.path.join(sum_dir, 'climate_historical.nc')
utils.compile_climate_input(
    gdirs_3r_a5, path=opath_climate, filename='climate_historical')

# if you get a long error log saying that "columns" can not be renamed it is often related to multiprocessing
cfg.PARAMS['use_multiprocessing'] = False
for m, model in enumerate(models):
    for member in range(members[m]):

        # Provide the path to the perturbation dataset
        i_folder_ptb = f"/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/03. Output files/01. Climate data/07. OGGM Perturbation input files/{timeframe}/{model}/{member}"
        ds_path = f"{i_folder_ptb}/{model}.00{member}.{y0_clim}_{ye_clim}.{timeframe}.perturbation.input.%.degC.nc"

        # Provide the sample ID to provide the processed pertrubations with the correct output suffix
        sample_id = f"{model}.00{member}"
        print(sample_id)

        workflow.execute_entity_task(process_perturbation_data, gdirs_3r_a5,
                                     ds_path=ds_path,
                                     y0=None, y1=None,
                                     output_filesuffix=f'_perturbation_{sample_id}')

        opath_perturbations = os.path.join(
            sum_dir, f'climate_historical_perturbation_{sample_id}.nc')
        utils.compile_climate_input(gdirs_3r_a5, path=opath_perturbations, filename='climate_historical',
                                    input_filesuffix=f'_perturbation_{sample_id}',
                                    use_compression=True)


# %% Cell 5: Perturb the climate historical with the processed Irr-perturbations, output is gcm file
cfg.PARAMS['use_multiprocessing'] = True
count = 0
# gdir = gdirs_3r_a5[0]
for gdir in gdirs_3r_a5:
    count += 1
    print(round((count*100)/len(gdirs_3r_a5), 2))
    # tasks.init_present_time_glacier(gdir)
    with xr.open_dataset(gdir.get_filepath('climate_historical')) as ds:
        ds = ds.load()
    for m, model in enumerate(models_shortlist):
        for member in range(members[m]):
            sample_id = f"{model}.00{member}"
            # make a copy of the historical climate
            clim_ptb = ds.copy()
            # open the perturbation dataset and add the perturbations
            with xr.open_dataset(gdir.get_filepath('climate_historical', filesuffix='_perturbation_{}'.format(sample_id))) as ds_ptb:
                ds_ptb = ds_ptb.load()
            clim_ptb['temp'] = clim_ptb.temp - ds_ptb.temp
            clim_ptb['prcp'] = clim_ptb.prcp - clim_ptb.prcp * ds_ptb.prcp
            clim_ptb = clim_ptb.dropna('time')
            clim_ptb.to_netcdf(gdir.get_filepath(
                'gcm_data', filesuffix='_perturbed_{}'.format(sample_id)))


# %% Cell 6: Run Mass Balance model for the glaciers (V = A*B)

for m, model in enumerate(models):
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
                    mbmod = massbalance.MonthlyTIModel(
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
                error_ids.append((gdir.rgi_id, model, member))
                continue
            mean_mb = np.mean(mb_ts)
            mb_ts_mean.append((gdir.rgi_id, mean_mb))

            if model == "W5E5":
                mean_mb_ext = np.mean(mb_ts_ext)
                mb_ts_mean_ext.append((gdir.rgi_id, mean_mb_ext))

        # create a dataframe with the mass balance data of all gdirs_3r_a5
        mb_df_mean = pd.DataFrame(mb_ts_mean, columns=['rgi_id', 'B'])
        mb_df_mean.to_csv(os.path.join(
            sum_dir, f'specific_massbalance_mean_{sample_id}.csv'), index=False)
        mb_ts_df = pd.DataFrame(
            mb_ts_all, columns=['rgi_id', 'Year', 'Mass_Balance'])
        mb_ts_df.to_csv(os.path.join(
            sum_dir, f'specific_massbalance_timeseries_{sample_id}.csv'), index=False)

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
                error_ids, columns=['rgi_id', 'Model', 'Member'])
            error_df.to_csv(os.path.join(
                log_dir, 'Error_Log.csv'), index=False)
# %% TEST Cell 6: Run Mass Balance model for the glaciers (V = A*B)

for m, model in enumerate(models):
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
            try:
                # provide the model flowlines and years for the mbmod
                fls = gdir.read_pickle('model_flowlines')
                years = np.arange(2000, 2014)

                if model == "W5E5":
                    # extend range for w5e5, to see match w. geodetic mass balance
                    # Get the calibrated mass-balance model - the default is to use OGGM's "MonthlyTIModel" and compute specific mb
                    mbmod = massbalance.MonthlyTIModel(
                        gdir, filename='climate_historical')
                    mb_ts = mbmod.get_specific_mb(fls=fls, year=years)

                    for year, mb in zip(years, mb_ts):
                        mb_ts_all.append((gdir.rgi_id, years, mb_ts))
                else:
                    mbmod = massbalance.MonthlyTIModel(
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
                error_ids.append((gdir.rgi_id, model, member))
                continue
            mean_mb = np.mean(mb_ts)
            mb_ts_mean.append((gdir.rgi_id, mean_mb))

        # create a dataframe with the mass balance data of all gdirs_3r_a5
        mb_df_mean = pd.DataFrame(mb_ts_mean, columns=['rgi_id', 'B'])
        mb_df_mean.to_csv(os.path.join(
            sum_dir, f'specific_massbalance_mean_{sample_id}_2000_2014.csv'), index=False)
        mb_ts_df = pd.DataFrame(
            mb_ts_all, columns=['rgi_id', 'Year', 'Mass_Balance'])
        mb_ts_df.to_csv(os.path.join(
            sum_dir, f'specific_massbalance_timeseries_{sample_id}_2000_2014.csv'), index=False)

        # Optionally save the list of error cases to a CSV for later review
        if error_ids:
            error_df = pd.DataFrame(
                error_ids, columns=['rgi_id', 'Model', 'Member'])
            error_df.to_csv(os.path.join(
                log_dir, 'Error_Log.csv'), index=False)
# %% Cell 7: Run the climate model - Save pkl after running is done , as running takes quite a while

cfg.PARAMS['continue_on_error'] = True
cfg.PARAMS['use_multiprocessing'] = False

# gdir = gdirs_3r_a5[0]
y0_clim = 1985
ye_clim = 2014

members = [1, 3, 4, 6]
models = ["IPSL-CM6", "E3SM", "CESM2", "CNRM"]
for m, model in enumerate(models):
    for member in range(members[m]):
        sample_id = f"{model}.00{member}"
        print(sample_id)
        workflow.execute_entity_task(
            tasks.init_present_time_glacier, gdirs_3r_a5)
        out_id = f'_perturbed_{sample_id}'
        opath = os.path.join(
            sum_dir, f'climate_run_output_perturbed_{sample_id}.nc')
        workflow.execute_entity_task(tasks.run_from_climate_data, gdirs_3r_a5,
                                     ys=y0_clim, ye=ye_clim,  # min_ys=None,
                                     max_ys=None, fixed_geometry_spinup_yr=None,
                                     store_monthly_step=False, store_model_geometry=None,
                                     store_fl_diagnostics=True, climate_filename='gcm_data',
                                     # mb_model=None, mb_model_class=<class 'oggm.core.massbalance.MonthlyTIModel'>,
                                     climate_input_filesuffix='_perturbed_{}'.format(
                                         sample_id),
                                     output_filesuffix=out_id,
                                     # init_model_filesuffix='_historical', #in case you want to start from a previous model state
                                     # init_model_yr=2015, init_model_fls=None,
                                     zero_initial_glacier=False, bias=0,
                                     temperature_bias=None, precipitation_factor=None)

        ds_ptb = utils.compile_run_output(
            gdirs_3r_a5, input_filesuffix=out_id, path=opath)  # compile the run output


# And run the climate model with reference data
workflow.execute_entity_task(tasks.run_from_climate_data, gdirs_3r_a5,
                             ys=y0_clim, ye=ye_clim,
                             output_filesuffix='_baseline_W5E5.000')
opath_base = os.path.join(sum_dir, 'climate_run_output_baseline_W5E5.000.nc')
ds_base = utils.compile_run_output(
    gdirs_3r_a5, input_filesuffix='_baseline_W5E5.000', path=opath_base)
