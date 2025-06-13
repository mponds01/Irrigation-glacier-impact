#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 23 12:44:11 2025

@author: magaliponds
"""


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
from datetime import datetime
#%%
# function_directory = "/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/src/03. Glacier simulations"
# sys.path.append(function_directory)
# from OGGM_data_processing import process_perturbation_data


#%% Cell 0: Initialize OGGM with the preferred model parameter set up

folder_path = '/Users/magaliponds/Documents/00. Programming'
wd_path = f'{folder_path}/04. Modelled perturbation-glacier interactions - R13-15 A+1km2/'
# wd_path = f'{folder_path}/03. Modelled perturbation-glacier interactions - R13-15 A+5km2/'
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
#%% Cell 1: Plot perturbations per model and member

members = [1, 3, 4, 4,6]
models = ["IPSL-CM6", "E3SM", "CESM2","NorESM", "CNRM"]
timeframe = "monthly"
fig,axes =plt.subplots(5,3,figsize=(8,8), sharex=True, sharey=True)
axes = axes.flatten()
i=0
for m, model in enumerate(models):
        for member in range(members[m]):
            ax = axes[i]
            if model =="IPSL-CM6" or member>=1:
                sample_id = f"{model}.00{member}"
                
                opath_perturbations = os.path.join(
                    sum_dir, f'climate_historical_perturbation_{sample_id}.nc')
                
                with xr.open_dataset(opath_perturbations) as ds:
                    ax.plot(ds.mean(dim='rgi_id').time, ds.mean(dim='rgi_id').temp, color='k')
                    # ax.plot(ds.mean(dim='rgi_id').time, ds.mean(dim='rgi_id').prcp, color='b')
                    ax.set_title(sample_id)
                i+=1
                print(i)

fig.suptitle("Monthly Temperature Perturbations", fontsize=14, y=0.93)
fig.text(0.04, 0.5, "Temperature perturbation [°C]", va='center', rotation='vertical', fontsize=14)            
# fig.suptitle("Monthly Precipitation Perturbations", fontsize=14, y=0.93)
# fig.text(0.04, 0.5, "Precipitation perturbation [%]", va='center', rotation='vertical', fontsize=14)            
plt.tight_layout(rect=[0.05, 0.03, 1, 0.95])  # Initial layout
fig.subplots_adjust(hspace=0.5)  # Increase vertical spacing        
#%% DOuble click on NorESM data                 

members = [4]#1, 3, 4, 4,6]
models = ["NorESM"]#"IPSL-CM6", "E3SM", "CESM2","NorESM", "CNRM"]
timeframe = "monthly"
axes = axes.flatten()
i=0
for m, model in enumerate(models):
        for member in range(members[m]):
            ax = axes[i]
            if model =="IPSL-CM6" or member>=1:
                sample_id = f"{model}.00{member}"
                print(sample_id)
                
                opath_perturbations = os.path.join(
                    sum_dir, f'climate_historical_perturbation_{sample_id}.nc')
                
                with xr.open_dataset(opath_perturbations) as ds:
                    # prcp = ds['prcp']

                    # # Boolean mask for outliers
                    # outlier_mask = (prcp > 10000) | (prcp < -10000)
                    
                    # # Count of outliers
                    # n_outliers = outlier_mask.sum().item()
                    
                    # # Total number of values
                    # total_values = prcp.size  # or len(prcp.values) if it's 1D
                    
                    # # Percentage of outliers
                    # percentage_outliers = (n_outliers / total_values) * 100|
                    
                    # print(f"Outliers: {n_outliers} of {total_values} ({percentage_outliers:.2f}%)")

                    outlier_mask = (ds['prcp'] > 1) | (ds['prcp'] < -1)

                    # Reduce over time dimension to find any outlier for each rgi_id
                    has_outliers = outlier_mask.any(dim='time')
                    
                    # Extract the rgi_ids where outliers are present
                    rgi_ids_with_outliers = ds['rgi_id'].where(has_outliers, drop=True)
                    
                    print(f"RGI IDs with outliers:\n{rgi_ids_with_outliers.values}")        
                    print(len(rgi_ids_with_outliers))
                   

#%% Go back to perturbation input file
members = [4]#1, 3, 4, 4,6]
models = ["NorESM"]#"IPSL-CM6", "E3SM", "CESM2","NorESM", "CNRM"]
timeframe = "monthly"
y0_clim=1985
ye_clim=2014
i=0
fig, axes = plt.subplots(3,3, figsize=(12,8))
axes = axes.flatten()
for m, model in enumerate(models):
        for member in range(members[m]):
            if model =="IPSL-CM6" or member>=1:
                sample_id = f"{model}.00{member}"
                
                i_folder_ptb = f"/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/03. Output files/01. Climate data/07. OGGM Perturbation input files/{timeframe}/{model}/{member}"
                ds_path = f"{i_folder_ptb}/{model}.00{member}.{y0_clim}_{ye_clim}.{timeframe}.perturbation.input.%.degC.nc"      
                with xr.open_dataset(ds_path, engine='h5netcdf') as ds:
                    ds=ds.mean(dim='time')
                    ds.temp.plot.imshow(x='lon', y='lat', ax=axes[3*i])
                    ds.prcp.plot.imshow(x='lon', y='lat', ax=axes[1+(3*i)], vmin=-0.2, vmax=0.2)
                    ds.hgt.plot.imshow(x='lon', y='lat', ax=axes[2+(3*i)])

            
                i+=1
                print(i)
                print(2+(3*i))

#%%

n_models = 3
fig, axes = plt.subplots(n_models, 3, figsize=(10, n_models * 3))
axes = np.array(axes)  # shape (n_models, 3)

# Storage for the mappables (images)
im_temp = []
im_prcp = []
im_hgt = []

for i, (model, member_count) in enumerate(zip(models, members)):
    for member in range(member_count):
        if model == "IPSL-CM6" or member >= 1:
            print(i)
            sample_id = f"{model}.00{member}"
            print(sample_id)
            i_folder_ptb = f"/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/03. Output files/01. Climate data/07. OGGM Perturbation input files/{timeframe}/{model}/{member}"
            ds_path = f"{i_folder_ptb}/{model}.00{member}.{y0_clim}_{ye_clim}.{timeframe}.perturbation.input.%.degC.nc"
            with xr.open_dataset(ds_path) as ds:
                ds=ds.mean(dim='time')
                im_temp.append(ds.temp.plot.imshow(ax=axes[i, 0], add_colorbar=False))
                im_prcp.append(ds.prcp.plot.imshow(ax=axes[i, 1], cmap='BrBG',vmin=-100000, vmax=100000, add_colorbar=False))
                im_hgt.append(ds.hgt.plot.imshow(ax=axes[i, 2], cmap='Greys', add_colorbar=False))
                axes[i, 0].set_ylabel(sample_id)
                

                i+=1

# Add one colorbar per column

fig.colorbar(im_temp[0], ax=axes[:, 0], orientation='horizontal', label='Temperature [°C]', fraction=0.05, pad=-0.25,location='bottom')
fig.colorbar(im_prcp[0], ax=axes[:, 1], orientation='horizontal', label='Precipitation', fraction=0.05,  pad=-0.25,location='bottom')
fig.colorbar(im_hgt[0], ax=axes[:, 2], orientation='horizontal', label='Height [m]', fraction=0.05,  pad=-0.25,location='bottom')

plt.tight_layout()
plt.show()                                
                    
                    
    #%%
    
    
y0_clim = 1985
ye_clim = 2014
y0_cf = 1901
ye_cf = 1985
import cftime
n_models = 3
fig, axes = plt.subplots(n_models, 3, figsize=(10, n_models * 3))
axes = np.array(axes)  # shape (n_models, 3)

# Storage for the mappables (images)
im_temp = []
im_prcp = []
im_hgt = []
members_averages = [1, 4]#3, 4, 6,4]
models_shortlist = ["IPSL-CM6", "NorESM"]#"E3SM", "CESM2", "CNRM", 

for i, (model, member_count) in enumerate(zip(models_shortlist, members_averages)):
    for member in range(member_count):
        if model == "IPSL-CM6" or member >= 1:
            sample_id = f"{model}.00{member}"
            print(sample_id)
            # base_folder_out = f"/Users/magaliponds/OneDrive - Vrije Universiteit Brussel/1. VUB/02. Coding/01. IRRMIP/03. Data/03. Output files/01. Climate data/01. Processed input data/Precipitation/{timeframe}/{model}/{member}"
            # diff_folder_out = f"/Users/magaliponds/OneDrive - Vrije Universiteit Brussel/1. VUB/02. Coding/01. IRRMIP/03. Data/03. Output files/01. Climate data/02. Perturbations/Precipitation/{timeframe}/{model}/{member}"

            # ofile_diff = f"{diff_folder_out}/{model}.pr.DIF.00{member}.{y0_clim}_{ye_clim}_{timeframe}_rel.nc"
            # ofile_irr = f"{base_folder_out}/{model}.pr.IRR.00{member}.{y0_clim}_{ye_clim}_{timeframe}_rel.nc"
            # ofile_noi = f"{base_folder_out}/{model}.pr.NOI.00{member}.{y0_clim}_{ye_clim}_{timeframe}_rel.nc"
            
            # i_folder_abs_p = f"/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/03. Output files/01. Climate data/01. Processed input data/Precipitation/monthly"
            # ds_path_irr = f"{i_folder_abs_p}/{model}/{member}/{model}.PR.IRR.00{member}.{y0_clim}_{ye_clim}_{timeframe}_abs.nc"
            # ds_path_noi = f"{i_folder_abs_p}/{model}/{member}/{model}.PR.NOI.00{member}.{y0_clim}_{ye_clim}_{timeframe}_abs.nc"
            # # with xr.open_dataset(ds_path_irr) as ds:
            #     # ds=ds.mean(dim='time')
            #     # print(ds.pr.max())
            #     # print(ds.pr.min())
            # folder_in = f"/Users/magaliponds/OneDrive - Vrije Universiteit Brussel/1. VUB/02. Coding/01. IRRMIP/03. Data/01. Input files/01. Climate data/{model}/{y0_clim}/"
            # ifile_NOI = f"{folder_in}/{model}.NOI.00{member}.{y0_clim}_{ye_clim}_selparam_monthly_total.nc"
            # ifile_IRR = f"{folder_in}/{model}.IRR.00{member}.{y0_clim}_{ye_clim}_selparam_monthly_total.nc"
            
            #CHECK PERTURBATIONS IN OGGM LOADED
            i_folder_ptb = f"/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/03. Output files/01. Climate data/07. OGGM Perturbation input files/{timeframe}/{model}/{member}"
            ds_path = f"{i_folder_ptb}/{model}.00{member}.{y0_clim}_{ye_clim}.{timeframe}.perturbation.input.%.degC.nc"
            
            #CHECK INDIVIDUAL PERTURBATION FILES
            climate_data_folder_path = f"/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/03. Output files/01. Climate data/04. Perturbation Timeseries/"
            climate_data_file_path_pr = f"{climate_data_folder_path}/Precipitation/{timeframe}/{model}/{member}/REGRID.perturbation.timeseries.PR.{model}.00{member}.{y0_clim}_{ye_clim}_{timeframe}.nc"
            climate_data_file_path_tas = f"{climate_data_folder_path}/Temperature/{timeframe}/{model}/{member}/REGRID.perturbation.timeseries.TEMP.{model}.00{member}.{y0_clim}_{ye_clim}_{timeframe}.nc"

            #CHECK OUTPUTS FROM PROCESSING
            o_folder_clim = f"/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/03. Output files/01. Climate data/07. OGGM Perturbation input files/{timeframe}/{model}/{member}"
            merged_data_out = f"{o_folder_clim}/{model}.00{member}.{y0_clim}_{ye_clim}.{timeframe}.perturbation.input.%.degC_nanzero.nc"
            
            with xr.open_dataset(ds_path, engine='netcdf4') as ds:
                ds = ds.where((ds.lon >= 60) & (ds.lon <= 109) & (
                    ds.lat >= 22) & (ds.lat <= 52), drop=True) 
                binary_nan_map = ds['prcp'].isnull().astype(int)
                binary_data_map = ds['prcp'].notnull().astype(int)
                fig,axes = plt.subplots(4,3, figsize=(12,8), sharex=True, sharey=True)
                axes=axes.flatten()
                for month in range(12):
                    ax=axes[month]
                    month+=1
                    if month<10:
                        month=f"0{month}"
                    # time = np.datetime64(f"1986-{month}-01")
                    nan_mask = ds.sel(time=f'1986-{month}-01')
                    data = np.ma.masked_where(np.isnan(nan_mask['prcp']), nan_mask['prcp'])
                    cmap = plt.get_cmap('BrBG').copy()
                    cmap.set_bad(color='red')  # Set NaNs to black
                    ax.imshow(data.T,  cmap=cmap,  aspect='auto')          
                    # data.plot.imshow(ax=ax, cmap=cmap)#cmap="BrBG")  # white = 1 (NaNs), black = 0 (valid)
                    # binary_data_map.sel(time=time).plot.imshow(ax=ax, cmap="binary")#cmap="BrBG")  # white = 1 (NaNs), black = 0 (valid)
                    # print(sample_id, month, ds.sel(time=time).prcp.min().values)
                plt.tight_layout()
                fig.suptitle(f" {sample_id} Data Availability Map : Red = NaN", fontsize=14, y=1.02)
                plt.show()
                # for year in range(1985,2015):
                #     for month in range(13):
                #         plt.figure()
                #         print(month)
                #         if month>=1:
                #             time = cftime.DatetimeNoLeap(year, month,1)

                #             if month==1 and year ==1985:
                #                 print(time)
                                
                #             elif month >=2 and year ==2015:
                #                 print(time)
                                
                #             else:
                               
                #                   
                #                 plt.show()
                                
                                
                
#%% Plot climate input files on map          
wd_path = f'{folder_path}/03. Modelled perturbation-glacier interactions - R13-15 A+5km2/'
sum_dir = f"{wd_path}/summary"
model ="NorESM"
for member in range(2):
    if member>=1:
        sample_id = f"{model}.00{member}"
        print(sample_id)
        # opath_perturbations = os.path.join(
        #     sum_dir, f'climate_historical_perturbation_{sample_id}.nc')
        opath_gcm_data = os.path.join(
            # sum_dir, 'climate_historical')
            sum_dir, f'gcm_data_perturbed_{sample_id}.nc')
        with xr.open_dataset(opath_gcm_data) as ds:
            has_nan = ds['prcp'].isnull().any(dim='time')  # True where NaNs exist for any time step
            n_rgi_with_nan = has_nan.sum().item()  # Total count of rgi_ids with NaNs
            
            # print(f"{n_rgi_with_nan} out of {ds.rgi_id.size} rgi_ids have NaN values.")
            nan_mask = ds.sel(time=slice(None, '1986-12-31'))
            data = np.ma.masked_where(np.isnan(nan_mask['prcp']), nan_mask['prcp'])

            cmap = plt.get_cmap('BrBG').copy()
            cmap.set_bad(color='red')  # Set NaNs to black
            # Plot heatmap: time on y-axis, rgi_id on x-axis
            plt.figure(figsize=(15, 6))
            plt.imshow(data.T,  cmap=cmap, aspect='auto')            
            plt.xlabel('Time')
            plt.ylabel('rgi_id')
            plt.title('NaN Pattern in temp Time Series (Red = NaN)')
            plt.colorbar(label='Precipitation values [%]')
            plt.tight_layout()
            plt.show()
    
    
    
#%% plot climate data rgi_id vs time matrix

timeframe="monthly"
model="NorESM"
members = [4,3,4,6,1]#4,
y0=1985
ye=2014

for m, model in enumerate([ "CESM2", "E3SM", "NorESM", "CNRM", "IPSL-CM6"]):#"NorESM",
    for member in range(members[m]):
        if model =="IPSL-CM6" or member>=1:
            
            sample_id = f"{model}.00{member}"
            climate_data_folder_path = f"/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/03. Output files/01. Climate data/03. Regridded Perturbations/"
            climate_data_file_path_pr = f"{climate_data_folder_path}/Precipitation/{timeframe}/{model}/{member}/REGRID.{model}.PR.DIF.00{member}.{y0}_{ye}_{timeframe}_rel.nc"
            climate_data_file_path_tas = f"{climate_data_folder_path}/Temperature/{timeframe}/{model}/{member}/REGRID.{model}.TEMP.DIF.00{member}.{y0}_{ye}_{timeframe}_abs.nc"#REGRID.perturbation.timeseries.TEMP.{model}.00{member}.{y0}_{ye}_{timeframe}.nc"
            climate_data_pr = xr.open_dataset(
                climate_data_file_path_pr, engine='h5netcdf').rename({'pr': 'prcp'})
            # climate_data_tas = xr.open_dataset(
            #     climate_data_file_path_tas, engine='h5netcdf').rename({'tas': 'temp'})
            
            # climate_data_pr = climate_data_pr.where((climate_data_pr.lon >= 60) & (climate_data_pr.lon <= 109) & (
            #     climate_data_pr.lat >= 22) & (climate_data_pr.lat <= 52), drop=True)
            
            fig, axes = plt.subplots(3, 4, figsize=(15, 8), sharex=True, sharey=True)
            axes = axes.flatten()
            im = None
            
            for month in range(1): #12
                ax = axes[month]
                m = month + 1
                nan_mask = climate_data_pr.sel(month=m)
                pr_data = nan_mask['prcp'].values
            
                # Mask NaNs for base colormap
                data_masked = np.ma.masked_where(np.isnan(pr_data), pr_data)
                
                cmap = plt.get_cmap('Greys').copy()
                cmap.set_bad(color='red')
                im = ax.imshow(pr_data, cmap=cmap, aspect='auto', vmin=-40, vmax=40)
                
                zero_mask = (pr_data == 0)
                if np.any(zero_mask):
                    rgba_overlay = np.zeros((*pr_data.shape, 4))
                    rgba_overlay[zero_mask] = [0.5, 0, 0.5, 1.0]  # RGB for purple, alpha=1
                    ax.imshow(rgba_overlay, aspect='auto')
                print(f"Number of zero values {sample_id} {month}: {np.sum(zero_mask)}")
                print(f"Number of nan values {sample_id} {month}: {np.isnan(pr_data).sum()}")
                    
                ax.set_title(f'Month {m}')
            
            fig.suptitle(F'Perturbation in precipitation ({sample_id})', fontsize=14)
            fig.colorbar(im, ax=axes, orientation='horizontal', shrink=0.8, label='Perturbation in Precipitation [%]')
            plt.show()
            
#%% Precipitation perturbation


timeframe="monthly"
model="NorESM"
members = [4,3,4,6,1]#4,
y0=1985
ye=2014

fig, axes = plt.subplots(5,3, figsize=(15, 12), sharex=True, sharey=True)
axes = axes.flatten()
counter=0

for m, model in enumerate([ "CESM2", "E3SM", "NorESM", "CNRM", "IPSL-CM6"]):#"NorESM",
    for member in range(members[m]):
        if model =="IPSL-CM6" or member>=1:
            ax=axes[counter]
            sample_id=f"{model}.00{member}"
            
            sample_id = f"{model}.00{member}"
            climate_data_folder_path = f"/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/03. Output files/01. Climate data/03. Regridded Perturbations/"
            climate_data_file_path_pr = f"{climate_data_folder_path}/Precipitation/{timeframe}/{model}/{member}/REGRID.{model}.PR.DIF.00{member}.{y0}_{ye}_{timeframe}_rel.nc"
            climate_data_file_path_tas = f"{climate_data_folder_path}/Temperature/{timeframe}/{model}/{member}/REGRID.{model}.TEMP.DIF.00{member}.{y0}_{ye}_{timeframe}_abs.nc"#REGRID.perturbation.timeseries.TEMP.{model}.00{member}.{y0}_{ye}_{timeframe}.nc"
            climate_data_pr = xr.open_dataset(
                climate_data_file_path_pr, engine='h5netcdf').rename({'pr': 'prcp'})
            # climate_data_tas = xr.open_dataset(
            #     climate_data_file_path_tas, engine='h5netcdf').rename({'tas': 'temp'})
            
            # climate_data_pr = climate_data_pr.where((climate_data_pr.lon >= 60) & (climate_data_pr.lon <= 109) & (
            #     climate_data_pr.lat >= 22) & (climate_data_pr.lat <= 52), drop=True)
            
            
            im = None
            
            for month in range(1): #12
                # ax = axes[month]
                m = month + 1
                nan_mask = climate_data_pr.sel(month=m)
                pr_data = nan_mask['prcp'].values
            
                # Mask NaNs for base colormap
                data_masked = np.ma.masked_where(np.isnan(pr_data), pr_data)
                
                cmap = plt.get_cmap('Greys').copy()
                cmap.set_bad(color='red')
                im = ax.imshow(pr_data, cmap=cmap, aspect='auto', vmin=-40, vmax=40)
                
                zero_mask = (pr_data == 0)
                if np.any(zero_mask):
                    rgba_overlay = np.zeros((*pr_data.shape, 4))
                    rgba_overlay[zero_mask] = [0.5, 0, 0.5, 1.0]  # RGB for purple, alpha=1
                    ax.imshow(rgba_overlay, aspect='auto')
                # print(f"Number of zero values {sample_id} {month}: {np.sum(zero_mask)}")
                # print(f"Number of nan values {sample_id} {month}: {np.isnan(pr_data).sum()}")
                    
                ax.set_title(f'{sample_id}, Month {m}', fontsize=14)
                counter+=1
                
fig.tight_layout()          
# fig.suptitle(F'Data availability per modelxmember combination', fontsize=14, bbox_to_anchor=(1,1))
fig.colorbar(im, ax=axes,label='Perturbation in Precipitation [%]')#, orientation='horizontal', shrink=0.8, label='Perturbation in Precipitation [%]')
plt.show()


#%% Temperature perturbation


timeframe="monthly"
model="NorESM"
members = [4,3,4,6,1]#4,
y0=1985
ye=2014

fig, axes = plt.subplots(5,3, figsize=(15, 12), sharex=True, sharey=True)
axes = axes.flatten()
counter=0

for m, model in enumerate([ "CESM2", "E3SM", "NorESM", "CNRM", "IPSL-CM6"]):#"NorESM",
    for member in range(members[m]):
        if model =="IPSL-CM6" or member>=1:
            ax=axes[counter]
            sample_id=f"{model}.00{member}"
            
            sample_id = f"{model}.00{member}"
            climate_data_folder_path = f"/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/03. Output files/01. Climate data/03. Regridded Perturbations/"
            climate_data_file_path_pr = f"{climate_data_folder_path}/Precipitation/{timeframe}/{model}/{member}/REGRID.{model}.PR.DIF.00{member}.{y0}_{ye}_{timeframe}_rel.nc"
            climate_data_file_path_tas = f"{climate_data_folder_path}/Temperature/{timeframe}/{model}/{member}/REGRID.{model}.TEMP.DIF.00{member}.{y0}_{ye}_{timeframe}_abs.nc"#REGRID.perturbation.timeseries.TEMP.{model}.00{member}.{y0}_{ye}_{timeframe}.nc"
            # climate_data_pr = xr.open_dataset(
                # climate_data_file_path_pr, engine='h5netcdf').rename({'pr': 'prcp'})
            climate_data_tas = xr.open_dataset(climate_data_file_path_tas, engine='h5netcdf').rename({'tas': 'temp'})
            
            # climate_data_pr = climate_data_pr.where((climate_data_pr.lon >= 60) & (climate_data_pr.lon <= 109) & (
            #     climate_data_pr.lat >= 22) & (climate_data_pr.lat <= 52), drop=True)
            
            
            im = None
            
            for month in range(1): #12
                # ax = axes[month]
                m = month + 1
                nan_mask = climate_data_tas.sel(month=m)
                tas_data = nan_mask['temp'].values
            
                # Mask NaNs for base colormap
                data_masked = np.ma.masked_where(np.isnan(pr_data), pr_data)
                
                cmap = plt.get_cmap('RdBu').copy()
                cmap.set_bad(color='grey')
                im = ax.imshow(tas_data, cmap=cmap, aspect='auto', vmin=-2, vmax=2)
                
                zero_mask = (pr_data == 0)
                if np.any(zero_mask):
                    rgba_overlay = np.zeros((*pr_data.shape, 4))
                    rgba_overlay[zero_mask] = [0.5, 0, 0.5, 1.0]  # RGB for purple, alpha=1
                    ax.imshow(rgba_overlay, aspect='auto')
                # print(f"Number of zero values {sample_id} {month}: {np.sum(zero_mask)}")
                # print(f"Number of nan values {sample_id} {month}: {np.isnan(pr_data).sum()}")
                    
                ax.set_title(f'{sample_id}, Month {m}', fontsize=14)
                counter+=1
                
fig.tight_layout()          
# fig.suptitle(F'Data availability per modelxmember combination', fontsize=14, bbox_to_anchor=(1,1))
fig.colorbar(im, ax=axes,label='Perturbation in Temperature [°C]')#, orientation='horizontal', shrink=0.8, label='Perturbation in Precipitation [%]')
plt.show()

#%% Check how many Nans when creating perturbation timeseries

timeframe="monthly"
model="NorESM"
members = [4,1]
y0=1985
ye=2014

for m, model in enumerate(["NorESM", "IPSL-CM6"]):
    for member in range(members[m]):
        if model =="IPSL-CM6" or member>=1:
            
            sample_id = f"{model}.00{member}"
            climate_data_folder_path = f"/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/03. Output files/01. Climate data/04. Perturbation timeseries/"
            climate_data_file_path_pr = f"{climate_data_folder_path}/Precipitation/{timeframe}/{model}/{member}/REGRID.perturbation.timeseries.PR.{model}.00{member}.{y0}_{ye}_{timeframe}.nc"
            climate_data_file_path_tas = f"{climate_data_folder_path}/Temperature/{timeframe}/{model}/{member}/REGRID.perturbation.timeseries.TEMP.{model}.00{member}.{y0}_{ye}_{timeframe}.nc"#REGRID.perturbation.timeseries.TEMP.{model}.00{member}.{y0}_{ye}_{timeframe}.nc"
            # o_folder_clim = f"/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/03. Output files/01. Climate data/07. OGGM Perturbation input files/{timeframe}/{model}/{member}"
            # merged_data_out = f"{o_folder_clim}/{model}.00{member}.{y0}_{ye}.{timeframe}.perturbation.input.%.degC.nc"
            climate_data = xr.open_dataset(
                climate_data_file_path_pr, engine='h5netcdf')#.rename({'pr': 'prcp'})
            # climate_data_tas = xr.open_dataset(
            #     climate_data_file_path_tas, engine='h5netcdf').rename({'tas': 'temp'})
            
            climate_data = climate_data.where((climate_data.lon >= 60) & (climate_data.lon <= 109) & (
                climate_data.lat >= 22) & (climate_data.lat <= 52), drop=True)
            # print("data loaded")
            nan_per_month = climate_data['pr'].isnull().groupby('time.month').sum(dim=['time', 'lon', 'lat'])
            print(sample_id, "nr of nans", nan_per_month)

            # fig, axes = plt.subplots(3, 4, figsize=(15, 8), sharex=True, sharey=True)
            # axes = axes.flatten()
            # im = None
            
            # for month in range(12):
            #     ax = axes[month]
            #     m = month + 1
            #     nan_mask = climate_data_pr.sel(month=m)
            #     pr_data = nan_mask['prcp'].values
            
            #     # Mask NaNs for base colormap
            #     data_masked = np.ma.masked_where(np.isnan(pr_data), pr_data)
                
            #     cmap = plt.get_cmap('Greys').copy()
            #     cmap.set_bad(color='red')
            #     im = ax.imshow(pr_data, cmap=cmap, aspect='auto', vmin=-40, vmax=40)
                
            #     zero_mask = (pr_data == 0)
            #     if np.any(zero_mask):
            #         rgba_overlay = np.zeros((*pr_data.shape, 4))
            #         rgba_overlay[zero_mask] = [0.5, 0, 0.5, 1.0]  # RGB for purple, alpha=1
            #         ax.imshow(rgba_overlay, aspect='auto')
            #     print(f"Number of zero values {sample_id} {month}: {np.sum(zero_mask)}")
                    
            #     ax.set_title(f'Month {m}')
            
            # fig.suptitle(F'Perturbation in precipitation ({sample_id})', fontsize=14)
            # fig.colorbar(im, ax=axes, orientation='horizontal', shrink=0.8, label='Perturbation in Precipitation [%]')
            # plt.show()


   
#%% Compile run output for  A>5km2

# wd_path_pkls = f'{wd_path}/pkls_subset_success/'

# gdirs_3r_a5 = []
# for filename in os.listdir(wd_path_pkls):
#     if filename.endswith('.pkl'):
#         # f'{gdir.rgi_id}.pkl')
#         file_path = os.path.join(wd_path_pkls, filename)
#         with open(file_path, 'rb') as f:
#             gdir = pickle.load(f)
#             gdirs_3r_a5.append(gdir)

# print("loaded gdirs")
# folder_path = '/Users/magaliponds/Documents/00. Programming'
# wd_path = f'{folder_path}/03. Modelled perturbation-glacier interactions - R13-15 A+5km2/'

models = ["NorESM","IPSL-CM6"]
members=[4,1]
for m, model in enumerate(models):
        for member in range(members[m]):
            if member >= 1 or model =="IPSL-CM6":
                sample_id = f"{model}.00{member}"
                print(f"▶️ Starting {sample_id}")

                # Do this once per sample, not per glacier
                
                opath=f'{sum_dir}/gcm_data_perturbed_{sample_id}.nc'
                # df_climate_data = utils.compile_climate_input(gdirs_3r_a5, path=opath, filename='gcm_data', input_filesuffix=f'_perturbed_{sample_id}', use_compression=True)
                # df_stats = utils.compile_glacier_statistics(gdirs, filesuffix=f"_perturbed_{sample_id}")
                print(f"✅ Finished {sample_id}")
                
                
                with xr.open_dataset(opath) as ds:
                    print(ds)
                    nan_mask = ds.sel(time=slice(None, '1986-12-31'))
                    data = np.ma.masked_where(np.isnan(nan_mask['prcp']), nan_mask['prcp'])

                    cmap = plt.get_cmap('BrBG').copy()
                    cmap.set_bad(color='red')  # Set NaNs to black
                    # Plot heatmap: time on y-axis, rgi_id on x-axis
                    plt.figure(figsize=(15, 6))
                    plt.imshow(data.T,  cmap=cmap, aspect='auto')            
                    plt.xlabel('Time')
                    plt.ylabel('rgi_id')
                    plt.title('NaN Pattern in temp Time Series (Red = NaN)')
                    plt.colorbar(label='Precipitation values [%]')
                    plt.tight_layout()
                    plt.show()
                    
                    
