# %% Cell 0: Load data packages
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 15:36:52 2024

@author: magaliponds

This script creates a master dataset from OGGM operations and pre-existing data (Hugonnet, RGI)
for P only and T only (senstivity analysis) simulations


It runs through the following steps
Cell 0: Load packages
Cell 0b: Set base parameters for OGGM
Cell 0c: Load color dictionaries
Cell 1a: Load gdirs_3r from pkl (this includes all glaciers)
Cell 1b: Load gdirs_3r_a1 (this includes all glaciers with Area >1km2) - only succesful subset
Cell 2: Create a dataset with Area, Volume, RGIdate and RGIID (all at RGI date and from RGI dataset)
Cell 3: Prepare data for plotting on map plot (in script 2.7)  
"""

#%% Cell 0: Load packages

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

#%% Cell 0b: Initialize OGGM with the preferred model parameter set up

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

# %% Cell 0c: Load color dictionaries 

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

# %% Cell 1a: Load gdirs_3r from pkl (fastest way to get started)

wd_path_pkls = f'{wd_path}/pkls/'

gdirs_3r = []
for filename in os.listdir(wd_path_pkls):
    if filename.endswith('.pkl'):
        # f'{gdir.rgi_id}.pkl')
        file_path = os.path.join(wd_path_pkls, filename)
        with open(file_path, 'rb') as f:
            gdir = pickle.load(f)
            gdirs_3r.append(gdir)
            

# %% Cell 1b: Load succesfull gdirs_3r from pkl (fastest way to get started)
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



# %% Cell 2: Update the master dataset with B, A, RGI ID and location for all the different sample ids (541x15=8115) and the region name for only PT

master_df = pd.read_csv(
    f"{wd_path}/masters/master_gdirs_r3_a1_rgi_date_A_V_RGIreg.csv")


members = [1, 3, 4, 6, 4]
models_shortlist = ["IPSL-CM6", "E3SM", "CESM2", "CNRM", "NorESM"]

for var in ["P","T"]:
    
    data = []
    data_cf = []
    rgi_ids = []
    labels = []
    rgi_ids_sel=[]
    
    # Iterate over models and members, collecting data for boxplots
    # only take the model shortlist as members are handled separately
    for m, model in enumerate(models_shortlist):
        for member in range(members[m]):
            if member >=1 or model =="IPSL-CM6":
                sample_id = f"{model}.0{member:02d}"  # Ensure leading zeros
                i_path = os.path.join(
                    sum_dir, f'specific_massbalance_mean_{sample_id}_{var}_only.csv')
        
                # Load the CSV file into a DataFrame and convert to xarray
                mb = pd.read_csv(i_path, index_col=0).to_xarray()
                print(sample_id, mb)
                # Collect B values for each model and member
                data.append(mb.B.values)            
        
                # Store RGI IDs only for the first model/member
                if m == 0 and member == 0:
                    rgi_ids.append(mb.rgi_id.values)
        
                labels.append(sample_id)
    i_path_base = os.path.join(sum_dir, 'specific_massbalance_mean_W5E5.000.csv')
    mb_base = pd.read_csv(i_path_base)
    mb_base = mb_base[mb_base['rgi_id'].isin(rgi_ids[0])]
    base_array = np.array(mb_base.B)
    # Convert the list of data into a NumPy array and transpose it
    data_array = np.array(data)
    # Shape: (number of B values, number of models * members)
    reshaped_data = data_array.T
    # Create a DataFrame for the reshaped data
    df = pd.DataFrame(reshaped_data, index=rgi_ids, columns=np.repeat(labels, 1))
    
    df['B_irr'] = base_array
    # df['B_cf'] = reshaped_data_cf
    
    df.rename_axis(index='rgi_id', inplace=True)
    df.reset_index(drop=False, inplace=True)
    
    # Step 1: Melt the DataFrame to get B_noirr values
    df_melted = pd.melt(df, id_vars='rgi_id', value_vars=[col for col in df.columns if col != 'B_irr'],
                        var_name='sample_id', value_name='B_noirr')
    
    # Step 2: Create a DataFrame with repeated B_irr values
    # Keep only the rgi_id and B_irr columns
    b_irr_repeated = df[['rgi_id', 'B_irr']].copy()
    b_irr_repeated = b_irr_repeated.merge(
        df_melted[['rgi_id', 'sample_id']], on='rgi_id')  # Ensure all combinations
    
    # Now merge B_irr with melted DataFrame
    df_complete = pd.merge(df_melted, b_irr_repeated, on=['rgi_id', 'sample_id'])
    df_complete['B_delta_irr'] = df_complete.B_irr-df_complete.B_noirr
    
    # Merge with rgis_complete
    # master_df = master_ds.to_dataframe()
    df_complete = pd.merge(df_complete, master_df, on='rgi_id', how='inner')
    
    
    new_order = ['rgi_id', 'rgi_region', 'rgi_subregion', 'full_name', 'cenlon', 'cenlat', 'rgi_date',
                 'rgi_area_km2', 'rgi_volume_km3', 'sample_id', 'B_noirr', 'B_irr', 'B_delta_irr']
    
    df_complete = df_complete[new_order]
    # print(df_complete[1:5])
    df_complete.to_csv(
        f"{wd_path}/masters/master_gdirs_r3_a1_rgi_date_A_V_RGIreg_B_{var}_only.csv")

#%% Cell 3: Prepare data for plotting on map plot


"""Proc\ess master for map plot """

for var in ["P","T"]:
    # Load datasets
    df = pd.read_csv(
        f"{wd_path}masters/master_gdirs_r3_a1_rgi_date_A_V_RGIreg_B_{var}_only.csv")
    # df = pd.read_csv(
    #     f"{wd_path}masters/master_gdirs_r3_a5_rgi_date_A_V_RGIreg_B_hugo.csv")
    master_ds = df[(~df['sample_id'].str.endswith('0')) |  # Exclude all the model averages ending with 0 except for IPSL
                   (df['sample_id'].str.startswith('IPSL'))]
    
    # Normalize values
    # divide all the B values with 1000 to transform to m w.e. average over 30 yrs
    # master_ds[['B_noirr', 'B_irr', 'B_delta_irr', 'B_cf',  "B_delta_cf"]] /= 1000
    master_ds.loc[:, ['B_noirr', 'B_irr', 'B_delta_irr']] /= 1000
    
    master_ds = master_ds[['rgi_id', 'rgi_region', 'rgi_subregion', 'full_name', 'cenlon', 'cenlat', 'rgi_date',
                           'rgi_area_km2', 'rgi_volume_km3', 'sample_id', 'B_noirr', 'B_irr', 'B_delta_irr']]
    
    # Define custom aggregation functions for grouping over the 11 member data
    aggregation_functions = {
        'rgi_region': 'first',
        'rgi_subregion': 'first',
        'full_name': 'first',
        'cenlon': 'first',
        'cenlat': 'first',
        'rgi_date': 'first',
        'rgi_area_km2': 'first',
        'rgi_volume_km3': 'first',
        'B_noirr': 'mean',
        'B_irr': 'mean',
        'B_delta_irr': 'mean',
        # 'B_cf': 'mean',
        # 'B_delta_cf': 'mean',
        'sample_id': 'first'
    }
    
    
    master_ds_avg = master_ds.groupby(['rgi_id'], as_index=False).agg({
        'B_delta_irr': 'mean',
        'B_noirr': 'mean',
        # lamda is anonmous functions, returns 11 member average
        'sample_id': lambda _: "11 member average",
        # take first value for all columns that are not in the list
        **{col: 'first' for col in master_ds.columns if col not in ['B_noirr', 'B_delta', 'sample_id']}
    })
    
    master_ds_avg.to_csv(
        f"{wd_path}masters/master_lon_lat_rgi_id.csv")
    # Aggregate data for scatter plot
    master_ds_avg['grid_lon'] = np.floor(master_ds_avg['cenlon'])
    master_ds_avg['grid_lat'] = np.floor(master_ds_avg['cenlat'])
    
    
    # Aggregate dataset, area-weighted BDelta Birr and Bnoirr, Sample id is replaced by 11 member average
    aggregated_ds = master_ds_avg.groupby(['grid_lon', 'grid_lat'], as_index=False).agg({
        'B_delta_irr': lambda x: (x * master_ds.loc[x.index, 'rgi_area_km2']).sum() / master_ds.loc[x.index, 'rgi_area_km2'].sum(),
        # 'B_delta_cf': lambda x: (x * master_ds.loc[x.index, 'rgi_area_km2']).sum() / master_ds.loc[x.index, 'rgi_area_km2'].sum(),
        'B_irr': lambda x: (x * master_ds.loc[x.index, 'rgi_area_km2']).sum() / master_ds.loc[x.index, 'rgi_area_km2'].sum(),
        'B_noirr': lambda x: (x * master_ds.loc[x.index, 'rgi_area_km2']).sum() / master_ds.loc[x.index, 'rgi_area_km2'].sum(),
        # 'B_cf': lambda x: (x * master_ds.loc[x.index, 'rgi_area_km2']).sum() / master_ds.loc[x.index, 'rgi_area_km2'].sum(),
        'rgi_area_km2': 'sum',  # Sum for area
        'rgi_volume_km3': 'sum',  # Sum for volume
        # 'sample_id': lambda _: "11 member average",
        # take first value for all columns that are not in the list
        **{col: 'first' for col in master_ds.columns if col not in ['B_noirr', 'B_irr', 'B_delta', 'sample_id', 'rgi_area_km2', 'rgi_volume_km3']}
    })
    
    aggregated_ds.rename(
        columns={'grid_lon': 'lon', 'grid_lat': 'lat'}, inplace=True)
    
    aggregated_ds.to_csv(
        f"{wd_path}masters/complete_master_processed_for_map_plot_{var}_only.csv")





