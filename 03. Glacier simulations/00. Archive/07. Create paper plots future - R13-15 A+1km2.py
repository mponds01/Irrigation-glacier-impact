#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 20:35:51 2025

@author: magaliponds
"""
import os
import sys
function_directory = "/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/src/03. Glacier simulations"
sys.path.append(function_directory)

#%% Cell 0: Load data
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
from oggm.tasks import process_cmip_data
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
import matplotlib.gridspec as gridspec

import matplotlib.lines as mlines
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import sys
from xarray.coding.times import CFDatetimeCoder
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from oggm.graphics import plot_centerlines


# from OGGM_data_processing import process_perturbation_data,custom_process_cmip_data,custom_process_gcm_data

# %% Cell 1: Initialize OGGM with the preferred model parameter set up


folder_path = '/Users/magaliponds/Documents/00. Programming'
wd_path = f'{folder_path}/04. Modelled perturbation-glacier interactions - R13-15 A+1km2/'
# wd_path_fut = f'{folder_path}/03. Modelled perturbation-glacier interactions Future - R13-15 A+1km2/'
os.makedirs(wd_path, exist_ok=True)
# os.makedirs(wd_path_fut, exist_ok=True)
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

#%% Cell 2b: Load gdirs_3r_a1 from pkl

wd_path_pkls = f'{wd_path}/pkls_subset_success/'

gdirs_3r_a1 = []
for filename in os.listdir(wd_path_pkls):
    if filename.endswith('.pkl'):
        # f'{gdir.rgi_id}.pkl')
        file_path = os.path.join(wd_path_pkls, filename)
        with open(file_path, 'rb') as f:
            gdir = pickle.load(f)
            gdirs_3r_a1.append(gdir)

print(len(gdirs_3r_a1))

#%% Cell 2c: Load dicts
   
#
# Your SSP color codes
colors_ssp = {
    'ssp585': '#951b1e',
    'ssp370': '#e71d25',
    'ssp245': '#f79420',
    'ssp126': '#173c66',
    'ssp119': '#00adcf'
}

# Function to convert hex to RGBA with alpha
def hex_to_rgba(hex_color, alpha=1.0):
    return clrs.to_rgba(hex_color, alpha=alpha)

# Compose your full colors dict
colors = {
    "irr": ["black", "#40E0D0"],  # Base colors

    "noi": ["dimgrey", "darkgrey"],  # Base colors

    # Future scenarios: using SSP colors directly
    "irr_fut": [
        colors_ssp['ssp126'],  # SSP126
        colors_ssp['ssp370']   # SSP370
    ],

    # Future scenarios: same SSP colors but with alpha 0.5
    "noi_fut": [
        hex_to_rgba(colors_ssp['ssp126'], 0.5),
        hex_to_rgba(colors_ssp['ssp370'], 0.5)
    ]
}

print(colors)
subregion_names = {
        "13-01": "Tian Shan",
        "13-02": "Pamir",
        "13-03": "Hindu Kush",
        "13-04": "Karakoram",
        "13-05": "West Himalaya",
        "13-06": "Central Himalaya",
        "13-07": "East Himalaya",
        "13-08": "Qiangtang",
        "13-09": "Kunlun",
        "14-01": "Inner Tibet",
        "14-02": "Nyainqentanglha",
        "14-03": "Hengduan Shan",
        "15-01": "Eastern Altun Shan",
        "15-02": "Qilian Shan",
        "15-03": "Eastern Tien Shan",
}

subregion_names_2 = {
        "13-01": "Hissar Alay",
        "13-02": "Pamir",
        "13-03": "West Tien Shan",
        "13-04": "East Tien Shan",
        "13-05": "West Kun Lun",
        "13-06": "East Kun Lun",
        "13-07": "Qilian Shan",
        "13-08": "Inner Tibet",
        "13-09": "Southeast Tibet",
        "14-01": "Hindu Kush",
        "14-02": "Karakoram",
        "14-03": "West Himalaya",
        "15-01": "Central Himalaya",
        "15-02": "East Himalaya",
        "15-03": "Hengduan Shan",
}

               
                    
#%%Cell 3: Plot future volume for 9 rgi ids, irr noi and 2 ssps

members = [4]
models = ["CESM2"]
timeframe = "monthly"
ssps = ["126","370"]#"126",
exp = ["IRR", "NOI"]
y0=2015
ye=2074
factor =1e-9 #conversion m3 to km3 

fig, axes = plt.subplots(3,3, figsize=(10,6))
axes = axes.flatten()

for g, gdir in enumerate(gdirs_3r_a1[1:10]):
    gdir_avg_past = []
    gdir_avg_fut = []
    for m, model in enumerate(models):
        for member in range(members[m]):
            for s, ssp in enumerate(ssps):
                for e, ex in enumerate(exp):
                    if member >= 1: #skip 0
                        input_filesuffix = f"_SSP{ssp}_{ex}.00{member}"
                        if ex =="NOI":
                            out_id_og = f'_perturbed_CESM2.00{member}'
                            out_id = f'_CESM2{input_filesuffix}_noi_bias'

                        else:
                            out_id_og = f'_baseline_W5E5.000'
                            out_id = f'_CESM2{input_filesuffix}_noi_bias'

                            
                        print(out_id)
                        color = colors[ex.lower()][0]
                        color_fut =colors[f"{ex.lower()}_fut"][s]
                        # linestyle = "solid" if year < 2015 else ("dashed" if s == 1 else "dotted")
                        linestyle = "dotted"#"dashed" if  s == 1 else "dotted"
                    
                        ax = axes[g]
                        with xr.open_dataset(gdir.get_filepath('model_diagnostics', filesuffix=out_id_og)) as ds:
                            ds.load()
                            ax.plot(ds.time, ds.volume_m3*factor, color=color)
                            gdir_avg_past.append(ds.volume_m3*factor)
                            
                        with xr.open_dataset(gdir.get_filepath('model_diagnostics', filesuffix=out_id)) as ds:
                            ds.load()
                            ax.plot(ds.time, ds.volume_m3*factor, color=color_fut, linestyle=linestyle)
                            gdir_avg_fut.append(ds.volume_m3*factor)
                        ax.set_title(gdir.rgi_id)
                        if g<=5:
                            ax.set_xticklabels([])
                        ax.axvline(x=2014, color='gray', linestyle='dotted')
    
                        
custom_lines = [
    Line2D([0], [0], color='grey', linestyle='solid', label='Historical, W5E5'),
    mpatches.Patch(color=colors['noi'][0], linestyle='solid', label='NOI'),

    Line2D([0], [0], color='grey', linestyle='dotted', label='SSP127, CESM2'),


    mpatches.Patch( color=colors['irr'][0], linestyle='solid', label='IRR'),
    Line2D([0], [0], color='grey', linestyle='dashed', label='SSP370, CESM2'),

]

# Add to legend
plt.subplots_adjust(hspace=0.4, wspace=0.3)
fig.legend(handles=custom_lines, loc='lower center', ncols =3, bbox_to_anchor=(.50,-0.05)) 
plt.show()      
        
# #%% OUTDATED - Cell 4: Create master for creating subplots - outdated
# exp = ["IRR", "NOI"]

# regions = [13, 14, 15]
# subregions = [9, 3, 3]
# ssps = ["126","370"]#"126",

# region_data = []
# members_averages = [4]
# models_shortlist = ["CESM2"]

        
# # Storage for time series
# subregion_series = {}  # subregion → DataArray[time, member]
# global_series = []     # total average over all subregions per member
# members_all = []       # (model, member_id) pairs to track 14-member average


# master = pd.read_csv(f"{wd_path}/masters/master_lon_lat_rgi_id.csv")
# path_past = os.path.join(sum_dir, f'climate_run_output_baseline_W5E5.000.nc')
# ds_w5e5 = xr.open_dataset(path_past)
# master = master[master["rgi_id"].isin(ds_w5e5.rgi_id.values)]

# initial_volume=ds_w5e5.sel(time="1985")
# initial_volume = initial_volume.volume.sum(dim='rgi_id').values


# # Create containers
# past_records = []
# future_records = []


# for reg, region in enumerate(regions):
#     for sub in range(subregions[reg]):
#         region_id = f"{region}-0{sub+1}"
#         filtered_master = master[master["rgi_subregion"] == region_id].copy()
#         subregion_id = region_id  # used for storage key
#         for e, ex in enumerate(exp):
#             for m, model in enumerate(models_shortlist):
#                 for member in range(members_averages[m]):
#                         if member<1:
#                             continue
#                         if member >= 1: #skip 0
#                             sample_id = f"{model}.00{member:01d}"
                            
                            
#                             if ex =="IRR":
#                                 ds_past = ds_w5e5
                                
#                             else:
#                                 path_past = os.path.join(sum_dir, f'climate_run_output_perturbed_CESM2.00{member}.nc')
#                                 ds_past = xr.open_dataset(path_past, engine='h5netcdf')
#                             ds_past_filtered = ds_past.sel(rgi_id=filtered_master.rgi_id.values)
#                             volume_past = ds_past_filtered.volume.sum(dim='rgi_id')
#                             volume_past = volume_past /initial_volume *100 -100
#                             past_records.append({
#                                 "exp": ex,
#                                 "sample_id": sample_id,
#                                 "subregion": subregion_id,
#                                 "time": ds_past.time.values,
#                                 "volume": volume_past.values
#                             })
                            
#                             for s, ssp in enumerate(ssps):
#                                 input_filesuffix = f"_SSP{ssp}_{ex}.00{member}"
#                                 out_id = f'_CESM2{input_filesuffix}'
#                                 print(out_id)
#                                 input_filesuffix = f"_SSP{ssp}_{ex}.00{member}"
#                                 path_fut = os.path.join(sum_dir, f'climate_run_output{input_filesuffix}_noi_bias.nc')
#                                 ds_fut = xr.open_dataset(path_fut, engine='h5netcdf')
                            
#                                 ds_fut_filtered = ds_fut.sel(rgi_id=filtered_master.rgi_id.values)
                                
#                                 volume_fut = ds_fut_filtered.volume.sum(dim='rgi_id')
#                                 volume_fut = volume_fut /initial_volume *100 -100
    
#                                 # Save to future record
#                                 future_records.append({
#                                     "exp": ex,
#                                     "sample_id": sample_id,
#                                     "ssp": ssp,
#                                     "subregion": subregion_id,
#                                     "time": ds_fut.time.values,
#                                     "volume": volume_fut.values
#                                 })                                
# flat_past = []
# for rec in past_records:
#     for t, v in zip(rec["time"], rec["volume"]):
#         flat_past.append({
#             "exp": rec["exp"],
#             "sample_id": rec["sample_id"],
#             "subregion": rec["subregion"],
#             "time": t,
#             "volume": v
#         })

# df_past = pd.DataFrame(flat_past)
# past_ds = df_past.set_index(["exp", "sample_id", "subregion", "time"]).to_xarray()

# flat_future = []
# for rec in future_records:
#     for t, v in zip(rec["time"], rec["volume"]):
#         flat_future.append({
#             "exp": rec["exp"],
#             "sample_id": rec["sample_id"],
#             "ssp": rec["ssp"],
#             "subregion": rec["subregion"],
#             "time": t,
#             "volume": v
#         })

# df_future = pd.DataFrame(flat_future)
# future_ds = df_future.set_index(["exp", "sample_id", "ssp", "subregion", "time"]).to_xarray()


# past_ds.to_netcdf(f"{wd_path}masters/master_volume_subregion_past.nc")
# future_ds.to_netcdf(f"{wd_path}masters/master_volume_subregion_future.nc")


#%% Cell 4:  Create master for making subplots

# Setup
exp = ["IRR", "NOI"]
regions = [13, 14, 15]
subregions = [9, 3, 3]
ssps = ["126", "370"]
members_averages = [4]
models_shortlist = ["CESM2"]

# Load master and W5E5 baseline
master = pd.read_csv(f"{wd_path}/masters/master_lon_lat_rgi_id.csv")
path_past = os.path.join(sum_dir, 'climate_run_output_baseline_W5E5.000.nc')
ds_w5e5 = xr.open_dataset(path_past)
master = master[master["rgi_id"].isin(ds_w5e5.rgi_id.values) & (master["rgi_id"] != "RGI60-13.48966")] 
initial_volume = ds_w5e5.sel(time="1985").volume.sum(dim='rgi_id').values

# Prepare containers
past_records = []
future_records = []

for reg, region in enumerate(regions):
    for sub in range(subregions[reg]):
        region_id = f"{region}-0{sub+1}"
        filtered_master = master[master["rgi_subregion"] == region_id].copy()
        
        subregion_id = region_id

        for ex in exp:
            volumes_past_all = []
            volumes_future_all = {ssp: [] for ssp in ssps}

            for model in models_shortlist:
                for member in range(members_averages[0]):
                    if member < 1:
                        continue

                    sample_id = f"{model}.00{member}"

                    # Load past dataset
                    if ex == "IRR":
                        ds_past = ds_w5e5
                    else:
                        path_past = os.path.join(sum_dir, f'climate_run_output_perturbed_CESM2.00{member}.nc')
                        ds_past = xr.open_dataset(path_past, engine='netcdf4')

                    ds_past_filtered = ds_past.sel(rgi_id=filtered_master.rgi_id.values)
                    volume_past = ds_past_filtered.volume.sum(dim='rgi_id')
                    # print(ds_past.time.values[0])
                    # print(ds_past.time.values[-1])   
                    # volume_past = volume_past / initial_volume * 100 - 100

                    volumes_past_all.append(volume_past.values)
                    if ex =="NOI" and member==1:
                        print(ds_past_filtered)

                    past_records.append({
                        "exp": ex,
                        "sample_id": sample_id,
                        "subregion": subregion_id,
                        "time": ds_past.time.values,
                        "volume": volume_past.values
                    })
                    # print(ds_past.time.values[0])
                    # print(ds_past.time.values[-1])  

                    for ssp in ssps:
                        # print(ex, ssp, member)

                        # print(ex)
                        input_filesuffix = f"_SSP{ssp}_{ex}.00{member}"
                        # print(input_filesuffix)
                        # path_fut = os.path.join(sum_dir, f'climate_run_output{input_filesuffix}.nc')
                        # if ex == "NOI":
                        path_fut = os.path.join(sum_dir, f'climate_run_output{input_filesuffix}_noi_bias.nc')
                        # else:
                            # path_fut = os.path.join(sum_dir, f'climate_run_output{input_filesuffix}.nc')

                        
                        try:
                            ds_fut = xr.open_dataset(path_fut, engine='netcdf4')
                        except FileNotFoundError:
                            print(f"Missing: {path_fut}")
                            continue
                        ds_fut_filtered = ds_fut.sel(rgi_id=filtered_master.rgi_id.values)
                        volume_fut = ds_fut_filtered.volume.sum(dim='rgi_id')
                        # volume_fut = volume_fut / initial_volume * 100 - 100
                        # print(ds_fut.volume.values[0])
                        # print(ds_fut.volume.values[-1]) 
                        volumes_future_all[ssp].append(volume_fut.values)
                        future_records.append({
                            "exp": ex,
                            "sample_id": sample_id,
                            "ssp": ssp,
                            "subregion": subregion_id,
                            "time": ds_fut.time.values,
                            "volume": volume_fut.values
                        })
                        # print(ds_fut.time.values[0])
                        # print(ds_fut.time.values[-1])     
                   
            # Add 3-member average (PAST)
            if volumes_past_all:
                volume_past_avg = np.mean(volumes_past_all, axis=0)
                past_records.append({
                    "exp": ex,
                    "sample_id": "3-member-avg",
                    "subregion": subregion_id,
                    "time": ds_past.time.values,
                    "volume": volume_past_avg
                })

            # Add 3-member average (FUTURE)
            for ssp in ssps:
                if volumes_future_all[ssp]:
                    volume_fut_avg = np.mean(volumes_future_all[ssp], axis=0)
                    future_records.append({
                        "exp": ex,
                        "sample_id": "3-member-avg",
                        "ssp": ssp,
                        "subregion": subregion_id,
                        "time": ds_fut.time.values,
                        "volume": volume_fut_avg
                    })
                    



# Convert to xarray
flat_past = []
for rec in past_records:
    for t, v in zip(rec["time"], rec["volume"]):
        flat_past.append({
            "exp": rec["exp"],
            "sample_id": rec["sample_id"],
            "subregion": rec["subregion"],
            "time": t,
            "volume": v
        })
df_past = pd.DataFrame(flat_past)
past_ds = df_past.set_index(["exp", "sample_id", "subregion", "time"]).to_xarray()

flat_future = []
for rec in future_records:
    for t, v in zip(rec["time"], rec["volume"]):
        flat_future.append({
            "exp": rec["exp"],
            "sample_id": rec["sample_id"],
            "ssp": rec["ssp"],
            "subregion": rec["subregion"],
            "time": t,
            "volume": v
        })
df_future = pd.DataFrame(flat_future)
future_ds = df_future.set_index(["exp", "sample_id", "ssp", "subregion", "time"]).to_xarray()

past_ds.to_netcdf(f"{wd_path}masters/master_volume_subregion_past.nc")
future_ds.to_netcdf(f"{wd_path}masters/master_volume_subregion_future_noi_bias.nc")
# future_ds.to_netcdf(f"{wd_path}masters/master_volume_subregion_future.nc")

#%% Cell 4b: Update master including total volume (HMA overall)

future_records_total = []
for e,ex in enumerate(["NOI","IRR"]):
    for ssp in ["126", "370"]:
            future_total_mean = future_ds.sel(exp=ex, sample_id="3-member-avg", ssp=ssp).sum(dim='subregion')
            future_records_total.append({
                "exp": ex,
                "sample_id": "3-member-avg",
                "ssp": ssp,
                "subregion": "total",
                "time": future_total_mean.time.values,
                "volume": future_total_mean.volume.values
            })
            

                    
flat_future_total = []
for rec in future_records_total:
    for t, v in zip(rec["time"], rec["volume"]):
        
        flat_future_total.append({
            "exp": rec["exp"],
            "sample_id": rec["sample_id"],
            "ssp": rec["ssp"],
            "subregion": rec["subregion"],
            "time": t,
            "volume": v
        })
ds_future_total = pd.DataFrame(flat_future_total).set_index(["exp", "sample_id", "ssp", "subregion", "time"]).to_xarray()

ds_with_total = xr.concat([future_ds, ds_future_total], dim='subregion')
ds_with_total.to_netcdf(f"{wd_path}masters/master_volume_subregion_future_noi_bias_incl_total.nc")
# Save to disk




#%% Cell 5: Create plots by subregion and overall (mosaic)  
 
regions = [13, 14, 15]
subregions = [9, 3, 3]

fig, ax = plt.subplots(figsize=(15,10))  # create a new figure

members_averages = [4]
models_shortlist = [ "CESM2"]

# define the variables for p;lotting

variable_names = ["Volume"]
variable_axes = ["Volume compared to 1985 All Forcings scenario [%]"]
use_multiprocessing = False

plt.rcParams.update({'font.size': 12})

#create a figure consisting of 1 large plot and several smaller ones for every subregion
fig = plt.figure(figsize=(15,8))
gs = gridspec.GridSpec(4, 6, figure=fig)  # Adjust the grid size as needed
ax_big = fig.add_subplot(gs[:3, :3])  # Spanning first 2 rows and 3 columns
small_axes = []
positions = [
    (0, 3), (0, 4), (0, 5), (1, 3), (1, 4), (1, 5),  # First row right side
    (2, 3),(2, 4),(2, 5),  # Third row
    (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5)  # Fourth row
]
axes_dict = {}  # Dictionary to store axes for later use
resp_values_dict={}
linestyles = ['solid', 'dashed']
past_ds = xr.open_dataset(f"{wd_path}masters/master_volume_subregion_past.nc")
future_ds = xr.open_dataset(f"{wd_path}masters/master_volume_subregion_future_noi_bias.nc")
# future_ds = xr.open_dataset(f"{wd_path}masters/master_volume_subregion_future.nc")
initial_vol = past_ds.sel(time=1985).volume
initial_volume_big=initial_vol.sel(exp="IRR").sel(sample_id="CESM2.001").sum(dim='subregion')
p=0
for reg, region in enumerate(regions):
    for sub in range(subregions[reg]):
        region_id = f"{region}-0{sub+1}"
        filtered_master = master[master["rgi_subregion"] == region_id].copy()
        subregion_id = region_id  # used for storage key
        initial_volume=initial_vol.sel(subregion=region_id).sel(exp="IRR").sel(sample_id="CESM2.001")
        pos = positions[p]
        ax = fig.add_subplot(gs[pos])
        ax.set_ylim(-100,10)
        ax.set_title(region_id)
        axes_dict[region_id] = ax
        # if p not in {0,3,6,9}:
        ax.set_yticks([])
        # if p <9:
        ax.set_xticks([])
        p+=1
        subregion_title = subregion_names_2[region_id]
        axes_dict[region_id].set_title(subregion_title, fontweight="bold", bbox=dict(
            facecolor='white', edgecolor='none', pad=1), fontsize=14)
        
        for time in ["past","future"]:
            for e, ex in enumerate(exp):
                color = colors[ex.lower()][0]
                if ex =="IRR" and time=="past":
                    axes_dict[region_id].plot(past_ds.time, past_ds.volume.sel(subregion=region_id).sel(exp=ex).sel(sample_id="CESM2.001")/initial_volume*100-100, linestyle='solid', color=color)
                    if region_id=="13-01": 
                       ax_big.plot(past_ds.time, past_ds.volume.sel(exp=ex).sel(sample_id="CESM2.001").sum(dim='subregion')/initial_volume_big*100-100, linestyle='solid', color=color, label="Historic W5E5 individual member")
                
                       
                else:
                    for m, model in enumerate(models_shortlist):
                        for member in range(members_averages[m]):
                            if member<1:
                                continue
                            if member >= 1: #skip 0
                                sample_id = f"{model}.00{member:01d}"
                                print(sample_id)
                                
                                if time=="past":
                                    axes_dict[region_id].plot(past_ds.time, past_ds.volume.sel(subregion=region_id).sel(exp='NOI').sel(sample_id=sample_id)/initial_volume*100-100, linestyle='dotted', color=color)
                                    if member==1:
                                        label="Historic NoIrr individual member"
                                    else:
                                        label='_nolegend_'
                                    
                                        ax_big.plot(past_ds.time, past_ds.volume.sel(exp='NOI').sel(sample_id=sample_id).sum(dim='subregion')/initial_volume_big*100-100,
                                               color=color, linewidth=2, linestyle='dotted', label=label)  
                                else:
                                    for (s,ssp) in enumerate(ssps):
                                        if member==1 and region_id=="13-01":
                                            label= f'Future {ex.lower()} SSP{ssp} individual member'
                                        else:
                                            label='_nolegend_'
                                        color_fut = colors[f"{ex.lower()}_fut"][s]

                                        axes_dict[region_id].plot(future_ds.time, future_ds.volume.sel(subregion=region_id).sel(exp=ex).sel(sample_id=sample_id).sel(ssp=ssp)/initial_volume*100-100, linestyle='dashed', color=color_fut, lw=1)
                                        ax_big.plot(future_ds.time, future_ds.volume.sel(exp=ex).sel(sample_id=sample_id).sel(ssp=ssp).sum(dim='subregion')/initial_volume_big*100-100, linestyle='dashed', color=color_fut, lw=1, label=label)
                                
                    if time=="past":            
                        std = past_ds.volume.sel(subregion=region_id).sel(exp='NOI').std(dim='sample_id')/initial_volume*100-100
                        mean = past_ds.volume.sel(subregion=region_id).sel(exp='NOI').mean(dim='sample_id')/initial_volume*100-100
                        
                        axes_dict[region_id].plot(past_ds.time, mean, linestyle='solid', color=color, lw=3)
                        # axes_dict[region_id].fill_between(past_ds.time, mean-std, mean+std, color=color, alpha=0.2)
                    else:
                        for (s,ssp) in enumerate(ssps):
                            print(ex)
                            color_fut = colors[f"{ex.lower()}_fut"][s]
                            if s==0:
                                linestyle=':'
                                marker='>'
                            else:
                                linestyle='--'
                                marker='o'

                            std = future_ds.volume.sel(subregion=region_id).sel(exp=ex).sel(ssp=ssp).std(dim='sample_id')/initial_volume*100-100
                            mean = future_ds.volume.sel(subregion=region_id).sel(exp=ex).sel(ssp=ssp).mean(dim='sample_id')/initial_volume*100-100
                            
                            axes_dict[region_id].plot(future_ds.time, mean, linestyle=linestyle, color=color_fut, lw=3)
                            # axes_dict[region_id].fill_between(future_ds.time, mean-std, mean+std, color=color_fut, alpha=0.2)
                            
                        
color=colors["noi"][0]    
std_big = (past_ds.volume.sel(exp='NOI').sum(dim='subregion').std(dim='sample_id')/initial_volume_big)*100-100
sum_big = (past_ds.volume.sel(exp='NOI').sel(sample_id='3-member-avg').sum(dim='subregion')/initial_volume_big)*100-100
ax_big.plot(past_ds.time, sum_big, linestyle='solid', color=color, label="Historic NoIrr 3-member average")
# ax_big.fill_between(past_ds.time, sum_big-std_big,sum_big+std_big, color=color, alpha=0.2)

for (s,ssp) in enumerate(ssps):
    for (e,ex) in enumerate(exp):
        if s==0:
            linestyle=':'
            marker='>'
        else:
            linestyle='--'
            marker='o'
        color=colors[f"{ex.lower()}_fut"][s]
        std_big = (future_ds.volume.sel(exp=ex).sel(ssp=ssp).sum(dim='subregion').std(dim='sample_id')/initial_volume_big)*100-100
        sum_big = (future_ds.volume.sel(exp=ex).sel(ssp=ssp).sel(sample_id='3-member-avg').sum(dim='subregion')/initial_volume_big)*100-100
        ax_big.plot(future_ds.time, sum_big,  color=color, label=f"Future {ex} SSP{ssp} 3-member average", ls=linestyle, linewidth=4)
        # ax_big.fill_between(future_ds.time, sum_big-std_big,sum_big+std_big, color=color, alpha=0.2)

ax_big.set_title("High Mountain Asia", fontweight="bold", bbox=dict(
    facecolor='white', edgecolor='none', pad=1), fontsize=14)   
ax_big.set_ylim(-100,10)
# ax_big.set_yticks([])
plt.subplots_adjust(hspace=0.5)#, wspace=0.2)
ax_big.legend()             
           

                
                 #loop trhough all the different subplots
            
    #         rgi_ids_region = [rgi_id for rgi_id in rgi_ids_test if rgi_id in subregion_mask]
            
    #         #plot baseline data per region
    #         baseline = baseline_all.where(
    #             baseline_all.rgi_id.isin(rgi_ids_region), drop=True) #filter baseline to region

            
    #         axes_dict[region_id].plot(baseline["time"], (baseline['volume'].sum(dim="rgi_id") * factors)/resp_values_dict[region_id]*100,
    #                 label=legendtext, color=colors[f"irr"][0], linewidth=2, zorder=15, linestyle=linestyles[f])
    #         #define files for climate runs for comitted and normal run 
            
            
           


    
    # ax_big.fill_between(climate_run_output_noirr["time"].values, min_values_noirr_all, max_values_noirr_all,
    #                 color=colors[f"{run_type}"][f], alpha=0.3, label=labeltext_range, zorder=16)
    # ax_big.axhline(100, color='black', linestyle='--',
    #             linewidth=1, zorder=1)  # Dashed line at 0
    # # Set labels and title for the combined plot
    # ax_big.set_ylabel("∆Volume compared to 1985 All Forcings [%]", fontweight='bold')
    # ax_big.set_xlabel("Time [year]")
    # ax_big.set_title(f"All Regions ", fontweight="bold", bbox=dict(
    #     facecolor='white', edgecolor='none', pad=1), fontsize=14)
    # ax_big.set_ylim(0,150)

    # # Adjust the legend
    # handles, labels = ax_big.get_legend_handles_labels()
    # ax_big.legend(handles, labels,
    #            ncol=2)
    
# plt.tight_layout()
# plt.subplots_adjust(hspace=0.4, wspace=0.2)
# plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0)
# # specify and create a folder for saving the data (if it doesn't exists already) and save the plot
# o_folder_data = f"/Users/magaliponds/OneDrive - Vrije Universiteit Brussel/1. VUB/02. Coding/01. IRRMIP/04. Figures/02. OGGM simulations/01. Modelled output/3r_a5/1. Volume/00. Combined"
# os.makedirs(o_folder_data, exist_ok=True)
# # o_file_name = f"{o_folder_data}/1985_2014.{timeframe}.delta.{variable_names[v]}.comitted_cst_test.png"
# plt.savefig(o_file_name, bbox_inches='tight')
            
# plt.show()       
        
        
 #%% Cell 6: Create plots by SSP       
 
exp = ["IRR", "NOI"]
regions = [13, 14, 15]
subregions = [9, 3, 3]
ssps = ["126", "370"]
members_averages = [4]
models_shortlist = ["CESM2"]

fig, axes = plt.subplots(1,2, figsize=(12,7))#, sharey=True)  # create a new figure
axes = axes.flatten()
members_averages = [4]
models_shortlist = [ "CESM2"]

# define the variables for p;lotting

variable_names = ["Volume"]
variable_axes = ["Volume compared to 1985 All Forcings scenario [%]"]
use_multiprocessing = False

plt.rcParams.update({'font.size': 12})

#create a figure consisting of 1 large plot and several smaller ones for every subregion
# gs = gridspec.GridSpec(4, 6, figure=fig)  # Adjust the grid size as needed
axes_dict = {}  # Dictionary to store axes for later use
resp_values_dict={}
linestyles = ['solid', 'dashed']
past_ds = xr.open_dataset(f"{wd_path}masters/master_volume_subregion_past.nc")
# future_ds = xr.open_dataset(f"{wd_path}masters/master_volume_subregion_future.nc")
future_ds = xr.open_dataset(f"{wd_path}masters/master_volume_subregion_future_noi_bias.nc")
initial_vol = past_ds.sel(time=1985).volume
initial_volume_big=initial_vol.sel(exp="IRR").sel(sample_id="CESM2.001").sum(dim='subregion')
p=0

p+=1

#Plot the data per member
for time in ["past","future"]:
    for e, ex in enumerate(exp):
        color = colors[ex.lower()][0]
        if ex =="IRR" and time=="past":
            # axes_dict[region_id].plot(past_ds.time, past_ds.volume.sel(subregion=region_id).sel(exp=ex).sel(sample_id="CESM2.001")/initial_volume*100-100, linestyle='solid', color=color)
            #past data plot irr
            axes[0].plot(past_ds.time, past_ds.volume.sel(exp=ex).sel(sample_id="CESM2.001").sum(dim='subregion')/initial_volume_big*100-100, linestyle='solid', color=color, label="Historic W5E5 individual member", lw=3)
            axes[1].plot(past_ds.time, past_ds.volume.sel(exp=ex).sel(sample_id="CESM2.001").sum(dim='subregion')/initial_volume_big*100-100, linestyle='solid', color=color, label="Historic W5E5 individual member", lw=3)
        
               
        else:
            #past data plot noirr
            for m, model in enumerate(models_shortlist):
                for member in range(members_averages[m]):
                    if member<1:
                        continue
                    if member >= 1: #skip 0
                        sample_id = f"{model}.00{member:01d}"
                        print(sample_id)
                        
                        if time=="past":
                            # axes_dict[region_id].plot(past_ds.time, past_ds.volume.sel(subregion=region_id).sel(exp='NOI').sel(sample_id=sample_id)/initial_volume*100-100, linestyle='dotted', color=color)
                            if member==1:
                                label="Historic NoIrr individual member"
                            else:
                                label='_nolegend_'
                            
                            axes[0].plot(past_ds.time, past_ds.volume.sel(exp='NOI').sel(sample_id=sample_id).sum(dim='subregion')/initial_volume_big*100-100,
                                   color=color, lw=1, linestyle='dotted', label=label)  
                            axes[1].plot(past_ds.time, past_ds.volume.sel(exp='NOI').sel(sample_id=sample_id).sum(dim='subregion')/initial_volume_big*100-100,
                                       color=color, lw=1, linestyle='dotted', label=label)  
                        else:
                            for (s,ssp) in enumerate(ssps):
                                ax = axes[s]
                                if member==1 and region_id=="13-01":
                                    label= f'Future {ex.lower()} SSP{ssp} individual member'
                                else:
                                    label='_nolegend_'
                                color_fut = colors[f"{ex.lower()}_fut"][s]

                                # axes_dict[region_id].plot(future_ds.time, future_ds.volume.sel(subregion=region_id).sel(exp=ex).sel(sample_id=sample_id).sel(ssp=ssp)/initial_volume*100-100, linestyle='dashed', color=color_fut, lw=1)
                                volume_evolution=future_ds.volume.sel(exp=ex).sel(sample_id=sample_id).sel(ssp=ssp).sum(dim='subregion')/initial_volume_big*100-100
                                ax.plot(future_ds.time, volume_evolution, linestyle='dashed', color=color_fut, lw=1, label=label)
                                
                        
color=colors["noi"][0]    
std_big = (past_ds.volume.sel(exp='NOI').sum(dim='subregion').std(dim='sample_id')/initial_volume_big)*100-100
sum_big = (past_ds.volume.sel(exp='NOI').sel(sample_id='3-member-avg').sum(dim='subregion')/initial_volume_big)*100-100
axes[0].plot(past_ds.time, sum_big, linestyle='solid', color=color, label="Historic NoIrr 3-member average", lw=3)
axes[1].plot(past_ds.time, sum_big, linestyle='solid', color=color, label="Historic NoIrr 3-member average", lw=3)
# ax_big.fill_between(past_ds.time, sum_big-std_big,sum_big+std_big, color=color, alpha=0.2)
annotations =np.zeros((2,2))

#plot future data
for (s,ssp) in enumerate(ssps):
    ax = axes[s]
    for (e,ex) in enumerate(exp):
        if s==0: #allocate linesyles and markers
            linestyle='--'
            marker='>'
        else:
            linestyle='--'
            marker='o'
        color=colors[f"{ex.lower()}_fut"][s]
        std_big = (future_ds.volume.sel(exp=ex).sel(ssp=ssp).sum(dim='subregion').std(dim='sample_id')/initial_volume_big)*100-100
        sum_big = (future_ds.volume.sel(exp=ex).sel(ssp=ssp).sel(sample_id='3-member-avg').sum(dim='subregion')/initial_volume_big)*100-100
        ax.plot(future_ds.time, sum_big,  color=color, label=f"Future {ex} SSP 3-member average", ls=linestyle, linewidth=3)
        annotation=np.round(sum_big[-1].values)
        annotations[s,e] = annotation
        
        
        # ax_big.fill_between(future_ds.time, sum_big-std_big,sum_big+std_big, color=color, alpha=0.2)
        
        ax.set_title(f"SSP{ssp}", bbox=dict(
            facecolor='white', edgecolor='none', pad=1), fontsize=14)   
        ax.set_ylim(-65,10)
        ax.set_xlim(1985,2075) 
    
#include annotations
for (s,ssp) in enumerate(ssps):
    # ax = axes[s]
    for (e,ex) in enumerate(exp):    
        color=colors[f"{ex.lower()}_fut"][s]  
        if ex == "IRR" or ssp=="370":
            offset = 0.5
        else:
            offset= - 2
        for line in range(4):
            annotation = annotations.flatten()[line]
            axes[s].axhline(annotation, linestyle='--', color='k', lw=1)
        annotation = annotations[s,e]
        axes[s].text(1990,annotation+offset, f"{ex} SSP{ssp} ∆V={annotation}%",color=color, fontweight='bold')


axes[1].set_yticks([])
# ax_big.set_yticks([])
plt.subplots_adjust(hspace=0.5, wspace=0.05)#, wspace=0.2)
axes[0].set_ylabel("Volume change (%, vs. 1985 historic")

legend_patches = [mpatches.Patch(color=colors['irr'][0], label='Historical (W5E5)'),
                  mpatches.Patch(color=colors['noi_fut'][0], label='Future NoIrr SSP126'),
                    Line2D([0], [0], color='grey', linestyle='dashed', linewidth=3, label=f'Future, 11-member mean'),
                    
                    mpatches.Patch(color=colors['noi'][0], label='Historical NoIrr'),
                    mpatches.Patch(color=colors['noi_fut'][1], label='Future NoIrr SSP370'),
                    Line2D([0], [0], color='grey', linestyle='solid', linewidth=3, label=f'Historic, 11-member mean'),
                    
                    mpatches.Patch(color=colors['irr_fut'][0], label='Future (CESM2) SSP126'),
                    Line2D([0], [0], color='grey', linestyle='dotted', linewidth=3, label=f'individual member mean'),
                    Line2D([0], [0], color='grey', linestyle='solid', linewidth=3, label=f'Historic, 11-member mean'),
                    
                    mpatches.Patch(color=colors['irr_fut'][1], label='Future (CESM2) SSP370'),
                    Line2D([0], [0], color='grey', linestyle='dotted', linewidth=3, label=f'individual member mean'),
                    
                    
                         ]
# plt.tight_layout()
fig.legend(handles=legend_patches, loc='lower center', ncols=4, bbox_to_anchor=(0.5,-0.05), )#,
          # bbox_to_anchor=(0.512, 0.96), ncols=5, fontsize=12,columnspacing=1.5)
# fig.subplots_adjust(right=0.97)


#%% Cell 7: Create overview plot volume change total, all SSPs and ∆V annotation (key figure) 
exp = ["IRR", "NOI"]
regions = [13, 14, 15]
subregions = [9, 3, 3]
ssps = ["126", "370"]
members_averages = [4]
models_shortlist = ["CESM2"]

fig, axes = plt.subplots(figsize=(12,7), sharey=True )  # create a new figure

# define the variables for p;lotting

variable_names = ["Volume"]
variable_axes = ["Volume compared to 1985 All Forcings scenario [%]"]
use_multiprocessing = False

plt.rcParams.update({'font.size': 12})

#create a figure consisting of 1 large plot and several smaller ones for every subregion
# gs = gridspec.GridSpec(4, 6, figure=fig)  # Adjust the grid size as needed
axes_dict = {}  # Dictionary to store axes for later use
resp_values_dict={}
linestyles = ['solid', 'dashed']
past_ds = xr.open_dataset(f"{wd_path}masters/master_volume_subregion_past.nc")
# future_ds = xr.open_dataset(f"{wd_path}masters/master_volume_subregion_future.nc")
future_ds = xr.open_dataset(f"{wd_path}masters/master_volume_subregion_future_noi_bias.nc")
initial_vol = past_ds.sel(time=1985).volume
initial_volume_big=initial_vol.sel(exp="IRR").sel(sample_id="CESM2.001").sum(dim='subregion')
p=0

p+=1

for time in ["past","future"]:
    for e, ex in enumerate(exp):
        color = colors[ex.lower()][0]
        if ex=="IRR":
            ls="solid"
        else:
            ls="dashed"
        if ex =="IRR" and time=="past":
            # axes_dict[region_id].plot(past_ds.time, past_ds.volume.sel(subregion=region_id).sel(exp=ex).sel(sample_id="CESM2.001")/initial_volume*100-100, linestyle='solid', color=color)
            axes.plot(past_ds.time, past_ds.volume.sel(exp=ex).sel(sample_id="CESM2.001").sum(dim='subregion')/initial_volume_big*100-100, linestyle=ls, color=color, label="Historic W5E5 individual member", lw=2)
            axes.plot(past_ds.time, past_ds.volume.sel(exp=ex).sel(sample_id="CESM2.001").sum(dim='subregion')/initial_volume_big*100-100, linestyle=ls, color=color, label="Historic W5E5 individual member", lw=2)
        
               
        else:
            volume_evolution_data=[]
            for m, model in enumerate(models_shortlist):
                for member in range(members_averages[m]):
                    if member<1:
                        continue
                    if member >= 1: #skip 0
                        sample_id = f"{model}.00{member:01d}"
                        print(sample_id)
                        
                        for (s, ssp) in enumerate(ssps):
                            ax = axes
                            color_fut = colors[f"{ex.lower()}_fut"][s]
                            volume_evolution_data = []  # reset per SSP
                    
                            for m, model in enumerate(models_shortlist):
                                for member in range(members_averages[m]):
                                    if member < 1:
                                        continue
                                    sample_id = f"{model}.00{member:01d}"
                                    
                                    if s==0 and member==1 and time=="past":
                                        # axes_dict[region_id].plot(past_ds.time, past_ds.volume.sel(subregion=region_id).sel(exp='NOI').sel(sample_id=sample_id)/initial_volume*100-100, linestyle='dotted', color=color)
                                        if member==1:
                                            label="Historic NoIrr individual member"
                                        else:
                                            label='_nolegend_'
                                        
                                        axes.plot(past_ds.time, past_ds.volume.sel(exp='NOI').sel(sample_id=sample_id).sum(dim='subregion')/initial_volume_big*100-100,
                                               color=color, lw=1, linestyle='dotted', label=label)  
                    
                                    try:
                                        volume_evolution = future_ds.volume.sel(exp=ex).sel(sample_id=sample_id).sel(ssp=ssp).sum(dim='subregion') / initial_volume_big * 100 - 100
                                        volume_evolution_data.append(volume_evolution)
                                        
                                        # Plot individual lines if needed
                                        if member == 1: #and region_id == "13-01":
                                            label = f'Future {ex.lower()} SSP-{ssp} individual member'
                                        else:
                                            label = '_nolegend_'
                                        # ax.plot(future_ds.time, volume_evolution, linestyle='dashed', color=color_fut, lw=1, label=label)
                    
                                    except KeyError:
                                        print(f"Data missing for {sample_id}, SSP{ssp}")
                    
                            # Compute min/max across members
                            if volume_evolution_data:
                                # Convert to DataArray for easier handling
                                volume_stack = xr.concat(volume_evolution_data, dim='member')
                                min_vol = volume_stack.min(dim='member')
                                max_vol = volume_stack.max(dim='member')
                    
                                ax.fill_between(
                                    future_ds.time,
                                    min_vol,
                                    max_vol,
                                    color=color_fut,
                                    alpha=0.1,
                                    label=f'SSP{ssp} envelope'
                                )

            
                                
                        
color=colors["noi"][0]    
std_big = (past_ds.volume.sel(exp='NOI').sum(dim='subregion').std(dim='sample_id')/initial_volume_big)*100-100
sum_big = (past_ds.volume.sel(exp='NOI').sel(sample_id='3-member-avg').sum(dim='subregion')/initial_volume_big)*100-100
min_big = (past_ds.volume.sel(exp='NOI').sum(dim='subregion').min(dim='sample_id')/initial_volume_big)*100-100
max_big = (past_ds.volume.sel(exp='NOI').sum(dim='subregion').max(dim='sample_id')/initial_volume_big)*100-100

axes.plot(past_ds.time, sum_big, linestyle='dashed', color=color, label="Historic NoIrr 3-member average", lw=3)
axes.plot(past_ds.time, sum_big, linestyle='dashed', color=color, label="Historic NoIrr 3-member average", lw=3)
axes.fill_between(past_ds.time, min_big,max_big, color=color, alpha=0.2)
# axes.fill_between(past_ds.time, sum_big-std_big,sum_big+std_big, color=color, alpha=0.2)
annotations =np.zeros((2,2))
ssp_anno=['1.26', '3.70']
for (s,ssp) in enumerate(ssps):
    
    ax = axes
    for (e,ex) in enumerate(exp):
        if ex=="IRR":
            ls="solid"
        else:
            ls="dashed"
            
        if s==0:
            marker='>'
        else:
            marker='o'
        color=colors[f"{ex.lower()}_fut"][s]
        std_big = (future_ds.volume.sel(exp=ex).sel(ssp=ssp).sum(dim='subregion').std(dim='sample_id')/initial_volume_big)*100-100
        sum_big = (future_ds.volume.sel(exp=ex).sel(ssp=ssp).sel(sample_id='3-member-avg').sum(dim='subregion')/initial_volume_big)*100-100
        ax.plot(future_ds.time, sum_big,  color=color, label=f"Future {ex} SSP 3-member average", ls=ls, linewidth=3)
        annotation=np.round(sum_big[-1].values, decimals=2)
        annotations[s,e] = annotation
        
        
 
greys = ['lightgrey', 'grey']
diffs=[-7,-8]
for (s,ssp) in enumerate(ssps):
    print(s)
    # if s==0:

    x = 2077+s*3-1 # or an integer like 2045 if x is in years
    y0 = annotations[s, 0]
    y1 = annotations[s, 1]
    
    # Add vertical line
    axes.add_line(mlines.Line2D([x, x], [y0, y1], color='black', lw=1))
    
    # Add horizontal caps
    cap_width = 1  # adjust width depending on your x-axis scale
    axes.add_line(mlines.Line2D([x - cap_width, x + cap_width], [y0, y0], color='black', lw=1))
    axes.add_line(mlines.Line2D([x - cap_width, x + cap_width], [y1, y1], color='black', lw=1))
    
    mid_y = (y0 + y1) / 2
    x_offset = 0.5  # or use +3 for integers
    axes.text(
        x + x_offset,  # to the right
        mid_y,         # middle of whisker
        fr'$∆V_{{\mathrm{{Irr,SSP{ssp_anno[s]}}}}}={diffs[s]}\%$',
        va='center',
        ha='left',
        fontsize=12,
        fontweight='bold'
    )

    # ax = axes[s]
    for (e,ex) in enumerate(exp):  
        
        
        color=colors[f"{ex.lower()}_fut"][s]  
        if ex == "IRR" or ssp=="370":
            offset = -2#0.5
        else:
            offset= 0.5#- 2
        for line in range(4):
            annotation = annotations.flatten()[line]
            axes.axhline(annotation, linestyle='--', color='black', lw=0.5,zorder=0)
        annotation = round(annotations[s,e],1)
        axes.text(1990,annotation+offset, f"{ex} SSP-{ssp_anno[s]} ∆V={annotation}%",color=color, fontweight='bold')

# axes[1].set_yticks([])
desired_years = [1985, 2015, 2045, 2075]
axes.set_xlim(1985,2100)
axes.set_ylim(-72,4)

# Apply to the axis
axes.set_xticks(desired_years)
axes.set_xticklabels([str(year) for year in desired_years])
# ax_big.set_yticks([])
plt.subplots_adjust(hspace=0.5, wspace=0.05)#, wspace=0.2)
axes.set_ylabel("Volume change (%, vs. 1985 historic")

legend_patches = [mpatches.Patch(color=colors['irr'][0], label='Historical (W5E5)           '),
                  mpatches.Patch(color=colors['noi'][0], label='Historical NoIrr'),
                  mpatches.Patch(color=colors['irr_fut'][0], label='Future (CESM2), SSP-1.26'),
                  mpatches.Patch(color=colors['noi_fut'][0], label='Future NoIrr, SSP-1.26'),
                  mpatches.Patch(color=colors['irr_fut'][1], label='Future (CESM2), SSP-3.70'),
                    mpatches.Patch(color=colors['noi_fut'][1], label='Future NoIrr, SSP-3.70'),
                    # Line2D([0], [0], color='grey', linestyle='dotted', linewidth=3, label=f'individual member mean'),
                    ]
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.3)     
                         
# plt.tight_layout()
fig.legend(handles=legend_patches, loc='upper right', ncols=1, bbox_to_anchor=(0.9,0.88), frameon=False, fontsize=12)#,
          # bbox_to_anchor=(0.512, 0.96), ncols=5, fontsize=12,columnspacing=1.5)
# fig.subplots_adjust(right=0.97)

fig_folder = '/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/04. Figures/99. Final Figures/'
# plt.savefig(f"{fig_folder}/01. EGU25/Future_Volume_Evolution_ssp126only.png") 
# plt.savefig(f"{fig_folder}/01. EGU25/Future_Volume_Evolution_noannotation.png")    
plt.savefig(f"{fig_folder}/Future_Volume_Evolution.png")       
        
        
        
        
#%% OUTDATED/NOT USED Cell 8: Create run with hydro plot - 

# melt_evolution_data=[]
# fig,ax  = plt.subplots(figsize=(12,8))

# # hydro_data_path_historic = f"{wd_path}/summary/hydro_run_output_baseline_W5E5.000.nc"
# # with xr.open_dataset(hydro_data_path_historic) as hydro_data_hist:
# #     ax.plot(hydro_data_hist.time, hydro_data_hist.sum(dim='rgi_id').melt_on_glacier, label=f"Historic (W5E5)", color='k', lw=3)


# markers=['s', 'o', '^']
# for s, ssp in enumerate(["126", "370"]):
#     for member in range(4):
#         if member>=1:
#             hydro_data_path = f"{wd_path}/summary/climate_run_output_perturbed_SSP{ssp}_CESM2.00{member}_hydro.nc"
#             with xr.open_dataset(hydro_data_path) as hydro_data:
#                 ax.plot(hydro_data.time, hydro_data.sum(dim='rgi_id').melt_on_glacier, label=f"CESM2.00{member} - SSP{ssp}", color=colors['noi_fut'][s], lw=1)
#                 melt_evolution_data.append(hydro_data.sum(dim='rgi_id').melt_on_glacier)
            
#     if melt_evolution_data:
#         # Convert to DataArray for easier handling
#         melt_stack = xr.concat(melt_evolution_data, dim='member')
#         min_melt = melt_stack.min(dim='member')
#         max_melt = melt_stack.max(dim='member')
#         mean_melt = melt_stack.mean(dim='member')
#         ax.plot(
#             hydro_data.time,
#             mean_melt,
#             color=colors['noi_fut'][s],
#             alpha=1,
#             label=f'SSP{ssp} mean',
#             lw=3)
#         ax.fill_between(
#             hydro_data.time,
#             min_melt,
#             max_melt,
#             color=colors['noi_fut'][s],
#             alpha=0.1,
#             label=f'SSP{ssp} envelope'
#         )
# ax.set_ylabel("Annual melt on glacier (m3)")

# %% Cell 9: Process hydro data to monthly timeseries and create output plot
#When changing the future yera insight in seasonality can be gained

wd_path="/Users/magaliponds/Documents/00. Programming/04. Modelled perturbation-glacier interactions - R13-15 A+1km2"
        
                
# fig, ax = plt.subplots(figsize=(5, 4), sharex=True)
monthly_member_data=[]
monthly_member_data_past=[]

members = [1, 3, 4, 6, 4]
models = ["IPSL-CM6", "E3SM", "CESM2", "CNRM", "NorESM"]
f, ax = plt.subplots(figsize=(9, 6));
for e,ex in enumerate(["irr", "noi"]):
    #First load past data
    for m,model in enumerate(models):
        for member in range(members[m]):
            if member>=1:
                sample_id = f"{model}.00{member}"
                if ex=="irr":
                    if member==1:
                        hydro_data_past = f"{wd_path}/summary/hydro_run_output_baseline_W5E5.000.nc" #open the W5E5 data in the irr case
                                                            # hydro_run_output_baseline_W5E5.000.nc
                    else:
                        break
                else:
                    hydro_data_past = f"{wd_path}/summary/hydro_run_output_perturbed_{sample_id}.nc" #open the sample id data for the noi case
                with xr.open_dataset(hydro_data_past) as ds:
                    sel_vars = [v for v in ds.variables if 'month_2d' not in ds[v].dims] #only convert the variables present
                    # print(sel_vars)
                    df_annual = ds[sel_vars].to_dataframe() 
                    runoff_vars = ['melt_off_glacier', 'melt_on_glacier', 'liq_prcp_off_glacier', 'liq_prcp_on_glacier', 'snowfall_on_glacier', 'snowfall_off_glacier']
                    df_runoff = df_annual[runoff_vars] * 1e-9
                    df_runoff = df_runoff[df_runoff.index.get_level_values('time') != 2074.0].groupby(level='time').sum() #oggm does not include last year in hydro simulation for current version
                    
                    runoff_vars_monthly = ['melt_off_glacier_monthly', 'melt_on_glacier_monthly', 'liq_prcp_off_glacier_monthly', 'liq_prcp_on_glacier_monthly','snowfall_on_glacier_monthly', 'snowfall_off_glacier_monthly']
                    monthly_runoff = ds[runoff_vars_monthly] 
                    monthly_runoff = monthly_runoff.sum(dim='rgi_id') *1e-9
                    # monthly_runoff = monthly_runoff.rolling(time=31, center=True, min_periods=1).mean() * 1e-9
                    
                    total_runoff = (
                        monthly_runoff['melt_off_glacier_monthly']
                        + monthly_runoff['melt_on_glacier_monthly']
                        + monthly_runoff['liq_prcp_on_glacier_monthly']
                        + monthly_runoff['liq_prcp_off_glacier_monthly']
                        )
                    
                    total_runoff = total_runoff.rename('total_runoff')
                    # if ex=='irr':
                    df_monthly = total_runoff.to_dataframe()
                    # df_monthly = df_monthly.stack().reset_index()
                    # df_monthly.columns = ['year', 'month', 'month_2d', 'runoff']
    
                    # Add metadata
                    df_monthly['ssp'] = f'SSP{ssp}'
                    df_monthly['experiment'] = ex
                    df_monthly['member'] = member
    
                    monthly_member_data_past.append(df_monthly)
                    for i, yr in enumerate([1985]):
                        if ex=='irr':
                            ls='-'
                        else:
                            ls=':'
                            total_runoff.sel(time=yr).plot(ax=ax, color=colors[f'{ex}'][0],  ls=ls, lw=1, zorder=10)#label=f'Year {yr}',
                            
                    plt.ylabel('Mt yr$^{-1}$');# plt.legend(loc='best');
                    plt.xlabel('Month'); plt.title('Total monthly runoff change with time');
    df_monthly_all_past = pd.concat(monthly_member_data_past, ignore_index=True)
    
    for i,yr in enumerate([1985]):
        if ex=='irr':
            ls='-'
        else:
            ls='--'
        df_subset = df_monthly_all_past[(df_monthly_all_past['experiment'] == ex) & (df_monthly_all_past['calendar_year'] == yr)]
        df_stats = df_subset.groupby('calendar_month_2d')['total_runoff'].agg(['min', 'max', 'mean']).reset_index()
        # df_avg_monthly = df_subset.groupby('month')['runoff'].mean()#.reset_index()
        ax.plot(df_stats['calendar_month_2d'], df_stats['mean'],color=colors[f'{ex}'][0], ls=ls, lw=3, zorder=100)#label=f'Year {yr}',
        ax.fill_between(df_stats['calendar_month_2d'], df_stats['min'], df_stats['max'], color=colors[f'{ex}'][1],alpha=0.3, zorder=50 )#, label='Range (min–max)')
                    
                    
    for s, ssp in enumerate(["126", "370"]):
        for member in range(4): #4
            if member>=1:
                if ex=="irr":
                    hydro_data_path = f"{wd_path}/summary/climate_run_output_perturbed_SSP{ssp}_CESM2.00{member}_hydro.nc"
                else:
                    hydro_data_path = f"{wd_path}/summary/climate_run_output_perturbed_SSP{ssp}_CESM2.00{member}_noi_bias_hydro.nc"
                with xr.open_dataset(hydro_data_path) as ds:
                    sel_vars = [v for v in ds.variables if 'month_2d' not in ds[v].dims] #only convert the variables present
                    df_annual = ds[sel_vars].to_dataframe() 
                    runoff_vars = ['melt_off_glacier', 'melt_on_glacier', 'liq_prcp_off_glacier', 'liq_prcp_on_glacier']
                    df_runoff = df_annual[runoff_vars] * 1e-9
                    df_runoff = df_runoff[df_runoff.index.get_level_values('time') != 2074.0].groupby(level='time').sum()
                    
                    runoff_vars_monthly = ['melt_off_glacier_monthly', 'melt_on_glacier_monthly', 'liq_prcp_off_glacier_monthly', 'liq_prcp_on_glacier_monthly']
                    monthly_runoff = ds[runoff_vars_monthly] 
                    monthly_runoff = monthly_runoff.sum(dim='rgi_id') *1e-9
                    total_runoff = (
                        monthly_runoff['melt_off_glacier_monthly']
                        + monthly_runoff['melt_on_glacier_monthly']
                        + monthly_runoff['liq_prcp_on_glacier_monthly']
                        + monthly_runoff['liq_prcp_off_glacier_monthly']
                        + monthly_runoff['snowfall_on_glacier_monthly']
                        + monthly_runoff['snowfall_off_glacier_monthly']

                        )
                                    
                    # monthly_runoff = monthly_runoff.rolling(time=31, center=True, min_periods=1).mean() * 1e-9
                    
                    #Claculate runoff vs precipitation
                    # tot_precip = ds['liq_prcp_off_glacier_monthly'] + ds['liq_prcp_on_glacier_monthly'] + ds['snowfall_off_glacier_monthly'] + ds['snowfall_on_glacier_monthly']
                    # tot_precip = tot_precip.sum(dim='rgi_id') *1e-9
                    total_runoff = total_runoff.rename('total_runoff')
                    df_monthly = total_runoff.to_dataframe() 
                    # df_monthly = df_monthly.stack().reset_index()
                    # df_monthly.columns = ['year', 'month', 'month_2d','runoff']
    
                    # Add metadata
                    df_monthly['ssp'] = f'SSP{ssp}'
                    df_monthly['experiment'] = ex
                    df_monthly['member'] = member
    
                    monthly_member_data.append(df_monthly)
                    for i, yr in enumerate([2073]):
                        if i==0:
                            colorlabel="_fut"
                        else:
                            colorlabel=""
                        if ex=='irr':
                            ls='-'
                        else:
                            ls=':'
                       
                        total_runoff.sel(time=yr).plot(ax=ax, color=colors[f'{ex}{colorlabel}'][s],  ls=ls, lw=0.5, zorder=10)#label=f'Year {yr}',
                        # print(ssp, member, yr, total_runoff.sel(time=yr).values)
                    plt.ylabel('Runoff [m$^3$]'); #plt.legend(loc='best');
                    plt.xlabel('Month'); plt.title('Monthly runoff in past/future');
                
            
        df_monthly_all = pd.concat(monthly_member_data, ignore_index=True)
        for i,yr in enumerate([2073]):
            if ex=='irr':
                ls='-'
            else:
                ls='--'
            df_subset = df_monthly_all[(df_monthly_all['ssp'] == f'SSP{ssp}') & (df_monthly_all['experiment'] == ex) & (df_monthly_all['calendar_year'] == yr)]
            # print(ex, ssp, "fut", df_subset)
            df_stats = df_subset.groupby('calendar_month_2d')['total_runoff'].agg(['min', 'max', 'mean']).reset_index()
            # df_avg_monthly = df_subset.groupby('month')['runoff'].mean()#.reset_index()
            ax.plot(df_stats['calendar_month_2d'], df_stats['mean'],color=colors[f'{ex}_fut'][s], ls=ls, lw=3, zorder=100)#label=f'Year {yr}',
            ax.fill_between(df_stats['calendar_month_2d'], df_stats['min'], df_stats['max'], color=colors[f'{ex}_fut'][s],alpha=0.3, zorder=50 )#, label='Range (min–max)')

            

legend_patches = [Line2D([0], [0], color=colors['irr'][0], lw=2, label='Historical (W5E5, 1985)'),#label='Historical (W5E5)'),
                    Line2D([0], [0], color=colors['noi'][0], lw=2, ls='--', label='Historical NoIrr (IRRMIP ensemble, 1985)'),
                    Line2D([0], [0], color=colors['irr_fut'][0], lw=2, label='Future (CESM2, 2073), SSP-1.26'),
                    Line2D([0], [0], color=colors['noi_fut'][0], lw=2, ls='--', label='Future NoIrr, SSP-1.26'),
                    Line2D([0], [0], color=colors['irr_fut'][1], lw=2, label='Future (CESM2, 2073), SSP-3.70'),
                    Line2D([0], [0], color=colors['noi_fut'][1], lw=2, ls='--', label='Future NoIrr, SSP-3.70'),
                    # Line2D([0], [0], color='grey', linestyle='dotted', linewidth=3, label='Individual Member Mean'),
                ]
        
f.legend(handles=legend_patches, loc='upper left', ncols=1, frameon=False, fontsize=12,bbox_to_anchor=(0.13,0.88), )#,
       
        
#%% Cell 10: Align data per subregion and create output dataset
# --- Config ---
members = [4]
models = ["CESM2"]
experiments = ["irr", "noi"]
ssps = ["126", "370"]

# --- Data accumulation ---
df_monthly_all = []
df_runoff_shares=[]

base_ds = pd.read_csv(f"{wd_path}/masters/master_gdirs_r3_a1_rgi_date_A_V_RGIreg.csv", usecols=['rgi_id','rgi_subregion']).set_index('rgi_id').to_xarray() #open rgi subregions per rgi_id
rgi_ids=[]
for ex in experiments:
    for m, model in enumerate(models):
        for member in range(members[m]):
            if member >= 1:
                sample_id = f"{model}.00{member}"
                if ex == "irr" and member == 1:
                    hydro_data_past = f"{wd_path}/summary/hydro_run_output_baseline_W5E5.000.nc"
                elif ex == "noi":
                    hydro_data_past = f"{wd_path}/summary/hydro_run_output_perturbed_{sample_id}.nc"
                else:
                    continue
                try:
                    with xr.open_dataset(hydro_data_past) as ds:
                        if m==0 and member==1:
                            rgi_ids = ds.rgi_id.values
                            
                        melt_on = ds['melt_on_glacier_monthly'] * 1e-9 #only working with monthly components, can delete for already averaged, but than also delete summation later
                        melt_off = ds['melt_off_glacier_monthly'] * 1e-9
                        prcp_on = ds['liq_prcp_on_glacier_monthly'] * 1e-9
                        prcp_off = ds['liq_prcp_off_glacier_monthly'] * 1e-9
                        snow_on = ds['snowfall_on_glacier_monthly'] * 1e-9
                        snow_off = ds['snowfall_off_glacier_monthly'] * 1e-9
                        
                        
                        total_runoff = melt_on + melt_off + prcp_on + prcp_off + snow_on + snow_off#sum all the runoff components
                        # total_runoff = melt_on + prcp_on #+ melt_off + prcp_on + prcp_off #for melt_on only

                        
                        eps = 0#1e-10
                        shares = xr.Dataset({
                            'share_melt_on': melt_on / (total_runoff + eps),
                            'share_melt_off': melt_off / (total_runoff + eps),
                            'share_prcp_on': prcp_on / (total_runoff + eps),
                            'share_prcp_off': prcp_off / (total_runoff + eps), 
                            'share_snow_on': snow_on / (total_runoff + eps),
                            'share_snow_off': snow_off / (total_runoff + eps), 
                        }) #calculate the share of each runoff component compared to total (make 0 zero if year runoff is 0) 
                        #Share is taken based on annual runoff values
                        
                        
                        runoff = xr.merge([total_runoff.to_dataset(name='runoff'), base_ds]).set_coords('rgi_subregion') #merge the runoff data with the subregion ids
                        runoff_subregions = runoff.groupby('rgi_subregion').sum(dim='rgi_id')
                        runoff_subregions['glacier_count'] = runoff['rgi_id'].groupby(runoff['rgi_subregion']).count()

                        # df_runoff = runoff_subregions.to_dataframe().reset_index()[['time',  'runoff', 'rgi_subregion']]
                        # df_runoff = df_runoff.rename(columns={'time': 'year'})
                        df_runoff = runoff_subregions.to_dataframe().reset_index()[['time', 'month_2d', 'runoff', 'rgi_subregion', 'glacier_count']]                
                        df_runoff = df_runoff.rename(columns={'time': 'year', 'month_2d': 'month'})
                        df_runoff['experiment'] = ex
                        df_runoff['ssp'] = "hist"
                        df_runoff['member'] = member
                        df_monthly_all.append(df_runoff) #crate table with experiments, ssps and member data
    
                        share_subregions = shares.assign_coords(rgi_subregion=base_ds.rgi_subregion).groupby('rgi_subregion').mean(dim='rgi_id')
                        df_shares = share_subregions.to_dataframe().reset_index()
                        df_shares['experiment'] = ex
                        df_shares['ssp'] = "hist"
                        df_shares['member'] = member
                        df_runoff_shares.append(df_shares)

                        
                except Exception as e:
                    print(f"Error loading {hydro_data_past}: {e}")

    for ssp in ssps:
        for member in range(1, 4):
            sample_id = f"CESM2.00{member}"
            if ex == "irr":
                hydro_data_path = f"{wd_path}/summary/climate_run_output_perturbed_SSP{ssp}_{sample_id}_hydro.nc"
            else:
                hydro_data_path = f"{wd_path}/summary/climate_run_output_perturbed_SSP{ssp}_{sample_id}_noi_bias_hydro.nc"
            try:
                with xr.open_dataset(hydro_data_path) as ds:
                    
                    melt_on = ds['melt_on_glacier_monthly'] * 1e-9
                    melt_off = ds['melt_off_glacier_monthly'] * 1e-9
                    prcp_on = ds['liq_prcp_on_glacier_monthly'] * 1e-9
                    prcp_off = ds['liq_prcp_off_glacier_monthly'] * 1e-9
                    snow_on = ds['snowfall_on_glacier_monthly'] * 1e-9
                    snow_off = ds['snowfall_off_glacier_monthly'] * 1e-9
                    total_runoff = melt_on + melt_off + prcp_on + prcp_off + snow_on + snow_off#sum all the runoff components
                    
                    eps = 0#1e-10
                    shares = xr.Dataset({
                        'share_melt_on': melt_on / (total_runoff + eps),
                        'share_melt_off': melt_off / (total_runoff + eps),
                        'share_prcp_on': prcp_on / (total_runoff + eps),
                        'share_prcp_off': prcp_off / (total_runoff + eps),
                    })
                
                    runoff = xr.merge([total_runoff.to_dataset(name='runoff'), base_ds]).set_coords('rgi_subregion')
                    runoff_subregions = runoff.groupby('rgi_subregion').sum(dim='rgi_id')
                    runoff_subregions['glacier_count'] = runoff['rgi_id'].groupby(runoff['rgi_subregion']).count()

                    # df_runoff = runoff_subregions.to_dataframe().reset_index()[['time',  'runoff', 'rgi_subregion']]
                    # df_runoff = df_runoff.rename(columns={'time': 'year'})
                    df_runoff = runoff_subregions.to_dataframe().reset_index()[['time', 'month_2d', 'runoff', 'rgi_subregion', 'glacier_count']]
                    df_runoff = df_runoff.rename(columns={'time': 'year', 'month_2d': 'month'})
                    df_runoff['experiment'] = ex
                    df_runoff['ssp'] = ssp
                    df_runoff['member'] = member
                    df_monthly_all.append(df_runoff)

                    share_subregions = shares.assign_coords(rgi_subregion=base_ds.rgi_subregion).groupby('rgi_subregion').mean(dim='rgi_id')
                    df_shares = share_subregions.to_dataframe().reset_index().rename(columns={'time': 'year', 'month_2d': 'month'})
                    df_shares['experiment'] = ex
                    df_shares['ssp'] = ssp
                    df_shares['member'] = member
                    df_runoff_shares.append(df_shares)
            except Exception as e:
                print(f"Error loading {hydro_data_path} pt 2: {e}")

# --- Combine and analyze ---
df_monthly_all = pd.concat(df_monthly_all, ignore_index=True)
df_annual = df_monthly_all.groupby(['year', 'experiment', 'ssp', 'member', 'rgi_subregion'])['runoff'].sum().reset_index()
df_runoff_shares = pd.concat(df_runoff_shares, ignore_index=True)

#filepaths for only melt runoff components
# opath_df_monthly = os.path.join(wd_path,'masters','hydro_output_monthly_subregions_melton_only.csv')
# opath_df_annual = os.path.join(wd_path,'masters','hydro_output_annual_subregions_melton_only.csv')

#file paths for only melt and precipitation runoff components
# opath_df_monthly = os.path.join(wd_path,'masters','hydro_output_monthly_subregions_meltprcpon_only.csv')
# opath_df_annual = os.path.join(wd_path,'masters','hydro_output_annual_subregions_meltprcpon_only.csv')

#file paths including all runoff components
opath_df_monthly = os.path.join(wd_path,'masters','hydro_output_monthly_subregions.csv')
opath_df_annual = os.path.join(wd_path,'masters','hydro_output_annual_subregions.csv')
opath_df_runoff_shares = os.path.join(wd_path,'masters','hydro_output_monthly_runoff_shares.csv')


df_monthly_all.to_csv(opath_df_monthly)
df_annual.to_csv(opath_df_annual)
df_runoff_shares.to_csv(opath_df_runoff_shares)

#%% Cell 11: Runoff plot including share by runoff component contribution to total _ v2

#when working with melton only
# opath_df_monthly = os.path.join(wd_path,'masters','hydro_output_monthly_subregions_melton_only.csv')
# opath_df_annual = os.path.join(wd_path,'masters','hydro_output_annual_subregions_melton_only.csv')

#when working with melton and prcp on only
opath_df_monthly = os.path.join(wd_path,'masters','hydro_output_monthly_subregions_meltprcpon_only.csv')
opath_df_annual = os.path.join(wd_path,'masters','hydro_output_annual_subregions_meltprcpon_only.csv')
# 
#when working with total runoff
# opath_df_monthly = os.path.join(wd_path,'masters','hydro_output_monthly_subregions.csv')
# opath_df_annual = os.path.join(wd_path,'masters','hydro_output_annual_subregions.csv')

opath_df_runoff_shares = os.path.join(wd_path,'masters','hydro_output_monthly_runoff_shares.csv')


df_annual = pd.read_csv(opath_df_annual).reset_index()
df_runoff_shares = pd.read_csv(opath_df_runoff_shares)
df_monthly = pd.read_csv(opath_df_monthly, dtype={'ssp': str, 'rgi_subregion': str, 'experiment': str})[['year','month','experiment','ssp','member', 'runoff','rgi_subregion']]
                         
df_avg_monthly = (df_monthly.groupby(['year','month', 'experiment', 'ssp', 'rgi_subregion',])['runoff'].mean().reset_index()) #average over members
df_avg_monthly = df_avg_monthly[df_avg_monthly['runoff'] != 0.00]
df_avg_jja = df_avg_monthly[df_avg_monthly['month'].isin([6, 7, 8])]
df_avg_subregions = (df_avg_jja.groupby(['year', 'experiment', 'ssp', 'rgi_subregion'])['runoff'].sum().reset_index()) #sum yo annual runoff for all subregions per year
# df_avg_subregions = (df_avg_monthly.groupby(['year', 'experiment', 'ssp', 'rgi_subregion'])['runoff'].sum().reset_index()) #sum yo annual runoff for all subregions per year
df_avg_subregions = df_avg_subregions[~df_avg_subregions['year'].isin([2074])] 
df_avg_all = (df_avg_subregions.groupby(['year', 'experiment', 'ssp'])['runoff'].sum().reset_index()) #sum for all of HMA

early_period = (1985, 2014)
late_period = (2044, 2074)

# Filter and compute early period averages
df_early_avg = (
    df_runoff_shares[(df_runoff_shares['year'] >= early_period[0]) & (df_runoff_shares['year'] <= early_period[1])]
    .groupby(['rgi_subregion','experiment'])[['share_melt_on', 'share_melt_off', 'share_prcp_on', 'share_prcp_off']]
    .mean()
)

# Same for late period
df_late_avg = (
    df_runoff_shares[(df_runoff_shares['year'] >= late_period[0]) & (df_runoff_shares['year'] <= late_period[1])]
    .groupby(['rgi_subregion','experiment'])[['share_melt_on', 'share_melt_off', 'share_prcp_on', 'share_prcp_off']]
    .mean()
)

fig,axes = plt.subplots(4,4,figsize=(12,8), sharex=True, sharey=False)
ax = axes.flatten()

c=1

component_colors = [
    'navy',  # Marine (melt on glacier)
    '#FFFFFF',  # Powder blue (melt off glacier)
    '#FFFFFF',  # Pine green (precip on glacier)
    '#FFFFFF',  # Tea green (precip off glacier)
]

component_colors_noirr = [
    'royalblue',  # Powder blue (melt on glacier noirr)], before #4682a9',
    '#FFFFFF',
    '#FFFFFF',
    '#FFFFFF',
]

for region in df_avg_annual['rgi_subregion'].unique():
    print(region)
    region_data = df_avg_subregions[df_avg_subregions['rgi_subregion'] == region] #check if glacier is in region
    #calculate relative runoff
    base_year = region_data[region_data['year']==1985][region_data['experiment']=='irr'].runoff.values
    region_data['runoff_relative']=((region_data['runoff']-base_year)/base_year)
    
    # calculate cumulative absolute total runoff
    # region_data['runoff_cumulative'] = region_data.sort_values([ 'year']).groupby([ 'experiment', 'ssp'])['runoff'].cumsum() #sorting as years need to be in order for accumulation
    
    for (exp, ssp) in region_data.groupby(['experiment', 'ssp']).groups.keys():
        # Skip 'hist' by itself — only process SSPs
        if ssp == 'hist':
            continue
        # if ssp=='126':
            # continue

        # Define index for colors or line styles
        add = "_fut"
        s = 0 if ssp == "126" else 1

        # Combine hist + this SSP for the same experiment
        hist_ssp = region_data[
            (region_data['experiment'] == exp) &
            (region_data['ssp'].isin(['hist', ssp]))
        ].copy()
        


        # Sort by time
        hist_ssp = hist_ssp.sort_values('year')
        
        hist_ssp['runoff_cumulative'] = hist_ssp.sort_values(['year']).groupby(['experiment'])['runoff'].cumsum() #sorting as years need to be in order for accumulation


        # Smooth
        hist_ssp['runoff_smoothed'] = (
            hist_ssp['runoff']
            .rolling(window=11, center=True, min_periods=1)
            .mean()
        )
        
        hist_ssp['relative_runoff_smoothed'] = (
            hist_ssp['runoff_relative']
            .rolling(window=11, center=True, min_periods=1)
            .mean()
        )

        hist_ssp['cumulative_runoff_smoothed'] = (
            hist_ssp['runoff_cumulative']
            .rolling(window=11, center=True, min_periods=1)
            .mean()
        )
        
        # Style settings
        ls = '--' if exp == "noi" else '-'
        lw = 1 if exp == "noi" else 2

        # Split into historical & future
        is_hist = hist_ssp['year'] < 2014
        is_future = hist_ssp['year'] >= 2014

        # Plot hist part (no add)
        ax[c].plot(
            hist_ssp.loc[is_hist, 'year'],
            # hist_ssp.loc[is_hist, 'relative_runoff_smoothed']*100,
            hist_ssp.loc[is_hist, 'runoff_cumulative'],
            # hist_ssp.loc[is_hist, 'runoff_smoothed'],
            label=f"{exp.upper()} hist",
            color=colors[f'{exp}'][0],
            linestyle=ls, linewidth=lw
        )

        # Plot future part (add)
        ax[c].plot(
            hist_ssp.loc[is_future, 'year'],
            # hist_ssp.loc[is_future, 'relative_runoff_smoothed']*100,
            hist_ssp.loc[is_future, 'runoff_cumulative'],
            # hist_ssp.loc[is_future, 'runoff_smoothed'],
            label=f"{exp.upper()} {ssp}",
            color=colors[f'{exp}{add}'][s],
            linestyle=ls, linewidth=lw
        )
        
        peak_year=hist_ssp.loc[hist_ssp['runoff_smoothed'].idxmax(), 'year']
        if peak_year<=2014:
            color='black'
        else:
            color=colors[f'{exp}{add}'][s]
        ax[c].axvline(peak_year, color=color, linestyle=ls, lw=1)
        ax[c].set_title(f'Region: {region}')
    #     # ax[c].set_ylim(-100,800)
    #     # if c not in [4,8,12]:
    #     #     ax[c].set_yticklabels([])
    
    
    # peak_year = region_data.loc[region_data['runoff'].idxmax(), 'year']
    # ax[c].axvline(2014, color='grey', linestyle='--', label='Historical/Future Split', lw=1)
    # ax[c].axvline(peak_year, color='red', linestyle=':', label=f'Peak Year: {peak_year}', lw=1)
    ax[c].set_title(f'Region: {region}')
    
    ax[c].grid(True, color='grey', ls='--', lw=0.5)
    avg_early = df_early_avg.loc[region]
    avg_late = df_late_avg.loc[region]
    
    
    #Add inset axes for pie charts
    inset_ax1 = inset_axes(ax[c], width="40%", height="40%", loc='upper left', borderpad=0)
    inset_ax2 = inset_axes(ax[c], width="40%", height="40%", loc='lower right', borderpad=0)
    wedges_outer, texts_outer = inset_ax1.pie(avg_early.loc[avg_late.index == 'noi'].values.flatten(),radius=1.0, labels=None, startangle=90,colors=component_colors_noirr,wedgeprops=dict(width=0.4, edgecolor='w'))
    wedges_inner, texts_inner = inset_ax1.pie(avg_early.loc[avg_late.index == 'irr'].values.flatten(),radius=0.7,labels=None, startangle=90,colors=component_colors,wedgeprops=dict(width=0.4, edgecolor='w'))
    centre_circle = plt.Circle((0, 0), 0.4, fc='white')
    inset_ax1.add_artist(centre_circle)
    inset_ax1.set(aspect='equal')
    inset_ax1.set_title('')  # Remove default title position

    #if error on file format (inf values) - probably updated the prcp/melton input files before without commenting out the shares file, that file only works for total runoff
    wedges_outer, texts_outer = inset_ax2.pie(avg_late.loc[avg_late.index == 'noi'].values.flatten(),radius=1.0, labels=None, startangle=90,colors=component_colors_noirr,wedgeprops=dict(width=0.4, edgecolor='w'))
    wedges_inner, texts_inner = inset_ax2.pie(avg_late.loc[avg_late.index == 'irr'].values.flatten(),radius=0.7,labels=None, startangle=90,colors=component_colors,wedgeprops=dict(width=0.4, edgecolor='w'))
    centre_circle = plt.Circle((0, 0), 0.4, fc='white')
    inset_ax2.add_artist(centre_circle)
    inset_ax2.set(aspect='equal')
    inset_ax2.set_title('')  # Remove default title position

    
    
    
    plt.tight_layout()
    c+=1


base_year_all =df_avg_all[df_avg_all['year']==1985][df_avg_all['experiment']=='irr'].runoff.values

#calculate relative runoff
df_avg_all['runoff_relative']=(df_avg_all['runoff']-base_year_all)/base_year_all


for (exp, ssp) in df_avg_all.groupby(['experiment', 'ssp']).groups.keys():
    
    # Skip 'hist' by itself — only process SSPs
    if ssp == 'hist':
        continue
    # if ssp=='126':
        # continue

    # Define index for colors or line styles
    add = "_fut"
    s = 0 if ssp == "126" else 1

    # Combine hist + this SSP for the same experiment
    hist_ssp = df_avg_all[
        (df_avg_all['experiment'] == exp) &
        (df_avg_all['ssp'].isin(['hist', ssp]))
    ].copy()

    # Sort by time
    hist_ssp = hist_ssp.sort_values('year')
    
    
    #calculate cumulative absolute total runoff
    hist_ssp['runoff_cumulative'] = hist_ssp.sort_values(['year']).groupby(['experiment'])['runoff'].cumsum() #sorting as years need to be in order for accumulation


    # Smooth
    hist_ssp['runoff_smoothed'] = (
        hist_ssp['runoff']
        .rolling(window=11, center=True, min_periods=1)
        .mean()
    )
    
    hist_ssp['relative_runoff_smoothed'] = (
        hist_ssp['runoff_relative']
        .rolling(window=11, center=True, min_periods=1)
        .mean()
    )
    
    hist_ssp['cumulative_runoff_smoothed'] = (
        hist_ssp['runoff_cumulative']
        .rolling(window=11, center=True, min_periods=1)
        .mean()
    )

    # Style settings
    ls = '--' if exp == "noi" else '-'
    lw = 1 if exp == "noi" else 2

    # Split into historical & future
    is_hist = hist_ssp['year'] < 2014
    is_future = hist_ssp['year'] >= 2014

    # Plot hist part (no add)
    ax[0].plot(
        hist_ssp.loc[is_hist, 'year'],
        # hist_ssp.loc[is_hist, 'relative_runoff_smoothed']*100,
        hist_ssp.loc[is_hist, 'runoff_cumulative'],
        # hist_ssp.loc[is_hist, 'runoff_smoothed'],
        label=f"{exp.upper()} hist",
        color=colors[f'{exp}'][0],
        linestyle=ls, linewidth=lw
    )

    # Plot future part (add)
    ax[0].plot(
        hist_ssp.loc[is_future, 'year'],
        # hist_ssp.loc[is_future, 'relative_runoff_smoothed']*100,
        hist_ssp.loc[is_future, 'runoff_cumulative'],
        # hist_ssp.loc[is_future, 'runoff_smoothed'],
        label=f"{exp.upper()} {ssp}",
        color=colors[f'{exp}{add}'][s],
        linestyle=ls, linewidth=lw
    )
    
    peak_year=hist_ssp.loc[hist_ssp['runoff_smoothed'].idxmax(), 'year']
    if peak_year<=2014:
        color='black'
    else:
        color=colors[f'{exp}{add}'][s]
    ax[0].axvline(peak_year, color=color, linestyle=ls, lw=1)


fig.text(
    0.5,  # to the right
    -0.02,         # middle of whisker
    'Years',
    va='center',
    ha='left',
    fontsize=12,
    # fontweight='bold',
    
)

fig.text(
    -0.01,  # to the right
    0.5,         # middle of whisker
    # 'Total annual runoff [km$^3$]',
    # 'Total cumulative annual runoff [km$^3$]',
    'Total annual runoff change [%]',
    va='center',
    ha='left',
    fontsize=12,
    # fontweight='bold'
    rotation=90
)

# peak_year = region_data.loc[region_data['runoff'].idxmax(), 'year']

# ax[0].axvline(2014, color='grey', linestyle='--', label='Historical/Future Split', lw=1)
# ax[0].axvline(peak_year, color='red', linestyle=':', label=f'Peak Year: {peak_year}', lw=1)

ax[0].set_title('High Mountain Asia', fontweight='bold')
ax[0].grid(True, color='grey', ls='--', lw=0.5)

avg_early = df_early_avg.groupby('experiment').mean()
avg_late = df_late_avg.groupby('experiment').mean()
inset_ax1 = inset_axes(ax[0], width="40%", height="40%", loc='upper left', borderpad=0)
inset_ax2 = inset_axes(ax[0], width="40%", height="40%", loc='lower right', borderpad=0)
# wedges, texts =  inset_ax1.pie(avg_early, labels=None, startangle=90, autopct=None, colors=component_colors)#lambda pct: f"{int(round(pct))}%"
wedges_outer, texts_outer = inset_ax1.pie(avg_early.loc[avg_late.index == 'noi'].values.flatten(),radius=1.0, labels=None, startangle=90,colors=component_colors_noirr,wedgeprops=dict(width=0.4, edgecolor='w'))
wedges_inner, texts_inner = inset_ax1.pie(avg_early.loc[avg_late.index == 'irr'].values.flatten(),radius=0.7,labels=None, startangle=90,colors=component_colors,wedgeprops=dict(width=0.4, edgecolor='w'))
centre_circle = plt.Circle((0, 0), 0.4, fc='white')
inset_ax1.add_artist(centre_circle)
inset_ax1.set(aspect='equal')
inset_ax1.set_title('')  # Remove default title position

wedges_outer, texts_outer = inset_ax2.pie(avg_late.loc[avg_late.index == 'noi'].values.flatten(),radius=1.0, labels=None, startangle=90,colors=component_colors_noirr,wedgeprops=dict(width=0.4, edgecolor='w'))
wedges_inner, texts_inner = inset_ax2.pie(avg_late.loc[avg_late.index == 'irr'].values.flatten(),radius=0.7,labels=None, startangle=90,colors=component_colors,wedgeprops=dict(width=0.4, edgecolor='w'))

    
centre_circle = plt.Circle((0, 0), 0.4, fc='white')
inset_ax2.add_artist(centre_circle)
inset_ax2.set(aspect='equal')
inset_ax2.set_title('')  # Remove default title position
inset_ax1.annotate('1985–2014', xy=(0.5, -0.15), ha='center', va='center', fontsize=10, xycoords='axes fraction')
inset_ax2.set_title('2044–2074', fontsize=10)

# Plot pie in lower-right


pie_patches = [
    mpatches.Patch(color=component_colors[0], label='Melt on glacier'),
    mpatches.Patch(color=component_colors_noirr[0], label='Melt on glacier NoIrr'),
    mlines.Line2D([0], [0], color='grey', linestyle='-',linewidth=1,label='Peak water'),
    mlines.Line2D([0], [0], color='grey', linestyle='--',linewidth=1,label='Peak water NoIrr'),
]

# Line legend (your original)
line_patches = [
    mpatches.Patch(color=colors['irr'][0], label='Historical (W5E5)'),
    mpatches.Patch(color=colors['noi'][0], label='Historical NoIrr'),
    mpatches.Patch(color=colors['irr_fut'][0], label='Future (CESM2), SSP-1.26'),
    mpatches.Patch(color=colors['noi_fut'][0], label='Future NoIrr, SSP-1.26'),
    mpatches.Patch(color=colors['irr_fut'][1], label='Future (CESM2), SSP-3.70'),
    mpatches.Patch(color=colors['noi_fut'][1], label='Future NoIrr, SSP-3.70'),
    
]

# Combine them
legend_patches =  line_patches + pie_patches 

# Add to figure/axes
fig.legend(handles=legend_patches, loc='lower center', frameon=False, ncols=5, bbox_to_anchor=(0.5,-0.14))

# plt.xlabel('Year')
# plt.ylabel('Runoff [km³]')
# plt.legend()   
plt.show()
#%% Cell 11b: make plot with relative difference


# opath_df_monthly = os.path.join(wd_path,'masters','hydro_output_monthly_subregions_melton_only.csv')
# opath_df_annual = os.path.join(wd_path,'masters','hydro_output_annual_subregions_melton_only.csv')

opath_df_monthly = os.path.join(wd_path,'masters','hydro_output_monthly_subregions.csv')
opath_df_annual = os.path.join(wd_path,'masters','hydro_output_annual_subregions.csv')
opath_df_runoff_shares = os.path.join(wd_path,'masters','hydro_output_monthly_runoff_shares.csv')


df_annual = pd.read_csv(opath_df_annual).reset_index()
df_runoff_shares = pd.read_csv(opath_df_runoff_shares)
df_monthly = pd.read_csv(opath_df_monthly, dtype={'ssp': str, 'rgi_subregion': str, 'experiment': str})[['year','month','experiment','ssp','member', 'runoff','rgi_subregion']]
                         
df_avg_monthly = (df_monthly.groupby(['year','month', 'experiment', 'ssp', 'rgi_subregion',])['runoff'].mean().reset_index()) #average over members
df_avg_monthly = df_avg_monthly[df_avg_monthly['runoff'] != 0.00]
df_avg_jja = df_avg_monthly[df_avg_monthly['month'].isin([6, 7, 8])]
df_avg_subregions = (df_avg_jja.groupby(['year', 'experiment', 'ssp', 'rgi_subregion'])['runoff'].sum().reset_index()) #sum yo annual runoff for all subregions per year
# df_avg_subregions = (df_avg_monthly.groupby(['year', 'experiment', 'ssp', 'rgi_subregion'])['runoff'].sum().reset_index()) #sum yo annual runoff for all subregions per year
df_avg_subregions = df_avg_subregions[~df_avg_subregions['year'].isin([2074])] 
df_avg_all = (df_avg_subregions.groupby(['year', 'experiment', 'ssp'])['runoff'].sum().reset_index()) #sum for all of HMA

early_period = (1985, 2014)
late_period = (2044, 2074)

# Filter and compute early period averages
df_early_avg = (
    df_runoff_shares[(df_runoff_shares['year'] >= early_period[0]) & (df_runoff_shares['year'] <= early_period[1])]
    .groupby(['rgi_subregion','experiment'])[['share_melt_on', 'share_melt_off', 'share_prcp_on', 'share_prcp_off']]
    .mean()
)

# Same for late period
df_late_avg = (
    df_runoff_shares[(df_runoff_shares['year'] >= late_period[0]) & (df_runoff_shares['year'] <= late_period[1])]
    .groupby(['rgi_subregion','experiment'])[['share_melt_on', 'share_melt_off', 'share_prcp_on', 'share_prcp_off']]
    .mean()
)

fig,axes = plt.subplots(4,4,figsize=(12,8), sharex=True, sharey=False)
ax = axes.flatten()

c=1

component_colors = [
    'navy',  # Marine (melt on glacier)
    '#FFFFFF',  # Powder blue (melt off glacier)
    '#FFFFFF',  # Pine green (precip on glacier)
    '#FFFFFF',  # Tea green (precip off glacier)
]

component_colors_noirr = [
    'royalblue',  # Powder blue (melt on glacier noirr)], before #4682a9',
    '#FFFFFF',
    '#FFFFFF',
    '#FFFFFF',
]

for region in df_avg_annual['rgi_subregion'].unique():
    print(region)
    region_data = df_avg_subregions[df_avg_subregions['rgi_subregion'] == region] #check if glacier is in region
    base_year = region_data[region_data['year']==1985][region_data['experiment']=='irr'].runoff.values
    region_data['runoff_relative']=((region_data['runoff']-base_year)/base_year)
    # Calculate the difference
    # diff = (region_data[region_data['experiment'] == 'irr']['runoff_relative'].values- region_data[region_data['experiment'] == 'noi']['runoff_relative'].values)
    
    # region_data['runoff_relative_difference'] = np.nan
    
    region_data = region_data.sort_values(['ssp', 'year', 'rgi_subregion', 'experiment']) #control if irr or noirr comes first in the group, then we abstract the values [i] from [i-1]
    region_data['runoff_relative_difference'] = (
        region_data.groupby(['ssp', 'year', 'rgi_subregion'])['runoff_relative'].diff().shift(-1)) #aligns the difference with the irr experiment
    region_data.loc[region_data['experiment'] != 'irr', 'runoff_relative_difference'] = np.nan #s4et the values in the runoff_relative for noi to nan
    
    # region_data.loc[region_data['experiment'] == 'irr', 'runoff_relative_difference'] = diff    
    print("base_year", base_year)
    for (exp, ssp) in region_data.groupby(['experiment', 'ssp']).groups.keys():
        # Skip 'hist' by itself — only process SSPs
        if ssp == 'hist': #decomment for combined smoothening
            continue
        # if ssp=='126':
            # continue

        # Define index for colors or line styles
        add = "_fut"
        s = 0 if ssp == "126" else 1

        # Combine hist + this SSP for the same experiment
        hist_ssp = region_data[
            (region_data['experiment'] == exp) 
            & (region_data['ssp'].isin(['hist',ssp])) #for combined smoothening
            # & (region_data['ssp'].isin([ssp])) #for single smoothening
        ].copy()

        # Sort by time
        hist_ssp = hist_ssp.sort_values('year')

        # Smooth
        hist_ssp['runoff_smoothed'] = (
            hist_ssp['runoff']
            .rolling(window=11, center=True, min_periods=1)
            .mean()
        )
        
        hist_ssp['relative_runoff_smoothed'] = (
            hist_ssp['runoff_relative']
            .rolling(window=11, center=True, min_periods=1)
            .mean()
        )
        
        # Split into historical & future
        is_hist = hist_ssp['year'] < 2014
        is_future = hist_ssp['year'] >= 2014
        # Style settings
        ls = '--' if exp == "noi" else '-'
        lw = 1 if exp == "noi" else 2
        if exp =="irr":
            hist_ssp['relative_runoff_difference_smoothed'] = (
                hist_ssp['runoff_relative_difference']
                .rolling(window=11, center=True, min_periods=1)
                .mean()
            )
            # Plot hist part (no add)
            ax[c].plot(
                hist_ssp.loc[is_hist, 'year'],
                hist_ssp.loc[is_hist, 'relative_runoff_difference_smoothed']*100,
                # hist_ssp.loc[is_hist, 'runoff_relative_difference']*100,
                label=f"{exp.upper()} hist",
                color=colors[f'{exp}'][0],
                linestyle=ls, linewidth=lw
            )

            # Plot future part (add)
            ax[c].plot(
                hist_ssp.loc[is_future, 'year'],
                hist_ssp.loc[is_future, 'relative_runoff_difference_smoothed']*100,
                # hist_ssp.loc[is_future, 'runoff_relative_difference']*100,
                label=f"{exp.upper()} {ssp}",
                color=colors[f'{exp}{add}'][s],
                linestyle=ls, linewidth=lw
            )
        
        peak_year=hist_ssp.loc[hist_ssp['runoff_smoothed'].idxmax(), 'year']
        if peak_year<=2014:
            color='black'
        else:
            color=colors[f'{exp}{add}'][s]
        # ax[c].axvline(peak_year, color=color, linestyle=ls, lw=1)
        # ax[c].set_title(f'Region: {region}')
    #     # ax[c].set_ylim(-100,800)
    #     # if c not in [4,8,12]:
    #     #     ax[c].set_yticklabels([])
    
    
    # peak_year = region_data.loc[region_data['runoff'].idxmax(), 'year']
    # ax[c].axvline(2014, color='grey', linestyle='--', label='Historical/Future Split', lw=1)
    # ax[c].axvline(peak_year, color='red', linestyle=':', label=f'Peak Year: {peak_year}', lw=1)
    ax[c].set_title(f'Region: {region}')
    
    ax[c].grid(True, color='grey', ls='--', lw=0.5)
    avg_early = df_early_avg.loc[region]
    avg_late = df_late_avg.loc[region]
    
    
    #Add inset axes for pie charts
    inset_ax1 = inset_axes(ax[c], width="40%", height="40%", loc='upper left', borderpad=0)
    inset_ax2 = inset_axes(ax[c], width="40%", height="40%", loc='lower right', borderpad=0)
    wedges_outer, texts_outer = inset_ax1.pie(avg_early.loc[avg_late.index == 'noi'].values.flatten(),radius=1.0, labels=None, startangle=90,colors=component_colors_noirr,wedgeprops=dict(width=0.4, edgecolor='w'))
    wedges_inner, texts_inner = inset_ax1.pie(avg_early.loc[avg_late.index == 'irr'].values.flatten(),radius=0.7,labels=None, startangle=90,colors=component_colors,wedgeprops=dict(width=0.4, edgecolor='w'))
    centre_circle = plt.Circle((0, 0), 0.4, fc='white')
    inset_ax1.add_artist(centre_circle)
    inset_ax1.set(aspect='equal')
    inset_ax1.set_title('')  # Remove default title position

    wedges_outer, texts_outer = inset_ax2.pie(avg_late.loc[avg_late.index == 'noi'].values.flatten(),radius=1.0, labels=None, startangle=90,colors=component_colors_noirr,wedgeprops=dict(width=0.4, edgecolor='w'))
    wedges_inner, texts_inner = inset_ax2.pie(avg_late.loc[avg_late.index == 'irr'].values.flatten(),radius=0.7,labels=None, startangle=90,colors=component_colors,wedgeprops=dict(width=0.4, edgecolor='w'))
    centre_circle = plt.Circle((0, 0), 0.4, fc='white')
    inset_ax2.add_artist(centre_circle)
    inset_ax2.set(aspect='equal')
    inset_ax2.set_title('')  # Remove default title position

    
    
    
    plt.tight_layout()
    c+=1


base_year_all =df_avg_all[df_avg_all['year']==1985][df_avg_all['experiment']=='irr'].runoff.values
df_avg_all['runoff_relative']=(df_avg_all['runoff']-base_year_all)/base_year_all
df_avg_all = df_avg_all.sort_values(['ssp', 'year', 'experiment']) #control if irr or noirr comes first in the group, then we abstract the values [i] from [i-1]
df_avg_all['runoff_relative_difference'] = (
    df_avg_all.groupby(['ssp', 'year'])['runoff_relative'].diff().shift(-1)) #aligns the difference with the irr experiment
df_avg_all.loc[df_avg_all['experiment'] != 'irr', 'runoff_relative_difference'] = np.nan #s4et the values in the runoff_relative for noi to nan

for (exp, ssp) in df_avg_all.groupby(['experiment', 'ssp']).groups.keys():
    # Skip 'hist' by itself — only process SSPs
    if ssp == 'hist': #for combined smoothening
        continue
    # if ssp=='126':
        # continue

    # Define index for colors or line styles
    add = "_fut"
    s = 0 if ssp == "126" else 1

    # Combine hist + this SSP for the same experiment
    hist_ssp = df_avg_all[
        (df_avg_all['experiment'] == exp) 
         &(df_avg_all['ssp'].isin(['hist',ssp])) #for combined smoothening
         # &(df_avg_all['ssp'].isin([ssp])) #for per item smoothening
    ].copy()

    # Sort by time
    hist_ssp = hist_ssp.sort_values('year')

    # Smooth
    hist_ssp['runoff_smoothed'] = (
        hist_ssp['runoff']
        .rolling(window=11, center=True, min_periods=1)
        .mean()
    )
    
    hist_ssp['relative_runoff_smoothed'] = (
        hist_ssp['runoff_relative']
        .rolling(window=11, center=True, min_periods=1)
        .mean()
    )

    # Style settings
    ls = '--' if exp == "noi" else '-'
    lw = 1 if exp == "noi" else 2

    # Split into historical & future
    is_hist = hist_ssp['year'] < 2014
    is_future = hist_ssp['year'] >= 2014
    
    if exp =="irr":
        hist_ssp['relative_runoff_difference_smoothed'] = (
            hist_ssp['runoff_relative_difference']
            .rolling(window=11, center=True, min_periods=1)
            .mean()
        )
        
        # Plot hist part (no add)
        ax[0].plot(
            hist_ssp.loc[is_hist, 'year'],
            hist_ssp.loc[is_hist, 'relative_runoff_difference_smoothed']*100,
            # hist_ssp.loc[is_hist, 'runoff_relative_difference']*100,
            label=f"{exp.upper()} hist",
            color=colors[f'{exp}'][0],
            linestyle=ls, linewidth=lw
        )

        # Plot future part (add)
        ax[0].plot(
            hist_ssp.loc[is_future, 'year'],
            hist_ssp.loc[is_future, 'relative_runoff_difference_smoothed']*100,
            # hist_ssp.loc[is_future, 'runoff_relative_difference']*100,
            label=f"{exp.upper()} {ssp}",
            color=colors[f'{exp}{add}'][s],
            linestyle=ls, linewidth=lw
        )

    # Plot hist part (no add)
    # ax[0].plot(
    #     hist_ssp.loc[is_hist, 'year'],
    #     hist_ssp.loc[is_hist, 'relative_runoff_smoothed']*100,
    #     # hist_ssp.loc[is_hist, 'runoff_smoothed'],
    #     label=f"{exp.upper()} hist",
    #     color=colors[f'{exp}'][0],
    #     linestyle=ls, linewidth=lw
    # )

    # # Plot future part (add)
    # ax[0].plot(
    #     hist_ssp.loc[is_future, 'year'],
    #     hist_ssp.loc[is_future, 'relative_runoff_smoothed']*100,
    #     # hist_ssp.loc[is_hist, 'runoff_smoothed'],
    #     label=f"{exp.upper()} {ssp}",
    #     color=colors[f'{exp}{add}'][s],
    #     linestyle=ls, linewidth=lw
    # )
    
    peak_year=hist_ssp.loc[hist_ssp['runoff_smoothed'].idxmax(), 'year']
    if peak_year<=2014:
        color='black'
    else:
        color=colors[f'{exp}{add}'][s]
    # ax[0].axvline(peak_year, color=color, linestyle=ls, lw=1)


fig.text(
    0.5,  # to the right
    -0.02,         # middle of whisker
    'Years',
    va='center',
    ha='left',
    fontsize=12,
    # fontweight='bold',
    
)

fig.text(
    -0.01,  # to the right
    0.5,         # middle of whisker
    'Difference in relative Runoff [Irr-NoIrr, %]',
    va='center',
    ha='left',
    fontsize=12,
    # fontweight='bold'
    rotation=90
)

# peak_year = region_data.loc[region_data['runoff'].idxmax(), 'year']

# ax[0].axvline(2014, color='grey', linestyle='--', label='Historical/Future Split', lw=1)
# ax[0].axvline(peak_year, color='red', linestyle=':', label=f'Peak Year: {peak_year}', lw=1)

ax[0].set_title('High Mountain Asia', fontweight='bold')
ax[0].grid(True, color='grey', ls='--', lw=0.5)

avg_early = df_early_avg.groupby('experiment').mean()
avg_late = df_late_avg.groupby('experiment').mean()
inset_ax1 = inset_axes(ax[0], width="40%", height="40%", loc='upper left', borderpad=0)
inset_ax2 = inset_axes(ax[0], width="40%", height="40%", loc='lower right', borderpad=0)
# wedges, texts =  inset_ax1.pie(avg_early, labels=None, startangle=90, autopct=None, colors=component_colors)#lambda pct: f"{int(round(pct))}%"
wedges_outer, texts_outer = inset_ax1.pie(avg_early.loc[avg_late.index == 'noi'].values.flatten(),radius=1.0, labels=None, startangle=90,colors=component_colors_noirr,wedgeprops=dict(width=0.4, edgecolor='w'))
wedges_inner, texts_inner = inset_ax1.pie(avg_early.loc[avg_late.index == 'irr'].values.flatten(),radius=0.7,labels=None, startangle=90,colors=component_colors,wedgeprops=dict(width=0.4, edgecolor='w'))
centre_circle = plt.Circle((0, 0), 0.4, fc='white')
inset_ax1.add_artist(centre_circle)
inset_ax1.set(aspect='equal')
inset_ax1.set_title('')  # Remove default title position

wedges_outer, texts_outer = inset_ax2.pie(avg_late.loc[avg_late.index == 'noi'].values.flatten(),radius=1.0, labels=None, startangle=90,colors=component_colors_noirr,wedgeprops=dict(width=0.4, edgecolor='w'))
wedges_inner, texts_inner = inset_ax2.pie(avg_late.loc[avg_late.index == 'irr'].values.flatten(),radius=0.7,labels=None, startangle=90,colors=component_colors,wedgeprops=dict(width=0.4, edgecolor='w'))

    
centre_circle = plt.Circle((0, 0), 0.4, fc='white')
inset_ax2.add_artist(centre_circle)
inset_ax2.set(aspect='equal')
inset_ax2.set_title('')  # Remove default title position
inset_ax1.annotate('1985–2014', xy=(0.5, -0.15), ha='center', va='center', fontsize=10, xycoords='axes fraction')
inset_ax2.set_title('2044–2074', fontsize=10)

# Plot pie in lower-right


pie_patches = [
    mpatches.Patch(color=component_colors[0], label='Melt on glacier'),
    mpatches.Patch(color=component_colors_noirr[0], label='Melt on glacier NoIrr'),
    mlines.Line2D([0], [0], color='grey', linestyle='-',linewidth=1,label='Peak water'),
    mlines.Line2D([0], [0], color='grey', linestyle='--',linewidth=1,label='Peak water NoIrr'),
]

# Line legend (your original)
line_patches = [
    mpatches.Patch(color=colors['irr'][0], label='Historical (W5E5)'),
    mpatches.Patch(color=colors['noi'][0], label='Historical NoIrr'),
    mpatches.Patch(color=colors['irr_fut'][0], label='Future (CESM2), SSP-1.26'),
    # mpatches.Patch(color=colors['noi_fut'][0], label='Future NoIrr, SSP-1.26'),
    mpatches.Patch(color=colors['irr_fut'][1], label='Future (CESM2), SSP-3.70'),
    # mpatches.Patch(color=colors['noi_fut'][1], label='Future NoIrr, SSP-3.70'),
    
]

# Combine them
legend_patches =  line_patches + pie_patches 

# Add to figure/axes
fig.legend(handles=legend_patches, loc='lower center', frameon=False, ncols=5, bbox_to_anchor=(0.5,-0.14))

# plt.xlabel('Year')
# plt.ylabel('Runoff [km³]')
# plt.legend()   
plt.show()


f#%% Plot average seasonality in past and future

opath_df_monthly = os.path.join(wd_path,'masters','hydro_output_monthly_subregions.csv')
opath_df_annual = os.path.join(wd_path,'masters','hydro_output_annual_subregions.csv')


df_annual = pd.read_csv(opath_df_annual).reset_index()
df_monthly = pd.read_csv(opath_df_monthly, dtype={'ssp': str, 'rgi_subregion': str, 'experiment': str})[['year','month','experiment','ssp','member', 'runoff','rgi_subregion']]
df_monthly = df_monthly[df_monthly['runoff'] != 0.00]

# print(df_monthly['year'].unique())  
# print(df_monthly['month'].unique())  
# print(df_monthly['experiment'].unique())  
# print(df_monthly['ssp'].unique())       
# print(df_monthly['member'].unique())
# print(df_monthly['rgi_subregion'].unique())                    
                         
df_avg_monthly = (df_monthly.groupby(['month', 'experiment', 'ssp', 'rgi_subregion'])['runoff'].mean().reset_index())
df_avg_monthly_total = (df_avg_monthly.groupby(['month', 'experiment', 'ssp'])['runoff'].sum().reset_index())


#%% Cell 12: Make seasonal runoff charts for all years and for all ssps and experiments (to be improved)
fig,axes = plt.subplots(4,4,figsize=(12,8), sharex=True, sharey=False)
ax = axes.flatten()

c=1
for region in df_avg_monthly['rgi_subregion'].unique():
    print(region)
    region_data = df_avg_monthly[df_avg_monthly['rgi_subregion'] == region]
    # base_year =region_data[region_data['year']==1985][region_data['experiment']=='irr'].runoff.values
    peak_month_prev=0
    for (exp, ssp), group in region_data.groupby(['experiment', 'ssp']):
        peak_month = group.groupby('month')['runoff'].sum().idxmax()
        peak_flow = group.groupby('month')['runoff'].sum().max()
        offset=0
        if peak_month == peak_month_prev:
            offset+=0.1
        print(exp)
        if ssp == 'W5E5':
            add = ""
            s = 0  # default index
        else:
            add = "_fut"
            s = 0 if ssp == "126" else 1
        # print(region, exp, ssp, s)
    
        ls = '--' if exp == "noi" else '-'
        lw = 2 if exp == "noi" else 3
        ax[c].plot(group['month'], group['runoff'],
                 label=f"{exp.upper()} {ssp}",
                 color=colors[f'{exp}{add}'][s],
                 linestyle=ls, linewidth=lw)
        
        ax[c].axvline(peak_month, lw=1,ls=ls, color=colors[f'{exp}{add}'][s])
        
    ax[c].set_title(f'{region}')
    ax[c].set_ylim(0,12000)
    if c not in [4,8,12]:
        ax[c].set_yticklabels([])
    
    ax[c].grid(False)#, color='grey', ls='--', lw=0.5)
    # plt.tight_layout()
    c+=1
    
    
for (exp, ssp), group in df_avg_monthly_total.groupby(['experiment', 'ssp']):
    peak_month = group.groupby('month')['runoff'].sum().idxmax()

    if ssp == 'W5E5':
        add = ""
        s = 0  # default index
    else:
        add = "_fut"
        s = 0 if ssp == "126" else 1

    ls = '--' if exp == "noi" else '-'
    lw = 2 if exp == "noi" else 3

    ax[0].plot(group['month'], group['runoff'],
             label=f"{exp.upper()} {ssp}",
             color=colors[f'{exp}{add}'][s],
             linestyle=ls, linewidth=lw)
    ax[0].axvline(peak_month, ls=ls, lw=2, color=colors[f'{exp}{add}'][s])

ax[0].set_title('High Mountain Asia', fontweight='bold')
ax[0].grid(False)#, color='grey', ls='--', lw=0.5)


handles, labels = ax[0].get_legend_handles_labels()
# Place a shared legend below the entire figure
custom_labels = ['Future (CESM2) SSP126', 'Future (CESM2) SSP370', 'Historic (W5E5)', 'Future NoIrr (CESM2) SSP126', 'Future NoIrr (CESM2) SSP370', 'Historic NoIrr (W5E5)',]
# custom_labels = custom_labels[::-1]

fig.legend(handles, custom_labels, loc='lower center', ncol=3, fontsize=10, bbox_to_anchor=(0.5,-0.05))


fig.supxlabel("Year")
fig.supylabel('Runoff [km³]')
plt.tight_layout()
plt.show()




 #%% Test figures: Boxplot in spread of average runoff and seasonality mean
# --- Boxplot: Historical vs Future ---
boxplot = 'on'
if boxplot == 'on':
    
    palette = {
        'irr_hist': colors['irr'][0],
        'noi_hist': colors['noi'][0],
        'irr_126': colors['irr_fut'][0],
        'noi_126': colors['noi_fut'][0],
        'irr_370': colors['irr_fut'][1],
        'noi_370': colors['noi_fut'][1],
    }
    
    hue_order = [
    'irr_W5E5',        # Historic Irr
    'noi_W5E5',        # Historic NoIrr
    'irr_126',      # Future Irr SSP126
    'noi_126',      # Future NoIrr SSP126
    'irr_370',      # Future Irr SSP370
    'noi_370',      # Future NoIrr SSP370
]
    
    
    labels=['Historic', 'Historic NoIrr', 'Future SSP126', 'Future NoIrr SSP126', 'Future SSP370','Future NoIrr SSP370']
    
    
    # df_annual['period'] = df_annual['year'].apply(lambda x: 'Historical' if x <= 2014 else 'Future')
    df_annual['unique_id'] = df_annual['experiment'].astype(str) + '_' + df_annual['ssp'].astype(str)
    fig,ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(ax=ax, data=df_annual, x='unique_id', y='runoff', hue='unique_id', order = hue_order, palette=palette)
    ax.set_xticklabels(labels, rotation=45)  # Rotate for readability
    plt.title('Spread in average annual Runoff')
    plt.ylabel('Total Annual Runoff')
    plt.xlabel('Period')
    plt.tight_layout()
    plt.show()

# --- Seasonality Plot ---
seasonality_plot ="on"
if seasonality_plot == 'on':
    df_seasonality = df_monthly_all.groupby(['experiment', 'ssp', 'month'])['runoff'].mean().reset_index()
    df_seasonality=df_seasonality[df_annual['year']!=2014][df_annual['year']!=2074]
    df_seasonality['unique_id'] = df_seasonality['experiment'].astype(str) + '_' + df_seasonality['ssp'].astype(str)
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df_seasonality, x='month', y='runoff', hue='unique_id', palette=palette, style='experiment', lw=2)
    plt.title('Runoff Seasonality')
    plt.xlabel('Month')
    plt.ylabel('Mean Monthly Runoff [km$^3$]')
    plt.grid(False)
    plt.tight_layout()
    plt.show()
      
seasonality_plot_decadal = "off"



#%% Cell 13: Decadal average runoff change (test plot)


# Prepare
df_filtered = df_monthly_all[
    (df_monthly_all['year'] != 2014) & (df_monthly_all['year'] != 2074)
].copy()
df_filtered['decade'] = (df_filtered['year'] // 10) * 10
df_seasonality = df_filtered.groupby(['ssp', 'experiment', 'month', 'decade'])['runoff'].mean().reset_index()

# Define color mapping for experiments
# Sort decades to apply increasing alpha
decade_list = sorted(df_seasonality['decade'].unique())

# Create figure with one subplot per SSP
ssps = sorted(df_seasonality['ssp'].unique(), reverse=True)
fig, axes = plt.subplots(2,3, figsize=(10, 5), sharey=True)

# if len(ssps) == 1:
# axes = axes.flatten()  # make iterable if only one
cmap = cm.get_cmap('viridis')
palette = [cmap(i) for i in np.linspace(0.3, 0.9, 10)]

for i, ssp in enumerate(ssps):
    print(i)
    df_ssp = df_seasonality[df_seasonality['ssp'] == ssp]
    if ssp=="W5E5":
        add=""
        # i=0
    else:
        add="_fut"

    for e, experiment in enumerate(df_ssp['experiment'].unique()):
        ax = axes[e,i]
        df_exp = df_ssp[df_ssp['experiment'] == experiment]
        if ssp=="W5E5":
            c=0
        else:
            c=i
        for j, decade in enumerate(decade_list):
            df_dec = df_exp[df_exp['decade'] == decade]
            color = palette[j % len(palette)]
            # alpha = 0 + 0.7 * (j / (len(decade_list)-1))  # fade from 0.3 to 1.0
            ax.plot(
                df_dec['month'],
                df_dec['runoff'],
                label=f'{experiment} {decade}',
                color=color,#colors[f'{experiment}{add}'][c],
                # linestyle=linestyles[j],# alpha=alpha,
                linewidth=2,
                # alpha=alpha|
            )
        label = experiment.upper()
        ax.set_title(f'{label}, SSP {ssp}')
        ax.set_xlabel('Month')
        ax.grid(False)
    if i == 0:
        ax.set_ylabel('Monthly Runoff [km³]')

fig.suptitle('Decadal Average Runoff Seasonality per experiment and scenario', fontsize=16)
# fig.legend(title='Experiment & Decade', loc='upper right')
plt.tight_layout(rect=[0, 0, 0.9, 0.95])

legend_elements = []
for j, decade in enumerate(decade_list):
    line = Line2D(
        [0], [0],
        color = palette[j % len(palette)],               # use a neutral preview color
        # linestyle=linestyles[j % len(linestyles)],
        linewidth=2,
        label=f"{decade}s"
    )
    legend_elements.append(line)

# Add to the plot (use any ax or fig)
fig.legend(
    handles=legend_elements,
    title="Decade",
    loc='upper right',
    bbox_to_anchor=(1.01, 0.85)
)
plt.show()
#%% Cell 13b: make dataset peak water per glacier aggregated 

members = [4]
models = ["CESM2"]
experiments = ["irr", "noi"]
ssps = ["126", "370"]

# --- Data accumulation ---
df_monthly_all = []
df_runoff_shares=[]

base_ds = pd.read_csv(f"{wd_path}/masters/master_gdirs_r3_a1_rgi_date_A_V_RGIreg.csv", usecols=['rgi_id','rgi_subregion']).set_index('rgi_id').to_xarray() #open rgi subregions per rgi_id

rgi_ids=[]
for ex in experiments:
    for m, model in enumerate(models):
        for member in range(members[m]):
            if member >= 1:
                sample_id = f"{model}.00{member}"
                if ex == "irr" and member == 1:
                    hydro_data_past = f"{wd_path}/summary/hydro_run_output_baseline_W5E5.000.nc" #open historic runoff data, only once
                elif ex == "noi":
                    hydro_data_past = f"{wd_path}/summary/hydro_run_output_perturbed_{sample_id}.nc" #open future runoff data
                else:
                    continue
                try:
                    with xr.open_dataset(hydro_data_past) as ds:
                        if m==0 and member==1:
                            rgi_ids = ds.rgi_id.values #read out rgi_ids
                        
                            #turn on for monthly
                        # melt_on = ds['melt_on_glacier_monthly'] * 1e-9 #only working with monthly components, can delete for already averaged, but than also delete summation later
                        # melt_off = ds['melt_off_glacier_monthly'] * 1e-9 #convert the runoff components to the right unit
                        # prcp_on = ds['liq_prcp_on_glacier_monthly'] * 1e-9
                        # prcp_off = ds['liq_prcp_off_glacier_monthly'] * 1e-9
                        # snow_on = ds['snowfall_on_glacier_monthly'] * 1e-9
                        # snow_off = ds['snowfall_off_glacier_monthly'] * 1e-9
                        
                        melt_on = ds['melt_on_glacier'] * 1e-9 #only working with monthly components, can delete for already averaged, but than also delete summation later
                        melt_off = ds['melt_off_glacier'] * 1e-9 #convert the runoff components to the right unit
                        prcp_on = ds['liq_prcp_on_glacier'] * 1e-9
                        prcp_off = ds['liq_prcp_off_glacier'] * 1e-9
                        snow_on = ds['snowfall_on_glacier'] * 1e-9
                        snow_off = ds['snowfall_off_glacier'] * 1e-9
                        
                        
                        total_runoff = melt_on + melt_off + prcp_on + prcp_off + snow_on + snow_off #sum all the runoff components
                        runoff = xr.merge([total_runoff.to_dataset(name='runoff'), base_ds]).set_coords('rgi_subregion') #merge the runoff data with the subregion ids

                        df_runoff = runoff.to_dataframe().reset_index()[['time',  'rgi_subregion', 'rgi_id', 'runoff']]                
                        df_runoff = df_runoff.rename(columns={'time': 'year'})
                        # df_runoff = runoff.to_dataframe().reset_index()[['time', 'month_2d', 'rgi_subregion', 'rgi_id', 'runoff']]                
                        # df_runoff = df_runoff.rename(columns={'time': 'year', 'month_2d': 'month'})
                        df_runoff['experiment'] = ex
                        df_runoff['ssp'] = "hist"
                        df_runoff['member'] = member
                        df_monthly_all.append(df_runoff) #crate table with experiments, ssps and member data
    
                except Exception as e:
                    print(f"Error loading {hydro_data_past}: {e}")

    #repeat the same but then for ssps
    for ssp in ssps:
        for member in range(1, 4):
            sample_id = f"CESM2.00{member}"
            if ex == "irr":
                hydro_data_path = f"{wd_path}/summary/climate_run_output_perturbed_SSP{ssp}_{sample_id}_hydro.nc"
            else:
                hydro_data_path = f"{wd_path}/summary/climate_run_output_perturbed_SSP{ssp}_{sample_id}_noi_bias_hydro.nc"
            try:
                with xr.open_dataset(hydro_data_path) as ds:
                    
                    melt_on = ds['melt_on_glacier'] * 1e-9
                    melt_off = ds['melt_off_glacier'] * 1e-9
                    prcp_on = ds['liq_prcp_on_glacier'] * 1e-9
                    prcp_off = ds['liq_prcp_off_glacier'] * 1e-9
                    snow_on = ds['snowfall_on_glacier'] * 1e-9
                    snow_off = ds['snowfall_off_glacier'] * 1e-9


                    # melt_on = ds['melt_on_glacier_monthly'] * 1e-9
                    # melt_off = ds['melt_off_glacier_monthly'] * 1e-9
                    # prcp_on = ds['liq_prcp_on_glacier_monthly'] * 1e-9
                    # prcp_off = ds['liq_prcp_off_glacier_monthly'] * 1e-9
                    # snow_on = ds['snowfall_on_glacier_monthly'] * 1e-9
                    # snow_off = ds['snowfall_off_glacier_monthly'] * 1e-9                    
                    
                    total_runoff = melt_on + melt_off + prcp_on + prcp_off + snow_on + snow_off #sum all the runoff components
                    
                    runoff = xr.merge([total_runoff.to_dataset(name='runoff'), base_ds]).set_coords('rgi_subregion') #merge the runoff data with the subregion ids

                    df_runoff = runoff.to_dataframe().reset_index()[['time', 'rgi_subregion', 'rgi_id', 'runoff']]                
                    df_runoff = df_runoff.rename(columns={'time': 'year'})
                    
                    # df_runoff = runoff.to_dataframe().reset_index()[['time', 'month_2d', 'rgi_subregion', 'rgi_id', 'runoff']]                
                    # df_runoff = df_runoff.rename(columns={'time': 'year', 'month_2d': 'month'})
                    df_runoff['experiment'] = ex
                    df_runoff['ssp'] = ssp
                    df_runoff['member'] = member
                    df_monthly_all.append(df_runoff) #crate table with experiments, ssps and member data
                    
            except Exception as e:
                print(f"Error loading {hydro_data_path} pt 2: {e}")

# --- Combine and analyze ---
df_monthly_all_concat = pd.concat(df_monthly_all, ignore_index=True)
df_annual=df_monthly_all_concat #if already selected upfront that annual instead of monthly
# df_annual = df_monthly_all.groupby(['year', 'experiment', 'ssp', 'member', 'rgi_subregion', 'rgi_id'])['runoff'].sum().reset_index()

#filepaths for only melt runoff components
# opath_df_monthly = os.path.join(wd_path,'masters','hydro_output_monthly_subregions_melton_only_per_rgi_id.csv')
# opath_df_annual = os.path.join(wd_path,'masters','hydro_output_annual_subregions_melton_only_per_rgi_id.csv')

#file paths for only melt and precipitation runoff components
# opath_df_monthly = os.path.join(wd_path,'masters','hydro_output_monthly_subregions_meltprcpon_only_per_rgi_id.csv')
# opath_df_annual = os.path.join(wd_path,'masters','hydro_output_annual_subregions_meltprcpon_only_per_rgi_id.csv')

#file paths including all runoff components
# opath_df_monthly = os.path.join(wd_path,'masters','hydro_output_monthly_subregions_per_rgi_id.csv')
opath_df_annual = os.path.join(wd_path,'masters','hydro_output_annual_subregions_per_rgi_id_preselected.csv')


# df_monthly_all.to_csv(opath_df_monthly)
df_annual.to_csv(opath_df_annual)

#%%  13c Process and calculate the peak water time per glacier

#operation is quite time consuming as contains many rgi_ids, years combinations
opath_df_annual = os.path.join(wd_path,'masters','hydro_output_annual_subregions_per_rgi_id_preselected.csv')
df_annual = pd.read_csv(opath_df_annual)
df_annual = df_annual.sort_values(['rgi_id', 'experiment', 'ssp', 'member', 'year']) #sort, becuase rolling only works for sorted years
df_annual['ssp'] = df_annual['ssp'].astype(str) #assure same type, otherwise strange grouping
df_annual['runoff_11yr_rolling'] = (
    df_annual.groupby(['rgi_id', 'experiment', 'ssp', 'member'])['runoff']
      .transform(lambda x: x.rolling(window=11, center=True, min_periods=1).mean()) #using transform returns a series aligned to the orgiinal df index, so ungrouped columns remain
)

#summarize the peak water year in a new table for the groupby columns (e.g. I want to know per rgi_id, exp, ssp, member)
group_cols = ['rgi_id', 'experiment', 'ssp', 'member']
# Get the index of max rolling runoff within each group
idx_max = df_annual.groupby(group_cols)['runoff_11yr_rolling'].idxmax()
idx_max_clean = idx_max.dropna().astype(int) #drop out na's if applicable

# Create a new column with NaN everywhere
df_annual['year_of_max_runoff'] = pd.NA

# Fill the year only at max index positions
df_annual.loc[idx_max_clean, 'year_of_max_runoff'] = df_annual.loc[idx_max_clean, 'year'] #look up the index id amongst years

summary = (
    df_annual.loc[idx_max_clean, group_cols + ['year', 'runoff_11yr_rolling']]
    .rename(columns={'year': 'year_of_max_runoff', 'runoff_11yr_rolling': 'max_runoff_11yr'})
    .reset_index(drop=True)
)

opath_df_summary = os.path.join(wd_path,'masters','hydro_output_peakwater_per_rgi_id.csv')
summary.to_csv(opath_df_summary)
#%% Cell 13d: Link rgi_id to coordinates and plot peak water with color bar




#%% Cell 14: Open and process reference data

rho = 0.917 #kg/m3
Rounce_path = f"/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/04. Reference/01. Climate data/Rounce/Global_reg_allvns_50sets_2000_2100-ssps.nc"#"R13_glac_mass_annual_50sets_2000_2100-rcp26.nc"
df_rounce_all = xr.open_dataset(Rounce_path)     

df_rounce = df_rounce_all.where(df_rounce_all.Region.isin(['Central Asia','South Asia West', 'South Asia East']), drop=True).sum(dim='region')
df_rounce['reg_mass_annual']/= rho*10e9

plt.figure(figsize=(8, 5))
for s,scenario in enumerate(df_rounce.Scenario.values):
    data = df_rounce.sel(scenario=s)
    climate_models = df_rounce['Climate_Model'].values
    model_indices = df_rounce['model'].values
    if s==0:
        valid_model_indices = model_indices[[not any(model.startswith(prefix) for prefix in ['BCC', 'FGOALS', 'CESM2', 'INM', 'MPI', 'NorESM']) for model in climate_models]]
        data = data.sel(model=valid_model_indices)
    models=data.Climate_Model.values
    mean_series_base_year = data.reg_mass_annual.mean(dim='model').sel(year=2000).values
    mean_series = data.reg_mass_annual.mean(dim='model').values
    for m, model in enumerate(models):
        print(model,m)
        # try:
        try:
        # Ensure data exists at base year (e.g., 2000)
            base_yr = data.sel(model=m).sel(year=2000).reg_mass_annual.values
            if not np.isnan(base_yr):  # Optional: skip NaNs
                plt.plot(data.sel(model=m).reg_mass_annual.values/base_yr*100, label=model, alpha=0.4, color=ssp_colors[scenario], ls=':')
                # plt.plot(data.year.values, data.sel(model=m).reg_mass_annual.values, label=model, alpha=0.4, color=ssp_colors[scenario], ls=':')
        except Exception as e:
            print(e)
            continue
    plt.plot(mean_series/mean_series_base_year*100, label='Multi-model mean', color=ssp_colors[scenario], ls='-', linewidth=2.5)
    # plt.plot(data.year.values, mean_series, label='Multi-model mean', color=ssp_colors[scenario], ls='-', linewidth=2.5)

plt.title(f'Regional Glacier Mass Change Rounce et al. (2023))')
# plt.ylabel('Remaining Glacier Volume [km${^3}$]')
plt.ylabel('Remaining Glacier Mass [%]')#' m w.e. yr$^{-1}$')
plt.xlabel('Year')
# plt.legend()
# plt.grid(True)
plt.tight_layout()
plt.show()

# df_rounce.reg_mass_annual.plot(colors='scenario')


#%% CEll 15: Plot reference data on top of our future simulations

exp = ["IRR"]#, "NOI"]#only focus on irr for comparison
regions = [13, 14, 15]
subregions = [9, 3, 3]
ssps = ["126", "370"]
members_averages = [4]
models_shortlist = ["CESM2"]

fig, axes = plt.subplots(figsize=(12,7), sharey=True )  # create a new figure

# define the variables for p;lotting

variable_names = ["Volume"]
variable_axes = ["Volume compared to 1985 All Forcings scenario [%]"]
use_multiprocessing = False

plt.rcParams.update({'font.size': 12})

#Plot data for future our model
axes_dict = {}  # Dictionary to store axes for later use
resp_values_dict={}
linestyles = ['solid', 'dashed']
past_ds = xr.open_dataset(f"{wd_path}/masters/master_volume_subregion_past.nc")
# future_ds = xr.open_dataset(f"{wd_path}masters/master_volume_subregion_future.nc")
future_ds = xr.open_dataset(f"{wd_path}/masters/master_volume_subregion_future_noi_bias.nc")
initial_vol = past_ds.sel(time=1985).volume*10e-9
initial_volume_big=initial_vol.sel(exp="IRR").sel(sample_id="CESM2.001").sum(dim='subregion')
initial_volume_pct_rounce = (past_ds.sel(time=2000).sel(exp="IRR").sel(sample_id="CESM2.001").volume.sum(dim='subregion')/initial_vol.sel(exp="IRR").sel(sample_id="CESM2.001").sum(dim='subregion'))*100
p=0

p+=1

past_ds['volume'] = past_ds['volume'] * 1e-9
future_ds['volume'] = future_ds['volume'] * 1e-9

for time in ["past","future"]:
    for e, ex in enumerate(exp):
        color = colors[ex.lower()][0]
        if ex=="IRR":
            ls="solid"
        else:
            ls="dashed"
        if ex =="IRR" and time=="past":
            # axes_dict[region_id].plot(past_ds.time, past_ds.volume.sel(subregion=region_id).sel(exp=ex).sel(sample_id="CESM2.001")/initial_volume*100-100, linestyle='solid', color=color)
            axes.plot(past_ds.time, past_ds.volume.sel(exp=ex).sel(sample_id="CESM2.001").sum(dim='subregion'), linestyle=ls, color=color, label="Historic W5E5 individual member", lw=2)
            axes.plot(past_ds.time, past_ds.volume.sel(exp=ex).sel(sample_id="CESM2.001").sum(dim='subregion'), linestyle=ls, color=color, label="Historic W5E5 individual member", lw=2)
            # axes.plot(past_ds.time, past_ds.volume.sel(exp=ex).sel(sample_id="CESM2.001").sum(dim='subregion')/initial_volume_big*100-100, linestyle=ls, color=color, label="Historic W5E5 individual member", lw=2)
            # axes.plot(past_ds.time, past_ds.volume.sel(exp=ex).sel(sample_id="CESM2.001").sum(dim='subregion')/initial_volume_big*100-100, linestyle=ls, color=color, label="Historic W5E5 individual member", lw=2)
        
               
        else:
            volume_evolution_data=[]
            for m, model in enumerate(models_shortlist):
                for member in range(members_averages[m]):
                    if member<1:
                        continue
                    if member >= 1: #skip 0
                        sample_id = f"{model}.00{member:01d}"
                        print(sample_id)
                        
                        for (s, ssp) in enumerate(ssps):
                            ax = axes
                            color_fut = colors[f"{ex.lower()}_fut"][s]
                            volume_evolution_data = []  # reset per SSP
                    
                            for m, model in enumerate(models_shortlist):
                                for member in range(members_averages[m]):
                                    if member < 1:
                                        continue
                                    sample_id = f"{model}.00{member:01d}"
                                    
                                    if s==0 and member==1 and time=="past":
                                        # axes_dict[region_id].plot(past_ds.time, past_ds.volume.sel(subregion=region_id).sel(exp='NOI').sel(sample_id=sample_id)/initial_volume*100-100, linestyle='dotted', color=color)
                                        if member==1:
                                            label="Historic NoIrr individual member"
                                        else:
                                            label='_nolegend_'
                                        
                                        # axes.plot(past_ds.time, past_ds.volume.sel(exp='NOI').sel(sample_id=sample_id).sum(dim='subregion'),
                                        #        color=color, lw=1, linestyle='dotted', label=label)  
                                        
                                        #uncomment for relative
                                        # axes.plot(past_ds.time, past_ds.volume.sel(exp='NOI').sel(sample_id=sample_id).sum(dim='subregion')/initial_volume_big*100-100,
                                        #        color=color, lw=1, linestyle='dotted', label=label)  
                    
                                    try:
                                        #uncomment for relative
                                        # volume_evolution = future_ds.volume.sel(exp=ex).sel(sample_id=sample_id).sel(ssp=ssp).sum(dim='subregion') / initial_volume_big * 100 - 100
                                        volume_evolution = future_ds.volume.sel(exp=ex).sel(sample_id=sample_id).sel(ssp=ssp).sum(dim='subregion') 
                                        volume_evolution_data.append(volume_evolution)
                                        
                                        # Plot individual lines if needed
                                        if member == 1: #and region_id == "13-01":
                                            label = f'Future {ex.lower()} SSP-{ssp} individual member'
                                        else:
                                            label = '_nolegend_'
                                        # ax.plot(future_ds.time, volume_evolution, linestyle='dashed', color=color_fut, lw=1, label=label)
                    
                                    except KeyError:
                                        print(f"Data missing for {sample_id}, SSP{ssp}")
                    
                            # Compute min/max across members
                            if volume_evolution_data:
                                # Convert to DataArray for easier handling
                                volume_stack = xr.concat(volume_evolution_data, dim='member')
                                min_vol = volume_stack.min(dim='member')
                                max_vol = volume_stack.max(dim='member')
                    
                                ax.fill_between(
                                    future_ds.time,
                                    min_vol,
                                    max_vol,
                                    color=color_fut,
                                    alpha=0.1,
                                    label=f'SSP{ssp} envelope'
                                )

            
                                
                        
color=colors["noi"][0]    

# std_big = past_ds.volume.sel(exp='NOI').sum(dim='subregion').std(dim='sample_id')
# sum_big = past_ds.volume.sel(exp='NOI').sel(sample_id='3-member-avg').sum(dim='subregion')
# min_big = past_ds.volume.sel(exp='NOI').sum(dim='subregion').min(dim='sample_id')
# max_big = past_ds.volume.sel(exp='NOI').sum(dim='subregion').max(dim='sample_id')

# axes.plot(past_ds.time, sum_big, linestyle='dashed', color=color, label="Historic NoIrr 3-member average", lw=3)
# axes.plot(past_ds.time, sum_big, linestyle='dashed', color=color, label="Historic NoIrr 3-member average", lw=3)
# axes.fill_between(past_ds.time, min_big,max_big, color=color, alpha=0.2)

# axes.fill_between(past_ds.time, sum_big-std_big,sum_big+std_big, color=color, alpha=0.2)
annotations =np.zeros((2,2))
ssp_anno=['1.26', '3.70']
for (s,ssp) in enumerate(ssps):
    
    ax = axes
    for (e,ex) in enumerate(exp):
        if ex=="IRR":
            ls="solid"
        else:
            ls="dashed"
            
        if s==0:
            marker='>'
        else:
            marker='o'
        print("init")
        color=colors[f"{ex.lower()}_fut"][s]
        std_big = future_ds.volume.sel(exp=ex).sel(ssp=ssp).sum(dim='subregion').std(dim='sample_id')
        sum_big = future_ds.volume.sel(exp=ex).sel(ssp=ssp).sel(sample_id='3-member-avg').sum(dim='subregion')
        
        #uncomment for relative
        # std_big = (future_ds.volume.sel(exp=ex).sel(ssp=ssp).sum(dim='subregion').std(dim='sample_id')/initial_volume_big)*100-100
        # sum_big = (future_ds.volume.sel(exp=ex).sel(ssp=ssp).sel(sample_id='3-member-avg').sum(dim='subregion')/initial_volume_big)*100-100
        ax.plot(future_ds.time, sum_big,  color=color, label=f"Future {ex} SSP 3-member average", ls=ls, linewidth=6)
        annotation=np.round(sum_big[-1].values, decimals=2)
        annotations[s,e] = annotation
        
        
 
desired_years = [1985, 2015, 2045, 2075]
axes.set_xlim(1985,2100)
# axes.set_ylim(-72,4)

# Apply to the axis
axes.set_xticks(desired_years)
axes.set_xticklabels([str(year) for year in desired_years])
# ax_big.set_yticks([])
plt.subplots_adjust(hspace=0.5, wspace=0.05)#, wspace=0.2)
axes.set_ylabel("Volume change (%, vs. 1985 historic")

legend_patches = [mpatches.Patch(color=colors['irr'][0], label='Historical (W5E5)           '),
                  mpatches.Patch(color=colors['noi'][0], label='Historical NoIrr'),
                  mpatches.Patch(color=colors['irr_fut'][0], label='Future (CESM2), SSP-1.26'),
                  mpatches.Patch(color=colors['noi_fut'][0], label='Future NoIrr, SSP-1.26'),
                  mpatches.Patch(color=colors['irr_fut'][1], label='Future (CESM2), SSP-3.70'),
                    mpatches.Patch(color=colors['noi_fut'][1], label='Future NoIrr, SSP-3.70'),
                    # Line2D([0], [0], color='grey', linestyle='dotted', linewidth=3, label=f'individual member mean'),
                    ]
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.3)     
                         
# plt.tight_layout()
# fig.legend(handles=legend_patches, loc='upper right', ncols=1, bbox_to_anchor=(0.9,0.88), frameon=False, fontsize=12)#,

# #Add Rounce data 
# rho = 0.917#*10**3 #kg/m3
# Rounce_path = f"/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/04. Reference/01. Climate data/Rounce/Global_reg_allvns_50sets_2000_2100-ssps.nc"#"R13_glac_mass_annual_50sets_2000_2100-rcp26.nc"
# df_rounce_all = xr.open_dataset(Rounce_path)     

# df_rounce = df_rounce_all.where(df_rounce_all.Region.isin(['Central Asia','South Asia West', 'South Asia East']), drop=True).sum(dim='region')
# df_rounce['reg_mass_annual']/=rho #/= rho

# for s,scenario in enumerate(df_rounce.Scenario.values):
#     data = df_rounce.sel(scenario=s)
#     climate_models = df_rounce['Climate_Model'].values
#     model_indices = df_rounce['model'].values
#     if s==0:
#         valid_model_indices = model_indices[[not any(model.startswith(prefix) for prefix in ['BCC', 'FGOALS', 'CESM2', 'INM', 'MPI', 'NorESM']) for model in climate_models]]
#         data = data.sel(model=valid_model_indices)
#     models=data.Climate_Model.values
#     mean_series_base_year = data.reg_mass_annual.mean(dim='model').sel(year=2000).values
#     mean_series = data.reg_mass_annual.mean(dim='model').values
#     for m, model in enumerate(models):
#         print(model,m)
#         # try:
#         try:
#         # Ensure data exists at base year (e.g., 2000)
#             base_yr = data.sel(model=m).sel(year=2000).reg_mass_annual.values
#             if not np.isnan(base_yr):  # Optional: skip NaNs
#                 plt.plot(data.sel(model=m).reg_mass_annual.values/base_yr*100, label=model, alpha=0.4, color=ssp_colors[scenario], ls=':')
#                 # plt.plot(data.year.values, data.sel(model=m).reg_mass_annual.values, label=model, alpha=0.4, color=ssp_colors[scenario], ls=':')
#         except Exception as e:
#             print(e)
#             continue
#     rounce_offset=(initial_volume_pct_rounce.values-100)
#     rounce_ts = (mean_series-mean_series_base_year)/mean_series_base_year*100 + rounce_offset
    
#     # axes.plot(data.year.values, rounce_ts, label=f'Rounce {ssp}', color=ssp_colors[scenario], ls=':', linewidth=3)
#     print("init")
#     glacier_area_share= 0.78
#     plt.plot(data.year.values, mean_series*glacier_area_share, label='Multi-model mean', color=ssp_colors[scenario], ls='-', linewidth=2.5)

# # plt.title(f'Regional Glacier Mass Change Rounce et al. (2023))')
# # # plt.ylabel('Remaining Glacier Volume [km${^3}$]')
# # plt.ylabel('Remaining Glacier Mass [%]')#' m w.e. yr$^{-1}$')
# # plt.xlabel('Year')
# # plt.legend()
# # plt.grid(True)


#Add harry and Aguayo data

Aguayo_path = f"/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/04. Reference/01. Climate data/Aguayo/merged_glacier_simulations.nc"#"R13_glac_mass_annual_50sets_2000_2100-rcp26.nc"
ds_aguayo_all = xr.open_dataset(Aguayo_path)
# fig, ax = plt.subplots(figsize=(10, 6))
markers = ['o', 'x', '^']
ls =[':','-.', '--']
for v, var in enumerate(['glogem', 'oggm', 'pygem']):
    da=ds_aguayo_all[var]
    print(len(da.model.values))

    # Loop over each scenario
    for scenario in ds_aguayo_all['scenario'].values:
        if scenario in ['ssp126','ssp370']:
            # Loop over each model
            da_list = []
            print(da)
            
            for model in da['model'].values:
                
                da_=da.sel(model=model, scenario=scenario)
                all_zero = (da_ < 1e-10)#.all(dim='model') #add tollerance for zero mask
                da_single = da_.where(~all_zero)
                print(model, da_single)
                da_list.append(da_single)
            
                # Optional: plot here if you want
                if model=="CESM2":
                    ax.plot(ds_aguayo_all['year'], da_single, ls=ls[v], lw=3,
                            color=colors_ssp[scenario],
                            # label=f'{scenario} - {model}') # Select the time series
                            )
    
            da_stack = xr.concat(da_list, dim='model')
    
            # Now compute stats
            da_mean = da_stack.mean(dim='model', skipna=True)
            da_min = da_stack.min(dim='model', skipna=True)
            da_max = da_stack.max(dim='model', skipna=True)
        
            # ax.plot(ds_aguayo_all['year'], da_mean, ls = ls[v],label=f'{scenario} - {var}', color=colors_ssp[scenario],lw=3)
            # Plot the shaded min-max range
            # ax.fill_between(ds_aguayo_all['year'], da_min, da_max, alpha=0.1, color=colors_ssp[scenario])
            

ax.set_title(f'Volume evolution over time for CESM2')
ax.set_xlabel('Year')
ax.set_ylabel('Volume [km$^3$]')
    # ax.legend(fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.tight_layout()
    # plt.show()
    

ssp_patches = [
    mpatches.Patch(color=color, label=scenario.upper())
    for scenario, color in colors_ssp.items()
]
    
line_legend = [
    mpatches.Patch(color='black', label='Historical (W5E5)'),
    Line2D([0], [0], color='grey', linestyle='solid', label='simulations'),
    Line2D([0], [0], color='grey', linestyle='dotted', label='GloGEM'),
    Line2D([0], [0], color='grey', linestyle='dashed', label='PyGEM'),
    Line2D([0], [0], color='grey', linestyle='dashdot', label='OGGM')

]

legend_patches=ssp_patches + line_legend
# Then pass this list to legend
ax.legend(handles=legend_patches, ncols=5, bbox_to_anchor=(0.5,-0.3), loc='lower center')

plt.tight_layout()
plt.show()

# df_rounce.reg_mass_annual.plot(colors='scenario')

#%% Cell 16: Plot reference data on top of our future simulations - relative to 2015

exp = ["IRR"]#, "NOI"]#only focus on irr for comparison
regions = [13, 14, 15]
subregions = [9, 3, 3]
ssps = ["126", "370"]
members_averages = [4]
models_shortlist = ["CESM2"]

fig, axes = plt.subplots(figsize=(12,7), sharey=True )  # create a new figure

# define the variables for p;lotting

variable_names = ["Volume"]
variable_axes = ["Volume compared to 1985 All Forcings scenario [%]"]
use_multiprocessing = False

plt.rcParams.update({'font.size': 12})

#Plot data for future our model
axes_dict = {}  # Dictionary to store axes for later use
resp_values_dict={}
linestyles = ['solid', 'dashed']
past_ds = xr.open_dataset(f"{wd_path}/masters/master_volume_subregion_past.nc")
# future_ds = xr.open_dataset(f"{wd_path}masters/master_volume_subregion_future.nc")
future_ds = xr.open_dataset(f"{wd_path}/masters/master_volume_subregion_future_noi_bias.nc")

initial_volume_pct_rounce = (past_ds.sel(time=2000).sel(exp="IRR").sel(sample_id="CESM2.001").volume.sum(dim='subregion')/initial_vol.sel(exp="IRR").sel(sample_id="CESM2.001").sum(dim='subregion'))*100
p=0

p+=1

past_ds['volume'] = past_ds['volume'] * 1e-9
future_ds['volume'] = future_ds['volume'] * 1e-9

for time in ["past","future"]:
    for e, ex in enumerate(exp):
        color = colors[ex.lower()][0]
        if ex=="IRR":
            ls="solid"
        else:
            ls="dashed"
               
        if time=="future":
            volume_evolution_data=[]
            for m, model in enumerate(models_shortlist):
                for member in range(members_averages[m]):
                    if member<1:
                        continue
                    if member >= 1: #skip 0
                        sample_id = f"{model}.00{member:01d}"
                        print(sample_id)
                        
                        for (s, ssp) in enumerate(ssps):
                            ax = axes
                            color_fut = colors[f"{ex.lower()}_fut"][s]
                            volume_evolution_data = []  # reset per SSP
                    
                            for m, model in enumerate(models_shortlist):
                                for member in range(members_averages[m]):
                                    if member < 1:
                                        continue
                                    sample_id = f"{model}.00{member:01d}"
                                    
                                    if s==0 and member==1 and time=="past":
                                        # axes_dict[region_id].plot(past_ds.time, past_ds.volume.sel(subregion=region_id).sel(exp='NOI').sel(sample_id=sample_id)/initial_volume*100-100, linestyle='dotted', color=color)
                                        if member==1:
                                            label="Historic NoIrr individual member"
                                        else:
                                            label='_nolegend_'
                                        
                                    try:
                                        #uncomment for relative
                                        volume_evolution = future_ds.volume.sel(exp=ex).sel(sample_id=sample_id).sel(ssp=ssp).sum(dim='subregion') / initial_volume_big.sel(ssp=ssp) * 100 - 100
                                        # volume_evolution = future_ds.volume.sel(exp=ex).sel(sample_id=sample_id).sel(ssp=ssp).sum(dim='subregion') 
                                        volume_evolution_data.append(volume_evolution)
                                        
                                        # Plot individual lines if needed
                                        if member == 1: #and region_id == "13-01":
                                            label = f'Future {ex.lower()} SSP-{ssp} individual member'
                                        else:
                                            label = '_nolegend_'
                                            
                                        #uncomment for plotting individual members
                                        # ax.plot(future_ds.time, volume_evolution, linestyle='dashed', color=color_fut, lw=1, label=label)
                    
                                    except KeyError:
                                        print(f"Data missing for {sample_id}, SSP{ssp}")
                    
                            # Compute min/max across members for each model
                            if volume_evolution_data:
                                # Convert to DataArray for easier handling
                                volume_stack = xr.concat(volume_evolution_data, dim='member')
                                min_vol = volume_stack.min(dim='member')
                                max_vol = volume_stack.max(dim='member')
                                
                                # ax.fill_between(
                                #     future_ds.time,
                                #     min_vol,
                                #     max_vol,
                                #     color=color_fut,
                                #     alpha=0.1,
                                #     label=f'SSP{ssp} envelope'
                                # )

            
                                
                        
annotations =np.zeros((2,2))
ssp_anno=['1.26', '3.70']
for (s,ssp) in enumerate(ssps):
    
    ax = axes
    for (e,ex) in enumerate(exp):
        if ex=="IRR":
            ls="solid"
        else:
            ls="dashed"
            
        if s==0:
            marker='>'
        else:
            marker='o'
        print("init")
        initial_vol = future_ds.sel(time=2015).sel(ssp=ssp).volume
        initial_volume_big=initial_vol.sel(exp="IRR").sel(sample_id="3-member-avg").sum(dim='subregion')
        
        color=colors[f"{ex.lower()}_fut"][s]
        filtered = future_ds.volume.sel(exp=ex, ssp=ssp).sel(sample_id=future_ds.sample_id != "3-member-avg").sum(dim='subregion')
        std_big = filtered.std(dim='sample_id')
        sum_big = future_ds.volume.sel(exp=ex).sel(ssp=ssp).sel(sample_id='3-member-avg').sum(dim='subregion')
        #uncomment for relative
        sum_big = (future_ds.volume.sel(exp=ex).sel(ssp=ssp).sel(sample_id='3-member-avg').sum(dim='subregion'))/initial_volume_big*100
        #uncomment for relative
        # std_big = (future_ds.volume.sel(exp=ex).sel(ssp=ssp).sum(dim='subregion').std(dim='sample_id')/initial_volume_big)*100-100
        # sum_big = (future_ds.volume.sel(exp=ex).sel(ssp=ssp).sel(sample_id='3-member-avg').sum(dim='subregion')/initial_volume_big)*100-100
        ax.plot(future_ds.time, sum_big,  color=color, label=f"Future {ex} SSP 3-member average", ls=ls, linewidth=6)
        annotation=np.round(sum_big[-1].values, decimals=2)
        annotations[s,e] = annotation
        
        
 
desired_years = [1985, 2015, 2045, 2075]
axes.set_xlim(1985,2100)
# axes.set_ylim(-72,4)

# Apply to the axis
axes.set_xticks(desired_years)
axes.set_xticklabels([str(year) for year in desired_years])
# ax_big.set_yticks([])
plt.subplots_adjust(hspace=0.5, wspace=0.05)#, wspace=0.2)
axes.set_ylabel("Volume change (%, vs. 1985 historic")

legend_patches = [mpatches.Patch(color=colors['irr'][0], label='Historical (W5E5)           '),
                  mpatches.Patch(color=colors['noi'][0], label='Historical NoIrr'),
                  mpatches.Patch(color=colors['irr_fut'][0], label='Future (CESM2), SSP-1.26'),
                  mpatches.Patch(color=colors['noi_fut'][0], label='Future NoIrr, SSP-1.26'),
                  mpatches.Patch(color=colors['irr_fut'][1], label='Future (CESM2), SSP-3.70'),
                    mpatches.Patch(color=colors['noi_fut'][1], label='Future NoIrr, SSP-3.70'),
                    # Line2D([0], [0], color='grey', linestyle='dotted', linewidth=3, label=f'individual member mean'),
                    ]
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.3)     
                         
# plt.tight_layout()
# fig.legend(handles=legend_patches, loc='upper right', ncols=1, bbox_to_anchor=(0.9,0.88), frameon=False, fontsize=12)#,

# #Add Rounce data 
# rho = 0.917#*10**3 #kg/m3
# Rounce_path = f"/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/04. Reference/01. Climate data/Rounce/Global_reg_allvns_50sets_2000_2100-ssps.nc"#"R13_glac_mass_annual_50sets_2000_2100-rcp26.nc"
# df_rounce_all = xr.open_dataset(Rounce_path)     

# df_rounce = df_rounce_all.where(df_rounce_all.Region.isin(['Central Asia','South Asia West', 'South Asia East']), drop=True).sum(dim='region')
# df_rounce['reg_mass_annual']/=rho #/= rho

# for s,scenario in enumerate(df_rounce.Scenario.values):
#     data = df_rounce.sel(scenario=s)
#     climate_models = df_rounce['Climate_Model'].values
#     model_indices = df_rounce['model'].values
#     if s==0:
#         valid_model_indices = model_indices[[not any(model.startswith(prefix) for prefix in ['BCC', 'FGOALS', 'CESM2', 'INM', 'MPI', 'NorESM']) for model in climate_models]]
#         data = data.sel(model=valid_model_indices)
#     models=data.Climate_Model.values
#     mean_series_base_year = data.reg_mass_annual.mean(dim='model').sel(year=2000).values
#     mean_series = data.reg_mass_annual.mean(dim='model').values
#     for m, model in enumerate(models):
#         print(model,m)
#         # try:
#         try:
#         # Ensure data exists at base year (e.g., 2000)
#             base_yr = data.sel(model=m).sel(year=2000).reg_mass_annual.values
#             if not np.isnan(base_yr):  # Optional: skip NaNs
#                 plt.plot(data.sel(model=m).reg_mass_annual.values/base_yr*100, label=model, alpha=0.4, color=ssp_colors[scenario], ls=':')
#                 # plt.plot(data.year.values, data.sel(model=m).reg_mass_annual.values, label=model, alpha=0.4, color=ssp_colors[scenario], ls=':')
#         except Exception as e:
#             print(e)
#             continue
#     rounce_offset=(initial_volume_pct_rounce.values-100)
#     rounce_ts = (mean_series-mean_series_base_year)/mean_series_base_year*100 + rounce_offset
    
#     # axes.plot(data.year.values, rounce_ts, label=f'Rounce {ssp}', color=ssp_colors[scenario], ls=':', linewidth=3)
#     print("init")
#     glacier_area_share= 0.78
#     plt.plot(data.year.values, mean_series*glacier_area_share, label='Multi-model mean', color=ssp_colors[scenario], ls='-', linewidth=2.5)

# # plt.title(f'Regional Glacier Mass Change Rounce et al. (2023))')
# # # plt.ylabel('Remaining Glacier Volume [km${^3}$]')
# # plt.ylabel('Remaining Glacier Mass [%]')#' m w.e. yr$^{-1}$')
# # plt.xlabel('Year')
# # plt.legend()
# # plt.grid(True)


#Add harry and Aguayo data

Aguayo_path = f"/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/04. Reference/01. Climate data/Aguayo/merged_glacier_simulations.nc"#"R13_glac_mass_annual_50sets_2000_2100-rcp26.nc"
ds_aguayo_all = xr.open_dataset(Aguayo_path)
# fig, ax = plt.subplots(figsize=(10, 6))
markers = ['o', 'x', '^']
ls =[':','-.', '--']
for v, var in enumerate(['glogem', 'oggm', 'pygem']):
    da=ds_aguayo_all[var]
    print(len(da.model.values))

    # Loop over each scenario
    for scenario in ds_aguayo_all['scenario'].values:
        if scenario in ['ssp126','ssp370']:
            # Loop over each model
            da_list = []
            print(da)
            
            for model in da['model'].values:
                
                da_=da.sel(model=model, scenario=scenario)
                all_zero = (da_ < 1e-10)#.all(dim='model') #add tollerance for zero mask
                da_single = da_.where(~all_zero)
                da_single = da_single.sel(year=slice(2015, None))
                print(model, da_single)
                da_list.append(da_single)
            
                # Optional: plot here if you want
                # if model=="CESM2":
                # ax.plot(ds_aguayo_all['year'], da_single, ls=ls[v], lw=3,
                #         color=colors_ssp[scenario],
                #         # label=f'{scenario} - {model}') # Select the time series
                #         )
    
            da_stack = xr.concat(da_list, dim='model')
    
            # Now compute stats
            da_mean = da_stack.mean(dim='model', skipna=True)
            da_mean_initial = da_mean.sel(year=2015)
        
            da_mean_relative = (da_mean)/da_mean_initial*100
            
            da_min = da_stack.min(dim='model', skipna=True)
            da_max = da_stack.max(dim='model', skipna=True)
            da_std = da_stack.std(dim='model')
            da_std_rel = da_std/da_mean_initial*100
        
            # ax.plot(ds_aguayo_all['year'], da_mean, ls = ls[v],label=f'{scenario} - {var}', color=colors_ssp[scenario],lw=3)
            #uncomment for relative plotting
            # ⚡️ Plot only years >= 2015
            years = da_mean['year'].values
            ax.plot(years, da_mean_relative, ls = ls[v],label=f'{scenario} - {var}', color=colors_ssp[scenario],lw=3)
            
            # Plot the shaded min-max range
            # ax.fill_between(ds_aguayo_all['year'], da_min, da_max, alpha=0.1, color=colors_ssp[scenario])
            #Plot shaded 1std range
            ax.fill_between(years, da_mean_relative-da_std_rel, da_mean_relative+da_std_rel, alpha=0.1, color=colors_ssp[scenario])
            

ax.set_title(f'Volume evolution over time for different models')
ax.set_xlabel('Year')
# ax.set_ylabel('Volume [km$^3$]')
ax.set_ylabel('Volume relative to 2015 [%]')
    # ax.legend(fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.tight_layout()
    # plt.show()
    

ssp_patches = [
    mpatches.Patch(color=color, label=scenario.upper())
    for scenario, color in colors_ssp.items()
]
    
line_legend = [
    mpatches.Patch(color='black', label='Historical (W5E5)'),
    Line2D([0], [0], color='grey', linestyle='solid', label='simulations'),
    Line2D([0], [0], color='grey', linestyle='dotted', label='GloGEM'),
    Line2D([0], [0], color='grey', linestyle='dashed', label='PyGEM'),
    Line2D([0], [0], color='grey', linestyle='dashdot', label='OGGM')

]

legend_patches=ssp_patches + line_legend
# Then pass this list to legend
ax.legend(handles=legend_patches, ncols=5, bbox_to_anchor=(0.5,-0.3), loc='lower center')

plt.tight_layout()
plt.show()

# df_rounce.reg_mass_annual.plot(colors='scenario')


#%% Test climate input data for run with hydro
out_ids = ["_baseline_W5E5.000", "_perturbed_CESM2.001","_perturbed_CESM2.002","_perturbed_CESM2.003"]
for gdir in gdirs_3r_a1[1:2]:
    for o, out_id in enumerate(out_ids):
        print(out_id)
        fpath = gdir.get_filepath('model_geometry', filesuffix=out_id)
        with xr.open_dataset(fpath) as ds:
            plot_centerlines(gdir)

# climate_path = os.path.join(sum_dir, "climate_run_output_baseline_W5E5.000.nc")
                            
# with xr.open_dataset(climate_path) as ds:
#     # print(ds.head())
#     print(ds.where(ds.time==2014, drop=True).volume.values)
#     # 
        
#%% Export list of rgi_ids to rodrigo

gdir_list=[]
for gdir in gdirs_3r_a1:
    gdir_list.append(gdir.rgi_id)
# Convert to DataFrame
df = pd.DataFrame({'rgi_id': gdir_list})

# Save to CSV
df.to_csv(os.path.join(wd_path, "masters", "rgi_id_overview.csv"), index=False)
