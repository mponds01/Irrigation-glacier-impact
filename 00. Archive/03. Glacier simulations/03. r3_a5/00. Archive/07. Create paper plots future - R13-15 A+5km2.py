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

#%%
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

from OGGM_data_processing import process_perturbation_data,custom_process_cmip_data,custom_process_gcm_data

# %% Cell 1: Initialize OGGM with the preferred model parameter set up
colors = {
    "irr": ["black", "#40E0D0"],  # Base: deep teal and turquoise
    "noi": ["dimgrey","darkgrey"],  # Golden yellow tones

    # Future scenarios
    #1f78b4
    "noi_fut": ["#6B8FC4", "#F1798D"],  # SSP126: orange, SSP370: darker orange
    "irr_fut": ["#1B3968", "#D6102A"],  # SSP126: light turquoise, SSP370: lighter turquoise
    # "noi_fut": ["darkorange", "tomato"],  # SSP126: orange, SSP370: darker orange
    # "irr_fut": ["cornflowerblue", "navy"],  # SSP126: light turquoise, SSP370: lighter turquoise
}

folder_path = '/Users/magaliponds/Documents/00. Programming'
wd_path = f'{folder_path}/03. Modelled perturbation-glacier interactions - R13-15 A+5km2/'
wd_path_fut = f'{folder_path}/03. Modelled perturbation-glacier interactions Future - R13-15 A+5km2/'
os.makedirs(wd_path, exist_ok=True)
os.makedirs(wd_path_fut, exist_ok=True)
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

#%% Cell 2b: Load gdirs_3r_a5 from pkl

wd_path_pkls_fut = f'{wd_path_fut}/pkls_subset_success/'

gdirs_3r_a5 = []
for filename in os.listdir(wd_path_pkls_fut):
    if filename.endswith('.pkl'):
        # f'{gdir.rgi_id}.pkl')
        file_path = os.path.join(wd_path_pkls_fut, filename)
        with open(file_path, 'rb') as f:
            gdir = pickle.load(f)
            gdirs_3r_a5.append(gdir)

print(len(gdirs_3r_a5))

#%% Load dicts
   
colors = {
    "irr": ["#000000", "#555555"],   # Base: deep teal and turquoise
    "noi": ["dimgrey","darkgrey"],  # Golden shades

    # Future scenarios
    # "noi_fut": ["#FFA500", "#E34234"],  # SSP126: orange, SSP370: red
    # "irr_fut": ["#A0E7E5", "#4682B4"],  # SSP126: light turquoise, SSP370: blue
    "irr_fut": ["#D6102A", "#1B3968"],  # SSP126: deep red, SSP370: deep blue
    "noi_fut": ["#F8AEB8", "#A4B9E1"],  # SSP126: pale rose, SSP370: light periwinkle
}

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

               
                    
#%% Plot for 9 rgi ids
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

for g, gdir in enumerate(gdirs_3r_a5[1:10]):
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
                            out_id = f'_CESM2{input_filesuffix}'

                            
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
        
d#%% create master for creating subplots - outdated
exp = ["IRR", "NOI"]

regions = [13, 14, 15]
subregions = [9, 3, 3]
ssps = ["126","370"]#"126",

region_data = []
members_averages = [4]
models_shortlist = ["CESM2"]

        
# Storage for time series
subregion_series = {}  # subregion → DataArray[time, member]
global_series = []     # total average over all subregions per member
members_all = []       # (model, member_id) pairs to track 14-member average


master = pd.read_csv(f"{wd_path}/masters/master_lon_lat_rgi_id.csv")
path_past = os.path.join(sum_dir, f'climate_run_output_baseline_W5E5.000.nc')
ds_w5e5 = xr.open_dataset(path_past)
master = master[master["rgi_id"].isin(ds_w5e5.rgi_id.values)]

initial_volume=ds_w5e5.sel(time="1985")
initial_volume = initial_volume.volume.sum(dim='rgi_id').values


# Create containers
past_records = []
future_records = []


for reg, region in enumerate(regions):
    for sub in range(subregions[reg]):
        region_id = f"{region}-0{sub+1}"
        filtered_master = master[master["rgi_subregion"] == region_id].copy()
        subregion_id = region_id  # used for storage key
        for e, ex in enumerate(exp):
            for m, model in enumerate(models_shortlist):
                for member in range(members_averages[m]):
                        if member<1:
                            continue
                        if member >= 1: #skip 0
                            sample_id = f"{model}.00{member:01d}"
                            
                            
                            if ex =="IRR":
                                ds_past = ds_w5e5
                                
                            else:
                                path_past = os.path.join(sum_dir, f'climate_run_output_perturbed_CESM2.00{member}.nc')
                                ds_past = xr.open_dataset(path_past, engine='h5netcdf')
                            ds_past_filtered = ds_past.sel(rgi_id=filtered_master.rgi_id.values)
                            volume_past = ds_past_filtered.volume.sum(dim='rgi_id')
                            volume_past = volume_past /initial_volume *100 -100
                            past_records.append({
                                "exp": ex,
                                "sample_id": sample_id,
                                "subregion": subregion_id,
                                "time": ds_past.time.values,
                                "volume": volume_past.values
                            })
                            
                            for s, ssp in enumerate(ssps):
                                input_filesuffix = f"_SSP{ssp}_{ex}.00{member}"
                                out_id = f'_CESM2{input_filesuffix}'
                                print(out_id)
                                input_filesuffix = f"_SSP{ssp}_{ex}.00{member}"
                                path_fut = os.path.join(sum_dir, f'climate_run_output{input_filesuffix}.nc')
                                ds_fut = xr.open_dataset(path_fut, engine='h5netcdf')
                            
                                ds_fut_filtered = ds_fut.sel(rgi_id=filtered_master.rgi_id.values)
                                
                                volume_fut = ds_fut_filtered.volume.sum(dim='rgi_id')
                                volume_fut = volume_fut /initial_volume *100 -100
    
                                # Save to future record
                                future_records.append({
                                    "exp": ex,
                                    "sample_id": sample_id,
                                    "ssp": ssp,
                                    "subregion": subregion_id,
                                    "time": ds_fut.time.values,
                                    "volume": volume_fut.values
                                })                                
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
future_ds.to_netcdf(f"{wd_path}masters/master_volume_subregion_future.nc")


#%% Create master for making subplots

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
                    print(ex, ssp, member)
                    volume_past = ds_past_filtered.volume.sum(dim='rgi_id')
                    # print(ds_past.time.values[0])
                    # print(ds_past.time.values[-1])   
                    # volume_past = volume_past / initial_volume * 100 - 100

                    volumes_past_all.append(volume_past.values)
                    print(len(volume_past.values))
                    print(len(volumes_past_all))
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
                        # print(ex)
                        input_filesuffix = f"_SSP{ssp}_{ex}.00{member}"
                        # print(input_filesuffix)
                        # path_fut = os.path.join(sum_dir, f'climate_run_output{input_filesuffix}.nc')
                        if ex == "NOI":
                            path_fut = os.path.join(sum_dir, f'climate_run_output{input_filesuffix}_noi_bias.nc')
                        else:
                            path_fut = os.path.join(sum_dir, f'climate_run_output{input_filesuffix}.nc')

                        
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

past_ds.to_netcdf(f"{wd_path}masters/master_volume_subregion_past_2014.nc")
future_ds.to_netcdf(f"{wd_path}masters/master_volume_subregion_future_noi_bias_2014.nc")
# future_ds.to_netcdf(f"{wd_path}masters/master_volume_subregion_future.nc")

#%%

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



#%% Create master for future data table



ds = xr.open_dataset(f"{wd_path}masters/master_volume_subregion_future_noi_bias.nc")
ds = ds.sel(time=2074).sel(sample_id='3-member-avg')

mean_list = []
for exp in ["IRR","NOI"]:
    for ssp in ["126", "370"]:
        mean_ds = ds.sel(ssp=ssp).sel(exp=exp)
        print(mean_ds)
    
    


# df_means = pd.DataFrame(mean_list, columns=["subregion", "V_2264_irr_delta","V_2264_noirr_delta"])

# df_means.to_csv(f"{wd_path}masters/mean_deltaV_Comitted.csv")

#%% Create plots by subregion        
 
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
        
        
 #%% Create plots by subregion - panel by SSP       
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

for time in ["past","future"]:
    for e, ex in enumerate(exp):
        color = colors[ex.lower()][0]
        if ex =="IRR" and time=="past":
            # axes_dict[region_id].plot(past_ds.time, past_ds.volume.sel(subregion=region_id).sel(exp=ex).sel(sample_id="CESM2.001")/initial_volume*100-100, linestyle='solid', color=color)
            axes[0].plot(past_ds.time, past_ds.volume.sel(exp=ex).sel(sample_id="CESM2.001").sum(dim='subregion')/initial_volume_big*100-100, linestyle='solid', color=color, label="Historic W5E5 individual member", lw=3)
            axes[1].plot(past_ds.time, past_ds.volume.sel(exp=ex).sel(sample_id="CESM2.001").sum(dim='subregion')/initial_volume_big*100-100, linestyle='solid', color=color, label="Historic W5E5 individual member", lw=3)
        
               
        else:
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
for (s,ssp) in enumerate(ssps):
    ax = axes[s]
    for (e,ex) in enumerate(exp):
        if s==0:
            linestyle='--'
            marker='>'
        else:
            linestyle='--'
            marker='o'
        color=colors[f"{ex.lower()}_fut"][0]
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
        
for (s,ssp) in enumerate(ssps):
    # ax = axes[s]
    for (e,ex) in enumerate(exp):    
        color=colors[f"{ex.lower()}_fut"][0]  
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
                    Line2D([0], [0], color='grey', linestyle='dashed', linewidth=3, label=f'Future, 11-member mean'),
                    mpatches.Patch(color=colors['noi'][0], label='Historical NoIrr'),
                    Line2D([0], [0], color='grey', linestyle='solid', linewidth=3, label=f'Historic, 11-member mean'),
                    mpatches.Patch(color=colors['irr_fut'][0], label='Future (W5E5)'),
                    Line2D([0], [0], color='grey', linestyle='dotted', linewidth=3, label=f'individual member mean'),
                    mpatches.Patch(color=colors['noi_fut'][0], label='Future NoIrr'),
                         ]
# plt.tight_layout()
fig.legend(handles=legend_patches, loc='lower center', ncols=4, bbox_to_anchor=(0.5,-0.05), )#,
          # bbox_to_anchor=(0.512, 0.96), ncols=5, fontsize=12,columnspacing=1.5)
# fig.subplots_adjust(right=0.97)


#%% Create plots by subregion -1 figure    
exp = ["IRR", "NOI"]
regions = [13, 14, 15]
subregions = [9, 3, 3]
ssps = ["126", "370"]
members_averages = [4]
models_shortlist = ["CESM2"]

fig, axes = plt.subplots(figsize=(12,7), sharey=True )  # create a new figure
# axes = axes.flatten()
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
past_ds = xr.open_dataset(f"{wd_path}masters/master_volume_subregion_past_2014.nc")
# future_ds = xr.open_dataset(f"{wd_path}masters/master_volume_subregion_future.nc")
future_ds = xr.open_dataset(f"{wd_path}masters/master_volume_subregion_future_noi_bias_2014.nc")
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
                                        if member == 1 and region_id == "13-01":
                                            label = f'Future {ex.lower()} SSP{ssp} individual member'
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

axes.plot(past_ds.time, sum_big, linestyle='dashed', color=color, label="Historic NoIrr 3-member average", lw=2)
axes.plot(past_ds.time, sum_big, linestyle='dashed', color=color, label="Historic NoIrr 3-member average", lw=2)
axes.fill_between(past_ds.time, min_big,max_big, color=color, alpha=0.2)
# axes.fill_between(past_ds.time, sum_big-std_big,sum_big+std_big, color=color, alpha=0.2)
annotations =np.zeros((2,2))
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
        ax.plot(future_ds.time, sum_big,  color=color, label=f"Future {ex} SSP 3-member average", ls=ls, linewidth=2)
        annotation=np.round(sum_big[-1].values)
        annotations[s,e] = annotation
        
        
 
greys = ['lightgrey', 'grey']
diffs=[-9,-11]
for (s,ssp) in enumerate(ssps):
    
    
    x = 2077+s*3 # or an integer like 2045 if x is in years
    y0 = annotations[s, 0]
    y1 = annotations[s, 1]
    
    # Add vertical line
    axes.add_line(mlines.Line2D([x, x], [y0, y1], color='black', lw=1))
    
    # Add horizontal caps
    cap_width = 1  # adjust width depending on your x-axis scale
    axes.add_line(mlines.Line2D([x - cap_width, x + cap_width], [y0, y0], color='black', lw=1))
    axes.add_line(mlines.Line2D([x - cap_width, x + cap_width], [y1, y1], color='black', lw=1))
    
    mid_y = (y0 + y1) / 2
    x_offset = 1  # or use +3 for integers
    axes.text(
        x + x_offset,  # to the right
        mid_y,         # middle of whisker
        fr'$V_{{\mathrm{{Irr,SSP{ssp}}}}}={diffs[s]}\%$',
        va='center',
        ha='left',
        fontsize=12,
        fontweight='bold'
    )

    # ax = axes[s]
    for (e,ex) in enumerate(exp):  
        
        
        color=colors[f"{ex.lower()}_fut"][s]  
        if ex == "IRR" or ssp=="370":
            offset = 0.5
        else:
            offset= - 2
        for line in range(4):
            annotation = annotations.flatten()[line]
            axes.axhline(annotation, linestyle='--', color='black', lw=0.5,zorder=0)
        annotation = annotations[s,e]
        axes.text(1990,annotation+offset, f"{ex} SSP{ssp} ∆V={annotation}%",color=color, fontweight='bold')

# axes[1].set_yticks([])
desired_years = [1985, 2015, 2045, 2075]
axes.set_xlim(1985,2100)
axes.set_ylim(-61,4)

# Apply to the axis
axes.set_xticks(desired_years)
axes.set_xticklabels([str(year) for year in desired_years])
# ax_big.set_yticks([])
plt.subplots_adjust(hspace=0.5, wspace=0.05)#, wspace=0.2)
axes.set_ylabel("Volume change (%, vs. 1985 historic")

legend_patches = [mpatches.Patch(color=colors['irr'][0], label='Historical (W5E5)'),
                  mpatches.Patch(color=colors['noi'][0], label='Historical NoIrr'),
                  mpatches.Patch(color=colors['irr_fut'][0], label='Future (W5E5), SSP126'),
                  mpatches.Patch(color=colors['noi_fut'][0], label='Future NoIrr, SSP126'),
                  mpatches.Patch(color=colors['irr_fut'][1], label='Future (W5E5), SSP370'),
                    mpatches.Patch(color=colors['noi_fut'][1], label='Future NoIrr, SSP370'),
                    # Line2D([0], [0], color='grey', linestyle='dotted', linewidth=3, label=f'individual member mean'),
                    ]
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.3)     
                         
# plt.tight_layout()
fig.legend(handles=legend_patches, loc='upper right', ncols=1, bbox_to_anchor=(0.9,0.88), frameon=False, fontsize=10)#,
          # bbox_to_anchor=(0.512, 0.96), ncols=5, fontsize=12,columnspacing=1.5)
# fig.subplots_adjust(right=0.97)

fig_folder = '/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/04. Figures/99. Final Figures/'
plt.savefig(f"{fig_folder}/Future_Volume_Evolution.png")       
        
        
        
        
        
        
        
        
        
        
        
        