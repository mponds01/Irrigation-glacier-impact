#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 20:35:51 2025

@author: magaliponds

This scripts creates the plots for future data in manuscript

It consists of the following steps:
Cell 0: Load custom OGGM functions
Cell 0b: Load packages
Cell 1: Initialize OGGM
Cell 2: Load gdirs 
Cell 2a: Print mean RGI date (for manuscript input)
Cell 2b: Load dictionaries 
Cell 3 Create master dataset for making subplots 
Cell 3b: Update master including volume overall (HMA)
Cell 4: Create overview plot volume change totals (with all SSPs and ∆V annotation - key figure)
Cell 5: Align data per subregion and create output dataset
Cell 6:  Runoff plot including share by runoff component contribution to total - averaged over members beforehand (key figure)
Cell 7: runoff plot timeline subregions (key figure, extended data)
Cell 8: make hydrological runoff plot with relative difference  
Cell 9: make dataset peak water per glacier aggregated (for scatter plot like Rounce has)
Cell 10: Convert runoff to annual, per rgi_ID
Cell 11: stack historical runoff data to different ssps so there is one continuous timeseries per ssp (1985-2014 + 2015-2074)
Cell 12:  Process and calculate the peak water time per glacier per member and across 3 members
Cell 13a:  Process and calculate the peak water time per subregion per member and across 3 members
Cell 13b: Link rgi subregions to coordinates 
Cell 13c: Link rgi_id to coordinates for filtered dataset
Cell 14a: Create peak water plots on map (key figure - extended data)
Cell 14b: Sense check for comparison - create runoff timeline overview
Cell 14c: create scatter plot with peak water years (key figure - subpanel 3)
Cell 15: Make total runoff overview - 3 panels, calling functions (key figure, not used)
Cell 15b: Make total runoff overview - 2 panels, calling functions (key figure)
Cell 16: Open and process reference data
Cell 16b: Plot reference data on top of our future simulations
Cell 16c: Plot reference data on top of our future simulations - relative to 2015
"""

#%% Cell 0: Load custom OGGM functions
import os
import sys
function_directory = "/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Research/01. IRRMIP/src/03. Glacier simulations"
sys.path.append(function_directory)

#%% Cell 0b: Load packages
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
from matplotlib.colors import LightSource
import matplotlib.ticker as mticker


import matplotlib.lines as mlines
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import sys
from xarray.coding.times import CFDatetimeCoder
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from oggm.graphics import plot_centerlines
from shapely.geometry import shape
import rasterio

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

#%% Cell 2: Load gdirs_3r_a1 from pkl

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

#%% 2a Print mean RGI date

dates = [gdir.rgi_date for gdir in gdirs_3r_a1]

# average over all glaciers
mean_date = np.median(dates)
print(f"Average RGI inventory year: {mean_date:.1f}")


#%% Cell 2b: Load dicts
   
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


#%% Cell 3:  Create master for making subplots

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
                # volume_past_avg = np.mean(volumes_past_all, axis=0)
                volume_past_avg = np.median(volumes_past_all, axis=0)
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
                    # volume_fut_avg = np.mean(volumes_future_all[ssp], axis=0)
                    volume_fut_avg = np.median(volumes_future_all[ssp], axis=0)
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

past_ds.to_netcdf(f"{wd_path}masters/master_volume_subregion_past_median.nc")
future_ds.to_netcdf(f"{wd_path}masters/master_volume_subregion_future_noi_bias_median.nc")

# past_ds.to_netcdf(f"{wd_path}masters/master_volume_subregion_past.nc")
# future_ds.to_netcdf(f"{wd_path}masters/master_volume_subregion_future_noi_bias.nc")
# future_ds.to_netcdf(f"{wd_path}masters/master_volume_subregion_future.nc")

#%% Cell 3b: Update master including total volume (HMA overall)

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



#%% Cell 4: Create overview plot volume change total, all SSPs and ∆V annotation (key figure) 


def historic_future_volume_evolution_hma(ax=None):
    exp = ["IRR", "NOI"]
    regions = [13, 14, 15]
    subregions = [9, 3, 3]
    ssps = ["126", "370"]
    members_averages = [4]
    models_shortlist = ["CESM2"]
    
    if ax==None:
        fig, ax = plt.subplots(figsize=(12,7), sharey=True )  # create a new figure
    
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
    # past_ds = xr.open_dataset(f"{wd_path}/masters/master_volume_subregion_past.nc")
    # # future_ds = xr.open_dataset(f"{wd_path}/masters/master_volume_subregion_future.nc")
    # future_ds = xr.open_dataset(f"{wd_path}/masters/master_volume_subregion_future_noi_bias.nc")
    past_ds = xr.open_dataset(f"{wd_path}/masters/master_volume_subregion_past_median.nc")
    future_ds = xr.open_dataset(f"{wd_path}/masters/master_volume_subregion_future_noi_bias_median.nc")
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
                ax.plot(past_ds.time, past_ds.volume.sel(exp=ex).sel(sample_id="CESM2.001").sum(dim='subregion')/initial_volume_big*100-100, linestyle=ls, color=color, label="Historic W5E5 individual member", lw=2)
                ax.plot(past_ds.time, past_ds.volume.sel(exp=ex).sel(sample_id="CESM2.001").sum(dim='subregion')/initial_volume_big*100-100, linestyle=ls, color=color, label="Historic W5E5 individual member", lw=2)
            
                   
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
                                # ax = axes
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
                                            
                                            ax.plot(past_ds.time, past_ds.volume.sel(exp='NOI').sel(sample_id=sample_id).sum(dim='subregion')/initial_volume_big*100-100,
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
                        
                                    # ax.fill_between(
                                    #     future_ds.time,
                                    #     min_vol,
                                    #     max_vol,
                                    #     color=color_fut,
                                    #     alpha=0.1,
                                    #     label=f'SSP{ssp} envelope'
                                    # )
    
                
                                    
                            
    color=colors["noi"][0]    
    std_big = (past_ds.volume.sel(exp='NOI').sum(dim='subregion').std(dim='sample_id')/initial_volume_big)*100#-100
    print(std_big)
    sum_big = (past_ds.volume.sel(exp='NOI').sel(sample_id='3-member-avg').sum(dim='subregion')/initial_volume_big)*100-100
    min_big = (past_ds.volume.sel(exp='NOI').sum(dim='subregion').min(dim='sample_id')/initial_volume_big)*100-100
    max_big = (past_ds.volume.sel(exp='NOI').sum(dim='subregion').max(dim='sample_id')/initial_volume_big)*100-100
    
    ax.plot(past_ds.time, sum_big, linestyle='dashed', color=color, label="Historic NoIrr 3-member average", lw=3)
    ax.plot(past_ds.time, sum_big, linestyle='dashed', color=color, label="Historic NoIrr 3-member average", lw=3)
    # ax.fill_between(past_ds.time, min_big,max_big, color=color, alpha=0.2)
    ax.fill_between(past_ds.time, sum_big-std_big,sum_big+std_big, color=color, alpha=0.2)
    
    annotations =np.zeros((2,2))
    ssp_anno=['1.26', '3.70']
    for (s,ssp) in enumerate(ssps):
        
        # ax = axes
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
            std_big = (future_ds.volume.sel(exp=ex).sel(ssp=ssp).sum(dim='subregion').std(dim='sample_id')/initial_volume_big)*100 #-100
            sum_big = (future_ds.volume.sel(exp=ex).sel(ssp=ssp).sel(sample_id='3-member-avg').sum(dim='subregion')/initial_volume_big)*100-100
            ax.fill_between(future_ds.time, sum_big-std_big,sum_big+std_big, color=color, alpha=0.2)
            ax.plot(future_ds.time, sum_big,  color=color, label=f"Future {ex} SSP 3-member average", ls=ls, linewidth=3)
            
            annotation=np.round(sum_big[-1].values, decimals=2)
            annotations[s,e] = annotation
    
    annotate="off"
    if annotate=="on":
                
                
         
        greys = ['lightgrey', 'grey']
        diffs=[-7,-8]
    
        for (s,ssp) in enumerate(ssps):
            print(s)
            # if s==0:
        
            x = 2077+s*3-1 # or an integer like 2045 if x is in years
            y0 = annotations[s, 0]
            y1 = annotations[s, 1]
            
            # Add vertical line
            ax.add_line(mlines.Line2D([x, x], [y0, y1], color='black', lw=1))
            
            # Add horizontal caps
            cap_width = 1  # adjust width depending on your x-axis scale
            ax.add_line(mlines.Line2D([x - cap_width, x + cap_width], [y0, y0], color='black', lw=1))
            ax.add_line(mlines.Line2D([x - cap_width, x + cap_width], [y1, y1], color='black', lw=1))
            
            mid_y = (y0 + y1) / 2
            x_offset = 0.5  # or use +3 for integers
            ax.text(
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
                    ax.axhline(annotation, linestyle='--', color='black', lw=0.5,zorder=0)
                annotation = round(annotations[s,e],1)
                ax.text(1990,annotation+offset, f"{ex} SSP-{ssp_anno[s]} ∆V={annotation}%",color=color, fontweight='bold')
    
        desired_years = [1985, 2015, 2045, 2075]
    
        ax.set_xlim(1985,2100 )
        ax.set_ylim(-72,4)
        
        # Apply to the axis
        ax.set_xticks(desired_years)
        ax.set_xticklabels([str(year) for year in desired_years])
    # ax_big.set_yticks([])
        plt.subplots_adjust(hspace=0.5, wspace=0.05)#, wspace=0.2)
    if ax==None: #only set axis label when not making into bigger subplots figure
        ax.set_ylabel("Volume change (%, vs. 1985 historic")
    
    legend_patches = [mpatches.Patch(color=colors['irr'][0], label='Historical (W5E5)           '),
                      mpatches.Patch(color=colors['noi'][0], label='Historical NoIrr'),
                      mpatches.Patch(color=colors['irr_fut'][0], label='Future (CESM2), SSP-1.26'),
                      mpatches.Patch(color=colors['noi_fut'][0], label='Future NoIrr, SSP-1.26'),
                      mpatches.Patch(color=colors['irr_fut'][1], label='Future (CESM2), SSP-3.70'),
                        mpatches.Patch(color=colors['noi_fut'][1], label='Future NoIrr, SSP-3.70'),
                        # Line2D([0], [0], color='grey', linestyle='dotted', linewidth=3, label=f'individual member mean'),
                        ]
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.3)     
    plt.grid(True, color='grey', ls='--', lw=0.5)
                             
    # plt.tight_layout()
    if ax==None:
        plt.legend(handles=legend_patches, loc='upper right', ncols=1, bbox_to_anchor=(0.9,0.88), frameon=False, fontsize=12)#,
              # bbox_to_anchor=(0.512, 0.96), ncols=5, fontsize=12,columnspacing=1.5)
    # fig.subplots_adjust(right=0.97)
    ax.grid(True, color='grey', ls='--', lw=0.5)
    ax.text(0.96,0.94,'a', transform=ax.transAxes,fontweight='bold', fontsize=12)
    
    ax.set_xlim(1980.55, 2078.45)
    ax.set_ylim(-66.08651008605958, 6.190124893188477)
    
    fig_folder = '/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Research/01. IRRMIP/04. Figures/99. Final Figures/'
    # plt.savefig(f"{fig_folder}/01. EGU25/Future_Volume_Evolution_ssp126only.png") 
    # plt.savefig(f"{fig_folder}/01. EGU25/Future_Volume_Evolution_noannotation.png")    
    # plt.savefig(f"{fig_folder}/Future_Volume_Evolution.png")       
    
    return ax

historic_future_volume_evolution_hma()

        
        
#%% Cell 5: Align data per subregion and create output dataset
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

#%%  Cell 6: Runoff plot including share by runoff component contribution to total (key figure) - averaged over members beforehand (key figure)

def annual_runoff_timeline_plot_hma_relative(ax=None):
    peak_water_dict={}
    peak_water_list=[]
    plotting_subregions='off' #specify if one plot or also subplots
    #when working with melton only
    opath_df_monthly_melt = os.path.join(wd_path,'masters','hydro_output_monthly_subregions_melton_only.csv')
    opath_df_annual_melt = os.path.join(wd_path,'masters','hydro_output_annual_subregions_melton_only.csv')
    
    #when working with melton and prcp on only
    # opath_df_monthly = os.path.join(wd_path,'masters','hydro_output_monthly_subregions_meltprcpon_only.csv')
    # opath_df_annual = os.path.join(wd_path,'masters','hydro_output_annual_subregions_meltprcpon_only.csv')
    # 
    #when working with total runoff|
    opath_df_monthly = os.path.join(wd_path,'masters','hydro_output_monthly_subregions.csv')
    opath_df_annual = os.path.join(wd_path,'masters','hydro_output_annual_subregions.csv')
    
    opath_df_runoff_shares = os.path.join(wd_path,'masters','hydro_output_monthly_runoff_shares.csv')
    
    #load annual data for all runoff
    df_annual = pd.read_csv(opath_df_annual).reset_index()
    df_monthly = pd.read_csv(opath_df_monthly, dtype={'ssp': str, 'rgi_subregion': str, 'experiment': str})[['year','month','experiment','ssp','member', 'runoff','rgi_subregion']]
    
    #load annual data for only melt
    df_annual_melt = pd.read_csv(opath_df_annual_melt).reset_index()
    df_monthly_melt = pd.read_csv(opath_df_monthly_melt, dtype={'ssp': str, 'rgi_subregion': str, 'experiment': str})[['year','month','experiment','ssp','member', 'runoff','rgi_subregion']]
     
    #process data for total runoff                  
    # df_avg_monthly = (df_monthly.groupby(['year','month', 'experiment', 'ssp', 'rgi_subregion',])['runoff'].mean().reset_index()) #average over members
    df_avg_monthly = (df_monthly.groupby(['year','month', 'experiment', 'ssp', 'rgi_subregion',])['runoff'].median().reset_index()) #average over members
    df_avg_monthly = df_avg_monthly[df_avg_monthly['runoff'] != 0.00] #exclude zeros as last year of simulation does not work
    # df_avg_jja = df_avg_monthly[df_avg_monthly['month'].isin([6, 7, 8])]
    df_avg_jja = df_avg_monthly[df_avg_monthly['month'].isin(np.arange(1,13,1))] #if we want full year
    df_avg_subregions = (df_avg_jja.groupby(['year', 'experiment', 'ssp', 'rgi_subregion'])['runoff'].sum().reset_index()) #sum yo annual runoff for all subregions per year
    df_avg_subregions = df_avg_subregions[~df_avg_subregions['year'].isin([2074])] 
    df_avg_all = (df_avg_subregions.groupby(['year', 'experiment', 'ssp'])['runoff'].sum().reset_index()) #sum for all subregions in HMA
    
    df_monthly_std_in = df_monthly[df_monthly['runoff'] != 0.00] #exclude zeros as last year of simulation does not work
    df_avg_monthly_for_std = (df_monthly_std_in.groupby(['year', 'experiment', 'ssp', 'member'])['runoff'].sum().reset_index()) #calculate annual sum of runoff for the different members for hma overall
    df_avg_monthly_for_std = df_avg_monthly_for_std[~df_avg_monthly_for_std['year'].isin([2074])] 
    df_avg_annual_std_all = (df_avg_monthly_for_std.groupby(['year', 'experiment', 'ssp'])['runoff'].std().reset_index()) #take std before rolling mean (otherwise std is smoothened too much, losing variability)
    df_avg_annual_std_all['runoff'] = df_avg_annual_std_all['runoff'].fillna(0) #hist irr is nan because only 1 member - make zero for plotting
    
    
    #process data for melt only runoff
    # df_avg_monthly_melt = (df_monthly_melt.groupby(['year','month', 'experiment', 'ssp', 'rgi_subregion',])['runoff'].mean().reset_index()) #average over members
    df_avg_monthly_melt = (df_monthly_melt.groupby(['year','month', 'experiment', 'ssp', 'rgi_subregion',])['runoff'].median().reset_index()) #median over members
    df_avg_monthly_melt = df_avg_monthly_melt[df_avg_monthly_melt['runoff'] != 0.00]
    # df_avg_jja_melt = df_avg_monthly_melt[df_avg_monthly_melt['month'].isin([6, 7, 8])]
    df_avg_jja_melt = df_avg_monthly_melt[df_avg_monthly_melt['month'].isin(np.arange(1,13,1))] #if we want full year
    df_avg_subregions_melt = (df_avg_jja_melt.groupby(['year', 'experiment', 'ssp', 'rgi_subregion'])['runoff'].sum().reset_index()) #sum yo annual runoff for all subregions per year
    df_avg_subregions_melt = df_avg_subregions_melt[~df_avg_subregions_melt['year'].isin([2074])] 
    df_avg_all_melt = (df_avg_subregions_melt.groupby(['year', 'experiment', 'ssp'])['runoff'].sum().reset_index()) #sum for all subregions in HMA
    
    df_monthly_std_in_melt = df_monthly_melt[df_monthly_melt['runoff'] != 0.00] #exclude zeros as last year of simulation does not work
    df_avg_monthly_for_std_melt = (df_monthly_std_in_melt.groupby(['year', 'experiment', 'ssp', 'member'])['runoff'].sum().reset_index()) #calculate annual sum of runoff for the different members for hma overall
    df_avg_monthly_for_std_melt = df_avg_monthly_for_std_melt[~df_avg_monthly_for_std_melt['year'].isin([2074])] 
    df_avg_annual_std_all_melt = (df_avg_monthly_for_std_melt.groupby(['year', 'experiment', 'ssp'])['runoff'].std().reset_index()) #take std before rolling mean (otherwise std is smoothened too much, losing variability)
    df_avg_annual_std_all_melt['runoff'] = df_avg_annual_std_all_melt['runoff'].fillna(0) #hist irr is nan because only 1 member - make zero for plotting
    
    if plotting_subregions =='on':
        if ax==None:
            fig,axes = plt.subplots(4,4,figsize=(12,8), sharex=True, sharey=False) #if including subregions
            ax = axes.flatten()
        
        
        c=1
        
        lw_tot=3 #specify linethickness for items
        lw_melt=1
        
        """ Plotting for subregions """
        
        for region in df_avg_subregions['rgi_subregion'].unique():
            print(region)
            region_data = df_avg_subregions[df_avg_subregions['rgi_subregion'] == region] #check if glacier is in region
            region_data_melt = df_avg_subregions_melt[df_avg_subregions_melt['rgi_subregion'] == region] #check if glacier is in region
            
            #calculate relative runoff
            base_year = region_data[region_data['year']==1985][region_data['experiment']=='irr'].runoff.values
            region_data['runoff_relative']=((region_data['runoff']-base_year)/base_year)
            
            base_year_melt = region_data[region_data['year']==1985][region_data['experiment']=='irr'].runoff.values
            region_data_melt['runoff_relative']=((region_data_melt['runoff']-base_year)/base_year) #respective to 1985 total runoff
            # region_data_melt['runoff_relative']=((region_data_melt['runoff']-base_year_melt)/base_year_melt) #respective to 1985 melt only runoff
            
            # calculate cumulative absolute total runoff
            # region_data['runoff_cumulative'] = region_data.sort_values([ 'year']).groupby([ 'experiment', 'ssp'])['runoff'].cumsum() #sorting as years need to be in order for accumulation
            
            for (exp, ssp) in region_data.groupby(['experiment', 'ssp']).groups.keys():
                # Skip 'hist' by itself — only process SSPs and combine with hist timeseries
                if ssp == 'hist':
                    continue 
                # if ssp=='126':
                    # continue
        
                # Define index for colors or line styles
                add = "_fut"
                s = 0 if ssp == "126" else 1
        
                # Combine hist + this SSP for the same experiment
                hist_ssp = region_data[(region_data['experiment'] == exp) &(region_data['ssp'].isin(['hist', ssp]))].copy()
                hist_ssp_melt = region_data_melt[(region_data_melt['experiment'] == exp) &(region_data_melt['ssp'].isin(['hist', ssp]))].copy()
        
                # Sort by time
                hist_ssp = hist_ssp.sort_values('year')
                hist_ssp_melt = hist_ssp_melt.sort_values('year')
                
                # Smooth
                hist_ssp['runoff_smoothed'] = (hist_ssp['runoff'].rolling(window=11, center=True, min_periods=1).mean())
                hist_ssp_melt['runoff_smoothed'] = (hist_ssp_melt['runoff'].rolling(window=11, center=True, min_periods=1).mean())        
                
                # hist_ssp['relative_runoff_smoothed'] = (hist_ssp['runoff_relative'].rolling(window=11, center=True, min_periods=1).mean())
                # hist_ssp_melt['relative_runoff_smoothed'] = (hist_ssp_melt['runoff_relative'].rolling(window=11, center=True, min_periods=1).mean())
        
                # Style settings
                ls = '--' if exp == "noi" else '-'
                # lw = 1 if exp == "noi" else 2
        
                # Split into historical & future
                is_hist = hist_ssp['year'] < 2014
                is_future = hist_ssp['year'] >= 2014
        
                # Plot hist part (no add)
                ax[c].plot(
                    hist_ssp.loc[is_hist, 'year'],
                    hist_ssp.loc[is_hist, 'relative_runoff_smoothed']*100,
                    # hist_ssp.loc[is_hist, 'runoff_cumulative'],
                    # hist_ssp.loc[is_hist, 'runoff_smoothed'],
                    label=f"{exp.upper()} hist",
                    color=colors[f'{exp}'][0],
                    linestyle=ls, linewidth=lw_tot, 
                )
                
                ax[c].plot(
                    hist_ssp_melt.loc[is_hist, 'year'],
                    hist_ssp_melt.loc[is_hist, 'relative_runoff_smoothed']*100,
                    # hist_ssp_melt.loc[is_hist, 'runoff_cumulative'],
                    # hist_ssp_melt.loc[is_hist, 'runoff_smoothed'],
                    label=f"{exp.upper()} hist",
                    color=colors[f'{exp}'][0],
                    linestyle=ls, linewidth=lw_melt
                )
        
                # Plot future part (add)
                ax[c].plot(
                    hist_ssp.loc[is_future, 'year'],
                    hist_ssp.loc[is_future, 'relative_runoff_smoothed']*100,
                    # hist_ssp.loc[is_future, 'runoff_cumulative'],
                    # hist_ssp.loc[is_future, 'runoff_smoothed'],
                    label=f"{exp.upper()} {ssp}",
                    color=colors[f'{exp}{add}'][s],
                    linestyle=ls, linewidth=lw_tot
                )
                
                ax[c].plot(
                    hist_ssp_melt.loc[is_future, 'year'],
                    hist_ssp_melt.loc[is_future, 'relative_runoff_smoothed']*100,
                    # hist_ssp_melt.loc[is_future, 'runoff_cumulative'],
                    # hist_ssp_melt.loc[is_future, 'runoff_smoothed'],
                    label=f"{exp.upper()} {ssp}",
                    color=colors[f'{exp}{add}'][s],
                    linestyle=ls, linewidth=lw_melt
                )
                
                peak_year=hist_ssp.loc[hist_ssp['runoff_smoothed'].idxmax(), 'year']
                peak_magnitude=hist_ssp.loc[hist_ssp['runoff_smoothed'].idxmax(), 'runoff_smoothed']
                
                
                
                peak_water_dict = {
                    'rgi_subregion':region,
                    'peak_water_year': peak_year,
                    'peak_magnitude': peak_magnitude,
                    'experiment': exp,
                    'ssp': ssp}
                
                peak_water_list.append(peak_water_dict)
                # print(peak_water_dict)
                
                if peak_year<=2014:
                    color='black'
                else:
                    color=colors[f'{exp}{add}'][s]
                ax[c].axvline(peak_year, color=color, linestyle=ls, lw=lw_tot-2)
                ax[c].set_title(f'Region: {region}')
        
            ax[c].set_title(f'Region: {region}')
            ax[c].grid(True, color='grey', ls='--', lw=0.5)
            
            plt.tight_layout()
            c+=1
    else:
        if ax==None:
            fig,ax = plt.subplots(1,1, figsize=(12,8), sharex=True, sharey=False)
        lw_tot=3
        lw_melt=2
        
    if plotting_subregions=='on':
        # if ax==None:
        ax = ax[0] #otherwise subscript problems
        
        
    base_year_all =df_avg_all[df_avg_all['year']==1985][df_avg_all['experiment']=='irr'].runoff.values
    #calculate relative runoff
    df_avg_all['runoff_relative']=(df_avg_all['runoff']-base_year_all)/base_year_all
    df_avg_annual_std_all['runoff_relative']=(df_avg_annual_std_all['runoff'])/base_year_all
    
    base_year_all_melt =df_avg_all_melt[df_avg_all_melt['year']==1985][df_avg_all_melt['experiment']=='irr'].runoff.values
    df_avg_all_melt['runoff_relative']=(df_avg_all_melt['runoff']-base_year_all)/base_year_all #relative to total runoff
    # df_avg_all_melt['runoff_relative']=(df_avg_all_melt['runoff']-base_year_all)/base_year_all_melt #relative to total runoff
    
    totals=[]
    
    for (exp, ssp) in df_avg_all.groupby(['experiment', 'ssp']).groups.keys():
        
        # Skip 'hist' by itself — only process SSPs
        if ssp == 'hist':
            continue
        # if ssp=='370':
        #     continue
    
        # Define index for colors or line styles
        add = "_fut"
        s = 0 if ssp == "126" else 1
    
        # Combine hist + this SSP for the same experiment
        hist_ssp = df_avg_all[(df_avg_all['experiment'] == exp) & (df_avg_all['ssp'].isin(['hist', ssp]))].copy()
        hist_ssp_melt = df_avg_all_melt[(df_avg_all_melt['experiment'] == exp) & (df_avg_all_melt['ssp'].isin(['hist', ssp]))].copy()
        hist_ssp_std = df_avg_annual_std_all[(df_avg_annual_std_all['experiment'] == exp) & (df_avg_annual_std_all['ssp'].isin(['hist', ssp]))].copy()
        hist_ssp_std_melt = df_avg_annual_std_all_melt[(df_avg_annual_std_all_melt['experiment'] == exp) & (df_avg_annual_std_all_melt['ssp'].isin(['hist', ssp]))].copy()
    
        # Sort by time
        hist_ssp = hist_ssp.sort_values('year')
        hist_ssp_melt = hist_ssp_melt.sort_values('year')    
        hist_ssp_std = hist_ssp_std.sort_values('year')
        hist_ssp_std_melt = hist_ssp_std_melt.sort_values('year')    
        
        #calculate cumulative absolute total runoff
        hist_ssp['runoff_cumulative'] = hist_ssp.sort_values(['year']).groupby(['experiment'])['runoff'].cumsum() #sorting as years need to be in order for accumulation
        hist_ssp_melt['runoff_cumulative'] = hist_ssp_melt.sort_values(['year']).groupby(['experiment'])['runoff'].cumsum() #sorting as years need to be in order for accumulation
    
        # Smooth
        hist_ssp['runoff_smoothed'] = (hist_ssp['runoff'].rolling(window=11, center=True, min_periods=1).mean())
        hist_ssp_melt['runoff_smoothed'] = (hist_ssp_melt['runoff'].rolling(window=11, center=True, min_periods=1).mean())
        hist_ssp_std['runoff_smoothed'] = (hist_ssp_std['runoff'].rolling(window=11, center=True, min_periods=1).mean())
        hist_ssp_std_melt['runoff_smoothed'] = (hist_ssp_std_melt['runoff'].rolling(window=11, center=True, min_periods=1).mean())
        
        hist_ssp['relative_runoff_smoothed'] = (hist_ssp['runoff_relative'].rolling(window=11, center=True, min_periods=1).mean())
        hist_ssp_std['relative_runoff_smoothed'] = (hist_ssp_std['runoff_relative'].rolling(window=11, center=True, min_periods=1).mean())
        hist_ssp_melt['relative_runoff_smoothed'] = (hist_ssp_melt['runoff_relative'].rolling(window=11, center=True, min_periods=1).mean())
        
        # hist_ssp['cumulative_runoff_smoothed'] = (hist_ssp['runoff_cumulative'].rolling(window=11, center=True, min_periods=1).mean())
        # hist_ssp_melt['cumulative_runoff_smoothed'] = (hist_ssp_melt['runoff_cumulative'].rolling(window=11, center=True, min_periods=1).mean())
    
        # Style settings
        ls = '--' if exp == "noi" else '-'
        # lw = 1 if exp == "noi" else 2
    
        # Split into historical & future
        is_hist = hist_ssp['year'] < 2014
        is_future = hist_ssp['year'] >= 2014
    
    
        #plot shading first
        #plot std hist
        # ax.fill_between(
        #     hist_ssp.loc[is_hist, 'year'],
        #     hist_ssp.loc[is_hist, 'runoff_smoothed']-hist_ssp_std.loc[is_hist, 'runoff_smoothed'],
        #     hist_ssp.loc[is_hist, 'runoff_smoothed']+hist_ssp_std.loc[is_hist, 'runoff_smoothed'], 
        #     label=f"{exp.upper()} hist",color=colors[f'{exp}'][0],
        #     linestyle=ls, linewidth=lw_tot, alpha=0.1     
        # )
        
        print(hist_ssp.loc[is_hist, 'relative_runoff_smoothed'])
        #Plot std for relative
        ax.fill_between(
            hist_ssp.loc[is_hist, 'year'],
            hist_ssp.loc[is_hist, 'relative_runoff_smoothed']*100-hist_ssp_std.loc[is_hist, 'relative_runoff_smoothed']*100,
            hist_ssp.loc[is_hist, 'relative_runoff_smoothed']*100+hist_ssp_std.loc[is_hist, 'relative_runoff_smoothed']*100, 
            label=f"{exp.upper()} hist",color=colors[f'{exp}{add}'][s],
            linestyle=ls, linewidth=lw_tot, alpha=0.1     
        )
        
        #plot std hist melt
        # ax.fill_between(
        #     hist_ssp_melt.loc[is_hist, 'year'],
        #     hist_ssp_melt.loc[is_hist, 'runoff_smoothed']-hist_ssp_std_melt.loc[is_hist, 'runoff_smoothed'],
        #     hist_ssp_melt.loc[is_hist, 'runoff_smoothed']+hist_ssp_std_melt.loc[is_hist, 'runoff_smoothed'], 
        #     label=f"{exp.upper()} hist",color=colors[f'{exp}'][0],
        #     linestyle=ls, linewidth=lw_melt, alpha=0.1     
        # )
        
        #plot std future
        # ax.fill_between(
        #     hist_ssp.loc[is_future, 'year'],
        #     hist_ssp.loc[is_future, 'runoff_smoothed']-hist_ssp_std.loc[is_future, 'runoff_smoothed'],
        #     hist_ssp.loc[is_future, 'runoff_smoothed']+hist_ssp_std.loc[is_future, 'runoff_smoothed'], 
        #     label=f"{exp.upper()} hist",color=colors[f'{exp}{add}'][s],
        #     linestyle=ls, linewidth=lw_tot, alpha=0.1     
        # )
        
        #Plot std for relative
        ax.fill_between(
            hist_ssp.loc[is_future, 'year'],
            hist_ssp.loc[is_future, 'relative_runoff_smoothed']*100-hist_ssp_std.loc[is_future, 'relative_runoff_smoothed']*100,
            hist_ssp.loc[is_future, 'relative_runoff_smoothed']*100+hist_ssp_std.loc[is_future, 'relative_runoff_smoothed']*100, 
            label=f"{exp.upper()} hist",color=colors[f'{exp}{add}'][s],
            linestyle=ls, linewidth=lw_tot, alpha=0.1     
        )
        
        # plot std future melt
        # ax.fill_between(
        #     hist_ssp_melt.loc[is_future, 'year'],
        #     hist_ssp_melt.loc[is_future, 'runoff_smoothed']-hist_ssp_std_melt.loc[is_future, 'runoff_smoothed'],
        #     hist_ssp_melt.loc[is_future, 'runoff_smoothed']+hist_ssp_std_melt.loc[is_future, 'runoff_smoothed'], 
        #     label=f"{exp.upper()} hist",color=colors[f'{exp}{add}'][s],
        #     linestyle=ls, linewidth=lw_melt, alpha=0.1     
        # )
    
        # Plot hist part (no add)
        ax.plot(
            hist_ssp.loc[is_hist, 'year'],
            hist_ssp.loc[is_hist, 'relative_runoff_smoothed']*100,
            # hist_ssp.loc[is_hist, 'runoff_cumulative'],
            # hist_ssp.loc[is_hist, 'runoff_smoothed'],
            label=f"{exp.upper()} hist",
            color=colors[f'{exp}'][0],
            linestyle=ls, linewidth=lw_tot       
        )
        
        # Plot hist part melt (no add)
        # ax.plot(
        #     hist_ssp_melt.loc[is_hist, 'year'],
        #     # hist_ssp.loc[is_hist, 'relative_runoff_smoothed']*100,
        #     # hist_ssp.loc[is_hist, 'runoff_cumulative'],
        #     hist_ssp_melt.loc[is_hist, 'runoff_smoothed'],
        #     label=f"{exp.upper()} hist",
        #     color=colors[f'{exp}'][0],
        #     linestyle=ls, linewidth=lw_melt)
    
        # Plot future part (add)
        ax.plot(
            hist_ssp.loc[is_future, 'year'],
            hist_ssp.loc[is_future, 'relative_runoff_smoothed']*100,
            # hist_ssp.loc[is_future, 'runoff_cumulative'],
            # hist_ssp.loc[is_future, 'runoff_smoothed'],
            label=f"{exp.upper()} {ssp}",
            color=colors[f'{exp}{add}'][s],
            linestyle=ls, linewidth=lw_tot
        )
        
        # Plot future part melt (add)
        # ax.plot(
        #     hist_ssp_melt.loc[is_future, 'year'],
        #     # hist_ssp_melt.loc[is_future, 'relative_runoff_smoothed']*100,
        #     # hist_ssp_melt.loc[is_future, 'runoff_cumulative'],
        #     hist_ssp_melt.loc[is_future, 'runoff_smoothed'],
        #     label=f"{exp.upper()} {ssp}",
        #     color=colors[f'{exp}{add}'][s],
        #     linestyle=ls, linewidth=lw_melt
        # )
        
        years = hist_ssp.loc[is_future, 'year'].values
        runoff = hist_ssp.loc[is_future, 'relative_runoff_smoothed'].values*100
        
        for y,r in zip(years,runoff):
            totals.append((exp, ssp, y, r))
        
        peak_year=hist_ssp.loc[hist_ssp['runoff_smoothed'].idxmax(), 'year']
        peak_magnitude=hist_ssp.loc[hist_ssp['runoff_smoothed'].idxmax(), 'relative_runoff_smoothed']
        
        print("High Mountain Asia", exp, ssp)
        print("Peak year", peak_year, "Peak Magnitude", peak_magnitude)
        print("final runoff 2074", hist_ssp.loc[hist_ssp['year']==2073]['runoff_smoothed'])
        
        peak_water_dict = {
            'rgi_subregion': 'hma',
            'peak_water_year': peak_year,
            'peak_magnitude': peak_magnitude,
            'experiment': exp,
            'ssp': ssp}
        
        peak_water_list.append(peak_water_dict)
        
        if peak_year<=2014:
            color='black'
        else:
            color=colors[f'{exp}{add}'][s]
        face_color='none' if exp=='noi' else color
        # ax.axvline(peak_year, color=color, linestyle=ls, lw=lw_tot-2)
        ax.scatter(peak_year, peak_magnitude*100,marker='o', s=100, facecolor=face_color, edgecolor=color, linewidths=5)  # thicker edge)
        # print(peak_year, peak_magnitude)
    
    records_df = pd.DataFrame(totals, columns=["experiment", "ssp", "year", "runoff"])
    print(records_df)
    df_pivot = records_df.pivot_table(index=["ssp", "year"], columns="experiment", values="runoff").reset_index()
    df_pivot["diff"] = df_pivot["irr"] - df_pivot["noi"]
    results = df_pivot.loc[df_pivot.groupby("ssp")["diff"].idxmax(), ["ssp", "year", "diff"]]
    print(results)
    
    if ax==None: #set figure labels later in overall figure
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
    
    if ax==None:
        ax.set_title('High Mountain Asia', fontweight='bold')
    ax.grid(True, color='grey', ls='--', lw=0.5)
    
    linestyle_patches = [
        mlines.Line2D([0], [0], color='grey', linestyle='-',linewidth=lw_tot,label='Total runoff'),
        mlines.Line2D([0], [0], color='grey', linestyle='--',linewidth=lw_tot,label='Total runoff NoIrr'),
        mlines.Line2D([0], [0], color='grey', linestyle='-',linewidth=lw_melt,label='Melt'),
        mlines.Line2D([0], [0], color='grey', linestyle='--',linewidth=lw_melt,label='Melt NoIrr'),
        
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
    legend_patches =  line_patches + linestyle_patches 
    
    ax.set_xlim(1980.55, 2078.45)
    ax.set_ylim(-1.850978400467218, 36.08317127736037)
    print(ax.get_xlim()[0])
    print(ax.get_xlim()[1])
    print(ax.get_ylim()[0])
    print(ax.get_ylim()[1])
    
    # Add to figure/axes
    if ax==None: #only set legend if single figure
        fig.legend(handles=legend_patches, loc='lower center', frameon=False, ncols=5, bbox_to_anchor=(0.5,-0.14))
    else:
        ax.text(0.96,0.94,'b', transform=ax.transAxes,fontweight='bold', fontsize=12)
    
    # plt.xlabel('Year')
    # plt.ylabel('Runoff [km³]')
    # plt.legend()
    # plt.show()
    opath_dict_pw = os.path.join(wd_path,'masters','hydro_peak_water_dictionary_subregions.csv')
    df = pd.DataFrame(peak_water_list)
    # df.to_csv(opath_dict_pw, index=False) #only run when subregions on

    return ax

# ax_timeline =  annual_runoff_timeline_plot_hma()
# ax_timeline.show()
annual_runoff_timeline_plot_hma_relative()
# 


#%% Cell 7: runoff plot timeline subregions (key figure, extended data)

def annual_runoff_timeline_plot_hma_subregions(ax=None):
    peak_water_dict={}
    peak_water_list=[]
    plotting_subregions='on' #specify if one plot or also subplots
    #when working with melton only
    opath_df_monthly_melt = os.path.join(wd_path,'masters','hydro_output_monthly_subregions_melton_only.csv')
    opath_df_annual_melt = os.path.join(wd_path,'masters','hydro_output_annual_subregions_melton_only.csv')
    
    #when working with melton and prcp on only
    # opath_df_monthly = os.path.join(wd_path,'masters','hydro_output_monthly_subregions_meltprcpon_only.csv')
    # opath_df_annual = os.path.join(wd_path,'masters','hydro_output_annual_subregions_meltprcpon_only.csv')
    # 
    #when working with total runoff
    opath_df_monthly = os.path.join(wd_path,'masters','hydro_output_monthly_subregions.csv')
    opath_df_annual = os.path.join(wd_path,'masters','hydro_output_annual_subregions.csv')
    
    opath_df_runoff_shares = os.path.join(wd_path,'masters','hydro_output_monthly_runoff_shares.csv')
    
    #load annual data for all runoff
    df_annual = pd.read_csv(opath_df_annual).reset_index()
    df_monthly = pd.read_csv(opath_df_monthly, dtype={'ssp': str, 'rgi_subregion': str, 'experiment': str})[['year','month','experiment','ssp','member', 'runoff','rgi_subregion']]
    
    #load annual data for only melt
    df_annual_melt = pd.read_csv(opath_df_annual_melt).reset_index()
    df_monthly_melt = pd.read_csv(opath_df_monthly_melt, dtype={'ssp': str, 'rgi_subregion': str, 'experiment': str})[['year','month','experiment','ssp','member', 'runoff','rgi_subregion']]
     
    #process data for total runoff                  
    # df_avg_monthly = (df_monthly.groupby(['year','month', 'experiment', 'ssp', 'rgi_subregion',])['runoff'].mean().reset_index()) #average over members
    df_avg_monthly = (df_monthly.groupby(['year','month', 'experiment', 'ssp', 'rgi_subregion',])['runoff'].median().reset_index()) #average over members
    df_avg_monthly = df_avg_monthly[df_avg_monthly['runoff'] != 0.00] #exclude zeros as last year of simulation does not work
    # df_avg_jja = df_avg_monthly[df_avg_monthly['month'].isin([6, 7, 8])]
    df_avg_jja = df_avg_monthly[df_avg_monthly['month'].isin(np.arange(1,13,1))] #if we want full year
    df_avg_subregions = (df_avg_jja.groupby(['year', 'experiment', 'ssp', 'rgi_subregion'])['runoff'].sum().reset_index()) #sum yo annual runoff for all subregions per year
    df_avg_subregions = df_avg_subregions[~df_avg_subregions['year'].isin([2074])] 
    df_avg_all = (df_avg_subregions.groupby(['year', 'experiment', 'ssp'])['runoff'].sum().reset_index()) #sum for all subregions in HMA
    
    df_monthly_std_in = df_monthly[df_monthly['runoff'] != 0.00] #exclude zeros as last year of simulation does not work
    df_avg_monthly_subregions_for_std = (df_monthly_std_in.groupby(['year', 'experiment', 'ssp', 'member','rgi_subregion'])['runoff'].sum().reset_index()) #calculate annual sum of runoff for the different members for hma overall
    df_avg_monthly_subregions_for_std = df_avg_monthly_subregions_for_std[~df_avg_monthly_subregions_for_std['year'].isin([2074])] 
    df_avg_annual_subregions_std_all = (df_avg_monthly_subregions_for_std.groupby(['year', 'experiment', 'ssp', 'rgi_subregion'])['runoff'].std().reset_index()) #take std before rolling mean (otherwise std is smoothened too much, losing variability)
    df_avg_annual_subregions_std_all['runoff'] = df_avg_annual_subregions_std_all['runoff'].fillna(0) #hist irr is nan because only 1 member - make zero for plotting
    
    
    df_monthly_std_in = df_monthly[df_monthly['runoff'] != 0.00] #exclude zeros as last year of simulation does not work
    df_avg_monthly_for_std = (df_monthly_std_in.groupby(['year', 'experiment', 'ssp', 'member'])['runoff'].sum().reset_index()) #calculate annual sum of runoff for the different members for hma overall
    df_avg_monthly_for_std = df_avg_monthly_for_std[~df_avg_monthly_for_std['year'].isin([2074])] 
    df_avg_annual_std_all = (df_avg_monthly_for_std.groupby(['year', 'experiment', 'ssp'])['runoff'].std().reset_index()) #take std before rolling mean (otherwise std is smoothened too much, losing variability)
    df_avg_annual_std_all['runoff'] = df_avg_annual_std_all['runoff'].fillna(0) #hist irr is nan because only 1 member - make zero for plotting    
    
    #process data for melt only runoff
    # df_avg_monthly_melt = (df_monthly_melt.groupby(['year','month', 'experiment', 'ssp', 'rgi_subregion',])['runoff'].mean().reset_index()) #average over members
    df_avg_monthly_melt = (df_monthly_melt.groupby(['year','month', 'experiment', 'ssp', 'rgi_subregion',])['runoff'].median().reset_index()) #median over members
    df_avg_monthly_melt = df_avg_monthly_melt[df_avg_monthly_melt['runoff'] != 0.00]
    # df_avg_jja_melt = df_avg_monthly_melt[df_avg_monthly_melt['month'].isin([6, 7, 8])]
    df_avg_jja_melt = df_avg_monthly_melt[df_avg_monthly_melt['month'].isin(np.arange(1,13,1))] #if we want full year
    df_avg_subregions_melt = (df_avg_jja_melt.groupby(['year', 'experiment', 'ssp', 'rgi_subregion'])['runoff'].sum().reset_index()) #sum yo annual runoff for all subregions per year
    df_avg_subregions_melt = df_avg_subregions_melt[~df_avg_subregions_melt['year'].isin([2074])] 
    df_avg_all_melt = (df_avg_subregions_melt.groupby(['year', 'experiment', 'ssp'])['runoff'].sum().reset_index()) #sum for all subregions in HMA
    
    df_monthly_std_in_melt = df_monthly_melt[df_monthly_melt['runoff'] != 0.00] #exclude zeros as last year of simulation does not work
    df_avg_monthly_for_std_melt = (df_monthly_std_in_melt.groupby(['year', 'experiment', 'ssp', 'member'])['runoff'].sum().reset_index()) #calculate annual sum of runoff for the different members for hma overall
    df_avg_monthly_for_std_melt = df_avg_monthly_for_std_melt[~df_avg_monthly_for_std_melt['year'].isin([2074])] 
    df_avg_annual_std_all_melt = (df_avg_monthly_for_std_melt.groupby(['year', 'experiment', 'ssp'])['runoff'].std().reset_index()) #take std before rolling mean (otherwise std is smoothened too much, losing variability)
    df_avg_annual_std_all_melt['runoff'] = df_avg_annual_std_all_melt['runoff'].fillna(0) #hist irr is nan because only 1 member - make zero for plotting
    
    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(4, 6, figure=fig)
    
    # Define axis positions with labels
    ax_pos = {
        'big': (slice(0, 3), slice(0, 3)),
        '13-01': (0,3), '13-02': (0, 4), '13-03': (0, 5),
        '13-04': (1,3), '13-05': (1, 4), '13-06': (1, 5),
        '13-07': (2, 3), '13-08': (2, 4), '13-09': (2, 5), 
        '14-01': (3, 0), '14-02': (3, 1), '14-03': (3, 2),'15-01': (3, 3), '15-02': (3, 4),'15-03': (3, 5)
    }
    
    axes = {key: fig.add_subplot(gs[pos]) for key, pos in ax_pos.items()}
    
    # for ax in axes.values():
    #     ax.set_xticks([]), ax.set_yticks([])
        
    if plotting_subregions =='on':
        # if ax==None:
        #     fig,axes = plt.subplots(4,4,figsize=(12,8), sharex=True, sharey=False) #if including subregions
        #     ax = axes.flatten()
        
        
        c=1
        
        lw_tot=2 #specify linethickness for items
        lw_melt=1
        
        """ Plotting for subregions """
        
        for region in df_avg_subregions['rgi_subregion'].unique():
            print(region)
            region_data = df_avg_subregions[df_avg_subregions['rgi_subregion'] == region] #check if glacier is in region
            region_data_melt = df_avg_subregions_melt[df_avg_subregions_melt['rgi_subregion'] == region] #check if glacier is in region
            region_data_std = df_avg_annual_subregions_std_all[df_avg_annual_subregions_std_all['rgi_subregion']==region]
            
            #calculate relative runoff
            base_year = region_data[region_data['year']==1985][region_data['experiment']=='irr'].runoff.values
            print(base_year)
            region_data['runoff_relative']=((region_data['runoff']-base_year)/base_year)*100+100
            print(region_data['runoff_relative'])
            region_data_std['runoff_relative']=(region_data_std['runoff'])/base_year
            
            base_year_melt = region_data[region_data['year']==1985][region_data['experiment']=='irr'].runoff.values
            region_data_melt['runoff_relative']=((region_data_melt['runoff'])/base_year)*100+100 #respective to 1985 total runoff
            # region_data_melt['runoff_relative']=((region_data_melt['runoff']-base_year_melt)/base_year_melt) #respective to 1985 melt only runoff
            
            # calculate cumulative absolute total runoff
            # region_data['runoff_cumulative'] = region_data.sort_values([ 'year']).groupby([ 'experiment', 'ssp'])['runoff'].cumsum() #sorting as years need to be in order for accumulation
            
            for (exp, ssp) in region_data.groupby(['experiment', 'ssp']).groups.keys():
                # Skip 'hist' by itself — only process SSPs and combine with hist timeseries
                if ssp == 'hist':
                    continue 
                # if ssp=='126':
                    # continue
        
                # Define index for colors or line styles
                add = "_fut"
                s = 0 if ssp == "126" else 1
        
                # Combine hist + this SSP for the same experiment
                hist_ssp = region_data[(region_data['experiment'] == exp) &(region_data['ssp'].isin(['hist', ssp]))].copy()
                hist_ssp_melt = region_data_melt[(region_data_melt['experiment'] == exp) &(region_data_melt['ssp'].isin(['hist', ssp]))].copy()
                hist_ssp_std = region_data_std[(region_data_std['experiment'] == exp) & (region_data_std['ssp'].isin(['hist', ssp]))].copy()
                # hist_ssp_std_melt = region_data_std_melt[(region_data_std_melt['experiment'] == exp) & (region_data_std_melt['ssp'].isin(['hist', ssp]))].copy()
        
                # Sort by time
                hist_ssp = hist_ssp.sort_values('year')
                hist_ssp_melt = hist_ssp_melt.sort_values('year')
                hist_ssp_std = hist_ssp_std.sort_values('year')
                # hist_ssp_std_melt = hist_ssp_std_melt.sort_values('year')    
                
                # Smooth
                hist_ssp['runoff_smoothed'] = (hist_ssp['runoff'].rolling(window=11, center=True, min_periods=1).mean())
                hist_ssp_melt['runoff_smoothed'] = (hist_ssp_melt['runoff'].rolling(window=11, center=True, min_periods=1).mean())        
                hist_ssp_std['runoff_smoothed'] = (hist_ssp_std['runoff'].rolling(window=11, center=True, min_periods=1).mean())
                # hist_ssp_std_melt['runoff_smoothed'] = (hist_ssp_std_melt['runoff'].rolling(window=11, center=True, min_periods=1).mean())
                
                hist_ssp['relative_runoff_smoothed'] = (hist_ssp['runoff_relative'].rolling(window=11, center=True, min_periods=1).mean())
                hist_ssp_std['relative_runoff_smoothed'] = (hist_ssp_std['runoff_relative'].rolling(window=11, center=True, min_periods=1).mean())
                # hist_ssp_melt['relative_runoff_smoothed'] = (hist_ssp_melt['runoff_relative'].rolling(window=11, center=True, min_periods=1).mean())
        
                # Style settings
                ls = '--' if exp == "noi" else '-'
                # lw = 1 if exp == "noi" else 2
        
                # Split into historical & future
                is_hist = hist_ssp['year'] < 2014
                is_future = hist_ssp['year'] >= 2014
                
                axes[region].fill_between(
                    hist_ssp.loc[is_hist, 'year'],
                    hist_ssp.loc[is_hist, 'runoff_smoothed']-hist_ssp_std.loc[is_hist, 'runoff_smoothed'],
                    hist_ssp.loc[is_hist, 'runoff_smoothed']+hist_ssp_std.loc[is_hist, 'runoff_smoothed'], 
                    # hist_ssp.loc[is_hist, 'relative_runoff_smoothed']-hist_ssp_std.loc[is_hist, 'relative_runoff_smoothed'],
                    # hist_ssp.loc[is_hist, 'relative_runoff_smoothed']+hist_ssp_std.loc[is_hist, 'relative_runoff_smoothed'],
                    label=f"{exp.upper()} hist",color=colors[f'{exp}'][0],
                    linestyle=ls, linewidth=lw_tot, alpha=0.1     
                )
                
                #plot std hist melt
                # ax.fill_between(
                #     hist_ssp_melt.loc[is_hist, 'year'],
                #     hist_ssp_melt.loc[is_hist, 'runoff_smoothed']-hist_ssp_std_melt.loc[is_hist, 'runoff_smoothed'],
                #     hist_ssp_melt.loc[is_hist, 'runoff_smoothed']+hist_ssp_std_melt.loc[is_hist, 'runoff_smoothed'], 
                #     label=f"{exp.upper()} hist",color=colors[f'{exp}'][0],
                #     linestyle=ls, linewidth=lw_melt, alpha=0.1     
                # )
                
                
                axes[region].fill_between(
                    hist_ssp.loc[is_future, 'year'],
                    hist_ssp.loc[is_future, 'runoff_smoothed']-hist_ssp_std.loc[is_future, 'runoff_smoothed'],
                    hist_ssp.loc[is_future, 'runoff_smoothed']+hist_ssp_std.loc[is_future, 'runoff_smoothed'], 
                    # hist_ssp.loc[is_future, 'relative_runoff_smoothed']-hist_ssp_std.loc[is_future, 'relative_runoff_smoothed'],
                    # hist_ssp.loc[is_future, 'relative_runoff_smoothed']+hist_ssp_std.loc[is_future, 'relative_runoff_smoothed'], 
                    label=f"{exp.upper()} hist",color=colors[f'{exp}{add}'][s],
                    linestyle=ls, linewidth=lw_tot, alpha=0.1 
                    )
                    
        
                # Plot hist part (no add)
                axes[region].plot(
                    hist_ssp.loc[is_hist, 'year'],
                    # hist_ssp.loc[is_hist, 'relative_runoff_smoothed'],
                    # hist_ssp.loc[is_hist, 'runoff_cumulative'],
                    hist_ssp.loc[is_hist, 'runoff_smoothed'],
                    label=f"{exp.upper()} hist",
                    color=colors[f'{exp}'][0],
                    linestyle=ls, linewidth=lw_tot, 
                )
                
                # plot std future melt
                # ax.fill_between(
                #     hist_ssp_melt.loc[is_future, 'year'],
                #     hist_ssp_melt.loc[is_future, 'runoff_smoothed']-hist_ssp_std_melt.loc[is_future, 'runoff_smoothed'],
                #     hist_ssp_melt.loc[is_future, 'runoff_smoothed']+hist_ssp_std_melt.loc[is_future, 'runoff_smoothed'], 
                #     label=f"{exp.upper()} hist",color=colors[f'{exp}{add}'][s],
                #     linestyle=ls, linewidth=lw_melt, alpha=0.1     
                # )
                
                
                # ax[c].plot(
                #     hist_ssp_melt.loc[is_hist, 'year'],
                #     # hist_ssp_melt.loc[is_hist, 'relative_runoff_smoothed']*100,
                #     # hist_ssp_melt.loc[is_hist, 'runoff_cumulative'],
                #     hist_ssp_melt.loc[is_hist, 'runoff_smoothed'],
                #     label=f"{exp.upper()} hist",
                #     color=colors[f'{exp}'][0],
                #     linestyle=ls, linewidth=lw_melt
                # )
        
                # Plot future part (add)
                axes[region].plot(
                    hist_ssp.loc[is_future, 'year'],
                    # hist_ssp.loc[is_future, 'relative_runoff_smoothed'],
                    # hist_ssp.loc[is_future, 'runoff_cumulative'],
                    hist_ssp.loc[is_future, 'runoff_smoothed'],
                    label=f"{exp.upper()} {ssp}",
                    color=colors[f'{exp}{add}'][s],
                    linestyle=ls, linewidth=lw_tot
                )
                
                # ax[c].plot(
                #     hist_ssp_melt.loc[is_future, 'year'],
                #     # hist_ssp_melt.loc[is_future, 'relative_runoff_smoothed']*100,
                #     # hist_ssp_melt.loc[is_future, 'runoff_cumulative'],
                #     hist_ssp_melt.loc[is_future, 'runoff_smoothed'],
                #     label=f"{exp.upper()} {ssp}",
                #     color=colors[f'{exp}{add}'][s],
                #     linestyle=ls, linewidth=lw_melt
                # )
                
                peak_year=hist_ssp.loc[hist_ssp['runoff_smoothed'].idxmax(), 'year']
                peak_magnitude=hist_ssp.loc[hist_ssp['runoff_smoothed'].idxmax(), 'runoff_smoothed']
                # peak_magnitude=hist_ssp.loc[hist_ssp['relative_runoff_smoothed'].idxmax(), 'relative_runoff_smoothed']
                
                peak_water_dict = {
                    'rgi_subregion':region,
                    'peak_water_year': peak_year,
                    'peak_magnitude': peak_magnitude,
                    'experiment': exp,
                    'ssp': ssp}
                
                peak_water_list.append(peak_water_dict)
                # print(peak_water_dict)
                
                if peak_year<=2014:
                    color='black'
                else:
                    color=colors[f'{exp}{add}'][s]
                face_color='none' if exp=='noi' else color
                # ax.axvline(peak_year, color=color, linestyle=ls, lw=lw_tot-2)
                axes[region].scatter(peak_year, peak_magnitude,marker='o', s=70, facecolor=face_color, edgecolor=color, linewidths=5)  # thicker edge)
                print(peak_year, peak_magnitude)
                # ax[c].axvline(peak_year, color=color, linestyle=ls, lw=lw_tot-2)
                # axes[region].set_title(f'Region: {region}')
                # axes[region].ticklabel_format(style='scientific', scilimits=(0,0), axis='y')
            axes[region].set_title(f'{subregion_names[region]}', fontweight='bold')
            axes[region].grid(True, color='grey', ls='--', lw=0.5)
            # axes[region].set_xlim([1985,2074])
            # axes[region].set_ylim([2000,60000])
            axes[region].set_xticks([])
            axes[region].tick_params(labelsize=14)

            
            plt.tight_layout()
            c+=1
    else:
        # if ax==None:
        #     fig,ax = plt.subplots(1,1, figsize=(16,8), sharex=True, sharey=False)
        lw_tot=3
        lw_melt=2
        
    if plotting_subregions=='on':
        # if ax==None:
        ax = axes["big"] #otherwise subscript problems
        
        
    base_year_all =df_avg_all[df_avg_all['year']==1985][df_avg_all['experiment']=='irr'].runoff.values
    #calculate relative runoff
    df_avg_all['runoff_relative']=(df_avg_all['runoff'])/base_year_all*100
    df_avg_annual_std_all['runoff_relative']=(df_avg_annual_std_all['runoff'])/base_year_all*100
    
    base_year_all_melt =df_avg_all_melt[df_avg_all_melt['year']==1985][df_avg_all_melt['experiment']=='irr'].runoff.values
    df_avg_all_melt['runoff_relative']=(df_avg_all_melt['runoff'])/base_year_all #relative to total runoff
    # df_avg_all_melt['runoff_relative']=(df_avg_all_melt['runoff']-base_year_all)/base_year_all_melt #relative to total runoff
    
    
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
        hist_ssp = df_avg_all[(df_avg_all['experiment'] == exp) & (df_avg_all['ssp'].isin(['hist', ssp]))].copy()
        hist_ssp_melt = df_avg_all_melt[(df_avg_all_melt['experiment'] == exp) & (df_avg_all_melt['ssp'].isin(['hist', ssp]))].copy()
        hist_ssp_std = df_avg_annual_std_all[(df_avg_annual_std_all['experiment'] == exp) & (df_avg_annual_std_all['ssp'].isin(['hist', ssp]))].copy()
        hist_ssp_std_melt = df_avg_annual_std_all_melt[(df_avg_annual_std_all_melt['experiment'] == exp) & (df_avg_annual_std_all_melt['ssp'].isin(['hist', ssp]))].copy()
    
        # Sort by time
        hist_ssp = hist_ssp.sort_values('year')
        hist_ssp_melt = hist_ssp_melt.sort_values('year')    
        hist_ssp_std = hist_ssp_std.sort_values('year')
        hist_ssp_std_melt = hist_ssp_std_melt.sort_values('year')    
        
        #calculate cumulative absolute total runoff
        hist_ssp['runoff_cumulative'] = hist_ssp.sort_values(['year']).groupby(['experiment'])['runoff'].cumsum() #sorting as years need to be in order for accumulation
        hist_ssp_melt['runoff_cumulative'] = hist_ssp_melt.sort_values(['year']).groupby(['experiment'])['runoff'].cumsum() #sorting as years need to be in order for accumulation
    
        # Smooth
        hist_ssp['runoff_smoothed'] = (hist_ssp['runoff'].rolling(window=11, center=True, min_periods=1).mean())
        hist_ssp_melt['runoff_smoothed'] = (hist_ssp_melt['runoff'].rolling(window=11, center=True, min_periods=1).mean())
        hist_ssp_std['runoff_smoothed'] = (hist_ssp_std['runoff'].rolling(window=11, center=True, min_periods=1).mean())
        hist_ssp_std_melt['runoff_smoothed'] = (hist_ssp_std_melt['runoff'].rolling(window=11, center=True, min_periods=1).mean())
        
        hist_ssp['relative_runoff_smoothed'] = (hist_ssp['runoff_relative'].rolling(window=11, center=True, min_periods=1).mean())
        hist_ssp_melt['relative_runoff_smoothed'] = (hist_ssp_melt['runoff_relative'].rolling(window=11, center=True, min_periods=1).mean())
        hist_ssp_std['relative_runoff_smoothed'] = (hist_ssp_std['runoff_relative'].rolling(window=11, center=True, min_periods=1).mean())
        
        print("smoothed", hist_ssp['relative_runoff_smoothed'])
        # hist_ssp['cumulative_runoff_smoothed'] = (hist_ssp['runoff_cumulative'].rolling(window=11, center=True, min_periods=1).mean())
        # hist_ssp_melt['cumulative_runoff_smoothed'] = (hist_ssp_melt['runoff_cumulative'].rolling(window=11, center=True, min_periods=1).mean())
    
        # Style settings
        ls = '--' if exp == "noi" else '-'
        # lw = 1 if exp == "noi" else 2
    
        # Split into historical & future
        is_hist = hist_ssp['year'] < 2014
        is_future = hist_ssp['year'] >= 2014
    
    
        #plot shading first
        #plot std hist
        ax.fill_between(
            hist_ssp.loc[is_hist, 'year'],
            hist_ssp.loc[is_hist, 'runoff_smoothed']-hist_ssp_std.loc[is_hist, 'runoff_smoothed'],
            hist_ssp.loc[is_hist, 'runoff_smoothed']+hist_ssp_std.loc[is_hist, 'runoff_smoothed'], 
            # hist_ssp.loc[is_hist, 'relative_runoff_smoothed']-hist_ssp_std.loc[is_hist, 'relative_runoff_smoothed'],
            # hist_ssp.loc[is_hist, 'relative_runoff_smoothed']+hist_ssp_std.loc[is_hist, 'relative_runoff_smoothed'], 
            label=f"{exp.upper()} hist",color=colors[f'{exp}'][0],
            linestyle=ls, linewidth=lw_tot, alpha=0.1     
        )
        
        #plot std hist melt
        # ax.fill_between(
        #     hist_ssp_melt.loc[is_hist, 'year'],
        #     hist_ssp_melt.loc[is_hist, 'runoff_smoothed']-hist_ssp_std_melt.loc[is_hist, 'runoff_smoothed'],
        #     hist_ssp_melt.loc[is_hist, 'runoff_smoothed']+hist_ssp_std_melt.loc[is_hist, 'runoff_smoothed'], 
        #     label=f"{exp.upper()} hist",color=colors[f'{exp}'][0],
        #     linestyle=ls, linewidth=lw_melt, alpha=0.1     
        # )
        
        #plot std future
        ax.fill_between(
            hist_ssp.loc[is_future, 'year'],
            hist_ssp.loc[is_future, 'runoff_smoothed']-hist_ssp_std.loc[is_future, 'runoff_smoothed'],
            hist_ssp.loc[is_future, 'runoff_smoothed']+hist_ssp_std.loc[is_future, 'runoff_smoothed'], 
            # hist_ssp.loc[is_future, 'relative_runoff_smoothed']-hist_ssp_std.loc[is_future, 'relative_runoff_smoothed'],
            # hist_ssp.loc[is_future, 'relative_runoff_smoothed']+hist_ssp_std.loc[is_future, 'relative_runoff_smoothed'], 
            label=f"{exp.upper()} hist",color=colors[f'{exp}{add}'][s],
            linestyle=ls, linewidth=lw_tot, alpha=0.1     
        )
        # plot std future melt
        # ax.fill_between(
        #     hist_ssp_melt.loc[is_future, 'year'],
        #     hist_ssp_melt.loc[is_future, 'runoff_smoothed']-hist_ssp_std_melt.loc[is_future, 'runoff_smoothed'],
        #     hist_ssp_melt.loc[is_future, 'runoff_smoothed']+hist_ssp_std_melt.loc[is_future, 'runoff_smoothed'], 
        #     label=f"{exp.upper()} hist",color=colors[f'{exp}{add}'][s],
        #     linestyle=ls, linewidth=lw_melt, alpha=0.1     
        # )
    
        # Plot hist part (no add)
        ax.plot(
            hist_ssp.loc[is_hist, 'year'],
            # hist_ssp.loc[is_hist, 'runoff_relative'],
            # hist_ssp.loc[is_hist, 'relative_runoff_smoothed'],
            # hist_ssp.loc[is_hist, 'runoff_cumulative'],
            hist_ssp.loc[is_hist, 'runoff_smoothed'],
            label=f"{exp.upper()} hist",
            color=colors[f'{exp}'][0],
            linestyle=ls, linewidth=lw_tot       
        )
        
        # Plot hist part melt (no add)
        # ax.plot(
        #     hist_ssp_melt.loc[is_hist, 'year'],
        #     # hist_ssp.loc[is_hist, 'relative_runoff_smoothed']*100,
        #     # hist_ssp.loc[is_hist, 'runoff_cumulative'],
        #     hist_ssp_melt.loc[is_hist, 'runoff_smoothed'],
        #     label=f"{exp.upper()} hist",
        #     color=colors[f'{exp}'][0],
        #     linestyle=ls, linewidth=lw_melt)
    
        # Plot future part (add)
        ax.plot(
            hist_ssp.loc[is_future, 'year'],
            # hist_ssp.loc[is_future, 'relative_runoff_smoothed'],
            # hist_ssp.loc[is_future, 'runoff_relative'],
            # hist_ssp.loc[is_future, 'runoff_cumulative'],
            hist_ssp.loc[is_future, 'runoff_smoothed'],
            label=f"{exp.upper()} {ssp}",
            color=colors[f'{exp}{add}'][s],
            linestyle=ls, linewidth=lw_tot
        )
        
        # Plot future part melt (add)
        # ax.plot(
        #     hist_ssp_melt.loc[is_future, 'year'],
        #     # hist_ssp_melt.loc[is_future, 'relative_runoff_smoothed']*100,
        #     # hist_ssp_melt.loc[is_future, 'runoff_cumulative'],
        #     hist_ssp_melt.loc[is_future, 'runoff_smoothed'],
        #     label=f"{exp.upper()} {ssp}",
        #     color=colors[f'{exp}{add}'][s],
        #     linestyle=ls, linewidth=lw_melt
        # )
        
        
        peak_year=hist_ssp.loc[hist_ssp['runoff_smoothed'].idxmax(), 'year']
        peak_magnitude=hist_ssp.loc[hist_ssp['runoff_smoothed'].idxmax(), 'runoff_smoothed']
        # peak_magnitude=hist_ssp.loc[hist_ssp['relative_runoff_smoothed'].idxmax(), 'relative_runoff_smoothed']
        
        peak_water_dict = {
            'rgi_subregion': 'hma',
            'peak_water_year': peak_year,
            'peak_magnitude': peak_magnitude,
            'experiment': exp,
            'ssp': ssp}
        
        peak_water_list.append(peak_water_dict)
        
        if peak_year<=2014:
            color='black'
        else:
            color=colors[f'{exp}{add}'][s]
        face_color='none' if exp=='noi' else color
        # ax.axvline(peak_year, color=color, linestyle=ls, lw=lw_tot-2)
        ax.scatter(peak_year, peak_magnitude,marker='o', s=100, facecolor=face_color, edgecolor=color, linewidths=5)  # thicker edge)
        print(peak_year, peak_magnitude)
        ax.ticklabel_format(style='scientific')
    
    if ax==None: #set figure labels later in overall figure
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
            'Total annual runoff [km$^3$]',
            # 'Total cumulative annual runoff [km$^3$]',
            # 'Total annual runoff change [%]',
            va='center',
            ha='left',
            fontsize=12,
            # fontweight='bold'
            rotation=90
        )
        
    
    ax.set_title('High Mountain Asia', fontweight='bold')
    ax.grid(True, color='grey', ls='--', lw=0.5)
    
    # line_patches = [mpatches.Patch(color=colors['irr'][0], label='Historical (W5E5)           '),
    #                   mpatches.Patch(color=colors['noi'][0], label='Historical NoIrr'),
    #                   mpatches.Patch(color=colors['irr_fut'][0], label='Future (CESM2), SSP-1.26'),
    #                   mpatches.Patch(color=colors['noi_fut'][0], label='Future NoIrr, SSP-1.26'),
    #                   mpatches.Patch(color=colors['irr_fut'][1], label='Future (CESM2), SSP-3.70'),
    #                     mpatches.Patch(color=colors['noi_fut'][1], label='Future NoIrr, SSP-3.70'),
    #                     ]

    # scatter_patches = [Line2D([0], [0], marker='o', lw=0,  markerfacecolor='grey', markeredgecolor='grey', mew=3, markersize=10, label='Peakwater Irr'),
    #                   Line2D([0], [0], marker='o', lw=0, markerfacecolor='none', markeredgecolor='grey', mew = 3, markersize=10, label='Peakwater NoIrr')]

    # legend_patches = line_patches+ scatter_patches
    # fig_tot.legend(handles=legend_patches, loc='lower center', ncols=4, bbox_to_anchor=(0.5,-0.1),frameon=False, fontsize=14)#,
    
    # Add to figure/axes
    if ax==None: #only set legend if single figure
        fig.legend(handles=legend_patches, loc='lower center', frameon=False, ncols=5, bbox_to_anchor=(0.5,-0.14))
   
    # plt.xlabel('Year')
    # plt.ylabel('Runoff [km³]')
    # plt.legend()
    # plt.show()
    opath_dict_pw = os.path.join(wd_path,'masters','hydro_peak_water_dictionary_subregions.csv')
    df = pd.DataFrame(peak_water_list)
    ax.tick_params(labelsize=14)
    # df.to_csv(opath_dict_pw, index=False) #only run when subregions on
    
    fig.text(-0.04, 0.5, 'Runoff (m³)', va='center', rotation='vertical', fontsize=14)
    fig.text(0.5, -0.02, 'Years', va='center', fontsize=14)
    
    
    
    # line_patches = [mpatches.Patch(color=colors['irr'][0], label='Historical (W5E5)           '),
    #                   mpatches.Patch(color=colors['noi'][0], label='Historical NoIrr'),
    #                   mpatches.Patch(color=colors['irr_fut'][0], label='Future (CESM2), SSP-1.26'),
    #                   mpatches.Patch(color=colors['noi_fut'][0], label='Future NoIrr, SSP-1.26'),
    #                   mpatches.Patch(color=colors['irr_fut'][1], label='Future (CESM2), SSP-3.70'),
    #                     mpatches.Patch(color=colors['noi_fut'][1], label='Future NoIrr, SSP-3.70'),
    #                     ]

    # scatter_patches = [Line2D([0], [0], marker='o', lw=0,  markerfacecolor='grey', markeredgecolor='grey', mew=3, markersize=10, label='Peakwater Irr'),
    #                   Line2D([0], [0], marker='o', lw=0, markerfacecolor='none', markeredgecolor='grey', mew = 3, markersize=10, label='Peakwater NoIrr')]
    
    
    line_patches = [
        mlines.Line2D([], [], color="darkgrey", linestyle='-', linewidth=2,
                      label='Irrigation'),
        mlines.Line2D([], [], color="lightgrey", linestyle='--', linewidth=2,
                      label='No Irrigation'),
        # mlines.Line2D([], [], color="white", linestyle='--', linewidth=2,
        #               label=''),
        
        
    ]
    
    color_patches = [mpatches.Patch(color=colors['irr'][0], label='Historical (W5E5)    '), 
                     mpatches.Patch(color=colors['irr_fut'][0], linestyle='-', linewidth=2,
                                   label='Future (CESM2), SSP1-2.6'),
                     mpatches.Patch(color=colors['irr_fut'][1], linestyle='-', linewidth=2,
                                   label='Future (CESM2), SSP3-7.0')]
                     
    
    scatter_patches = [Line2D([0], [0], marker='o', lw=0,  markerfacecolor='grey', markeredgecolor='grey', mew=3, markersize=10, label='Peakwater Irr'),
                      Line2D([0], [0], marker='o', lw=0, markerfacecolor='none', markeredgecolor='grey', mew = 3, markersize=10, label='Peakwater NoIrr')]

    legend_patches = color_patches + line_patches + scatter_patches
    
    axes["big"].legend(handles=legend_patches, loc='upper left', ncols=1, frameon=False, fontsize=14)#,
    


    return ax

ax_timeline =  annual_runoff_timeline_plot_hma_subregions()
# 
#%% Cell 8: make hydrological runoff plot with relative difference  
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
# df_avg_jja = df_avg_monthly[df_avg_monthly['month'].isin([6, 7, 8])]
df_avg_jja = df_avg_monthly[df_avg_monthly['month'].isin(np.arange(1,13,1))] #if we want full year

df_avg_subregions = (df_avg_jja.groupby(['year', 'experiment', 'ssp', 'rgi_subregion'])['runoff'].sum().reset_index()) #sum yo annual runoff for all subregions per year
# df_avg_subregions = (df_avg_monthly.groupby(['year', 'experiment', 'ssp', 'rgi_subregion'])['runoff'].sum().reset_index()) #sum yo annual runoff for all subregions per year
df_avg_subregions = df_avg_subregions[~df_avg_subregions['year'].isin([2074])] 
df_avg_all = (df_avg_subregions.groupby(['year', 'experiment', 'ssp'])['runoff'].sum().reset_index()) #sum for all of HMA

early_period = (1985, 2014)
late_period = (2044, 2074)

# Filter and compute early period averages
# df_early_avg = (
#     df_runoff_shares[(df_runoff_shares['year'] >= early_period[0]) & (df_runoff_shares['year'] <= early_period[1])]
#     .groupby(['rgi_subregion','experiment'])[['share_melt_on', 'share_melt_off', 'share_prcp_on', 'share_prcp_off']]
#     .mean()
# )

# # Same for late period
# df_late_avg = (
#     df_runoff_shares[(df_runoff_shares['year'] >= late_period[0]) & (df_runoff_shares['year'] <= late_period[1])]
#     .groupby(['rgi_subregion','experiment'])[['share_melt_on', 'share_melt_off', 'share_prcp_on', 'share_prcp_off']]
#     .mean()
# )

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

for region in df_avg_subregions['rgi_subregion'].unique():
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
    
    #Add inset axes for pie charts
    # inset_ax1 = inset_axes(ax[c], width="40%", height="40%", loc='upper left', borderpad=0)
    # inset_ax2 = inset_axes(ax[c], width="40%", height="40%", loc='lower right', borderpad=0)
    # wedges_outer, texts_outer = inset_ax1.pie(avg_early.loc[avg_late.index == 'noi'].values.flatten(),radius=1.0, labels=None, startangle=90,colors=component_colors_noirr,wedgeprops=dict(width=0.4, edgecolor='w'))
    # wedges_inner, texts_inner = inset_ax1.pie(avg_early.loc[avg_late.index == 'irr'].values.flatten(),radius=0.7,labels=None, startangle=90,colors=component_colors,wedgeprops=dict(width=0.4, edgecolor='w'))
    # centre_circle = plt.Circle((0, 0), 0.4, fc='white')
    # inset_ax1.add_artist(centre_circle)
    # inset_ax1.set(aspect='equal')
    # inset_ax1.set_title('')  # Remove default title position

    # wedges_outer, texts_outer = inset_ax2.pie(avg_late.loc[avg_late.index == 'noi'].values.flatten(),radius=1.0, labels=None, startangle=90,colors=component_colors_noirr,wedgeprops=dict(width=0.4, edgecolor='w'))
    # wedges_inner, texts_inner = inset_ax2.pie(avg_late.loc[avg_late.index == 'irr'].values.flatten(),radius=0.7,labels=None, startangle=90,colors=component_colors,wedgeprops=dict(width=0.4, edgecolor='w'))
    # centre_circle = plt.Circle((0, 0), 0.4, fc='white')
    # inset_ax2.add_artist(centre_circle)
    # inset_ax2.set(aspect='equal')
    # inset_ax2.set_title('')  # Remove default title position

    
    
    
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

# avg_early = df_early_avg.groupby('experiment').mean()
# avg_late = df_late_avg.groupby('experiment').mean()
# inset_ax1 = inset_axes(ax[0], width="40%", height="40%", loc='upper left', borderpad=0)
# inset_ax2 = inset_axes(ax[0], width="40%", height="40%", loc='lower right', borderpad=0)
# # wedges, texts =  inset_ax1.pie(avg_early, labels=None, startangle=90, autopct=None, colors=component_colors)#lambda pct: f"{int(round(pct))}%"
# wedges_outer, texts_outer = inset_ax1.pie(avg_early.loc[avg_late.index == 'noi'].values.flatten(),radius=1.0, labels=None, startangle=90,colors=component_colors_noirr,wedgeprops=dict(width=0.4, edgecolor='w'))
# wedges_inner, texts_inner = inset_ax1.pie(avg_early.loc[avg_late.index == 'irr'].values.flatten(),radius=0.7,labels=None, startangle=90,colors=component_colors,wedgeprops=dict(width=0.4, edgecolor='w'))
# centre_circle = plt.Circle((0, 0), 0.4, fc='white')
# inset_ax1.add_artist(centre_circle)
# inset_ax1.set(aspect='equal')
# inset_ax1.set_title('')  # Remove default title position

# wedges_outer, texts_outer = inset_ax2.pie(avg_late.loc[avg_late.index == 'noi'].values.flatten(),radius=1.0, labels=None, startangle=90,colors=component_colors_noirr,wedgeprops=dict(width=0.4, edgecolor='w'))
# wedges_inner, texts_inner = inset_ax2.pie(avg_late.loc[avg_late.index == 'irr'].values.flatten(),radius=0.7,labels=None, startangle=90,colors=component_colors,wedgeprops=dict(width=0.4, edgecolor='w'))

    
# centre_circle = plt.Circle((0, 0), 0.4, fc='white')
# inset_ax2.add_artist(centre_circle)
# inset_ax2.set(aspect='equal')
# inset_ax2.set_title('')  # Remove default title position
# inset_ax1.annotate('1985–2014', xy=(0.5, -0.15), ha='center', va='center', fontsize=10, xycoords='axes fraction')
# inset_ax2.set_title('2044–2074', fontsize=10)

# Plot pie in lower-right


pie_patches = [
    # mpatches.Patch(color=component_colors[0], label='Melt on glacier'),
    # mpatches.Patch(color=component_colors_noirr[0], label='Melt on glacier NoIrr'),
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
# plt.show()


#%% Cell 9: make dataset peak water per glacier aggregated 

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
                        melt_on = ds['melt_on_glacier_monthly'] * 1e-9 #only working with monthly components, can delete for already averaged, but than also delete summation later
                        melt_off = ds['melt_off_glacier_monthly'] * 1e-9 #convert the runoff components to the right unit
                        prcp_on = ds['liq_prcp_on_glacier_monthly'] * 1e-9
                        prcp_off = ds['liq_prcp_off_glacier_monthly'] * 1e-9
                        snow_on = ds['snowfall_on_glacier_monthly'] * 1e-9
                        snow_off = ds['snowfall_off_glacier_monthly'] * 1e-9
                        
                        # melt_on = ds['melt_on_glacier'] * 1e-9 #only working with monthly components, can delete for already averaged, but than also delete summation later
                        # melt_off = ds['melt_off_glacier'] * 1e-9 #convert the runoff components to the right unit
                        # prcp_on = ds['liq_prcp_on_glacier'] * 1e-9
                        # prcp_off = ds['liq_prcp_off_glacier'] * 1e-9
                        # snow_on = ds['snowfall_on_glacier'] * 1e-9
                        # snow_off = ds['snowfall_off_glacier'] * 1e-9
                        
                        
                        total_runoff = melt_on + melt_off + prcp_on + prcp_off + snow_on + snow_off #sum all the runoff components
                        runoff = xr.merge([total_runoff.to_dataset(name='runoff'), base_ds]).set_coords('rgi_subregion') #merge the runoff data with the subregion ids

                        # df_runoff = runoff.to_dataframe().reset_index()[['time',  'rgi_subregion', 'rgi_id', 'runoff']]                
                        # df_runoff = df_runoff.rename(columns={'time': 'year'})
                        df_runoff = runoff.to_dataframe().reset_index()[['time', 'month_2d', 'rgi_subregion', 'rgi_id', 'runoff']]                
                        df_runoff = df_runoff.rename(columns={'time': 'year', 'month_2d': 'month'})
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
                    
                    # melt_on = ds['melt_on_glacier'] * 1e-9
                    # melt_off = ds['melt_off_glacier'] * 1e-9
                    # prcp_on = ds['liq_prcp_on_glacier'] * 1e-9
                    # prcp_off = ds['liq_prcp_off_glacier'] * 1e-9
                    # snow_on = ds['snowfall_on_glacier'] * 1e-9
                    # snow_off = ds['snowfall_off_glacier'] * 1e-9


                    melt_on = ds['melt_on_glacier_monthly'] * 1e-9
                    melt_off = ds['melt_off_glacier_monthly'] * 1e-9
                    prcp_on = ds['liq_prcp_on_glacier_monthly'] * 1e-9
                    prcp_off = ds['liq_prcp_off_glacier_monthly'] * 1e-9
                    snow_on = ds['snowfall_on_glacier_monthly'] * 1e-9
                    snow_off = ds['snowfall_off_glacier_monthly'] * 1e-9                    
                    
                    total_runoff = melt_on + melt_off + prcp_on + prcp_off + snow_on + snow_off #sum all the runoff components
                    
                    runoff = xr.merge([total_runoff.to_dataset(name='runoff'), base_ds]).set_coords('rgi_subregion') #merge the runoff data with the subregion ids

                    # df_runoff = runoff.to_dataframe().reset_index()[['time', 'rgi_subregion', 'rgi_id', 'runoff']]                
                    # df_runoff = df_runoff.rename(columns={'time': 'year'})
                    
                    df_runoff = runoff.to_dataframe().reset_index()[['time', 'month_2d', 'rgi_subregion', 'rgi_id', 'runoff']]                
                    df_runoff = df_runoff.rename(columns={'time': 'year', 'month_2d': 'month'})
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
opath_df_monthly = os.path.join(wd_path,'masters','hydro_output_monthly_per_rgi_id.csv')
# opath_df_annual = os.path.join(wd_path,'masters','hydro_output_annual_subregions_per_rgi_id_preselected.csv')


df_monthly_all_concat.to_csv(opath_df_monthly)
# df_annual.to_csv(opath_df_annual)


#%% Cell 10: Convert runoff to annual, per rgi_ID

"""Group into annual - else kernel dies"""
opath_df_monthly = os.path.join(wd_path,'masters','hydro_output_monthly_per_rgi_id.csv') #work from monthly as annual processing comes later
df_monthly = pd.read_csv(opath_df_monthly)
df_monthly['ssp'] = df_monthly['ssp'].astype(str) #assure same type, otherwise strange grouping

df_avg_monthly = df_monthly
# df_avg_monthly = (df_monthly.groupby(['year','month', 'experiment', 'ssp', 'rgi_subregion'])['runoff'].median().reset_index()) # #group into subregions
# df_avg_monthly = (df_monthly.groupby(['year','month', 'experiment', 'ssp', 'rgi_subregion','rgi_id'])['runoff'].median().reset_index()) #average over members
df_avg_monthly = df_avg_monthly[df_avg_monthly['runoff'] != 0.00] #exclude zeros as last year of simulation does not work
# df_avg_jja = df_avg_monthly[df_avg_monthly['month'].isin([6, 7, 8])]
df_avg_jja = df_avg_monthly[df_avg_monthly['month'].isin(np.arange(1,13,1))] #if we want full year
df_avg_subregions = (df_avg_jja.groupby(['year', 'experiment', 'ssp', 'member', 'rgi_subregion','rgi_id'])['runoff'].sum().reset_index()) #sum yo annual runoff for all subregions per year
df_avg_subregions = df_avg_subregions[~df_avg_subregions['year'].isin([2014,2074])] 
# df_avg_all = (df_avg_subregions.groupby(['year', 'experiment', 'ssp','rgi_id'])['runoff'].sum().reset_index()) #sum for all subregions in HMA

opath_df_annual = os.path.join(wd_path,'masters','hydro_output_annual_per_rgi_id.csv') #work from monthly as annual processing comes later
df_avg_subregions.to_csv(opath_df_annual)


#%% Cell 11: stack historical to different ssps
# Separate historical and future data

opath_df_annual = os.path.join(wd_path,'masters','hydro_output_annual_per_rgi_id.csv') #work from annual, and align per member 
df_annual=pd.read_csv(opath_df_annual)

hist_df = df_annual[df_annual['ssp'] == 'hist'].copy()
future_df = df_annual[df_annual['ssp'] != 'hist'].copy()

# Step 1: NOI case → match by member
hist_noi = hist_df[hist_df['experiment'] == 'noi']
future_noi = future_df[future_df['experiment'] == 'noi']

noi_list = []
for ssp in future_noi['ssp'].unique():
    for member in [1, 2, 3]:
        hist_part = hist_noi[hist_noi['member'] == member].copy()
        hist_part['ssp'] = ssp  # update SSP label
        future_part = future_noi[(future_noi['ssp'] == ssp) & (future_noi['member'] == member)]
        noi_list.append(pd.concat([hist_part, future_part]))

df_noi = pd.concat(noi_list, ignore_index=True)

# For IRR → hist has only member=1, duplicate to all future members and update ssp
hist_irr = hist_df[hist_df['experiment'] == 'irr']
future_irr = future_df[future_df['experiment'] == 'irr']

irr_list = []
for ssp in future_irr['ssp'].unique():
    for member in [1, 2, 3]:
        hist_part = hist_irr.copy()
        hist_part['member'] = member
        hist_part['ssp'] = ssp  # update SSP label
        future_part = future_irr[(future_irr['ssp'] == ssp) & (future_irr['member'] == member)]
        irr_list.append(pd.concat([hist_part, future_part]))

df_irr = pd.concat(irr_list, ignore_index=True)

# Combine both
df_combined = pd.concat([df_noi, df_irr], ignore_index=True)
opath_df_annual = os.path.join(wd_path,'masters','hydro_output_annual_per_rgi_id_preselected_stacked_histssp.csv')
df_combined.to_csv(opath_df_annual)

df_combined_subregions = df_combined.groupby(['year', 'experiment', 'ssp', 'member', 'rgi_subregion'])['runoff'].sum().reset_index()
opath_df_annual_subregions = os.path.join(wd_path,'masters','hydro_output_annual_per_subregion_preselected_stacked_histssp.csv')
df_combined_subregions.to_csv(opath_df_annual_subregions)



#%% Cell 12:  Process and calculate the peak water time per glacier per member and across 3 members

#operation is quite time consuming as contains many rgi_ids, years combinations


opath_df_annual = os.path.join(wd_path,'masters','hydro_output_annual_per_rgi_id_preselected_stacked_histssp.csv')
df_annual = pd.read_csv(opath_df_annual)


# df_annual=df_avg_subregions
df_annual['runoff_11yr_rolling'] = (
    df_annual.groupby(['rgi_subregion', 'experiment', 'ssp','member', 'rgi_id', ])['runoff']
      .transform(lambda x: x.rolling(window=11, center=True, min_periods=1).mean()) #using transform returns a series aligned to the orgiinal df index, so ungrouped columns remain
)

# Calculate the median across members
df_median = (
    df_annual.groupby(['rgi_subregion', 'experiment', 'ssp', 'rgi_id', 'year'], as_index=False)['runoff_11yr_rolling']
    .median())

group_cols = ['rgi_subregion', 'experiment', 'ssp','member','rgi_id']

# Add a 'member' label for the median
df_median['member'] = 'member_median'

# Append to df_annual
df_annual_with_median = pd.concat([df_annual, df_median], ignore_index=True)

# Now continue with the same peak water calculation but using df_annual_with_median
idx_max_all = df_annual_with_median.groupby(group_cols)['runoff_11yr_rolling'].idxmax()
idx_max_clean_all = idx_max_all.dropna().astype(int)

df_annual_with_median['year_of_max_runoff'] = pd.NA
df_annual_with_median.loc[idx_max_clean_all, 'year_of_max_runoff'] = df_annual_with_median.loc[idx_max_clean_all, 'year']

summary_all = (
    df_annual_with_median.loc[idx_max_clean_all, group_cols + ['year', 'runoff_11yr_rolling']]
    .rename(columns={'year': 'year_of_max_runoff', 'runoff_11yr_rolling': 'max_runoff_11yr'})
    .reset_index(drop=True)
)


opath_df_summary = os.path.join(wd_path,'masters','hydro_output_peakwater_per_rgi_id_member_avg.csv')
summary_all.to_csv(opath_df_summary)

#%% Cell 13a:  Process and calculate the peak water time per subregion per member and across 3 members

#operation is quite time consuming as contains many rgi_ids, years combinations


opath_df_annual = os.path.join(wd_path,'masters','hydro_output_annual_per_subregion_preselected_stacked_histssp.csv')
df_annual = pd.read_csv(opath_df_annual)


# df_annual=df_avg_subregions
df_annual['runoff_11yr_rolling'] = (
    df_annual.groupby(['rgi_subregion', 'experiment', 'ssp','member' ])['runoff']
      .transform(lambda x: x.rolling(window=11, center=True, min_periods=1).mean()) #using transform returns a series aligned to the orgiinal df index, so ungrouped columns remain
)

# Calculate the median across members
df_median = (
    df_annual.groupby(['rgi_subregion', 'experiment', 'ssp',  'year'], as_index=False)['runoff_11yr_rolling']
    .median())



group_cols = ['rgi_subregion', 'experiment', 'ssp','member']

# Add a 'member' label for the median
df_median['member'] = 'member_median'

df_median.to_csv(os.path.join(wd_path,'masters','hydro_output_peakwater_per_subregion_member_avg_ts.csv'))

# Append to df_annual
df_annual_with_median = pd.concat([df_annual, df_median], ignore_index=True)

# Now continue with the same peak water calculation but using df_annual_with_median
idx_max_all = df_annual_with_median.groupby(group_cols)['runoff_11yr_rolling'].idxmax()
idx_max_clean_all = idx_max_all.dropna().astype(int)

df_annual_with_median['year_of_max_runoff'] = pd.NA
df_annual_with_median.loc[idx_max_clean_all, 'year_of_max_runoff'] = df_annual_with_median.loc[idx_max_clean_all, 'year']

summary_all = (
    df_annual_with_median.loc[idx_max_clean_all, group_cols + ['year', 'runoff_11yr_rolling']]
    .rename(columns={'year': 'year_of_max_runoff', 'runoff_11yr_rolling': 'max_runoff_11yr'})
    .reset_index(drop=True)
)

opath_df_summary = os.path.join(wd_path,'masters','hydro_output_peakwater_per_subregion_member_avg.csv')
summary_all.to_csv(opath_df_summary)
#%% Cell 13b: Link rgi subregions to coordinates 

opath_dict_pw = os.path.join(wd_path,'masters','hydro_output_peakwater_per_subregion_member_avg.csv')
pw = pd.read_csv(opath_dict_pw)

df = pd.read_csv(
    f"{wd_path}masters/master_gdirs_r3_a1_rgi_date_A_V_RGIreg.csv")

master_ds = df.reset_index(drop=True)
master_ds = master_ds[['rgi_id', 'rgi_region', 'rgi_subregion', 'full_name', 'cenlon', 'cenlat', 'rgi_date',
                       'rgi_area_km2', 'rgi_volume_km3']]

master_ds_unique = master_ds.drop_duplicates(subset='rgi_subregion')
merged_df = pw.merge(master_ds_unique, on='rgi_subregion', how='left')

merged_df.loc[merged_df['rgi_subregion'] == 'hma', 'full_name'] = 'High Mountain Asia'

opath_merged_df = os.path.join(wd_path,'masters','hydro_output_peakwater_per_subregion_member_avg_rgi_info.csv')
merged_df.to_csv(opath_merged_df)

#%% Cell 13c: Link rgi_id to coordinates for filtered dataset

opath_df_summary = os.path.join(wd_path,'masters','hydro_output_peakwater_per_rgi_id_member_avg.csv')
# opath_df_summary = os.path.join(wd_path,'masters','hydro_output_peakwater_per_subregion_member_avg.csv')
summary = pd.read_csv(opath_df_summary)

plot_df = summary.dropna(subset=['year_of_max_runoff'])
plot_df = plot_df[plot_df['member']=='member_median'] #only plot for member median
plot_df['ssp'] = plot_df['ssp'].astype(str) #assure same type, otherwise strange grouping
plot_df['experiment'] = plot_df['experiment'].astype(str) #assure same type, otherwise strange grouping


#merge with info on location and size

print("merging datasets")
df = pd.read_csv(
    f"{wd_path}masters/master_gdirs_r3_a1_rgi_date_A_V_RGIreg_B_hugo.csv")
master_ds = df[(~df['sample_id'].str.endswith('0')) |  # Exclude all the model averages ending with 0 except for IPSL
               (df['sample_id'].str.startswith('IPSL'))]
master_ds = master_ds.reset_index(drop=True)
master_ds = master_ds[['rgi_id', 'rgi_region', 'rgi_subregion', 'full_name', 'cenlon', 'cenlat', 'rgi_date',
                       'rgi_area_km2', 'rgi_volume_km3']]

merged_sum = summary.merge(master_ds, on='rgi_id', how='left')

opath_df_summary = os.path.join(wd_path,'masters','hydro_output_peakwater_per_rgi_id_rgi_info.csv')
merged_sum.to_csv(opath_df_summary)
#%% Cell 14a: Create peak water plots on map (key figure - extended data)

# merged_sum = plot_df.merge(master_ds, on='rgi_subregion', how='left')

opath_df_summary = os.path.join(wd_path,'masters','hydro_output_peakwater_per_rgi_id_rgi_info.csv')
plot_df = pd.read_csv(opath_df_summary)
plot_df=plot_df[plot_df['member']=='member_median']
#regrid based on 1x1 degree grid
plot_df['lon_bin'] = plot_df['cenlon'].floordiv(1).astype(int)
plot_df['lat_bin'] = plot_df['cenlat'].floordiv(1).astype(int)
plot_df_grid = plot_df.groupby(['lon_bin', 'lat_bin','experiment','ssp']).agg({
          'year_of_max_runoff': 'mean',
          'max_runoff_11yr': 'sum',  # or 'mean' depending on your logic
          'rgi_area_km2': 'sum',  # or 'mean' depending on your logic
      }) .reset_index()
      
      

print("start plotting")
"""load basin outlines"""
path = "/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Research/01. IRRMIP/03. Data/04. Reference/02. Hydro data/global_30_minute_potential_network_v601_asc/g_basin.asc"
with rasterio.open(path) as src:
    data = src.read(1)
    transform = src.transform

# Convert raster to polygons
geoms, basin_ids = [], []
unique_basins = np.unique(data)
for basin_id in unique_basins:
    if basin_id > 0 and not np.isnan(basin_id):
        mask = (data == basin_id)
        shapes = rasterio.features.shapes(mask.astype(np.uint8), mask=mask, transform=transform)
        for geom, val in shapes:
            if val == 1:
                geoms.append(shape(geom))
                basin_ids.append(basin_id)

gdf = gpd.GeoDataFrame({'basin_id': basin_ids, 'geometry': geoms}, crs='EPSG:4326')

# Clip geometries to HMA
gdf_hma = gdf.cx[65:110, 23:47]

#Load shapefile
shapefile_path = '/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Research/01. IRRMIP/03. Data/01. Input files/03. Shapefile/Karakoram/Pan-Tibetan Highlands/Pan-Tibetan Highlands (Liu et al._2022)/Shapefile/Pan-Tibetan Highlands (Liu et al._2022)_P.shp'
shp = gpd.read_file(shapefile_path)
target_crs = 'EPSG:4326'
shp = shp.to_crs(target_crs)

fig, axes = plt.subplots(2,2, figsize=(10, 7), subplot_kw={'projection': ccrs.PlateCarree()})
axes=axes.flatten()
"""Plot for ssps x experiment combinations"""
axid=0
scatter="off"
for exp in ['irr', 'noi']:
    for ssp in [126,370]:
        ax=axes[axid]
        
        plot_df_sel = plot_df_grid[(plot_df_grid['experiment'] == exp) & (plot_df_grid['ssp'] == ssp)]
        # plot_df_sel = plot_df[(plot_df['experiment'] == exp) & (plot_df['ssp'] == ssp)]
        
        # area_scaling = 80  # adjust to your liking
        # marker_sizes = plot_df_sel['rgi_area_km2'] / area_scaling
        marker_sizes=(np.sqrt(plot_df_sel['rgi_area_km2'])/5)**1.3
        
        
        # Create plot
        
        #include dem in background
        dem_path = "/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Research/01. IRRMIP/03. Data/01. Input files/02. DEM/ETOPO2v2c_f4.nc"
        with xr.open_dataset(dem_path) as ds:
            dem=ds['z']
        
        # Print dataset info to find variable names
        
        dem = dem.where((dem.x >= 65) & (dem.x <= 110) & (dem.y >= 23) & (dem.y <= 47), drop=True)
        dx =(dem['x'][1]-dem['x'][0]).values
        dy =(dem['y'][1]-dem['y'][0]).values
        z = dem.values
        x, y = np.meshgrid(dem.x, dem.y)
        
        ls = LightSource(azdeg=315, altdeg=45)
        
        im = ax.imshow(ls.hillshade(z, vert_exag=10, dx=0.01, dy=0.01),extent=[dem.x.min(), dem.x.max(), dem.y.min(), dem.y.max()],
                  transform=ccrs.PlateCarree(), alpha=0.1, cmap="grey_r")  # Adjust alpha for effect
        
        # Add geographical features
        ax.add_feature(cfeature.RIVERS, edgecolor='blue', facecolor='none', linewidth=0.6, alpha=0.6)
        
        # Add basemap features
        ax.add_feature(cfeature.COASTLINE)
        
        # Add big catchment outlines
        # gdf.plot(ax=ax, edgecolor='grey', facecolor='none', alpha=0.7)
        
        #plot shapefile of hma
        shp.plot(ax=ax, edgecolor='black', linewidth=1, facecolor='none')

        
        
            # Define colormap similar to your reference (red-yellow-white-blue reversed)
        cmap = plt.cm.get_cmap('RdYlBu') #for rounce
        # cmap = plt.cm.jet_r#get_cmap('gist_rainbow') #for huss and hock
        
        # Set normalization range
        # norm = clrs.Normalize(vmin=1980, vmax=2100)
        norm = clrs.Normalize(vmin=2020, vmax=2080)
            # 
            # Scatter plot with updated colormap and normalization
            # scatter = ax.scatter(
            #     plot_df_sel['cenlon'], plot_df_sel['cenlat'],
            #     c=plot_df_sel['year_of_max_runoff'], cmap=cmap, norm=norm,
            #     s=marker_sizes, alpha=0.7, edgecolor='none',
            #     transform=ccrs.PlateCarree())
            
        scatter = ax.scatter(
            plot_df_sel['lon_bin'], plot_df_sel['lat_bin'],
            c=plot_df_sel['year_of_max_runoff'], cmap=cmap, norm=norm,
            s=marker_sizes, alpha=0.7, edgecolor='none',
            transform=ccrs.PlateCarree())
            
            # Colorbar with custom settings
        
        
        # Title and labels
        ssp_txt="3-7.0" if ssp==370 else "1-2.6"
        exp_txt="Irr" if exp=="irr" else "NoIrr"
        index = ["a.","b.","c.","d."][axid]
        ax.set_title(f'{index} SSP{ssp_txt}, {exp_txt}',  # so coordinates are relative to axis
            loc='right', va='top',    # align to top-right corner
            fontsize=12,              # optional: adjust font size
            fontweight='bold'
        )
        
        ax.set_extent([dem.x.min(), dem.x.max(), dem.y.min(), dem.y.max()], crs=ccrs.PlateCarree())
        axid+=1
        
        
"""Include legends"""

custom_sizes = [200, 500, 2000,5000]#500, 1000, 2000, 3000]  # Example sizes for the legend (area)
size_labels = [f"{size:.0f}" for size in custom_sizes]  # Create labels for sizes

# Create legend handles (scatter points with different sizes)
legend_handles = [
    plt.scatter([], [], s=(np.sqrt(size)/5)**1.3, edgecolor='k', facecolor='none')
    for size in custom_sizes
]  # Adjust size factor if needed
text_handles = [Line2D([0], [0], linestyle="none", label=label) for label in size_labels]



# Create a separate axis for the legend
cax = fig.add_axes([0.9,0.05, 0.2,0.4])  # [left, bottom, width, height]
cax.set_frame_on(False)  # Hide frame
cax.set_xticks([])  # Remove x-ticks
cax.set_yticks([])  # Remove y-ticks
cbar = fig.colorbar(scatter, ax=cax, orientation='vertical', fraction=0.4, pad=0)
cbar.set_label('Peakwater [year]')   


cax_legend = fig.add_axes([0.95,0.5, 0.2,0.5])
# Remove axis visuals
cax_legend.set_frame_on(False)  # Hide frame
cax_legend.set_xticks([])  # Remove x-ticks
cax_legend.set_yticks([])  # Remove y-ticks

# Add legend to the separate axis
legend = cax_legend.legend(
    legend_handles, size_labels, loc="center",
    fontsize=12, ncol=1,frameon=False, title="Total Area \n km$^2$", title_fontsize=12, scatterpoints=1, labelspacing=1)

plt.tight_layout()
# plt.savefig(os.join_path())
fig_path = '/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Research/01. IRRMIP/04. Figures/98. Final Figures A+1km2/'
os.makedirs(f"{fig_path}/Extended_Data", exist_ok=True)
plt.savefig(f"{fig_path}/Extended_Data/Peak_water_mapplot.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
plt.show()

#%% Cell 14b: Sense check for comparison - create runoff timeline overview
opath_df_summary_ts = os.path.join(wd_path,'masters','hydro_output_peakwater_per_subregion_member_avg_ts.csv')
df_annual_with_median =pd.read_csv(opath_df_summary_ts)

fig,axes = plt.subplots(5,3, figsize=(10,7))
axes = axes.flatten()

for s, ssp in enumerate([126,370]):
    for exp in ['irr', 'noi']:
        ls='--' if exp=='noi' else '-'
        for idx, subregion in enumerate(df_annual_with_median.rgi_subregion.unique()):
            ts = df_annual_with_median[(df_annual_with_median['rgi_subregion']==subregion)
                                       & (df_annual_with_median['experiment']==exp)
                                       & (df_annual_with_median['ssp']==ssp)
                                       & (df_annual_with_median['member']=="member_median")]
            
            axes[idx].plot(ts.year, ts.runoff_11yr_rolling, color=colors[f'{exp}_fut'][s], ls=ls)
        
        

 
#%% Cell 14c: create scatter plot with peak water years (key figure - subpanel 3)
# 
def peak_water_scatter_plot(ax=None):
# ax=None
# === Load Data ===
    # opath_df_summary = os.path.join(wd_path, 'masters', 'hydro_output_peakwater_per_rgi_id_subregions.csv')
    opath_df_summary = os.path.join(wd_path, 'masters', 'hydro_peak_water_dictionary_subregions_rgi_info.csv') 
    # opath_df_summary = os.path.join(wd_path, 'masters', 'hydro_output_peakwater_per_subregion_member_avg_rgi_info.csv')
    
    
    # opath_df_summary = os.path.join(wd_path, 'masters', 'hydro_output_peakwater_per_subregion_subregions.csv')
    merged_sum = pd.read_csv(opath_df_summary)
    
    merged_sum.loc[merged_sum['rgi_subregion'] == "13-02", 'full_name'] = "Pamir"
    merged_sum.loc[merged_sum['rgi_subregion'] == "13-04", 'full_name'] = "East Tien Shan"
    merged_sum.loc[merged_sum['rgi_subregion'] == "13-06", 'full_name'] = "East Kun Lun"
    
    merged_sum = merged_sum.rename(columns={
        'peak_water_year': 'avg_year_of_max_runoff'})
    
    print(merged_sum.columns)
    # === Group Definitions ===
    group_cols_subregions = ['full_name', 'experiment', 'ssp']
    group_cols_hma = ['experiment', 'ssp']
    
    # === Calculate Averages ===
    # avg_subregions = (
    #     # median_df.groupby(group_cols_subregions)['year_of_max_runoff']
    #     merged_sum.groupby(group_cols_subregions)['year_of_max_runoff']
    #     .mean().reset_index(name='avg_year_of_max_runoff')
    # )
    
    
    
    # avg_hma = (
    #     # median_df.groupby(group_cols_hma)['year_of_max_runoff']
    #     merged_sum.groupby(group_cols_hma)['year_of_max_runoff']
    #     .mean().reset_index(name='avg_year_of_max_runoff')
    # )
    
    # avg_hma['full_name'] = 'High Mountain Asia'
    
    # === Combine Subregions + HMA ===
    # avg_total = pd.concat([avg_hma, avg_subregions], ignore_index=True)
    
    avg_total = merged_sum
    
    # === Reverse Subregion Order ===
    # subregion_order = avg_total['full_name'].drop_duplicates().tolist()[::-1]
    subregion_list = avg_total['full_name'].drop_duplicates().tolist()
    subregion_order =  subregion_list[:-1][::-1] + [subregion_list[-1]]
    
    # Optional: map subregions to y positions
    y_positions = {subregion: idx for idx, subregion in enumerate(subregion_order)}
    
    # === Plot ===
    if ax==None:
        fig, ax = plt.subplots(figsize=(8, 6))
        internal_plot=True
    else:
        internal_plot=False
    
    add = "_fut"
    s = 0 if ssp == "126" else 1
    
    for subregion, idx in y_positions.items():
        print(subregion)
        subset = avg_total[avg_total['full_name'] == subregion]
        for _, row in subset.iterrows():
            exp = f"{row['experiment']}_fut"
            ssp_code = {126: 0, 370: 1}[row['ssp']]
            color=colors[exp][ssp_code]
            facecolor= 'none' if row['experiment']=='noi' else color
            ax.scatter(
                # row['year_of_max_runoff'],
                row['avg_year_of_max_runoff'],
                [idx],
                label=subregion,
                facecolor=facecolor,
                edgecolor=color,
                linewidths=3,
                s=100
            )
            
        offset=0.5
        
        ax.text(
        x= 1983,  # choose a reasonable x-position (left of scatter points)
        y=idx,
        s=subregion,
        va='center',
        ha='left',
        fontsize=12
        )
        
        if idx % 2 == 0:
            ax.axhspan(
                idx - 0.5, idx + 0.5,
                color='lightgrey',
                alpha=0.3,
                zorder=0
            )
            
    ax.set_ylim(-0.5, len(subregion_order) - 0.5)
    
    # if internal_plot==True:
    #     ax.set_xlim(1985,2073)
    # ax.set_yticks([])
    ax.tick_params(axis='y', left=False, labelleft=False)
    
    # ax.set_xlabel("Mean Year of Max Runoff")
    # ax.set_ylabel("Region (HMA + Subregions)")
    return ax
# peak_water_scatter_plot()



#%% Cell 15: Make total runoff overview - 3 panels, calling functions (key figure)

fig_tot, axes_tot = plt.subplots(3,1, figsize=(11,12), constrained_layout=False, gridspec_kw={'height_ratios': [1, 1, 2]}, sharex=True)  # bottom plot = 2x height)
historic_future_volume_evolution_hma(ax=axes_tot[0]) #plot future glacier volume evolution
axes_tot[0].set_title('a. Glacier volume evolution', fontweight='bold', loc='right')
axes_tot[0].set_ylabel('Volume [% rel. to 1985 irr]', fontsize=14)

print("starting annual runoff plot")
annual_runoff_timeline_plot_hma(ax=axes_tot[1]) #plot total runoff over time in second panel
axes_tot[1].set_title('b. Total annual runoff', fontweight='bold', loc='right')
axes_tot[1].set_ylabel('Total annual runoff [km$^3$]', fontsize=14)
axes_tot[1].yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
axes_tot[1].ticklabel_format(axis='y', style='sci', scilimits=(3,3))

peak_water_scatter_plot(ax=axes_tot[2])
axes_tot[2].set_title('c. Average peak water', fontweight='bold', loc='right')
axes_tot[2].set_ylabel('Subregions', fontsize=14)

fig_tot.text(0.5,0,'Years', fontsize=14)

line_patches = [mpatches.Patch(color=colors['irr'][0], label='Historical (W5E5)           '),
                  mpatches.Patch(color=colors['noi'][0], label='Historical NoIrr'),
                  mpatches.Patch(color=colors['irr_fut'][0], label='Future (CESM2), SSP-1.26'),
                  mpatches.Patch(color=colors['noi_fut'][0], label='Future NoIrr, SSP-1.26'),
                  mpatches.Patch(color=colors['irr_fut'][1], label='Future (CESM2), SSP-3.70'),
                    mpatches.Patch(color=colors['noi_fut'][1], label='Future NoIrr, SSP-3.70'),
                    ]

scatter_patches = [Line2D([0], [0], marker='o', lw=0,  markerfacecolor='grey', markeredgecolor='grey', mew=3, markersize=10, label='Peakwater Irr'),
                  Line2D([0], [0], marker='o', lw=0, markerfacecolor='none', markeredgecolor='grey', mew = 3, markersize=10, label='Peakwater NoIrr')]

legend_patches = line_patches+ scatter_patches
fig_tot.legend(handles=legend_patches, loc='lower center', ncols=4, bbox_to_anchor=(0.5,-0.1),frameon=False, fontsize=12)#,
    

plt.tight_layout()

# plt.savefig(os.join_path())
fig_path = '/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Research/01. IRRMIP/04. Figures/98. Final Figures A+1km2/'
# plt.savefig(f"{fig_path}/Future_Runoff_Evolution.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
plt.show()
plt.show()


#%% Cell 15b: Make total runoff overview - 2 panels, calling functions (key figure)

fig_tot, axes_tot = plt.subplots(2,1, figsize=(8,8), constrained_layout=False, gridspec_kw={'height_ratios': [1, 1]}, sharex=True, dpi=300)  # bottom plot = 2x height)
historic_future_volume_evolution_hma(ax=axes_tot[0]) #plot future glacier volume evolution
# axes_tot[0].set_title('a. Glacier volume evolution', fontweight='bold', loc='right')
axes_tot[0].set_ylabel('Volume [% rel. to 1985 irr]', fontsize=14)

print("starting annual runoff plot")
annual_runoff_timeline_plot_hma_relative(ax=axes_tot[1]) #plot total runoff over time in second panel
# axes_tot[1].set_title('b. Total annual runoff', fontweight='bold', loc='right')
# axes_tot[1].set_ylabel('Total annual runoff [km$^3$]', fontsize=14)
axes_tot[1].set_ylabel('Total annual runoff change [%]', fontsize=14)
axes_tot[1].yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
# axes_tot[1].ticklabel_format(axis='y', style='sci', scilimits=(3,3))

# peak_water_scatter_plot(ax=axes_tot[2])
# axes_tot[2].set_title('c. Average peak water', fontweight='bold', loc='right')
# axes_tot[2].set_ylabel('Subregions', fontsize=14)

fig_tot.text(0.5,0,'Years', fontsize=14)

# line_patches = [mpatches.Patch(color=colors['irr'][0], label='Historical (W5E5)           '),
#                   mpatches.Patch(color=colors['noi'][0], label='Historical NoIrr'),
#                   mpatches.Patch(color=colors['irr_fut'][0], label='Future (CESM2), SSP-1.26'),
#                   mpatches.Patch(color=colors['noi_fut'][0], label='Future NoIrr, SSP-1.26'),
#                   mpatches.Patch(color=colors['irr_fut'][1], label='Future (CESM2), SSP-3.70'),
#                     mpatches.Patch(color=colors['noi_fut'][1], label='Future NoIrr, SSP-3.70'),
#                     ]

line_patches = [
    mlines.Line2D([], [], color="darkgrey", linestyle='-', linewidth=2,
                  label='Irrigation'),
    mlines.Line2D([], [], color="lightgrey", linestyle='--', linewidth=2,
                  label='No Irrigation'),
    mlines.Line2D([], [], color="white", linestyle='--', linewidth=2,
                  label=''),
    
    
]

color_patches = [mpatches.Patch(color=colors['irr'][0], label='Historical (W5E5)    '), 
                 mpatches.Patch(color=colors['irr_fut'][0], linestyle='-', linewidth=2,
                               label='Future (CESM2), SSP1-2.6'),
                 mpatches.Patch(color=colors['irr_fut'][1], linestyle='-', linewidth=2,
                               label='Future (CESM2), SSP3-7.0')]
                 

scatter_patches = [Line2D([0], [0], marker='o', lw=0,  markerfacecolor='grey', markeredgecolor='grey', mew=3, markersize=10, label='Peakwater Irr'),
                  Line2D([0], [0], marker='o', lw=0, markerfacecolor='none', markeredgecolor='grey', mew = 3, markersize=10, label='Peakwater NoIrr')]

# legend_patches = line_patches+ scatter_patches
# fig_tot.legend(handles=legend_patches, loc='lower center', ncols=4, bbox_to_anchor=(0.5,-0.1),frameon=False, fontsize=12)#,
    
axes_tot[0].legend(handles=line_patches+color_patches, loc='lower left', bbox_to_anchor=(0.01,0.02), ncols=1, frameon='true')
axes_tot[1].legend(handles=scatter_patches, loc='upper left', bbox_to_anchor=(0.01,0.95), ncols=1, frameon='true')

plt.tight_layout()

# plt.savefig(os.join_path())
fig_path = '/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Research/01. IRRMIP/04. Figures/98. Final Figures A+1km2/'
# plt.savefig(f"{fig_path}/Future_Runoff_Evolution.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
plt.show()
plt.show()








#%% Cell 16: Open and process reference data

rho = 0.917 #kg/m3
Rounce_path = f"/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Research/01. IRRMIP/03. Data/04. Reference/01. Climate data/Rounce/Global_reg_allvns_50sets_2000_2100-ssps.nc"#"R13_glac_mass_annual_50sets_2000_2100-rcp26.nc"
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




#%% Cell 16b: Plot reference data on top of our future simulations

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
# Rounce_path = f"/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Research/01. IRRMIP/03. Data/04. Reference/01. Climate data/Rounce/Global_reg_allvns_50sets_2000_2100-ssps.nc"#"R13_glac_mass_annual_50sets_2000_2100-rcp26.nc"
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

Aguayo_path = f"/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Research/01. IRRMIP/03. Data/04. Reference/01. Climate data/Aguayo/merged_glacier_simulations.nc"#"R13_glac_mass_annual_50sets_2000_2100-rcp26.nc"
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

#%% Cell 16c: Plot reference data on top of our future simulations - relative to 2015

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
# Rounce_path = f"/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Research/01. IRRMIP/03. Data/04. Reference/01. Climate data/Rounce/Global_reg_allvns_50sets_2000_2100-ssps.nc"#"R13_glac_mass_annual_50sets_2000_2100-rcp26.nc"
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

Aguayo_path = f"/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Research/01. IRRMIP/03. Data/04. Reference/01. Climate data/Aguayo/merged_glacier_simulations.nc"#"R13_glac_mass_annual_50sets_2000_2100-rcp26.nc"
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


