# -*- coding: utf-8 -*-

#%% Plot temperature and precipitation
import oggm
from oggm import cfg, utils, workflow, tasks, DEFAULT_BASE_URL

import geopandas as gpd
import numpy as np
import os
from oggm.shop import rgitopo
from oggm import GlacierDirectory
from oggm.core import massbalance, flowline
# Libs
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from oggm.shop import cru, w5e5
from oggm.tasks import process_w5e5_data

import matplotlib.pyplot as plt
from oggm import graphics

import pandas as pd
import logging
import shutil
import argparse
import time
import logging
import json


#%% Provide modelling parameters

folder_path='/Users/magaliponds/Documents/00. Programming'
wd_path=f'{folder_path}/06. Modelled perturbation-glacier interactions/'

y0_clim=1985
ye_clim=2014
y0_spinup=1980
ye_spinup=2020

cmap=cm.get_cmap('bone')
colors=[cmap(i / 9) for i in range(10)]

# Specify the RGI region you are interested in, for example, region 13 (Central Asia)
rgi_reg = '13'  # this must fit to example glacier(s), if starting from level 0
rgi_ids = ["RGI60-13.40102", # Area >10km2
    "RGI60-13.39195", # Area >10km2
    "RGI60-13.36881", # Area >10km2
    "RGI60-13.38969", # Area >10km2
    "RGI60-13.37184", # Area >10km2
    "RGI60-13.00967", # Area <10km2
    "RGI60-13.40982", # Area <10km2
    "RGI60-13.41891", # WGMS observation
    "RGI60-13.23659", # WGMS observation
] 

base_url = ('https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/2023.3/elev_bands/W5E5/')
prepro_level=3
sum_dir=os.path.join(wd_path, 'summary')
#Provide output names:

#%% Cell 11: Run OGGM
members=[1,3,4,6,1]
models = "W5E5","E3SM","CESM2","CNRM","IPSL-CM6"
timeframe="monthly"


#%% Plot temperatur and precipitation - against reference climate
colors = {
    "W5E5": ["#000000"],  # Black
    "E3SM": ["#785EF0", "#8F7BF1", "#A6A8F2"],  # Darker to lighter shades of purple
    "CESM2": ["#DC267F", "#E58A9E", "#F0A2B6", "#F7BCC4"],  # Darker to lighter shades of pink
    "CNRM": ["#FE6100", "#FE7D33", "#FE9A66", "#FEB799", "#FECDB5", "#FEF1E1"],  # Darker to lighter shades of orange
    "IPSL-CM6": ["#FFB000"]  # Dark purple to lighter shades
}

members=[3,4,6,1,1]
models = ["E3SM","CESM2","CNRM","IPSL-CM6","W5E5"]
markers=["o","x"]
for v,var in enumerate(["prcp", "temp"]):
    fig,axes=plt.subplots(3,3,figsize=(15,8))
    for r, rgi_id in enumerate(rgi_ids):
        row = r // axes.shape[1]
        col = r % axes.shape[1]
        

        for m, model in enumerate(models):
            if m==4:
                for member in range(members[m]):
                    #Create a sample-id tag to list model restuls
                    sample_id= f"{model}.00{member}" #p
                    # Why no valid volume file?? 
                    opath = os.path.join(wd_path, "per_glacier", rgi_id[:-6], rgi_id[:-3],rgi_id, 'climate_historical_{}.nc'.format(sample_id))
                
                    # ds_diag_ext = pd.read_csv(opath)
                    # ds_diag_ext = xr.Dataset.from_dataframe(ds_diag_ext)
                    ds_diag_ext=xr.open_dataset(opath)
                    # print(model, " ", ds_diag_ext.time)
                    axes[row,col].scatter(ds_diag_ext["time"], ds_diag_ext[var], label=sample_id, marker=markers[v], color=colors[model][member])
                    axes[row,col].set_title(rgi_id)
                    if r == 3:
                        axes[row,col].set_ylabel(ds_diag_ext[var].long_name+" ["+ds_diag_ext[var].units+"]")
                    if r >= 6:
                        axes[row, col].set_xlabel("Time [year]")
                    # print(ds_diag_ext["time"])

        opath_ref = os.path.join(wd_path, "per_glacier", rgi_id[:-6], rgi_id[:-3],rgi_id, 'climate_historical.nc')
        # print("Reference", " ", ds_diag_ext.time)

        ds_diag_ext_ref = xr.open_dataset(opath_ref)
        ds_diag_ext_ref = ds_diag_ext_ref.sel(time=slice('1985-01-01', '2014-12-31'))
        axes[row,col].scatter(ds_diag_ext_ref["time"], ds_diag_ext_ref[var], label="Reference data", marker=markers[v], color="red")
    # Adjust the legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=5)
    plt.tight_layout()
    plt.show()
    o_folder_data=f"/Users/magaliponds/OneDrive - Vrije Universiteit Brussel/1. VUB/02. Coding/01. IRRMIP/04. Figures/02. OGGM simulations/00. Processed Climate/{var}/"
    os.makedirs(f"{o_folder_data}/", exist_ok=True)
    o_file_name=f"{o_folder_data}/{var}.1985_2014.{timeframe}.processed.climate.data.combined.png"
    # plt.savefig(o_file_name, bbox_inches='tight')
        
#%% Plot temperatur and precipitation - processed climate input by model member, RGI
colors = {
    "W5E5": ["#000000"],#"#000000"],  # Black
    "E3SM": ["#785EF0", "#8F7BF1", "#A6A8F2"],  # Darker to lighter shades of purple
    "CESM2": ["#DC267F", "#E58A9E", "#F0A2B6", "#F7BCC4"],  # Darker to lighter shades of pink
    "CNRM": ["#FE6100", "#FE7D33", "#FE9A66", "#FEB799", "#FECDB5", "#FEF1E1"],  # Darker to lighter shades of orange
    "IPSL-CM6": ["#FFB000"]  # Dark purple to lighter shades
}

members=[3,4,6,1,1]
models = ["E3SM","CESM2","CNRM","IPSL-CM6","W5E5"]
markers=["o","x"]
for v,var in enumerate(["prcp", "temp"]):
    fig,axes=plt.subplots(3,3,figsize=(15,8))
    for r, rgi_id in enumerate(rgi_ids):
        row = r // axes.shape[1]
        col = r % axes.shape[1]        

        for m, model in enumerate(models):
            for member in range(members[m]):
                #Create a sample-id tag to list model restuls
                sample_id= f"{model}.00{member}" #p
                # Why no valid volume file?? 
                opath = os.path.join(wd_path, "per_glacier", rgi_id[:-6], rgi_id[:-3],rgi_id, 'climate_historical_{}.nc'.format(sample_id))
                # ds_diag_ext = pd.read_csv(opath)
                # ds_diag_ext = xr.Dataset.from_dataframe(ds_diag_ext)
                ds_diag_ext=xr.open_dataset(opath)
                axes[row,col].scatter(ds_diag_ext["time"], ds_diag_ext[var], label=sample_id, marker=markers[v], color=colors[model][member])
                axes[row,col].set_title(rgi_id)
                if r == 3:
                    axes[row,col].set_ylabel(ds_diag_ext[var].long_name+" ["+ds_diag_ext[var].units+"]")
                if r >= 6:
                    axes[row, col].set_xlabel("Time [year]")
                # print(ds_diag_ext["time"])


    # Adjust the legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=5)
    plt.tight_layout()
    plt.show()
    o_folder_data=f"/Users/magaliponds/OneDrive - Vrije Universiteit Brussel/1. VUB/02. Coding/01. IRRMIP/04. Figures/02. OGGM simulations/00. Processed Climate/{var}/"
    os.makedirs(f"{o_folder_data}/", exist_ok=True)
    o_file_name=f"{o_folder_data}/{var}.1985_2014.{timeframe}.processed.climate.data.combined.png"
    # plt.savefig(o_file_name, bbox_inches='tight')
        
#%% Plot simulated volume for each modelxmember combination

# Define the members and models

def plot_volume_area_evolution_over_time_combined(averaged):
    members = [3,4, 6, 1,1]
    # members=[1,1,1,1,1,]
    models = ["E3SM", "CESM2", "CNRM", "IPSL-CM6", "W5E5"]
    variables =["volume_m3","area_m2"]

    variable_names =["Volume","Area"]
    variable_axes =["Volume [m$^3$]","Area [m$^2$]"]
    
    # Define the colors for each model's members
    colors = {
        "W5E5": ["#000000"],  # Black
        "E3SM": ["#785EF0", "#8F7BF1", "#A6A8F2"],  # Darker to lighter shades of purple
        "CESM2": ["#DC267F", "#E58A9E", "#F0A2B6", "#F7BCC4"],  # Darker to lighter shades of pink
        "CNRM": ["#FE6100", "#FE7D33", "#FE9A66", "#FEB799", "#FECDB5", "#FEF1E1"],  # Darker to lighter shades of orange
        "IPSL-CM6": ["#FFB000"]  # Dark purple to lighter shades
    }
    
    # Create the figure and axes
    
    
    for v,var in enumerate(variables):
        fig, axes = plt.subplots(3,3, figsize=(15, 8))
        for r, rgi_id in enumerate(rgi_ids):
            row = r // axes.shape[1]
            col = r % axes.shape[1]
            axes[row, col].set_title(rgi_id)

        
            if r == 0 or r == 3 or r == 6:
                axes[row, col].set_ylabel(variable_axes[v])
            if r >= 6:
                axes[row, col].set_xlabel("Time [year]")
        
            for m, model in enumerate(models):
                
                for i in range(members[m]):
                    # Create a sample-id tag to list model results
                    sample_id = f"{model}.00{i}" 
                    # Define the output path

                    opath = os.path.join(wd_path, "per_glacier", rgi_id[:-6], rgi_id[:-3], rgi_id, 'model_diagnostics_{}_output_climate_run.nc'.format(sample_id))
                    
                    # Load the dataset
                    ds_diag_ext = xr.open_dataset(opath)

                    # Plot the data with the appropriate color
                    
                    
                    if m==4:
                        axes[row, col].plot(ds_diag_ext["time"], ds_diag_ext[var], label=sample_id, color=colors[model][i], linewidth=1)
                        # axes[row, col].plot(ds_diag_ext_test["time"], ds_diag_ext_test[var], label="historical reference", color=colors[model][i], linewidth=2)
                    elif i==0:
                        axes[row, col].plot(ds_diag_ext["time"], ds_diag_ext[var], label=sample_id, color=colors[model][i], linewidth=3, alpha=1)
                    elif averaged==False:
                        axes[row, col].plot(ds_diag_ext["time"], ds_diag_ext[var], label=sample_id, color=colors[model][i], linewidth=3, linestyle="dotted")
                        
            if model=="W5E5":
                wd_path_test=f'{folder_path}/07. check perturbation-glacier interactions/'
                opath_2 = os.path.join(wd_path_test, "per_glacier", rgi_id[:-6], rgi_id[:-3], rgi_id, 'model_diagnostics_spinup_historical.nc')
                ds_diag_ext_test = xr.open_dataset(opath_2)
                ds_diag_ext_test = ds_diag_ext_test.where((ds_diag_ext_test['time'] >= 1985) & (ds_diag_ext_test['time'] < 2014))
                axes[row, col].plot(ds_diag_ext_test["time"], ds_diag_ext_test[var], label="W5E5.000 pre-pro 5", color="black", linewidth=2, linestyle="dotted")
                
        # Adjust the legend
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=5)
        plt.tight_layout()
        
        if averaged==True:
            suffix="member.avg"
        else:
            suffix="all.members"
        
        o_folder_data=f"/Users/magaliponds/OneDrive - Vrije Universiteit Brussel/1. VUB/02. Coding/01. IRRMIP/04. Figures/02. OGGM simulations/01. Modelled output/0{v+1}. {variable_names[v]}/00. Combined"
        # print(o_folder_data)
        os.makedirs(f"{o_folder_data}/", exist_ok=True)
        o_file_name=f"{o_folder_data}/1985_2014.{timeframe}.delta.{variable_names[v]}.combined.{suffix}.png"
        plt.savefig(o_file_name, bbox_inches='tight')
    return
plot_volume_area_evolution_over_time_combined(True)
# plot_volume_area_evolution_over_time_combined(False)

#%% Same code but now per model

def plot_volume_area_evolution_over_time_by_model():
    
    variables =["volume_m3","area_m2"]
    variable_names =["Volume","Area"]
    variable_axes =["Volume [m$^3$]","Area [m$^2$]"]
    
    members = [3, 4, 6, 1]
    # members=[1,1,1,1,1,]
    models = ["E3SM", "CESM2", "CNRM", "IPSL-CM6"]
    
    # Define the colors for each model's members
    colors = {
        "W5E5": ["#000000"],  # Black
        "E3SM": ["#785EF0", "#8F7BF1", "#A6A8F2"],  # Darker to lighter shades of purple
        "CESM2": ["#DC267F", "#E58A9E", "#F0A2B6", "#F7BCC4"],  # Darker to lighter shades of pink
        "CNRM": ["#FE6100", "#FE7D33", "#FE9A66", "#FEB799", "#FECDB5", "#FEF1E1"],  # Darker to lighter shades of orange
        "IPSL-CM6": ["#FFB000"]  # Dark purple to lighter shades
    }
    for v,var in enumerate(variables):
        for m, model in enumerate(models):
            fig,axes = plt.subplots(3,3,figsize=(15,8))
            
            for r, rgi_id in enumerate(rgi_ids):
                row = r // axes.shape[1]
                col = r % axes.shape[1]
                axes[row, col].set_title(rgi_id)
                if r == 0 or r == 3 or r == 6:
                    axes[row, col].set_ylabel(variable_axes[v])
                if r >= 6:
                    axes[row, col].set_xlabel("Time [year]")
                #Add baseline plot
                opath = os.path.join(wd_path, "per_glacier", rgi_id[:-6], rgi_id[:-3], rgi_id, 'model_diagnostics_W5E5.000_output_climate_run.nc')
                ds_baseline = xr.open_dataset(opath)
                axes[row,col].plot(ds_baseline["time"], ds_baseline[var], label="W5E5.000", color="black", linewidth=3)
                
                for i in range(members[m]):
                    sample_id = f"{model}.00{i}" 
                    # Create a sample-id tag to list model results
                    # Plot the baseline (W5E5)
                    
                
                    # Define the output path
                    opath = os.path.join(wd_path, "per_glacier", rgi_id[:-6], rgi_id[:-3], rgi_id, f'model_diagnostics_{sample_id}_output_climate_run.nc')
                    
                    # Load the dataset
                    ds_diag_ext = xr.open_dataset(opath)
                    
                    # Plot the data with the appropriate color and style
                    
                    linestyle = 'dotted' if i > 0 else '-'
                    axes[row,col].plot(ds_diag_ext["time"], ds_diag_ext[var], label=f'{sample_id}', color=colors[model][i], linewidth=2, linestyle=linestyle)
        
                    
           
            # Adjust the legend
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.52, 0), ncol=3)
            plt.tight_layout()
                    
            o_folder_data=f"/Users/magaliponds/OneDrive - Vrije Universiteit Brussel/1. VUB/02. Coding/01. IRRMIP/04. Figures/02. OGGM simulations/01. Modelled output/0{v+1}. {variable_names[v]}/0{m+1}. {model}"
            os.makedirs(f"{o_folder_data}/", exist_ok=True)
            o_file_name=f"{o_folder_data}/1985_2014.{timeframe}.delta.{variable_names[v]}.combined.{model}.png"
            plt.savefig(o_file_name, bbox_inches='tight')
            plt.show()
            
    return

plot_volume_area_evolution_over_time_by_model()  
 

#%%  Cell 12: Plot MB comparison for geodetic vs modelle baselined 

#%% Plot simulated volume for each modelxmember combination

# Define the members and models

def plot_mass_balance_evolution_over_time_combined(averaged):
    

    members = [3, 4, 6, 1,1]
    # members=[1,1,1,1,1,]
    models = ["E3SM", "CESM2", "CNRM", "IPSL-CM6", "W5E5"]
    
    # Define the colors for each model's members
    colors = {
        "W5E5": ["#000000"],  # Black
        "E3SM": ["#785EF0", "#8F7BF1", "#A6A8F2"],  # Darker to lighter shades of purple
        "CESM2": ["#DC267F", "#E58A9E", "#F0A2B6", "#F7BCC4"],  # Darker to lighter shades of pink
        "CNRM": ["#FE6100", "#FE7D33", "#FE9A66", "#FEB799", "#FECDB5", "#FEF1E1"],  # Darker to lighter shades of orange
        "IPSL-CM6": ["#FFB000"]  # Dark purple to lighter shades
    }
    
    # Create the figure and axes
    
    fig, axes = plt.subplots(3,3, figsize=(15, 8))
    for r, rgi_id in enumerate(rgi_ids):
        row = r // axes.shape[1]
        col = r % axes.shape[1]
        axes[row, col].set_title(rgi_id)
    
        if r == 3:
            axes[row, col].set_ylabel("Specific Mass Balance [mm w.e. yr$-1$]")
        if r >= 6:
            axes[row, col].set_xlabel("Time [year]")
        
        mbdf = utils.get_geodetic_mb_dataframe()
        years = np.arange(1985, 2020) 
        mb = pd.DataFrame({'years': years})
        mb['years'] = pd.to_datetime(mb['years'], format='%Y')
        sel = mbdf.loc[rgi_id].set_index('period')*1000
       
        # Load & plot Hugonnet data for 2000-2010
        _mb, _err = sel.loc['2000-01-01_2020-01-01'][['dmdtda', 'err_dmdtda']]
        axes[row,col].fill_between([2000, 2020], [_mb-_err, _mb-_err], [_mb+_err, _mb+_err], alpha=0.5, color='gray')
        axes[row,col].plot([2000, 2020], [_mb, _mb], color='gray', label='Hugonnet 2000-2010')

        opath_test = os.path.join(sum_dir, 'fixed_geometry_mass_balance_climate_test.csv')
        ds_diag_ext_test = pd.read_csv(opath_test)
        ds_diag_ext_test = xr.Dataset.from_dataframe(ds_diag_ext_test)    
        ds_diag_ext_test = ds_diag_ext_test.rename({'Unnamed: 0': "time"})
        ds_diag_ext_test = ds_diag_ext_test.where((ds_diag_ext_test['time'] >= 1985) & (ds_diag_ext_test['time'] < 2014))

        
        axes[row, col].plot(ds_diag_ext_test["time"], ds_diag_ext_test[rgi_id], label="W5E5 OGGM pre-pro 5", color="black", linewidth=2, linestyle="dotted")

        
        for m, model in enumerate(models):
            for i in range(members[m]):
                # Create a sample-id tag to list model results
                sample_id = f"{model}.00{i}" 
                # Define the output path
                opath = os.path.join(sum_dir, 'fixed_geometry_mass_balance_{}.csv'.format(sample_id))
                opath_2 = os.path.join(sum_dir, 'specific_mb_{}.csv'.format(sample_id))

               

    
                 
                # Load the dataset
                ds_diag_ext = pd.read_csv(opath)
                ds_diag_ext = xr.Dataset.from_dataframe(ds_diag_ext)    
                ds_diag_ext = ds_diag_ext.rename({'Unnamed: 0': "time"})
                ds_diag_ext = ds_diag_ext.where((ds_diag_ext['time'] >= 1985) & (ds_diag_ext['time'] < 2014))
                
                ds_diag_ext_test = pd.read_csv(opath_2)
                ds_diag_ext_test = xr.Dataset.from_dataframe(ds_diag_ext_test)    
                ds_diag_ext_test = ds_diag_ext_test.rename({'Year': "time"})
                ds_diag_ext_test = ds_diag_ext_test.where((ds_diag_ext_test['time'] >= 1985) & (ds_diag_ext_test['time'] < 2015))
                print(ds_diag_ext_test)
                filtered_data = ds_diag_ext_test.where(ds_diag_ext_test["Glacier"] == rgi_id, drop=True)

                
                
                # print(ds_diag_ext.data_vars)
                
                mb_2010_2020 = ds_diag_ext.where((ds_diag_ext['time'] >= 2000) & (ds_diag_ext['time'] < 2014))
                mean_mb_2010_2020 = mb_2010_2020[rgi_id].mean()
                
                
                # Plot the data with the appropriate color
                
                if m==4:
                    axes[row, col].plot(ds_diag_ext["time"], ds_diag_ext[rgi_id], label=sample_id, color=colors[model][i], linewidth=2)
                    axes[row,col].plot([2000, 2020], [mean_mb_2010_2020, mean_mb_2010_2020], color=colors[model][i], linestyle='dashed', label='OGGM modelled mean 2000-2020')
                elif i==0:
                    axes[row, col].plot(ds_diag_ext["time"], ds_diag_ext[rgi_id], label=sample_id, color=colors[model][i], linewidth=3, alpha=1)
                    # axes[row,col].plot([2000, 2020], [mean_mb_2010_2020, mean_mb_2010_2020], color=colors[model][i], linestyle='dashed', label=False)
                    # axes[row, col].plot(filtered_data["time"], filtered_data["Mass Balance"], label="new mb model", color=colors[model][i], linewidth=3, alpha=1, linestyle="-.")

                elif averaged==False:
                    axes[row, col].plot(ds_diag_ext["time"], ds_diag_ext[rgi_id], label=sample_id, color=colors[model][i], linewidth=3, linestyle="dotted")
                    axes[row,col].plot([2000, 2020], [mean_mb_2010_2020, mean_mb_2010_2020], color=colors[model][i], linestyle='dashed', label=False)
                
                
                # Plot (mean) modelled data 
                
    # Adjust the legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=5)
    plt.tight_layout()
    
    if averaged==True:
        suffix="member.avg"
    else:
        suffix="all.members"
    
    o_folder_data=f"/Users/magaliponds/OneDrive - Vrije Universiteit Brussel/1. VUB/02. Coding/01. IRRMIP/04. Figures/02. OGGM simulations/01. Modelled output/03. Mass Balance/00. Combined"
    print(o_folder_data)
    os.makedirs(f"{o_folder_data}/", exist_ok=True)
    o_file_name=f"{o_folder_data}/1985_2014.{timeframe}.delta.mb.combined.{suffix}.png"
    # plt.savefig(o_file_name, bbox_inches='tight')
    return

plot_mass_balance_evolution_over_time_combined(True)
# plot_mass_balance_evolution_over_time_combined(False)

#%% Plot mass balance over time by model

def plot_mb_evolution_over_time_by_model():
    
    
    members = [3, 4, 6, 1]
    # members=[1,1,1,1,1,]
    models = ["E3SM", "CESM2", "CNRM", "IPSL-CM6"]
    
    # Define the colors for each model's members
    colors = {
        "W5E5": ["#000000"],  # Black
        "E3SM": ["#785EF0", "#8F7BF1", "#A6A8F2"],  # Darker to lighter shades of purple
        "CESM2": ["#DC267F", "#E58A9E", "#F0A2B6", "#F7BCC4"],  # Darker to lighter shades of pink
        "CNRM": ["#FE6100", "#FE7D33", "#FE9A66", "#FEB799", "#FECDB5", "#FEF1E1"],  # Darker to lighter shades of orange
        "IPSL-CM6": ["#FFB000"]  # Dark purple to lighter shades
    }
    for m, model in enumerate(models):
        # if m==0:
        fig,axes = plt.subplots(3,3,figsize=(15,8))
        
        for r, rgi_id in enumerate(rgi_ids):
            row = r // axes.shape[1]
            col = r % axes.shape[1]
            axes[row, col].set_title(rgi_id)
            if r == 0 or r == 3 or r == 6:
                axes[row, col].set_ylabel("Specific Mass Balance [mm w.e. yr$-1$]")
            if r >= 6:
                axes[row, col].set_xlabel("Time [year]")
            #Add baseline plot
      
            opath = os.path.join(sum_dir, 'fixed_geometry_mass_balance_climate_W5E5.000.csv')
            # Load the dataset
            ds_baseline = pd.read_csv(opath)
            ds_baseline = xr.Dataset.from_dataframe(ds_baseline)    
            ds_baseline = ds_baseline.rename({'Unnamed: 0': "time"})
            axes[row,col].plot(ds_baseline["time"], ds_baseline[rgi_id], label="W5E5.000", color="black", linewidth=3)
            
            for i in range(members[m]):
                sample_id = f"{model}.00{i}" 
                # Create a sample-id tag to list model results
                # Plot the baseline (W5E5)
                
            
                # Define the output path
                opath = os.path.join(sum_dir, f'fixed_geometry_mass_balance_climate_{sample_id}.csv')
                opath_base = os.path.join(sum_dir, f'fixed_geometry_mass_balance_climate_test.csv')
                
                # Load the dataset
                ds_diag_ext = pd.read_csv(opath)
                ds_diag_ext = xr.Dataset.from_dataframe(ds_diag_ext)    
                ds_diag_ext = ds_diag_ext.rename({'Unnamed: 0': "time"})
                
                # Plot the data with the appropriate color and style
                
                # linestyle = 'dotted' if i > 0 else '-'
                axes[row,col].plot(ds_diag_ext["time"], ds_diag_ext[rgi_id], label=f'{sample_id}', color=colors[model][i], linewidth=2)
                mb_2010_2020 = ds_diag_ext.where((ds_diag_ext['time'] >= 2000) & (ds_diag_ext['time'] < 2014))
                mean_mb_2010_2020 = mb_2010_2020[rgi_id].mean()
                # Plot (mean) modelled data 
                axes[row,col].plot([2000, 2020], [mean_mb_2010_2020, mean_mb_2010_2020], color=colors[model][i], linestyle='dashed', label='OGGM modelled mean 2000-2020')
               
                if i==0:
                # Load & plot Hugonnet data for 2000-2010
                    _mb, _err = sel.loc['2000-01-01_2020-01-01'][['dmdtda', 'err_dmdtda']]
                    axes[row,col].fill_between([2000, 2020], [_mb-_err, _mb-_err], [_mb+_err, _mb+_err], alpha=0.5, color='gray')
                    axes[row,col].plot([2000, 2020], [_mb, _mb], color='gray', label='Hugonnet 2000-2010')
                
               
                
       
        # Adjust the legend
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.52, 0), ncol=3)
        plt.tight_layout()
                
        o_folder_data=f"/Users/magaliponds/OneDrive - Vrije Universiteit Brussel/1. VUB/02. Coding/01. IRRMIP/04. Figures/02. OGGM simulations/01. Modelled output/03. Mass Balance/0{m+1}. {model}"
        os.makedirs(f"{o_folder_data}/", exist_ok=True)
        o_file_name=f"{o_folder_data}/1985_2014.{timeframe}.delta.mb.combined.{model}.png"
        plt.savefig(o_file_name, bbox_inches='tight')
        plt.show()
        
    return

plot_mb_evolution_over_time_by_model()  

#%% Plot mass balacne with Multiple Flowline model
# cfg.add_to_basenames('model_flowlines_dyn_melt_f_calib.pkl', 'model_flowlines.pkl', 'Adding the dynamic melt calibration flowlines')


cfg.PARAMS['store_model_geometry']=True
panels=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

mbdf = utils.get_geodetic_mb_dataframe()
years = np.arange(1985, 2020) 
mb = pd.DataFrame({'years': years})
mb['years'] = pd.to_datetime(mb['years'], format='%Y')

fig_mb, axes_mb = plt.subplots(3,3, figsize=(20, 12), sharex=True)
axes_mb = axes_mb.flatten()  # Flatten the 2D array of axes to simplify indexing

for (i, gdir) in enumerate(gdirs):
    for m,model in enumerate(models):
        
        mbmod = massbalance.MultipleFlowlineMassBalance(gdir) 
        # First check the parameters used for running the flmod model per flowline
        # for flmod in mbmod.flowline_mb_models:
            # print(gdir.rgi_id, ' - melt_f:', flmod.melt_f, ', temp bias:', f'{flmod.temp_bias:.2f}', ', prcp factor:', f'{flmod.prcp_fac:.2f}')
            
        fls = gdir.read_pickle('model_flowlines_dyn_melt_f_calib.pkl') #model_flowlines_dyn_melt_f_calib.pkl
        # fls=xr.open_dataset(os.path.join(wd_path, "per_glacier", rgi_id[:-6], rgi_id[:-3],rgi_id, "fl_diagnostics_CESM2.002_output_climate_run.nc"))
        print(fls)
        B = mbmod.get_specific_mb(fls=fls, year=years)
        mb['B'] = B
        
        # Load & plot Hugonnet data for 2000-2010
        mb_2010_2020 = mb[(mb['years'] >= '2000-01-01') & (mb['years'] < '2020-01-01')]
        mean_mb_2010_2020 = mb_2010_2020['B'].mean()
        sel = mbdf.loc[gdir.rgi_id].set_index('period') * 1000
       
        # Load & plot Hugonnet data for 2000-2010
        _mb, _err = sel.loc['2000-01-01_2020-01-01'][['dmdtda', 'err_dmdtda']]
        axes_mb[i].fill_between([2000, 2020], [_mb-_err, _mb-_err], [_mb+_err, _mb+_err], alpha=0.5, color='C0')
        axes_mb[i].plot([2000, 2020], [_mb, _mb], color='C0', label='Hugonnet 2000-2010')
        
        # Plot (mean) modelled data 
        axes_mb[i].plot([2000, 2020], [mean_mb_2010_2020, mean_mb_2010_2020], color='k', linestyle='dashed', label='OGGM modelled mean 2000-2020')
        # axes_mb[i].plot([2010, 2020], [mean_mb_2020, mean_mb_2020], color='C1', linestyle='dashed', label='OGGM modelled mean 2010-2020')
        axes_mb[i].plot(years, mb.B, c='k', label='OGGM Modelled')
        
        # Format graphs
        axes_mb[i].set_title(gdir.rgi_id, fontsize=20)
        axes_mb[i].annotate(panels[i],xy=(0.98, 0.98), xycoords='axes fraction',
                                        fontsize=24, weight='medium', ha='right', va='top')#,
                                        #bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))
        # axes_mb[i].plot(years, mb.B, c='k', label='OGGM Modelled')
        
        if i % 3 == 0:  # Only set ylabel on the left column
            axes_mb[i].set_ylabel('B [mm w.e.]', fontsize=16)
        axes_mb[i].tick_params(axis='both', which='major', labelsize=16) 


plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
plt.legend(loc='lower center', bbox_to_anchor=(-0.75, -0.5), ncols=3)
plt.show()



#%%  Cell 12: Plot MB comparison for geodetic vs modelled data 

# cfg.add_to_basenames('model_flowlines_dyn_melt_f_calib.pkl', 'model_flowlines.pkl', 'Adding the dynamic melt calibration flowlines')


cfg.PARAMS['store_model_geometry']=True
panels=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

colors = {
    "W5E5": ["#000000"],  # Black
    "E3SM": ["#785EF0", "#8F7BF1", "#A6A8F2"],  # Darker to lighter shades of purple
    "CESM2": ["#DC267F", "#E58A9E", "#F0A2B6", "#F7BCC4"],  # Darker to lighter shades of pink
    "CNRM": ["#FE6100", "#FE7D33", "#FE9A66", "#FEB799", "#FECDB5", "#FEF1E1"],  # Darker to lighter shades of orange
    "IPSL-CM6": ["#FFB000"]  # Dark purple to lighter shades
}


members=[1,3,4,6,1]
models = "W5E5","E3SM","CESM2","CNRM","IPSL-CM6"
timeframe="monthly"
download_gdirs=False
reset=False

     
fig_mb, axes_mb = plt.subplots(3,3, figsize=(20, 12), sharex=True)
axes_mb = axes_mb.flatten()  # Flatten the 2D array of axes to simplify indexing

for (i, gdir) in enumerate(gdirs):
    # Load & plot Hugonnet data for 2000-2010
    _mb, _err = sel.loc['2000-01-01_2020-01-01'][['dmdtda', 'err_dmdtda']]
    axes_mb[i].fill_between([2000, 2020], [_mb-_err, _mb-_err], [_mb+_err, _mb+_err], alpha=0.5, color='C0')
    axes_mb[i].plot([2000, 2020], [_mb, _mb], color='C0', label='Hugonnet 2000-2010')

    for m, model in enumerate(models):
        for member in range(members[m]):
            #Create a sample-id tag to list model restuls
            sample_id= f"{model}.00{i}" #provide label for all run output
            
            #initiate the model with custom climate data (model-member combination)
            initiate_OGGM(timeframe, model, member, reset, wd_path)   
         
            
            mbdf = utils.get_geodetic_mb_dataframe()
            years = np.arange(1985, 2020) 
            mb = pd.DataFrame({'years': years})
            mb['years'] = pd.to_datetime(mb['years'], format='%Y')
            fls = gdir.read_pickle('model_flowlines_dyn_melt_f_calib.pkl') #model_flowlines_dyn_melt_f_calib.pkl
            # Load & plot Hugonnet data for 2000-2010
            
            sel = mbdf.loc[gdir.rgi_id].set_index('period') * 1000
               
            
            
             
                
            mbmod = massbalance.MultipleFlowlineMassBalance(gdir) 
            # First check the parameters used for running the flmod model per flowline
            for flmod in mbmod.flowline_mb_models:
                print(gdir.rgi_id, ' - melt_f:', flmod.melt_f, ', temp bias:', f'{flmod.temp_bias:.2f}', ', prcp factor:', f'{flmod.prcp_fac:.2f}')
                
            B = mbmod.get_specific_mb(fls=fls, year=years)
            mb['B'] = B
            mb_2000_2020 = mb[(mb['years'] >= '2000-01-01') & (mb['years'] < '2020-01-01')]
            mean_mb_2000_2020 = mb_2010_2020['B'].mean()
            
            # Plot (mean) modelled data 
            axes_mb[i].plot([2000, 2020], [mean_mb_2000_2020, mean_mb_2000_2020], color=colors[model][member], linestyle='dashed', label='OGGM modelled mean 2000-2020')
            # axes_mb[i].plot([2010, 2020], [mean_mb_2020, mean_mb_2020], color='C1', linestyle='dashed', label='OGGM modelled mean 2010-2020')
            axes_mb[i].plot(years, mb.B, c=colors[model][member], label='OGGM Modelled')
                    
            # Format graphs
            axes_mb[i].set_title(gdir.rgi_id, fontsize=20)
            axes_mb[i].annotate(panels[i],xy=(0.98, 0.98), xycoords='axes fraction',
                                            fontsize=24, weight='medium', ha='right', va='top')#,
                                            #bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))
            # axes_mb[i].plot(years, mb.B, c='k', label='OGGM Modelled')
            
            if i % 3 == 0:  # Only set ylabel on the left column
                axes_mb[i].set_ylabel('B [mm w.e.]', fontsize=16)
            axes_mb[i].tick_params(axis='both', which='major', labelsize=16) 
            
            
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
    plt.legend(loc='lower center', bbox_to_anchor=(-0.75, -0.5), ncols=3)



