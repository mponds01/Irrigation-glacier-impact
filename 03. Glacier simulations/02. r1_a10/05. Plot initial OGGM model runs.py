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

#%%

ds = xr.open_dataset("/Users/magaliponds/Downloads/orog_W5E5v2.0.nc")
ds.orog.attrs

#%% Provide modelling parameters

folder_path='/Users/magaliponds/Documents/00. Programming'
wd_path=f'{folder_path}/06. Modelled perturbation-glacier interactions/'
wd_path_test=f'{folder_path}/07. check perturbation-glacier interactions/'

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


#%% Plot temperatur and precipitation - against prepro-5

  
colors = {
    "W5E5": ["#000000"],#"#000000"],  # Black
    "E3SM": ["#785EF0", "#8F7BF1", "#A6A8F2"],  # Darker to lighter shades of purple
    "CESM2": ["#DC267F", "#E58A9E", "#F0A2B6", "#F7BCC4"],  # Darker to lighter shades of pink
    "CNRM": ["#FE6100", "#FE7D33", "#FE9A66", "#FEB799", "#FECDB5", "#FEF1E1"],  # Darker to lighter shades of orange
    "IPSL-CM6": ["#FFB000"]  # Dark purple to lighter shades
}

# members=[3,4,6,1,1]
members=[1,1,1,1,1]

models = ["E3SM","CESM2","CNRM","IPSL-CM6","W5E5"]
markers=["o","x"]
for v,var in enumerate(["prcp", "temp"]):
    fig,axes=plt.subplots(3,3,figsize=(22,8))
    

    for r, rgi_id in enumerate(rgi_ids):
        row = r // axes.shape[1]
        col = r % axes.shape[1]    
        opath_base = os.path.join(wd_path, "per_glacier", rgi_id[:-6], rgi_id[:-3],rgi_id, 'climate_historical_W5E5.000_ext.nc')
        ds_diag_base=xr.open_dataset(opath_base)
        
        opath_prepro = os.path.join(wd_path_test, "per_glacier", rgi_id[:-6], rgi_id[:-3],rgi_id, 'climate_historical.nc')
        ds_diag_prepro=xr.open_dataset(opath_prepro)


        
        ds_diag_dif=ds_diag_prepro-ds_diag_base
        axes[row,col].scatter(ds_diag_prepro["time"], ds_diag_prepro[var], label='W5E5 prepro L5',  color='grey', linestyle='dashed')#marker=markers[v],
        axes[row,col].scatter(ds_diag_base["time"], ds_diag_base[var], label='W5E5 prepro 3',  color='darkgrey', linestyle='dashed')#marker=markers[v],
        axes[row,col].set_title(rgi_id)
        if r == 3:
            axes[row,col].set_ylabel(ds_diag_base[var].long_name+" ["+ds_diag_base[var].units+"]")
        if r >= 6:
            axes[row, col].set_xlabel("Time [year]")
        # print(ds_diag_ext["time"])
        
        # Create secondary y-axis
        axes2 = axes[row,col].twinx()
        axes2.plot(ds_diag_dif["time"], ds_diag_dif[var], label='Difference L5-L3',  color='red')#, linestyle='dashed')#marker=markers[v],
        axes2.set_ylabel('Difference L3-L5')
        axes2.set_ylim(-1,1)
                    


    # Adjust the legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles2, labels2 = axes2.get_legend_handles_labels()
    handles += handles2
    labels += labels2
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=5)
    plt.tight_layout()
    plt.show()
    o_folder_data=f"/Users/magaliponds/OneDrive - Vrije Universiteit Brussel/1. VUB/02. Coding/01. IRRMIP/04. Figures/02. OGGM simulations/00. Processed Climate/{var}/"
    os.makedirs(f"{o_folder_data}/", exist_ok=True)
    o_file_name=f"{o_folder_data}/{var}.1985_2014.{timeframe}.processed.climate.data.prepro.difference.png"
    plt.savefig(o_file_name, bbox_inches='tight')
    


#%% Plot temperatur and precipitation - against prepro-5
colors = {
    "W5E5": ["#000000"],#"#000000"],  # Black
    "E3SM": ["#785EF0", "#8F7BF1", "#A6A8F2"],  # Darker to lighter shades of purple
    "CESM2": ["#DC267F", "#E58A9E", "#F0A2B6", "#F7BCC4"],  # Darker to lighter shades of pink
    "CNRM": ["#FE6100", "#FE7D33", "#FE9A66", "#FEB799", "#FECDB5", "#FEF1E1"],  # Darker to lighter shades of orange
    "IPSL-CM6": ["#FFB000"]  # Dark purple to lighter shades
}

# members=[3,4,6,1,1]
members=[1,1,1,1,1]

models = ["E3SM","CESM2","CNRM","IPSL-CM6","W5E5"]
markers=["o","x"]
colors=['lightblue', 'pink']
linecolors=['blue', 'red']

for v,var in enumerate(["prcp", "temp"]):
    fig,axes=plt.subplots(3,3,figsize=(15,15))
    # fig_L3,axes_L3=plt.subplots(3,3,figsize=(15,15))
    

    for r, rgi_id in enumerate(rgi_ids):
        row = r // axes.shape[1]
        col = r % axes.shape[1]    
        opath_base_custom = os.path.join(wd_path, "per_glacier", rgi_id[:-6], rgi_id[:-3],rgi_id, 'climate_historical_W5E5.000_ext.nc')
        opath_base_L3 = os.path.join(wd_path, "per_glacier", rgi_id[:-6], rgi_id[:-3],rgi_id, 'climate_historicalL3.nc')
        ds_diag_custom=xr.open_dataset(opath_base_custom)
        ds_diag_L3=xr.open_dataset(opath_base_L3)

        
        opath_prepro = os.path.join(wd_path_test, "per_glacier", rgi_id[:-6], rgi_id[:-3],rgi_id, 'climate_historical.nc')
        ds_diag_prepro=xr.open_dataset(opath_prepro)
        ds_diag_prepro_L3=ds_diag_prepro.where(ds_diag_prepro.time >= np.datetime64('1979'), drop=True)

        # x_smooth_L3 = np.linspace(ds_diag_prepro_L3[var].min(), ds_diag_prepro_L3[var].max(), 500)

        # coefficients_L3 = np.polyfit(ds_diag_prepro_L3[var], ds_diag_L3[var], 1)
        # trendline_L3 = np.poly1d(coefficients_L3)
        
        # axes_L3[row,col].scatter( ds_diag_prepro_L3[var], ds_diag_L3[var], color='pink', marker='+', label="Monthly L3-L5 data")#, linestyle='dashed')#marker=markers[v],
        # axes_L3[row,col].plot( x_smooth_L3, trendline_L3(x_smooth_L3),  color='red', linestyle='dashed', label="1:1 prepro line")#marker=markers[v],
        # axes_L3[row,col].plot( ds_diag_prepro_L3[var], ds_diag_prepro_L3[var],  color='black', linestyle='dashed', label="1:1 prepro line")#marker=markers[v],
        # axes_L3[row,col].set_title(rgi_id)
        # axes_L3[row,col].axis('equal')

        
        # if r in [3]:
        #     axes_L3[row,col].set_ylabel(f"{ds_diag_L3[var].long_name} in {ds_diag_L3[var].units} - prepro L3")
        # if r >= 6:
        #     axes_L3[row, col].set_xlabel(f"{ds_diag_prepro[var].long_name} in {ds_diag_prepro[var].units} - prepro L5")
        # # print(ds_diag_ext["time"])
        
        
        ds_diag_prepro= ds_diag_prepro.where(ds_diag_prepro.time< np.datetime64('2015'), drop=True)
        
        # Fit a linear trend line (1st-degree polynomial)
        coefficients = np.polyfit(ds_diag_prepro[var], ds_diag_custom[var], 1)
        trendline = np.poly1d(coefficients)
        x_smooth = np.linspace(ds_diag_prepro[var].min(), ds_diag_prepro[var].max(), 500)

        
        axes[row,col].scatter( ds_diag_prepro[var], ds_diag_custom[var], color=colors[v], marker='+', label="Monthly L5-custom data")#, linestyle='dashed')#marker=markers[v],
        axes[row,col].plot( ds_diag_prepro[var], ds_diag_prepro[var],  color='black', linestyle='dashed', label="1:1 prepro line")#marker=markers[v],
        axes[row,col].plot( x_smooth, trendline(x_smooth),  color=linecolors[v], linestyle='dashed', label='Fitted data line')#marker=markers[v],
        axes[row,col].set_title(rgi_id)
        axes[row,col].axis('equal')

        
        if r in [3]:
            axes[row,col].set_ylabel(f"{ds_diag_L3[var].long_name} in {ds_diag_L3[var].units} - custom processing")
        if r >= 6:
            axes[row, col].set_xlabel(f"{ds_diag_prepro[var].long_name} in {ds_diag_prepro[var].units} - prepro L5")
            
        # print(ds_diag_ext["time"])
        
        
    

    # Adjust the legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=5)

    plt.tight_layout()
    plt.show()
    
    # handles2, labels2 = axes_L3[0, 0].get_legend_handles_labels()
    # fig_L3.legend(handles2, labels2, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=5)

    plt.tight_layout()
    plt.show()
    
    # o_folder_data=f"/Users/magaliponds/OneDrive - Vrije Universiteit Brussel/1. VUB/02. Coding/01. IRRMIP/04. Figures/02. OGGM simulations/00. Processed Climate/{var}/"
    # os.makedirs(f"{o_folder_data}/", exist_ok=True)
    # o_file_name=f"{o_folder_data}/{var}.1985_2014.{timeframe}.processed.climate.data.prepro.compared.png"
    # plt.savefig(o_file_name, bbox_inches='tight')
    
  
    
#%% Plot volume and area for each sample_id

def plot_volume_area_evolution_over_time_combined(averaged):
    timeframe ="monthly"
    members = [3, 4, 6, 1, 1]
    models = ["E3SM", "CESM2", "CNRM", "IPSL-CM6", "W5E5"]
    variables = [ "volume", "area"]#"volume",
    factors = [10**-9, 10**-6]

    variable_names = ["Volume", "Area"]
    variable_axes = ["Volume [km$^3$]", "Area [km$^2$]"]

    colors = {
        "W5E5": ["#000000"],
        
        "E3SM": ["#785EF0", "#8F7BF1", "#A6A8F2"],
        "CESM2": ["#DC267F", "#E58A9E", "#F0A2B6", "#F7BCC4"],
        "CNRM": ["#FE6100", "#FE7D33", "#FE9A66", "#FEB799", "#FECDB5", "#FEF1E1"],
        "IPSL-CM6": ["#FFB000"]
    }

    for v, var in enumerate(variables):
        fig, axes = plt.subplots(3, 3, figsize=(15, 8))
        axes = axes.flatten()  # Flatten the axes array to simplify indexing

        for m, model in enumerate(models):
            if averaged==False:
                members=members
            else:
                members=[1,1,1,1,1]
                
            for i in range(members[m]):
                sample_id = f"{model}.00{i}"
                opath = os.path.join(sum_dir, f'climate_run_output_{sample_id}.nc')
                ds_diag_ext = xr.open_dataset(opath)
                
                for r, rgi_id in enumerate(rgi_ids):
                    ax = axes[r]  # Select the correct subplot for the current rgi_id
                    ax.set_title(rgi_id)

                    ds_filtered = ds_diag_ext.where(ds_diag_ext.rgi_id == rgi_id, drop=True)

                    if r == 0 or r == 3 or r == 6:
                        ax.set_ylabel(variable_axes[v])
                    if r >= 6:
                        ax.set_xlabel("Time [year]")

                    if m == 4:
                        ax.plot(ds_filtered["time"], ds_filtered[var] * factors[v], label=sample_id, color=colors[model][i], linewidth=1)
                    elif i == 0:
                        ax.plot(ds_filtered["time"], ds_filtered[var] * factors[v], label=sample_id, color=colors[model][i], linewidth=3, alpha=1)
                    elif averaged==False:
                        ax.plot(ds_filtered["time"], ds_filtered[var] * factors[v], label=sample_id, color=colors[model][i], linewidth=3, linestyle="dotted")
                    

                    if model == "W5E5":
                        wd_path_test = f'{folder_path}/07. check perturbation-glacier interactions/'
                        opath_2 = os.path.join(wd_path_test, "summary", "spinup_historical_control_run.nc") #Plot the file from preprocessing level 5
                        ds_diag_ext_test = xr.open_dataset(opath_2)
                        ds_diag_ext_test = ds_diag_ext_test.where((ds_diag_ext_test['time'] >= 1985) & (ds_diag_ext_test['time'] < 2014))
                        ds_diag_ext_test = ds_diag_ext_test.where(ds_diag_ext_test.rgi_id == rgi_id, drop=True)
                        ax.plot(ds_diag_ext_test["time"], ds_diag_ext_test[var] * factors[v], label="W5E5.000 pre-pro 5", color="black", linewidth=2, linestyle="dashed")
                        
                    
                        # opath_3 = os.path.join(wd_path, "summary", "spinup_historical_run_output_before_W5E5.000.nc")
                        # ds_diag_ext_test2 = xr.open_dataset(opath_3)
                        # ds_diag_ext_test2 = ds_diag_ext_test2.where((ds_diag_ext_test2['time'] >= 1985) & (ds_diag_ext_test2['time'] < 2014))
                        # ds_diag_ext_test2 = ds_diag_ext_test2.where(ds_diag_ext_test2.rgi_id == rgi_id, drop=True)
                        # ax.plot(ds_diag_ext_test2["time"], ds_diag_ext_test2[var] * factors[v], label="W5E5.000 pre-pro 3 before dyn spinup", color="blue", linewidth=2, linestyle="dotted")
                        
                        opath_4 = os.path.join(wd_path, "summary", "spinup_historical_run_output_after_W5E5.000.nc")
                        ds_diag_ext_test4 = xr.open_dataset(opath_4)
                        ds_diag_ext_test4 = ds_diag_ext_test4.where((ds_diag_ext_test4['time'] >= 1985) & (ds_diag_ext_test4['time'] < 2014))
                        ds_diag_ext_test4 = ds_diag_ext_test4.where(ds_diag_ext_test4.rgi_id == rgi_id, drop=True)
                        ax.plot(ds_diag_ext_test4["time"], ds_diag_ext_test4[var] * factors[v], label="W5E5.000 pre-pro 3 after dyn spinup", color="red", linestyle=":")

            axes[r].set_title(rgi_id)

        # Adjust the legend
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=5)
        plt.tight_layout()

        suffix = "member.avg" if averaged else "all.members"
        o_folder_data = f"/Users/magaliponds/OneDrive - Vrije Universiteit Brussel/1. VUB/02. Coding/01. IRRMIP/04. Figures/02. OGGM simulations/01. Modelled output/0{v + 1}. {variable_names[v]}/00. Combined"
        os.makedirs(o_folder_data, exist_ok=True)
        o_file_name = f"{o_folder_data}/1985_2014.{timeframe}.delta.{variable_names[v]}.combined.{suffix}.png"
        plt.savefig(o_file_name, bbox_inches='tight')

    return

# Call the function with appropriate parameters
plot_volume_area_evolution_over_time_combined(True)

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
 

#%%  Cell 12: Test

for r, rgi_id in enumerate(rgi_ids):
    if r==0:
        opath = os.path.join(wd_path, "per_glacier", rgi_id[:-6], rgi_id[:-3], rgi_id, 'model_geometry_dynamic_spinup.nc')
        ds= xr.open_dataset(opath)
        print(ds.time.values)


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

        opath_test = os.path.join(wd_path_test, 'summary', 'spinup_historical_mb_control_run.csv')
        
        ds_diag_ext_test = pd.read_csv(opath_test)
        ds_diag_ext_test = xr.Dataset.from_dataframe(ds_diag_ext_test)    
        ds_diag_ext_test = ds_diag_ext_test.rename({'Unnamed: 0': "time"})
        ds_diag_ext_test = ds_diag_ext_test.where((ds_diag_ext_test['time'] >= 1985) & (ds_diag_ext_test['time'] < 2014))
        axes[row, col].plot(ds_diag_ext_test["time"], ds_diag_ext_test[rgi_id], label="W5E5 OGGM pre-pro 5", color="black", linewidth=2, linestyle="dotted")

        opath_test2 = os.path.join(wd_path, 'summary', 'historical_run_massbalance_before_output.csv')

        ds_diag_ext_test2 = pd.read_csv(opath_test2)
        ds_diag_ext_test2 = xr.Dataset.from_dataframe(ds_diag_ext_test2)    
        ds_diag_ext_test2 = ds_diag_ext_test2.rename({'Unnamed: 0': "time"})
        ds_diag_ext_test2 = ds_diag_ext_test2.where((ds_diag_ext_test2['time'] >= 1985) & (ds_diag_ext_test2['time'] < 2014))
        axes[row, col].plot(ds_diag_ext_test2["time"], ds_diag_ext_test2[rgi_id], label="before spinup", color="green", linewidth=2, linestyle="dotted")
        print(ds_diag_ext_test2)
        
        opath_test3 = os.path.join(wd_path, 'summary', 'historical_run_massbalance_output.csv')

        ds_diag_ext_test3 = pd.read_csv(opath_test2)
        ds_diag_ext_test3 = xr.Dataset.from_dataframe(ds_diag_ext_test3)    
        ds_diag_ext_test3 = ds_diag_ext_test3.rename({'Unnamed: 0': "time"})
        ds_diag_ext_test3 = ds_diag_ext_test3.where((ds_diag_ext_test3['time'] >= 1985) & (ds_diag_ext_test3['time'] < 2014))
        axes[row, col].plot(ds_diag_ext_test3["time"], ds_diag_ext_test3[rgi_id], label="after spinup", color="blue", linewidth=2, linestyle="dotted")
        print(ds_diag_ext_test2)
        
        

        
        # for m, model in enumerate(models):
        #     if averaged==True:
        #         members=[1,1,1,1,1]
            
        #     for i in range(members[m]):
        #         # Create a sample-id tag to list model results
        #         sample_id = f"{model}.00{i}" 
        #         # Define the output path
        #         opath = os.path.join(sum_dir, 'compiled_mass_balance_output_{}.csv'.format(sample_id))

              
        #         # Load the dataset
        #         ds_diag_ext = pd.read_csv(opath)
        #         ds_diag_ext = xr.Dataset.from_dataframe(ds_diag_ext)    
        #         ds_diag_ext = ds_diag_ext.rename({'Unnamed: 0': "time"})
        #         ds_diag_ext = ds_diag_ext.where((ds_diag_ext['time'] >= 1985) & (ds_diag_ext['time'] < 2014))
                
        #         # print(ds_diag_ext.data_vars)
                
        #         mb_2010_2020 = ds_diag_ext.where((ds_diag_ext['time'] >= 2000) & (ds_diag_ext['time'] < 2014))
        #         mean_mb_2010_2020 = mb_2010_2020[rgi_id].mean()
                
                
        #         # Plot the data with the appropriate color
                
        #         if m==4:
        #             axes[row, col].plot(ds_diag_ext["time"], ds_diag_ext[rgi_id], label=sample_id, color=colors[model][i], linewidth=2)
        #             axes[row,col].plot([2000, 2020], [mean_mb_2010_2020, mean_mb_2010_2020], color=colors[model][i], linestyle='dashed', label='OGGM modelled mean 2000-2020')
        #         elif i==0:
        #             axes[row, col].plot(ds_diag_ext["time"], ds_diag_ext[rgi_id], label=sample_id, color=colors[model][i], linewidth=3, alpha=1)
        #             # axes[row,col].plot([2000, 2020], [mean_mb_2010_2020, mean_mb_2010_2020], color=colors[model][i], linestyle='dashed', label=False)
        #             # axes[row, col].plot(filtered_data["time"], filtered_data["Mass Balance"], label="new mb model", color=colors[model][i], linewidth=3, alpha=1, linestyle="-.")

        #         elif averaged==False:
        #             axes[row, col].plot(ds_diag_ext["time"], ds_diag_ext[rgi_id], label=sample_id, color=colors[model][i], linewidth=3, linestyle="dotted")
        #             axes[row,col].plot([2000, 2020], [mean_mb_2010_2020, mean_mb_2010_2020], color=colors[model][i], linestyle='dashed', label=False)
                
                
        #         # Plot (mean) modelled data 
                
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



