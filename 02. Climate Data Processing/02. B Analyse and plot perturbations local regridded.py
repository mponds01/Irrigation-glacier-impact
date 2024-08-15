#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 14:14:34 2024

@author: magaliponds
"""

import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import geopandas as gpd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

"""
This script performs an initial data screening, in 3 sections:
    1. checks what the monthly irrigation, controlrun and difference in P look like, plotted in a geoplot
    2. Includes the shapefile plot for the Karakoram Area
"""
    

""" 0 . Input files """


def P_T_overview_plot(model,var, timeframe, mode, diftype, inputplot, plotsave):
    
    if mode == 'dif':
        mode_suff = 'total'
    if mode == 'std':
        mode_suff = 'std'

    
    folder=f"/Users/magaliponds/OneDrive - Vrije Universiteit Brussel/1. VUB/02. Coding/01. IRRMIP/02. Data/01. Input files/{model}/01. Regridded Data/"
        
    ifile_IRR=f"REGRID.{model}.IRR.000.1985_2014_selparam_monthly_{mode_suff}.nc"
    ifile_NOI=f"REGRID.{model}.NOI.000.1985_2014_selparam_monthly_{mode_suff}.nc"
    
    
    

    # shapefile_path = '/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/01. Input files/Shapefile/glac_inv/glac_inv_all_lowres.shp'
    
    """ Part 0 - Create variability for differnt timescales in: amount of subplots, time-averaging used """
    #adjust figure sizes towards type of plot
    if timeframe=='monthly': 
        figsize=(16, 9)
        fig, axes = plt.subplots(nrows=3, ncols=4, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=figsize)
        time_signature = 'ymon'
        timestamps =['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN','JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'] 
        time_averaging = 'time.month'
        time_type='month'
        col_wrap = 4
    if timeframe =='seasonal':
        figsize=(9, 7)
        fig, axes = plt.subplots(nrows=2, ncols=2, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=figsize)
        time_signature ='yseas'
        timestamps =['DJF','MAM','JJA','SON']
        time_averaging = 'time.season'
        time_type='season'
        col_wrap = 2
    if timeframe =='annual':
        figsize=(7, 5)
        fig, axes = plt.subplots(nrows=1, ncols=1, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=figsize)
        time_signature ='year'
        timestamps=['YEAR']
        time_averaging = 'time.year'
        time_type='year'
        col_wrap = 1
    
    """Part 1 - load and correct the data"""
    
    #first for each variable load the data, for precipitation this consists of Snow & Rain from atmosphere, converted from [mm/day]
    ifile_IRR =xr.open_dataset(folder+ifile_IRR)
    ifile_NOI =xr.open_dataset(folder+ifile_NOI)
   
    #only for model IPSL-CM6 change dimensions of time (rename), as not possible in bash
    if model== "IPSL-CM6":
        ifile_IRR = ifile_IRR.rename({'time_counter': 'time'})
        ifile_NOI = ifile_NOI.rename({'time_counter': 'time'})
   

        
    if var=="Precipitation":

        baseline_R = ifile_NOI.pr*86400
        irrigation_R = ifile_IRR.pr*86400

        if model in ["CESM2", "E3SM"]:
            baseline_S = ifile_NOI.sn*86400
            baseline=baseline_R+baseline_S
            irrigation_S =ifile_IRR.sn*86400
            irrigation=irrigation_R+irrigation_S
            #for CESM3 precipitation variables are average --> to do: calculate monthly totals
            # if model =="CESM2":
            irrigation=irrigation*30
            baseline=baseline*30
        else:
            baseline=baseline_R
            irrigation=irrigation_R
        
        
    elif var =="Temperature":
        baseline=ifile_NOI.tas
        irrigation=ifile_IRR.tas
        
    
    baseline = baseline.where((baseline.lon >= 60) & (baseline.lon <= 109) & (baseline.lat >= 22) & (baseline.lat <= 52), drop=True) 
    baseline["time"] = baseline.time.astype("datetime64[ns]")
    # baseline = baseline.groupby(time_averaging).mean()
    
    irrigation = irrigation.where((irrigation.lon >= 60) & (irrigation.lon <= 109) & (irrigation.lat >= 22) & (irrigation.lat <= 52), drop=True) 
    irrigation["time"] = irrigation.time.astype("datetime64[ns]")
    # irrigation = irrigation.groupby(time_averaging).mean()
    
    # if timeframe=='annual':
    #     baseline = baseline.mean(dim='year')
    #     irrigation = irrigation.mean(dim='year')
    
    if var =="Temperature":
        if timeframe=='annual':
            baseline = baseline.mean(dim='time')
            irrigation = irrigation.mean(dim='time')
        else:
            baseline = baseline.groupby(time_averaging).mean(dim='time')
            irrigation = irrigation.groupby(time_averaging).mean(dim='time')
        
    if var =="Precipitation":
        
        if timeframe=='seasonal':
            irrigation = irrigation.resample(time='QS-DEC').sum(dim='time')
            baseline = baseline.resample(time='QS-DEC').sum(dim='time')

        if timeframe=='annual':
            irrigation = irrigation.groupby(time_averaging).sum(dim='time')
            irrigation=irrigation.mean(dim='year')
            baseline = baseline.groupby(time_averaging).sum(dim='time')
            baseline = baseline.mean(dim='year')
        else:
            irrigation=irrigation.groupby(time_averaging).mean(dim='time')
            baseline=baseline.groupby(time_averaging).mean(dim='time')
    
    if mode =='std':
        diff = (irrigation+baseline)/2
    if var =="Precipitation":
        if diftype=='rel':
            diff = ((irrigation-baseline)/baseline *100) 
    if var =="Temperature" or diftype=='abs':
        diff = irrigation-baseline 
        
    diff = diff.where((diff.lon >= 60) & (diff.lon <= 109) & (diff.lat >= 22) & (diff.lat <= 52), drop=True)

    #create local minima/maxima, for axis of plot
    local_min_diff = diff.quantile(0.25)
    local_max_diff = diff.quantile(0.75)
    
    #Provide cbar ranges and colors for plots for different variables, modes (dif/std) and difference types (abs/rel)    
    if var=="Precipitation":
        if mode =='dif' and diftype=='rel':
            vmin=-40
            vmax=40
            zero_scaled = (abs(vmin)/(abs(vmin)+abs(vmax)))
            colors = [(0, 'xkcd:mocha'),(zero_scaled, 'xkcd:white'),(1, 'xkcd:aquamarine')]
            custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors) 
        if mode =='dif' and diftype=='abs':
            vmin=-50
            vmax=75
            zero_scaled = (abs(vmin)/(abs(vmin)+abs(vmax)))
            colors = [(0, 'xkcd:mocha'),(zero_scaled, 'xkcd:white'),(1, 'xkcd:aquamarine')]
            custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors) 
        if mode =='std':
            vmin=-40
            vmax=40
            zero_scaled = (abs(vmin)/(abs(vmin)+abs(vmax)))
            colors = [(0, 'xkcd:peach'),(zero_scaled, 'xkcd:white'),(1, 'xkcd:light aquamarine')]
            custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors) 
        
    elif var =="Temperature":
        if mode =='dif': 
            vmin=-1.5
            vmax=1.5
            zero_scaled = (abs(vmin)/(abs(vmin)+abs(vmax)))
            colors = [(0, 'cornflowerblue'),(zero_scaled, 'xkcd:white'),(1, 'xkcd:tomato')]
            custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
        if mode =='std':
            vmin=-1.5
            vmax=1.5
            zero_scaled = (abs(vmin)/(abs(vmin)+abs(vmax)))
            colors = [(0, 'xkcd:lightblue'),(zero_scaled, 'xkcd:white'),(1, 'xkcd:pink')]
            custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors) 
        


    """ Part 2 - Shapefile outline for Karakoram Area to be included"""
    #path to alternative shapefile
    # shapefile_path = '/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/01. Input files/Shapefile/glac_inv/glac_inv_all_lowres.shp'
    
    # shapefile_path = '/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/01. Input files/Shapefile/Liu_2022/Pan_Tibetan_shape.shp'
    # shapefile_path = '/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/01. Input files/Shapefile/QGIS/Karakoram_Study_Area.shp'
    shapefile_path ='/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/01. Input files/Shapefile/Karakoram/Pan-Tibetan Highlands/Pan-Tibetan Highlands (Liu et al._2022)/Shapefile/Pan-Tibetan Highlands (Liu et al._2022)_P.shp'
    shp = gpd.read_file(shapefile_path)
    target_crs='EPSG:4326'
    shp = shp.to_crs(target_crs)
    
    
    """ Part 3 - Create subplots for IRR, NOI and DIF"""

           
    for time_idx, timestamp_name in enumerate(timestamps):
       
       #indicate the column and row of the subplot to plot in
       if timeframe=='monthly': 
           row = (time_idx) // 4  # Calculate row index
           col = (time_idx) % 4
           ax = axes[row, col]
       if timeframe =='seasonal':
           row = (time_idx) // 2
           col = (time_idx) % 2
           ax = axes[row, col]
       if timeframe =='annual':
           row  = 0
           col = 0
           ax = axes
           
       """ 3A Plotting data, incl karakoram outline """
       #select relevant month/season and only 1 year for annual to plot and annotate
       if timeframe=='annual':
           diff_sel = diff
       else:
           diff_sel = diff.isel({diff.dims[0]: time_idx})
      
       #plot the data incl the outline of the karakoram shapefile, setting the colors, but excluding the shapefile

       im = diff_sel.plot(ax=ax, vmin=vmin, vmax=vmax, extend='both', transform=ccrs.PlateCarree(), cmap=custom_cmap, add_colorbar=False)
       shp.plot(ax=ax, edgecolor='black', linewidth=1, facecolor='none')
       # shp.plot(ax=ax, edgecolor='red', linewidth=1, facecolor='none')

       ax.coastlines(resolution='10m')  
       
       
       #include month as a label, instead as on top off data
       ax.set_title('')
       ax.annotate(timestamp_name, xy=(1, 1), xytext=(-10, -10), xycoords='axes fraction', textcoords='offset points',
                 ha='right', va='top', fontsize=15, bbox=dict(boxstyle='square', fc='white', alpha=1))
       
       
       """ 3B - Min and Max value annotation"""
       #Include annotation for min and max values in every subplot, excluding NaN from min/max creation
       diff_sel_min = diff_sel.fillna(diff_sel.max())
       diff_sel_max = diff_sel.fillna(diff_sel.min())
       
       #find min and max values in gridcell 
       min_value_index = np.unravel_index(np.argmin(diff_sel_min.values), diff_sel_min.shape)
       max_value_index = np.unravel_index(np.argmax(diff_sel_max.values), diff_sel_max.shape)
       
       # Extract longitude and latitude corresponding to the minimum and maximum value indices
       min_lon, min_lat = diff_sel.lon.values[min_value_index[1]], diff_sel.lat.values[min_value_index[0]]
       max_lon, max_lat = diff_sel.lon.values[max_value_index[1]], diff_sel.lat.values[max_value_index[0]]
       
       # Plot the dot on the subplot
       ax.plot(min_lon, min_lat, marker='o', markersize=7, color='blue')  # Adjust marker properties as needed
       ax.plot(max_lon, max_lat, marker='o', markersize=7, color='red')  # Adjust marker properties as needed

       # #indicate annotations for min and max values in plot, formatted as percenteges for rel precipitation differences
       min_value = diff_sel[min_value_index]
       max_value = diff_sel[max_value_index]
       
       if timeframe =='annual':
           ax.annotate(f'Min: {min_value:.1f}',xy=(65,50), xytext=(65,50),fontsize=15, ha='left', va='top')
           ax.annotate(f'Max: {max_value:.1f}', xy=(78,50), xytext=(78,50),fontsize=15, ha='left', va='top')
         
           ax.plot(64,49.4, marker='o', color='blue', markersize=5)
           ax.plot(77,49.4, marker='o', color='red', markersize=5)
       else:
           ax.annotate(f'Min: {min_value:.1f}',xy=(64,50), xytext=(64,50),fontsize=15, ha='left', va='top')
           ax.annotate(f'Max: {max_value:.1f}', xy=(84,50), xytext=(84,50),fontsize=15, ha='left', va='top')
         
           ax.plot(62,49, marker='o', color='blue', markersize=5)
           ax.plot(82,49, marker='o', color='red', markersize=5)
           
           
       # Set the map gridlines
       gl = ax.gridlines(draw_labels=True)
       gl.top_labels = False
       gl.right_labels = False
       
       # Set x-ticks using latitude values
       if col == 0:      
           gl.ylabel_style = {'size': 15} 
           gl.ylocator = plt.MaxNLocator(nbins=3)

       else:
           gl.left_labels = False
       
       if (timeframe=='monthly' and row==2) or (timeframe=='seasonal' and row==1) or (timeframe=='annual'):
           gl.xlabel_style = {'size': 15} 
           gl.xlocator = plt.MaxNLocator(nbins=3)


       else:
           gl.bottom_labels = False
           
       gl.xformatter = LONGITUDE_FORMATTER
       gl.yformatter = LATITUDE_FORMATTER

       
    """ 3C Add color bar for entire plot"""
    #add cbar in the figure, for overall figure, not subplots
    cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])  # Define the position of the colorbar
    cbar = fig.colorbar(im, cax=cbar_ax, extend='both')    
    
        #adjust subplot spacing to be smaller
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1)
    
   # Set label and tick parameters for the colorbar
    if var == "Precipitation":
        if mode =='std' or diftype =='abs':
            unit='mm'
        else:
            unit = '%'
    elif var == "Temperature":
        unit = 'K'
    else:
        unit = 'Unknown'

    """4 Include labels for the cbar and for the y and x axis"""
    # Increase distance between colorbar label and colorbar
    cbar.ax.yaxis.labelpad = 20
    if mode =='dif':
        cbar.set_label(f'$\Delta$ {var} [{unit}]', size='15')
    if mode =='std':
        cbar.set_label(f'{var} - model member std [{unit}]', size='15')
    cbar.ax.tick_params(labelsize=15)
    
    if timeframe=='monthly':
        fig.text(0.5, 0.03, 'Longitude', ha='center', fontsize=15)
        fig.text(0.03, 0.5, 'Latitude', va='center', rotation='vertical', fontsize=15)
        fig.text(0.5, 0.92, model , ha='center', fontsize=20)
    else:
        fig.text(0.5, 0.01, 'Longitude', ha='center', fontsize=15)
        fig.text(-0.02, 0.5, 'Latitude', va='center', rotation='vertical', fontsize=15)
        fig.text(0.5, 0.92, model , ha='center', fontsize=20)

        
   
    
    if plotsave =='save': 
        if var=="Precipitation":
            plt.savefig(f'/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/04. Output files/Zoom/1985/{timeframe}/{var}/1985.2014.{var}.{timeframe}.{mode}.{diftype}.{model}.zoom.png', bbox_inches='tight')
        else:
            plt.savefig(f'/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/04. Output files/Zoom/1985/{timeframe}/{var}/1985.2014.{var}.{timeframe}.{mode}.abs.{model}.zoom.png', bbox_inches='tight')
 
    plt.show()

    # """ Part 5: Test input data by creaing a plots of the input data"""
    
    # if inputplot =='on':
    #     #assign specific scales for testing absolute IRR and NOI values --> differences are in different order of magnitude
    #     if var =="Temperature":
    #         vmin_test = 240
    #         vmax_test=300
    #     if var =="Precipitation":
    #         vmin_test=0
    #         vmax_test=150
            
    #     irr_plot=irrigation.plot(x="lon", y="lat", col=time_type, col_wrap=col_wrap, vmax= vmax_test, vmin=vmin_test)
    #     irr_plot=plt.gca()
    #     irr_plot.text(figsize[0], figsize[1], 'IRR', ha='center', rotation='horizontal', color='red')

    #     noi_plot=baseline.plot(x="lon", y="lat", col=time_type, col_wrap=col_wrap, vmax= vmax_test, vmin=vmin_test)
    #     noi_plot=plt.gca()
    #     noi_plot.text(figsize[0], figsize[1], 'NOI', ha='center', rotation='horizontal', color='red')
        
    return

# for model in ["IPSL-CM6","E3SM","CESM2", "CNRM"]: 
#     for var in ["Temperature", "Precipitation"]:
#         for timeframe in ["annual","seasonal","monthly"]:
#             for mode in ['dif']:#, 'std']:
#                 if var =="Precipitation" and mode=='dif':
#                     diftypes=['abs','rel']
#                 else:
#                     diftypes=['abs']
                
#                 for dif in diftypes:
#                     P_T_overview_plot(model, var, timeframe, mode, dif, 'off', 'save')
P_T_overview_plot("CNRM", "Precipitation", "annual", "dif", "abs", 'off', 'nosave')

                
