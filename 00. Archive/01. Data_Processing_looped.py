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

"""
This script performs an initial data screening, in 3 sections:
    1. checks what the monthly irrigation, controlrun and difference in P look like, plotted in a geoplot
    2. Includes the shapefile plot for the Karakoram Area
"""
    

""" 0 . Input files """


def P_T_overview_plot_looped(var, timeframe, mode, diftype, inputplot, plotsave):
    
    all_data = []
    members = [1,2,3]
   

    if mode == 'dif':
        mode_suff = 'total'
    if mode == 'std':
        mode_suff = 'std'

    
    
    """ Part 0 - Create variability for differnt timescales in: amount of subplots, time-averaging used """
    #adjust figure sizes towards type of plot
    if timeframe=='monthly': 
        figsize=(20, 13)
        fig, axes = plt.subplots(nrows=3, ncols=4, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=figsize)
        time_signature = 'ymon'
        timestamps =['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN','JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'] 
        time_averaging = 'time.month'
        time_type='month'
        col_wrap = 4
    if timeframe =='seasonal':
        figsize=(13, 12)
        fig, axes = plt.subplots(nrows=2, ncols=2, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=figsize)
        time_signature ='yseas'
        timestamps =['DJF','MAM','JJA','SON']
        time_averaging = 'time.season'
        time_type='season'
        col_wrap = 2
    if timeframe =='annual':
        figsize=(10.5, 9)
        fig, axes = plt.subplots(nrows=1, ncols=1, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=figsize)
        time_signature ='year'
        timestamps=['YEAR']
        time_averaging = 'time.year'
        time_type='year'
        col_wrap = 1
    
    """Part 1 - load and correct the data"""
    for m in members:
        folder="/Users/magaliponds/OneDrive - Vrije Universiteit Brussel/1. VUB/02. Coding/01. IRRMIP/02. Data/01. Input files/DIF00/"
        ifile_IRR=f"CESM2.IRR.00{m}.clm.h0.1981_2014_selparam_monthly_{mode_suff}.nc"
        ifile_NOI=f"CESM2.NOI.00{m}.clm.h0.1981_2014_selparam_monthly_{mode_suff}.nc"
        shapefile_path = '/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/01. Input files/Shapefile/glac_inv/glac_inv_all_lowres.shp'
        
        #first for each variable load the data, for precipitation this consists of Snow & Rain from atmosphere, converted from [mm/day]
        if var=="Precipitation":
            baseline_R = xr.open_dataset(folder+ifile_NOI).RAIN_FROM_ATM*86400*30
            baseline_S = xr.open_dataset(folder+ifile_NOI).SNOW_FROM_ATM*86400*30
            baseline=baseline_R+baseline_S
            
            irrigation_R = xr.open_dataset(folder+ifile_IRR).RAIN_FROM_ATM*86400*30
            irrigation_S = xr.open_dataset(folder+ifile_IRR).SNOW_FROM_ATM*86400*30
            irrigation=irrigation_R+irrigation_S
        elif var =="Temperature":
            baseline=xr.open_dataset(folder+ifile_NOI).TSA
            irrigation=xr.open_dataset(folder+ifile_IRR).TSA
            
        
        baseline = baseline.where((baseline.lon >= 60) & (baseline.lon <= 100) & (baseline.lat >= 20) & (baseline.lat <= 50), drop=True) 
        baseline["time"] = baseline.time.astype("datetime64[ns]")
        baseline = baseline.groupby(time_averaging).mean()
        
        irrigation = irrigation.where((irrigation.lon >= 60) & (irrigation.lon <= 100) & (irrigation.lat >= 20) & (irrigation.lat <= 50), drop=True) 
        irrigation["time"] = irrigation.time.astype("datetime64[ns]")
        irrigation = irrigation.groupby(time_averaging).mean()
        
        if timeframe=='annual':
            baseline = baseline.mean(dim='year')
            irrigation = irrigation.mean(dim='year')
        
        if mode =='std':
            diff = (irrigation+baseline)/2
        if var =="Precipitation":
            if diftype=='rel':
                diff = ((irrigation-baseline)/baseline *100) 
        if var =="Temperature" or diftype=='abs':
            diff = irrigation-baseline 
        
        # diff.plot(x="lon", y="lat", col="month", col_wrap=4, vmax= 100, cmap='BrBG')
        plt.show()

        #store the difference matrix for each member in 1 data array
        all_data.append(diff)
    
    #stach the matrixes next to eachother, so that they can be averaged over the member dimension
    diff_matrix = np.stack(all_data,axis=0)
    diff_matrix = np.mean(diff_matrix, axis=0)    
   
    #assure the matrix is stacked as xr datatype, so that the data can be processed according to existing code
    diff = xr.DataArray(diff_matrix)

    if timeframe == 'annual':
        diff = diff.rename({'dim_0':'lat', 'dim_1': 'lon'})
    else:
        diff = diff.rename({'dim_0':'time','dim_1': 'lat', 'dim_2': 'lon'})
    
    # Update the dimension names to reflect longitude and latitude
    diff = diff.assign_coords(lon=baseline.lon, lat=baseline.lat)
    
    # Calculate the mean along the members dimension
    
    # Ensure the data is within the desired latitude and longitude range
    diff = diff.where((diff.lon >= 60) & (diff.lon <= 100) & (diff.lat >= 20) & (diff.lat <= 50),drop=True)

    #Provide cbar ranges and colors for plots for different variables, modes (dif/std) and difference types (abs/rel)    
    if var=="Precipitation":
        if mode =='dif' and diftype=='rel':
            vmin=-40
            vmax=40
            zero_scaled = (abs(vmin)/(abs(vmin)+abs(vmax)))
            colors = [(0, 'xkcd:mocha'),(zero_scaled, 'xkcd:white'),(1, 'xkcd:aquamarine')]
            custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors) 
        if mode =='dif' and diftype=='abs':
            vmin=-20
            vmax=20
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
    shapefile_path = '/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/01. Input files/Shapefile/glac_inv/glac_inv_all_lowres.shp'
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
       ax.annotate(f'Min: {min_value:.1f}',xy=(62,48), xytext=(62,48),fontsize=15, ha='left', va='top')
       ax.annotate(f'Max: {max_value:.1f}', xy=(78,48), xytext=(78,48),fontsize=15, ha='left', va='top')
      
       ax.plot(61,47.4, marker='o', color='blue', markersize=5)
       ax.plot(77,47.4, marker='o', color='red', markersize=5)
       
       # Set the map gridlines
       gl = ax.gridlines(draw_labels=True)
       gl.top_labels = False
       gl.right_labels = False
       
       # Set x-ticks using latitude values
       if col == 0:      
           gl.ylabel_style = {'size': 15} 
       else:
           gl.left_labels = False
       
       if (timeframe=='monthly' and row==2) or (timeframe=='seasonal' and row==1) or (timeframe=='annual'):
           gl.xlabel_style = {'size': 15} 
       else:
           gl.bottom_labels = False
       
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
        fig.text(0.5, 0.05, 'Longitude', ha='center', fontsize=15)
        fig.text(0.05, 0.5, 'Latitude', va='center', rotation='vertical', fontsize=15)
    else:
        fig.text(0.5, 0.02, 'Longitude', ha='center', fontsize=15)
        fig.text(0.02, 0.5, 'Latitude', va='center', rotation='vertical', fontsize=15)
   
    
    if plotsave =='save': 
        if var=="Precipitation":

            plt.savefig(f'/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/04. Output files/CESM2/Test/1981.2014.{var}.{timeframe}.{mode}.{diftype}.zoom.avgtest.png', bbox_inches='tight')
        else:
            plt.savefig(f'/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/04. Output files/CESM2/Test/1981.2014.{var}.{timeframe}.{mode}.abs.zoom.avgtest.png', bbox_inches='tight')
 
    plt.show()

    """ Part 5: Test input data by creaing a plots of the input data"""
    
    if inputplot =='on':
        #assign specific scales for testing absolute IRR and NOI values --> differences are in different order of magnitude
        if var =="Temperature":
            vmin_test = 240
            vmax_test=300
        if var =="Precipitation":
            vmin_test=0
            vmax_test=150
            
        irr_plot=irrigation.plot(x="lon", y="lat", col=time_type, col_wrap=col_wrap, vmax= vmax_test, vmin=vmin_test)
        irr_plot=plt.gca()
        irr_plot.text(figsize[0], figsize[1], 'IRR', ha='center', rotation='horizontal', color='red')

        noi_plot=baseline.plot(x="lon", y="lat", col=time_type, col_wrap=col_wrap, vmax= vmax_test, vmin=vmin_test)
        noi_plot=plt.gca()
        noi_plot.text(figsize[0], figsize[1], 'NOI', ha='center', rotation='horizontal', color='red')
        
    return

# for var in ["Precipitation"]:#"Temperature", 
#     for timeframe in ["monthly","seasonal","annual"]:
#         for mode in ["dif"]:
#             if var =="Precipitation" and mode=="dif":
#                 diftypes=['abs','rel']
#             else:
#                 diftypes=['abs']
#             for dif in diftypes:
#                 P_T_overview_plot_looped(var, timeframe, mode, dif, 'off', 'save')

P_T_overview_plot_looped("Precipitation", "monthly", "dif", "abs", 'off', 'nosave')

                
