# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import math
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import geopandas as gpd
from matplotlib.ticker import ScalarFormatter

"""
#print all variables, change timeframe to monthly, seasonal, annual, daily for other data splits
""" 
# fn  = f'/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/01. Input files/monthly/CESM2.DIF.000.clm.h0.1981_2014.ymonmean.total.nc'
# ds = nc.Dataset(fn, 'r')
# print(ds)
 
# for var in ds.variables:
#     print(var)
# Adjusted longitude data (-180 to 180)




def individual_figures_zoom_subplot(var, model,member,total_members, zoom, timeframe, stat):
    
    
    #indicate variable indicators for different timeframes - e.g. for ymon for loading the file and timestamps to plot on the figure
    if timeframe == 'monthly':
        time_signature = 'ymon'
        timestamps =['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN','JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'] 
        #['January', 'February', 'March', 'April', 'May', 'June','July', 'August', 'September', 'October', 'November', 'December']
    if timeframe == 'seasonal':
        time_signature ='yseas'
        timestamps =['DJF','MAM','JJA','SON']
    if timeframe =='daily':
        time_signature = 'yday'
    if timeframe == 'annual':
        time_signature ='year'
        timestamps=['YEAR']
    
    #load data from the indicated path (downloaded after processing in VSC)
    fn  = f'/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/01. Input files/{timeframe}/{model}.DIF.00{member}.clm.h0.1981_2014.{time_signature}{stat}.total.nc'
    ds = nc.Dataset(fn, 'r')    

    #load lon- & latitude of global dataset
    lon = ds.variables['lon'][:]
    lat = ds.variables['lat'][:]
    
    #indicate what area to be plotted (e.g. karakoram only or global)
    if zoom =='on':
        lon_range = (65, 104)
        lat_range = (20, 50)
    if zoom =='off':
          lon_range = (-180, 180)
          lat_range = (-180, 180)
     
    #find indices corresponding to specified zoom
    lat_indices = np.where((lat >= lat_range[0]) & (lat <= lat_range[1]))[0]
    lon_indices = np.where((lon >= lon_range[0]) & (lon <= lon_range[1]))[0]
    
    #create arrays to store local min/max and compare between all 3 members of the cilmate model --> only abs min/max of model are used for legend
    local_min=0
    local_max=0
    min_scale = np.zeros((total_members, len(timestamps)))
    max_scale = np.zeros((total_members, len(timestamps)))
    
    var_zoom = ds.variables[var][:, lat_indices, lon_indices]
    
    #adjust legend to be equal for all members in the model, could be removed if we take average of all members
    for i in np.arange(0, total_members,1):
        print(i)
        for t, timestamp_name in enumerate(timestamps):
            fn_member = f'/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/01. Input files/{timeframe}/{model}.DIF.00{i}.clm.h0.1981_2014.{time_signature}{stat}.total.nc'
            ds_member=nc.Dataset(fn_member)
            ds_all = ds_member.variables[var][:, lat_indices, lon_indices]
    
            min_scale[i,t]=np.min(ds_all[0,:,:])
            max_scale[i,t]=np.max(ds_all[0,:,:])        
            # print(np.argsort(ds_all)[-5:])
            
            #test plot for legend data
            print(min_scale[i,t], max_scale[i,t])

            if i==total_members-1 and t==len(timestamps)-1:
                local_min=np.min(min_scale)
                local_max=np.max(max_scale)       
    
    #use only variable data corresponding to indicated zoom
      
    #set legend scale & tick interval for each variable 
    if var=='RAIN_FROM_ATM_PERDAY' or var=='RAIN_FROM_ATM_PERDAY_REL':
        #define legend scale based on local min & max values        
        rounded_max=local_max#np.round(local_max,10) 
        rounded_min=local_min#np.round(local_min,10) 
        levels = np.linspace(local_min, local_max, 1000) 
        
        #centre scale around zero
        zero_scaled = (abs(rounded_min)/(abs(rounded_min)+abs(rounded_max)))
        if stat=='mean':
            colors = [(0, 'xkcd:mocha'),(zero_scaled, 'white'),(1, 'xkcd:aquamarine')]
            custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)  
        if stat =='std':
            colors = [(0, 'xkcd:tomato'),(zero_scaled, 'white'),(1, 'xkcd:purple')]
            custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)  
        variable_type = 'RAIN_FROM_ATM_PERDAY'
    if var=='TSA' or var=='TSA_REL':
        #define legend scale based on local min& max values
        rounded_min = math.floor(local_min * 2) / 2
        rounded_max = math.ceil(local_max * 2) / 2
        levels = np.arange(rounded_min, rounded_max +0.1, 0.1) 
        #centre scale around zero
        zero_scaled = (abs(rounded_min)/(abs(rounded_min)+abs(rounded_max)))
        if stat=='mean':
           colors = [(0, 'blue'),(zero_scaled, 'white'),(1, 'red')]
           custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
        if stat =='std':
           colors = [(0, 'xkcd:lightgreen'),(zero_scaled, 'white'),(1, 'xkcd:gold')]
           custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)   
        variable_type = 'TSA'

    #defined units and change_mode for plotting (change mode used to store data)
    if var =='TSA' or var=='RAIN_FROM_ATM_PERDAY':
        unit = ds.variables[var].units
        change_mode ='absolute'
    else:
        unit = '-'
        change_mode ='relative'

    #load shapefile, in right projection, to plot karakoram glacier outlines   
    shapefile_path = '/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/01. Input files/Shapefile/glac_inv/glac_inv_all_lowres.shp'
    shp = gpd.read_file(shapefile_path)
    target_crs='EPSG:4326'
    shp = shp.to_crs(target_crs)
    
    #adjust figure sizes towards type of plot
    if timeframe=='monthly': 
        fig, axes = plt.subplots(nrows=3, ncols=4, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(20, 13))
    if timeframe =='seasonal':
        fig, axes = plt.subplots(nrows=2, ncols=2, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(14, 13))
    if timeframe =='annual':
        fig, axes = plt.subplots(nrows=1, ncols=1, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10, 9))
            
    #loop through data for different timestamps (e.g. 4 seasons, 12 months or 1 year)
    for time_idx, timestamp_name in enumerate(timestamps):
        
        print(timestamp_name,'done')
        # Get variable data for the current timestamp (e.g. season, month or entire year)
        var_zoom_time = var_zoom[time_idx, :, :]
        
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
                
       
        # Plot data on each subplot   
        im = ax.contourf(lon[lon_indices], lat[lat_indices], var_zoom_time,
                  transform=ccrs.PlateCarree(), cmap=custom_cmap, vmin=rounded_min, vmax=rounded_max,levels=levels, extend='both')     
       
        # Add gridlines and ticks
        ax.coastlines(resolution='10m')   
        
        #include outline of glaciers in karakoram area in case zoom is on
        if zoom=='on':
             shp.plot(ax=ax, edgecolor='black', linewidth=2, facecolor='none')
        
        #Determine min and max values, their locations and annotate
        min_value_index = np.unravel_index(np.argmin(var_zoom_time), var_zoom_time.shape)
        max_value_index = np.unravel_index(np.argmax(var_zoom_time), var_zoom_time.shape)
        min_lon, min_lat = lon[lon_indices][min_value_index[1]], lat[lat_indices][min_value_index[0]]
        max_lon, max_lat = lon[lon_indices][max_value_index[1]], lat[lat_indices][max_value_index[0]]

        min_value = var_zoom_time[min_value_index]
        max_value = var_zoom_time[max_value_index]
       
        ax.annotate(f'Min: {min_value:.2e}',xy=(67,48), xytext=(67,48),fontsize=10, ha='left', va='top')
        ax.annotate(f'Max: {max_value:.2e}', xy=(79,48), xytext=(79,48),fontsize=10, ha='left', va='top')
        ax.plot(min_lon, min_lat, marker='o', color='blue', markersize=10)
        ax.plot(max_lon, max_lat, marker='o', color='red', markersize=10)
        
        # Set the map gridlines
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
        
        #annotate month number
        ax.annotate(timestamp_name, xy=(1, 1), xytext=(-10, -10), xycoords='axes fraction', textcoords='offset points',
                  ha='right', va='top', fontsize=20, bbox=dict(boxstyle='square', fc='white', alpha=1))
        
       #indicate only axis labels for outer plots
        if col == 0:
            gl.ylabel_style = {'size': 16} 
        else:
            gl.left_labels = False
        
        if (timeframe=='monthly' and row==2) or (timeframe=='seasonal' and row==1) or (timeframe=='annual'):
            gl.xlabel_style = {'size': 16} 
        else:
            gl.bottom_labels = False

    #add an legend for all plots, specifying location, ticks and label
    cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])  # Define the position of the colorbar
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(f'$\Delta$ {var} [{unit}]', size='16')
    cbar.ax.tick_params(labelsize=16)
    
    #format the ticks to be mathmathmatical, to the power x, with only 1 decimal
    class ScalarFormatterForceFormat(ScalarFormatter):
        def _set_format(self):  # Override function that finds format to use.
            self.format = "%1.1f"  # Give format here
    formatter = ScalarFormatterForceFormat()
    formatter.set_powerlimits((0,0)) 
    cbar.ax.yaxis.set_major_formatter(formatter)


    # Increase distance between colorbar label and colorbar
    cbar.ax.yaxis.labelpad = 30
    #increase disctance between ylabel and cbar
    cbar.ax.yaxis.offsetText.set_fontsize(16)
    
    #adjust subplot spacing to be smaller
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1)
   
    #include axis labels and title
    fig.text(0.5, 0.04, 'Longitude', ha='center', fontsize=20)
    fig.text(0.04, 0.5, 'Latitude', va='center', rotation='vertical', fontsize=20)
    if member=='0': 
        fig.text(0.5, 0.92, f'{model}.ensemble, 1981-2014, {timeframe} {stat}', ha='center', fontsize=20)
    else:
        fig.text(0.5, 0.92, f'{model}.00{member}, 1981-2014, {timeframe} {stat}', ha='center', fontsize=20)
    
    #indicate suffix for saving file
    if zoom=='on':
        zoomed ="karakoram"
    else:
        zoomed="global"
     
    #save figure
    plt.savefig(f'/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/04. Output files/{model}/{variable_type}/00{member}/1981_2014.{timeframe}.{time_signature}dif.00{member}.{change_mode}.{stat}.{zoomed}.png', bbox_inches='tight')
    # plt.show()

    return 

""" 

Loop through all possible combinations if variables, timeframes, model members and zoom modes

"""

variable_types = ['RAIN_FROM_ATM_PERDAY_REL']#'RAIN_FROM_ATM_REL','RAIN_FROM_ATM', 'TSA', 'TSA_REL']#'RAIN_FROM_ATM_REL']#
timeframes = ['monthly']#, 'seasonal', 'monthly']#, 'daily']
members =[3]#1,2,3]
zoom_modes =['on']#, 'off']

for v, var_types in enumerate(variable_types):
    for t, timeframe in enumerate(timeframes):
        for m, member in enumerate(members):
            for z, zoom in enumerate(zoom_modes):
                if (timeframe=='monthly'):
                    stats=['mean']
                else:
                    stats=['mean']#,'std']
                for s,stat in enumerate(stats):
                    print(var_types, timeframe, member, zoom, stat)
                      #only DIF00 map input used for legend scale
                    # figures = individual_figures_zoom_subplot(var_types,'CESM2', member, 4, zoom, timeframe, stat)

# figures = individual_figures_zoom_subplot('RAIN_FROM_ATM_REL','CESM2','0', 1, 'on', 'annual','mean')
                
"""
Graveyard
"""

def individual_figures_zoom_subplot_test(var, model,member,total_members, zoom, timeframe, stat, datatype):
    
    
    #indicate variable indicators for different timeframes - e.g. for ymon for loading the file and timestamps to plot on the figure
    if timeframe == 'monthly':
        time_signature = 'ymon'
        timestamps =['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN','JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'] 
        #['January', 'February', 'March', 'April', 'May', 'June','July', 'August', 'September', 'October', 'November', 'December']
    if timeframe == 'seasonal':
        time_signature ='yseas'
        timestamps =['DJF','MAM','JJA','SON']
    if timeframe =='daily':
        time_signature = 'yday'
    if timeframe == 'annual':
        time_signature ='year'
        timestamps=['YEAR']
    
    #load data from the indicated path (downloaded after processing in VSC)
    if datatype == 'DIF':
        fn  = f'/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/01. Input files/{timeframe}/{model}.{datatype}.00{member}.clm.h0.1981_2014.{time_signature}{stat}.total.nc'
    else:
        fn  = f'/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/01. Input files/{timeframe}/{model}.{datatype}.00{member}.clm.h0.1981_2014.{time_signature}{stat}.nc'

    ds = nc.Dataset(fn, 'r')    

    #load lon- & latitude of global dataset
    lon = ds.variables['lon'][:]
    lat = ds.variables['lat'][:]
    
    #indicate what area to be plotted (e.g. karakoram only or global)
    if zoom =='on':
        lon_range = (65, 104)
        lat_range = (20, 50)
    if zoom =='off':
          lon_range = (-180, 180)
          lat_range = (-180, 180)
     
    #find indices corresponding to specified zoom
    lat_indices = np.where((lat >= lat_range[0]) & (lat <= lat_range[1]))[0]
    lon_indices = np.where((lon >= lon_range[0]) & (lon <= lon_range[1]))[0]
    
    #create arrays to store local min/max and compare between all 3 members of the cilmate model --> only abs min/max of model are used for legend
    local_min=0
    local_max=0
    min_scale = np.zeros((total_members, len(timestamps)))
    max_scale = np.zeros((total_members, len(timestamps)))
    
    var_zoom = ds.variables[var][:, lat_indices, lon_indices]
    
    #adjust legend to be equal for all members in the model, could be removed if we take average of all members
    for i in np.arange(0, total_members,1):
        print(i)
        for t, timestamp_name in enumerate(timestamps):
            fn_member = f'/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/01. Input files/{timeframe}/{model}.DIF.00{i}.clm.h0.1981_2014.{time_signature}{stat}.total.nc'
            ds_member=nc.Dataset(fn_member)
            ds_all = ds_member.variables[var][:, lat_indices, lon_indices]
    
            min_scale[i,t]=np.min(ds_all[0,:,:])
            max_scale[i,t]=np.max(ds_all[0,:,:])        
            # print(np.argsort(ds_all)[-5:])
            
            #test plot for legend data
            print(min_scale[i,t], max_scale[i,t])
           


            levels=np.linspace(min_scale[i,t], max_scale[i,t],100)
            im = plt.pcolormesh(lon[lon_indices], lat[lat_indices], ds_all[t,:,:],
                      vmin=min_scale[i,t], vmax=max_scale[i,t])#levels=levels,, extend='both'
            plt.colorbar(im, label='Value')
            plt.show()
            if i==total_members-1 and t==len(timestamps)-1:
                local_min=np.min(min_scale)
                local_max=np.max(max_scale)       
    
    #use only variable data corresponding to indicated zoom
      
    #set legend scale & tick interval for each variable 
    if var=='RAIN_FROM_ATM_PERDAY' or var=='RAIN_FROM_ATM_PERDAY_REL':
        #define legend scale based on local min & max values        
        rounded_max=local_max#np.round(local_max,10) 
        rounded_min=local_min#np.round(local_min,10) 
        levels = np.linspace(local_min, local_max, 1000) 
        
        #centre scale around zero
        zero_scaled = (abs(rounded_min)/(abs(rounded_min)+abs(rounded_max)))
        if stat=='mean':
            colors = [(0, 'xkcd:mocha'),(zero_scaled, 'white'),(1, 'xkcd:aquamarine')]
            custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)  
        if stat =='std':
            colors = [(0, 'xkcd:tomato'),(zero_scaled, 'white'),(1, 'xkcd:purple')]
            custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)  
        variable_type = 'RAIN_FROM_ATM_PERDAY'
    if var=='TSA' or var=='TSA_REL':
        #define legend scale based on local min& max values
        rounded_min = math.floor(local_min * 2) / 2
        rounded_max = math.ceil(local_max * 2) / 2
        levels = np.arange(rounded_min, rounded_max +0.1, 0.1) 
        #centre scale around zero
        zero_scaled = (abs(rounded_min)/(abs(rounded_min)+abs(rounded_max)))
        if stat=='mean':
            colors = [(0, 'blue'),(zero_scaled, 'white'),(1, 'red')]
            custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
        if stat =='std':
            colors = [(0, 'xkcd:lightgreen'),(zero_scaled, 'white'),(1, 'xkcd:gold')]
            custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)   
        variable_type = 'TSA'

    #defined units and change_mode for plotting (change mode used to store data)
    if var =='TSA' or var=='RAIN_FROM_ATM_PERDAY':
        unit = ds.variables[var].units
        change_mode ='absolute'
    else:
        unit = '-'
        change_mode ='relative'

    #load shapefile, in right projection, to plot karakoram glacier outlines   
    shapefile_path = '/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/01. Input files/Shapefile/glac_inv/glac_inv_all_lowres.shp'
    shp = gpd.read_file(shapefile_path)
    target_crs='EPSG:4326'
    shp = shp.to_crs(target_crs)
    
    #adjust figure sizes towards type of plot
    if timeframe=='monthly': 
        fig, axes = plt.subplots(nrows=3, ncols=4, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(20, 13))
    if timeframe =='seasonal':
        fig, axes = plt.subplots(nrows=2, ncols=2, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(14, 13))
    if timeframe =='annual':
        fig, axes = plt.subplots(nrows=1, ncols=1, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10, 9))
            
    #loop through data for different timestamps (e.g. 4 seasons, 12 months or 1 year)
    for time_idx, timestamp_name in enumerate(timestamps):
        
        print(timestamp_name,'done')
        # Get variable data for the current timestamp (e.g. season, month or entire year)
        var_zoom_time = var_zoom[time_idx, :, :]
        
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
                
       
        # Plot data on each subplot   
        im = ax.contourf(lon[lon_indices], lat[lat_indices], var_zoom_time,
                  transform=ccrs.PlateCarree(), cmap=custom_cmap, vmin=rounded_min, vmax=rounded_max,levels=levels, extend='both')     
       
        # Add gridlines and ticks
        ax.coastlines(resolution='10m')   
        
        #include outline of glaciers in karakoram area in case zoom is on
        if zoom=='on':
              shp.plot(ax=ax, edgecolor='black', linewidth=2, facecolor='none')
        
        #Determine min and max values, their locations and annotate
        min_value_index = np.unravel_index(np.argmin(var_zoom_time), var_zoom_time.shape)
        max_value_index = np.unravel_index(np.argmax(var_zoom_time), var_zoom_time.shape)
        min_lon, min_lat = lon[lon_indices][min_value_index[1]], lat[lat_indices][min_value_index[0]]
        max_lon, max_lat = lon[lon_indices][max_value_index[1]], lat[lat_indices][max_value_index[0]]

        min_value = var_zoom_time[min_value_index]
        max_value = var_zoom_time[max_value_index]
       
        ax.annotate(f'Min: {min_value:.2e}',xy=(67,48), xytext=(67,48),fontsize=10, ha='left', va='top')
        ax.annotate(f'Max: {max_value:.2e}', xy=(79,48), xytext=(79,48),fontsize=10, ha='left', va='top')
        ax.plot(min_lon, min_lat, marker='o', color='blue', markersize=10)
        ax.plot(max_lon, max_lat, marker='o', color='red', markersize=10)
        
        # Set the map gridlines
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
        
        #annotate month number
        ax.annotate(timestamp_name, xy=(1, 1), xytext=(-10, -10), xycoords='axes fraction', textcoords='offset points',
                  ha='right', va='top', fontsize=20, bbox=dict(boxstyle='square', fc='white', alpha=1))
        
        #indicate only axis labels for outer plots
        if col == 0:
            gl.ylabel_style = {'size': 16} 
        else:
            gl.left_labels = False
        
        if (timeframe=='monthly' and row==2) or (timeframe=='seasonal' and row==1) or (timeframe=='annual'):
            gl.xlabel_style = {'size': 16} 
        else:
            gl.bottom_labels = False

    #add an legend for all plots, specifying location, ticks and label
    cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])  # Define the position of the colorbar
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(f'$\Delta$ {var} [{unit}]', size='16')
    cbar.ax.tick_params(labelsize=16)
    
    #format the ticks to be mathmathmatical, to the power x, with only 1 decimal
    class ScalarFormatterForceFormat(ScalarFormatter):
        def _set_format(self):  # Override function that finds format to use.
            self.format = "%1.1f"  # Give format here
    formatter = ScalarFormatterForceFormat()
    formatter.set_powerlimits((0,0)) 
    cbar.ax.yaxis.set_major_formatter(formatter)


    # Increase distance between colorbar label and colorbar
    cbar.ax.yaxis.labelpad = 30
    #increase disctance between ylabel and cbar
    cbar.ax.yaxis.offsetText.set_fontsize(16)
    
    #adjust subplot spacing to be smaller
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1)
   
    #include axis labels and title
    fig.text(0.5, 0.04, 'Longitude', ha='center', fontsize=20)
    fig.text(0.04, 0.5, 'Latitude', va='center', rotation='vertical', fontsize=20)
    if member=='0': 
        fig.text(0.5, 0.92, f'{model}.ensemble, 1981-2014, {timeframe} {stat}', ha='center', fontsize=20)
    else:
        fig.text(0.5, 0.92, f'{model}.00{member}, 1981-2014, {timeframe} {stat}', ha='center', fontsize=20)
    
    #indicate suffix for saving file
    if zoom=='on':
        zoomed ="karakoram"
    else:
        zoomed="global"
     
    #save figure
    plt.savefig(f'/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/04. Output files/{model}/{variable_type}/00{member}/1981_2014.{timeframe}.{time_signature}dif.00{member}.{change_mode}.{stat}.{zoomed}.png', bbox_inches='tight')
    # plt.show()

    return 

 figures = individual_figures_zoom_subplot('RAIN_FROM_ATM_REL','CESM2','0', 1, 'on', 'annual','mean', 'NOI')


