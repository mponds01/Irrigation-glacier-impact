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
import geopandas as gpd
from shapely.geometry import box

fn  = '/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/01. Input files/CESM2.DIF.001.clm.h0.1981_2014.ymondif.nc'
ds = nc.Dataset(fn, 'r')


"""
#print all variables
"""

# for var in ds.variables:
    # print(var)
    
""" 
load the variables
"""
P = ds.variables['RAIN_FROM_ATM'][:]
T = ds.variables['TSA']
lon = ds.variables['lon'][:]
lat = ds.variables['lat'][:]

""" 
load long and latitude zoom
"""




months = ['01', '02', '03', '04', '05', '06',
          '07', '08', '09', '10', '11', '12']

def individual_figures_zoom(var, lon, lat, modelname):
# Plot precipitation data for each month
    figures = []
    lon_range = (65, 100)
    lat_range = (20, 50)
    
    lat_indices = np.where((lat >= lat_range[0]) & (lat <= lat_range[1]))[0]
    lon_indices = np.where((lon >= lon_range[0]) & (lon <= lon_range[1]))[0]
    var_zoom = ds.variables[var][:, lat_indices, lon_indices]
    global_min=np.min(var_zoom)
    global_max=np.max(var_zoom)
    
   
    unit = ds.variables[var].units

    if var=='RAIN_FROM_ATM':
        cmap = cm.get_cmap('BrBG')
        rounded_min=np.round(global_min,7)
        rounded_max=np.round(global_max,7)
        levels = np.linspace(global_min, global_max, 100) 
        

    else:
        cmap = cm.get_cmap('RdBu')
        rounded_min = math.floor(global_min * 2) / 2
        rounded_max = math.ceil(global_max * 2) / 2
        levels = np.arange(rounded_min, rounded_max +0.1, 0.1) 
    
    shapefile_path = '/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/01. Input files/Shapefile/glac_inv/glac_inv_all.shp'
    shp = gpd.read_file(shapefile_path)
    target_crs='EPSG:4326'
    shp = shp.to_crs(target_crs)

    # bbox = box(lon_range[0], lat_range[0], lon_range[1], lat_range[1])   
    # shp_clipped = shp[shp.geometry.intersects(bbox)]
    
    rows=3
    cols=4
    subplot_width=6
    subplot_height=6
    total_w=cols*subplot_width
    total_h=rows*subplot_height

    fig_tot,ax_tot = plt.subplots(rows, cols,figsize=(total_w,total_h), gridspec_kw={'hspace': 0.02, 'wspace': 0.02})
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(resolution='10m')
    ax_tot=ax_tot.flatten()
    
    for month_idx, month_name in enumerate(months):
        
        print(month_name,'done')
        # Get precipitation data for the current month
        var_zoom_month = var_zoom[month_idx, :, :]
        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.coastlines(resolution='10m')
        
       
    
        # Plot the data on the map for worldwide
        # im = ax.contourf(lon, lat, P_month, transform=ccrs.PlateCarree())
        
        # for only selected area
        im = ax.contourf(lon[lon_indices], lat[lat_indices], var_zoom_month,
                          transform=ccrs.PlateCarree(), levels=levels, cmap=cmap, vmin=rounded_min, vmax=rounded_max)
        # shp.plot(ax=ax, edgecolor='black')
        # Add a colorbar        
        # if month_idx==11:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f'$\Delta$ {var} [{unit}]', size='14')
        cbar.ax.tick_params(labelsize=14)
        plt.set_cmap(cmap)
        
    
        # Set title
        # plt.suptitle(f'{var} change')
        plt.title(f'{modelname}, 1981-2014', fontsize='14')
        
        plt.annotate(month_idx+1, xy=(1, 1), xytext=(-10, -10), xycoords='axes fraction', textcoords='offset points',
                 ha='right', va='top', fontsize=18, bbox=dict(boxstyle='square', fc='white', alpha=1))
        
        # Set the map gridlines
        gl = ax.gridlines(draw_labels=True)
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xlabel_style = {'size': 14}  # Set font size for x-axis labels
        gl.ylabel_style = {'size': 14} 
       
        
    
        # Show the plot
        # figures.append(fig)
    return figures
# After you're done, close the NetCDF file

# figures = individual_figures_zoom('RAIN_FROM_ATM',lon,lat, 'CNRM-2.001')

def individual_figures_zoom_subplot(var, lon, lat, modelname):
# Plot precipitation data for each month
    figures = []
    lon_range = (65, 100)
    lat_range = (20, 50)
    
    lat_indices = np.where((lat >= lat_range[0]) & (lat <= lat_range[1]))[0]
    lon_indices = np.where((lon >= lon_range[0]) & (lon <= lon_range[1]))[0]
    var_zoom = ds.variables[var][:, lat_indices, lon_indices]
    global_min=np.min(var_zoom)
    global_max=np.max(var_zoom)
    
   
    unit = ds.variables[var].units

    if var=='RAIN_FROM_ATM':
        cmap = cm.get_cmap('BrBG')
        rounded_min=np.round(global_min,7)
        rounded_max=np.round(global_max,7)
        levels = np.linspace(global_min, global_max, 100) 
        

    else:
        cmap = cm.get_cmap('RdBu')
        rounded_min = math.floor(global_min * 2) / 2
        rounded_max = math.ceil(global_max * 2) / 2
        levels = np.arange(rounded_min, rounded_max +0.1, 0.1) 
    
    shapefile_path = '/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/01. Input files/Shapefile/glac_inv/glac_inv_all.shp'
    shp = gpd.read_file(shapefile_path)
    target_crs='EPSG:4326'
    shp = shp.to_crs(target_crs)

    # bbox = box(lon_range[0], lat_range[0], lon_range[1], lat_range[1])   
    # shp_clipped = shp[shp.geometry.intersects(bbox)]
    
    rows=3
    cols=4
    subplot_width=6
    subplot_height=6
    total_w=cols*subplot_width
    total_h=rows*subplot_height

    fig_tot,ax_tot = plt.subplots(rows, cols,figsize=(total_w,total_h), gridspec_kw={'hspace': 0.02, 'wspace': 0.02})
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(resolution='10m')
    ax_tot=ax_tot.flatten()
    
    for month_idx, month_name in enumerate(months):
        
        print(month_name,'done')
        # Get precipitation data for the current month
        var_zoom_month = var_zoom[month_idx, :, :]
        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.coastlines(resolution='10m')
        
       
    
        # Plot the data on the map for worldwide
        # im = ax.contourf(lon, lat, P_month, transform=ccrs.PlateCarree())
        
        # for only selected area
        im = ax.contourf(lon[lon_indices], lat[lat_indices], var_zoom_month,
                          transform=ccrs.PlateCarree(), levels=levels, cmap=cmap, vmin=rounded_min, vmax=rounded_max)
        # shp.plot(ax=ax, edgecolor='black')
        # Add a colorbar        
        # if month_idx==11:
        
    
        # Set title
        # plt.suptitle(f'{var} change')
        plt.title(f'{modelname}, 1981-2014', fontsize='14')
        
        plt.annotate(month_idx+1, xy=(1, 1), xytext=(-10, -10), xycoords='axes fraction', textcoords='offset points',
                 ha='right', va='top', fontsize=18, bbox=dict(boxstyle='square', fc='white', alpha=1))
        # Set the map gridlines
        gl = ax.gridlines(draw_labels=True)
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xlabel_style = {'size': 14}  # Set font size for x-axis labels
        gl.ylabel_style = {'size': 14} 
        figures.append(fig)
        
        if month_idx+1==12:
            fig = plt.figure(figsize=(10, 8))
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.coastlines(resolution='10m')
            im = ax.contourf(lon[lon_indices], lat[lat_indices], var_zoom_month,
                              transform=ccrs.PlateCarree(), levels=levels, cmap=cmap, vmin=rounded_min, vmax=rounded_max)
            cbar = plt.colorbar(im, ax=ax, orientation="horizontal")
            cbar.set_label(f'$\Delta$ {var} [{unit}]', size='14')
            cbar.ax.tick_params(labelsize=14)
            plt.set_cmap(cmap)
            figures.append(fig)
       
        
    
        # Show the plot
    return figures
# After you're done, close the NetCDF file

figures = individual_figures_zoom_subplot('RAIN_FROM_ATM',lon,lat, 'CNRM-2.001')

