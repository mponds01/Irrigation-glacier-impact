# -*- coding: utf-8 -*-


import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import geopandas as gpd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import os 

#%%
def plot_P_T_perturbed_baseline(model, scale, var, timeframe, plotsave):    
    
    """ Part 0 - Set plotting parameters"""
    #adjust figure sizes towards type of plot
    if scale=="Global":
        if timeframe=='monthly': 
            figsize=(25, 12)
            fig, axes = plt.subplots(nrows=3, ncols=4, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=figsize)
            time_signature = 'ymon'
            timestamps =['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN','JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'] 
            time_averaging = 'time.month'
            time_type='month'
            col_wrap = 4
        if timeframe =='seasonal':
            figsize=(12,7.5)
            fig, axes = plt.subplots(nrows=2, ncols=2, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=figsize)
            time_signature ='yseas'
            timestamps =['DJF','MAM','JJA','SON']
            time_averaging = 'time.season'
            time_type='season'
            col_wrap = 2
        if timeframe =='annual':
            figsize=(7,5)
            fig, axes = plt.subplots(nrows=1, ncols=1, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=figsize)
            time_signature ='year'
            timestamps=['YEAR']
            time_averaging = 'time.year'
            time_type='year'
            col_wrap = 1
            
    if scale =="Local":
        if timeframe=='monthly': 
            figsize=(18, 10)
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
    
    #Provide cbar ranges and colors for plots for different variables, modes (dif/std) and difference types (abs/rel)    
    if var=="Precipitation":
        var_suffix="PR"
        var_idx='prcp'
        vmin=0
        if timeframe=="monthly":
            vmax=60
        if timeframe=="seasonal":
            vmax=180
        if timeframe=="annual":
            vmax=720
        # zero_scaled = (abs(vmin)/(abs(vmin)+abs(vmax)))
        colors = [(vmin, 'xkcd:white'),(1, 'xkcd:aquamarine')]
        custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors) 
    elif var =="Temperature":
        var_suffix="TEMP"
        var_idx='temp'
        vmin=250
        vmax=290
        zero_scaled = (273-vmin)/(vmax-vmin)
        colors = [(0, 'cornflowerblue'),(zero_scaled, 'xkcd:white'),(1, 'xkcd:tomato')]
        custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
    
    
    """ Part 1: Load the input data (derived in function process_P_T_perturbations) """
    
    base_folder_in = f"/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/03. Output files/01. Climate data/05. OGGM Climate input files/{timeframe}/{model}/{member}"
    ifile_base=f"{base_folder_in}/{model}.00{member}.1985_2014.{timeframe}.perturbed.climate.input.nc"
    
    data = xr.open_dataset(ifile_base)[var_idx]

    if scale=="Local":
        data = data.where((data.lon >= 60) & (data.lon <= 109) & (data.lat >= 22) & (data.lat <= 52), drop=True)
    #create local minima/maxima, for axis of plot
    # local_min_data = data.quantile(0.25)
    # local_max_data = data.quantile(0.75)
    
    """ Part 2 - Shapefile outline for Karakoram Area to be included"""
    #path to  shapefile
    shapefile_path ='/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/01. Input files/03. Shapefile/Karakoram/Pan-Tibetan Highlands/Pan-Tibetan Highlands (Liu et al._2022)/Shapefile/Pan-Tibetan Highlands (Liu et al._2022)_P.shp'
    shp = gpd.read_file(shapefile_path)
    target_crs='EPSG:4326'
    shp = shp.to_crs(target_crs)
    
    
    """ Part 3 - Create subplots for IRR, NOI and DIF"""
    
    #first create output folders for the data
    
    o_folder_data=f"/Users/magaliponds/OneDrive - Vrije Universiteit Brussel/1. VUB/02. Coding/01. IRRMIP/04. Figures/01. Climate data/03. Perturbed Baselines/{scale}/{var}/{model}/{member}"  
    print(data)
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
         # if scale=="Local": 
         time_dim_name = list(data.dims)[0]
         # if scale=="Global":
         #     if timeframe!='annual':
         #         time_dim_name = list(data.dims)[2]
         #     else:
         #         time_dim_dame = list(data.dims)
             
         # if timeframe=='annual':
         #    data_sel = data
         # else:
         data_sel = data.isel({time_dim_name: time_idx})
         
         #make into dataframe, else it doesnt work
         if isinstance(data_sel, xr.Dataset):
             data_sel = data_sel[list(data_sel.data_vars.keys())[0]]
       
         #plot the data incl the outline of the karakoram shapefile, setting the colors, but excluding the shapefile
         im = data_sel.plot.imshow(ax=ax, vmin=vmin, vmax=vmax, extend='both', transform=ccrs.PlateCarree(), cmap=custom_cmap, add_colorbar=False)
         shp.plot(ax=ax, edgecolor='black', linewidth=1, facecolor='none')
         # shp.plot(ax=ax, edgecolor='red', linewidth=1, facecolor='none')
 
         ax.coastlines(resolution='10m')  
        
        
         #include month as a label, instead as on top off data
         ax.set_title('')
         ax.annotate(timestamp_name, xy=(1, 1), xytext=(-10, -10), xycoords='axes fraction', textcoords='offset points',
                  ha='right', va='top', fontsize=15, bbox=dict(boxstyle='square', fc='white', alpha=1))
        
        
         """ 3B - Min and Max value annotation"""
         #Include annotation for min and max values in every subplot, excluding NaN from min/max creation
         data_sel_min = data_sel.fillna(data_sel.max())
         data_sel_max = data_sel.fillna(data_sel.min())
        
         #find min and max values in gridcell 
         min_value_index = np.unravel_index(np.argmin(data_sel_min.values), data_sel_min.shape)
         max_value_index = np.unravel_index(np.argmax(data_sel_max.values), data_sel_max.shape)
         
         # Extract longitude and latitude corresponding to the minimum and maximum value indices
         min_lon, min_lat = data_sel.lon.values[min_value_index[1]], data_sel.lat.values[min_value_index[0]]
         max_lon, max_lat = data_sel.lon.values[max_value_index[1]], data_sel.lat.values[max_value_index[0]]
        
         # Plot the dot on the subplot
         ax.plot(min_lon, min_lat, marker='o', markersize=7, color='blue')  # Adjust marker properties as needed
         ax.plot(max_lon, max_lat, marker='o', markersize=7, color='red')  # Adjust marker properties as needed
 
         # #indicate annotations for min and max values in plot, formatted as percenteges for rel precipitation differences
         min_value = data_sel[min_value_index]
         max_value = data_sel[max_value_index]
         
         if scale=="Local":
             if timeframe =='annual':
                 ax.annotate(f'Min: {min_value:.1f}',xy=(65,50), xytext=(65,50),fontsize=15, ha='left', va='top')
                 ax.annotate(f'Max: {max_value:.1f}', xy=(78,50), xytext=(78,50),fontsize=15, ha='left', va='top')
              
                 ax.plot(64,49.4, marker='o', color='blue', markersize=5)
                 ax.plot(77,49.4, marker='o', color='red', markersize=5)
             else:
                 ax.annotate(f'Min: {min_value:.1f}',xy=(64,50), xytext=(64,50),fontsize=14, ha='left', va='top')
                 ax.annotate(f'Max: {max_value:.1f}', xy=(84,50), xytext=(84,50),fontsize=14, ha='left', va='top')
              
                 ax.plot(62,49, marker='o', color='blue', markersize=5)
                 ax.plot(82,49, marker='o', color='red', markersize=5)
         if scale=="Global":
             if timeframe =='monthly':
                 ax.annotate(f'Min: {min_value:.1f}',xy=(0,-75), xytext=(0,-75),fontsize=15, ha='left', va='top')
                 ax.annotate(f'Max: {max_value:.1f}', xy=(85,-75), xytext=(85,-75),fontsize=15, ha='left', va='top')
                 ax.plot(-4,-78, marker='o', color='blue', markersize=5)
                 ax.plot(80,-78, marker='o', color='red', markersize=5)
             else:
                 ax.annotate(f'Min: {min_value:.1f}',xy=(0,-75), xytext=(0,-75),fontsize=15, ha='left', va='top')
                 ax.annotate(f'Max: {max_value:.1f}', xy=(85,-75), xytext=(85,-75),fontsize=15, ha='left', va='top')
                 ax.plot(-4,-79, marker='o', color='blue', markersize=5)
                 ax.plot(80,-79, marker='o', color='red', markersize=5)
            
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
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.05, hspace=0.05)
     
    # Set label and tick parameters for the colorbar
    if var == "Precipitation":
        unit = 'mm'
    elif var == "Temperature":         
        unit = 'K'
    
        
    """4 Include labels for the cbar and for the y and x axis"""
    # Increase distance between colorbar label and colorbar
    cbar.ax.yaxis.labelpad = 20
    cbar.set_label(f'{var} [{unit}]', size='15')     
     
    cbar.ax.tick_params(labelsize=15)
     
    if timeframe=='monthly':
        fig.text(0.5, 0.03, 'Longitude', ha='center', fontsize=15)
        fig.text(0.03, 0.5, 'Latitude', va='center', rotation='vertical', fontsize=15)
    else:         
        fig.text(0.5, 0.01, 'Longitude', ha='center', fontsize=15)
        fig.text(-0.02, 0.5, 'Latitude', va='center', rotation='vertical', fontsize=15)
        
    if model=="W5E5":
        fig.text(0.5, 0.92,str(model) + " baseline climate", ha='center', fontsize=15)
    else:
        fig.text(0.5, 0.92, str(model) + " 00" + str(member) + " perturbed climate", ha='center', fontsize=15)
            
    if plotsave =='save': 
        # os.makedirs(f"o_folder_base/{scale}/{timeframe}/{var}/", exist_ok=True)
        os.makedirs(f"{o_folder_data}/", exist_ok=True)
        o_file_name=f"{o_folder_data}/{model}.00{member}.1985_2014.{timeframe}.perturbed.climate.input.png"
        plt.savefig(o_file_name, bbox_inches='tight')
        
    plt.show()
    return


#%%

members=[3,4,6,1] #1
for (m,model) in enumerate(["E3SM","CESM2","CNRM","IPSL-CM6"]):#"W5E5",
    for member in range(members[m]):
        print(member)
        for scale in ["Local"]:#,"Global"]:
            for var in ["Precipitation","Temperature"]:#, "Precipitation"]:
                for timeframe in ["annual","seasonal","monthly"]:
                    
                    # plot_P_T_perturbations(model, scale, var, timeframe, mode, dif,"save")
                    plot_P_T_perturbed_baseline(model, scale, var, timeframe,"save")
                        