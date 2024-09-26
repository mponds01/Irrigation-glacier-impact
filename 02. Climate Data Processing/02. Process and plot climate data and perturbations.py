#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 14:14:34 2024

@author: magaliponds
"""

"""
This script performs an initial data screening, in 3 sections:
    1. checks what the monthly irrigation, controlrun and difference in P look like, plotted in a geoplot
    2. Includes the shapefile plot for the Karakoram Area
"""


# %% Cell 1: Process the climate data - stored as netCDF output




import pandas as pd
from calendar import monthrange
import calendar
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import geopandas as gpd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import os
def process_P_T_perturbations(model, member, var, timeframe, mode, diftype):

    if mode == 'dif':
        mode_suff = 'total'
    if mode == 'std':
        mode_suff = 'std'

    # Enable variability for differnt timescales in: amount of subplots, time-averaging used
    if timeframe == 'monthly':
        time_averaging = 'time.month'
    if timeframe == 'seasonal':
        time_averaging = 'time.season'
    if timeframe == 'annual':
        time_averaging = 'time.year'

    """ Part 0 - Load and open the climate data"""
    # paths to climate data
    folder_in = f"/Users/magaliponds/OneDrive - Vrije Universiteit Brussel/1. VUB/02. Coding/01. IRRMIP/03. Data/01. Input files/01. Climate data/{model}/"
    ifile_IRR = f"{model}.IRR.00{member}.1985_2014_selparam_monthly_{mode_suff}.nc"
    ifile_NOI = f"{model}.NOI.00{member}.1985_2014_selparam_monthly_{mode_suff}.nc"
    # first for each variable load the data, for precipitation this consists of Snow & Rain from atmosphere, converted from [mm/day]
    ifile_IRR = xr.open_dataset(folder_in+ifile_IRR)
    ifile_NOI = xr.open_dataset(folder_in+ifile_NOI)

    """Part 1 - Correct the data"""
    # only for model IPSL-CM6 change dimensions of time (rename), as not possible in bash
    if model == "IPSL-CM6":
        ifile_IRR = ifile_IRR.rename({'time_counter': 'time'})
        ifile_NOI = ifile_NOI.rename({'time_counter': 'time'})

    if var == "Precipitation":

        baseline_R = ifile_NOI.pr*86400
        irrigation_R = ifile_IRR.pr*86400
        if model in ["CESM2", "E3SM"]:
            baseline_S = ifile_NOI.sn*86400
            baseline = baseline_R+baseline_S
            irrigation_S = ifile_IRR.sn*86400
            irrigation = irrigation_R+irrigation_S

            days_in_month_da = baseline.time.dt.days_in_month
            days_in_month_broadcasted = days_in_month_da.broadcast_like(
                baseline)

            irrigation = irrigation*days_in_month_broadcasted
            baseline = baseline*days_in_month_broadcasted
            # rename the variables in the dataset
            if irrigation.name == None:
                irrigation.name = 'pr'
            if baseline.name == None:
                baseline.name = 'pr'

        else:
            baseline = baseline_R
            irrigation = irrigation_R
    elif var == "Temperature":
        baseline = ifile_NOI.tas
        irrigation = ifile_IRR.tas

    """Part 2 - Enable time averaging for the climate data (monthly, seasonal or annual)"""

    # reformat the time dimension of the data for resampling
    baseline["time"] = baseline.time.astype("datetime64[ns]")
    irrigation["time"] = irrigation.time.astype("datetime64[ns]")

    # calculate monthly averages (different from precipitation where calculate average monthly totals)
    if var == "Temperature":
        if timeframe == 'annual':
            baseline = baseline.mean(dim='time')
            irrigation = irrigation.mean(dim='time')
        else:
            baseline = baseline.groupby(time_averaging).mean(dim='time')
            irrigation = irrigation.groupby(time_averaging).mean(dim='time')
    # different approach for precipitation as we are interested in the total amount of precipitation on a seasonal and annual basis
    # the input comes from monthly totals, so we don't need to sum over monthly data anymore
    if var == "Precipitation":
        if timeframe == 'seasonal':
            irrigation = irrigation.resample(time='QS-DEC').sum(dim='time')
            baseline = baseline.resample(time='QS-DEC').sum(dim='time')
        if timeframe == 'annual':
            irrigation = irrigation.groupby(time_averaging).sum(dim='time')
            irrigation = irrigation.mean(dim='year')
            baseline = baseline.groupby(time_averaging).sum(dim='time')
            baseline = baseline.mean(dim='year')
        else:
            irrigation = irrigation.groupby(time_averaging).mean(dim='time')
            baseline = baseline.groupby(time_averaging).mean(dim='time')

    # for std the difference is calculated as the average between std of irrigation and baseline - not used in data output
    if mode == 'std':
        diff = (irrigation+baseline)/2
    # calculate relative precipitation perturbation compared to baseline
    if var == "Precipitation":
        if diftype == 'rel':
            diff = ((irrigation-baseline)/irrigation * 100)
            mask = np.isinf(diff.values)
            diff.values[mask] = 0
            # set infinite values to 0, as these values result from very very small differences
        var_suffix = "PR"
    # calculate absolute temperature perturbation compared to baseline
    if diftype == 'abs':
        diff = irrigation-baseline

    diff = diff.to_dataset()
    irrigation = irrigation.to_dataset()
    baseline = baseline.to_dataset()
    # Add attributes to all the output files
    if var == "Temperature":
        var_suffix = "TEMP"
        diff.tas.attrs['units'] = "K"
        diff.tas.attrs['long_name'] = "Perturbation in temperature 2m"
        baseline.tas.attrs['units'] = "K"
        baseline.tas.attrs['long_name'] = "Temperature 2m"
        irrigation.tas.attrs['units'] = "K"
        irrigation.tas.attrs['long_name'] = "Temperature 2m"
    if var == "Precipitation":
        diff.pr.attrs['units'] = "%"
        diff.pr.attrs[
            'long_name'] = f"perturbation in precipitation: ((irrigation-baseline)/irrigation * 100), base: kg/m2, ({timeframe})"
        baseline.pr.attrs['units'] = "kg/m2"
        baseline.pr.attrs['long_name'] = f"Precip Total liq+sol ({timeframe})"
        irrigation.pr.attrs['units'] = "kg/m2"
        irrigation.pr.attrs['long_name'] = f"Precip Total liq+sol ({timeframe})"

    # save difference and processed input files:
    base_folder_out = f"/Users/magaliponds/OneDrive - Vrije Universiteit Brussel/1. VUB/02. Coding/01. IRRMIP/03. Data/03. Output files/01. Climate data/01. Processed input data/{var}/{timeframe}/{model}/{member}"
    diff_folder_out = f"/Users/magaliponds/OneDrive - Vrije Universiteit Brussel/1. VUB/02. Coding/01. IRRMIP/03. Data/03. Output files/01. Climate data/02. Perturbations/{var}/{timeframe}/{model}/{member}"
    os.makedirs(diff_folder_out, exist_ok=True)
    os.makedirs(base_folder_out, exist_ok=True)

    ofile_diff = f"{diff_folder_out}/{model}.{var_suffix}.DIF.00{member}.1985_2014_{timeframe}_{diftype}.nc"
    ofile_irr = f"{base_folder_out}/{model}.{var_suffix}.IRR.00{member}.1985_2014_{timeframe}_{diftype}.nc"
    ofile_noi = f"{base_folder_out}/{model}.{var_suffix}.NOI.00{member}.1985_2014_{timeframe}_{diftype}.nc"

    diff.to_netcdf(ofile_diff)
    irrigation.to_netcdf(ofile_irr)
    baseline.to_netcdf(ofile_noi)

    return


# %% Cell 2: Process baseline

# for model in ["W5E5"]:

#     for member in [0]:
#         for var in ["Precipitation", "Temperature"]:
#             for timeframe in ["annual", "seasonal", "monthly"]:
#                     for diftype in ['abs']:
#                         for y0 in [1985,1901]:
#                             print(var, timeframe, y0)

def process_P_T_baseline(model, member, var, timeframe, diftype, y0, ye):

    var == "Precipitation"
    if timeframe == 'seasonal':
        time_averaging = "QS-DEC"
    if timeframe == 'annual':
        time_averaging = 'YE'

    """ Part 0 - Load and open the climate data"""
    # paths to climate data

    w5e5_path = '/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/01. Input files/01. Climate data/W5E5/cluster.klima.uni-bremen.de/~oggm/climate/gswp3-w5e5/unflattened/2023.2/monthly'
    pr_path = f'{w5e5_path}/gswp3-w5e5_obsclim_pr_global_monthly_1901_2019.nc'
    tas_path = f'{w5e5_path}/gswp3-w5e5_obsclim_tas_global_monthly_1901_2019.nc'

    if var == "Precipitation":
        ifile = xr.open_dataset(pr_path)
        var_suffix = "PR"
    else:
        ifile = xr.open_dataset(tas_path)
        var_suffix = "TEMP"

    # select only the data for the relevant timeframe
    # base_range = pd.date_range(start='1985-01-01', end='2014-12-31', freq='MS')

    base_range = pd.date_range(start=f'{y0}-01-01', end=f'{ye}-12-31')

    # Use 'isel' method to filter by date range
    baseline = ifile.sel(time=slice(base_range.min(), base_range.max()))

    """Part 1 - Correct the data"""
    # Data is provided in Precipitation flux (kg/m2/s) on a monthly basis

    if var == "Precipitation":

        # Get the number of days in the corresponding month
        dates = pd.to_datetime(baseline.time.values)
        days_in_month_series = pd.Series(
            [monthrange(d.year, d.month)[1] for d in dates])

        # Convert days_in_month_series to xarray DataArray
        days_in_month_da = xr.DataArray(
            days_in_month_series.values,
            coords=[baseline.time],
            dims=['time']
        )

        # Broadcast days_in_month_da to match the shape of baseline
        days_in_month_broadcasted = days_in_month_da.expand_dims(
            {'lat': baseline.lat, 'lon': baseline.lon}, axis=[1, 2])

        # Replace 30 with the actual number of days in the corresponding month
        baseline['pr'] = baseline.pr * 86400 * \
            days_in_month_broadcasted  # to go to monthly total flux
        # print(baseline)
        # baseline = baseline.pr*86400*30 #to go to monthly total flux

    # """Part 2 - Enable time averaging for the climate data (monthly, seasonal or annual)"""

    # reformat the time dimension of the data for resampling
    baseline["time"] = baseline.time.astype("datetime64[ns]")

    # calculate monthly averages (different from precipitation where calculate average monthly totals)
    if timeframe != "monthly":
        if var == "Temperature":
            baseline = baseline.resample(time=time_averaging).mean(dim='time')
        # different approach for precipitation as we are interested in the total amount of precipitation on a seasonal and annual basis
        # the input comes from monthly totals, so we don't need to sum over monthly data anymore
        if var == "Precipitation":
            baseline = baseline.resample(time=time_averaging).sum(dim='time')

    if var == "Temperature":

        baseline.tas.attrs['units'] = "K"
        baseline.tas.attrs['long_name'] = "Temperature 2m, ({timeframe})"
        # print(baseline.tas.attrs)
    if var == "Precipitation":
        baseline.pr.attrs['units'] = "kg/(m2)"
        baseline.pr.attrs['long_name'] = "Precip Total liq+sol, ({timeframe})"
        # print(baseline.pr.attrs)

    # save difference and processed input files:
    base_folder_out = f"/Users/magaliponds/OneDrive - Vrije Universiteit Brussel/1. VUB/02. Coding/01. IRRMIP/03. Data/03. Output files/01. Climate data/01. Processed input data/{var}/{timeframe}/{model}/{member}"
    os.makedirs(base_folder_out, exist_ok=True)

    ofile_baseline = f"{base_folder_out}/{model}.{var_suffix}.BASE.00{member}.{y0}_{ye}_{timeframe}_{diftype}.nc"
    baseline.to_netcdf(ofile_baseline)
    return


# %% Cell 3: Plotting the climate data - only for perturbations

def plot_P_T_perturbations(model, scale, var, timeframe, mode, diftype, plotsave):
    """ Part 0 - Set plotting parameters"""
    # adjust figure sizes towards type of plot
    if scale == "Global":
        if timeframe == 'monthly':
            figsize = (25, 12)
            fig, axes = plt.subplots(nrows=3, ncols=4, subplot_kw={
                                     'projection': ccrs.PlateCarree()}, figsize=figsize)
            time_signature = 'ymon'
            timestamps = ['JAN', 'FEB', 'MAR', 'APR', 'MAY',
                          'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
            time_averaging = 'time.month'
            time_type = 'month'
            col_wrap = 4
        if timeframe == 'seasonal':
            figsize = (12, 7.5)
            fig, axes = plt.subplots(nrows=2, ncols=2, subplot_kw={
                                     'projection': ccrs.PlateCarree()}, figsize=figsize)
            time_signature = 'yseas'
            timestamps = ['DJF', 'MAM', 'JJA', 'SON']
            time_averaging = 'time.season'
            time_type = 'season'
            col_wrap = 2
        if timeframe == 'annual':
            figsize = (7, 5)
            fig, axes = plt.subplots(nrows=1, ncols=1, subplot_kw={
                                     'projection': ccrs.PlateCarree()}, figsize=figsize)
            time_signature = 'year'
            timestamps = ['YEAR']
            time_averaging = 'time.year'
            time_type = 'year'
            col_wrap = 1

    if scale == "Local":
        if timeframe == 'monthly':
            figsize = (18, 10)
            fig, axes = plt.subplots(nrows=3, ncols=4, subplot_kw={
                                     'projection': ccrs.PlateCarree()}, figsize=figsize)
            time_signature = 'ymon'
            timestamps = ['JAN', 'FEB', 'MAR', 'APR', 'MAY',
                          'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
            time_averaging = 'time.month'
            time_type = 'month'
            col_wrap = 4
        if timeframe == 'seasonal':
            figsize = (9, 7)
            fig, axes = plt.subplots(nrows=2, ncols=2, subplot_kw={
                                     'projection': ccrs.PlateCarree()}, figsize=figsize)
            time_signature = 'yseas'
            timestamps = ['DJF', 'MAM', 'JJA', 'SON']
            time_averaging = 'time.season'
            time_type = 'season'
            col_wrap = 2
        if timeframe == 'annual':
            figsize = (7, 5)
            fig, axes = plt.subplots(nrows=1, ncols=1, subplot_kw={
                                     'projection': ccrs.PlateCarree()}, figsize=figsize)
            time_signature = 'year'
            timestamps = ['YEAR']
            time_averaging = 'time.year'
            time_type = 'year'
            col_wrap = 1

    # Provide cbar ranges and colors for plots for different variables, modes (dif/std) and difference types (abs/rel)
    if var == "Precipitation":
        var_suffix = "PR"
        if mode == 'dif' and diftype == 'rel':
            mode_suff = 'total'
            vmin = -40
            vmax = 40
            zero_scaled = (abs(vmin)/(abs(vmin)+abs(vmax)))
            colors = [(0, 'xkcd:mocha'), (zero_scaled,
                                          'xkcd:white'), (1, 'xkcd:aquamarine')]
            custom_cmap = LinearSegmentedColormap.from_list(
                'custom_cmap', colors)
        if mode == 'dif' and diftype == 'abs':
            mode_suff = 'total'
            vmin = -50
            vmax = 75
            zero_scaled = (abs(vmin)/(abs(vmin)+abs(vmax)))
            colors = [(0, 'xkcd:mocha'), (zero_scaled,
                                          'xkcd:white'), (1, 'xkcd:aquamarine')]
            custom_cmap = LinearSegmentedColormap.from_list(
                'custom_cmap', colors)
        if mode == 'std':
            mode_suff = 'std'
            vmin = -40
            vmax = 40
            zero_scaled = (abs(vmin)/(abs(vmin)+abs(vmax)))
            colors = [(0, 'xkcd:peach'), (zero_scaled, 'xkcd:white'),
                      (1, 'xkcd:light aquamarine')]
            custom_cmap = LinearSegmentedColormap.from_list(
                'custom_cmap', colors)

    elif var == "Temperature":
        var_suffix = "TEMP"
        if mode == 'dif':
            mode_suff = 'total'
            vmin = -1.5
            vmax = 1.5
            zero_scaled = (abs(vmin)/(abs(vmin)+abs(vmax)))
            colors = [(0, 'cornflowerblue'), (zero_scaled,
                                              'xkcd:white'), (1, 'xkcd:tomato')]
            custom_cmap = LinearSegmentedColormap.from_list(
                'custom_cmap', colors)
        if mode == 'std':
            mode_suff = 'std'
            vmin = -1.5
            vmax = 1.5
            zero_scaled = (abs(vmin)/(abs(vmin)+abs(vmax)))
            colors = [(0, 'xkcd:lightblue'), (zero_scaled,
                                              'xkcd:white'), (1, 'xkcd:pink')]
            custom_cmap = LinearSegmentedColormap.from_list(
                'custom_cmap', colors)

    """ Part 1: Load the input data (derived in function process_P_T_perturbations) """

    base_folder_in = f"/Users/magaliponds/OneDrive - Vrije Universiteit Brussel/1. VUB/02. Coding/01. IRRMIP/03. Data/03. Output files/01. Climate data/01. Processed input data/{var}/{timeframe}/{model}/{member}"
    diff_folder_in = f"/Users/magaliponds/OneDrive - Vrije Universiteit Brussel/1. VUB/02. Coding/01. IRRMIP/03. Data/03. Output files/01. Climate data/02. Perturbations/{var}/{timeframe}/{model}/{member}"

    ifile_diff = f"{diff_folder_in}/{model}.{var_suffix}.DIF.00{member}.1985_2014_{timeframe}_{diftype}.nc"
    ifile_diff = f"{diff_folder_in}/{model}.{var_suffix}.DIF.00{member}.1985_2014_{timeframe}_{diftype}.nc"
    ifile_irr = f"{base_folder_in}/{model}.{var_suffix}.IRR.00{member}.1985_2014_{timeframe}_{diftype}.nc"
    ifile_noi = f"{base_folder_in}/{model}.{var_suffix}.NOI.00{member}.1985_2014_{timeframe}_{diftype}.nc"

    diff = xr.open_dataset(ifile_diff)
    irrigation = xr.open_dataarray(ifile_irr)
    baseline = xr.open_dataarray(ifile_noi)

    if var == "Temperature":
        irrigation = irrigation-273.15
        baseline = baseline-273.15

    if scale == "Local":
        diff = diff.where((diff.lon >= 60) & (diff.lon <= 109) & (
            diff.lat >= 22) & (diff.lat <= 52), drop=True)
    # create local minima/maxima, for axis of plot
    local_min_diff = diff.quantile(0.25)
    local_max_diff = diff.quantile(0.75)

    """ Part 2 - Shapefile outline for Karakoram Area to be included"""
    # path to  shapefile
    shapefile_path = '/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/01. Input files/03. Shapefile/Karakoram/Pan-Tibetan Highlands/Pan-Tibetan Highlands (Liu et al._2022)/Shapefile/Pan-Tibetan Highlands (Liu et al._2022)_P.shp'
    shp = gpd.read_file(shapefile_path)
    target_crs = 'EPSG:4326'
    shp = shp.to_crs(target_crs)

    """ Part 3 - Create subplots for IRR, NOI and DIF"""

    # first create output folders for the data
    o_folder_base = f"/Users/magaliponds/OneDrive - Vrije Universiteit Brussel/1. VUB/02. Coding/01. IRRMIP/04. Figures/01. Climate data/01. Input data/{scale}/{var}/{model}/{member}"
    o_folder_diff = f"/Users/magaliponds/OneDrive - Vrije Universiteit Brussel/1. VUB/02. Coding/01. IRRMIP/04. Figures/01. Climate data/02. Perturbations/{scale}/{var}/{model}/{member}"

    for time_idx, timestamp_name in enumerate(timestamps):

        # indicate the column and row of the subplot to plot in
        if timeframe == 'monthly':
            row = (time_idx) // 4  # Calculate row index
            col = (time_idx) % 4
            ax = axes[row, col]
        if timeframe == 'seasonal':
            row = (time_idx) // 2
            col = (time_idx) % 2
            ax = axes[row, col]
        if timeframe == 'annual':
            row = 0
            col = 0
            ax = axes

        """ 3A Plotting data, incl karakoram outline """
        # select relevant month/season and only 1 year for annual to plot and annotate
        if scale == "Local":
            time_dim_name = list(diff.dims)[0]
        if scale == "Global":
            if timeframe != 'annual':
                time_dim_name = list(diff.dims)[2]

        if timeframe == 'annual':
            diff_sel = diff
        else:
            diff_sel = diff.isel({time_dim_name: time_idx})

        # make into dataframe, else it doesnt work
        if isinstance(diff_sel, xr.Dataset):
            diff_sel = diff_sel[list(diff_sel.data_vars.keys())[0]]

        # plot the data incl the outline of the karakoram shapefile, setting the colors, but excluding the shapefile
        im = diff_sel.plot.imshow(ax=ax, vmin=vmin, vmax=vmax, extend='both',
                                  transform=ccrs.PlateCarree(), cmap=custom_cmap, add_colorbar=False)
        shp.plot(ax=ax, edgecolor='black', linewidth=1, facecolor='none')
        # shp.plot(ax=ax, edgecolor='red', linewidth=1, facecolor='none')

        ax.coastlines(resolution='10m')

        # include month as a label, instead as on top off data
        ax.set_title('')
        ax.annotate(timestamp_name, xy=(1, 1), xytext=(-10, -10), xycoords='axes fraction', textcoords='offset points',
                    ha='right', va='top', fontsize=15, bbox=dict(boxstyle='square', fc='white', alpha=1))

        """ 3B - Min and Max value annotation"""
        # Include annotation for min and max values in every subplot, excluding NaN from min/max creation
        diff_sel_min = diff_sel.fillna(diff_sel.max())
        diff_sel_max = diff_sel.fillna(diff_sel.min())

        # find min and max values in gridcell
        min_value_index = np.unravel_index(
            np.argmin(diff_sel_min.values), diff_sel_min.shape)
        max_value_index = np.unravel_index(
            np.argmax(diff_sel_max.values), diff_sel_max.shape)

        # Extract longitude and latitude corresponding to the minimum and maximum value indices
        min_lon, min_lat = diff_sel.lon.values[min_value_index[1]
                                               ], diff_sel.lat.values[min_value_index[0]]
        max_lon, max_lat = diff_sel.lon.values[max_value_index[1]
                                               ], diff_sel.lat.values[max_value_index[0]]

        # Plot the dot on the subplot
        ax.plot(min_lon, min_lat, marker='o', markersize=7,
                color='blue')  # Adjust marker properties as needed
        ax.plot(max_lon, max_lat, marker='o', markersize=7,
                color='red')  # Adjust marker properties as needed

        # #indicate annotations for min and max values in plot, formatted as percenteges for rel precipitation differences
        min_value = diff_sel[min_value_index]
        max_value = diff_sel[max_value_index]

        if scale == "Local":
            if timeframe == 'annual':
                ax.annotate(f'Min: {min_value:.1f}', xy=(65, 50), xytext=(
                    65, 50), fontsize=15, ha='left', va='top')
                ax.annotate(f'Max: {max_value:.1f}', xy=(78, 50), xytext=(
                    78, 50), fontsize=15, ha='left', va='top')

                ax.plot(64, 49.4, marker='o', color='blue', markersize=5)
                ax.plot(77, 49.4, marker='o', color='red', markersize=5)
            else:
                ax.annotate(f'Min: {min_value:.1f}', xy=(64, 50), xytext=(
                    64, 50), fontsize=14, ha='left', va='top')
                ax.annotate(f'Max: {max_value:.1f}', xy=(84, 50), xytext=(
                    84, 50), fontsize=14, ha='left', va='top')

                ax.plot(62, 49, marker='o', color='blue', markersize=5)
                ax.plot(82, 49, marker='o', color='red', markersize=5)
        if scale == "Global":
            if timeframe == 'monthly':
                ax.annotate(f'Min: {min_value:.1f}', xy=(
                    0, -75), xytext=(0, -75), fontsize=15, ha='left', va='top')
                ax.annotate(f'Max: {max_value:.1f}', xy=(
                    85, -75), xytext=(85, -75), fontsize=15, ha='left', va='top')
                ax.plot(-4, -78, marker='o', color='blue', markersize=5)
                ax.plot(80, -78, marker='o', color='red', markersize=5)
            else:
                ax.annotate(f'Min: {min_value:.1f}', xy=(
                    0, -75), xytext=(0, -75), fontsize=15, ha='left', va='top')
                ax.annotate(f'Max: {max_value:.1f}', xy=(
                    85, -75), xytext=(85, -75), fontsize=15, ha='left', va='top')
                ax.plot(-4, -79, marker='o', color='blue', markersize=5)
                ax.plot(80, -79, marker='o', color='red', markersize=5)

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

        if (timeframe == 'monthly' and row == 2) or (timeframe == 'seasonal' and row == 1) or (timeframe == 'annual'):
            gl.xlabel_style = {'size': 15}
            gl.xlocator = plt.MaxNLocator(nbins=3)

        else:
            gl.bottom_labels = False

        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

    """ 3C Add color bar for entire plot"""
    # add cbar in the figure, for overall figure, not subplots
    # Define the position of the colorbar
    cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(im, cax=cbar_ax, extend='both')

    # adjust subplot spacing to be smaller
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1,
                        top=0.9, wspace=0.05, hspace=0.05)

    # Set label and tick parameters for the colorbar
    if var == "Precipitation":
        if mode == 'std' or diftype == 'abs':
            unit = 'mm'
        else:
            unit = '%'
    elif var == "Temperature":
        unit = '°C'
    else:
        unit = 'Unknown'

    """4 Include labels for the cbar and for the y and x axis"""
    # Increase distance between colorbar label and colorbar
    cbar.ax.yaxis.labelpad = 20
    if mode == 'dif':
        cbar.set_label(f'$\Delta$ {var} [{unit}]', size='15')
        if mode == 'std':
            cbar.set_label(f'{var} - model member std [{unit}]', size='15')
    cbar.ax.tick_params(labelsize=15)

    if timeframe == 'monthly':
        fig.text(0.5, 0.03, 'Longitude', ha='center', fontsize=15)
        fig.text(0.03, 0.5, 'Latitude', va='center',
                 rotation='vertical', fontsize=15)
        fig.text(0.5, 0.92, model, ha='center', fontsize=20)
    else:
        fig.text(0.5, 0.01, 'Longitude', ha='center', fontsize=15)
        fig.text(-0.02, 0.5, 'Latitude', va='center',
                 rotation='vertical', fontsize=15)
        fig.text(0.5, 0.92, model, ha='center', fontsize=20)

    if plotsave == 'save':
        # os.makedirs(f"o_folder_base/{scale}/{timeframe}/{var}/", exist_ok=True)
        os.makedirs(f"{o_folder_diff}/", exist_ok=True)
        o_file_name = f"{o_folder_diff}/{model}.{var_suffix}.DIF.00{member}.1985_2014_{timeframe}_{diftype}.png"
        plt.savefig(o_file_name, bbox_inches='tight')
    plt.show()
    return

# %% Cell 4: Plotting the climate data - for perturbations and input data


def plot_P_T_input_perturbations(plotvar, model, scale, var, timeframe, mode, diftype, plotsave):
    """ Part 0 - Set plotting parameters"""
    # adjust figure sizes towards type of plot
    if scale == "Global":
        if timeframe == 'monthly':
            figsize = (25, 12)
            fig, axes = plt.subplots(nrows=3, ncols=4, subplot_kw={
                                     'projection': ccrs.PlateCarree()}, figsize=figsize)
            time_signature = 'ymon'
            timestamps = ['JAN', 'FEB', 'MAR', 'APR', 'MAY',
                          'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
            time_averaging = 'time.month'
            time_type = 'month'
            col_wrap = 4
        if timeframe == 'seasonal':
            figsize = (12, 7.5)
            fig, axes = plt.subplots(nrows=2, ncols=2, subplot_kw={
                                     'projection': ccrs.PlateCarree()}, figsize=figsize)
            time_signature = 'yseas'
            timestamps = ['DJF', 'MAM', 'JJA', 'SON']
            time_averaging = 'time.season'
            time_type = 'season'
            col_wrap = 2
        if timeframe == 'annual':
            figsize = (7, 5)
            fig, axes = plt.subplots(nrows=1, ncols=1, subplot_kw={
                                     'projection': ccrs.PlateCarree()}, figsize=figsize)
            time_signature = 'year'
            timestamps = ['YEAR']
            time_averaging = 'time.year'
            time_type = 'year'
            col_wrap = 1

    if scale == "Local":
        if timeframe == 'monthly':
            figsize = (18, 10)
            fig, axes = plt.subplots(nrows=3, ncols=4, subplot_kw={
                                     'projection': ccrs.PlateCarree()}, figsize=figsize)
            time_signature = 'ymon'
            timestamps = ['JAN', 'FEB', 'MAR', 'APR', 'MAY',
                          'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
            time_averaging = 'time.month'
            time_type = 'month'
            col_wrap = 4
        if timeframe == 'seasonal':
            figsize = (9, 7)
            fig, axes = plt.subplots(nrows=2, ncols=2, subplot_kw={
                                     'projection': ccrs.PlateCarree()}, figsize=figsize)
            time_signature = 'yseas'
            timestamps = ['DJF', 'MAM', 'JJA', 'SON']
            time_averaging = 'time.season'
            time_type = 'season'
            col_wrap = 2
        if timeframe == 'annual':
            figsize = (7, 5)
            fig, axes = plt.subplots(nrows=1, ncols=1, subplot_kw={
                                     'projection': ccrs.PlateCarree()}, figsize=figsize)
            time_signature = 'year'
            timestamps = ['YEAR']
            time_averaging = 'time.year'
            time_type = 'year'
            col_wrap = 1

    # Provide cbar ranges and colors for plots for different variables, modes (dif/std) and difference types (abs/rel)
    if var == "Precipitation":
        var_suffix = "PR"
        if plotvar != "DIF":
            vmin = 0
            vmax = 150
        if mode == 'dif' and diftype == 'rel':
            mode_suff = 'total'
            vmin = -40
            vmax = 40
            zero_scaled = (abs(vmin)/(abs(vmin)+abs(vmax)))
            colors = [(0, 'xkcd:mocha'), (zero_scaled,
                                          'xkcd:white'), (1, 'xkcd:aquamarine')]
            custom_cmap = LinearSegmentedColormap.from_list(
                'custom_cmap', colors)
        if mode == 'dif' and diftype == 'abs':
            mode_suff = 'total'
            vmin = -50
            vmax = 75
            zero_scaled = (abs(vmin)/(abs(vmin)+abs(vmax)))
            colors = [(0, 'xkcd:mocha'), (zero_scaled,
                                          'xkcd:white'), (1, 'xkcd:aquamarine')]
            custom_cmap = LinearSegmentedColormap.from_list(
                'custom_cmap', colors)
        if mode == 'std':
            mode_suff = 'std'
            vmin = -40
            vmax = 40
            zero_scaled = (abs(vmin)/(abs(vmin)+abs(vmax)))
            colors = [(0, 'xkcd:peach'), (zero_scaled, 'xkcd:white'),
                      (1, 'xkcd:light aquamarine')]
            custom_cmap = LinearSegmentedColormap.from_list(
                'custom_cmap', colors)

    elif var == "Temperature":
        var_suffix = "TEMP"
        if plotvar == "DIF":
            vmin = -1.5
            vmax = 1.5
        else:
            vmin = 240
            vmax = 300

        if mode == 'dif':
            mode_suff = 'total'
            zero_scaled = (abs(vmin)/(abs(vmin)+abs(vmax)))
            colors = [(0, 'cornflowerblue'), (zero_scaled,
                                              'xkcd:white'), (1, 'xkcd:tomato')]
            custom_cmap = LinearSegmentedColormap.from_list(
                'custom_cmap', colors)
        if mode == 'std':
            mode_suff = 'std'
            zero_scaled = (abs(vmin)/(abs(vmin)+abs(vmax)))
            colors = [(0, 'xkcd:lightblue'), (zero_scaled,
                                              'xkcd:white'), (1, 'xkcd:pink')]
            custom_cmap = LinearSegmentedColormap.from_list(
                'custom_cmap', colors)

    """ Part 1: Load the input data (derived in function process_P_T_perturbations) """

    base_folder_in = f"/Users/magaliponds/OneDrive - Vrije Universiteit Brussel/1. VUB/02. Coding/01. IRRMIP/03. Data/03. Output files/01. Climate data/01. Processed input data/{var}/{timeframe}/{model}/{member}"
    diff_folder_in = f"/Users/magaliponds/OneDrive - Vrije Universiteit Brussel/1. VUB/02. Coding/01. IRRMIP/03. Data/03. Output files/01. Climate data/02. Perturbations/{var}/{timeframe}/{model}/{member}"

    if plotvar == "DIF":
        ifile_diff = f"{diff_folder_in}/{model}.{var_suffix}.{plotvar}.00{member}.1985_2014_{timeframe}_{diftype}.nc"
    else:
        ifile_diff = f"{base_folder_in}/{model}.{var_suffix}.{plotvar}.00{member}.1985_2014_{timeframe}_{diftype}.nc"

    # ifile_irr=f"{base_folder_in}/{model}.{var_suffix}.IRR.00{member}.1985_2014_{timeframe}_{mode_suff}_{diftype}.nc"
    # ifile_noi=f"{base_folder_in}/{model}.{var_suffix}.NOI.00{member}.1985_2014_{timeframe}_{mode_suff}_{diftype}.nc"

    diff = xr.open_dataset(ifile_diff)
    # irrigation =  xr.open_dataarray(ifile_irr)
    # baseline =  xr.open_dataarray(ifile_noi)

    if scale == "Local":
        diff = diff.where((diff.lon >= 60) & (diff.lon <= 109) & (
            diff.lat >= 22) & (diff.lat <= 52), drop=True)
    # create local minima/maxima, for axis of plot
    local_min_diff = diff.quantile(0.25)
    local_max_diff = diff.quantile(0.75)

    """ Part 2 - Shapefile outline for Karakoram Area to be included"""
    # path to  shapefile
    shapefile_path = '/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/01. Input files/03. Shapefile/Karakoram/Pan-Tibetan Highlands/Pan-Tibetan Highlands (Liu et al._2022)/Shapefile/Pan-Tibetan Highlands (Liu et al._2022)_P.shp'
    shp = gpd.read_file(shapefile_path)
    target_crs = 'EPSG:4326'
    shp = shp.to_crs(target_crs)

    """ Part 3 - Create subplots for IRR, NOI and DIF"""

    # first create output folders for the data
    if plotvar == "DIF":
        o_folder_diff = f"/Users/magaliponds/OneDrive - Vrije Universiteit Brussel/1. VUB/02. Coding/01. IRRMIP/04. Figures/01. Climate data/02. Perturbations/{scale}/{var}/{model}/{member}"
    else:
        o_folder_diff = f"/Users/magaliponds/OneDrive - Vrije Universiteit Brussel/1. VUB/02. Coding/01. IRRMIP/04. Figures/01. Climate data/01. Input data/{scale}/{var}/{model}/{member}/{plotvar}"

    for time_idx, timestamp_name in enumerate(timestamps):

        # indicate the column and row of the subplot to plot in
        if timeframe == 'monthly':
            row = (time_idx) // 4  # Calculate row index
            col = (time_idx) % 4
            ax = axes[row, col]
        if timeframe == 'seasonal':
            row = (time_idx) // 2
            col = (time_idx) % 2
            ax = axes[row, col]
        if timeframe == 'annual':
            row = 0
            col = 0
            ax = axes

        """ 3A Plotting data, incl karakoram outline """
        # select relevant month/season and only 1 year for annual to plot and annotate
        if scale == "Local":
            time_dim_name = list(diff.dims)[0]
        if scale == "Global":
            if timeframe != 'annual':
                time_dim_name = list(diff.dims)[2]

        if timeframe == 'annual':
            diff_sel = diff
        else:
            diff_sel = diff.isel({time_dim_name: time_idx})

        # make into dataframe, else it doesnt work
        if isinstance(diff_sel, xr.Dataset):
            diff_sel = diff_sel[list(diff_sel.data_vars.keys())[0]]

        # plot the data incl the outline of the karakoram shapefile, setting the colors, but excluding the shapefile
        im = diff_sel.plot.imshow(ax=ax, vmin=vmin, vmax=vmax, extend='both',
                                  transform=ccrs.PlateCarree(), cmap=custom_cmap, add_colorbar=False)
        shp.plot(ax=ax, edgecolor='black', linewidth=1, facecolor='none')
        # shp.plot(ax=ax, edgecolor='red', linewidth=1, facecolor='none')

        ax.coastlines(resolution='10m')

        # include month as a label, instead as on top off data
        ax.set_title('')
        ax.annotate(timestamp_name, xy=(1, 1), xytext=(-10, -10), xycoords='axes fraction', textcoords='offset points',
                    ha='right', va='top', fontsize=15, bbox=dict(boxstyle='square', fc='white', alpha=1))

        """ 3B - Min and Max value annotation"""
        # Include annotation for min and max values in every subplot, excluding NaN from min/max creation
        diff_sel_min = diff_sel.fillna(diff_sel.max())
        diff_sel_max = diff_sel.fillna(diff_sel.min())

        # find min and max values in gridcell
        min_value_index = np.unravel_index(
            np.argmin(diff_sel_min.values), diff_sel_min.shape)
        max_value_index = np.unravel_index(
            np.argmax(diff_sel_max.values), diff_sel_max.shape)

        # Extract longitude and latitude corresponding to the minimum and maximum value indices
        min_lon, min_lat = diff_sel.lon.values[min_value_index[1]
                                               ], diff_sel.lat.values[min_value_index[0]]
        max_lon, max_lat = diff_sel.lon.values[max_value_index[1]
                                               ], diff_sel.lat.values[max_value_index[0]]

        # Plot the dot on the subplot
        ax.plot(min_lon, min_lat, marker='o', markersize=7,
                color='blue')  # Adjust marker properties as needed
        ax.plot(max_lon, max_lat, marker='o', markersize=7,
                color='red')  # Adjust marker properties as needed

        # #indicate annotations for min and max values in plot, formatted as percenteges for rel precipitation differences
        min_value = diff_sel[min_value_index]
        max_value = diff_sel[max_value_index]

        if scale == "Local":
            if timeframe == 'annual':
                ax.annotate(f'Min: {min_value:.1f}', xy=(65, 50), xytext=(
                    65, 50), fontsize=15, ha='left', va='top')
                ax.annotate(f'Max: {max_value:.1f}', xy=(78, 50), xytext=(
                    78, 50), fontsize=15, ha='left', va='top')

                ax.plot(64, 49.4, marker='o', color='blue', markersize=5)
                ax.plot(77, 49.4, marker='o', color='red', markersize=5)
            else:
                ax.annotate(f'Min: {min_value:.1f}', xy=(64, 50), xytext=(
                    64, 50), fontsize=14, ha='left', va='top')
                ax.annotate(f'Max: {max_value:.1f}', xy=(84, 50), xytext=(
                    84, 50), fontsize=14, ha='left', va='top')

                ax.plot(62, 49, marker='o', color='blue', markersize=5)
                ax.plot(82, 49, marker='o', color='red', markersize=5)
        if scale == "Global":
            if timeframe == 'monthly':
                ax.annotate(f'Min: {min_value:.1f}', xy=(
                    0, -75), xytext=(0, -75), fontsize=15, ha='left', va='top')
                ax.annotate(f'Max: {max_value:.1f}', xy=(
                    85, -75), xytext=(85, -75), fontsize=15, ha='left', va='top')
                ax.plot(-4, -78, marker='o', color='blue', markersize=5)
                ax.plot(80, -78, marker='o', color='red', markersize=5)
            else:
                ax.annotate(f'Min: {min_value:.1f}', xy=(
                    0, -75), xytext=(0, -75), fontsize=15, ha='left', va='top')
                ax.annotate(f'Max: {max_value:.1f}', xy=(
                    85, -75), xytext=(85, -75), fontsize=15, ha='left', va='top')
                ax.plot(-4, -79, marker='o', color='blue', markersize=5)
                ax.plot(80, -79, marker='o', color='red', markersize=5)

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

        if (timeframe == 'monthly' and row == 2) or (timeframe == 'seasonal' and row == 1) or (timeframe == 'annual'):
            gl.xlabel_style = {'size': 15}
            gl.xlocator = plt.MaxNLocator(nbins=3)

        else:
            gl.bottom_labels = False

        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

    """ 3C Add color bar for entire plot"""
    # add cbar in the figure, for overall figure, not subplots
    # Define the position of the colorbar
    cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(im, cax=cbar_ax, extend='both')

    # adjust subplot spacing to be smaller
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1,
                        top=0.9, wspace=0.05, hspace=0.05)

    # Set label and tick parameters for the colorbar
    if var == "Precipitation":
        if mode == 'std' or diftype == 'abs':
            unit = 'mm'
        else:
            unit = '%'
    elif var == "Temperature":
        unit = '°C'
    else:
        unit = 'Unknown'

    """4 Include labels for the cbar and for the y and x axis"""
    # Increase distance between colorbar label and colorbar
    cbar.ax.yaxis.labelpad = 20
    if mode == 'dif':
        cbar.set_label(f'$\Delta$ {var} [{unit}]', size='15')
        if mode == 'std':
            cbar.set_label(f'{var} - model member std [{unit}]', size='15')
    cbar.ax.tick_params(labelsize=15)

    if timeframe == 'monthly':
        fig.text(0.5, 0.03, 'Longitude', ha='center', fontsize=15)
        fig.text(0.03, 0.5, 'Latitude', va='center',
                 rotation='vertical', fontsize=15)
        fig.text(0.5, 0.92, model, ha='center', fontsize=20)
    else:
        fig.text(0.5, 0.01, 'Longitude', ha='center', fontsize=15)
        fig.text(-0.02, 0.5, 'Latitude', va='center',
                 rotation='vertical', fontsize=15)
        fig.text(0.5, 0.92, model, ha='center', fontsize=20)
    if plotvar != "DIF":
        fig.text((figsize[0]+0.5)/figsize[0], (figsize[1]-0.5)/figsize[1], str(
            plotvar), ha='center', rotation='horizontal', color='red', fontsize=20)

    if plotsave == 'save':
        if plotvar == "DIF":
            # os.makedirs(f"o_folder_base/{scale}/{timeframe}/{var}/", exist_ok=True)
            os.makedirs(f"{o_folder_diff}/", exist_ok=True)
            o_file_name = f"{o_folder_diff}/{model}.{var_suffix}.{plotvar}.00{member}.1985_2014_{timeframe}_{mode_suff}_{diftype}.png"
            plt.savefig(o_file_name, bbox_inches='tight')
        else:
            os.makedirs(f"{o_folder_diff}/", exist_ok=True)
            o_file_name = f"{o_folder_diff}/{model}.{var_suffix}.{plotvar}.00{member}.1985_2014_{timeframe}_{mode_suff}_{diftype}.png"
            plt.savefig(o_file_name, bbox_inches='tight')
    plt.show()
    return


# %% Cell 5: Run the perturbation processing for all climate models, members etc.
members = [1, 3, 4, 6]
# members = [1, 1, 1, 1]

for (m, model) in enumerate(["IPSL-CM6", "E3SM", "CESM2", "CNRM"]):
    for member in range(members[m]):
        for var in ["Precipitation"]:  # "Temperature"]:  # ,"Temperature"]:
            for timeframe in ["annual", "seasonal", "monthly"]:
                for mode in ['dif']:  # , 'std']:
                    if var == "Precipitation" and mode == 'dif':
                        diftypes = ['abs', 'rel']
                    else:
                        diftypes = ['abs']
                    for dif in diftypes:
                        print(var, model, member, timeframe, dif)
                        process_P_T_perturbations(
                            model, member, var, timeframe, mode, dif)


# %% Cell 6: Run the baseline processing

ye = [2014, 2020]
for model in ["W5E5"]:

    for member in [0]:
        for var in ["Precipitation", "Temperature"]:
            for timeframe in ["annual", "seasonal", "monthly"]:
                for diftype in ['abs']:

                    for (y, y0) in enumerate([1985, 1901]):
                        ye_sel = ye[y]

                        print(var, timeframe, y0, ye_sel)
                        process_P_T_baseline(
                            model, member, var, timeframe, diftype, y0, ye_sel)


# %% Cell 5: Run the functions for all different combinations to generate output datasets and plots
members = [1, 3, 4, 6]

for (m, model) in enumerate(["IPSL-CM6", "E3SM", "CESM2", "CNRM"]):
    for member in range(members[m]):
        print(member)
        for scale in ["Local"]:  # ,"Global"]:
            for plotvar in ["DIF"]:  # "IRR", "NOI"]:#,"DIF"]:
                # "Temperature", "Precipitation"]:
                for var in ["Precipitation"]:
                    for timeframe in ["annual", "seasonal", "monthly"]:
                        for mode in ['dif']:  # , 'std']:

                            if plotvar == "DIF" and var == "Precipitation" and mode == 'dif':
                                diftypes = ["rel"]  # abs','rel']
                            else:
                                diftypes = ["abs"]

                            for dif in diftypes:
                                print(dif)
                                plot_P_T_perturbations(
                                    model, scale, var, timeframe, mode, dif, "save")
                                # plot_P_T_input_perturbations(plotvar, model, scale, var, timeframe, mode, dif,"save")


# %%% TEST
def process_P_T_perturbations(model, member, var, timeframe, mode, diftype):

    if mode == 'dif':
        mode_suff = 'total'
    if mode == 'std':
        mode_suff = 'std'

    # Enable variability for differnt timescales in: amount of subplots, time-averaging used
    if timeframe == 'monthly':
        time_averaging = 'time.month'
    if timeframe == 'seasonal':
        time_averaging = 'time.season'
    if timeframe == 'annual':
        time_averaging = 'time.year'

    """ Part 0 - Load and open the climate data"""
    # paths to climate data
    folder_in = f"/Users/magaliponds/OneDrive - Vrije Universiteit Brussel/1. VUB/02. Coding/01. IRRMIP/03. Data/01. Input files/01. Climate data/{model}/"
    ifile_IRR = f"{model}.IRR.00{member}.1985_2014_selparam_monthly_{mode_suff}.nc"
    ifile_NOI = f"{model}.NOI.00{member}.1985_2014_selparam_monthly_{mode_suff}.nc"
    # first for each variable load the data, for precipitation this consists of Snow & Rain from atmosphere, converted from [mm/day]
    ifile_IRR = xr.open_dataset(folder_in+ifile_IRR)
    ifile_NOI = xr.open_dataset(folder_in+ifile_NOI)

    print(model, ifile_IRR.pr.units)
    print(model, ifile_IRR.pr.units)


for (m, model) in enumerate(["IPSL-CM6", "E3SM", "CESM2", "CNRM"]):
    for member in range(members[m]):
        for var in ["Precipitation"]:  # ,"Temperature"]:
            for timeframe in ["annual", "seasonal", "monthly"]:
                for mode in ['dif']:  # , 'std']:
                    if var == "Precipitation" and mode == 'dif':
                        diftypes = ['abs', 'rel']
                    else:
                        diftypes = ['abs']
                    for dif in diftypes:
                        print(model, member, timeframe, dif)
                        process_P_T_perturbations(
                            model, member, var, timeframe, mode, dif)
