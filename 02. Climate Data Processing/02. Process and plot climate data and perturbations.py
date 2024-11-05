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


# %%

import os
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.crs as ccrs
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import geopandas as gpd
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import xarray as xr
import calendar
from calendar import monthrange
import pandas as pd
diff_folder_in = f"/Users/magaliponds/OneDrive - Vrije Universiteit Brussel/1. VUB/02. Coding/01. IRRMIP/03. Data/03. Output files/01. Climate data/02. Perturbations/Precipitation/monthly/CESM2/0/"
ifile_diff = f"{diff_folder_in}/CESM2.PR.DIF.000.1985_2014_monthly_abs.nc"
ds = xr.open_dataset(ifile_diff)
# %% Cell 1: Process perturbations for IRR-NOI comparison


def process_P_T_perturbations(model, member, var, timeframe, mode, diftype, y0, ye):

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
    folder_in = f"/Users/magaliponds/OneDrive - Vrije Universiteit Brussel/1. VUB/02. Coding/01. IRRMIP/03. Data/01. Input files/01. Climate data/{model}/{y0}/"
    ifile_IRR = f"{model}.IRR.00{member}.{y0}_{ye}_selparam_monthly_{mode_suff}.nc"
    ifile_NOI = f"{model}.NOI.00{member}.{y0}_{ye}_selparam_monthly_{mode_suff}.nc"
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

    ofile_diff = f"{diff_folder_out}/{model}.{var_suffix}.DIF.00{member}.{y0}_{ye}_{timeframe}_{diftype}.nc"
    ofile_irr = f"{base_folder_out}/{model}.{var_suffix}.IRR.00{member}.{y0}_{ye}_{timeframe}_{diftype}.nc"
    ofile_noi = f"{base_folder_out}/{model}.{var_suffix}.NOI.00{member}.{y0}_{ye}_{timeframe}_{diftype}.nc"
    print(ofile_diff)

    diff.to_netcdf(ofile_diff)
    irrigation.to_netcdf(ofile_irr)
    baseline.to_netcdf(ofile_noi)

    return

# %% Cell 2: Process perturbations for IRR-counterfactual comparison


def process_P_T_perturbations_counterfactual(model, member, var, timeframe, mode, diftype):

    y0_cf = 1901
    ye_cf = 1930
    y0 = 1985
    ye = 2014
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
    ifile_IRR = f"{y0}/{model}.IRR.00{member}.{y0}_{ye}_selparam_monthly_{mode_suff}.nc"
    ifile_IRR_cf = f"{y0_cf}/{model}.IRR.00{member}.{y0_cf}_{ye_cf}_selparam_monthly_{mode_suff}.nc"
    # ifile_NOI = f"{model}.NOI.00{member}.{y0}_{ye}_selparam_monthly_{mode_suff}.nc"
    # first for each variable load the data, for precipitation this consists of Snow & Rain from atmosphere, converted from [mm/day]
    ifile_IRR = xr.open_dataset(folder_in+ifile_IRR)
    ifile_IRR_cf = xr.open_dataset(folder_in+ifile_IRR_cf)
    # ifile_NOI = xr.open_dataset(folder_in+ifile_NOI)

    """Part 1 - Correct the data"""
    # only for model IPSL-CM6 change dimensions of time (rename), as not possible in bash
    if model == "IPSL-CM6":
        ifile_IRR = ifile_IRR.rename({'time_counter': 'time'})
        ifile_IRR_cf = ifile_IRR_cf.rename({'time_counter': 'time'})

    if var == "Precipitation":

        baseline_R = ifile_IRR_cf.pr*86400
        irrigation_R = ifile_IRR.pr*86400
        if model in ["CESM2", "E3SM"]:
            baseline_S = ifile_IRR_cf.sn*86400
            baseline = baseline_R+baseline_S
            irrigation_S = ifile_IRR.sn*86400
            irrigation = irrigation_R+irrigation_S

            days_in_month_da_base = baseline.time.dt.days_in_month
            days_in_month_broadcasted_base = days_in_month_da_base.broadcast_like(
                baseline)

            days_in_month_da = irrigation.time.dt.days_in_month
            days_in_month_broadcasted = days_in_month_da.broadcast_like(
                baseline)
            irrigation = irrigation*days_in_month_broadcasted
            baseline = baseline*days_in_month_broadcasted_base
            # rename the variables in the dataset
            if irrigation.name == None:
                irrigation.name = 'pr'
            if baseline.name == None:
                baseline.name = 'pr'

        else:
            baseline = baseline_R
            irrigation = irrigation_R
    elif var == "Temperature":
        baseline = ifile_IRR_cf.tas
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

    ofile_diff = f"{diff_folder_out}/{model}.{var_suffix}.DIF.00{member}.{y0_cf}_{y0}_{timeframe}_{diftype}_counterfactual.nc"
    ofile_irr = f"{base_folder_out}/{model}.{var_suffix}.IRR.00{member}.{y0_cf}_{y0}_{timeframe}_{diftype}_counterfactual.nc"
    ofile_noi = f"{base_folder_out}/{model}.{var_suffix}.NOI.00{member}.{y0_cf}_{y0}_{timeframe}_{diftype}_counterfactual.nc"

    diff.to_netcdf(ofile_diff)
    irrigation.to_netcdf(ofile_irr)
    baseline.to_netcdf(ofile_noi)

    return

# %% Cell 3: Process baseline


def process_P_T_baseline(model, member, var, timeframe, diftype, y0, ye):

    if timeframe == 'seasonal':
        time_averaging = "QS-DEC"
    if timeframe == 'annual':
        time_averaging = 'YE'

    """ Part 0 - Load and open the climate data"""
    # paths to climate data
    if model == "W5E5":
        w5e5_path = '/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/01. Input files/01. Climate data/W5E5/cluster.klima.uni-bremen.de/~oggm/climate/gswp3-w5e5/unflattened/2023.2/monthly'
        pr_path = f'{w5e5_path}/gswp3-w5e5_obsclim_pr_global_monthly_1901_2019.nc'
        tas_path = f'{w5e5_path}/gswp3-w5e5_obsclim_tas_global_monthly_1901_2019.nc'
    if model == "CRU":
        cru_path = '/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/01. Input files/01. Climate data/CRU/'
        pr_path = f"{cru_path}/cru_ts4.08.1901.2023.pre.dat.nc"
        tas_path = f"{cru_path}/cru_ts4.08.1901.2023.tmp.dat.nc"

    if var == "Precipitation":
        ifile = xr.open_dataset(pr_path)
        var_suffix = "PR"
        if model == "CRU":
            ifile = ifile.rename({'pre': 'pr'})
            ifile = ifile[['pr']]
        # ifile=ifile['pr']

    else:
        ifile = xr.open_dataset(tas_path)
        var_suffix = "TEMP"
        if model == "CRU":
            ifile = ifile.rename({'tmp': 'tas'})
            ifile = ifile[['tas']]

    # select only the data for the relevant timeframe
    baseline = ifile.sel(time=slice(f'{y0}-01-01', f'{ye}-12-31'))

    """Part 1 - Correct the data"""
    # Data is provided in Precipitation flux (kg/m2/s) on a monthly basis

    if var == "Precipitation" and model == "W5E5":  # CRU is already provided in mm/month

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

    """Part 2 - Enable time averaging for the climate data (monthly, seasonal or annual)"""

    # reformat the time dimension of the data for resampling
    baseline["time"] = baseline.time.astype("datetime64[ns]")

    # calculate monthly averages (different from precipitation where calculate average monthly totals)
    if timeframe != "monthly":
        if var == "Temperature":
            baseline = baseline.resample(time=time_averaging).mean(
                dim='time')  # mean(dim='time')
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
    print(ofile_baseline)
    baseline.to_netcdf(ofile_baseline)
    return


for model in ["CRU", "W5E5"]:

    for member in [0]:
        for var in ["Precipitation", "Temperature"]:
            for timeframe in ["annual", "seasonal", "monthly"]:
                for diftype in ['abs']:
                    for y0 in [1985]:  # ,1901]:
                        ye = 2014

                        print(var, timeframe, y0)
                        process_P_T_baseline(
                            model, member, var, timeframe, diftype, y0, ye)


# %% Cell 4: Plotting the climate data - only for perturbations

def plot_P_T_perturbations(model, scale, var, timeframe, mode, diftype, plotsave, y0, ye):
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

    base_folder_in = f"/Users/magaliponds/OneDrive - Vrije Universiteit Brussel/1. VUB/02. Coding/01. IRRMIP/03. Data/03. Output files/01. Climate data/01. Processed input data/{var}/{timeframe}/{model}/{member}/"
    diff_folder_in = f"/Users/magaliponds/OneDrive - Vrije Universiteit Brussel/1. VUB/02. Coding/01. IRRMIP/03. Data/03. Output files/01. Climate data/02. Perturbations/{var}/{timeframe}/{model}/{member}/"

    ifile_diff = f"{diff_folder_in}/{model}.{var_suffix}.DIF.00{member}.{y0}_{ye}_{timeframe}_{diftype}.nc"
    ifile_diff = f"{diff_folder_in}/{model}.{var_suffix}.DIF.00{member}.{y0}_{ye}_{timeframe}_{diftype}.nc"
    ifile_irr = f"{base_folder_in}/{model}.{var_suffix}.IRR.00{member}.{y0}_{ye}_{timeframe}_{diftype}.nc"
    ifile_noi = f"{base_folder_in}/{model}.{var_suffix}.NOI.00{member}.{y0}_{ye}_{timeframe}_{diftype}.nc"

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
        o_file_name = f"{o_folder_diff}/{model}.{var_suffix}.DIF.00{member}.{y0}_{ye}_{timeframe}_{diftype}.png"
        plt.savefig(o_file_name, bbox_inches='tight')
    plt.show()
    return

# %% CEll 5: Plot the perturbations for IRR-counterfactual


def plot_P_T_perturbations_counterfactual(model, scale, var, timeframe, mode, diftype, plotsave):
    y0_cf = 1901
    ye_cf = 1930
    y0 = 1985
    ye = 2014
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

    # base_folder_in = f"/Users/magaliponds/OneDrive - Vrije Universiteit Brussel/1. VUB/02. Coding/01. IRRMIP/03. Data/03. Output files/01. Climate data/01. Processed input data/{var}/{timeframe}/{model}/{member}"
    diff_folder_in = f"/Users/magaliponds/OneDrive - Vrije Universiteit Brussel/1. VUB/02. Coding/01. IRRMIP/03. Data/03. Output files/01. Climate data/02. Perturbations/{var}/{timeframe}/{model}/{member}"

    ifile_diff = f"{diff_folder_in}/{model}.{var_suffix}.DIF.00{member}.{y0_cf}_{y0}_{timeframe}_{diftype}_counterfactual.nc"
    diff = xr.open_dataset(ifile_diff)

    # if var == "Temperature":
    #     irrigation = irrigation-273.15
    #     baseline = baseline-273.15

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
    # o_folder_base = f"/Users/magaliponds/OneDrive - Vrije Universiteit Brussel/1. VUB/02. Coding/01. IRRMIP/04. Figures/01. Climate data/01. Input data/{scale}/{var}/{model}/{member}"
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
        o_file_name = f"{o_folder_diff}/{model}.{var_suffix}.DIF.00{member}.{y0_cf}_{y0}_{timeframe}_{diftype}_counterfactual.png"
        plt.savefig(o_file_name, bbox_inches='tight')
    plt.show()
    return

# %% Cell 6: Plotting the climate data - for perturbations and input data


def plot_P_T_input_perturbations(plotvar, model, scale, var, timeframe, mode, diftype, plotsave, y0, ye):
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
        ifile_diff = f"{diff_folder_in}/{model}.{var_suffix}.{plotvar}.00{member}.{y0}_{ye}_{timeframe}_{diftype}.nc"
    else:
        ifile_diff = f"{base_folder_in}/{model}.{var_suffix}.{plotvar}.00{member}.{y0}_{ye}_{timeframe}_{diftype}.nc"

    # ifile_irr=f"{base_folder_in}/{model}.{var_suffix}.IRR.00{member}.{y0}_{ye}_{timeframe}_{mode_suff}_{diftype}.nc"
    # ifile_noi=f"{base_folder_in}/{model}.{var_suffix}.NOI.00{member}.{y0}_{ye}_{timeframe}_{mode_suff}_{diftype}.nc"

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
            o_file_name = f"{o_folder_diff}/{model}.{var_suffix}.{plotvar}.00{member}.{y0}_{ye}_{timeframe}_{mode_suff}_{diftype}.png"
            plt.savefig(o_file_name, bbox_inches='tight')
        else:
            os.makedirs(f"{o_folder_diff}/", exist_ok=True)
            o_file_name = f"{o_folder_diff}/{model}.{var_suffix}.{plotvar}.00{member}.{y0}_{ye}_{timeframe}_{mode_suff}_{diftype}.png"
            plt.savefig(o_file_name, bbox_inches='tight')
    plt.show()
    return


# %% Cell 7: Run the perturbation processing for all climate models, members etc.
members = [4]  # 1, 3, 4, 6, 3]
# members = [1, 1, 1, 1]
y0 = 1985
ye = 2014
# IPSL-CM6", "E3SM", "CESM2", "CNRM", "NorESM"]):
for (m, model) in enumerate(["NorESM"]):
    for member in range(members[m]):
        # "Precipitation"]:  # "Temperature"]:  # ,"Temperature"]:
        for var in ["Temperature", "Precipitation"]:  # "Temperature"]:
            for timeframe in ["annual", "seasonal", "monthly"]:
                for mode in ['dif']:  # , 'std']:
                    if var == "Precipitation" and mode == 'dif':
                        diftypes = ['abs', 'rel']
                    else:
                        diftypes = ['abs']
                    for dif in diftypes:
                        if member == 3:
                            print(var, model, member, timeframe, dif)

                            process_P_T_perturbations(
                                model, member, var, timeframe, mode, dif, y0, ye)
                            process_P_T_perturbations_counterfactual(
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


# %% Cell 8: Run the plotting functions for all different combinations to generate output datasets and plots
members = [3]  # 1, 3, 4, 6, 3]
y0 = 1985
ye = 2014
# "IPSL-CM6", "E3SM", "CESM2", "CNRM", "NorESM"]):
for (m, model) in enumerate(["NorESM"]):
    for member in range(members[m]):
        print(member)
        for scale in ["Local"]:  # ,"Global"]:
            for plotvar in ["DIF"]:  # "IRR", "NOI"]:#,"DIF"]:
                # "Temperature", "Precipitation"]:
                for var in ["Precipitation", "Temperature"]:
                    for timeframe in ["annual", "seasonal", "monthly"]:
                        for mode in ['dif']:  # , 'std']:

                            if plotvar == "DIF" and var == "Precipitation" and mode == 'dif':
                                diftypes = ["rel"]  # abs','rel']
                            else:
                                diftypes = ["abs"]

                            for dif in diftypes:
                                print(dif)
                                plot_P_T_perturbations(
                                    model, scale, var, timeframe, mode, dif, "save", y0, ye)
                                # plot_P_T_perturbations_counterfactual(
                                #     model, scale, var, timeframe, mode, dif, "save")
                                # plot_P_T_input_perturbations(plotvar, model, scale, var, timeframe, mode, dif,"save")

# %%


def plot_subplots(index, subplots, annotation, diff, timestamps, axes, shp, custom_cmap, timeframe, scale, title, vmin=None, vmax=None):

    for time_idx, timestamp_name in enumerate(timestamps):

        # Determine subplot location based on timeframe
        if timeframe == 'monthly':
            row = time_idx // 4  # Calculate row index
            col = time_idx % 4
            ax = axes[row, col]
        elif timeframe == 'seasonal':
            row = time_idx // 2
            col = time_idx % 2
            ax = axes[row, col]
        elif timeframe == 'annual':
            row, col = 0, 0
            ax = axes

        # Select time dimension based on scale and timeframe
        if scale == "Local":
            time_dim_name = list(diff.dims)[0]
        elif scale == "Global" and timeframe != 'annual':
            time_dim_name = list(diff.dims)[2]

        # Select relevant data slice
        if timeframe == 'annual':
            diff_sel = diff
        else:
            diff_sel = diff.isel({time_dim_name: time_idx})

        # Convert Dataset to DataArray if necessary
        if isinstance(diff_sel, xr.Dataset):
            diff_sel = diff_sel[list(diff_sel.data_vars.keys())[0]]

        # Plot data and the Karakoram outline
        im = diff_sel.plot.imshow(ax=ax, vmin=vmin, vmax=vmax, extend='both',
                                  transform=ccrs.PlateCarree(), cmap=custom_cmap, add_colorbar=False)
        shp.plot(ax=ax, edgecolor='black', linewidth=1, facecolor='none')
        ax.coastlines(resolution='10m')
        # Find min and max values, and annotate them
        diff_sel_min = diff_sel.fillna(diff_sel.max())
        diff_sel_max = diff_sel.fillna(diff_sel.min())

        min_value_index = np.unravel_index(
            np.argmin(diff_sel_min.values), diff_sel_min.shape)
        max_value_index = np.unravel_index(
            np.argmax(diff_sel_max.values), diff_sel_max.shape)

        # Extract coordinates for min and max values
        min_lon, min_lat = diff_sel.lon.values[min_value_index[1]
                                               ], diff_sel.lat.values[min_value_index[0]]
        max_lon, max_lat = diff_sel.lon.values[max_value_index[1]
                                               ], diff_sel.lat.values[max_value_index[0]]

        # Plot markers for min and max values
        ax.plot(min_lon, min_lat, marker='o',
                markersize=7, color='blue')  # Min marker
        ax.plot(max_lon, max_lat, marker='o',
                markersize=7, color='red')  # Max marker

        min_value = diff_sel[min_value_index]
        max_value = diff_sel[max_value_index]

        # Annotate with the timestamp & min max values
        ax.annotate(annotation, xy=(1, 1), xytext=(-10, -10), xycoords='axes fraction',
                    textcoords='offset points', ha='right', va='top', fontsize=12,
                    bbox=dict(boxstyle='square', fc='white', alpha=1))

        if index == "A":

            ax.annotate(f'Min: {min_value:.1f}', xy=(65, 50), xytext=(
                65, 50), fontsize=12, ha='left', va='top')
            ax.annotate(f'Max: {max_value:.1f}', xy=(81, 50), xytext=(
                81, 50), fontsize=12, ha='left', va='top')
            ax.plot(64, 49.5, marker='o', color='blue', markersize=5)
            ax.plot(79, 49.5, marker='o', color='red', markersize=5)

        # Set gridlines and labels
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False

        if index in ["A", "D"]:
            gl.ylabel_style = {'size': 12}
            gl.ylocator = plt.MaxNLocator(nbins=3)
        else:
            gl.left_labels = False

        # (timeframe == 'monthly' and row == 2) or (timeframe == 'seasonal' and row == 1) or (timeframe == 'annual'):
        if index in ["D", "E", "F"] or subplots != "on":
            gl.xlabel_style = {'size': 12}
            gl.xlocator = plt.MaxNLocator(nbins=3)
        else:
            gl.bottom_labels = False

        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        """4 Include labels for the cbar and for the y and x axis"""

        ax.set_title(title)

    return im


# def plot_P_T_perturbations_avg(scale, var, timeframe, mode, diftype, plotsave):
""" Part 0 - Set plotting parameters"""


y0 = 1985  # if running from 1901 to 1985, than indicate extra id of counterfactual to access the data
ye = 2014
extra_id = ""  # "_counterfactual"
scale = "Local"
subplots = "off"
for var in ["Temperature"]:  # "Temperature"]:  # ,"Temperature"]:
    for timeframe in ["annual"]:  # :, "seasonal", "monthly"]:
        for mode in ['dif']:  # , 'std']:
            if var == "Precipitation" and mode == 'dif':
                diftypes = ['abs', 'rel']
            else:
                diftypes = ['abs']
            for dif in diftypes:
                print(var, timeframe, dif)
                diftype = dif
                # plot_P_T_perturbations_avg(scale,var, timeframe, mode, dif, "off")
                # adjust figure sizes towards type of plot# adjust figure sizes towards type of plot
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
                        figsize = (7, 5)  # (50, 25)#(7, 5)
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
                        # fig, axes = plt.subplots(nrows=3, ncols=4, subplot_kw={
                        #                          'projection': ccrs.PlateCarree()}, figsize=figsize)
                        time_signature = 'ymon'
                        timestamps = ['JAN', 'FEB', 'MAR', 'APR', 'MAY',
                                      'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
                        time_averaging = 'time.month'
                        time_type = 'month'
                        col_wrap = 4
                    if timeframe == 'seasonal':
                        figsize = (9, 7)
                        # fig, axes = plt.subplots(nrows=2, ncols=2, subplot_kw={
                        #                          'projection': ccrs.PlateCarree()}, figsize=figsize)
                        time_signature = 'yseas'
                        timestamps = ['DJF', 'MAM', 'JJA', 'SON']
                        time_averaging = 'time.season'
                        time_type = 'season'
                        col_wrap = 2
                    if timeframe == 'annual':
                        figsize = (7, 5)
                        # fig, axes = plt.subplots(nrows=1, ncols=5, subplot_kw={
                        #                          'projection': ccrs.PlateCarree()}, figsize=figsize)
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
                        vmin = -20
                        vmax = 20
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

                members = [1, 3, 4, 6]  # ,4]
                # members = [1, 1, 1, 1]
                all_diff = []  # create a dataset to add all member differences
                all_model_diffs = []
                models = ["IPSL-CM6", "E3SM", "CESM2", "CNRM"]  # ,"NorESM"]

                for (m, model) in enumerate(models):
                    model_diff = []
                    for member in range(members[m]):
                        # only open data for non model averages (except for IPSL-CM6 as only one member)
                        if model == "IPSL-CM6" or member != 0:

                            # Part 1: Delete
                            diff_folder_in = f"/Users/magaliponds/OneDrive - Vrije Universiteit Brussel/1. VUB/02. Coding/01. IRRMIP/03. Data/03. Output files/01. Climate data/03. Regridded Perturbations/{var}/{timeframe}/{model}/{member}"
                            ifile_diff = f"{diff_folder_in}/REGRID.{model}.{var_suffix}.DIF.00{member}.{y0}_{ye}_{timeframe}_{diftype}{extra_id}.nc"
                            diff = xr.open_dataset(ifile_diff)

                            if scale == "Local":  # scale the data to the local scale
                                diff = diff.where((diff.lon >= 60) & (diff.lon <= 109) & (
                                    diff.lat >= 22) & (diff.lat <= 52), drop=True)
                            # loose all the filtered data (nan)
                            diff_clean = diff.dropna(dim="lon", how="all")
                            # include the values in the list for caluclating the avg difference by model
                            model_diff.append(diff_clean)
                            # include the values in the list for caluclating the avg difference over all models
                            all_diff.append(diff_clean)
                    all_model_diff = xr.concat(
                        model_diff, dim="models").mean(dim="models")  # concatenate all models into a list averaged by model
                    all_model_diffs.append(all_model_diff)
                all_model_diffs_avg = xr.concat(
                    all_model_diffs, dim="models")  # concatenate all models
                all_diffs_avg = xr.concat(all_diff, dim="models").mean(
                    dim="models")  # concatenate all members and calculate the mean over all the models

                """ Part 2 - Shapefile outline for Karakoram Area to be included"""
                # path to  shapefile
                shapefile_path = '/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/01. Input files/03. Shapefile/Karakoram/Pan-Tibetan Highlands/Pan-Tibetan Highlands (Liu et al._2022)/Shapefile/Pan-Tibetan Highlands (Liu et al._2022)_P.shp'
                shp = gpd.read_file(shapefile_path)
                target_crs = 'EPSG:4326'
                shp = shp.to_crs(target_crs)

                indices = ["A", "B", "C", "D", "E"]  # ,"F"]

                # Create the mosaic plot
                if subplots == "on":
                    layout = """
                    AAB
                    AAC
                    DEF
                    """
                    fig, axes = plt.subplot_mosaic(layout, subplot_kw={'projection': ccrs.PlateCarree()},
                                                   figsize=figsize,
                                                   gridspec_kw={'wspace': 0, 'hspace': 0.4})
                else:
                    layout = """
                    AA
                    AA
                    """
                    fig, axes = plt.subplot_mosaic(layout, subplot_kw={'projection': ccrs.PlateCarree()},
                                                   figsize=figsize,
                                                   gridspec_kw={'wspace': 0, 'hspace': 0.4})
                # plot the irrmip difference
                im = plot_subplots(indices[0], subplots, (sum(members)-len(models)+1),
                                   all_diffs_avg, timestamps, axes[indices[0]], shp, custom_cmap, timeframe, scale, f"IRRMIP", vmin=vmin, vmax=vmax)
                if subplots == "on":
                    for (m, model) in enumerate(models):
                        print(m+1)
                        annotation = members[m] - \
                            1 if model != "IPSL-CM6" else members[m]
                        im_model = plot_subplots(indices[m+1], subplots, annotation,
                                                 all_model_diffs_avg.sel(models=m), timestamps, axes[indices[m+1]], shp, custom_cmap, timeframe, scale, model, vmin=vmin, vmax=vmax)

                """ 3C Add color bar for entire plot"""
                # add cbar in the figure, for overall figure, not subplots
                # Define the position of the colorbar
                cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
                cbar = fig.colorbar(im, cax=cbar_ax, extend='both')

                # Increase distance between colorbar label and colorbar
                cbar.ax.yaxis.labelpad = 20
                if mode == 'dif':
                    cbar.set_label(f'$\Delta$ {var} [{unit}]', size='15')
                    if mode == 'std':
                        cbar.set_label(
                            f'{var} - model member std [{unit}]', size='15')
                cbar.ax.tick_params(labelsize=12)

                # adjust subplot spacing to be smaller
                plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1,
                                    top=0.9, wspace=0.05, hspace=0.05)

                # if plotsave == 'save':
                # os.makedirs(f"o_folder_base/{scale}/{timeframe}/{var}/", exist_ok=True)
                # os.makedirs(f"{o_folder_diff}/", exist_ok=True)
                # o_file_name = f"{o_folder_diff}/{model}.{var_suffix}.DIF.00{member}.{y0}_{ye}_{timeframe}_{diftype}.png"
                # plt.savefig(o_file_name, bbox_inches='tight')
                plt.show()
                # return
