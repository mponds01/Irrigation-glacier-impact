#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 14:14:34 2024

@author: magaliponds
"""

import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

"""
This script performs an initial data screening, in 3 sections:
    1. checks what the monthly irrigation, controlrun and difference in P look like, plotted in a geoplot
    2. checks the completeness of data, using a timeseries plot
    3. tests what components should be part of total precipitaiton (received from atmosphere)
"""
    

folder="/Users/magaliponds/OneDrive - Vrije Universiteit Brussel/1. VUB/02. Coding/01. IRRMIP/02. Data/01. Input files/E3SM/"
ifile_IRR="E3SM.IRR.000.1981_2014_selparam_monthly_total.nc"
ifile_NOI="E3SM.NOI.000.1981_2014_selparam_monthly_total.nc"


"""Part 1 - create differences plot"""

baseline_R = xr.open_dataset(folder+ifile_NOI).RAIN*86400
baseline_S = xr.open_dataset(folder+ifile_NOI).SNOW*86400
baseline=baseline_R+baseline_S
baseline = baseline.where((baseline.lon >= 60) & (baseline.lon <= 100) & (baseline.lat >= 20) & (baseline.lat <= 50), drop=True) 
baseline["time"] = baseline.time.astype("datetime64[ns]")
baseline = baseline.groupby('time.month').mean()
# baseline.plot(x="lon", y="lat", col="month", col_wrap=4, vmax= 100)

irrigation_R = xr.open_dataset(folder+ifile_IRR).RAIN*86400
irrigation_S = xr.open_dataset(folder+ifile_IRR).SNOW*86400
irrigation=irrigation_R+irrigation_S
irrigation = irrigation.where((irrigation.lon >= 60) & (irrigation.lon <= 100) & (irrigation.lat >= 20) & (irrigation.lat <= 50), drop=True) 
irrigation["time"] = irrigation.time.astype("datetime64[ns]")
irrigation = irrigation.groupby('time.month').mean()
# irrigation.plot(x="lon", y="lat", col="month", col_wrap=4, vmax= 100)

# ((irrigation-baseline)/baseline *100).plot(x="lon", y="lat", col="month", col_wrap=4, vmin =-50, vmax=50, cmap="bwr")

shapefile_path = '/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/01. Input files/Shapefile/glac_inv/glac_inv_all_lowres.shp'

"""Part 2 - check timeseries"""
# baseline = xr.open_dataset(folder+ifile_NOI).RAIN*86400*30
# baseline = baseline.where((baseline.lon >= 60) & (baseline.lon <= 100) & (baseline.lat >= 20) & (baseline.lat <= 50), drop=True) 
# baseline["time"] = baseline.time.astype("datetime64[ns]")

# baseline.where(baseline['time.year'] > 1970, drop=True).sel(lat = 40, lon = 80, method = "nearest").plot()

"""Part 3 - Baseline check variables sums"""
# baseline_RFA = xr.open_dataset(folder+ifile_NOI).RAIN_FROM_ATM*86400*30
# baseline_RFA = baseline_RFA.where((baseline_RFA.lon >= 60) & (baseline_RFA.lon <= 100) & (baseline_RFA.lat >= 20) & (baseline_RFA.lat <= 50), drop=True) 
# baseline_RFA["time"] = baseline_RFA.time.astype("datetime64[ns]")

# baseline_SFA = xr.open_dataset(folder+ifile_NOI).SNOW_FROM_ATM*86400*30
# baseline_SFA = baseline_SFA.where((baseline_SFA.lon >= 60) & (baseline_SFA.lon <= 100) & (baseline_SFA.lat >= 20) & (baseline_SFA.lat <= 50), drop=True) 
# baseline_SFA["time"] = baseline_SFA.time.astype("datetime64[ns]")



baseline_R = xr.open_dataset(folder+ifile_NOI).RAIN*86400*30
baseline_R = baseline_R.where((baseline_R.lon >= 60) & (baseline_R.lon <= 100) & (baseline_R.lat >= 20) & (baseline_R.lat <= 50), drop=True) 
baseline_R["time"] = baseline_R.time.astype("datetime64[ns]")

baseline_S = xr.open_dataset(folder+ifile_NOI).SNOW*86400*30 
baseline_S = baseline_S.where((baseline_S.lon >= 60) & (baseline_S.lon <= 100) & (baseline_S.lat >= 20) & (baseline_S.lat <= 50), drop=True) 
baseline_S["time"] = baseline_S.time.astype("datetime64[ns]")

# #make sum for total precipitation received form atm 
# baseline_ATM = baseline_RFA + baseline_SFA
# baseline_ATM = baseline_ATM.where((baseline_ATM.lon >= 60) & (baseline_ATM.lon <= 100) & (baseline_ATM.lat >= 20) & (baseline_ATM.lat <= 50), drop=True) 
# baseline_ATM["time"] = baseline_ATM.time.astype("datetime64[ns]")
# # baseline_ATM = baseline_ATM.groupby('time.month').mean()

#make sum for total precipition after re-partitioning
baseline_PART = baseline_R + baseline_S
baseline_PART = baseline_PART.where((baseline_PART.lon >= 60) & (baseline_PART.lon <= 100) & (baseline_PART.lat >= 20) & (baseline_PART.lat <= 50), drop=True) 
baseline_PART["time"] = baseline_PART.time.astype("datetime64[ns]")
baseline_PART = baseline_PART.groupby('time.month').mean()

#create figure for total snow redistribution
# plt.figure()
# baseline_PART.where(baseline_PART['time.year'] > 1970, drop=True).sel(lat = 40, lon = 80, method = "nearest").plot(label="R+S", color="thistle")
# baseline_ATM.where(baseline_ATM['time.year'] > 1970, drop=True).sel(lat = 40, lon = 80, method = "nearest").plot(label="RFA+SFA", color="purple", linestyle=':', linewidth=1)
# plt.legend()
# plt.title("Total P before and after repartitioning")
# plt.ylabel("Monthly Precipitation [mm]")

plt.figure()
baseline_R.sel(lat = 40, lon = 80, method = "nearest").plot(label="R", color="peachpuff")
baseline_S.sel(lat = 40, lon = 80, method = "nearest").plot(label="S", color="lightblue")
# baseline_PART.sel(lat = 40, lon = 80, method = "nearest").plot(label="R+S", color="purple", linestyle=':', linewidth=1)

# baseline_RFA.where(baseline_RFA['time.year'] > 1970, drop=True).sel(lat = 40, lon = 80, method = "nearest").plot(label="RFA", color="lightsalmon", linestyle=':', linewidth=1)
# baseline_SFA.where(baseline_SFA['time.year'] > 1970, drop=True).sel(lat = 40, lon = 80, method = "nearest").plot(label="SFA", color = 'dodgerblue', linestyle=':', linewidth=1)

plt.legend()
plt.ylabel("Monthly Precipitation [mm]")
plt.title("P variables before and after repartitioning")

# plt.figure()
# baseline_SFA.where(baseline_SFA['time.year'] > 1970, drop=True).sel(lat = 40, lon = 80, method = "nearest").plot(label="SFA", color="dodgerblue")
# baseline_RFA.where(baseline_RFA['time.year'] > 1970, drop=True).sel(lat = 40, lon = 80, method = "nearest").plot(label="RFA", color="lightsalmon")
# baseline_ATM.where(baseline_ATM['time.year'] > 1970, drop=True).sel(lat = 40, lon = 80, method = "nearest").plot(label="RFA+SFA", color="purple", linestyle=':', linewidth=1)

# plt.legend()
# plt.title("Components of total Precipitation")
# plt.ylabel("Monthly Precipitation [mm]")



