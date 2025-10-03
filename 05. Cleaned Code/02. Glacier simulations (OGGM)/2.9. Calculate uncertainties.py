# -*- coding: utf-8 -*-
"""
Created on Wed Oct 1  3 15:36:52 2024

@author: magaliponds

This script calculates the uncertainties for all analysis done

Cell 0: Load data packages
Cell 0b: Set base parameters (dictionaries)
Cell 1: Calculate uncertainties on past MB
Cell 2: Calculate uncertainties on past volume timeseries
Cell 3a: Calculate uncertainties on future volume timeseries - ∆V
Cell 3b: Calculate uncertainties on future volume timeseries - Irr and NoIrr
Cell 4a: Calculate uncertainties on for comitted loss - hma overall
Cell 4b: Calculate uncertainties for comitted loss - subregion specific
Cell 5: Calculate uncertainties on past and future runoff timeseries


"""
#%% Cell 0a: Load data packages
# -*- coding: utf-8 -*-import oggm
# from OGGM_data_processing import process_perturbation_data
# import mpl_axes_aligner
import concurrent.futures
from shapely.geometry import LineString, MultiLineString
import ast
import string
from matplotlib.lines import Line2D
import oggm
from oggm import utils, cfg, workflow, tasks, DEFAULT_BASE_URL, graphics, global_tasks
from oggm.core import massbalance, flowline
from oggm.utils import floatyear_to_date, hydrodate_to_calendardate
from oggm.sandbox import distribute_2d
from oggm.sandbox.edu import run_constant_climate_with_bias
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as clrs
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import ConnectionPatch
import xarray as xr
import os
import seaborn as sns
import salem
from matplotlib.ticker import FuncFormatter

import pandas as pd
import numpy as np
from matplotlib import animation
from IPython.display import HTML, display

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats
from matplotlib.legend_handler import HandlerTuple
from matplotlib.colors import LightSource
from tqdm import tqdm
import pickle
import sys
import textwrap
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize, ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from shapely.geometry import Point
import matplotlib.gridspec as gridspec
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.patches import Rectangle, ConnectionPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


function_directory = "/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Research/01. IRRMIP/src/05. Cleaned Code/ 02. Glacier simulations (OGGM)"
sys.path.append(function_directory)

#%% Cell 0b: Set base parameters
# colors = {

#     "irr": ["#000000", "#777777"],
#     "noirr": ["#6363FF", "#B1B1FF"],
#     # much lighter versions of noirr colors
#     "noirr_com": ["#A6A6FF", "#E0E0FF"],
#     # much lighter grey fade of irr colors
#     "irr_com": ["#C0C0C0", "#E0E0E0"],
#     "cf": ["#FF5722", "#FFA780"],
#     "Yellow": ["#FFC107", "#FFE08A"]
# }
xkcd_colors = clrs.XKCD_COLORS

# colors = {
#     "irr": ["#000000", "#555555"],  # Black and dark gray
#     # Darker brown and golden yellow for contrast
#     # "noirr": ["#f5bf03","#fbeaac"],#["#8B5A00", "#D4A017"], #fdde6c
#     "noirr": ["dimgrey","darkgrey"],#["#8B5A00", "#D4A017"], #fdde6c
#     # "noirr_com": ["#E3C565", "#F6E3B0"],  # Lighter, distinguishable tan shades #
#     # "noirr_com": ["#380282", "#ceaefa"],  # Lighter, distinguishable tan shades #
#     "noirr_com": ["#FFC107", "#FFF3CD"],  # Lighter, distinguishable tan shades #
#     "irr_com": ["#fe4b03", "#D0D0D0"],  # Light gray, no change
#     # "irr_com": ["#B5B5B5", "#D0D0D0"],  # Light gray, no change

#     "cf": ["#004C4C", "#40E0D0"],
#     "cf_com": ["#008B8B", "#40E0D0"],
#     "cline": ["dimgrey", '#FFC107']
# }

colors_ssp = {
    'ssp585': '#951b1e',
    'ssp370': '#e71d25',
    'ssp245': '#f79420',
    'ssp126': '#173c66',
    'ssp119': '#00adcf'
}

# Function to convert hex to RGBA with alpha
def hex_to_rgba(hex_color, alpha=1.0):
    return clrs.to_rgba(hex_color, alpha=alpha)

colors = {
    "irr": ["#000000", "#555555"],  # Black and dark gray
    "noirr": ["dimgrey","darkgrey"],#["#8B5A00", "#D4A017"], #fdde6c
    # "noirr_com": ["#FFC107", "#FFF3CD"],  # Lighter, distinguishable tan shades #
    # "irr_com": ["#fe4b03", "#D0D0D0"],  # Light gray, no change
    # "noirr_com": ["#FFA500", "#FFD580"],
    # "irr_com": ["#00796B", "#388E3C"],    # Teal and deep green (distinct from SSPs)
    # "noirr_com": ["#80CBC4", "#A5D6A7"],  # Light turquoise and mint
    "irr_com": ["#A0522D", "#BF6C38"],     # Dry, earthy rust tones
    "noirr_com": ["#E6A87D", "#F3D4B3"],   # Muted peach and tan
    "cf": ["#004C4C", "#40E0D0"],
    "cf_com": ["#008B8B", "#40E0D0"],
    "cline": ["dimgrey", '#E6A87D'],
    "cline_2": ["black", 'dimgrey'],
    "irr_fut": [
        colors_ssp['ssp126'],  # SSP126
        colors_ssp['ssp370']   # SSP370
    ],

    # Future scenarios: same SSP colors but with alpha 0.5
    "noirr_fut": [
        hex_to_rgba(colors_ssp['ssp126'], 0.5),
        hex_to_rgba(colors_ssp['ssp370'], 0.5)
    ]
}



region_colors = {13: 'blue', 14: 'crimson', 15: 'orange'}


colors_models = {
    "W5E5": ["#000000"],  # "#000000"],  # Black
    # Darker to lighter shades of purple
    "E3SM": ["#785EF0", "#8F7BF1", "#A6A8F2"],
    # Darker to lighter shades of pink
    "CESM2": ["#DC267F", "#E58A9E", "#F0A2B6", "#F7BCC4"],
    # Darker to lighter shades of orange
    "CNRM": ["#FE6100", "#FE7D33", "#FE9A66", "#FEB799", "#FECDB5", "#FEF1E1"],
    "IPSL-CM6": ["#FFB000"],
    "NorESM": ["#FFC107", "#FFE08A"]  # Dark purple to lighter shades
}
members = [1, 3, 4, 6, 4, 1]
members_averages = [1, 2, 3, 5, 3]
models = ["IPSL-CM6", "E3SM", "CESM2", "CNRM", "NorESM", "W5E5"]
models_shortlist = ["IPSL-CM6", "E3SM", "CESM2", "CNRM", "NorESM"]
timeframe = "monthly"

y0_clim = 1985
ye_clim = 2014


fig_path = '/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Research/01. IRRMIP/04. Figures/98. Final Figures A+1km2/'
folder_path = '/Users/magaliponds/Documents/00. Programming'
wd_path = f'{folder_path}/04. Modelled perturbation-glacier interactions - R13-15 A+1km2/'
sum_dir = os.path.join(wd_path, 'summary')
#%% Cell 1: Past MB uncertainties

df = pd.read_csv(
    f"{wd_path}masters/master_gdirs_r3_a1_rgi_date_A_V_RGIreg_B_hugo_Vcom.csv")

master_ds = df[(~df['sample_id'].str.endswith('0')) |  # Exclude all the model averages ending with 0 except for IPSL
               (df['sample_id'].str.startswith('IPSL'))]

master_ds = master_ds[['rgi_id', 'rgi_region', 'rgi_subregion','full_name', 'cenlon', 'cenlat', 'rgi_date', 'rgi_area_km2','rgi_volume_km3', 'sample_id', 'B_noirr', 'B_irr', 'B_delta', 'B_hugo','Area', 'errB_hugo','V_1985_irr']]


master_ds = master_ds[master_ds['B_irr'].notna()]

# calculate for each subregion the average B_irr and B_noirr for all the different sample ids
master_ds_avg_subregions = master_ds.groupby(['rgi_subregion','sample_id'], as_index=False).agg({  # calculate the average regional mass balance over subregion per sample id
    # 'B_delta': 'mean',
    'B_delta': lambda x: (x * master_ds.loc[x.index, 'rgi_area_km2']).sum() / master_ds.loc[x.index, 'rgi_area_km2'].sum(), #area weighted mean mass balance in the region
    'Area': 'sum',
    'errB_hugo': 'mean',
    'rgi_date':'mean',
    'rgi_volume_km3': 'sum',
    'rgi_area_km2':'sum',
    **{col: 'first' for col in master_ds.columns if col not in ['B_delta']}})[[ 'rgi_region', 'rgi_subregion',
       'full_name', 'B_delta', 'rgi_area_km2',
       'rgi_volume_km3', 'sample_id',  'Area', 'errB_hugo']]


#calculate the std across the sample_id dimension                                                                                        
std_per_subregion = (
    master_ds_avg_subregions
    .groupby("rgi_subregion")[["B_delta"]]
    .std()
    .reset_index()
    .rename(columns={"B_delta": "B_delta_std"})
)

#calculate the mean for all hma for different sample ids
master_ds_avg = master_ds.groupby(['sample_id'], as_index=False).agg({  # calculate the average mass balance over subregion per sample id
    'B_delta': lambda x: (x * master_ds.loc[x.index, 'rgi_area_km2']).sum() / master_ds.loc[x.index, 'rgi_area_km2'].sum(), #area weighted mean mass balance in the region
    'Area': 'sum',
    'errB_hugo': 'mean',
    'rgi_date':'mean',
    'rgi_volume_km3': 'sum',
    'rgi_area_km2':'sum',
    **{col: 'first' for col in master_ds.columns if col not in ['B_delta']}})[[ 'rgi_region', 'rgi_subregion',
       'full_name', 'B_delta', 'rgi_area_km2',
       'rgi_volume_km3', 'sample_id',  'Area', 'errB_hugo']]

   
#calculate the std for all hma across sample id dimension
std_hma = (
    master_ds_avg
    .groupby("rgi_subregion")[["B_delta"]]
    .std()
    .reset_index()
    .rename(columns={"B_delta": "B_delta_std"})
)
std_hma["rgi_subregion"] = "HMA"
std_B_concat = pd.concat([std_per_subregion,std_hma], ignore_index=True)  # stack rows
std_B_concat['B_delta_std_95CI']=std_B_concat['B_delta_std']*1.96

std_B_concat.to_csv(f"{wd_path}masters/master_B_std_subregions.csv")



#%% Cell 2: Past Volume uncertainties

#calculation of uncertainties:
    
# --- calculate std across the 14 members, for each timestep, for each subregion and HMA as a total ---
ds_subregions = xr.open_dataset(f"{wd_path}masters/master_volume_ts_subregions_members.nc")
ds_subregions_med = xr.open_dataset(f"{wd_path}masters/master_volume_ts_subregions_members_median.nc")
# std_subregions = ds_subregions_med.drop_sel(member="14-member-med").volume.std(dim="member").rename("std_volume")



global_full_med = xr.open_dataset(f"{wd_path}masters/master_volume_ts_global_members_median.nc")
past_ds = xr.open_dataset(f"{wd_path}/masters/master_volume_subregion_past_median.nc")
# std_hma = (
#     global_full_med.drop_sel(member="14-member-med")
#     .volume.std(dim="member")
#     .rename("std_volume")
#     .expand_dims(subregion=["HMA"])
# )

# std_complete = xr.concat([std_subregions, std_hma], dim="subregion")
# std_complete.to_netcdf(f"{wd_path}masters/master_V_std_subregions.nc")

# --- create difference timeseries --- ### 

#for different subregions
initial_timeseries = past_ds.sel(exp="IRR", sample_id="3-member-avg").volume # initial_timeseries: dims (subregion, time)
vol = ds_subregions_med.drop_sel(member="14-member-med").volume # membered volumes: dims (subregion, time, member)
vol_al, init_al = xr.align(vol, initial_timeseries, join="inner", exclude=["member"]) # (recommended) align to make sure subregion/time labels match
volume_anom_hist = ((init_al - vol_al )/init_al.sel(time=1985)*100).rename("volume_anom") #difference irr-noi in absolute terms
std_subregions = ( volume_anom_hist.std(dim="member").rename("std_volume"))

initial_timeseries_hma = past_ds.sel(exp="IRR", sample_id="3-member-avg").volume.sum(dim='subregion') # initial_timeseries: dims (subregion, time)
vol_hma = global_full_med.drop_sel(member="14-member-med").volume
vol_al_hma, init_al_hma = xr.align(vol_hma, initial_timeseries_hma, join="inner", exclude=["member"])
volume_anom_hist_hma = ((init_al_hma - vol_al_hma )/init_al_hma.sel(time=1985)*100).rename("volume_anom")
std_hma = ( volume_anom_hist_hma.std(dim="member").rename("std_volume").expand_dims(subregion=["HMA"]))

std_complete = xr.concat([std_subregions, std_hma], dim="subregion")
std_complete.to_netcdf(f"{wd_path}masters/master_V_std_subregions.nc")

# # --- build initial volumes as an xarray DataArray (no DataFrame) --- for relative std calculation
# # per-subregion initial volume (IRR, 1985,w5e5 data based)
# subregions_initial_da = (
#     past_ds
#     .sel(exp="IRR", time=1985, sample_id="3-member-avg")
#     .volume                               # dims: subregion
# ).rename("initial_volume")

# # HMA initial = sum over all subregions, cast to subregion dim "HMA"
# hma_initial_scalar = subregions_initial_da.sum(dim="subregion")
# hma_initial_da = (
#     hma_initial_scalar
#     .rename("initial_volume")
#     .expand_dims({"subregion": ["HMA"]})
# )

# # Combine: initial volumes for all subregions + HMA
# initial_volume = xr.concat([subregions_initial_da, hma_initial_da], dim="subregion")
# # (optional) ensure same subregion order as std_complete
# initial_volume = initial_volume.reindex(subregion=std_complete["subregion"])

# --- compute absolute & relative 95% CI ---
ci95_rel = (1.96 * std_complete).rename("ci95_rel")  # same units as volume
# protect against divide-by-zero
# den = initial_volume.where(initial_volume != 0)
# ci95_rel = (ci95_abs / den *100).rename("ci95_rel")       # fraction; multiply by 100 for %

# --- save (optional) ---
ci95_rel.to_netcdf(f"{wd_path}masters/master_V_ci95_rel_subregions.nc")
# ci95_rel.to_netcdf(f"{wd_path}masters/master_V_ci95_rel_subregions.nc")

# --- if you want a CSV snapshot (e.g., year 2014) ---
out_2014 = xr.merge([std_complete, ci95_rel]).sel(time=2014)
out_2014 = out_2014.reset_coords(["hydro_year", "hydro_month", "calendar_year", "calendar_month"], drop=True)
out_2014_df = out_2014.to_dataframe()
out_2014_df.to_csv(f"{wd_path}masters/master_V_std_ci_2014.csv")


#%% Cell 3a: Future Volume uncertainties - change

# 1) Open the dataset
future_ds = xr.open_dataset(f"{wd_path}/masters/master_volume_subregion_future_noi_bias_median.nc")
# future_ds_std_subregions = future_ds.drop_sel(sample_id="3-member-avg").volume.std(dim="sample_id").rename("std_volume")
future_ds_hma = future_ds.sum(dim="subregion")                             # aggregate across subregions)

#Calculate the difference between the irr and the noirr scenarios
future_diff = future_ds.sel(exp="IRR")-future_ds.sel(exp="NOI")
future_ds_std_subregions = future_diff.drop_sel(sample_id="3-member-avg").volume.std(dim="sample_id").rename("std_volume")

future_diff_hma = future_ds_hma.sel(exp="IRR")-future_ds_hma.sel(exp="NOI")
future_ds_std_hma = future_diff_hma.drop_sel(sample_id="3-member-avg").volume.std(dim="sample_id").rename("std_volume").expand_dims(subregion=["HMA"]) 

# 2) Compute the std across sample members, excluding "3-member-avg"
# future_ds_std_hma = (
#      master_ds_hma.drop_sel(sample_id="3-member-avg") #drop the 3 member dimension
#     .volume
#     .std(dim="sample_id", skipna=True)                # take std across 3 different members
#     .rename("std_volume")                           #rename volume to std_volume
#     .expand_dims(subregion=["HMA"])                   # make 'subregion' a proper dimension for concat
# )

# 3) Concatenate with existing subregion std (must share other dims: exp, time, ssp, ...)
std_complete_future = xr.concat([future_ds_std_subregions, future_ds_std_hma], dim="subregion")
std_complete_future.to_netcdf(f"{wd_path}masters/master_V_future_std_subregions.nc")

std_subregions_2073_df = std_complete_future.sel(time=2073).to_dataframe()
std_subregions_2073_df.to_csv(f"{wd_path}masters/master_V_future_std_subregions_2073.csv")

#use following command for multi indexing
# std_subregions_2073_df.loc[("NOI", "126", "HMA")]


# 3) Build the fixed historic baseline per subregion (incl. HMA)

past_ds = xr.open_dataset(f"{wd_path}/masters/master_volume_subregion_past_median.nc")

subregions_initial_da = (
    past_ds
    .sel(exp="IRR", time=1985, sample_id="3-member-avg")
    .volume                             # dims: subregion
).rename("initial_volume")

hma_initial_scalar = subregions_initial_da.sum(dim="subregion")

hma_initial_da = (
    hma_initial_scalar
    .rename("initial_volume")
    .expand_dims({"subregion": ["HMA"]})  # make it length-1 along 'subregion'
)

initial_volume = xr.concat([subregions_initial_da, hma_initial_da], dim="subregion")

# Make sure ordering/labels line up with std_complete
initial_volume = initial_volume.reindex(subregion=std_complete_future["subregion"])

# 4) Compute absolute and relative (baseline) 95% CI for ALL FUTURE TIMES/SCENARIOS
#    This keeps the denominator fixed at the historic baseline for every time/exp/ssp.
ci95_abs = (1.96 * std_complete_future).rename("ci95_abs")

# guard against zeros in the baseline
den = initial_volume.where(initial_volume != 0)

ci95_rel_baseline = (ci95_abs / den).rename("ci95_rel_baseline")   # fraction (e.g., 0.12 = 12%)
ci95_rel_baseline_pct = (ci95_rel_baseline * 100).rename("ci95_rel_baseline_pct")

# # 5) (Optional) also store relative std (σ / baseline) if you want the raw ratio without 1.96
std_rel_baseline = (std_complete_future / den).rename("std_rel_baseline")
std_rel_baseline_pct = (std_rel_baseline * 100).rename("std_rel_baseline_pct")

# 6) Save full time series for the future (all times/exp/ssp/subregion)
xr.merge([std_complete_future, std_rel_baseline, std_rel_baseline_pct, ci95_abs, ci95_rel_baseline, ci95_rel_baseline_pct]).to_netcdf(
    f"{wd_path}masters/master_V_uncertainty_future_baseline1985.nc"
)

# 7) If you need a CSV snapshot for a particular year (e.g., 2073)
snap_2073 = xr.merge([std_complete_future,std_rel_baseline, std_rel_baseline_pct,  ci95_abs, ci95_rel_baseline, ci95_rel_baseline_pct]).sel(time=2073)

# drop aux time coords if present
drop_coords = [c for c in ["hydro_year","hydro_month","calendar_year","calendar_month"] if c in snap_2073.coords]
snap_2073 = snap_2073.reset_coords(drop_coords, drop=True)

snap_2073_df = snap_2073.to_dataframe()
snap_2073_df.to_csv(f"{wd_path}masters/master_V_uncertainty_2073_baseline1985.csv")

#%% Cell 3b: Calculate future uncertainties per irr and noirr
# 1) Open the dataset
future_ds = xr.open_dataset(f"{wd_path}/masters/master_volume_subregion_future_noi_bias_median.nc")
future_ds_std_subregions = future_ds.drop_sel(sample_id="3-member-avg").volume.std(dim="sample_id").rename("std_volume")
future_ds_hma = future_ds.sum(dim="subregion")                             # aggregate across subregions)

#Calculate the difference between the irr and the noirr scenarios


# 2) Compute the std across sample members, excluding "3-member-avg"
future_ds_std_hma = (
     future_ds_hma.drop_sel(sample_id="3-member-avg") #drop the 3 member dimension
    .volume
    .std(dim="sample_id", skipna=True)                # take std across 3 different members
    .rename("std_volume")                           #rename volume to std_volume
    .expand_dims(subregion=["HMA"])                   # make 'subregion' a proper dimension for concat
)

# 3) Concatenate with existing subregion std (must share other dims: exp, time, ssp, ...)
std_complete_future = xr.concat([future_ds_std_subregions, future_ds_std_hma], dim="subregion")
std_complete_future.to_netcdf(f"{wd_path}masters/master_V_future_std_subregions.nc")

std_subregions_2073_df = std_complete_future.sel(time=2073).to_dataframe()
std_subregions_2073_df.to_csv(f"{wd_path}masters/master_V_future_std_subregions_2073.csv")

#use following command for multi indexing
# std_subregions_2073_df.loc[("NOI", "126", "HMA")]


# 3) Build the fixed historic baseline per subregion (incl. HMA)

past_ds = xr.open_dataset(f"{wd_path}/masters/master_volume_subregion_past_median.nc")

subregions_initial_da = (
    past_ds
    .sel(exp="IRR", time=1985, sample_id="3-member-avg")
    .volume                             # dims: subregion
).rename("initial_volume")

hma_initial_scalar = subregions_initial_da.sum(dim="subregion")

hma_initial_da = (
    hma_initial_scalar
    .rename("initial_volume")
    .expand_dims({"subregion": ["HMA"]})  # make it length-1 along 'subregion'
)

initial_volume = xr.concat([subregions_initial_da, hma_initial_da], dim="subregion")

# Make sure ordering/labels line up with std_complete
initial_volume = initial_volume.reindex(subregion=std_complete_future["subregion"])

# 4) Compute absolute and relative (baseline) 95% CI for ALL FUTURE TIMES/SCENARIOS
#    This keeps the denominator fixed at the historic baseline for every time/exp/ssp.
ci95_abs = (1.96*std_complete_future).rename("ci95_abs")

# guard against zeros in the baseline
den = initial_volume.where(initial_volume != 0)

ci95_rel_baseline = (ci95_abs / den).rename("ci95_rel_baseline")   # fraction (e.g., 0.12 = 12%)
ci95_rel_baseline_pct = (ci95_rel_baseline * 100).rename("ci95_rel_baseline_pct")

# # 5) (Optional) also store relative std (σ / baseline) if you want the raw ratio without 1.96
std_rel_baseline = (std_complete_future / den).rename("std_rel_baseline")
std_rel_baseline_pct = (std_rel_baseline * 100).rename("std_rel_baseline_pct")

# 6) Save full time series for the future (all times/exp/ssp/subregion)
xr.merge([std_complete_future, std_rel_baseline, std_rel_baseline_pct, ci95_abs, ci95_rel_baseline, ci95_rel_baseline_pct]).to_netcdf(
    f"{wd_path}masters/master_V_uncertainty_future_baseline1985_absolute.nc"
)

# 7) If you need a CSV snapshot for a particular year (e.g., 2073)
snap_2073 = xr.merge([std_complete_future, std_rel_baseline, std_rel_baseline_pct, ci95_abs, ci95_rel_baseline, ci95_rel_baseline_pct]).sel(time=2073)

# drop aux time coords if present
drop_coords = [c for c in ["hydro_year","hydro_month","calendar_year","calendar_month"] if c in snap_2073.coords]
snap_2073 = snap_2073.reset_coords(drop_coords, drop=True)

snap_2073_df = snap_2073.to_dataframe()
snap_2073_df.to_csv(f"{wd_path}masters/master_V_uncertainty_2073_baseline1985_absolute.csv")

#%% Cell 4a: Calculate uncertainties for comitted loss - hma total

output_nc_path = os.path.join(wd_path, "masters", "master_comitted_volume_timeseries_individual_member.nc")
tlines = xr.open_dataset(output_nc_path).volume_percent-100 #because calculated as percentage of starting volume and not percentage change

irr_timeseries  = (tlines.sel(experiment="Irr", model="W5E5", scenario="committed", member=0))
noirr_timeseries = (tlines.sel(experiment="NoIrr", scenario="committed").drop_sel(model=["W5E5", "avg"]))

irr_ref = irr_timeseries.reset_coords(drop=True) # Drop conflicting scalar coords from the IRR reference
noirr_al, irr_al = xr.align(noirr_timeseries, irr_ref, join="left", exclude=("model", "member")) #Align on time (keep NOIRR's time grid); this also handles any label mismatches safely
delta = (noirr_al - irr_al).rename("delta") #Subtract: IRR will broadcast across NOIRR's (model, member)
delta_after = delta.sel(time=slice(2015, None))

std_all = delta_after.std(dim=('model','member'), skipna=True, ddof=0)
std_noirr = noirr_al.std(dim=('model','member'), skipna=True, ddof=0)
std_all_2264 = std_all.sel(time=2264)#*1.96
std_noirr_2264 = std_noirr.sel(time=2264)#*1.96



#%% Cell 4b: Calculate uncertainties for comitted loss - subregions

df = pd.read_csv(
    f"{wd_path}masters/master_gdirs_r3_a1_rgi_date_A_V_RGIreg_B_hugo_Vcom.csv")#
                                                                                        

master_ds = df[(~df['sample_id'].str.endswith('0')) |  # Exclude all the model averages ending with 0 except for IPSL
               (df['sample_id'].str.startswith('IPSL'))]
master_ds=master_ds[~master_ds['sample_id'].str.startswith('IPSL')]

master_ds_tot_vol = master_ds.groupby(['rgi_subregion', 'sample_id'], as_index=False).agg({  # calculate the 14 member average for noirr and delta, for irr the first value can be taken as they are all the same
    'V_2264_noirr': 'sum',
    'V_2264_irr': 'sum',
    'V_2014_noirr': 'sum',
    'V_2014_irr': 'sum',
    'V_1985_irr': 'sum',
    'rgi_id': lambda _: "regional total",
    # lamda is anonmous functions, returns 11 member average
    # take first value for all columns that are not in the list
    **{col: 'first' for col in master_ds.columns if col not in ['V_2264_noirr', 'V_2264_irr','V_1985_irr', 'sample_id', 'rgi_id']} #take first for rgi_id, rgi_region, full_name, cenlon, cenlat, rgi_date, rgi_areakm2
})[['rgi_subregion','sample_id','V_2264_irr', 'V_1985_irr','V_2264_noirr']]

# master_ds_tot_vol = master_ds_tot_vol.set_index('rgi_subregion')
initial_irr_ds = master_ds_tot_vol.groupby('rgi_subregion')['V_1985_irr'].first()
den = master_ds_tot_vol['rgi_subregion'].map(initial_irr_ds).replace(0, np.nan)
master_ds_tot_vol['delta_V_2264'] = master_ds_tot_vol['V_2264_irr']-master_ds_tot_vol['V_2264_noirr'] * 100
# master_ds_tot_vol['delta_V_2264_rel_pct'] = master_ds_tot_vol['delta_V_2264'] / den
# optional percent:
# master_ds_tot_vol['delta_V_2264_rel_pct'] = master_ds_tot_vol['delta_V_2264_rel'] * 100

# --- std across sample_id for each subregion (use the aggregated table) ---
value_cols = ['V_2264_irr', 'V_2264_noirr', 'delta_V_2264']
std_df = (
    master_ds_tot_vol
      .groupby('rgi_subregion', as_index=False)[value_cols]
      .agg(lambda s: s.std(ddof=1))  # sample std
      .rename(columns={c: f'std_{c}' for c in value_cols})
)

# (optional) relative stds with respect to the same baseline
std_df = std_df.merge(
    initial_irr_ds.rename('V_1985_irr_base').reset_index(),
    on='rgi_subregion', how='left'
)
std_df['std_V_2264_irr_rel']  = std_df['std_V_2264_irr'] / std_df['V_1985_irr_base'] #* 1.96 
std_df['std_V_2264_noirr_rel'] = std_df['std_V_2264_noirr'] / std_df['V_1985_irr_base'] #* 1.96
std_df['std_delta_V_2264_rel'] = std_df['std_delta_V_2264'] / std_df['V_1985_irr_base'] #* 1.96/
std_df['std_V_2264_irr_rel_ci95']  = std_df['std_V_2264_irr_rel']* 1.96
std_df['std_V_2264_noirr_rel_ci95'] = std_df['std_V_2264_noirr_rel'] * 1.96
std_df['std_delta_V_2264_rel_cia95'] = std_df['std_delta_V_2264_rel']  * 1.96

std_df.to_csv(f"{wd_path}masters/master_committed_std_subregions.csv")




#%% Cell 5: Calculate runoff uncertainties HMA overall


#when working with melton only
opath_df_monthly_melt = os.path.join(wd_path,'masters','hydro_output_monthly_subregions_melton_only.csv')
opath_df_annual_melt = os.path.join(wd_path,'masters','hydro_output_annual_subregions_melton_only.csv')

#when working with melton and prcp on only
#when working with total runoff|
opath_df_monthly = os.path.join(wd_path,'masters','hydro_output_monthly_subregions.csv')
opath_df_annual = os.path.join(wd_path,'masters','hydro_output_annual_subregions.csv')

opath_df_runoff_shares = os.path.join(wd_path,'masters','hydro_output_monthly_runoff_shares.csv')

#load annual data for all runoff
df_annual = pd.read_csv(opath_df_annual).reset_index()
df_monthly = pd.read_csv(opath_df_monthly, dtype={'ssp': str, 'rgi_subregion': str, 'experiment': str})[['year','month','experiment','ssp','member', 'runoff','rgi_subregion']]

#load annual data for only melt
df_annual_melt = pd.read_csv(opath_df_annual_melt).reset_index()
df_monthly_melt = pd.read_csv(opath_df_monthly_melt, dtype={'ssp': str, 'rgi_subregion': str, 'experiment': str})[['year','month','experiment','ssp','member', 'runoff','rgi_subregion']]
 
#process data for total runoff                  
# df_avg_monthly = (df_monthly.groupby(['year','month', 'experiment', 'ssp', 'rgi_subregion',])['runoff'].mean().reset_index()) #average over members
df_avg_monthly = (df_monthly.groupby(['year','month', 'experiment', 'ssp', 'rgi_subregion',])['runoff'].median().reset_index()) #average over members
df_avg_monthly = df_avg_monthly[df_avg_monthly['runoff'] != 0.00] #exclude zeros as last year of simulation does not work
# df_avg_jja = df_avg_monthly[df_avg_monthly['month'].isin([6, 7, 8])]
df_avg_jja = df_avg_monthly[df_avg_monthly['month'].isin(np.arange(1,13,1))] #if we want full year
df_avg_subregions = (df_avg_jja.groupby(['year', 'experiment', 'ssp', 'rgi_subregion'])['runoff'].sum().reset_index()) #sum yo annual runoff for all subregions per year
df_avg_subregions = df_avg_subregions[~df_avg_subregions['year'].isin([2074])] 
df_avg_all = (df_avg_subregions.groupby(['year', 'experiment', 'ssp'])['runoff'].sum().reset_index()) #sum for all subregions in HMA

df_monthly_std_in = df_monthly[df_monthly['runoff'] != 0.00] #exclude zeros as last year of simulation does not work
df_avg_monthly_for_std = (df_monthly_std_in.groupby(['year', 'experiment', 'ssp', 'member'])['runoff'].sum().reset_index()) #calculate annual sum of runoff for the different members for hma overall
df_avg_monthly_for_std = df_avg_monthly_for_std[~df_avg_monthly_for_std['year'].isin([2074])] 
df_avg_annual_std_all = (df_avg_monthly_for_std.groupby(['year', 'experiment', 'ssp'])['runoff'].std().reset_index()) #take std before rolling mean (otherwise std is smoothened too much, losing variability)
df_avg_annual_std_all['runoff'] = df_avg_annual_std_all['runoff'].fillna(0) #hist irr is nan because only 1 member - make zero for plotting


#process data for melt only runoff
# df_avg_monthly_melt = (df_monthly_melt.groupby(['year','month', 'experiment', 'ssp', 'rgi_subregion',])['runoff'].mean().reset_index()) #average over members
df_avg_monthly_melt = (df_monthly_melt.groupby(['year','month', 'experiment', 'ssp', 'rgi_subregion',])['runoff'].median().reset_index()) #median over members
df_avg_monthly_melt = df_avg_monthly_melt[df_avg_monthly_melt['runoff'] != 0.00]
# df_avg_jja_melt = df_avg_monthly_melt[df_avg_monthly_melt['month'].isin([6, 7, 8])]
df_avg_jja_melt = df_avg_monthly_melt[df_avg_monthly_melt['month'].isin(np.arange(1,13,1))] #if we want full year
df_avg_subregions_melt = (df_avg_jja_melt.groupby(['year', 'experiment', 'ssp', 'rgi_subregion'])['runoff'].sum().reset_index()) #sum yo annual runoff for all subregions per year
df_avg_subregions_melt = df_avg_subregions_melt[~df_avg_subregions_melt['year'].isin([2074])] 
df_avg_all_melt = (df_avg_subregions_melt.groupby(['year', 'experiment', 'ssp'])['runoff'].sum().reset_index()) #sum for all subregions in HMA

df_monthly_std_in_melt = df_monthly_melt[df_monthly_melt['runoff'] != 0.00] #exclude zeros as last year of simulation does not work
df_avg_monthly_for_std_melt = (df_monthly_std_in_melt.groupby(['year', 'experiment', 'ssp', 'member'])['runoff'].sum().reset_index()) #calculate annual sum of runoff for the different members for hma overall
df_avg_monthly_for_std_melt = df_avg_monthly_for_std_melt[~df_avg_monthly_for_std_melt['year'].isin([2074])]

df_avg_monthly_for_std_126 = (df_avg_monthly_for_std[df_avg_monthly_for_std['ssp'].isin(['hist', '126'])].copy())
df_avg_monthly_for_std_370 = (df_avg_monthly_for_std_melt[df_avg_monthly_for_std['ssp'].isin(['hist', '370'])].copy())
df_avg_monthly_for_std_melt_126 = (df_avg_monthly_for_std_melt[df_avg_monthly_for_std_melt['ssp'].isin(['hist', '126'])].copy())
df_avg_monthly_for_std_melt_370 = (df_avg_monthly_for_std_melt[df_avg_monthly_for_std_melt['ssp'].isin(['hist', '370'])].copy())

# df_avg_monthly_for_std_melt_126.loc[:, 'ssp'] = '126' #set all ssp values to 126 create one smoothened timeseries from history onward
# df_avg_monthly_for_std_melt_370.loc[:, 'ssp'] = '370'
# df_avg_monthly_for_std_126.loc[:, 'ssp'] = '126' #set all ssp values to 126 create one smoothened timeseries from history onward
# df_avg_monthly_for_std_370.loc[:, 'ssp'] = '370'

cols_group = ['experiment', 'ssp', 'member'] 

df_126 = df_avg_monthly_for_std_126.sort_values(cols_group + ['year']).copy() #sort all values
df_370 = df_avg_monthly_for_std_370.sort_values(cols_group + ['year']).copy()
df_melt_126 = df_avg_monthly_for_std_melt_126.sort_values(cols_group + ['year']).copy() #sort all values
df_melt_370 = df_avg_monthly_for_std_melt_370.sort_values(cols_group + ['year']).copy()

df_126['runoff_smoothed'] = (df_126.groupby(cols_group, group_keys=False)['runoff'].transform(lambda s: s.rolling(window=11, center=True, min_periods=1).mean()))
df_370['runoff_smoothed'] = (df_370.groupby(cols_group, group_keys=False)['runoff'].transform(lambda s: s.rolling(window=11, center=True, min_periods=1).mean()))
df_melt_126['runoff_smoothed'] = (df_melt_126.groupby(cols_group, group_keys=False)['runoff'].transform(lambda s: s.rolling(window=11, center=True, min_periods=1).mean()))
df_melt_370['runoff_smoothed'] = (df_melt_370.groupby(cols_group, group_keys=False)['runoff'].transform(lambda s: s.rolling(window=11, center=True, min_periods=1).mean()))



# df_avg_annual_std_all_melt = (df_avg_monthly_for_std_melt.groupby(['year', 'experiment', 'ssp'])['runoff'].std().reset_index()) #take std before rolling mean (otherwise std is smoothened too much, losing variability)
# df_avg_annual_std_all_melt['runoff'] = df_avg_annual_std_all_melt['runoff'].fillna(0) #hist irr is nan because only 1 member - make zero for plotting
    
    
base_year_all =df_avg_all[df_avg_all['year']==1985][df_avg_all['experiment']=='irr'].runoff.values
#calculate relative runoff
# df_avg_all['runoff_relative']=(df_avg_all['runoff']-base_year_all)/base_year_all
# df_avg_annual_std_all['runoff_relative']=(df_avg_annual_std_all['runoff'])/base_year_all

# base_year_all_melt =df_avg_all_melt[df_avg_all_melt['year']==1985][df_avg_all_melt['experiment']=='irr'].runoff.values
# df_avg_all_melt['runoff_relative']=(df_avg_all_melt['runoff']-base_year_all_melt)/base_year_all_melt #relative to total runoff
# df_avg_annual_std_all_melt['runoff_relative']=(df_avg_annual_std_all_melt['runoff'])/base_year_all_melt
# df_avg_annual_std_all_melt['runoff_relative_smoothened']=(df_avg_annual_std_all_melt['runoff'])/base_year_all_melt
# # df_avg_all_melt['runoff_relative']=(df_avg_all_melt['runoff']-base_year_all)/base_year_all_melt #relative to total runoff

totals=[]

# df = df_avg_monthly_for_std.copy()
df = df_126.copy()
# df = df_370.copy()
# df = df_melt_370.copy()
# df = df_melt_126.copy()
df['experiment'] = df['experiment'].str.lower().replace({'noi': 'noirr'})

# 1) Fixed HMA baseline (scalar): 1985, IRR, hist
base_series = df.query("year == 1985 and experiment == 'irr' and ssp == 'hist'")['runoff']
baseline = float(base_series.iloc[0]) if len(base_series) else np.nan
if baseline == 0: baseline = np.nan  # avoid /0

# 2) HIST: IRR (single) − NOIRR (members) → per-member deltas


irr_hist = (df.query("ssp == 'hist' and experiment == 'irr'")
              .drop(columns='member', errors='ignore')
              .groupby(['year','ssp'], as_index=False)[['runoff','runoff_smoothed']].first()
              .rename(columns={'runoff':'runoff_irr', 'runoff_smoothed': 'runoff_irr_smoothed'}))

noirr_hist = (df.query("ssp == 'hist' and experiment == 'noirr'")
                .rename(columns={'runoff':'runoff_noirr', 'runoff_smoothed': 'runoff_noirr_smoothed'}))

delta_hist = (noirr_hist
                .merge(irr_hist, on=['year','ssp'], how='inner')
                # 1) absolute deltas
                .assign(
                    delta_abs=lambda d: d['runoff_irr'] - d['runoff_noirr'],
                    delta_abs_smoothed=lambda d: d['runoff_irr_smoothed'] - d['runoff_noirr_smoothed'],
                )
                # 2) relative deltas (use baseline once)
                .assign(
                    delta_rel=lambda d: d['delta_abs'] / baseline,
                    delta_rel_pct=lambda d: d['delta_rel'] * 100,
                    delta_rel_smoothed=lambda d: d['delta_abs_smoothed'] / baseline,
                    delta_rel_smoothed_pct=lambda d: d['delta_rel_smoothed'] * 100,
                )
            )

# 3) FUTURE: pair by member
irr_fut = (df.query("ssp != 'hist' and experiment == 'irr'")
             .rename(columns={'runoff':'runoff_irr', 'runoff_smoothed': 'runoff_irr_smoothed'}))
noirr_fut = (df.query("ssp != 'hist' and experiment == 'noirr'")
               .rename(columns={'runoff':'runoff_noirr', 'runoff_smoothed': 'runoff_noirr_smoothed'}))

paired_fut = (noirr_fut
                .merge(irr_fut, on=['year','ssp', 'member'], how='inner')
                # 1) absolute deltas
                .assign(
                    delta_abs=lambda d: d['runoff_irr'] - d['runoff_noirr'],
                    delta_abs_smoothed=lambda d: d['runoff_irr_smoothed'] - d['runoff_noirr_smoothed'],
                )
                # 2) relative deltas (use baseline once)
                .assign(
                    delta_rel=lambda d: d['delta_abs'] / baseline,
                    delta_rel_pct=lambda d: d['delta_rel'] * 100,
                    delta_rel_smoothed=lambda d: d['delta_abs_smoothed'] / baseline,
                    delta_rel_smoothed_pct=lambda d: d['delta_rel_smoothed'] * 100,
                )
            )

# 4) Per-member deltas table (absolute & relative)
delta_detail = pd.concat(
    [delta_hist[['year','ssp','member','delta_abs','delta_rel','delta_rel_pct', 'delta_rel_smoothed', 'delta_rel_smoothed_pct']],
     paired_fut[['year','ssp','member','delta_abs','delta_rel','delta_rel_pct', 'delta_rel_smoothed', 'delta_rel_smoothed_pct']]],
    ignore_index=True
)

# 5) Std over members for each (year, ssp) + 95% CI
std_delta = (delta_detail.groupby(['year','ssp'], as_index=False)
             .agg(std_delta_abs=('delta_abs', lambda s: s.std(ddof=1)),
                  std_delta_rel=('delta_rel', lambda s: s.std(ddof=1)),
                  std_delta_rel_smoothed=('delta_rel_smoothed', lambda s: s.std(ddof=1)),
                  n_members=('member','nunique')))

std_delta['ci95_abs']     = 1.96 * std_delta['std_delta_abs']
std_delta['ci95_rel_pct'] = 1.96 * std_delta['std_delta_rel'] * 100
std_delta['ci95_rel_smooted_pct'] = 1.96 * std_delta['std_delta_rel_smoothed'] * 100

delta_median = (
    delta_detail
      .groupby(['year','ssp'], as_index=False)
      .agg(mean_delta_abs=('delta_abs', 'median'),
           mean_delta_rel=('delta_rel', 'median'),
           mean_delta_rel_pct=('delta_rel_pct', 'median'),
           mean_delta_rel_smoothed=('delta_rel_smoothed', 'median'),
           mean_delta_rel_smoothed_pct=('delta_rel_smoothed_pct', 'median')))

std_delta = std_delta.merge(delta_median, on=['year','ssp'], how='left')

std_delta.to_csv(f'{wd_path}/masters/spread_hist_ssp_totalrunoff_delta_126.csv')
# std_delta.to_csv(f'{wd_path}/masters/spread_hist_ssp_totalrunoff_delta_370.csv')
# std_delta.to_csv(f'{wd_path}/masters/spread_hist_ssp_melt_delta_126.csv')
# std_delta.to_csv(f'{wd_path}/masters/spread_hist_ssp_melt_delta_370.csv')

