# -*- coding: utf-8 -*-
# %% Cell 0: Load data packages
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 15:36:52 2024

@author: magaliponds

This script runs makes plots of the selected 3 regions (13-14-15) with Areas larger than 5km2 (3r_a5)

"""

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


function_directory = "/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/src/03. Glacier simulations"
sys.path.append(function_directory)
# %% Cell 1: Set base parameters
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

colors = {
    "irr": ["#000000", "#555555"],  # Black and dark gray
    # Darker brown and golden yellow for contrast
    "noirr": ["#f5bf03","#fbeaac"],#["#8B5A00", "#D4A017"], #fdde6c
    "noirr_com": ["#E3C565", "#F6E3B0"],  # Lighter, distinguishable tan shades #
    "irr_com": ["#B5B5B5", "#D0D0D0"],  # Light gray, no change
    "cf": ["#004C4C", "#40E0D0"],
    "cf_com": ["#008B8B", "#40E0D0"],
    "cline": ["#e17701", '#ff9408']
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


fig_path = '/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/04. Figures/02. OGGM simulations/01. Modelled output/3r_a5/'
folder_path = '/Users/magaliponds/Documents/00. Programming'
wd_path = f'{folder_path}/03. Modelled perturbation-glacier interactions - R13-15 A+5km2/'
sum_dir = os.path.join(wd_path, 'summary')

# %% Cell 1b: Load gdirs

wd_path_pkls = f'{wd_path}/pkls_subset_success/'

gdirs_3r_a5 = []
for filename in os.listdir(wd_path_pkls):
    if filename.endswith('.pkl'):
        # f'{gdir.rgi_id}.pkl')
        file_path = os.path.join(wd_path_pkls, filename)
        with open(file_path, 'rb') as f:
            gdir = pickle.load(f)
            gdirs_3r_a5.append(gdir)
#%% Cell 2a: map plot - prepare data

"""Process master for map plot """

# Load datasets
df = pd.read_csv(
    f"{wd_path}masters/master_gdirs_r3_a5_rgi_date_A_V_RGIreg_B_T_only.csv")
# df = pd.read_csv(
#     f"{wd_path}masters/master_gdirs_r3_a5_rgi_date_A_V_RGIreg_B_hugo.csv")
master_ds = df[(~df['sample_id'].str.endswith('0')) |  # Exclude all the model averages ending with 0 except for IPSL
               (df['sample_id'].str.startswith('IPSL'))]

# Normalize values
# divide all the B values with 1000 to transform to m w.e. average over 30 yrs
# master_ds[['B_noirr', 'B_irr', 'B_delta_irr', 'B_cf',  "B_delta_cf"]] /= 1000
master_ds.loc[:, ['B_noirr', 'B_irr', 'B_delta_irr']] /= 1000

master_ds = master_ds[['rgi_id', 'rgi_region', 'rgi_subregion', 'full_name', 'cenlon', 'cenlat', 'rgi_date',
                       'rgi_area_km2', 'rgi_volume_km3', 'sample_id', 'B_noirr', 'B_irr', 'B_delta_irr']]

# Define custom aggregation functions for grouping over the 11 member data
aggregation_functions = {
    'rgi_region': 'first',
    'rgi_subregion': 'first',
    'full_name': 'first',
    'cenlon': 'first',
    'cenlat': 'first',
    'rgi_date': 'first',
    'rgi_area_km2': 'first',
    'rgi_volume_km3': 'first',
    'B_noirr': 'mean',
    'B_irr': 'mean',
    'B_delta_irr': 'mean',
    'sample_id': 'first'
}


master_ds_avg = master_ds.groupby(['rgi_id'], as_index=False).agg({
    'B_delta_irr': 'mean',
    'B_noirr': 'mean',
    # lamda is anonmous functions, returns 11 member average
    'sample_id': lambda _: "11 member average",
    # take first value for all columns that are not in the list
    **{col: 'first' for col in master_ds.columns if col not in ['B_noirr', 'B_delta', 'sample_id']}
})

master_ds_avg.to_csv(
    f"{wd_path}masters/master_lon_lat_rgi_id.csv")
# Aggregate data for scatter plot
master_ds_avg['grid_lon'] = np.floor(master_ds_avg['cenlon'])
master_ds_avg['grid_lat'] = np.floor(master_ds_avg['cenlat'])


# Aggregate dataset, area-weighted BDelta Birr and Bnoirr, Sample id is replaced by 11 member average
aggregated_ds = master_ds_avg.groupby(['grid_lon', 'grid_lat'], as_index=False).agg({
    'B_delta_irr': lambda x: (x * master_ds.loc[x.index, 'rgi_area_km2']).sum() / master_ds.loc[x.index, 'rgi_area_km2'].sum(),
    'B_irr': lambda x: (x * master_ds.loc[x.index, 'rgi_area_km2']).sum() / master_ds.loc[x.index, 'rgi_area_km2'].sum(),
    'B_noirr': lambda x: (x * master_ds.loc[x.index, 'rgi_area_km2']).sum() / master_ds.loc[x.index, 'rgi_area_km2'].sum(),
    'rgi_area_km2': 'sum',  # Sum for area
    'rgi_volume_km3': 'sum',  # Sum for volume
    # 'sample_id': lambda _: "11 member average",
    # take first value for all columns that are not in the list
    **{col: 'first' for col in master_ds.columns if col not in ['B_noirr', 'B_irr', 'B_delta', 'sample_id', 'rgi_area_km2', 'rgi_volume_km3']}
})

aggregated_ds.rename(
    columns={'grid_lon': 'lon', 'grid_lat': 'lat'}, inplace=True)

aggregated_ds.to_csv(
    f"{wd_path}masters/complete_master_processed_for_map_plot_T_only.csv")

#%% Cell 2b: map plot - Process ∆ volume data

climate_run_output_irr = xr.open_dataset(os.path.join(
    sum_dir, f'climate_run_output_baseline_W5E5.000.nc'))
initial_volume=climate_run_output_irr.volume.sel(time=1985) #only select first year for initial value

delta_volume_irr = climate_run_output_irr.volume - initial_volume

all_datasets = []
for m, model in enumerate(models_shortlist): #Load the data for other models and calculate respective loss for each 
    for j in range(members_averages[m]):
        sample_id = f"{model}.00{j + 1}" if members_averages[m] > 1 else f"{model}.000"
        
        climate_run_output_noirr = xr.open_dataset(os.path.join(
            sum_dir, f'climate_run_output_perturbed_{sample_id}_T_only.nc'))
        delta_volume_noirr = climate_run_output_noirr.volume - initial_volume
        delta_volume_noirr_rel = xr.where(delta_volume_irr != 0, delta_volume_noirr / delta_volume_irr, 0)
                
        yearly_mean_volume = climate_run_output_noirr.mean(dim='time')
        
        ds_new = xr.Dataset({
            "delta_volume_irr": delta_volume_irr,
            "delta_volume_noirr": delta_volume_noirr,
            "sample_id": [sample_id],
            "delta_volume_noirr_rel": delta_volume_noirr_rel
        })

        # Add a new coordinate for sample_id
        # ds_new = ds_new.expand_dims(dim={"sample_id": [sample_id]})

        # Append to list
        all_datasets.append(ds_new)
        
volume_change_ds = xr.concat(all_datasets, dim="sample_id")
volume_change_ds["delta_volume_noirr_rel"].attrs["description"] = "Relative volume change ∆ NoIrr (resp. Irr-1985)/ ∆ Irr (resp. Irr-1985)"
# Group by rgi_id and time, then compute the mean for numeric variables
volume_ds_avg = volume_change_ds.mean(dim="sample_id").mean(dim="time") #30-yr average values for all 14 members in the model ensemble


opath_bymember = os.path.join(wd_path,"masters", 'delta_volume_evolution_bymember_T_only.nc')
volume_change_ds.to_netcdf(opath_bymember)

opath_averaged = os.path.join(wd_path,"masters", 'delta_volume_evolution_ensemble_average_T_only.nc')
opath_averaged_csv = os.path.join(wd_path,"masters", 'delta_volume_evolution_ensemble_average_T_only.csv')
volume_ds_avg.to_netcdf(opath_averaged)

volume_df_avg = volume_ds_avg.to_dataframe()

master_df = pd.read_csv(
    f"{wd_path}masters/master_lon_lat_rgi_id.csv")[['rgi_id', 'cenlon','cenlat', 'rgi_area_km2']]
volume_df_avg = volume_df_avg.merge(
    master_df, on="rgi_id", how="left")

# Aggregate data for scatter plot
volume_df_avg['grid_lon'] = np.floor(volume_df_avg['cenlon'])
volume_df_avg['grid_lat'] = np.floor(volume_df_avg['cenlat'])

# Aggregate dataset, area-weighted BDelta Birr and Bnoirr, Sample id is replaced by 11 member average
volume_df_avg = volume_df_avg.groupby(['grid_lon', 'grid_lat'], as_index=False).agg({
    'delta_volume_irr': lambda x: (x * volume_df_avg.loc[x.index, 'rgi_area_km2']).sum() / volume_df_avg.loc[x.index, 'rgi_area_km2'].sum(),
    'delta_volume_noirr': lambda x: (x * volume_df_avg.loc[x.index, 'rgi_area_km2']).sum() / volume_df_avg.loc[x.index, 'rgi_area_km2'].sum(),
    'delta_volume_noirr_rel': lambda x: (x * volume_df_avg.loc[x.index, 'rgi_area_km2']).sum() / volume_df_avg.loc[x.index, 'rgi_area_km2'].sum(),
    'rgi_area_km2': 'sum',  # Sum for area
    # 'sample_id': lambda _: "11 member average",
    # take first value for all columns that are not in the list
    **{col: 'first' for col in volume_df_avg.columns if col not in ['delta_volume_irr', 'delta_volume_noirr', 'delta_volume_noirr_rel', 'rgi_area_km2']}
})

volume_df_avg.rename(
    columns={'grid_lon': 'lon', 'grid_lat': 'lat'}, inplace=True)

volume_df_avg.to_csv(opath_averaged_csv)





#%% Cell 2c: map plot - prepare volume evolution

#tracker 
regions = [13, 14, 15]
subregions = [9, 3, 3]

region_data = []
members_averages = [2, 3,  3,  5, 1]
models_shortlist = ["E3SM", "CESM2",  "NorESM",  "CNRM", "IPSL-CM6"]

        
# Storage for time series
subregion_series = {}  # subregion → DataArray[time, member]
global_series = []     # total average over all subregions per member
members_all = []       # (model, member_id) pairs to track 14-member average

for m, model in enumerate(models_shortlist):
    for j in range(members_averages[m]):
        member_id = f"{model}.00{j + 1}" if members_averages[m] > 1 else f"{model}.000"
        sample_id = member_id  # used in filename

        # Open NetCDF file
        path = os.path.join(sum_dir, f'climate_run_output_perturbed_{sample_id}_P_only.nc')
        ds = xr.open_dataset(path, engine="h5netcdf")
        members_all.append(member_id)

        # Per subregion
        for reg, region in enumerate(regions):
            for sub in range(subregions[reg]):
                region_id = f"{region}-0{sub+1}"

                subregion_ds = master_ds_avg[
                    master_ds_avg['rgi_subregion'].str.contains(region_id)
                ]

                if subregion_ds.empty:
                    continue

                # Filter data
                ds_filtered = ds.where(
                    ds['rgi_id'].isin(subregion_ds.rgi_id.values), drop=True)

                # Sum volume over glaciers in this subregion
                vol_timeseries = ds_filtered["volume"].sum(dim="rgi_id")  # dims: time

                key = region_id
                if key not in subregion_series:
                    subregion_series[key] = []

                subregion_series[key].append(vol_timeseries.assign_coords(member=member_id))

        # Add global average (sum over all glaciers, no filter)
        global_total = ds["volume"].sum(dim="rgi_id")  # dims: time
        global_series.append(global_total.assign_coords(member=member_id))

# ---------------------------------------
#  Convert per-subregion data to Dataset
subregion_datasets = {}
for region_id, series_list in subregion_series.items():
    da = xr.concat(series_list, dim="member")
    da = da.transpose("time", "member")
    da_avg = da.mean(dim="member").assign_coords(member="14-member-avg")
    da_full = xr.concat([da, da_avg.expand_dims(dim="member")], dim="member")
    subregion_datasets[region_id] = da_full

# Merge all subregions into one dataset with region_id as a new dimension
ds_subregions = xr.concat(
    [ds.expand_dims(subregion=[key]) for key, ds in subregion_datasets.items()],
    dim="subregion"
)

# ---------------------------------------
# Create global average dataset
global_all = xr.concat(global_series, dim="member").transpose("time", "member")
global_avg = global_all.mean(dim="member").assign_coords(member="14-member-avg")
global_full = xr.concat([global_all, global_avg.expand_dims(dim="member")], dim="member")

# ---------------------------------------
# Now you have:
# - ds_subregions: [time, member, subregion]
# - global_full:   [time, member]

# You can save them if needed:
ds_subregions.to_netcdf(f"{wd_path}masters/master_volume_ts_subregions_members_P_only.nc")
global_full.to_netcdf(f"{wd_path}masters/master_volume_ts_global_members_P_only.nc")


#%% Cell 3: Create map plot - ∆B or ∆V

members_averages = [2, 3,  3,  5, 1]
models_shortlist = ["E3SM", "CESM2",  "NorESM",  "CNRM"]#, "IPSL-CM6"]


""" Load data"""
aggregated_ds = pd.read_csv(
    f"{wd_path}masters/complete_master_processed_for_map_plot_P_only.csv")

aggregated_ds_vol = pd.read_csv(
    f"{wd_path}masters/delta_volume_evolution_ensemble_average_P_only.csv")

gdf = gpd.GeoDataFrame(aggregated_ds, geometry=gpd.points_from_xy(
    aggregated_ds['lon'] +0.5 , aggregated_ds['lat'] + 0.5),
    crs="EPSG:4326")

gdf_volume = gpd.GeoDataFrame(aggregated_ds_vol, geometry=gpd.points_from_xy(
    aggregated_ds_vol['lon'] +0.5 , aggregated_ds_vol['lat'] + 0.5),
    crs="EPSG:4326")

""" Plot shaded relief """
fig, ax = plt.subplots(figsize=(15,12), subplot_kw={
                       'projection': ccrs.PlateCarree()})
dem_path = "/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/01. Input files/02. DEM/ETOPO2v2c_f4.nc"
with xr.open_dataset(dem_path) as ds:
    dem=ds['z']

# Print dataset info to find variable names

dem = dem.where((dem.x >= 54.5) & (dem.x <= 115) & (dem.y >= 16) & (dem.y <= 55), drop=True)
dx =(dem['x'][1]-dem['x'][0]).values
dy =(dem['y'][1]-dem['y'][0]).values
z = dem.values
x, y = np.meshgrid(dem.x, dem.y)

# Create a LightSource object for hillshading
ls = LightSource(azdeg=315, altdeg=45)


# Overlay the hillshade (for relief effect)
im = ax.imshow(ls.hillshade(z, vert_exag=10, dx=0.01, dy=0.01),extent=[dem.x.min(), dem.x.max(), dem.y.min(), dem.y.max()],
          transform=ccrs.PlateCarree(), alpha=0.1, cmap="grey_r")  # Adjust alpha for effect

# Add geographical features
# ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
# ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle="--")
# ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='none')
ax.add_feature(cfeature.RIVERS, edgecolor='blue', facecolor='none', linewidth=0.6, alpha=0.6)

"""Set up plot incl shapefile"""


subregions_path = "/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/02. QGIS/RGI outlines/GTN-G_O2regions_selected_clipped.shp"
subregions = gpd.read_file(subregions_path).to_crs('EPSG:4326')
ax.spines['geo'].set_visible(False)

# Optionally, remove gridlines
# Remove gridlines properly
gl = ax.gridlines(draw_labels=False)  # Create gridlines without labels
gl.xlines = False  # Remove longitude lines
gl.ylines = False  # Remove latitude lines


""" Plot subregions"""
# define movements for the annotation of subregions
movements = {
    '13-01': [0, -0.2], #A
    '13-02': [4.8, 2], #
    '13-03': [-0.2, -0.6], #B
    '13-04': [-0.6, 0], #C
    '13-05': [-4.6,2.5],#-0.2, 0.8], #E
    '13-06': [-10, 0], #F
    '13-07': [-1, 1], #G
    '13-08': [-2, -0.5], #J
    '13-09': [1.5, 0], #O
    '14-01': [0.5, -0.8], #H
    '14-02': [2,-1.8],#[1, -0.4], #I
    '14-03': [3.4, -3.5], #K
    '15-01': [0, 0], #L
    '15-02': [-2, -1], #M
    '15-03': [3.5, 4.5], #N
}
# annotate subregions
# i=0
   
# Create a ListedColormap with some colors (example)
""" Include the Mass Balance plot including color bar"""

h_leg = 0.65
w_leg = 0.646

cax = fig.add_axes([0.76, h_leg-0.015, 0.03, 0.06])  # [left, bottom, width, height]

scatter_data = "B"

# Define vmin and vmax for the color scale (asymmetric)
vmin, vmax = -0.2, 0.7  # Asymmetric range
start =(abs(vmin)/(vmax-vmin))
#Trim cmap so that white is at 0
n_colors=256
half=n_colors//2
original_cmap = plt.cm.bwr.reversed()  # Blue-White-Red colormap

colors_trimmed = [
    (0.0, "lightcoral"),   # lighter red at -0.2
    (0.222, "white"),   # white at 0.0 → (0 - (-0.2)) / (0.7 - (-0.2)) = ~0.222
    (1.0, "#0203e2")    # dark blue at 0.7
]

# Create custom colormap
custom_cmap = LinearSegmentedColormap.from_list("custom_trimmed_rwb", colors_trimmed, N=256)

# Normalize to match data range
norm = Normalize(vmin=-0.2, vmax=0.7)

# Create the colorbar
cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=custom_cmap ),
                    cax=cax, orientation="vertical")

# Define **evenly spaced** tick positions
num_ticks = 2  # Adjust as needed
tick_values = np.linspace(vmin, vmax, num_ticks)  # Ensures equal spacing

# Apply ticks and labels
cbar.set_ticks(tick_values)
cbar.set_ticklabels([f'{b:.1f}' for b in tick_values])
# Define the boundaries for each color block
cbar.ax.tick_params(labelsize=18) 


# Set the colorbar label
cbar.set_label('∆B$_{Irr}$ \nm yr$^{-1}$', fontsize=18, ha='left', rotation=0, labelpad=0, linespacing=0.8)
cbar.ax.yaxis.label.set_position((1.1, 1.4))  # (X, Y) - Adjust Y to move up
cbar.ax.yaxis.set_label_position('left')

gdf["marker_size"]=(np.sqrt(gdf['rgi_area_km2'])*5)**1.3
scatter = ax.scatter(gdf.geometry.x, gdf.geometry.y,
                     s=gdf['marker_size'], c=gdf['B_delta_irr'], cmap=custom_cmap, norm=norm, edgecolor='k', alpha=0.8)

#plot subregion lines



for attribute, subregion in subregions.groupby('o2region'):
   
# Uncomment for coloring of specific subregions

    linecolor = "black"  # if subregion.o2region.values in highlighted_subregions else "black"
    # subregion.plot(ax=ax, edgecolor='yellow', linewidth=4,
    subregion.plot(ax=ax, edgecolor='black', linewidth=1.2,
                  facecolor='none') # facecolor=region_colors[i],, alpha=0.4 # Plot the subregion
    # i+=1
    
""" Add volume legend"""
custom_sizes = [200, 2000]#500, 1000, 2000, 3000]  # Example sizes for the legend (area)
size_labels = [f"{size:.0f}" for size in custom_sizes]  # Create labels for sizes

# Create legend handles (scatter points with different sizes)
legend_handles = [
    plt.scatter([], [], s=(np.sqrt(size)*5)**1.3, edgecolor='k', facecolor='none')
    for size in custom_sizes
]  # Adjust size factor if needed
text_handles = [Line2D([0], [0], linestyle="none", label=label) for label in size_labels]



# Create a separate axis for the legend
cax_legend = fig.add_axes([w_leg -0.015, h_leg, 0.15, 0.1])  # [left, bottom, width, height]

# Remove axis visuals
cax_legend.set_frame_on(False)  # Hide frame
cax_legend.set_xticks([])  # Remove x-ticks
cax_legend.set_yticks([])  # Remove y-ticks

# Add legend to the separate axis
legend = cax_legend.legend(
    legend_handles, size_labels, loc="center",
    fontsize=18, ncol=1,frameon=False, title="Total Area \n km$^2$", title_fontsize=18, scatterpoints=1, labelspacing=1)

    # handler_map={tuple: HandlerTuple(ndivide=None)}, labelspacing=0.3, columnspacing=1, handletextpad=0.3)

# Adjust legend marker positions (stacked effect)

""" Add call out plots for the different regions"""
# Define and iterate over grid layout
layout = [["13-01", "13-03", "13-04", "14-02", "13-05", "13-06"], ["13-02", "", "", "","", "13-07"], [
    "14-01", "", "", "","", "13-08"], ["","","", "","", "13-09"], ["","","14-03", "15-01", "15-02", "15-03"]]
w = 0.14
w_space=0.014#0.035
h = w/0.17*0.17#w/0.17*0.15 
h_space=0.04 #0.06
start_x=0.07#-0.12
start_y = 0.82 #0.9
y_buffer=0.03 #0.05
nr_cols=6
nr_rows=5

# grid_positions = [[-0.15+ col * (w + 0.06), 0.82 - (h + 0.09) * row - 0.05, w, h]
#                   if layout[row][col] else None for row in range(4) for col in range(5)]
grid_positions = [[start_x + col * (w + w_space), start_y - (h + h_space) * row - y_buffer, w, h]
                  if layout[row][col] else None for row in range(nr_rows) for col in range(nr_cols)]

x_min, x_max = start_x, start_x + nr_cols * (w + w_space)
y_min, y_max = start_y, start_y - (h + h_space) * nr_rows - y_buffer


subregion_ids = list(string.ascii_uppercase)  
i=0
# print(subregion_ids)  

ds_subregions = xr.open_dataset(f"{wd_path}masters/master_volume_ts_subregions_members_P_only.nc")
global_full = xr.open_dataset(f"{wd_path}masters/master_volume_ts_global_members_P_only.nc")

for idx, pos in enumerate(grid_positions):
    if pos:
        subregion_id=subregion_ids[i]
        i+=1
        ax_callout = fig.add_axes(pos, ylim=(-30,15),facecolor='white')#'#E9E9E9'whitesmoke
    
        region_id = layout[idx // nr_cols][idx % nr_cols] #index columns and rows, // is rounded by full nrs
        
        
        # Find the corresponding subregion to add axes
        subregion = subregions[subregions['o2region'] == region_id] #convert to meters
        #load data in the plots
        subregion_ds = master_ds_avg[master_ds_avg['rgi_subregion'].str.contains(
            f"{region_id}")]
        mask = (subregion_ds["rgi_subregion"] == region_id)
        subregion_name = subregion_ds.full_name.iloc[0] #if error reload data above

        # Baseline and model plotting
        baseline_path = os.path.join(
            wd_path, "summary", f"climate_run_output_baseline_W5E5.000.nc")
        baseline = xr.open_dataset(baseline_path)
        rgi_ids_in_baseline = baseline['rgi_id'].values
        matching_rgi_ids = np.intersect1d(
            rgi_ids_in_baseline, subregion_ds.rgi_id.values)
        baseline_filtered = baseline.sel(rgi_id=matching_rgi_ids)
        # Check if there are any matching rgi_id values
        # Ensure that rgi_id values exist in both datasets
        
        initial_volume = baseline_filtered['volume'].sum(dim="rgi_id")[0].values

        # Plot model member data
        ax_callout.plot(baseline_filtered["time"].values, baseline_filtered['volume'].sum(dim="rgi_id")/initial_volume*100-100,
                        label="W5E5.000", color=colors['irr'][0], linewidth=2, zorder=15)
        filtered_member_data = []
        
        
        # Temporarily commented to speed up the plotting

        for m, model in enumerate(models_shortlist):
            for j in range(members_averages[m]):
                sample_id = f"{model}.00{j + 1}" if members_averages[m] > 1 else f"{model}.000"
                # with xr.open_dataset(os.path.join(
                #     sum_dir, f'climate_run_output_perturbed_{sample_id}.nc')) as climate_run_output:
                #     climate_run_output = climate_run_output.where(
                #         climate_run_output['rgi_id'].isin(subregion_ds.rgi_id.values), drop=True)
                climate_run_output = ds_subregions.sel(member=sample_id).sel(subregion=region_id) 
                # ax_callout.plot(climate_run_output["time"].values, climate_run_output["volume"]/initial_volume*100-100, label=sample_id, color="grey", linewidth=1, linestyle="dotted")
                # ax_callout.plot(climate_run_output["time"].values, climate_run_output["volume"].sum(
                #     dim="rgi_id")/initial_volume*100-100, label=sample_id, color="grey", linewidth=1, linestyle="dotted")
                # filtered_member_data.append(
                #     climate_run_output["volume"].sum(dim="rgi_id").values/initial_volume*100-100)
                

        # Mean and range plotting
        mean_values = ds_subregions.sel(subregion=region_id).sel(member='14-member-avg').volume.values/initial_volume*100-100 #np.mean(filtered_member_data, axis=0).flatten()
        min_values = ds_subregions.sel(subregion=region_id).min(dim="member").volume.values/initial_volume*100-100 #np.min(filtered_member_data, axis=0).flatten()
        max_values = ds_subregions.sel(subregion=region_id).max(dim="member").volume.values/initial_volume*100-100# np.max(filtered_member_data, axis=0).flatten()
        std_values = ((ds_subregions.sel(subregion=region_id).volume / initial_volume) * 100 - 100).std(dim="member").values
        ax_callout.plot(climate_run_output["time"].values, mean_values,
                        color=colors['noirr'][0], linestyle='solid', lw=2, label=f"{sum(members_averages)}-member average")
        # ax_callout.fill_between(
        #     climate_run_output["time"].values, min_values, max_values, color=colors['noirr'][1], alpha=0.3)#color="lightblue", alpha=0.3)
        ax_callout.fill_between(climate_run_output["time"].values, (mean_values-std_values), (mean_values+std_values), color=colors['noirr'][1], alpha=0.3, label=r"Historical NoIrr 1$\sigma$" )
        ax_callout.plot(climate_run_output["time"].values, np.ones(len(climate_run_output["time"]))*100-100, color="black", linewidth=1, linestyle="dashed")

        # Subplot formatting
        if region_id=="13-02":
            subregion_title ="Pamir"
        elif region_id=="13-04":
            subregion_title ="East Tien Shan"
        elif region_id=="13-06":
            subregion_title ="East Kun Lun"
        else:
            subregion_title=subregion_name
            
        ax_callout.set_title(f"{subregion_title}", bbox=dict(
            facecolor='none', edgecolor='none', pad=1), fontsize=18) # fontweight="bold"
        # Count the number of glaciers (assuming each 'rgi_id' represents a glacier)
        glacier_count = subregion_ds['rgi_id'].nunique()
        subregion_volume = round(subregion_ds['rgi_volume_km3'].sum())
        # Add number of glaciers as a text annotation in the lower left corner
        ax_callout.text(0.05, 0.05, f"#:{glacier_count}, V:{subregion_volume}",
                        transform=ax_callout.transAxes, fontsize=18, verticalalignment='bottom', fontstyle='italic')
        ax_callout.text(0.85, 0.8, f"{subregion_id}",
                        transform=ax_callout.transAxes, fontsize=18, verticalalignment='bottom'  ,fontweight='bold')
        
        boundary = subregion.geometry.boundary.iloc[0]
        if isinstance(boundary, MultiLineString):
            boundary = list(boundary.geoms)[0]
        boundary_coords = list(boundary.coords)
        boundary_x, boundary_y = boundary_coords[0]  # First point on the boundary
        boundary_x -= movements[region_id][0]
        boundary_y -= movements[region_id][1]
        # Annotate or place text near the boundary
        ax.text(boundary_x, boundary_y, f"{subregion_id}",
                horizontalalignment='center', fontsize=18, color='black', fontweight='bold')
       
        
        ax_callout.tick_params(axis='x', labelbottom=False)
        if idx % nr_cols != 0:
            ax_callout.tick_params(axis='y', labelleft=False)
        ax_callout.xaxis.set_tick_params(labelsize=18)
        ax_callout.yaxis.set_tick_params(labelsize=18)
            
        callout_x, callout_y, callout_w, callout_h = pos

# Create a new figure for the\small legend plot
fig_legend = fig.add_axes([start_x, 0.069, w*2.1, h*2.2], ylim=(-8,3), facecolor="white")##e9e9e9")  # make twice as large as the callout plots #w*2.4, h*2.15
total_initial_volume = baseline['volume'].sum(dim="rgi_id")[0].values
vol_display="relative"

fig_legend.plot(baseline["time"].values, baseline['volume'].sum(dim="rgi_id")/total_initial_volume*100-100,
                label="Historical, W5E5", color= colors['irr'][0], linewidth=2, zorder=15)
member_data = []


# Temporarily commented to speed up the plotting

for m, model in enumerate(models_shortlist):
    for j in range(members_averages[m]):
        if j ==1 and m==1:
            legendlabel = 'Historical NoIrr member'
        else:
            legendlabel = '_nolegend_'
        sample_id = f"{model}.00{j + 1}" if members_averages[m] > 1 else f"{model}.000"
        # with xr.open_dataset(os.path.join(
        #     sum_dir, f'climate_run_output_perturbed_{sample_id}.nc')) as climate_run_output:
        climate_run_output = global_full.sel(member=sample_id)
        fig_legend.plot(climate_run_output["time"].values, climate_run_output["volume"]/total_initial_volume*100-100, label=legendlabel, color=colors['noirr'][0], linewidth=2, linestyle="dotted")
            # fig_legend.plot(climate_run_output["time"].values, climate_run_output["volume"].sum(
            #     dim="rgi_id")/total_initial_volume*100-100, label=legendlabel, color=colors['noirr'][0], linewidth=2, linestyle="dotted")
        # member_data.append(
        #     climate_run_output["volume"].sum(dim="rgi_id").values/total_initial_volume*100-100)

# Mean and range plotting
mean_values = global_full.sel(member='14-member-avg').volume.values/total_initial_volume*100-100 #np.mean(member_data, axis=0).flatten()
min_values = global_full.volume.min(dim='member').values/total_initial_volume*100-100 #np.min(member_data, axis=0).flatten()
max_values = global_full.volume.max(dim='member').values/total_initial_volume*100-100 #np.max(member_data, axis=0).flatten()
std_values = ((global_full.volume / global_full.volume.isel(time=0)) * 100 - 100).std(dim="member").values
# std_values = global_full.volume.std(dim='member').values #np.std(member_data, axis=0).flatten()  # Replace min with std

fig_legend.plot(climate_run_output["time"].values, mean_values,
                color=colors['noirr'][0], linestyle='solid', lw=2, label=f"Historical NoIrr avg.")
fig_legend.fill_between(climate_run_output["time"].values, (mean_values-std_values), (mean_values+std_values), color=colors['noirr'][1], alpha=0.3, label=r"Historical NoIrr 1$\sigma$" )
fig_legend.plot(climate_run_output["time"].values, np.zeros(len(climate_run_output["time"])), color="k", linewidth=1, linestyle="dashed")

total_nr = len(master_ds_avg)
total_volume = round(master_ds_avg['rgi_volume_km3'].sum())
# Annotate number of glaciers (just for example)
fig_legend.text(0.05, 0.05, f'#:{total_nr}, V:{total_volume} km$^3$', transform=fig_legend.transAxes,
                fontsize=18, verticalalignment='bottom', horizontalalignment='left',
                bbox=dict(facecolor='none', edgecolor='none', pad=2), fontstyle='italic')
handles, labels = fig_legend.get_legend_handles_labels()

# Define the custom order (indices correspond to handles/labels order)
custom_order = [0,3,2,1]#[0,1,2]#,1]#[0,3,2,1]  # Example: Rearrange items by index

# Apply the custom order
fig_legend.legend([handles[i] for i in custom_order], 
          [labels[i] for i in custom_order], 
          loc='lower left', bbox_to_anchor=(0.0, 0.1), ncol=1, fontsize=18,
                            frameon=False, labelspacing=0.4)

# Set labels for axes
# fig_legend.set_xlabel('Time', fontsize=20, labelpad=10 )#fontweight="bold"
fig_legend.set_ylabel('∆ Volume [%, vs. 1985]', fontsize=18, labelpad=10) #, fontweight="bold"
fig_legend.set_title('High Mountain Asia', fontsize=18, fontweight='bold')
fig_legend.xaxis.set_tick_params(labelsize=18)
fig_legend.yaxis.set_tick_params(labelsize=18)

# Remove tick marks but keep the tick labels


fig.tight_layout()
fig_folder = os.path.join(fig_path, "04. Map")
os.makedirs(fig_folder, exist_ok=True)
fig_path = '/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/04. Figures/99. Final Figures/'

plt.savefig(f"{fig_path}/00. Appendix/Map_Plot_sA_c{scatter_data}_boxV_IRR_1985_2014_P_only.png", dpi=300, bbox_inches="tight", pad_inches=0.1)



plt.show()


#%% Table
# Specify Paths

o_folder_data = (
    "/Users/magaliponds/OneDrive - Vrije Universiteit Brussel/1. VUB/02. Coding/01. IRRMIP/03. Data/03. Output Files/02. OGGM/02. Volume Area simulations"
)
o_file_data = f"{o_folder_data}/1985_2014.{timeframe}.delta.Volume.subregions_T_only.csv"
unit="km3"
if unit=="Gt":
    rho=0.85 #Gt/km³
    o_file_data_processed = f"{o_folder_data}/1985_2014.{timeframe}.delta.Volume.subregions_paper_output_table_gt_T_only.csv"

else:
    rho=1
    o_file_data_processed = f"{o_folder_data}/1985_2014.{timeframe}.delta.Volume.subregions_paper_output_table_absolute_T_only.csv"

# Load dataset
ds = pd.read_csv(o_file_data)
ds_subregions = ds[ds.time == 2014.0][[
    "subregion",
    "volume_loss_percentage_irr",
    "volume_loss_percentage_noirr",
    "volume_irr",
    "volume_noirr",
]]

# Process total volumes
ds_total = ds.groupby("time").sum()[[
    "volume_irr",
    "volume_noirr",
]].round(2)

# Calculate total row
df_total = pd.DataFrame({
    "subregion": ["total"],
    "volume_loss_percentage_irr": ((ds_total.loc[2014.0, "volume_irr"] - ds_total.loc[1985.0, "volume_irr"]) / ds_total.loc[1985.0, "volume_irr"]) * 100,
    "volume_loss_percentage_noirr": ((ds_total.loc[2014.0, "volume_noirr"] - ds_total.loc[1985.0, "volume_irr"]) / ds_total.loc[1985.0, "volume_irr"]) * 100,
    "volume_irr": (ds_total.loc[2014.0, "volume_irr"] - ds_total.loc[1985.0, "volume_irr"])*rho,
    "volume_noirr": (ds_total.loc[2014.0, "volume_noirr"] - ds_total.loc[1985.0, "volume_irr"])*rho
})

# Combine subregions with total
ds_all_losses = pd.concat([ds_subregions, df_total], ignore_index=True)

# Calculate deltas
ds_all_losses["delta_irr"] = ds_all_losses["volume_loss_percentage_irr"] - ds_all_losses["volume_loss_percentage_noirr"]
ds_all_losses["delta_irr_abs"] = ds_all_losses["volume_irr"] - ds_all_losses["volume_noirr"]

# Load RGI data
df = pd.read_csv(
    f"{wd_path}masters/master_gdirs_r3_a5_rgi_date_A_V_RGIreg_B_P_only.csv")
df["subregion"] = df["rgi_subregion"].str.replace(
    "-", ".").str.strip().str.lower()


# Aggregate RGI data
areas = df.groupby("subregion")["rgi_area_km2"].sum().reset_index(name="area")
volume = df.groupby("subregion")[
    "rgi_volume_km3"].sum().reset_index(name="volume")
nr_glaciers = df.groupby("subregion")[
    "rgi_id"].count().reset_index(name="nr_glaciers")
subregion_name = df.groupby("subregion")["full_name"].first().reset_index(name="name")
subregion_map = {
    "13.02": "Pamir",
    "13.04": "East Tien Shan",
    "13.06": "East Kun Lun"
}
subregion_name["name"] = subregion_name["subregion"].map(subregion_map).fillna(subregion_name["name"])

 
# Add total rows
areas = pd.concat([areas, pd.DataFrame(
    {"subregion": ["total"], "area": [areas["area"].sum()]})], ignore_index=True)
volume = pd.concat([volume, pd.DataFrame({"subregion": ["total"], "volume": [
                   volume["volume"].sum()]})], ignore_index=True)
nr_glaciers = pd.concat([nr_glaciers, pd.DataFrame({"subregion": [
                        "total"], "nr_glaciers": [nr_glaciers["nr_glaciers"].sum()]})], ignore_index=True)
names = pd.concat([subregion_name, pd.DataFrame({"subregion": [
                        "total"], "name": ["High Mountain Asia"]})], ignore_index=True)
areas['area']=areas['area'].round().astype(int)
volume['volume']=volume['volume'].round().astype(int)
# Merge RGI info into ds_all_losses
ds_all_losses['subregion'] = areas["subregion"]
ds_all_losses = ds_all_losses.merge(areas, on="subregion", how="left")
ds_all_losses = ds_all_losses.merge(volume, on="subregion", how="left")
ds_all_losses = ds_all_losses.merge(nr_glaciers, on="subregion", how="left")
ds_all_losses = ds_all_losses.merge(names, on="subregion", how="left")

# ds_all_losses['error_margin_noi_rel'] = ds_all_losses['error_margin_noi_abs'] / \
    # ds_all_losses['volume_noirr']*100
# ds_all_losses['error_margin_cf_rel'] = ds_all_losses['error_margin_cf_abs'] / \
    # ds_all_losses['volume_cf']*100
# Final selection of columns and save to CSV
ds_all_losses = ds_all_losses[[
    "subregion", "name", "nr_glaciers", "area", f"volume",
    "volume_loss_percentage_irr", "volume_irr",
    "volume_loss_percentage_noirr", "volume_noirr",  "delta_irr", "delta_irr_abs"#"error_margin_noi_rel", "error_margin_noi_abs",
    # "volume_loss_percentage_cf", "volume_cf", "error_margin_cf_rel", "error_margin_cf_abs", "delta_cf",

]].round(1)

ds_all_losses.to_csv(o_file_data_processed)   
       
    
    
    
    

#%% Cell6b: TP plot - 1d legend



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

        if scale=="Local":
            shp.plot(ax=ax, edgecolor='black', linewidth=1, facecolor='none')
        ax.coastlines(resolution='10m')

        # Set gridlines and labels
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
        if index in ["A","C"]:
            gl.left_labels = False
        if index in ["A","C","B"]:
            gl.bottom_labels = False

    
        gl.ylabel_style = {'size': 14}
        gl.ylocator = plt.MaxNLocator(nbins=3)
    
        gl.xlabel_style = {'size': 14}
        gl.xlocator = plt.MaxNLocator(nbins=3)
        

        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        """4 Include labels for the cbar and for the y and x axis"""

        ax.set_title(title, fontsize=14)

    return im


# def plot_P_T_perturbations_avg(scale, var, timeframe, mode, diftype, plotsave):
""" Part 0 - Set plotting parameters"""


# y0 = 1985  # if running from 1901 to 1985, than indicate extra id of counterfactual to access the data
# ye = 2014
# if running from 1901 to 1985, than indicate extra id of counterfactual to access the data
y0s = [1985]#, 1901]
yes = [2014]#, 1985]
extra_ids = [""]#, "_counterfactual"]
ptb_types = ["Irr"]#, "NoForcing"]
scale = "Global"
subplots = "off"
all_vars_p = []
all_vars_t = []

all_vars_local_p = []
all_vars_local_t = []
all_model_diffs_avg_p = []
all_model_diffs_avg_t = []
all_member_diff_p = []
all_member_diff_t = []
# "Temperature"]:  # ,"Temperature"]:
for var in ["Temperature", "Precipitation"]:
    for timeframe in ["annual"]:  # :, "seasonal", "monthly"]:
        for mode in ['dif']:  # , 'std']:
            for y, y0 in enumerate(y0s):
                ye = yes[y]
                extra_id = extra_ids[y]
                ptbtype = ptb_types[y]
                if var == "Precipitation" and mode == 'dif':
                    diftypes = ['rel']
                else:
                    diftypes = ['abs']
                for dif in diftypes:
                    print(var, timeframe, dif)
                    diftype = dif
                    timestamps = "YEAR"
                    time_averaging = 'time.year'
                    time_type = 'year'
                    col_wrap = 1

                    # Provide cbar ranges and colors for plots for different variables, modes (dif/std) and difference types (abs/rel)
                    if var == "Precipitation":
                        variable_name = "pr"
                        var_suffix = "PR"
                        if mode == 'dif' and diftype == 'rel':
                            mode_suff = 'total'
                            unit = '%'

                    elif var == "Temperature":
                        variable_name = "tas"
                        var_suffix = "TEMP"
                        mode_suff = 'total'
                        
                        unit = '°C'
                    vmin_t = -1#-1.5
                    vmax_t = 1#1.5
                    zero_scaled_t = (abs(vmin_t)/(abs(vmin_t)+abs(vmax_t)))
                    # colors = [(0, 'xkcd:mocha'), (zero_scaled_t,
                    #                                   'xkcd:white'), (1, 'xkcd:aquamarine')]
                    # custom_cmap_p =  LinearSegmentedColormap.from_list(
                    #     'custom_cmap', colors)
                    brbg11 = plt.get_cmap('BrBG', 11)
                    colors = brbg11([i for i in range(brbg11.N)[2:-2]])  # Get list of 11 RGBA colors
                    custom_cmap_p = LinearSegmentedColormap.from_list('BrBG11', colors) 

                    # If you want a ListedColormap for use in pcolormesh, imshow, etc.
                    # custom_cmap_p = ListedColormap(brbg11.colors)
                    
                    vmin_p = -20
                    vmax_p = 20
                    zero_scaled_p = (abs(vmin_p)/(abs(vmin_p)+abs(vmax_p)))
                           
                    colors = [(0, 'cornflowerblue'), (zero_scaled_p,
                                                      'xkcd:white'), (1, 'xkcd:tomato')]
                    custom_cmap_t = LinearSegmentedColormap.from_list(
                         'custom_cmap', colors)

                    members = [1, 3, 4, 6, 4]
                    # members = [1, 1, 1, 1]
                    all_diff = []  # create a dataset to add all member differences
                    all_model_diffs = []
                    models = ["IPSL-CM6", "E3SM", "CESM2", "CNRM", "NorESM"]
                    lonmin, lonmax, latmin, latmax = [60,109,22,52]
                    all_member_diffs = []
                    for (m, model) in enumerate(models):
                        model_diff = []
                        for member in range(members[m]):
                            member_diff=[]
                            # only open data for non model averages (except for IPSL-CM6 as only one member)
                            if model == "IPSL-CM6" or member != 0:

                                # Part 1: Delete
                                diff_folder_in = f"/Users/magaliponds/OneDrive - Vrije Universiteit Brussel/1. VUB/02. Coding/01. IRRMIP/03. Data/03. Output files/01. Climate data/03. Regridded Perturbations/{var}/{timeframe}/{model}/{member}"
                                ifile_diff = f"{diff_folder_in}/REGRID.{model}.{var_suffix}.DIF.00{member}.{y0}_{ye}_{timeframe}_{diftype}{extra_id}.nc"
                                diff =  xr.open_dataset(ifile_diff)

                                if scale == "Local":  # scale the data to the local scale
                                    diff = diff.where((diff.lon >= 60) & (diff.lon <= 109) & (
                                        diff.lat >= 22) & (diff.lat <= 52), drop=True)
                                    print(diff.lon)
                                    
                                    
                                # loose all the filtered data (nan)
                                diff_clean = diff.dropna(dim="lon", how="all")
                                # include the values in the list for caluclating the avg difference by model
                                model_diff.append(diff_clean)
                                member_diff.append(diff_clean)
                                all_diff.append(diff_clean)
                                all_member_diff = xr.concat(
                                    member_diff, dim="member") 
                                all_member_diffs.append(all_member_diff)
                                
                                # include the values in the list for caluclating the avg difference over all models
                               
                        all_model_diff = xr.concat(
                            model_diff, dim="models").mean(dim="models")  # concatenate all models into a list averaged by model
                        all_model_diffs.append(all_model_diff)
                     # concatenate all models into a list averaged by model
                    
                    all_model_diffs_avg = xr.concat(
                        all_model_diffs, dim="models")  # concatenate all models
                    all_diffs_avg = xr.concat(all_diff, dim="models").mean(
                        dim="models")  # concatenate all members and calculate the mean over all the models
                    all_diffs_avg_local = all_diffs_avg.where((all_diffs_avg.lon >= 60) & (all_diffs_avg.lon <= 109) & (
                        all_diffs_avg.lat >= 22) & (all_diffs_avg.lat <= 52), drop=True)
                    o_folder = f"/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/03. Output files/01. Climate data/08. Processed Perturbations Plots"
                    os.makedirs(o_folder, exist_ok=True)
                    o_file=f"{o_folder}/{scale}_{var}_processed.nc"
                    # all_diffs_avg.to_netcdf(o_file)
                    
                    # Unpack axes
                    if var == "Precipitation":
                        all_vars_p = all_diffs_avg
                        all_vars_local_p = all_diffs_avg_local
                        all_model_diffs_avg_p = all_model_diffs_avg
                        all_member_diff_p = all_member_diffs
                    elif var =="Temperature":
                        all_vars_t = all_diffs_avg
                        all_vars_local_t = all_diffs_avg_local
                        all_model_diffs_avg_t = all_model_diffs_avg
                        all_member_diff_t = all_member_diffs
                    
                


""" Part 2 - Shapefile outline for Karakoram Area to be included"""
# path to  shapefile
shapefile_path = '/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/01. Input files/03. Shapefile/Karakoram/Pan-Tibetan Highlands/Pan-Tibetan Highlands (Liu et al._2022)/Shapefile/Pan-Tibetan Highlands (Liu et al._2022)_P.shp'
shp = gpd.read_file(shapefile_path)
target_crs = 'EPSG:4326'
shp = shp.to_crs(target_crs)

indices = ["A", "B","C","D"]#, "E", "F"]

# Create the mosaic plot

layout = """
AB
CD
"""
figsize = (18.15,10) #based on aspect ratio of global and zoomed

fig, axes = plt.subplot_mosaic(layout, subplot_kw={'projection': ccrs.PlateCarree()},
                               figsize=figsize,
                               gridspec_kw={'wspace': 0.05, 'width_ratios': [2.0, 1.63], 'hspace': 0.05})#,'height_ratios': [1, 1]})
axes['A'].set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
axes['C'].set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
axes['B'].set_extent([lonmin, lonmax, latmin, latmax], crs=ccrs.PlateCarree())
axes['D'].set_extent([lonmin, lonmax, latmin, latmax], crs=ccrs.PlateCarree())
# plot the irrmip difference
im_t_g = plot_subplots(indices[0], subplots, (sum(members)-len(models)+1),
                   all_vars_t, timestamps, axes[indices[0]], shp, custom_cmap_t, timeframe, "Global", "Global", vmin=vmin_t, vmax=vmax_t)
im_t_l = plot_subplots(indices[1], subplots, (sum(members)-len(models)+1),
                   all_vars_local_t, timestamps, axes[indices[1]], shp, custom_cmap_t, timeframe, "Local", "High Mountain Asia", vmin=vmin_t, vmax=vmax_t)
im_p_g = plot_subplots(indices[2], subplots, (sum(members)-len(models)+1),
                   all_vars_p, timestamps, axes[indices[2]], shp, custom_cmap_p, timeframe, "Global", "", vmin=vmin_p, vmax=vmax_p)
im_p_l = plot_subplots(indices[3], subplots, (sum(members)-len(models)+1),
                   all_vars_local_p, timestamps, axes[indices[3]], shp, custom_cmap_p, timeframe, "Local", "", vmin=vmin_p, vmax=vmax_p)

bbox = Rectangle((lonmin, latmin),              # lower-left corner
    lonmax-lonmin,                    # width = 109 - 60
    latmax-latmin,                    # height = 52 - 22
    linewidth=2,
    edgecolor='black',
    facecolor='none',
    linestyle='-',
    transform=ccrs.PlateCarree(),
    zorder=10
)

line_top = ConnectionPatch(
    xyA=(lonmax, latmax), coordsA=axes['A'].transData,
    xyB=(lonmin, latmax), coordsB=axes['B'].transData,
    axesA=axes['A'], axesB=axes['B'],
    color='black', linestyle='-', linewidth=1
)
    
line_bottom = ConnectionPatch(
    xyA=(lonmax, latmin), coordsA=axes['A'].transData,
    xyB=(lonmin, latmin), coordsB=axes['B'].transData,
    axesA=axes['A'], axesB=axes['B'],
    color='black', linestyle='-', linewidth=1
)

# Add to the global axes (they own the lines)
axes['A'].add_artist(line_top)
axes['A'].add_artist(line_bottom)
axes['A'].add_patch(bbox)

bbox2 = Rectangle((lonmin, latmin),              # lower-left corner
    lonmax-lonmin,                    # width = 109 - 60
    latmax-latmin,                    # height = 52 - 22
    linewidth=2,
    edgecolor='black',
    facecolor='none',
    linestyle='-',
    transform=ccrs.PlateCarree(),
    zorder=10
)


line_top = ConnectionPatch(
    xyA=(lonmax, latmax), coordsA=axes['C'].transData,
    xyB=(lonmin, latmax), coordsB=axes['D'].transData,
    axesA=axes['C'], axesB=axes['D'],
    color='black', linestyle='-', linewidth=1
)
    
line_bottom = ConnectionPatch(
    xyA=(lonmax, latmin), coordsA=axes['C'].transData,
    xyB=(lonmin, latmin), coordsB=axes['D'].transData,
    axesA=axes['C'], axesB=axes['D'],
    color='black', linestyle='-', linewidth=1
)
# Add to the global axes (they own the lines)
axes['C'].add_artist(line_top)
axes['C'].add_artist(line_bottom)
axes['C'].add_patch(bbox2)
        
for i, item in enumerate(all_member_diff_p):
    print(f"Item {i}: type = {type(item)}")
    

# signed_p = [np.sign(diff_p['pr']) for diff_p in all_member_diff_p]

# signed_t = [np.sign(diff_t['tas']) for diff_t in all_member_diff_t]
# # Concatenate along 'members' (existing dim)
# sign_diff_p = xr.concat(signed_p, dim='members')
# sign_diff_t = xr.concat(signed_t, dim='members')

sign_diff_p = xr.concat(
    [np.sign(ds['pr']) for ds in all_member_diff_p if isinstance(ds, xr.Dataset) and 'pr' in ds],
    dim='members'
)

sign_diff_t = xr.concat(
    [np.sign(ds['tas']) for ds in all_member_diff_t if isinstance(ds, xr.Dataset) and 'tas' in ds],
    dim='members'
)

# 2. Subset the region of interest: South Asia (approx.)
region_bounds = {
    "lon_min": 60,
    "lon_max": 109,
    "lat_min": 22,
    "lat_max": 52,
}

sign_diff_p = sign_diff_p.sel(
    lon=slice(region_bounds["lon_min"], region_bounds["lon_max"]),
    lat=slice(region_bounds["lat_max"], region_bounds["lat_min"])
)

sign_diff_t = sign_diff_t.sel(
    lon=slice(region_bounds["lon_min"], region_bounds["lon_max"]),
    lat=slice(region_bounds["lat_max"], region_bounds["lat_min"])
)

# 3. Compute agreement mask: where >80% of members agree on sign
within_threshold_p = (abs(sign_diff_p.mean(dim="members")) > 0.8)
within_threshold_t = (abs(sign_diff_t.mean(dim="members")) > 0.8)

# 4. Ensure masks are 2D, squeeze any singleton dimensions
within_threshold_2d_p = within_threshold_p.astype(int).squeeze()
within_threshold_2d_t = within_threshold_t.astype(int).squeeze()

# 5. Prepare meshgrid for plotting
lon2d, lat2d = np.meshgrid(all_diffs_avg_local.lon, all_diffs_avg_local.lat)

# 6. Plotting with contourf and hatching
for key, mask in zip(["B", "D"], [within_threshold_2d_t, within_threshold_2d_p]):
    axes[key].contourf(
        lon2d, lat2d, mask,
        levels=[0.5, 1.5],
        colors='none',
        hatches=['////'],
        transform=ccrs.PlateCarree()
    )




""" 3C Add color bar for entire plot"""

# add cbar in the figure, for overall figure, not subplots
# Define the position of the colorbar
# 
# cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.78])
cbar_ax_t = fig.add_axes([0.9, 0.53, 0.015, 0.35])  # [left, bottom, width, height]
cbar_ax_p = fig.add_axes([0.9, 0.1, 0.015, 0.35])

cbar_t = fig.colorbar(im_t_l, cax=cbar_ax_t, extend='both')
cbar_p = fig.colorbar(im_p_l, cax=cbar_ax_p, extend='both')

# Increase distance between colorbar label and colorbar
cbar_t.ax.yaxis.labelpad = 20
cbar_t.set_label('∆ Temperature [°C]', size=15)
cbar_t.ax.tick_params(labelsize=12)

cbar_p.ax.yaxis.labelpad = 20
cbar_p.set_label('∆ Precipitation [%]', size=15)
cbar_p.ax.tick_params(labelsize=12)

hedging_patch = mpatches.Patch(
    label='> 80% of model members agree on sign of change', hatch='////', edgecolor='black', facecolor='none')

# Add the custom legend to the plot
fig.legend(handles=[hedging_patch], loc='lower center', bbox_to_anchor=(0.53, 0.02), fontsize=14)

# if plotsave == 'save':
o_folder_diff = "/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/04. Figures/01. Climate data/02. Perturbations/"

os.makedirs(f"{o_folder_diff}/", exist_ok=True)
o_file_name = f"{o_folder_diff}/Mosaic.{var}.{y0}_{ye}_{timeframe}_{diftype}{extra_id}.png"
# plt.savefig(o_file_name, bbox_inches='tight')

plt.show()
