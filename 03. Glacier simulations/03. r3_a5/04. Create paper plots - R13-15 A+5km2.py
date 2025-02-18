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

from tqdm import tqdm
import pickle
import sys
import textwrap
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from shapely.geometry import Point
import matplotlib.gridspec as gridspec


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

colors = {
    "irr": ["#000000", "#555555"],  # Black and dark gray
    # Darker brown and golden yellow for contrast
    "noirr": ["#8B5A00", "#D4A017"],
    "noirr_com": ["#E3C565", "#F6E3B0"],  # Lighter, distinguishable tan shades
    "irr_com": ["#B5B5B5", "#D0D0D0"],  # Light gray, no change
    "cf": ["#008B8B", "#40E0D0"],
    "cf_com": ["#008B8B", "#40E0D0"]
}


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
    f"{wd_path}masters/master_gdirs_r3_a5_rgi_date_A_V_RGIreg_B_hugo.csv")
# df = pd.read_csv(
#     f"{wd_path}masters/master_gdirs_r3_a5_rgi_date_A_V_RGIreg_B_hugo.csv")
master_ds = df[(~df['sample_id'].str.endswith('0')) |  # Exclude all the model averages ending with 0 except for IPSL
               (df['sample_id'].str.startswith('IPSL'))]

# Normalize values
# divide all the B values with 1000 to transform to m w.e. average over 30 yrs
master_ds[['B_noirr', 'B_irr', 'B_delta_irr', 'B_cf',  "B_delta_cf"]] /= 1000

master_ds = master_ds[['rgi_id', 'rgi_region', 'rgi_subregion', 'full_name', 'cenlon', 'cenlat', 'rgi_date',
                       'rgi_area_km2', 'rgi_volume_km3', 'sample_id', 'B_noirr', 'B_irr', 'B_delta_irr', 'B_cf',  "B_delta_cf"]]

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
    'B_cf': 'mean',
    'B_delta_cf': 'mean',
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
    'B_delta_cf': lambda x: (x * master_ds.loc[x.index, 'rgi_area_km2']).sum() / master_ds.loc[x.index, 'rgi_area_km2'].sum(),
    'B_irr': lambda x: (x * master_ds.loc[x.index, 'rgi_area_km2']).sum() / master_ds.loc[x.index, 'rgi_area_km2'].sum(),
    'B_noirr': lambda x: (x * master_ds.loc[x.index, 'rgi_area_km2']).sum() / master_ds.loc[x.index, 'rgi_area_km2'].sum(),
    'B_cf': lambda x: (x * master_ds.loc[x.index, 'rgi_area_km2']).sum() / master_ds.loc[x.index, 'rgi_area_km2'].sum(),
    'rgi_area_km2': 'sum',  # Sum for area
    'rgi_volume_km3': 'sum',  # Sum for volume
    # 'sample_id': lambda _: "11 member average",
    # take first value for all columns that are not in the list
    **{col: 'first' for col in master_ds.columns if col not in ['B_noirr', 'B_irr', 'B_delta', 'sample_id', 'rgi_area_km2', 'rgi_volume_km3']}
})

aggregated_ds.rename(
    columns={'grid_lon': 'lon', 'grid_lat': 'lat'}, inplace=True)

aggregated_ds.to_csv(
    f"{wd_path}masters/complete_master_processed_for_map_plot.csv")

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
            sum_dir, f'climate_run_output_perturbed_{sample_id}.nc'))
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


opath_bymember = os.path.join(wd_path,"masters", 'delta_volume_evolution_bymember.nc')
volume_change_ds.to_netcdf(opath_bymember)

opath_averaged = os.path.join(wd_path,"masters", 'delta_volume_evolution_ensemble_average.nc')
opath_averaged_csv = os.path.join(wd_path,"masters", 'delta_volume_evolution_ensemble_average.csv')
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
#%% Cell 3: Create map plot _ ∆B or ∆V
models_shortlist = ["E3SM", "CESM2",  "NorESM",  "CNRM"]#, "IPSL-CM6"]

""" Load data"""
aggregated_ds = pd.read_csv(
    f"{wd_path}masters/complete_master_processed_for_map_plot.csv")

aggregated_ds_vol = pd.read_csv(
    f"{wd_path}masters/delta_volume_evolution_ensemble_average.csv")

gdf = gpd.GeoDataFrame(aggregated_ds, geometry=gpd.points_from_xy(
    aggregated_ds['lon'] +0.5 , aggregated_ds['lat'] + 0.5),
    crs="EPSG:4326")

gdf_volume = gpd.GeoDataFrame(aggregated_ds_vol, geometry=gpd.points_from_xy(
    aggregated_ds_vol['lon'] +0.5 , aggregated_ds_vol['lat'] + 0.5),
    crs="EPSG:4326")

"""Set up plot incl shapefile"""
# Plot setup and plot shapefile
fig, ax = plt.subplots(figsize=(15,12), subplot_kw={
                       'projection': ccrs.PlateCarree()})
# ax.set_extent([80, 107, 23, 48], crs=ccrs.PlateCarree())
# # Load shapefiles
shapefile_path = '/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/01. Input files/03. Shapefile/Karakoram/Pan-Tibetan Highlands/Pan-Tibetan Highlands (Liu et al._2022)/Shapefile/Pan-Tibetan Highlands (Liu et al._2022)_P.shp'
shp = gpd.read_file(shapefile_path).to_crs('EPSG:4326')
shp.plot(ax=ax, edgecolor='k', linewidth=0, facecolor='gainsboro')

subregions_path = "/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/02. QGIS/RGI outlines/GTN-G_O2regions_selected_clipped.shp"
subregions = gpd.read_file(subregions_path).to_crs('EPSG:4326')
ax.spines['geo'].set_visible(False)

# Optionally, remove gridlines
# Remove gridlines properly
gl = ax.gridlines(draw_labels=False)  # Create gridlines without labels
gl.xlines = False  # Remove longitude lines
gl.ylines = False  # Remove latitude lines


""" Plot subregions"""
# centroids = subregions.geometry.centroid
#
# define movements for the annotation of subregions
movements = {
    '13-01': [0, -0.2], #A
    '13-02': [4.8, 2], #F
    '13-03': [-0.2, -0.6], #B
    '13-04': [-0.6, 0], #C
    '13-05': [-0.2, 0.8], #D
    '13-06': [-10, 0], #E
    '13-07': [-1, 1], #G
    '13-08': [-2, -0.5], #J
    '13-09': [1.5, 0], #O
    '14-01': [0.5, -0.8], #H
    '14-02': [1, -0.4], #I
    '14-03': [3.4, -3.5], #K
    '15-01': [0, 0], #L
    '15-02': [-2, -1], #M
    '15-03': [3.5, 4.5], #N
}
# annotate subregions
# i=0
colormap = cm.get_cmap("tab20", 15)  # Alternatives: "Set3", "Paired", "tab10"
# region_colors = [colormap(i) for i in range(15)]

for attribute, subregion in subregions.groupby('o2region'):
   
# Uncomment for coloring of specific subregions

    linecolor = "black"  # if subregion.o2region.values in highlighted_subregions else "black"
    subregion.plot(ax=ax, edgecolor='grey', linewidth=1,
                  facecolor='none') # facecolor=region_colors[i],, alpha=0.4 # Plot the subregion
    # i+=1

# Create a ListedColormap with some colors (example)
""" Include the Mass Balance plot including color bar"""

# Create the BoundaryNorm with the defined boundaries
# listed_colors = [
#     (0.3, 0, 0),     # Very dark red
#     (0.6, 0, 0),    # Darker red for values below -0.7
#     (1, 0, 0),      # Red for values between -0.7 and -0.6
#     (1, 0.2, 0.2),  # Lighter red for values between -0.6 and -0.5
#     (1, 0.4, 0.4),  # Lighter red for values between -0.5 and -0.45
#     (1, 0.6, 0.6),  # Lighter red for values between -0.45 and -0.4
#     (1, 0.8, 0.8),  # Lightest red for values above -0.4
#     (1, 1, 1),      # White for -0.2 to -0.1
#     # (0.9, 0.9, 1),  # Light blue for -0.1 to 0 (this is distinct)
#     (0.8, 0.8, 1),  # Light blue for values 0 to 0.1
#     (0.5, 0.5, 1),  # Medium blue for 0.1 to 0.2
#     (0, 0, 0.7),     # Darker blue for values above 0.3
#     (0, 0, 0.4)     # Darker blue for values above 0.3
# ]
listed_colors = [
    # Red Shades (3 levels)
    (0.6, 0, 0),    # Dark Red
    (1, 0.2, 0.2),  # Medium Red
    (1, 0.6, 0.6),  # Light Red

    (1, 1, 1),      # White (Neutral, for near-zero values)

    # Blue Shades (7 levels)
    (0.8, 0.8, 1),  # Light Blue
    (0.6, 0.6, 1),  # Slightly Darker Blue
    (0.4, 0.4, 1),  # Medium-Light Blue
    (0.2, 0.2, 1),  # Medium Blue
    (0, 0, 0.8),    # Medium-Dark Blue
    (0, 0, 0.6),    # Darker Blue
    (0, 0, 0.4)     # Darkest Blue
]    

cax = fig.add_axes([0.31, -0.14, 0.82, 0.02])  # [left, bottom, width, height]

scatter_data = "B"

if scatter_data=="B":
    custom_cmap = clrs.ListedColormap(listed_colors)
    boundaries = [-0.35, -
                  0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75]
    boundaries_ticks = [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    
    norm = clrs.BoundaryNorm(boundaries, custom_cmap.N, clip=False)
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm),
                        cax=cax, boundaries=boundaries, ticks=boundaries_ticks,  orientation="horizontal")
    cbar.set_label('∆B (m w.e. yr$^{-1}$)', #'$∆B_{Irr-NoIrr}$ (m w.e. yr$^{-1}$)'
               fontsize=16, fontweight="bold")  # Label for the colorbar
    cbar.ax.set_xticklabels([f'{b:.1f}' for b in boundaries_ticks])
    scatter = ax.scatter(gdf.geometry.x, gdf.geometry.y,
                         s=np.sqrt(gdf['rgi_area_km2'])**1.7, c=gdf['B_delta_irr'], cmap=custom_cmap, norm=norm, edgecolor='k', alpha=1)

# Define the boundaries for each color block

elif scatter_data=="V":

    custom_cmap = clrs.ListedColormap(listed_colors)
    boundaries = [-87.5,-62.5,-37.5 ,-12.5,12.5, 37.5, 62.5, 87.5, 112.5, 137.5,162.5,187.5]  # Define the boundaries for each color block
    # Adjust the boundaries_ticks to match the boundarie
    boundaries_ticks = [-75,-50, -25,0, 25,50, 75,100,125,150,175]
    # boundaries_ticks = [-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3]
    # Create the BoundaryNorm with the defined boundaries
    norm = clrs.BoundaryNorm(boundaries, custom_cmap.N, clip=False)
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm),
                    cax=cax, boundaries=boundaries, ticks=boundaries_ticks,  orientation="horizontal")
    cbar.set_label('∆V$_{NoIrr/Irr}$ (%)', #'$∆B_{Irr-NoIrr}$ (m w.e. yr$^{-1}$)',
               fontsize=16, fontweight="bold")  
    # Adjust labels to 1 decimal place
    cbar.ax.set_xticklabels([f'{b:.1f}' for b in boundaries_ticks])
    scatter = ax.scatter(gdf_volume.geometry.x, gdf_volume.geometry.y,
                     s=np.sqrt(gdf_volume['rgi_area_km2'])**1.7, c=gdf_volume['delta_volume_noirr_rel']*100, cmap=custom_cmap, norm=norm, edgecolor='k', alpha=1)
 # Create the colorbar
cbar.ax.tick_params(labelsize=14) 

""" Add volume legend"""
custom_sizes = [200, 500, 1000, 2000, 3000]  # Example sizes for the legend (area)
size_labels = [f"{size:.0f}" for size in custom_sizes]  # Create labels for sizes

# Create legend handles (scatter points with different sizes)
legend_handles = [
    plt.scatter([], [], s=np.sqrt(size)**1.7, edgecolor='k', facecolor='none')
    for size in custom_sizes
]  # Adjust size factor if needed
text_handles = [Line2D([0], [0], linestyle="none", label=label) for label in size_labels]

h_leg = -0.18
w_leg = -0.01
# Create a separate axis for the legend
cax_legend = fig.add_axes([w_leg, h_leg, 0.15, 0.1])  # [left, bottom, width, height]

# Remove axis visuals
cax_legend.set_frame_on(False)  # Hide frame
cax_legend.set_xticks([])  # Remove x-ticks
cax_legend.set_yticks([])  # Remove y-ticks

# Add legend to the separate axis
legend = cax_legend.legend(
    legend_handles, size_labels, loc="upper center",
    fontsize=14, ncol=5,frameon=False, title="Total Area (km$^2$)", title_fontsize=16,  
    handler_map={tuple: HandlerTuple(ndivide=None)}, labelspacing=0.3, columnspacing=1, handletextpad=0.3
)


# for i, text in enumerate(legend.get_texts()):
#     text.set_y(text.get_position()[1] -60)
#     text.set_x(text.get_position()[0] -90)    
    
legend.get_title().set_position((0,-120))
legend.get_title().set_fontweight("bold")

""" Add call out plots for the different regions"""
# Define and iterate over grid layout
layout = [["13-01", "13-03", "13-04", "14-02", "13-05", "13-06"], ["13-02", "", "", "","", "13-07"], [
    "14-01", "", "", "","", "13-08"], ["","","", "","", "13-09"], ["","","14-03", "15-01", "15-02", "15-03"]]
w = 0.18 #0.17
w_space=0.035
h = w/0.17*0.15 #0.14
h_space=0.06
start_x=-0.12#-0.07
start_y = 0.9 #0.82
y_buffer=0.05
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
print(subregion_ids)  
for idx, pos in enumerate(grid_positions):
    if pos:
        subregion_id=subregion_ids[i]
        i+=1
        ax_callout = fig.add_axes(pos, ylim=(70,115))
    
        region_id = layout[idx // nr_cols][idx % nr_cols] #index columns and rows, // is rounded by full nrs
        print(region_id)
        
        
        # Find the corresponding subregion to add axes
        subregion = subregions[subregions['o2region'] == region_id] #convert to meters
        #load data in the plots
        subregion_ds = master_ds_avg[master_ds_avg['rgi_subregion'].str.contains(
            f"{region_id}")]
        mask = (subregion_ds["rgi_subregion"] == region_id)
        subregion_name = subregion_ds.full_name.iloc[0]

        # Baseline and model plotting
        baseline_path = os.path.join(
            wd_path, "summary", f"climate_run_output_baseline_W5E5.000.nc")
        baseline = xr.open_dataset(baseline_path)
        # Check if there are any matching rgi_id values
        # Ensure that rgi_id values exist in both datasets
        rgi_ids_in_baseline = baseline['rgi_id'].values
        matching_rgi_ids = np.intersect1d(
            rgi_ids_in_baseline, subregion_ds.rgi_id.values)
        baseline_filtered = baseline.sel(rgi_id=matching_rgi_ids)
        initial_volume = baseline_filtered['volume'].sum(dim="rgi_id")[0].values

        # Plot model member data
        vol_display="relative" #"relative"
        if vol_display =="absolute":
            ax_callout.plot(baseline_filtered["time"].values, baseline_filtered['volume'].sum(dim="rgi_id") * 1e-9,
                            label="W5E5.000", color="black", linewidth=2, zorder=15)
            filtered_member_data = []
            
            
            # Temporarily commented to speed up the plotting
    
            for m, model in enumerate(models_shortlist):
                for j in range(members_averages[m]):
                    sample_id = f"{model}.00{j + 1}" if members_averages[m] > 1 else f"{model}.000"
                    climate_run_output = xr.open_dataset(os.path.join(
                        sum_dir, f'climate_run_output_perturbed_{sample_id}.nc'))
                    climate_run_output = climate_run_output.where(
                        climate_run_output['rgi_id'].isin(subregion_ds.rgi_id.values), drop=True)
                    ax_callout.plot(climate_run_output["time"].values, climate_run_output["volume"].sum(
                        dim="rgi_id") * 1e-9, label=sample_id, color="grey", linewidth=1, linestyle="dotted")
                    filtered_member_data.append(
                        climate_run_output["volume"].sum(dim="rgi_id").values * 1e-9)
    
            # Mean and range plotting
            mean_values = np.mean(filtered_member_data, axis=0).flatten()
            min_values = np.min(filtered_member_data, axis=0).flatten()
            max_values = np.max(filtered_member_data, axis=0).flatten()
            ax_callout.plot(climate_run_output["time"].values, mean_values,
                            color=colors['noirr'][0], linestyle='solid', lw=2, label=f"{sum(members_averages)}-member average")
            ax_callout.fill_between(
                climate_run_output["time"].values, min_values, max_values, color=colors['noirr'][0], alpha=0.3)#color="lightblue", alpha=0.3)
            ax_callout.plot(climate_run_output["time"].values, np.ones(len(climate_run_output["time"]))*initial_volume, color="k", linewidth=1, linestyle="dashed")
            
        if vol_display=="relative":
            ax_callout.plot(baseline_filtered["time"].values, baseline_filtered['volume'].sum(dim="rgi_id")/initial_volume*100,
                            label="W5E5.000", color="black", linewidth=2, zorder=15)
            filtered_member_data = []
            
            
            # Temporarily commented to speed up the plotting
    
            for m, model in enumerate(models_shortlist):
                for j in range(members_averages[m]):
                    sample_id = f"{model}.00{j + 1}" if members_averages[m] > 1 else f"{model}.000"
                    climate_run_output = xr.open_dataset(os.path.join(
                        sum_dir, f'climate_run_output_perturbed_{sample_id}.nc'))
                    climate_run_output = climate_run_output.where(
                        climate_run_output['rgi_id'].isin(subregion_ds.rgi_id.values), drop=True)
                    
                    ax_callout.plot(climate_run_output["time"].values, climate_run_output["volume"].sum(
                        dim="rgi_id")/initial_volume*100, label=sample_id, color="grey", linewidth=1, linestyle="dotted")
                    filtered_member_data.append(
                        climate_run_output["volume"].sum(dim="rgi_id").values/initial_volume*100)
    
            # Mean and range plotting
            mean_values = np.mean(filtered_member_data, axis=0).flatten()
            min_values = np.min(filtered_member_data, axis=0).flatten()
            max_values = np.max(filtered_member_data, axis=0).flatten()
            ax_callout.plot(climate_run_output["time"].values, mean_values,
                            color=colors['noirr'][0], linestyle='solid', lw=2, label=f"{sum(members_averages)}-member average")
            ax_callout.fill_between(
                climate_run_output["time"].values, min_values, max_values, color=colors['noirr'][1], alpha=0.3)#color="lightblue", alpha=0.3)
            ax_callout.plot(climate_run_output["time"].values, np.ones(len(climate_run_output["time"]))*100, color="k", linewidth=1, linestyle="dashed")
        
            

        # # Subplot formatting
        if region_id=="13-02":
            subregion_title ="Pamir"
        elif region_id=="13-04":
            subregion_title ="East Tien Shan"
        elif region_id=="13-06":
            subregion_title ="East Kun Lun"
        else:
            subregion_title=subregion_name
            
        ax_callout.set_title(f"{subregion_title}", fontweight="bold", bbox=dict(
            facecolor='white', edgecolor='none', pad=1), fontsize=16)
        # Count the number of glaciers (assuming each 'rgi_id' represents a glacier)
        glacier_count = subregion_ds['rgi_id'].nunique()
        # Add number of glaciers as a text annotation in the lower left corner
        ax_callout.text(0.05, 0.05, f"{glacier_count}",
                        transform=ax_callout.transAxes, fontsize=14, verticalalignment='bottom', fontstyle='italic')
        ax_callout.text(0.89, 0.8, f"{subregion_id}",
                        transform=ax_callout.transAxes, fontsize=16, verticalalignment='bottom', fontweight='bold')
        
        boundary = subregion.geometry.boundary.iloc[0]
        if isinstance(boundary, MultiLineString):
            boundary = list(boundary.geoms)[0]
        boundary_coords = list(boundary.coords)
        boundary_x, boundary_y = boundary_coords[0]  # First point on the boundary
        boundary_x -= movements[region_id][0]
        boundary_y -= movements[region_id][1]
        # Annotate or place text near the boundary
        ax.text(boundary_x, boundary_y, f"{subregion_id}",
                horizontalalignment='center', fontsize=16, color='black', fontweight='bold')
       
        
        
        # ax_callout.set_xlim(-3, 3)
        # ax_callout.set_ylim(0, 20)
        if idx < len(grid_positions) - nr_cols:
            ax_callout.tick_params(axis='x', labelbottom=False)
        if idx % nr_cols != 0:
            ax_callout.tick_params(axis='y', labelleft=False)
        ax_callout.xaxis.set_tick_params(labelsize=14)
        ax_callout.yaxis.set_tick_params(labelsize=14)
            
        
            
                # Get main map extents
        
        callout_x, callout_y, callout_w, callout_h = pos
        # If applicable, plot the centroid connection lines here
        # if not subregion.empty:
            
        #     centroid = subregion.geometry.centroid.iloc[0]
        #     centroid_x, centroid_y = centroid.x, centroid.y
            
        #     polygon = subregion.geometry.iloc[0]  # Extract the polygon
        #     boundary = polygon.boundary  # Get the polygon boundary
        
        #     # Find the closest point on the boundary to the callout box
        #     callout_point = Point(callout_x, callout_y)
        #     closest_point = boundary.interpolate(boundary.project(callout_point))
        #     closest_x, closest_y = closest_point.x, closest_point.y

        #     # Step 4: Compute callout box corners in lat/lon
        #     top_left = (callout_x - (pos[2] * (x_max - x_min) / 2), callout_y + (pos[3] * (y_max - y_min) / 2))
        #     top_right = (callout_x + (pos[2] * (x_max - x_min) / 2), callout_y + (pos[3] * (y_max - y_min) / 2))
        #     bottom_left = (callout_x - (pos[2] * (x_max - x_min) / 2), callout_y - (pos[3] * (y_max - y_min) / 2))
        #     bottom_right = (callout_x + (pos[2] * (x_max - x_min) / 2), callout_y - (pos[3] * (y_max - y_min) / 2))

        #     # Find the closest corner to the centroid
        #     corners = [top_left, top_right, bottom_left, bottom_right]
        #     distances = [np.linalg.norm([closest_x - x, closest_y - y]) for x, y in corners]
        #     closest_corner = corners[np.argmin(distances)]
            
        #     lines = ConnectionPatch([closest_x, closest_y], closest_corner, coordsA="data", coordsB="figure fraction", axesA=ax, axesB=ax_callout, arrowstyle='->', capstyle='round',
        #                                        connectionstyle='angle', patchA=None, patchB=None, shrinkA=0.0, shrinkB=0.0, mutation_scale=10.0, mutation_aspect=None, clip_on=False)
        #     ax.add_artist(lines)


            # # Draw a dashed line connecting centroid to the closest callout corner
            # ax_overlay.plot([centroid_x, closest_corner[0]], [centroid_y, closest_corner[1]], 
            #                 color="black", linewidth=1, linestyle="dashed")

# Sample data for the example plot (volume vs. time)



time = np.linspace(1985, 2015, 5)  # Simulated time points
volume_irr = [30, 28, 27, 25, 24]  # Simulated volume data for Irr
volume_noirr = [30, 27, 25, 23, 20]  # Simulated volume data for NoIrr
volume_members1 = [31, 28, 26, 24, 22]  # Individual members
volume_members2 = [29, 26, 24, 22, 18]  # Individual members

# Create a new figure for the\small legend plot
fig_legend = fig.add_axes([-0.12, -0.03, w*2.15, h*2.4], ylim=(92,103))  # make twice as large as the callout plots
total_initial_volume = baseline['volume'].sum(dim="rgi_id")[0].values
vol_display="relative"
if vol_display =="absolute":
    fig_legend.plot(baseline["time"].values, baseline['volume'].sum(dim="rgi_id") * 1e-9,
                    label="Irr (W5E5.000)", color="black", linewidth=2, zorder=15)
    member_data = []
    
    # Temporarily commented to speed up the plotting

    for m, model in enumerate(models_shortlist):
        for j in range(members_averages[m]):
            if j ==1 and m==1:
                legendlabel = 'NoIrr (individual member)'
            else:
                legendlabel = '_nolegend_'
            sample_id = f"{model}.00{j + 1}" if members_averages[m] > 1 else f"{model}.000"
            climate_run_output = xr.open_dataset(os.path.join(
                sum_dir, f'climate_run_output_perturbed_{sample_id}.nc'))
            fig_legend.plot(climate_run_output["time"].values, climate_run_output["volume"].sum(
                dim="rgi_id") * 1e-9, label=legendlabel, color="grey", linewidth=1, linestyle="dotted")
            member_data.append(
                climate_run_output["volume"].sum(dim="rgi_id").values * 1e-9)

    # Mean and range plotting
    mean_values = np.mean(member_data, axis=0).flatten()
    min_values = np.min(member_data, axis=0).flatten()
    max_values = np.max(member_data, axis=0).flatten()
    fig_legend.plot(climate_run_output["time"].values, mean_values,
                    color=colors['noirr'][0], linestyle='solid', lw=2, label=f"NoIrr {sum(members_averages)}-member average")
    fig_legend.fill_between(
        climate_run_output["time"].values, min_values, max_values, color=colors['noirr'][1], alpha=0.3,label=f"NoIrr {sum(members_averages)}-member range" )
    fig_legend.plot(climate_run_output["time"].values, np.ones(len(climate_run_output["time"]))*total_initial_volume, color="k", linewidth=1, linestyle="dashed")
    
if vol_display=="relative":
    fig_legend.plot(baseline["time"].values, baseline['volume'].sum(dim="rgi_id")/total_initial_volume*100,
                    label="W5E5.000", color="black", linewidth=2, zorder=15)
    member_data = []
    
    
    # Temporarily commented to speed up the plotting

    for m, model in enumerate(models_shortlist):
        for j in range(members_averages[m]):
            if j ==1 and m==1:
                legendlabel = 'NoIrr (individual member)'
            else:
                legendlabel = '_nolegend_'
            sample_id = f"{model}.00{j + 1}" if members_averages[m] > 1 else f"{model}.000"
            climate_run_output = xr.open_dataset(os.path.join(
                sum_dir, f'climate_run_output_perturbed_{sample_id}.nc'))
            fig_legend.plot(climate_run_output["time"].values, climate_run_output["volume"].sum(
                dim="rgi_id")/total_initial_volume*100, label=legendlabel, color="grey", linewidth=1, linestyle="dotted")
            member_data.append(
                climate_run_output["volume"].sum(dim="rgi_id").values/total_initial_volume*100)

    # Mean and range plotting
    mean_values = np.mean(member_data, axis=0).flatten()
    min_values = np.min(member_data, axis=0).flatten()
    max_values = np.max(member_data, axis=0).flatten()
    fig_legend.plot(climate_run_output["time"].values, mean_values,
                    color=colors['noirr'][0], linestyle='solid', lw=2, label=f"NoIrr {sum(members_averages)}-member average")
    fig_legend.fill_between(
        climate_run_output["time"].values, min_values, max_values, color=colors['noirr'][1], alpha=0.3, label=f"NoIrr {sum(members_averages)}-member range" )
    fig_legend.plot(climate_run_output["time"].values, np.ones(len(climate_run_output["time"]))*100, color="k", linewidth=1, linestyle="dashed")
    
# Plot the sample data
# fig_legend.plot(time, volume_irr, label='AllForcings (W5E5)',
#                 color='black', linewidth=2)  # Black line for Irr
# fig_legend.plot(time, volume_noirr, label=f'NoIrr ({sum(members_averages)}-member average)',
#                 color='blue', linestyle='-', linewidth=2)  # Blue line for NoIrr average
# fig_legend.plot(time, volume_members1, label='NoIrr (individual member)', color='grey',
#                 linestyle='dotted', linewidth=1)  # Dotted grey for individual members
# fig_legend.plot(time, volume_members2, label='', color='grey',
#                 linestyle='dotted', linewidth=1)  # Dotted grey for individual members

# Shade for the range
# total_members=np.sum(members_averages)
# fig_legend.fill_between(time, volume_members2, volume_members1,
#                         color='lightblue', alpha=0.3, label=f'NoIrr {sum(members_averages)}-member range')  # Shading for range

# Annotate number of glaciers (just for example)
fig_legend.text(0.05, 0.05, '# of glaciers', transform=fig_legend.transAxes,
                fontsize=14, verticalalignment='bottom', horizontalalignment='left',
                bbox=dict(facecolor='white', edgecolor='none', pad=2))

fig_legend.text(0.6, 0.92, "ALL SUBREGIONS",
                transform=fig_legend.transAxes, fontsize=16, verticalalignment='bottom', fontweight='bold')

fig_legend.legend(loc='lower left', bbox_to_anchor=(0.02, 0.1), ncol=1, fontsize=14,
                  frameon=False, labelspacing=0.4)

# Set labels for axes
fig_legend.set_xlabel('Time', fontsize=16, labelpad=10, fontweight="bold")
fig_legend.set_ylabel('∆ Volume [%]', fontsize=16, labelpad=10, fontweight="bold")
fig_legend.set_title('High Mountain Asia', fontsize=16, fontweight='bold')
fig_legend.xaxis.set_tick_params(labelsize=14)
fig_legend.yaxis.set_tick_params(labelsize=14)

# Remove tick marks but keep the tick labels
# fig_legend.tick_params(axis='both', which='major', length=0)
# fig_legend.set_xticklabels([])  # Removes x-axis tick labels
# fig_legend.set_yticklabels([])  # Removes y-axis tick labels


fig.tight_layout()
fig_folder = os.path.join(fig_path, "04. Map")
os.makedirs(fig_folder, exist_ok=True)
plt.savefig(f"{fig_folder}/Map_Plot_sA_c{scatter_data}_boxV_IRR_2000_2014.png", dpi=300, bbox_inches="tight", pad_inches=0.1)



plt.show()


# %% Cell 4a: Create nan mask for ids without succesful comitted run


members = [3, 4, 6, 4, 1, 1]
models = ["E3SM", "CESM2", "CNRM", "NorESM", "W5E5"]#, "IPSL-CM6"]

overview_df = pd.DataFrame()

for m, model in enumerate(models):
    for member in range(members[m]):
        df_tot = pd.DataFrame()
        sample_id = f"{model}.00{member}"

        for f, filepath in enumerate([f"climate_run_output_baseline_W5E5.000.nc", f"climate_run_output_baseline_W5E5.000_comitted_random.nc"]):
            calendar_year = 2014
            if model != "W5E5":
                filepath = [f'climate_run_output_perturbed_{sample_id}.nc',
                            f'climate_run_output_perturbed_{sample_id}_comitted_random.nc'][f]
                # f'climate_run_output_perturbed_{sample_id}_comitted_cst.nc'][f]

            ds = xr.open_dataset(os.path.join(
                # log_dir, f"stats_perturbed_{sample_id}_climate_run_test.csv"))
                sum_dir, filepath))
            ds_zeros = ds.volume.sel(time=2014).to_dataframe()
            df_individual = ds_zeros[["calendar_year", "volume"]].reset_index()
            # df_tot = pd.concat([df_tot, df_individual], ignore_index=True)

            if df_tot.empty:
                df_tot = df_individual
            if f == 1:
                # Merge based on rgi_id
                df_tot = pd.merge(
                    df_tot,
                    df_individual[["rgi_id", "volume", "calendar_year"]],
                    on="rgi_id",
                    how="outer",
                    suffixes=("", "_noirr")
                )
            # if f == 2:
            #     # Merge based on rgi_id
            #     df_tot = pd.merge(
            #         df_tot,
            #         df_individual[["rgi_id", "volume", "calendar_year"]],
            #         on="rgi_id",
            #         how="outer",
            #         suffixes=("", "_noforcing")
            #     )
        df_zero_volume = df_tot[
            pd.isna(df_tot["volume_noirr"])|  pd.isna(df_tot["volume"])] #| pd.isna(df_tot["volume_noforcing"]) |
    overview_df = pd.concat([overview_df, df_zero_volume], ignore_index=True)


unique_rgi_ids = overview_df['rgi_id'].unique()
print(len(unique_rgi_ids))
unique_rgi_ids = pd.DataFrame(unique_rgi_ids, columns=['rgi_ids'])
unique_rgi_ids.to_csv(os.path.join(
    wd_path, 'masters', 'nan_mask_comitted_random.csv'))
    # wd_path, 'masters', 'nan_mask_comitted_random_noIPSL.csv'))

# Save the result to a CSV
output_path = os.path.join(
    wd_path, "masters", "nan_mask_all_models_volume_comitted_random.csv")
# wd_path, "masters", "nan_mask_all_models_volume_comitted_random_noIPSL.csv")
unique_rgi_ids.to_csv(output_path, index=False)
#%% Cell 4b: Create comitted mass loss plot


regions = [13, 14, 15]
subregions = [9, 3, 3]

fig, ax = plt.subplots(figsize=(15,10))  # create a new figure

members_averages = [2, 3,  3,  6, 1]
models_shortlist = ["E3SM", "CESM2",  "NorESM",  "CNRM"]#, "IPSL-CM6"]

# define the variables for p;lotting
factors = [10**-9]

variable_names = ["Volume"]
variable_axes = ["Volume compared to 1985 All Forcings scenario [%]"]
use_multiprocessing = False

rgi_ids_test=[]
subset_gdirs = gdirs_3r_a5
for gdir in subset_gdirs[:100]:
     rgi_ids_test.append(gdir.rgi_id)
     
 #Exclude error ids
nan_mask = pd.read_csv(os.path.join(
    wd_path, "masters", "nan_mask_all_models_volume_comitted_random.csv")).rgi_ids
# Remove duplicates if needed
nan_mask = set(pd.DataFrame({'rgi_id': nan_mask.unique()}).rgi_id.to_numpy())
rgi_ids_test = [rgi_id for rgi_id in rgi_ids_test if rgi_id not in nan_mask]
   
aggregated_ds_total = pd.read_csv(
    f"{wd_path}masters/master_lon_lat_rgi_id.csv") #load dataset - needed to filter rgi_ids per region    
aggregated_ds = aggregated_ds_total[aggregated_ds_total['rgi_id'].isin(rgi_ids_test)]
plt.rcParams.update({'font.size': 12})

#create a figure consisting of 1 large plot and several smaller ones for every subregion
fig = plt.figure(figsize=(15,8))
gs = gridspec.GridSpec(4, 6, figure=fig)  # Adjust the grid size as needed
ax_big = fig.add_subplot(gs[:3, :3])  # Spanning first 2 rows and 3 columns
small_axes = []
positions = [
    (0, 3), (0, 4), (0, 5), (1, 3), (1, 4), (1, 5),  # First row right side
    (2, 3),(2, 4),(2, 5),  # Third row
    (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5)  # Fourth row
]
axes_dict = {}  # Dictionary to store axes for later use
resp_values_dict={}
linestyles = ['solid', 'dashed']
for f, filepath in enumerate(["climate_run_output_baseline_W5E5.000.nc", f"climate_run_output_baseline_W5E5.000_comitted_random.nc"]):
    if f != 0:
        run_type = "noirr"
        legend_id = "committed"
        bar_values = 50
        run_label = "NoIrr"

    else:
        run_type = "irr"
        legend_id = ""
        bar_values = 20
    # load and plot the baseline data
    baseline_path = os.path.join(
        wd_path, "summary", filepath)
    baseline_all = xr.open_dataset(baseline_path)
    print(len(baseline_all.rgi_id))
    baseline_all = baseline_all.where(
        baseline_all.rgi_id.isin(rgi_ids_test), drop=True)
    
    #define the 1985 value for relative volume calculation
    if f == 0:
        resp_values_dict['Total'] = (baseline_all['volume'].sum(dim="rgi_id")[0].values * factors)[0]
        
        legendtext=f"W5E5.000"
    else:
        legendtext="W5E5.000, comitted run"
    ax_big.plot(baseline_all["time"], (baseline_all["volume"].sum(dim="rgi_id") * factors)/resp_values_dict['Total']*100,
            label=legendtext, color=colors[f"irr"][0], linewidth=2, zorder=15, linestyle=linestyles[f])
    nan_runs_noirr = []
    
    run_type = "noirr"
    legend_id = "" #comitted
    bar_values = 50
    run_label = "NoIrr"
    member_data_noirr_all = []   
    p=0 #set to zero for plot indices
    for reg, region in enumerate(regions):
        for sub in range(subregions[reg]):
            member_data_noirr_region = []
            region_id = f"{region}.0{sub+1}"
            print(region_id)
            pos = positions[p]

            if f==0: 
                ax = fig.add_subplot(gs[pos])
                ax.set_ylim(0,150)
                ax.set_title(region_id)
                axes_dict[region_id] = ax
                # if p not in {0,3,6,9}:
                ax.set_yticks([])
                # if p <9:
                ax.set_xticks([])
            p+=1 #loop trhough all the different subplots
            
            subregion_ds = aggregated_ds[aggregated_ds['rgi_subregion'].str.contains(
                f"{region}.0{sub+1}")]
            print(f"{region}.0{sub+1}")
            subregion_mask = set(pd.DataFrame({'rgi_id': subregion_ds.rgi_id.unique()}).rgi_id.to_numpy())

            # rgi_ids_region = [] #create filter for data not in that region
            # for gdir in subset_gdirs:
            #      rgi_ids_region.append(gdir.rgi_id)
            rgi_ids_region = [rgi_id for rgi_id in rgi_ids_test if rgi_id in subregion_mask]
            
            #plot baseline data per region
            baseline = baseline_all.where(
                baseline_all.rgi_id.isin(rgi_ids_region), drop=True) #filter baseline to region
            if f == 0:
                resp_values_dict[region_id] = baseline['volume'].sum(dim="rgi_id")[0].values * factors

            
            axes_dict[region_id].plot(baseline["time"], (baseline['volume'].sum(dim="rgi_id") * factors)/resp_values_dict[region_id]*100,
                    label=legendtext, color=colors[f"irr"][0], linewidth=2, zorder=15, linestyle=linestyles[f])
            #define files for climate runs for comitted and normal run 
            
            for m, model in enumerate(models_shortlist):
                for i in range(members_averages[m]):
                    sample_id = f"{model}.00{i}"

                    filepath = [f'climate_run_output_perturbed_{sample_id}.nc',
                                f'climate_run_output_perturbed_{sample_id}_comitted_random.nc',
                                ][f]
                    climate_run_opath_noirr = os.path.join(
                        sum_dir, filepath)  # f'climate_run_output_perturbed_{sample_id}_comitted.nc')
                    climate_run_output_noirr_all = xr.open_dataset(
                        climate_run_opath_noirr)
                    climate_run_output_noirr_selected = climate_run_output_noirr_all.where(
                        climate_run_output_noirr_all.rgi_id.isin(rgi_ids_test), drop=True)
                    if reg==0: #include the 3-region total climate run data for every model
                        member_data_noirr_all.append((climate_run_output_noirr_selected["volume"].sum(
                            dim="rgi_id").values)*factors/resp_values_dict['Total']*100)  
                    climate_run_output_noirr = climate_run_output_noirr_selected.where(
                         climate_run_output_noirr_selected.rgi_id.isin(rgi_ids_region), drop=True)   

                    # add all the summed volumes/areas to the member list, so a multi-member average can be calculated
                    member_data_noirr_region.append((climate_run_output_noirr["volume"].sum(
                        dim="rgi_id").values)*factors/resp_values_dict[region_id]*100)  
                    
                    if members_averages[m] > 1:
                        i += 1
                        label = None
                    else:
                        label = "GCM member"
                    
            # stack the member data
            stacked_member_data_region = np.stack(member_data_noirr_region)

            mean_values_noirr_region = np.median(
                stacked_member_data_region, axis=0).flatten()
            
            # mean_values_noirr = np.mean(stacked_member_data, axis=0).flatten()
            if f==0:
                labeltext = None#f"{run_label} (14-member avg) {legend_id}"
                # labeltext = f"{run_label} ({sum(members_averages)}-member avg) {legend_id}"
            else:
                labeltext = None
                
            axes_dict[region_id].plot(climate_run_output_noirr["time"].values, mean_values_noirr_region,
                    color=colors[f"{run_type}"][0], linestyle=linestyles[f], lw=2, label=labeltext, zorder=5)

            # calculate and plot volume/area 10-member min and max for ribbon
            min_values_noirr = np.min(stacked_member_data_region, axis=0).flatten()
            max_values_noirr = np.max(stacked_member_data_region, axis=0).flatten()
            if f==0:
                labeltext = "NoIrr 14-member avg" #f"{run_label} (14-member range) {legend_id}"
                labeltext_range="NoIrr 14-member range"
                # labeltext = f"{run_label} ({sum(members_averages)}-member range) {legend_id}"
            else:
                labeltext=label="NoIrr 14-member avg, comitted run"
                labeltext_range=None
            axes_dict[region_id].fill_between(climate_run_output_noirr["time"].values, min_values_noirr, max_values_noirr,
                            color=colors[f"{run_type}"][f], alpha=0.3, label=None, zorder=16)
           
            try:
                subregion_name = subregion_ds.full_name.iloc[0]
            except:
                subregion_name="no ids in subset"
            if region_id=="13.02":
                subregion_title ="Pamir"
            elif region_id=="13.04":
                subregion_title ="East Tien Shan"
            elif region_id=="13.06":
                subregion_title ="East Kun Lun"
            else:
                subregion_title=subregion_name
            
            axes_dict[region_id].set_title(subregion_title, fontweight="bold", bbox=dict(
                facecolor='white', edgecolor='none', pad=1), fontsize=14)

    stacked_member_data_all = np.stack(member_data_noirr_all)
    mean_values_noirr_all = np.median(
        stacked_member_data_all, axis=0).flatten()
    ax_big.plot(climate_run_output_noirr_all["time"].values, mean_values_noirr_all,
            color=colors[f"{run_type}"][0], linestyle=linestyles[f], lw=2, label=labeltext, zorder=5)
    # calculate and plot volume/area 10-member min and max for ribbon
    min_values_noirr_all = np.min(stacked_member_data_all, axis=0).flatten()
    max_values_noirr_all = np.max(stacked_member_data_all, axis=0).flatten()
    ax_big.fill_between(climate_run_output_noirr["time"].values, min_values_noirr_all, max_values_noirr_all,
                    color=colors[f"{run_type}"][f], alpha=0.3, label=labeltext_range, zorder=16)
    # Set labels and title for the combined plot
    ax_big.set_ylabel("∆Volume compared to 1985 All Forcings [%]", fontweight='bold')
    ax_big.set_xlabel("Time [year]")
    ax_big.set_title(f"All Regions ", fontweight="bold", bbox=dict(
        facecolor='white', edgecolor='none', pad=1), fontsize=14)
    ax_big.set_ylim(0,150)

    # Adjust the legend
    handles, labels = ax_big.get_legend_handles_labels()
    ax_big.legend(handles, labels,
               ncol=2)
    
plt.tight_layout()
plt.subplots_adjust(hspace=0.4, wspace=0.2)
plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0)
# specify and create a folder for saving the data (if it doesn't exists already) and save the plot
o_folder_data = f"/Users/magaliponds/OneDrive - Vrije Universiteit Brussel/1. VUB/02. Coding/01. IRRMIP/04. Figures/02. OGGM simulations/01. Modelled output/3r_a5/1. Volume/00. Combined"
os.makedirs(o_folder_data, exist_ok=True)
o_file_name = f"{o_folder_data}/1985_2014.{timeframe}.delta.Volume.comitted_random_subplots.png"
# o_file_name = f"{o_folder_data}/1985_2014.{timeframe}.delta.{variable_names[v]}.comitted_cst_test.png"
plt.savefig(o_file_name, bbox_inches='tight')
            
plt.show()

               


#%% Cell 5: Updated table
    
 
# Specify Paths
o_folder_data = (
    "/Users/magaliponds/OneDrive - Vrije Universiteit Brussel/1. VUB/02. Coding/01. IRRMIP/03. Data/03. Output Files/02. OGGM/02. Volume Area simulations"
)
o_file_data = f"{o_folder_data}/1985_2014.{timeframe}.delta.Volume.subregions.csv"
unit="km3"
if unit=="Gt":
    rho=0.917 #Gt/km³
    o_file_data_processed = f"{o_folder_data}/1985_2014.{timeframe}.delta.Volume.subregions_paper_output_table_gt.csv"

else:
    rho=1
    o_file_data_processed = f"{o_folder_data}/1985_2014.{timeframe}.delta.Volume.subregions_paper_output_table.csv"

# Load dataset
ds = pd.read_csv(o_file_data)
ds_subregions = ds[ds.time == 2014.0][[
    "subregion",
    "volume_loss_percentage_irr",
    "volume_loss_percentage_noirr",
    "volume_loss_percentage_cf",
    "volume_irr",
    "volume_noirr",
    "volume_cf"
]]

# Process total volumes
ds_total = ds.groupby("time").sum()[[
    "volume_irr",
    "volume_noirr",
    "volume_cf"
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
ds_all_losses["delta_irr"] = ds_all_losses["volume_loss_percentage_noirr"] / \
    ds_all_losses["volume_loss_percentage_irr"]*100

# Load RGI data
df = pd.read_csv(
    f"{wd_path}masters/master_gdirs_r3_a5_rgi_date_A_V_RGIreg.csv")
df["subregion"] = df["rgi_subregion"].str.replace(
    "-", ".").str.strip().str.lower()

# Aggregate RGI data
areas = df.groupby("subregion")["rgi_area_km2"].sum().reset_index(name="area")
volume = df.groupby("subregion")[
    "rgi_volume_km3"].sum().reset_index(name="volume")
nr_glaciers = df.groupby("subregion")[
    "rgi_id"].count().reset_index(name="nr_glaciers")

# Add total rows
areas = pd.concat([areas, pd.DataFrame(
    {"subregion": ["total"], "area": [areas["area"].sum()]})], ignore_index=True)
volume = pd.concat([volume, pd.DataFrame({"subregion": ["total"], "volume": [
                   volume["volume"].sum()]})], ignore_index=True)
nr_glaciers = pd.concat([nr_glaciers, pd.DataFrame({"subregion": [
                        "total"], "nr_glaciers": [nr_glaciers["nr_glaciers"].sum()]})], ignore_index=True)


# Process uncertainties
# for scenario in ["noi", "cf"]:
#     o_file_data_uncertainties = f"{o_folder_data}/1985_2014.{timeframe}.delta.Volume.subregions.uncertainties.{scenario}.csv"

#     # Load uncertainties - lowest and heightest modelled values
#     ds_uncertainties = pd.read_csv(o_file_data_uncertainties, index_col=0)
#     ds_uncertainties.index.name = "subregion"

#     # Calculate confidence intervals
#     mean_values = ds_uncertainties.mean(axis=1)
#     sem_values = ds_uncertainties.sem(axis=1)
#     confidence_level = 0.90
#     degrees_of_freedom = ds_uncertainties.shape[1] - 1
#     critical_value = stats.t.ppf(
#         (1 + confidence_level) / 2, degrees_of_freedom)
#     margin_of_error_abs = critical_value * sem_values

#     # Create confidence intervals DataFrame
#     confidence_intervals = pd.DataFrame({
#         f"error_margin_{scenario}_abs": margin_of_error_abs,
#     }).reset_index()

#     # Calculate total uncertainties
#     ds_uncertainties_total = ds_uncertainties.sum().round(2)
#     mean_values_total = ds_uncertainties_total.mean()
#     sem_values_total = ds_uncertainties_total.sem()
#     critical_value_total = stats.t.ppf(
#         (1 + confidence_level) / 2, degrees_of_freedom)
#     margin_of_error_total_abs = critical_value_total * sem_values_total

#     df_uncertainties_total = pd.DataFrame({
#         "subregion": ["total"],
#         f"error_margin_{scenario}_abs": [margin_of_error_total_abs],
#     })

#     # Combine with total
#     confidence_intervals = pd.concat(
#         [confidence_intervals, df_uncertainties_total], ignore_index=True)

#     # Merge confidence intervals into ds_all_losses
#     ds_all_losses = ds_all_losses.merge(
#         confidence_intervals, on="subregion", how="left")

# Merge RGI info into ds_all_losses
ds_all_losses['subregion'] = areas["subregion"]
ds_all_losses = ds_all_losses.merge(areas, on="subregion", how="left")
ds_all_losses = ds_all_losses.merge(volume, on="subregion", how="left")
ds_all_losses = ds_all_losses.merge(nr_glaciers, on="subregion", how="left")

# ds_all_losses['error_margin_noi_rel'] = ds_all_losses['error_margin_noi_abs'] / \
    # ds_all_losses['volume_noirr']*100
# ds_all_losses['error_margin_cf_rel'] = ds_all_losses['error_margin_cf_abs'] / \
    # ds_all_losses['volume_cf']*100
# Final selection of columns and save to CSV
ds_all_losses = ds_all_losses[[
    "subregion", "nr_glaciers", "area", f"volume",
    "volume_loss_percentage_irr", "volume_irr",
    "volume_loss_percentage_noirr", "volume_noirr",  "delta_irr",#"error_margin_noi_rel", "error_margin_noi_abs",
    # "volume_loss_percentage_cf", "volume_cf", "error_margin_cf_rel", "error_margin_cf_abs", "delta_cf",

]].round(2)

ds_all_losses.to_csv(o_file_data_processed)   
    
    
    
    
    
    
    
    