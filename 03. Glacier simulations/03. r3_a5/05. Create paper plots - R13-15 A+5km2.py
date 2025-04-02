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
from matplotlib.colors import LightSource
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
    "noirr": ["#f5bf03","#fdde6c"],#["#8B5A00", "#D4A017"],
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



#%% Cell 3: Create map plot - ∆B or ∆V
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

""" Plot shaded relief """
fig, ax = plt.subplots(figsize=(15,12), subplot_kw={
                       'projection': ccrs.PlateCarree()})
dem_path = "/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/01. Input files/02. DEM/ETOPO2v2c_f4.nc"
ds = xr.open_dataset(dem_path)

# Print dataset info to find variable names
dem=ds['z']
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
# Plot setup and plot shapefile

# ax.set_extent([80, 107, 23, 48], crs=ccrs.PlateCarree())
# # Load shapefiles
# shapefile_path = '/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/01. Input files/03. Shapefile/Karakoram/Pan-Tibetan Highlands/Pan-Tibetan Highlands (Liu et al._2022)/Shapefile/Pan-Tibetan Highlands (Liu et al._2022)_P.shp'
# shp = gpd.read_file(shapefile_path).to_crs('EPSG:4326')
# shp.plot(ax=ax, edgecolor='k', linewidth=2, facecolor='none')

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
start =(abs(vmin)/(vmax-vmin))+0.14
#Trim cmap so that white is at 0
original_cmap = plt.cm.bwr.reversed()  # Blue-White-Red colormap
custom_cmap = clrs.LinearSegmentedColormap.from_list(
    "trimmed_rwb", original_cmap(np.linspace(start, 1, 256))  # Trims first 20% of the colors
)
# Ensure white is at 0 without forcing symmetry
norm = clrs.Normalize(vmin=vmin,vmax=vmax)

# Create the colorbar
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm),
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

scatter = ax.scatter(gdf.geometry.x, gdf.geometry.y,
                     s=(np.sqrt(gdf['rgi_area_km2'])*5)**1.3, c=gdf['B_delta_irr'], cmap=custom_cmap, norm=norm, edgecolor='k', alpha=1)

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
# for i, handle in enumerate(legend.legendHandles):
#     handle.set_sizes([custom_sizes[i] * 0.1])  # Adjust scaling if needed

# for i, text in enumerate(legend.get_texts()):
#     text.set_y(text.get_position()[1] -60)
#     text.set_x(text.get_position()[0] -90)    
    
# legend.get_title().set_position((0,-120))
# legend.get_title().set_fontweight("bold")

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
print(subregion_ids)  
for idx, pos in enumerate(grid_positions):
    if pos:
        subregion_id=subregion_ids[i]
        i+=1
        ax_callout = fig.add_axes(pos, ylim=(-30,15),facecolor='whitesmoke')#'#E9E9E9'whitesmoke
    
        region_id = layout[idx // nr_cols][idx % nr_cols] #index columns and rows, // is rounded by full nrs
        print(region_id)
        
        
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
        # Check if there are any matching rgi_id values
        # Ensure that rgi_id values exist in both datasets
        rgi_ids_in_baseline = baseline['rgi_id'].values
        matching_rgi_ids = np.intersect1d(
            rgi_ids_in_baseline, subregion_ds.rgi_id.values)
        baseline_filtered = baseline.sel(rgi_id=matching_rgi_ids)
        initial_volume = baseline_filtered['volume'].sum(dim="rgi_id")[0].values

        # Plot model member data
        ax_callout.plot(baseline_filtered["time"].values, baseline_filtered['volume'].sum(dim="rgi_id")/initial_volume*100-100,
                        label="W5E5.000", color=colors['cf'][0], linewidth=2, zorder=15)
        filtered_member_data = []
        
        
        # Temporarily commented to speed up the plotting

        for m, model in enumerate(models_shortlist):
            for j in range(members_averages[m]):
                sample_id = f"{model}.00{j + 1}" if members_averages[m] > 1 else f"{model}.000"
                climate_run_output = xr.open_dataset(os.path.join(
                    sum_dir, f'climate_run_output_perturbed_{sample_id}.nc'))
                climate_run_output = climate_run_output.where(
                    climate_run_output['rgi_id'].isin(subregion_ds.rgi_id.values), drop=True)
                
                # ax_callout.plot(climate_run_output["time"].values, climate_run_output["volume"].sum(
                #     dim="rgi_id")/initial_volume*100-100, label=sample_id, color="grey", linewidth=1, linestyle="dotted")
                filtered_member_data.append(
                    climate_run_output["volume"].sum(dim="rgi_id").values/initial_volume*100-100)

        # Mean and range plotting
        mean_values = np.mean(filtered_member_data, axis=0).flatten()
        min_values = np.min(filtered_member_data, axis=0).flatten()
        max_values = np.max(filtered_member_data, axis=0).flatten()
        ax_callout.plot(climate_run_output["time"].values, mean_values,
                        color=colors['noirr'][0], linestyle='solid', lw=2, label=f"{sum(members_averages)}-member average")
        ax_callout.fill_between(
            climate_run_output["time"].values, min_values, max_values, color=colors['noirr'][1], alpha=0.3)#color="lightblue", alpha=0.3)
        ax_callout.plot(climate_run_output["time"].values, np.ones(len(climate_run_output["time"]))*100-100, color="black", linewidth=1, linestyle="dashed")

        # # Subplot formatting
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
       
        
        # ax_callout.set_xlim(-3, 3)
        # ax_callout.set_ylim(0, 20)
        # if idx < len(grid_positions) - nr_cols:
        ax_callout.tick_params(axis='x', labelbottom=False)
        if idx % nr_cols != 0:
            ax_callout.tick_params(axis='y', labelleft=False)
        ax_callout.xaxis.set_tick_params(labelsize=18)
        ax_callout.yaxis.set_tick_params(labelsize=18)
            
        callout_x, callout_y, callout_w, callout_h = pos

# Create a new figure for the\small legend plot
fig_legend = fig.add_axes([start_x, 0.069, w*2.1, h*2.2], ylim=(-8,3), facecolor="whitesmoke")##e9e9e9")  # make twice as large as the callout plots #w*2.4, h*2.15
total_initial_volume = baseline['volume'].sum(dim="rgi_id")[0].values
vol_display="relative"

fig_legend.plot(baseline["time"].values, baseline['volume'].sum(dim="rgi_id")/total_initial_volume*100-100,
                label="Historical, W5E5", color= colors['cf'][0], linewidth=2, zorder=15)
member_data = []


# Temporarily commented to speed up the plotting

for m, model in enumerate(models_shortlist):
    for j in range(members_averages[m]):
        if j ==1 and m==1:
            legendlabel = 'Historical NoIrr member'
        else:
            legendlabel = '_nolegend_'
        sample_id = f"{model}.00{j + 1}" if members_averages[m] > 1 else f"{model}.000"
        climate_run_output = xr.open_dataset(os.path.join(
            sum_dir, f'climate_run_output_perturbed_{sample_id}.nc'))
        fig_legend.plot(climate_run_output["time"].values, climate_run_output["volume"].sum(
            dim="rgi_id")/total_initial_volume*100-100, label=legendlabel, color=colors['noirr'][0], linewidth=2, linestyle="dotted")
        member_data.append(
            climate_run_output["volume"].sum(dim="rgi_id").values/total_initial_volume*100-100)

# Mean and range plotting
mean_values = np.mean(member_data, axis=0).flatten()
min_values = np.min(member_data, axis=0).flatten()
max_values = np.max(member_data, axis=0).flatten()
std_values = np.std(member_data, axis=0).flatten()  # Replace min with std

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
custom_order = [0,2,1]#[0,3,2,1]  # Example: Rearrange items by index

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
#%% Cell 4b: Create comitted mass loss plot transient with panels


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
            axes_dict[region_id].axhline(100, color='black', linestyle='--',
                        linewidth=1, zorder=1)  # Dashed line at 0
           
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
    ax_big.axhline(100, color='black', linestyle='--',
                linewidth=1, zorder=1)  # Dashed line at 0
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

        
 #%% Cell 4c: Comitted mass loss bocplots
# Function to plot the boxplots
def plot_boxplot(data, position, label, color, alpha, is_noirr):
    bp = ax.boxplot(data, patch_artist=True, labels=[""],#label if is_noirr else [""],  # only set labels for Noirr
                    vert=True, widths=0.3,
                    boxprops=dict(facecolor=color, alpha=alpha,
                                  edgecolor='none'),
                    medianprops=dict(color='black', linewidth=2),
                    positions=[position], showfliers=False, zorder=2)
    

    # Get the median value from the boxplot
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, zorder=0)
    ax.tick_params(labelsize=10)
    if p>0:
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        fontweight="medium"
    else: 
        ax.set_ylabel('Volume change (%, vs. 1985 historic)', labelpad=15, fontsize=12)
        ax.tick_params(left=True, bottom=False, labelleft=True, labelbottom=False, labelsize=12)
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        fontweight="bold"
    if is_noirr:
        ax.set_xlabel(label[0], rotation=90, fontweight=fontweight, fontsize=12)

    median_value = bp['medians'][0].get_xdata()[0]
    ax.set_ylim(-100, 140)


    # ax.text(median_value + 10, position, f'{median_value:.1f}',  # Place above for Irr
    #         va='center', ha='center', fontsize=10, color='black', #fontweight='bold',
    #         # bbox=dict(boxstyle='round,pad=0.001',
    #         #           facecolor='white', edgecolor='none'),
    #         zorder=3)

    return bp




regions = [13, 14, 15]
subregions = [9, 3, 3]

df = pd.read_csv(
    f"{wd_path}masters/master_gdirs_r3_a5_rgi_date_A_V_RGIreg_B_hugo_Vcom.csv")

master_ds = df[(~df['sample_id'].str.endswith('0')) |  # Exclude all the model averages ending with 0 except for IPSL
               (df['sample_id'].str.startswith('IPSL'))]

master_ds['V_2264_noirr'] = (master_ds['V_2264_noirr']-master_ds['V_1985_irr'])/master_ds['V_1985_irr']*100
master_ds['V_2264_irr'] = (master_ds['V_2264_irr']-master_ds['V_1985_irr'])/master_ds['V_1985_irr']*100
master_ds['V_2014_noirr'] = (master_ds['V_2014_noirr']-master_ds['V_1985_irr'])/master_ds['V_1985_irr']*100
master_ds['V_2014_irr'] = (master_ds['V_2014_irr']-master_ds['V_1985_irr'])/master_ds['V_1985_irr']*100
master_ds = master_ds[['rgi_id', 'rgi_region', 'rgi_subregion', 'full_name', 'cenlon', 'cenlat', 'rgi_date',
                       'rgi_area_km2', 'rgi_volume_km3', 'sample_id', 
                       'V_2264_irr','V_2264_noirr','V_2014_irr','V_2014_noirr', 'V_1985_irr']]


master_ds_avg = master_ds.groupby(['rgi_id'], as_index=False).agg({  # calculate the 11 member average for noirr and delta, for irr the first value can be taken as they are all the same
    'V_2264_noirr': 'mean',
    'V_2264_irr': 'mean',
    'V_2014_noirr': 'mean',
    'V_2014_irr': 'mean',
    # lamda is anonmous functions, returns 11 member average
    'sample_id': lambda _: "11 member average",
    # take first value for all columns that are not in the list
    **{col: 'first' for col in master_ds.columns if col not in ['V_2264_noirr', 'V_2264_irr','V_2014_noirr', 'V_2014_irr', 'sample_id']}
})


# Aggregate dataset, area-weighted BDelta Birr and Bnoirr,
master_ds_area_weighted = master_ds_avg.groupby(['rgi_subregion'], as_index=False).agg({
    'V_2264_noirr': lambda x: (x * master_ds.loc[x.index, 'rgi_area_km2']).sum() / master_ds.loc[x.index, 'rgi_area_km2'].sum(),
    'V_2264_irr': lambda x: (x * master_ds.loc[x.index, 'rgi_area_km2']).sum() / master_ds.loc[x.index, 'rgi_area_km2'].sum(),
    'V_2014_noirr': lambda x: (x * master_ds.loc[x.index, 'rgi_area_km2']).sum() / master_ds.loc[x.index, 'rgi_area_km2'].sum(),
    'V_2014_irr': lambda x: (x * master_ds.loc[x.index, 'rgi_area_km2']).sum() / master_ds.loc[x.index, 'rgi_area_km2'].sum(),
    'rgi_area_km2': 'sum',  # Sum for area
    'rgi_volume_km3': 'sum',  # Sum for volume
    'V_1985_irr': 'sum',  # Sum for volume
    # 'sample_id': lambda _: "11 member average",
    # take first value for all columns that are not in the list
    **{col: 'first' for col in master_ds.columns if col not in ['V_2264_irr', 'V_2264_noirr','V_1985_irr','rgi_area_km2', 'rgi_volume_km3']}
})

master_ds_avg=master_ds_avg[master_ds_avg['V_2264_irr'].notna()]
master_ds_area_weighted=master_ds_area_weighted[master_ds_area_weighted['V_2264_irr'].notna()]

use_weights = True
v_space_noi2 = 1.6  # Vertical space between irr and noirr boxplots
v_space_irr2 = 0.8  # Vertical space between irr and noirr boxplots
v_space_noi1 = 1.2  # Vertical space between irr and noirr boxplots
v_space_irr1 = 0.4  # Vertical space between irr and noirr boxplots

# Storage for combined data
all_noirr, all_irr = [], []
position_counter = 1


cumulative_index = 14

# Initialize plot
fig = plt.figure(figsize=(16, 6))
gs = gridspec.GridSpec(1, 16, width_ratios=[1.5] + [1]*15)  # First axis twice as wide

axes = [fig.add_subplot(gs[i]) for i in range(16)]
# fig, axes = plt.subplots(1,16, figsize=(15, 6), sharey=True)
print(axes)
# axes = axes.flatten()
p=1

# Example usage: Main loop through regions and subregions
for r, region in enumerate((regions)):
    for sub in range(list((subregions))[r]):
        # Filter subregion-specific data
        ax = axes[p]
        print(f"{region}.0{sub+1}")
       

        subregion_ds = master_ds_avg[master_ds_avg['rgi_subregion'].str.contains(
            f"{region}.0{sub+1}")]

        # Calculate mean values for Noirr and Irr
        noirr_mean = master_ds_area_weighted['V_2264_noirr'][cumulative_index]
        irr_mean = master_ds_area_weighted['V_2264_irr'][cumulative_index]

        # Set color and label based on the region
        try:
            label = [subregion_ds.full_name.iloc[0]]
        except:
            label = [f"{region}-0{sub+1}"]
        
        if region==13:
            if sub ==1:
                label = ["Pamir"]
            elif sub ==3:
                label = ["East Tien Shan"]
            elif sub ==5:
                label = ["East Kun Lun"]
        # Plot Noirr and Irr boxplots
        
        box_irr = plot_boxplot(
            subregion_ds['V_2014_irr'], position_counter + v_space_irr1, label="", color=colors['cf'][0], alpha=1, is_noirr=False)
        box_irr = plot_boxplot(
            subregion_ds['V_2264_irr'], position_counter + v_space_irr2, label="", color=colors['cf'][1], alpha=1, is_noirr=False)
        box_noirr = plot_boxplot(
            subregion_ds['V_2014_noirr'], position_counter + v_space_noi1, label, color=colors['noirr'][0], alpha=1, is_noirr=True)
        box_noirr = plot_boxplot(
            subregion_ds['V_2264_noirr'], position_counter + v_space_noi2, label, color=colors['noirr'][1], alpha=1, is_noirr=True)

        
        # Annotate the number of glaciers and delta between the two columns
        num_glaciers = len(subregion_ds)
        # delta = noirr_mean - irr_mean

        # Display number of glaciers and delta
        initial_volume = round(sum(subregion_ds['V_1985_irr'].values)*1e-9)
        ax.text(1.1, 120, 
                 f'{initial_volume}\n({num_glaciers})',  # \nΔ = {delta:.2f}',
                 va='center', ha='left', fontsize=12, color='black',
                 backgroundcolor="white",  zorder=10)

        p+=1
        
    

overall_area_weighted_mean = {
    'V_2264_irr': (master_ds_avg['V_2264_irr'] * master_ds_avg['rgi_area_km2']).sum() / master_ds_avg['rgi_area_km2'].sum(),
    'V_2264_noirr': (master_ds_avg['V_2264_noirr'] * master_ds_avg['rgi_area_km2']).sum() / master_ds_avg['rgi_area_km2'].sum(),
    'V_2014_irr': (master_ds_avg['V_2014_irr'] * master_ds_avg['rgi_area_km2']).sum() / master_ds_avg['rgi_area_km2'].sum(),
    'V_2014_noirr': (master_ds_avg['V_2014_noirr'] * master_ds_avg['rgi_area_km2']).sum() / master_ds_avg['rgi_area_km2'].sum(),
    'total_area_km2': master_ds_avg['rgi_area_km2'].sum(),  # Total area sum
    'V_1985_irr': master_ds_avg['V_1985_irr'].sum(),  # Total area sum
    # Total volume sum
    'total_volume_km3': master_ds_avg['rgi_volume_km3'].sum()
}

# # Plot overall average boxplots for Irr and Noirr
p=0
ax = axes[p]


avg_irr = plot_boxplot(master_ds_avg['V_2014_irr'],  position_counter +
                       v_space_irr1, "", color=colors['cf'][0], alpha=1, is_noirr=False)
avg_irr = plot_boxplot(master_ds_avg['V_2264_irr'],  position_counter +
                       v_space_irr2, "", color=colors['cf'][1], alpha=1, is_noirr=False)
avg_noi = plot_boxplot(master_ds_avg['V_2014_noirr'], position_counter + v_space_noi1, [
    "High Mountain Asia"], color=colors['noirr'][0], alpha=1, is_noirr=True)
avg_noi = plot_boxplot(master_ds_avg['V_2264_noirr'], position_counter + v_space_noi2, [
    "High Mountain Asia"], color=colors['noirr'][1], alpha=1, is_noirr=True)


initial_total_volume = round(overall_area_weighted_mean['V_1985_irr']*1e-9)
# Annotate the number of glaciers for the overall average
ax.text(1.1, 120, 
         f"V:{initial_total_volume}\n(#:{len(master_ds_avg)})",# Display total number of glaciers
         va='center', ha='left', fontsize=12, color='black',
          backgroundcolor="white", zorder=10)#fontstyle='italic',

# position_counter += 4


# Create custom legend elements for mean (dot) and median (stripe)
mean_dot = Line2D([0], [0], marker='o', color='w',
                  label='Area-weighted Mean (dot)', markerfacecolor='black', markersize=10)
median_stripe = Line2D([0], [0], color='black', lw=2, label='Median (stripe)')

# Add a legend for regions, mean (dot), and median (stripe)
region_legend_patches = [mpatches.Patch(color=colors['cf'][0], label='Historical, W5E5'),
                         mpatches.Patch(color=colors['cf'][1], label='Historical, comitted'),
                         mpatches.Patch(color=colors['noirr'][0], label='Historical NoIrr'),
                         mpatches.Patch(color=colors['noirr'][1], label='Historical NoIrr, comitted'),
                         ]

# Append the custom legend items for the mean dot and median stripe
region_legend_patches += [median_stripe]

# Create the legend with the updated patches
# fig.legend(handles=region_legend_patches, loc='center right',
#            bbox_to_anchor=(1.15, 0.6), ncols=1)
fig.legend(handles=region_legend_patches, loc='lower center',
          bbox_to_anchor=(0.5, -0.05), ncols=5, fontsize=12)




# Extend the ylims for more padding
# y_min, y_max = fig.get_ylim()  # Get current y-axis limits
# padding = 1  # Adjust this value as needed
# ax.set_ylim(y_min - padding, y_max + padding)

# Adjust layout and display the plot
plt.tight_layout()
fig_folder = os.path.join(fig_path, "03. Mass Balance", "Boxplot")
# os.makedirs(fig_folder, exist_ok=True)
# plt.savefig(
#     f"{fig_folder}/Gaussian_distribution_total_region_by_region_median.png")
plt.show()           

 #%% OUTDATED BOXPLOT
# Function to plot the boxplots
def plot_boxplot(data, position, label, color, alpha, is_noirr):
    bp = ax.boxplot(data, patch_artist=True, labels=label if is_noirr else [""],  # only set labels for Noirr
                    vert=False, widths=0.7,
                    boxprops=dict(facecolor=color, alpha=alpha,
                                  edgecolor='none'),
                    medianprops=dict(color='black', linewidth=2),
                    positions=[position], showfliers=False, zorder=2)

    # Get the median value from the boxplot
    median_value = bp['medians'][0].get_xdata()[0]

    ax.text(median_value + 10, position, f'{median_value:.1f}',  # Place above for Irr
            va='center', ha='center', fontsize=10, color='black', #fontweight='bold',
            # bbox=dict(boxstyle='round,pad=0.001',
            #           facecolor='white', edgecolor='none'),
            zorder=3)
    ax.set_ylim(-100, 140)


    return bp


# Initialize plot
fig, ax = plt.subplots(figsize=(9, 9))

df = pd.read_csv(
    f"{wd_path}masters/master_gdirs_r3_a5_rgi_date_A_V_RGIreg_B_hugo_Vcom.csv")

master_ds = df[(~df['sample_id'].str.endswith('0')) |  # Exclude all the model averages ending with 0 except for IPSL
               (df['sample_id'].str.startswith('IPSL'))]

master_ds['V_2264_noirr'] = (master_ds['V_2264_noirr']-master_ds['V_1985_irr'])/master_ds['V_1985_irr']*100
master_ds['V_2264_irr'] = (master_ds['V_2264_irr']-master_ds['V_1985_irr'])/master_ds['V_1985_irr']*100
master_ds = master_ds[['rgi_id', 'rgi_region', 'rgi_subregion', 'full_name', 'cenlon', 'cenlat', 'rgi_date',
                       'rgi_area_km2', 'rgi_volume_km3', 'sample_id', 
                       'V_2264_irr','V_2264_noirr', 'V_1985_irr']]


master_ds_avg = master_ds.groupby(['rgi_id'], as_index=False).agg({  # calculate the 11 member average for noirr and delta, for irr the first value can be taken as they are all the same
    'V_2264_noirr': 'mean',
    'V_2264_irr': 'mean',
    # lamda is anonmous functions, returns 11 member average
    'sample_id': lambda _: "11 member average",
    # take first value for all columns that are not in the list
    **{col: 'first' for col in master_ds.columns if col not in ['V_2264_noirr', 'V_2264_irr', 'sample_id']}
})


# Aggregate dataset, area-weighted BDelta Birr and Bnoirr,
master_ds_area_weighted = master_ds_avg.groupby(['rgi_subregion'], as_index=False).agg({
    'V_2264_noirr': lambda x: (x * master_ds.loc[x.index, 'rgi_area_km2']).sum() / master_ds.loc[x.index, 'rgi_area_km2'].sum(),
    'V_2264_irr': lambda x: (x * master_ds.loc[x.index, 'rgi_area_km2']).sum() / master_ds.loc[x.index, 'rgi_area_km2'].sum(),
    'rgi_area_km2': 'sum',  # Sum for area
    'rgi_volume_km3': 'sum',  # Sum for volume
    'V_1985_irr': 'sum',  # Sum for volume
    # 'sample_id': lambda _: "11 member average",
    # take first value for all columns that are not in the list
    **{col: 'first' for col in master_ds.columns if col not in ['V_2264_irr', 'V_2264_noirr','V_1985_irr','rgi_area_km2', 'rgi_volume_km3']}
})

master_ds_avg=master_ds_avg[master_ds_avg['V_2264_irr'].notna()]
master_ds_area_weighted=master_ds_area_weighted[master_ds_area_weighted['V_2264_irr'].notna()]

use_weights = True
v_space_noi = 0.8  # Vertical space between irr and noirr boxplots
v_space_irr = 1.6  # Vertical space between irr and noirr boxplots

# Storage for combined data
all_noirr, all_irr = [], []
position_counter = 1


cumulative_index = 14


p=0
# Example usage: Main loop through regions and subregions
for r, region in enumerate(reversed(regions)):
    for sub in reversed(range(list(reversed(subregions))[r])):
        # Filter subregion-specific data

        subregion_ds = master_ds_avg[master_ds_avg['rgi_subregion'].str.contains(
            f"{region}.0{sub+1}")]

        # Calculate mean values for Noirr and Irr
        noirr_mean = master_ds_area_weighted['V_2264_noirr'][cumulative_index]
        irr_mean = master_ds_area_weighted['V_2264_irr'][cumulative_index]

        # Set color and label based on the region
        # color = region_colors[region]
        try:
            label = [subregion_ds.full_name.iloc[0]]
        except:
            label = [f"{region}-0{sub+1}"]
        
        if region==13:
            if sub ==1:
                label = ["Pamir"]
            elif sub ==3:
                label = ["East Tien Shan"]
            elif sub ==5:
                label = ["East Kun Lun"]
        # Plot Noirr and Irr boxplots
        box_noirr = plot_boxplot(
            subregion_ds['V_2264_noirr'], position_counter + v_space_noi, label, color=colors['noirr'][0], alpha=0.5, is_noirr=True)
        box_irr = plot_boxplot(
            subregion_ds['V_2264_irr'], position_counter + v_space_irr, label="", color=colors['irr'][0], alpha=0.5, is_noirr=False)

        # Annotate the number of glaciers and delta between the two columns
        num_glaciers = len(subregion_ds)
        delta = noirr_mean - irr_mean

        # Display number of glaciers and delta
        initial_volume = round(sum(subregion_ds['V_1985_irr'].values)*1e-9)
        plt.text(120, position_counter + (v_space_noi ),
                 f'{initial_volume} ({num_glaciers})',  # \nΔ = {delta:.2f}',
                 va='center', ha='left', fontsize=10, color='black',
                 backgroundcolor="white",  zorder=1)

        # Increment position counter for the next subregion
        position_counter += 4
        cumulative_index -= 1
    

overall_area_weighted_mean = {
    'V_2264_irr': (master_ds_avg['V_2264_irr'] * master_ds_avg['rgi_area_km2']).sum() / master_ds_avg['rgi_area_km2'].sum(),
    'V_2264_noirr': (master_ds_avg['V_2264_noirr'] * master_ds_avg['rgi_area_km2']).sum() / master_ds_avg['rgi_area_km2'].sum(),
    'total_area_km2': master_ds_avg['rgi_area_km2'].sum(),  # Total area sum
    'V_1985_irr': master_ds_avg['V_1985_irr'].sum(),  # Total area sum
    # Total volume sum
    'total_volume_km3': master_ds_avg['rgi_volume_km3'].sum()
}

# Plot overall average boxplots for Irr and Noirr
avg_noi = plot_boxplot(master_ds_avg['V_2264_noirr'], position_counter + v_space_noi, [
    "Average"], color=colors['noirr'][1], alpha=1, is_noirr=True)
avg_irr = plot_boxplot(master_ds_avg['V_2264_irr'],  position_counter +
                       v_space_irr, "", color=colors['irr'][1], alpha=1, is_noirr=False)

initial_total_volume = round(overall_area_weighted_mean['V_1985_irr']*1e-9)
# Annotate the number of glaciers for the overall average
plt.text(120, position_counter + (v_space_noi ), 
         f"V: {initial_total_volume}·10^9 km³\n(#: {len(master_ds_avg)})",# Display total number of glaciers
         va='center', ha='left', fontsize=10, color='black',
          backgroundcolor="white", zorder=1, fontweight="bold")#fontstyle='italic',

position_counter += 4


# Create custom legend elements for mean (dot) and median (stripe)
mean_dot = Line2D([0], [0], marker='o', color='w',
                  label='Area-weighted Mean (dot)', markerfacecolor='black', markersize=10)
median_stripe = Line2D([0], [0], color='black', lw=2, label='Median (stripe)')

# Add a legend for regions, mean (dot), and median (stripe)
region_legend_patches = [mpatches.Patch(color=colors['irr'][1], label='AllForcings'),
                         mpatches.Patch(color=colors['noirr'][1], label='NoIrr (14-member average)'),
                         ]

# Append the custom legend items for the mean dot and median stripe
region_legend_patches += [median_stripe]

# Create the legend with the updated patches
# ax.legend(handles=region_legend_patches, loc='center right',
#            bbox_to_anchor=(1.5, 0.5), ncols=1)
ax.legend(handles=region_legend_patches, loc='lower center',
          bbox_to_anchor=(0.5, -0.12), ncols=3)

ax.axvline(x=0, color='black', linestyle='--', linewidth=1, zorder=0)
ax.set_ylabel('Regions and Subregions', labelpad=15)
ax.set_xlabel('Average 250yr comitted volume change (%, vs. 1985)')
ax.set_xlim(-100, 140)

# Extend the ylims for more padding
y_min, y_max = ax.get_ylim()  # Get current y-axis limits
# padding = 1  # Adjust this value as needed
# ax.set_ylim(y_min - padding, y_max + padding)

# Adjust layout and display the plot
plt.tight_layout()
fig_folder = os.path.join(fig_path, "03. Mass Balance", "Boxplot")
os.makedirs(fig_folder, exist_ok=True)
# plt.savefig(
#     f"{fig_folder}/Gaussian_distribution_total_region_by_region_median.png")
plt.show()           


 #%% Cell 5: Updated table
    
 
# Specify Paths
o_folder_data = (
    "/Users/magaliponds/OneDrive - Vrije Universiteit Brussel/1. VUB/02. Coding/01. IRRMIP/03. Data/03. Output Files/02. OGGM/02. Volume Area simulations"
)
o_file_data = f"{o_folder_data}/1985_2014.{timeframe}.delta.Volume.subregions.csv"
unit="km3"
if unit=="Gt":
    rho=0.85 #Gt/km³
    o_file_data_processed = f"{o_folder_data}/1985_2014.{timeframe}.delta.Volume.subregions_paper_output_table_gt.csv"

else:
    rho=1
    o_file_data_processed = f"{o_folder_data}/1985_2014.{timeframe}.delta.Volume.subregions_paper_output_table_absolute.csv"

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
ds_all_losses["delta_irr"] = ds_all_losses["volume_loss_percentage_irr"] - ds_all_losses["volume_loss_percentage_noirr"]
ds_all_losses["delta_irr_abs"] = ds_all_losses["volume_irr"] - ds_all_losses["volume_noirr"]

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
    
    
    
    
    
    
    
    
#%% Cell 6: Temperature plot - 2d legend

def create_precip_temp_colormap(
    precip_range=(-20, 20), temp_range=(-1.5, 1.5), grid_size=256
):
    color_dry_warm = np.array([1.0, 0.4, 0.0])
    color_dry_cold = np.array([0.6, 0.0, 0.6])
    color_wet_warm = np.array([0.6, 1.0, 0.2])
    color_wet_cold = np.array([0.0, 0.4, 1.0])
    color_center  = np.array([1.0, 1.0, 1.0])

    custom_colormap = np.zeros((grid_size, grid_size, 3))
    for i in range(grid_size):
        for j in range(grid_size):
            x = -1 + 2 * j / (grid_size - 1)
            y = -1 + 2 * i / (grid_size - 1)

            wx = (x + 1) / 2
            wy = (y + 1) / 2

            top = (1 - wx) * color_dry_warm + wx * color_wet_warm
            bottom = (1 - wx) * color_dry_cold + wx * color_wet_cold
            color = (1 - wy) * bottom + wy * top

            r = np.sqrt(x**2 + y**2)
            fade = np.clip(1 - r, 0, 1)
            color = (1 - fade) * color + fade * color_center

            custom_colormap[i, j] = color

    def colormap_callable(delta_temp, delta_precip):
        rows = ((delta_temp - temp_range[0]) / (temp_range[1] - temp_range[0]) * (grid_size - 1)).astype(int)
        cols = ((delta_precip - precip_range[0]) / (precip_range[1] - precip_range[0]) * (grid_size - 1)).astype(int)
        rows = np.clip(rows, 0, grid_size - 1)
        cols = np.clip(cols, 0, grid_size - 1)
        return custom_colormap[rows, cols]

    return custom_colormap, colormap_callable

def plot_colormap_legend(ax,colormap_grid, precip_range=(-20, 20), temp_range=(-1.5, 1.5)):
    """
    Plot the 2D color legend for the custom ΔPrecip–ΔTemp colormap.

    Parameters:
        colormap_grid: The [grid_size x grid_size x 3] RGB array from create_precip_temp_colormap()
        precip_range: Tuple of (min, max) precipitation anomaly
        temp_range: Tuple of (min, max) temperature anomaly
    """
    ax.imshow(colormap_grid, origin='lower',
              extent=[precip_range[0], precip_range[1], temp_range[0], temp_range[1]],
              aspect='auto')
    ax.set_xlabel("ΔPrecipitation (%)", fontsize=12)
    ax.set_ylabel("ΔTemperature (°C)", fontsize=12)
    ax.tick_params(labelsize=12)
    ax.set_title("Legend", fontsize=12)
    ax.grid(True)
    plt.tight_layout()
    # plt.show()


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
            
        # --- NEW: If colormap is callable, apply it manually ---
        if callable(custom_cmap):
            # Assuming `diff_sel` contains both temp and precip deltas
            delta_temp = diff_sel['tas'] if 'tas' in diff_sel else diff_sel
            delta_precip = diff_sel['pr'] if 'pr' in diff_sel else diff_sel

            delta_temp_np = delta_temp.values
            delta_precip_np = delta_precip.values
            rgb_img = custom_cmap(delta_temp_np, delta_precip_np)
            im = ax.imshow(rgb_img,
                           extent=[diff_sel.lon.min(), diff_sel.lon.max(),
                                   diff_sel.lat.min(), diff_sel.lat.max()],
                           origin='lower',
                           transform=ccrs.PlateCarree())
        else:
            im = diff_sel.plot.imshow(ax=ax, vmin=vmin, vmax=vmax, extend='both',
                                      transform=ccrs.PlateCarree(), cmap=custom_cmap, add_colorbar=False)

        # # Plot data and the Karakoram outline
        # im = diff_sel.plot.imshow(ax=ax, vmin=vmin, vmax=vmax, extend='both',
        #                           transform=ccrs.PlateCarree(), cmap=custom_cmap, add_colorbar=False)

        if scale=="Local":
            shp.plot(ax=ax, edgecolor='black', linewidth=1, facecolor='none')
        ax.coastlines(resolution='10m')

        # Set gridlines and labels
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False

    
        gl.ylabel_style = {'size': 12}
        gl.ylocator = plt.MaxNLocator(nbins=3)
    
        gl.xlabel_style = {'size': 12}
        gl.xlocator = plt.MaxNLocator(nbins=3)
        

        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        """4 Include labels for the cbar and for the y and x axis"""

        ax.set_title(title)

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
                            vmin = -20
                            vmax = 20
                            zero_scaled = (abs(vmin)/(abs(vmin)+abs(vmax)))
                            colors = [(0, 'xkcd:mocha'), (zero_scaled,
                                                          'xkcd:white'), (1, 'xkcd:aquamarine')]
                            # custom_cmap = LinearSegmentedColormap.from_list(
                            #     'custom_cmap', colors)
                            unit = '%'


                    elif var == "Temperature":
                        variable_name = "tas"
                        var_suffix = "TEMP"
                    
                        mode_suff = 'total'
                        vmin = -1.5
                        vmax = 1.5
                        zero_scaled = (abs(vmin)/(abs(vmin)+abs(vmax)))
                        colors = [(0, 'cornflowerblue'), (zero_scaled,
                                                          'xkcd:white'), (1, 'xkcd:tomato')]
                        custom_cmap = LinearSegmentedColormap.from_list(
                            'custom_cmap', colors)
                        unit = '°C'

                    members = [1, 3, 4, 6, 4]
                    # members = [1, 1, 1, 1]
                    all_diff = []  # create a dataset to add all member differences
                    all_model_diffs = []
                    models = ["IPSL-CM6", "E3SM", "CESM2", "CNRM", "NorESM"]
                    lonmin, lonmax, latmin, latmax = [60,109,22,52]

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
                    elif var =="Temperature":
                        all_vars_t = all_diffs_avg
                        all_vars_local_t = all_diffs_avg_local
                    
                
all_vars_2d = xr.merge([all_vars_p, all_vars_t])
all_vars_local_2d = xr.merge([all_vars_local_p, all_vars_local_t])

# 2D blended colormap
colormap_grid, custom_cmap = create_precip_temp_colormap(
    precip_range=(-20, 20), temp_range=(-1.5, 1.5)
)


""" Part 2 - Shapefile outline for Karakoram Area to be included"""
# path to  shapefile
shapefile_path = '/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/01. Input files/03. Shapefile/Karakoram/Pan-Tibetan Highlands/Pan-Tibetan Highlands (Liu et al._2022)/Shapefile/Pan-Tibetan Highlands (Liu et al._2022)_P.shp'
shp = gpd.read_file(shapefile_path)
target_crs = 'EPSG:4326'
shp = shp.to_crs(target_crs)

indices = ["A", "B"]#, "E", "F"]

# Create the mosaic plot

layout = """
AB
"""
figsize = (18.15,5) #based on aspect ratio of global and zoomed

fig, axes = plt.subplot_mosaic(layout, subplot_kw={'projection': ccrs.PlateCarree()},
                               figsize=figsize,
                               gridspec_kw={'wspace': 0.05, 'width_ratios': [2.0, 1.63]})#,'height_ratios': [1, 1]})
axes['A'].set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
axes['B'].set_extent([lonmin, lonmax, latmin, latmax], crs=ccrs.PlateCarree())
# plot the irrmip difference
im = plot_subplots(indices[0], subplots, (sum(members)-len(models)+1),
                   all_vars_2d, timestamps, axes[indices[0]], shp, custom_cmap, timeframe, "Global", "Global", vmin=vmin, vmax=vmax)
im = plot_subplots(indices[1], subplots, (sum(members)-len(models)+1),
                   all_vars_local_2d, timestamps, axes[indices[1]], shp, custom_cmap, timeframe, "Local", "High Mountain Asia", vmin=vmin, vmax=vmax)

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

sign_diff = xr.concat([np.sign(diff[variable_name])
                      for diff in all_model_diffs], dim="models")
agreement_on_sign = (
    abs(sign_diff.mean(dim="models")) > 0.8)

# Step 3: Combine the conditions to create the final mask
within_threshold = agreement_on_sign.sel(lon=slice(lonmin, lonmax), lat=slice(latmax, latmin))
# Step 4: Convert the mask to 2D by removing the singleton 'variable' dimension if needed
within_threshold_2d = within_threshold.astype(
    int).squeeze()

# Step 6: Overlay the mask with dots in areas where agreement criteria are met
axes[indices[1]].contourf(
    all_diffs_avg_local.lon, all_diffs_avg_local.lat, within_threshold_2d,
    levels=[0.5, 1.5], colors='none', hatches=['////'], transform=ccrs.PlateCarree()
)


""" 3C Add color bar for entire plot"""
cbar_ax = fig.add_axes([0.94, 0.1, 0.1, 0.78])
plot_colormap_legend(cbar_ax, colormap_grid, precip_range=(-20, 20), temp_range=(-1.5, 1.5))
# add cbar in the figure, for overall figure, not subplots
# Define the position of the colorbar
# 
# cbar = fig.colorbar(im, cax=cbar_ax, extend='both')

# # Increase distance between colorbar label and colorbar
# cbar.ax.yaxis.labelpad = 20
# if mode == 'dif':
#     # cbar.set_label(f'$\Delta_{{ptb_type}}_{,{var}}$ [{unit}]', size='15')
#     cbar.set_label(
#         rf'$\Delta_{{{ptbtype}, {var}}}$ [{unit}]', size=15)
#     if mode == 'std':
#         cbar.set_label(
#             f'{var} - model member std [{unit}]', size='15')
# cbar.ax.tick_params(labelsize=12)

# # adjust subplot spacing to be smaller
# plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1,
#                     top=0.9, wspace=0.05, hspace=0.05)

hedging_patch = mpatches.Patch(
    label='< 80% of model members agree on sign of change', hatch='////', edgecolor='black', facecolor='none')

# Add the custom legend to the plot
fig.legend(handles=[
                        hedging_patch], loc='lower center', bbox_to_anchor=(0.5, -0.05), fontsize=12)

# if plotsave == 'save':
o_folder_diff = "/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/04. Figures/01. Climate data/02. Perturbations/"

os.makedirs(f"{o_folder_diff}/", exist_ok=True)
o_file_name = f"{o_folder_diff}/Mosaic.{var}.{y0}_{ye}_{timeframe}_{diftype}{extra_id}.png"
# plt.savefig(o_file_name, bbox_inches='tight')

plt.show()
    
    
    


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
                    colors = [(0, 'xkcd:mocha'), (zero_scaled_t,
                                                      'xkcd:white'), (1, 'xkcd:aquamarine')]
                    custom_cmap_p = LinearSegmentedColormap.from_list(
                        'custom_cmap', colors)
                    
                    vmin_p = -20
                    vmax_p = 20
                    zero_scaled_p = (abs(vmin)/(abs(vmin)+abs(vmax)))
                           
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
                    elif var =="Temperature":
                        all_vars_t = all_diffs_avg
                        all_vars_local_t = all_diffs_avg_local
                        all_model_diffs_avg_t = all_model_diffs_avg
                    
                


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
                               gridspec_kw={'wspace': 0.0, 'width_ratios': [2.0, 1.63], 'hspace': 0.2})#,'height_ratios': [1, 1]})
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

sign_diff_p = xr.concat([np.sign(diff['pr'])
                      for diff in all_model_diffs_avg_p], dim="models")
sign_diff_t = xr.concat([np.sign(diff['tas'])
                      for diff in all_model_diffs_avg_t], dim="models")
within_threshold_p = (
    abs(sign_diff_p.mean(dim="models")) > 0.8)
within_threshold_t = (
    abs(sign_diff_t.mean(dim="models")) > 0.8)

# # Step 3: Combine the conditions to create the final mask
# within_threshold_p = agreement_on_sign_p.sel(lon=slice(lonmin, lonmax), lat=slice(latmax, latmin))
# within_threshold_t = agreement_on_sign_t.sel(lon=slice(lonmin, lonmax), lat=slice(latmax, latmin))
# Step 4: Convert the mask to 2D by removing the singleton 'variable' dimension if needed
within_threshold_2d_p = within_threshold_p.astype(
    int).squeeze()
within_threshold_2d_t = within_threshold_t.astype(
    int).squeeze()

# Step 6: Overlay the mask with dots in areas where agreement criteria are met
axes[indices[1]].contourf(
    all_diffs_avg_local.lon, all_diffs_avg_local.lat, within_threshold_2d_t,
    levels=[0.5, 1.5], colors='none', hatches=['////'], transform=ccrs.PlateCarree()
)
axes[indices[3]].contourf(
    all_diffs_avg_local.lon, all_diffs_avg_local.lat, within_threshold_2d_p,
    levels=[0.5, 1.5], colors='none', hatches=['////'], transform=ccrs.PlateCarree()
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
fig.legend(handles=[hedging_patch], loc='lower center', bbox_to_anchor=(0.53, 0), fontsize=12)

# if plotsave == 'save':
o_folder_diff = "/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/04. Figures/01. Climate data/02. Perturbations/"

os.makedirs(f"{o_folder_diff}/", exist_ok=True)
o_file_name = f"{o_folder_diff}/Mosaic.{var}.{y0}_{ye}_{timeframe}_{diftype}{extra_id}.png"
# plt.savefig(o_file_name, bbox_inches='tight')

plt.show()


#%% 
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

precip_min, precip_max = -20, 20
temp_min, temp_max = -1.5, 1.5

# Grid resolution
grid_size = 256

# Define corner RGB colors
color_dry_warm = np.array([1.0, 0.4, 0.0])   # Top-left (Dry + Warm)
color_dry_cold = np.array([0.6, 0.0, 0.6])   # Bottom-left (Dry + Cold)
color_wet_warm = np.array([0.6, 1.0, 0.2])   # Top-right (Wet + Warm)
color_wet_cold = np.array([0.0, 0.4, 1.0])   # Bottom-right (Wet + Cold)
color_center  = np.array([1.0, 1.0, 1.0])    # Center (Neutral)

# Create continuous colormap
custom_colormap = np.zeros((grid_size, grid_size, 3))
for i in range(grid_size):  # Temp (Y)
    for j in range(grid_size):  # Precip (X)
        # Normalize to [-1, 1]
        x = -1 + 2 * j / (grid_size - 1)  # ΔP
        y = -1 + 2 * i / (grid_size - 1)  # ΔT

        wx = (x + 1) / 2  # 0 to 1
        wy = (y + 1) / 2

        top = (1 - wx) * color_dry_warm + wx * color_wet_warm
        bottom = (1 - wx) * color_dry_cold + wx * color_wet_cold
        color = (1 - wy) * bottom + wy * top

        # Radial fade to white at center
        r = np.sqrt(x**2 + y**2)
        fade = np.clip(1 - r, 0, 1)
        color = (1 - fade) * color + fade * color_center

        custom_colormap[i, j] = color

# Generate mock ΔT and ΔP data
lon = np.linspace(-180, 180, 200)
lat = np.linspace(-90, 90, 100)
lon2d, lat2d = np.meshgrid(lon, lat)
delta_temp = np.random.uniform(temp_min, temp_max, size=lon2d.shape)
delta_precip = np.random.uniform(precip_min, precip_max, size=lon2d.shape)

# Normalize to [0, grid_size - 1]
rows = ((delta_temp - temp_min) / (temp_max - temp_min) * (grid_size - 1)).astype(int)
cols = ((delta_precip - precip_min) / (precip_max - precip_min) * (grid_size - 1)).astype(int)
rows = np.clip(rows, 0, grid_size - 1)
cols = np.clip(cols, 0, grid_size - 1)

# Apply colormap
colored_map = custom_colormap[rows, cols]

# Plotting
fig = plt.figure(figsize=(14, 6))

# Map
ax_map = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
ax_map.imshow(colored_map, extent=[-180, 180, -90, 90], origin='lower', transform=ccrs.PlateCarree())
ax_map.coastlines()
ax_map.set_title("Smooth 2D Color Blend")

# Legend
ax_legend = fig.add_subplot(1, 2, 2)
ax_legend.imshow(custom_colormap, origin='lower',
                 extent=[precip_min, precip_max, temp_min, temp_max],
                 aspect='auto')
ax_legend.set_xlabel("ΔPrecipitation")
ax_legend.set_ylabel("ΔTemperature")
ax_legend.set_title("Smooth 2D Color Legend")
ax_legend.grid(True)

plt.tight_layout()
plt.show()


   
#%% Presentation plot: Increased Irrigation Area

folder_data=o_folder_data = ("/Users/magaliponds/OneDrive - Vrije Universiteit Brussel/1. VUB/02. Coding/01. IRRMIP/03. Data/01. Input files/Irrigation_Data_Table.csv"
                       )

inputdata=pd.read_csv(folder_data, header=0, index_col="Year")
# Define blue color shades and hatch patterns
# colors = ['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087']
cmap = cm.get_cmap('Blues', 6)  # Get 5 different shades of blue
colors = [cmap(i) for i in np.linspace(0.2, 1, 6)]  # Adjust brightness

hatch_patterns = ['//', '\\', '//', '\\', '//']

plt.figure(figsize=(8,5))
stacked_areas = plt.stackplot(inputdata.index,
              inputdata["USA (Mha)"],
              inputdata["Pakistan (Mha)"],
              inputdata["China (Mha)"],
              inputdata["India (Mha)"],
              inputdata["Other countries (Mha)"],
              labels=["USA", "Pakistan", "China", "India", "Other countries"],
              alpha=0.7, colors=colors)  # Adjust transparency

# Annotate country names on the plot
country_labels = ["USA", "Pakistan", "China", "India", "Other \n countries","total"]
mid_year = inputdata.index[len(inputdata) // 2]  # Select a middle year for annotation
cumulative_values = np.zeros(len(inputdata))  # Initialize cumulative sum for stacking
stacked_data = np.cumsum(inputdata.values, axis=1)
for i, country in enumerate(inputdata.columns[:-1]):
    mid_year = inputdata.index[len(inputdata) // 2]  # Middle year for annotation
    mid_index = 10#len(inputdata) // 2  # Middle index of data
    country_label=country_labels[i]
    
    # Compute the middle value of the stacked region
    if i == 0:
        mid_value = inputdata.iloc[mid_index, i] / 2  # First country is at the bottom
    else:
        mid_value = (stacked_data[mid_index, i] + stacked_data[mid_index, i - 1]) / 2  # Middle of stacked area

    # Annotate the country in the correct position
    plt.text(1998, mid_value, country_label, fontsize=12,  ha='right', va='center')
             # bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# Labels and title
plt.xlabel("Year")

# Set xticks to match the "Year" column (index-based ticks)

plt.ylabel("Area Equipped for Irrigation (Mha)")
plt.title("Global Irrigation Expansion Over Time")
# plt.legend(loc="upper left")

# Show the plot
plt.show()

    
    
    
    
    
    
    
    
    
    
    
    