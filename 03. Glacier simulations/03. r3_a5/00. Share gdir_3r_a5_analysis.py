# -*- coding: utf-8 -*-
import multiprocessing
import traceback
import logging
from multiprocessing import Pool, set_start_method, get_context
from multiprocessing import Process
import concurrent.futures
from matplotlib.lines import Line2D
import oggm
from oggm import utils, cfg, workflow, tasks, DEFAULT_BASE_URL, graphics, global_tasks
from oggm.core import massbalance, flowline
from oggm.utils import floatyear_to_date, hydrodate_to_calendardate,compile_glacier_statistics
from oggm.sandbox import distribute_2d
from oggm.sandbox.edu import run_constant_climate_with_bias
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as clrs
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

from tqdm import tqdm
import pickle
import sys
import textwrap
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib import patches
function_directory = "/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/src/03. Glacier simulations"
sys.path.append(function_directory)

# %% Cell 0: Set base parameters

colors = {
    "W5E5": ["#000000"],  # "#000000"],  # Black
    # Darker to lighter shades of purple
    "E3SM": ["#785EF0", "#8F7BF1", "#A6A8F2"],
    # Darker to lighter shades of pink
    "CESM2": ["#DC267F", "#E58A9E", "#F0A2B6", "#F7BCC4"],
    # Darker to lighter shades of orange
    "CNRM": ["#FE6100", "#FE7D33", "#FE9A66", "#FEB799", "#FECDB5", "#FEF1E1"],
    "IPSL-CM6": ["#FFB000"]  # Dark purple to lighter shades
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

# %% Cell 1: Initialize OGGM with the preferred model parameter set up

folder_path = '/Users/magaliponds/Documents/00. Programming'
wd_path = f'{folder_path}/03. Modelled perturbation-glacier interactions - R13-15 A+5km2/'
os.makedirs(wd_path, exist_ok=True)
cfg.initialize(logging_level='WARNING')
cfg.PATHS['working_dir'] = utils.mkdir(wd_path, reset=False)

# make a sum dir
sum_dir = os.path.join(wd_path, 'summary')
os.makedirs(sum_dir, exist_ok=True)

# make a logging directory
log_dir = os.path.join(wd_path, "log")
os.makedirs(log_dir, exist_ok=True)

# Make a pkl directory
pkls = os.path.join(wd_path, "pkls")
os.makedirs(pkls, exist_ok=True)


cfg.PARAMS['baseline_climate'] = "GSWP3-W5E5"


cfg.PARAMS['store_model_geometry'] = True

# %% Cell 1a: Load gdirs_3r from pkl (fastest way to get started)

wd_path_pkls = f'{wd_path}/pkls/'

gdirs_3r = []
for filename in os.listdir(wd_path_pkls):
    if filename.endswith('.pkl'):
        # f'{gdir.rgi_id}.pkl')
        file_path = os.path.join(wd_path_pkls, filename)
        with open(file_path, 'rb') as f:
            gdir = pickle.load(f)
            gdirs_3r.append(gdir)


#%%

cfg.PARAMS['use_multiprocessing'] = True
cfg.PARAMS['core'] = 9 # 🔧 set number of cores

def main():
    
    # Compile the stats into a single DataFrame
    ofilesuffix='gdir_3r_a5_share_VA_analysis.csv'
    opath=f'{wd_path}/masters/gdir_3r_a5_share_VA_analysis.csv'
    df = compile_glacier_statistics(gdirs_3r, path=opath)
    
    # Define thresholds
    thresholds = [5, 2, 1]
    summary_data = []
    
    # Total area and volume (for % calculation)
    total_area = df['area_km2'].sum()
    total_volume = df['volume_km3'].sum()
    
    for t in thresholds:
        sub = df[df['area_km2'] > t]
        n_glaciers = len(sub)
        area = sub['area_km2'].sum()
        volume = sub['volume_km3'].sum()
        summary_data.append({
            'Threshold (A > km²)': t,
            '# Glaciers': n_glaciers,
            'Total Area (km²)': area,
            'Area % of Total': area / total_area * 100,
            'Total Volume (km³)': volume,
            'Volume % of Total': volume / total_volume * 100
        })
    
    # Create summary table
    summary_df = pd.DataFrame(summary_data)
    
    # Display the result
    import ace_tools as tools; tools.display_dataframe_to_user(name="Glacier Area Threshold Summary", dataframe=summary_df)
    
    summary_df.to_csv(f'{opath}/gdir_3r_a5_share_VA_analysis_summary.csv')
    
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()        
    
#%%
opath=f'{wd_path}/masters/gdir_3r_a5_share_VA_analysis.csv'
df = pd.read_csv(opath)
thresholds = [5,2,1, 0.5]
summary_data = []

# Total area and volume (for % calculation)
total_area = df['rgi_area_km2'].sum()
total_volume = df['inv_volume_km3'].sum()
total_glaciers = len(df)

for t in thresholds:
    print("threshold",t)
    sub = df[df['rgi_area_km2'] > t]
    n_glaciers = len(sub)
    area = sub['rgi_area_km2'].sum()
    volume = sub['inv_volume_km3'].sum()
    summary_data.append({
        'threshold area': t,
        'nr_glaciers': n_glaciers,
        'nr_glaciers_share': n_glaciers/total_glaciers,
        'area_km2': area,
        'area_share': area/total_area,
        'volume_km3': volume,
        'volume_share': volume/total_volume,

    })
    
# Create summary table
summary_df = pd.DataFrame(summary_data)

summary_df.to_csv(f'{wd_path}/masters/gdir_3r_a5_share_VA_analysis_summary.csv')
    
#%%
summary_df = pd.read_csv(f'{wd_path}/masters/gdir_3r_a5_share_VA_analysis_summary.csv')
# Define parameters

colors = ['coral', 'lightcoral', 'pink', 'papayawhip']  # Colors for 70%, 80%, 90%
radius_levels = [1.3, 1.0, 0.7, 0.4]  # Outer to inner rings
ring_width = 0.3

# Set up the figure and axis
fig, ax = plt.subplots(1,3,figsize=(8, 8))
ax=ax.flatten()

# Loop over each ring layer
for (v, var) in enumerate(['nr_glaciers_share','area_share', 'volume_share']):
    percentages = summary_df[var].values # Percent of each ring to color
    for pct, color, r in zip(percentages, colors, radius_levels):
        values = [pct * 100, (1 - pct) * 100]  # colored %, white remainder
        print(values)
        wedges, _ = ax[v].pie(
            values,
            radius=r,
            colors=[color, 'lightgrey'],
            startangle=90,
            wedgeprops=dict(width=ring_width, edgecolor='white')
        )
        # Calculate angle and position for annotation
        angle = 90 + (pct * 360)+10 #/ 2
        x = r * 0.85 * np.cos(np.radians(angle))
        y = r * 0.85 * np.sin(np.radians(angle))
        
        # Add annotation to the chart
        ax[v].text(x, y, f"{int(pct*100)}", ha='center', va='center', fontsize=12)#, weight='bold')

    
    # Add a small white center for donut effect
    centre_circle = plt.Circle((0, 0), 0.1, color='white')
    ax[v].add_artist(centre_circle)
    ax[v].set_title(var)

handles = [
    mpatches.Patch(facecolor=colors[0], edgecolor='white', label=summary_df['threshold area'][0]),
    mpatches.Patch(facecolor=colors[1], edgecolor='white', label=summary_df['threshold area'][1]),
    mpatches.Patch(facecolor=colors[2], edgecolor='white', label=summary_df['threshold area'][2]),
    mpatches.Patch(facecolor=colors[3], edgecolor='white', label=summary_df['threshold area'][3])
]

# Add the legend to the plot
ax[1].legend(handles=handles, title="Area threshold", loc="lower center", bbox_to_anchor=(0.5, -0.5), ncols=2)
# Title and layout
# ax.set_title("Custom Concentric Rings\nOuter: 70%, Middle: 80%, Inner: 90%", fontsize=14)
plt.tight_layout()
# plt.legend(loc='lower center')
plt.show()

