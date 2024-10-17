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
import concurrent.futures
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
function_directory = "/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/src/03. Glacier simulations"
sys.path.append(function_directory)

# %% Cell 1: Set base parameters

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

members = [1, 3, 4, 6, 1]
members_averages = [1, 2, 3, 5]
models = ["IPSL-CM6", "E3SM", "CESM2", "CNRM", "W5E5"]
models_shortlist = ["IPSL-CM6", "E3SM", "CESM2", "CNRM"]
timeframe = "monthly"

y0_clim = 1985
ye_clim = 2014


fig_path = '/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/04. Figures/02. OGGM simulations/01. Modelled output/3r_a5/'
folder_path = '/Users/magaliponds/Documents/00. Programming'
wd_path = f'{folder_path}/03. Modelled perturbation-glacier interactions - R13-15 A+5km2/'
sum_dir = os.path.join(wd_path, 'summary')


# %% Cell 2a: plot mass balance data in histograms and gaussians (subplots and one plot for all member options, boxplot and gaussian only option)

def gaussian(x, mu, sigma, A):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))
# Helper function to plot histogram and Gaussian


def plot_gaussian(ax, mb, bin_size, color, label, zorder, linestyle='-', gaussian_only=False):
    if not gaussian_only:
        # Plot histogram if gaussian_only is False
        n, bins, _ = ax.hist(mb.B, bins=bin_size, align='left', rwidth=0.8,
                             facecolor=color, edgecolor=color, alpha=0.6, zorder=zorder)
    else:
        # Compute histogram without plotting
        n, bins = np.histogram(mb.B, bins=bin_size)

    # Gaussian curve fitting and plot
    bin_centers = (bins[:-1] + bins[1:]) / 2
    params, _ = curve_fit(gaussian, bin_centers, n, p0=[
                          np.mean(mb.B), np.std(mb.B), np.max(n)])
    x = np.linspace(mb.B.min(), mb.B.max(), 100)
    ax.plot(x, gaussian(x, *params), color=color, label=label,
            zorder=zorder*20, linestyle=linestyle)


# Configuration variables
gaussian_only = True  # Flag to use Gaussian fit only
alpha_set = 0.8

mb_members = []

# Initialize plot

fig, ax = plt.subplots(1, 1, figsize=(10, 6), sharex=True, sharey=True)

# Load baseline data
i_path_base = os.path.join(sum_dir, 'specific_massbalance_mean_W5E5.000.csv')
mb_base = pd.read_csv(i_path_base, index_col=0).to_xarray()
bin_size = np.linspace(mb_base.B.min(), mb_base.B.max(),
                       int(len(mb_base.B) / 10))

# Plot baseline once if not using subplots
baseline_plotted = True
plot_gaussian(ax, mb_base, bin_size, "black", "W5E5.000",
              zorder=1, gaussian_only=gaussian_only)

# Iterate over models and members
for m, model in enumerate(models_shortlist):  # only perturbed models
    for member in range(members_averages[m]):
        # calculate the member averages
        if members_averages[m] > 1:
            member += 1
        linestyle = ":"  # define the linestyle for all model members
        # provide the sample id (members only) to open the datasets
        sample_id = f"{model}.00{member}"

        # provide the path to datafile
        i_path = os.path.join(
            sum_dir, f'specific_massbalance_mean_{sample_id}.csv')
        mb = pd.read_csv(i_path, index_col=0).to_xarray()  # open datafile

        # add the data to the master dataset - needed for averaging
        mb_members.append(mb.B.values)

        # Plot data by member
        plot_gaussian(ax, mb, bin_size, colors[model][member], sample_id,
                      zorder=members[m] + member + 1, linestyle=linestyle, gaussian_only=gaussian_only)

    # Add annotations and labels
    if m == 3:  # only for the last model plot
        # define number of glaciers for annotation
        total_glaciers = len(mb.B.values)
        ax.annotate(f"Total # of glaciers: {total_glaciers}", xy=(
            0.65, 0.98), xycoords='axes fraction', fontsize=10, verticalalignment='top')  # annotate amount of glaciers
        ax.annotate("Time period: 1985-2014", xy=(0.65, 0.92), xycoords='axes fraction',
                    fontsize=10, verticalalignment='top')  # annotate time period


# calculate the 10-member average
mb_member_average = np.mean(mb_members, axis=0)
# load dataframe structure in order to plot gaussian
mb_ds_members = mb.copy().to_dataframe()
mb_ds_members['B'] = mb_member_average  # add the data to the dataframe
plot_gaussian(ax, mb_ds_members, bin_size, "black", "11-member average",
              zorder=members[m] + member + 1, linestyle="--", gaussian_only=gaussian_only)

# format the plot
ax.set_ylabel("# of glaciers [-]")
ax.set_xlabel("Mean specific mass balance [mm w.e. yr$^{-1}$]")

ax.legend(loc="upper left", bbox_to_anchor=(0, 1))
ax.set_xlim(-1250, 1000)
ax.axvline(0, color='k', linestyle="dashed", lw=1, zorder=20)
plt.tight_layout()

fig_folder = os.path.join(fig_path, "03. Mass Balance", "Histogram")
os.makedirs(fig_folder, exist_ok=True)
plt.savefig(f"{fig_folder}/Gaussian_distribution_total_region_by_member.png")

plt.show()

# %% Cell 2b plot mass balance data in histograms and gaussians - by subregion (subplots and one plot for all member options, histogram and gaussian only option)

# Function to calculate weighted or simple mean


def gaussian(x, mu, sigma, A):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

# Helper function to plot area-weighted histogram and Gaussian


def plot_gaussian(ax, mb, bin_size, color, label, zorder, linestyle='-', gaussian_only=False):
    # Assuming 'area' is the area of each glacier in mb
    glacier_areas = mb['rgi_area_km2'].values

    if not gaussian_only:
        # Plot histogram, weighted by glacier area
        n, bins, _ = ax.hist(mb.B, bins=bin_size, align='left', rwidth=0.8,
                             weights=glacier_areas,  # This applies area weighting to the histogram
                             facecolor=color, edgecolor=color, alpha=0.2, zorder=zorder, label=f"hist {label}")
    else:
        # Compute histogram without plotting, also weighted by glacier area
        n, bins = np.histogram(mb.B, bins=bin_size, weights=glacier_areas)

    # Gaussian curve fitting and plot
    bin_centers = (bins[:-1] + bins[1:]) / 2
    params, _ = curve_fit(gaussian, bin_centers, n, p0=[
                          np.mean(mb.B), np.std(mb.B), np.max(n)])
    x = np.linspace(mb.B.min(), mb.B.max(), 100)

    # Plot the Gaussian curve with the fitted parameters
    ax.plot(x, gaussian(x, *params), color=color, label=label,
            zorder=zorder*20, linestyle=linestyle)


# Configuration variables
alpha_set = 0.8

regions = [13, 14, 15]
subregions = [9, 3, 3]

# Initialize plot
n_rows = 5
n_cols = 3
subplots = True

if subplots == True:
    gaussian_only = False  # Flag to use Gaussian fit only

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 6), sharex=True, sharey=False, gridspec_kw={
        'hspace': 0.15, 'wspace': 0.25})  # create a new figure
    axes = axes.flatten()
else:
    gaussian_only = True  # Flag to use Gaussian fit only
    fig, axes = plt.subplots(figsize=(8, 5), sharex=True, sharey=False, gridspec_kw={
        'hspace': 0.15, 'wspace': 0.25})  # create a new figure


# Define distinct colors for regions
region_colors = {
    13: 'blue',    # Blue for region 13
    14: 'crimson',    # Pink for region 14
    15: 'orange'   # Orange for region 15
}

plot_index = 0


# Load dataset
ds = pd.read_csv(
    f"{wd_path}masters/master_gdirs_r3_a5_rgi_date_A_V_B_hugo.csv")

# Filter out 'sample_id's that end with '0', except for 'IPSL-CM6.000'
ds = ds[(~ds['sample_id'].str.endswith('0')) |
        (ds['sample_id'] == 'IPSL-CM6.000')]

# Define custom aggregation functions for grouping
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
    'B_delta': 'mean',
    'sample_id': 'first'
}

# Step 1: Group by 'rgi_id', apply custom aggregation
grouped_ds = ds.groupby('rgi_id').agg(aggregation_functions).reset_index()

# Step 2: Replace the 'sample_id' column with "11-member average"
grouped_ds['sample_id'] = '11-member average'

# Step 3: Keep only the required columns
grouped_ds = grouped_ds[['rgi_id', 'rgi_region', 'rgi_subregion', 'full_name', 'cenlon', 'cenlat', 'rgi_date',
                         'rgi_area_km2', 'rgi_volume_km3', 'sample_id', 'B_noirr', 'B_irr', 'B_delta']]

# Overwrite the original dataset with the grouped one
ds = grouped_ds


for r, region in enumerate(regions):
    for sub in range(subregions[r]):
        region_id = f"{region}.0{sub+1}"
        print(region_id)
        color = region_colors[region]
        subregion_ds = ds[ds['rgi_subregion'].str.contains(
            f"{region}.0{sub+1}")]
        print(subregion_ds.rgi_id)
        if subplots == True:
            ax = axes[plot_index]
        else:
            ax = axes

        # Load baseline data
        mb_base = subregion_ds[['rgi_id', 'rgi_area_km2', 'B_irr']].rename(
            columns={'B_irr': 'B'})  # working with 11 member averages

        bin_size = np.linspace(mb_base.B.min(), mb_base.B.max(), 10)

        # Plot baseline once if not using subplots
        baseline_plotted = True
        plot_gaussian(ax, mb_base, bin_size, "black", "W5E5.000",
                      zorder=1, gaussian_only=False)

        # Iterate over models and members
        # only perturbed models
        for m, model in enumerate(models_shortlist[1]):
            for member in range(1):  # range(members_averages[m]):
                linestyle = '-'
                # provide the path to datafile
                filtered_mb = subregion_ds[['rgi_id', 'rgi_area_km2', 'B_noirr']].rename(columns={
                                                                                         'B_noirr': 'B'})

                # Plot data by member
                plot_gaussian(ax, filtered_mb, bin_size, color, "11-member average",  # colors[model][member]
                              zorder=members[m] + member + 1, linestyle=linestyle, gaussian_only=gaussian_only)

                # add vertical line at x=0
                ax.axvline(0, color='k', linestyle="dashed", lw=1, zorder=20)

            # Add annotations and labels
            if subplots == True and m == 1:
                total_glaciers = len(filtered_mb.B.values)
                ax.annotate(f"{region_id}", xy=(
                    0.02, 0.95), xycoords='axes fraction', fontsize=10, verticalalignment='top', fontweight='bold')  # annotate amount of glaciers
                ax.annotate(f"{total_glaciers}", xy=(
                    0.85, 0.95), xycoords='axes fraction', fontsize=10, verticalalignment='top', fontstyle='normal')  # annotate amount of glaciers
            elif subplots == False and m == 1 and r == 1:
                total_glaciers = len(ds.B_irr.values)
                ax.annotate(f"Region 13,14,15 A > 5 km$^2$", xy=(
                    0.02, 0.95), xycoords='axes fraction', fontsize=10, verticalalignment='top', fontweight='bold')  # annotate amount of glaciers
                ax.annotate(f"{total_glaciers}", xy=(
                    0.85, 0.95), xycoords='axes fraction', fontsize=10, verticalalignment='top', fontstyle='normal')  # annotate amount of glaciers

        plot_index += 1
        # calculate the 10-member average
        # mb_member_average = np.mean(mb_members, axis=0)
        # # load dataframe structure in order to plot gaussian
        # mb_ds_members = filtered_mb.copy().to_dataframe()
        # mb_ds_members['B'] = mb_member_average  # add the data to the dataframe
        # plot_gaussian(ax, mb_ds_members, bin_size, color, "11-member average",
        #               zorder=members[m] + member + 1, linestyle="--", gaussian_only=False)

# format the plot
# if subplots==True:
fig.text(0.32, 0.04,
         "Mean specific mass balance [mm w.e. yr$^{-1}$]", va='center', rotation='horizontal', fontsize=12)
fig.text(0.04, 0.5, "Area [km$^{2}$]",
         va='center', rotation='vertical', fontsize=12)
# else:
# fig.text(0.5, 0.04, "Mean specific mass balance [mm w.e. yr$^{-1}$]", va='center', rotation='horizontal', fontsize=12)
# fig.text(0.04, 0.5, "Area [km$^{2}$]", va='center', rotation='vertical', fontsize=12)

# Create a legend for the regions and distinguish irr/noirr
region_legend_patches = [
    mpatches.Patch(color='blue', label='NoIrr Region 13 (Blue)'),
    mpatches.Patch(color='crimson', label='NoIrr Region 14 (Pink)'),
    mpatches.Patch(color='orange', label='NoIrr Region 15 (Orange)'),
    mpatches.Patch(color='black', label='Irr All Regions W5E5.000 (Grey)'),

    # Create custom lines with specific line styles
    mlines.Line2D([], [], color='grey', linestyle='-',
                  label='NoIrr - 11-member average'),
    mlines.Line2D([], [], color='grey', linestyle=':',
                  label='NoIrr - Individual member'),
    mlines.Line2D([], [], color='black', linestyle='-',
                  label='Irr - W5E5.000'),

]

if subplots == False:
    plt.legend(handles=region_legend_patches, loc="center",
               bbox_to_anchor=(0.5, -0.3), ncols=3)
else:
    plt.legend(handles=region_legend_patches, loc="center",
               bbox_to_anchor=(-0.8, -1.3), ncols=3)

ax.set_xlim(-1500, 1000)

plt.tight_layout()

fig_folder = os.path.join(fig_path, "03. Mass Balance", "Histogram")
os.makedirs(fig_folder, exist_ok=True)
if subplots == False:
    plt.savefig(
        f"{fig_folder}/Gaussian_distribution_total_region_by_region_1plot.png")

else:
    plt.savefig(
        f"{fig_folder}/Gaussian_distribution_total_region_by_region_subplots.png")
plt.show()


# %% Cell 3a: Plot the specific mass balances using boxplots, different subsets, weighted by region - print median
# Initialize plot
fig, ax = plt.subplots(figsize=(10, 10))

# Load dataset
ds = pd.read_csv(
    f"{wd_path}masters/master_gdirs_r3_a5_rgi_date_A_V_B_hugo.csv")

# Filter out 'sample_id's that end with '0', except for 'IPSL_CM6.000'
ds = ds[(~ds['sample_id'].str.endswith('0')) |
        (ds['sample_id'] == 'IPSL-CM6.000')]

# Define a custom aggregation function
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
    'B_delta': 'mean',
    'sample_id': 'first'
}

# Step 1: Group by 'rgi_id' and 'sample_id', apply the custom aggregation
grouped_ds = ds.groupby(['rgi_id']).agg(aggregation_functions).reset_index()

# Step 2: Replace the 'sample_id' column with "11-member average"
grouped_ds['sample_id'] = '11-member average'

# Step 3: Keep only the required columns
grouped_ds = grouped_ds[['rgi_id', 'rgi_region', 'rgi_subregion',
                         'full_name', 'cenlon', 'cenlat', 'rgi_date', 'rgi_area_km2',
                         'rgi_volume_km3', 'sample_id', 'B_noirr', 'B_irr', 'B_delta']]

# If you want to overwrite the original DataFrame with the grouped one
ds = grouped_ds

regions = [13, 14, 15]
subregions = [9, 3, 3]
region_colors = {13: 'blue', 14: 'crimson', 15: 'orange'}
v_space = 0.75

# Storage for combined data
all_noirr, all_irr = [], []
position_counter = 1

# Function to calculate either the weighted or simple mean


def calc_mean(df, col, output_col_name, use_weights=True):
    if use_weights:
        # Calculate the weighted mean based on 'rgi_area_km2' as the weight
        mean = df.groupby('rgi_id', group_keys=False).apply(
            lambda g: pd.Series({
                output_col_name: np.average(g[col], weights=g['rgi_area_km2'])
            })
        ).reset_index()
    else:
        # Calculate the regular mean if no weights are provided
        mean = df.groupby('rgi_id')[col].mean().reset_index().rename(
            columns={col: output_col_name})

    return mean


# Function to plot the boxplots
def plot_boxplot(data, position, label, color, alpha, is_noirr):
    return ax.boxplot(data, patch_artist=True, labels=label if is_noirr else [""],  # only set labels for Noirr
                      vert=False, widths=0.35,
                      boxprops=dict(facecolor=color, alpha=alpha,
                                    edgecolor='none'),
                      medianprops=dict(color='black'),
                      positions=[position], showfliers=False, zorder=2)


# Set this flag to True to use weighted mean, False for unweighted
use_weights = False

for r, region in enumerate(regions):
    for sub in range(subregions[r]):
        subregion_ds = ds[ds['rgi_subregion'].str.contains(
            f"{region}.0{sub+1}")]

        noirr_mean = calc_mean(subregion_ds, 'B_noirr',
                               'B_noirr_mean', use_weights)
        irr_mean = calc_mean(subregion_ds, 'B_irr', 'B_irr_mean', use_weights)

        all_noirr.append(noirr_mean['B_noirr_mean'].values)
        all_irr.append(irr_mean['B_irr_mean'].values)

        color = region_colors[region]
        label = [f"{region}-0{sub+1}"]
        box_noirr = plot_boxplot(
            noirr_mean['B_noirr_mean'], position_counter, label, color, 0.5, is_noirr=True)
        box_irr = plot_boxplot(
            irr_mean['B_irr_mean'], position_counter + v_space, "", color, 1.0, is_noirr=False)

        # Annotate medians for the regions
        for box, median_value, offset in zip([box_noirr, box_irr],
                                             [noirr_mean['B_noirr_mean'].median(
                                             ), irr_mean['B_irr_mean'].median()],
                                             [-1, 0.5]):
            x_median, y_median = box['medians'][0].get_xydata()[1]
            plt.text(x_median - 1, y_median + offset, f'{median_value:.1f}',
                     va='center', ha='center', fontsize=10, color='black',
                     backgroundcolor="white", zorder=1)

        # Annotate glacier count for the regions
        plt.text(900, position_counter + (v_space / 2), len(noirr_mean),
                 va='center', ha='center', fontsize=10, color='black',
                 backgroundcolor="white", fontstyle="italic", zorder=1)

        position_counter += 3

# Plot overall average boxplots
all_noirr_combined = np.concatenate(all_noirr)
all_irr_combined = np.concatenate(all_irr)

# Create a boxplot for the overall Irr and Noirr mean
avg_noirr = plot_boxplot(all_noirr_combined, position_counter, [
                         "Average"], 'grey', 0.5, is_noirr=True)
avg_irr = plot_boxplot(all_irr_combined, position_counter +
                       v_space, "", 'black', 1.0, is_noirr=False)

# Annotate the median values for the average irr/noirr
for box, median_value, offset in zip([avg_noirr, avg_irr],
                                     [np.median(all_noirr_combined),
                                      np.median(all_irr_combined)],
                                     [-1, 0.5]):
    x_median, y_median = box['medians'][0].get_xydata()[1]
    plt.text(x_median - 1, y_median + offset, f'{median_value:.1f}',
             va='center', ha='center', fontsize=10, color='black',
             bbox=dict(boxstyle='round,pad=0.001',
                       facecolor='white', edgecolor='none'),
             zorder=1)

# Annotate the number of glaciers
plt.text(900, position_counter + (v_space / 2), len(all_irr_combined),  # Plot number of glaciers
         va='center', ha='center', fontsize=10, color='black',
         fontstyle='italic', backgroundcolor="white", zorder=1)

plt.text(900, position_counter + 2, "# of glaciers",  # Plot number of glaciers title
         va='center', ha='center', fontsize=10, color='black', fontstyle='italic', backgroundcolor="white", zorder=1)

# Add legend and customize plot
region_legend_patches = [mpatches.Patch(color=color, label=f'Region {r} ({color.capitalize()})')
                         for r, color in region_colors.items()]
region_legend_patches += [mpatches.Patch(color='grey', label='Average (NoIrr)'),
                          mpatches.Patch(color='black', label='Average (Irr)')]

ax.legend(handles=region_legend_patches, loc='lower center',
          bbox_to_anchor=(0.5, -0.15), ncols=3)
ax.axvline(x=0, color='black', linestyle='--', linewidth=1, zorder=0)
ax.set_xlabel('Mean B_noirr and B_irr')
ax.set_ylabel('Regions and Subregions')
ax.set_xlim(-1350, 1000)
plt.tight_layout()

fig_folder = os.path.join(fig_path, "03. Mass Balance", "Boxplot")
os.makedirs(fig_folder, exist_ok=True)
plt.savefig(
    f"{fig_folder}/Gaussian_distribution_total_region_by_region_median.png")
plt.show()


# %% Cell 4b: Plot the specific mass balances using boxplots, different subsets, weighted by region - print area weighted mean
# Initialize plot
fig, ax = plt.subplots(figsize=(10, 10))

# Load dataset
ds = pd.read_csv(
    f"{wd_path}masters/master_gdirs_r3_a5_rgi_date_A_V_B_hugo.csv")

# Filter out 'sample_id's that end with '0', except for 'IPSL-CM6.000'
ds = ds[(~ds['sample_id'].str.endswith('0')) |
        (ds['sample_id'] == 'IPSL-CM6.000')]

# Define custom aggregation functions for grouping
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
    'B_delta': 'mean',
    'sample_id': 'first'
}

# Step 1: Group by 'rgi_id', apply custom aggregation
grouped_ds = ds.groupby('rgi_id').agg(aggregation_functions).reset_index()

# Step 2: Replace the 'sample_id' column with "11-member average"
grouped_ds['sample_id'] = '11-member average'

# Step 3: Keep only the required columns
grouped_ds = grouped_ds[['rgi_id', 'rgi_region', 'rgi_subregion', 'full_name', 'cenlon', 'cenlat', 'rgi_date',
                         'rgi_area_km2', 'rgi_volume_km3', 'sample_id', 'B_noirr', 'B_irr', 'B_delta']]

# Overwrite the original dataset with the grouped one
ds = grouped_ds
use_weights = True
# Define regions and subregions to loop through
regions = [13, 14, 15]
subregions = [9, 3, 3]  # Subregions count for each region
region_colors = {13: 'blue', 14: 'crimson', 15: 'orange'}
v_space = 0.75  # Vertical space between irr and noirr boxplots

# Storage for combined data
all_noirr, all_irr = [], []
position_counter = 1

# Function to calculate weighted or simple mean


def calc_mean(df, col, output_col_name, use_weights=True):
    if use_weights:
        # Calculate weighted mean based on 'rgi_area_km2' as the weight
        mean = df.groupby('rgi_id', group_keys=False).apply(
            lambda g: pd.Series({
                output_col_name: np.average(g[col], weights=g['rgi_area_km2'])
            })
        ).reset_index()
    else:
        # Calculate the simple mean if no weights are used
        mean = df.groupby('rgi_id')[col].mean().reset_index().rename(
            columns={col: output_col_name})
    return mean

# Function to plot boxplots with black dot for mean and annotation


def plot_boxplot(data, position, label, color, alpha, is_noirr):
    # Create the boxplot
    box = ax.boxplot(data, patch_artist=True, labels=label if is_noirr else [""],  # Only set labels for Noirr
                     vert=False, widths=0.35,
                     boxprops=dict(facecolor=color, alpha=alpha,
                                   edgecolor='none'),
                     medianprops=dict(color='black'),
                     positions=[position], showfliers=False, zorder=2)

    # Calculate the regional mean
    mean_value = np.mean(data)

    # Plot black dot at the mean position
    # 'ko' for black dot ('k' = black, 'o' = dot)
    ax.plot(mean_value, position, 'ko', zorder=3)

    # Annotate text: above the boxplot for Irr, below for Noirr
    if is_noirr:
        plt.text(mean_value, position - 0.7, f'{mean_value:.1f}',  # Place below for Noirr
                 va='center', ha='center', fontsize=10, color='black',
                 bbox=dict(boxstyle='round,pad=0.001',
                           facecolor='white', edgecolor='none'),
                 zorder=3)
    else:
        plt.text(mean_value, position + 0.7, f'{mean_value:.1f}',  # Place above for Irr
                 va='center', ha='center', fontsize=10, color='black',
                 bbox=dict(boxstyle='round,pad=0.001',
                           facecolor='white', edgecolor='none'),
                 zorder=3)

    return box


# Example usage: Main loop through regions and subregions
for r, region in enumerate(regions):
    for sub in range(subregions[r]):
        # Filter subregion-specific data
        subregion_ds = ds[ds['rgi_subregion'].str.contains(
            f"{region}.0{sub+1}")]

        # Calculate mean values for Noirr and Irr
        noirr_mean = calc_mean(subregion_ds, 'B_noirr',
                               'B_noirr_mean', use_weights)
        irr_mean = calc_mean(subregion_ds, 'B_irr', 'B_irr_mean', use_weights)

        all_noirr.append(noirr_mean['B_noirr_mean'].values)
        all_irr.append(irr_mean['B_irr_mean'].values)

        # Set color and label based on the region
        color = region_colors[region]
        label = [f"{region}-0{sub+1}"]

        # Plot Noirr and Irr boxplots
        box_noirr = plot_boxplot(
            noirr_mean['B_noirr_mean'].values, position_counter, label, color, 0.5, is_noirr=True)
        box_irr = plot_boxplot(
            irr_mean['B_irr_mean'].values, position_counter + v_space, "", color, 1.0, is_noirr=False)

        # Annotate the number of glaciers and delta between the two columns
        num_glaciers = len(noirr_mean)
        delta = np.mean(noirr_mean['B_noirr_mean'].values) - \
            np.mean(irr_mean['B_irr_mean'].values)

        # Display number of glaciers and delta
        plt.text(850, position_counter + (v_space / 2),
                 f'{num_glaciers} ',  # \nΔ = {delta:.2f}',
                 va='center', ha='center', fontsize=10, color='black',
                 backgroundcolor="white", fontstyle="italic", zorder=1)

        # Increment position counter for the next subregion
        position_counter += 3

# Combine data for overall average plots
all_noirr_combined = np.concatenate(all_noirr)
all_irr_combined = np.concatenate(all_irr)

# Plot overall average boxplots for Irr and Noirr
avg_noirr = plot_boxplot(all_noirr_combined, position_counter, [
                         "Average"], 'grey', 0.5, is_noirr=True)
avg_irr = plot_boxplot(all_irr_combined, position_counter +
                       v_space, "", 'black', 1.0, is_noirr=False)


# Annotate the number of glaciers for the overall average
plt.text(850, position_counter + (v_space / 2), f'{len(all_irr_combined)}',  # Display total number of glaciers
         va='center', ha='center', fontsize=10, color='black',
         fontstyle='italic', backgroundcolor="white", zorder=1)
plt.text(850, position_counter + 2, "# of glaciers",  # Plot number of glaciers title
         va='center', ha='center', fontsize=10, color='black', fontstyle='italic', backgroundcolor="white", zorder=1)

# Create custom legend elements for mean (dot) and median (stripe)
mean_dot = Line2D([0], [0], marker='o', color='w',
                  label='Area-weighted Mean (dot)', markerfacecolor='black', markersize=10)
median_stripe = Line2D([0], [0], color='black', lw=2, label='Median (stripe)')

# Add a legend for regions, mean (dot), and median (stripe)
region_legend_patches = [mpatches.Patch(color=color, label=f'Region {r} ({color.capitalize()})')
                         for r, color in region_colors.items()]
region_legend_patches += [mpatches.Patch(color='grey', label='Average (NoIrr)'),
                          mpatches.Patch(color='black', label='Average (Irr)')]

# Append the custom legend items for the mean dot and median stripe
region_legend_patches += [mean_dot, median_stripe]

# Create the legend with the updated patches
ax.legend(handles=region_legend_patches, loc='lower center',
          bbox_to_anchor=(1, -0.15), ncols=3)
# ax.legend(handles=region_legend_patches, loc='lower center',
#           bbox_to_anchor=(0.5, -0.15), ncols=3)

ax.axvline(x=0, color='black', linestyle='--', linewidth=1, zorder=0)
ax.set_xlabel('Mean B_noirr and B_irr')
ax.set_ylabel('Regions and Subregions')
ax.set_xlim(-1350, 1000)

# Adjust layout and display the plot
plt.tight_layout()
fig_folder = os.path.join(fig_path, "03. Mass Balance", "Boxplot")
os.makedirs(fig_folder, exist_ok=True)
plt.savefig(
    f"{fig_folder}/Gaussian_distribution_total_region_by_region_areaweighted_mean.png")
plt.show()


# %% Cell 5a: Create output plots for Area and volume

# define the variables for p;lotting
variables = ["volume", "area"]
factors = [10**-9, 10**-6]

variable_names = ["Volume", "Area"]
variable_axes = ["Volume [km$^3$]", "Area [km$^2$]"]
use_multiprocessing = False

# repeat the loop for area and volume
for v, var in enumerate(variables):
    print(var)
    # create a new plot for both area and volume
    fig, ax = plt.subplots(figsize=(10, 6))  # create a new figure

    # create a timeseries for all the model members to add the data, needed to calculate averages later
    member_data = []

    # load and plot the baseline data
    baseline_path = os.path.join(
        wd_path, "summary", f"climate_run_output_baseline_W5E5.000.nc")
    baseline = xr.open_dataset(baseline_path)
    ax.plot(baseline["time"], baseline[var].sum(dim="rgi_id") * factors[v],
            label="W5E5.000", color=colors["W5E5"][0], linewidth=2, zorder=15)

    # loop through all the different model x member combinations
    for m, model in enumerate(models_shortlist):
        for i in range(members_averages[m]):

            # make sure the counter for sample ids starts with 001, 000 are averages of all members by model
            # IPSL-CM6 only has 1 member, so the sample_id must end with 000
            if members_averages[m] > 1:
                i += 1

            sample_id = f"{model}.00{i}"

            # load and plot the data from the climate output run
            climate_run_opath = os.path.join(
                sum_dir, f'climate_run_output_perturbed_{sample_id}.nc')
            climate_run_output = xr.open_dataset(climate_run_opath)
            ax.plot(climate_run_output["time"], climate_run_output[var].sum(dim="rgi_id") * factors[v],
                    label=sample_id, color=colors[model][i], linewidth=2, linestyle="dotted")

            # add all the summed volumes/areas to the member list, so a multi-member average can be calculated
            member_data.append(climate_run_output[var].sum(
                dim="rgi_id").values * factors[v])

    # stack the member data
    stacked_member_data = np.stack(member_data)

    # calculate and plot volume/area 10-member mean
    mean_values = np.mean(stacked_member_data, axis=0).flatten()
    ax.plot(climate_run_output["time"].values, mean_values,
            color="black", linestyle='dashed', lw=2, label="10-member average")

    # calculate and plot volume/area 10-member min and max for ribbon
    min_values = np.min(stacked_member_data, axis=0).flatten()
    max_values = np.max(stacked_member_data, axis=0).flatten()
    ax.fill_between(climate_run_output["time"].values, min_values, max_values,
                    color="lightblue", alpha=0.3, label=f"10-member range", zorder=16)

    # Set labels and title for the combined plot
    ax.set_ylabel(variable_axes[v])
    ax.set_xlabel("Time [year]")
    ax.set_title(f"Summed {variable_names[v]}, RGI 13, A >10 km$^2$")

    # Adjust the legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
               bbox_to_anchor=(0.5, -0.15), ncol=4)
    plt.tight_layout()

    # specify and create a folder for saving the data (if it doesn't exists already) and save the plot
    o_folder_data = f"/Users/magaliponds/OneDrive - Vrije Universiteit Brussel/1. VUB/02. Coding/01. IRRMIP/04. Figures/02. OGGM simulations/01. Modelled output/3r_a5/0{v + 1}. {variable_names[v]}/00. Combined"
    os.makedirs(o_folder_data, exist_ok=True)
    o_file_name = f"{o_folder_data}/1985_2014.{timeframe}.delta.{variable_names[v]}.combined.png"
    plt.savefig(o_file_name, bbox_inches='tight')

# %% Cell 6b: Create output plots for Area and volume - per region

ds = pd.read_csv(
    f"{wd_path}masters/master_gdirs_r3_a5_rgi_date_A_V_B_hugo.csv")

# Filter out 'sample_id's that end with '0', except for 'IPSL-CM6.000'
ds = ds[(~ds['sample_id'].str.endswith('0')) |
        (ds['sample_id'] == 'IPSL-CM6.000')]

# Define custom aggregation functions for grouping
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
    'B_delta': 'mean',
    'sample_id': 'first'
}

# Step 1: Group by 'rgi_id', apply custom aggregation
grouped_ds = ds.groupby('rgi_id').agg(aggregation_functions).reset_index()

# Step 2: Replace the 'sample_id' column with "11-member average"
grouped_ds['sample_id'] = '11-member average'

# Step 3: Keep only the required columns
grouped_ds = grouped_ds[['rgi_id', 'rgi_region', 'rgi_subregion', 'full_name', 'cenlon', 'cenlat', 'rgi_date',
                         'rgi_area_km2', 'rgi_volume_km3', 'sample_id', 'B_noirr', 'B_irr', 'B_delta']]

# Overwrite the original dataset with the grouped one
ds = grouped_ds

# define the variables for p;lotting
variables = ["volume", "area"]
factors = [10**-9, 10**-6]

variable_names = ["Volume", "Area"]
variable_axes = ["Volume [km$^3$]", "Area [km$^2$]"]
use_multiprocessing = False
regions = [13, 14, 15]
subregions = [9, 3, 3]
region_colors = {
    13: 'blue',    # Blue for region 13
    14: 'crimson',    # Pink for region 14
    15: 'orange'   # Orange for region 15
}


def custom_function(rgi_id, pattern):
    return np.char.startswith(rgi_id, pattern)


# repeat the loop for area and volume
for v, var in enumerate(variables):
    print(var)
    # create a new plot for both area and volume
    n_rows = 5
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6), sharex=True, sharey=False, gridspec_kw={
                             'hspace': 0.15, 'wspace': 0.25})  # create a new figure
    plot_index = 0
    axes = axes.flatten()
    for r, region in enumerate(regions):
        for subregion in range(subregions[r]):
            ax = axes[plot_index]

            subregion = subregion+1
            region_id = f"{region}.0{subregion}"
            print(region_id)
            subregion_ds = ds[ds['rgi_subregion'].str.contains(
                f"{region_id}")]  # based on input from above

            # create a timeseries for all the model members to add the data, needed to calculate averages later
            filtered_member_data = []

            # load and plot the baseline data
            baseline_path = os.path.join(
                wd_path, "summary", f"climate_run_output_baseline_W5E5.000.nc")
            baseline = xr.open_dataset(baseline_path)
            filtered_baseline = baseline.where(
                baseline['rgi_id'].isin(subregion_ds.rgi_id.values), drop=True)
            print(len(filtered_baseline.rgi_id))
            # # Drop NaN values for rgi_id, but preserve the time dimension
            # filtered_baseline = filtered_baseline.dropna(
            #     dim='rgi_id', how='all')

            ax.plot(filtered_baseline["time"], filtered_baseline[var].sum(dim="rgi_id") * factors[v],
                    label="W5E5.000", color=colors["W5E5"][0], linewidth=2, zorder=15)

            # loop through all the different model x member combinations
            for m, model in enumerate(models_shortlist):
                for i in range(members_averages[m]):

                    # make sure the counter for sample ids starts with 001, 000 are averages of all members by model
                    # IPSL-CM6 only has 1 member, so the sample_id must end with 000
                    if members_averages[m] > 1:
                        i += 1

                    sample_id = f"{model}.00{i}"

                    # load and plot the data from the climate output run
                    climate_run_opath = os.path.join(
                        sum_dir, f'climate_run_output_perturbed_{sample_id}.nc')
                    climate_run_output = xr.open_dataset(climate_run_opath)
                    filtered_climate_run_output = climate_run_output.where(
                        climate_run_output['rgi_id'].isin(subregion_ds.rgi_id.values), drop=True)
                    ax.plot(filtered_climate_run_output["time"], filtered_climate_run_output[var].sum(dim="rgi_id") * factors[v],
                            label=sample_id, color=colors[model][i], linewidth=2, linestyle="dotted")

                    # add all the summed volumes/areas to the member list, so a multi-member average can be calculated
                    filtered_member_data.append(filtered_climate_run_output[var].sum(
                        dim="rgi_id").values * factors[v])

            # stack the member data
            stacked_member_data = np.stack(filtered_member_data)

            # calculate and plot volume/area 10-member mean
            mean_values = np.mean(stacked_member_data, axis=0).flatten()
            ax.plot(climate_run_output["time"].values, mean_values,
                    color="black", linestyle='dashed', lw=2, label="10-member average")

            # calculate and plot volume/area 10-member min and max for ribbon
            min_values = np.min(stacked_member_data, axis=0).flatten()
            max_values = np.max(stacked_member_data, axis=0).flatten()
            ax.fill_between(climate_run_output["time"].values, min_values, max_values,
                            color="lightblue", alpha=0.3, label=f"10-member range", zorder=16)

            # Determine row and column indices
            row = plot_index // n_cols
            col = plot_index % n_cols
            plot_index += 1

            # Add y label if it's the first column or bottom row
            if col == 0 and row == 2:
                # Set labels and title for the combined plot
                ax.set_ylabel(variable_axes[v])
            if row == n_rows-1:
                ax.set_xlabel("Time [year]")

            # Annotate region ID in the lower-left corner
            ax.text(0.05, 0.05, f'{region_id}', transform=ax.transAxes,
                    fontsize=12, verticalalignment='bottom', horizontalalignment='left', fontweight='bold')
            num_glaciers = len(filtered_baseline.rgi_id.values)
            ax.text(0.8, 0.05, f'{num_glaciers}', transform=ax.transAxes,
                    fontsize=12, verticalalignment='bottom', horizontalalignment='left')

    # Adjust the legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center',
               bbox_to_anchor=(1.01, 0.5), ncol=1)
    # ax.set_title(f"{region_id}")
    plt.tight_layout()

    # specify and create a folder for saving the data (if it doesn't exists already) and save the plot
    o_folder_data = f"/Users/magaliponds/OneDrive - Vrije Universiteit Brussel/1. VUB/02. Coding/01. IRRMIP/04. Figures/02. OGGM simulations/01. Modelled output/3r_a5/0{v + 1}. {variable_names[v]}/01. Subregions"
    os.makedirs(o_folder_data, exist_ok=True)
    o_file_name = f"{o_folder_data}/1985_2014.{timeframe}.delta.{variable_names[v]}.subregions.png"
    plt.savefig(o_file_name, bbox_inches='tight')


# %% Cell 7: Correlation curve

hugo_ds = pd.read_csv(
    f"{wd_path}masters/master_gdirs_r3_a5_rgi_date_A_V_B_hugo.csv")

hugo_df = hugo_ds[['rgi_id', 'B_hugo']]

members = [1, 1, 1, 1, 1]
models = ["W5E5", "E3SM", "CESM2", "CNRM", "IPSL-CM6"]

plt.figure(figsize=(10, 6))

hugo_df = hugo_df.rename(columns={'B_hugo': 'B'})
# plt.scatter(mb_base.rgi_id, mb_base.B,color=colors["W5E5"][0])
for m, model in enumerate(models):
    for member in range(members[m]):
        print(model)

        sample_id = f"{model}.00{member}"

        if model == "W5E5":
            i_path_base = os.path.join(
                sum_dir, 'specific_massbalance_mean_extended_W5E5.000.csv')
            mb = pd.read_csv(i_path_base, index_col=0).to_xarray()
        else:
            i_path = os.path.join(
                sum_dir, f'specific_massbalance_mean_{sample_id}.csv')
            mb = pd.read_csv(i_path, index_col=0).to_xarray()

        mb['B'] = mb['B']/1000
        # mb = mb.rename({'B':'B_oggm'})
        mb['B'].attrs['units'] = "m w.e. yr-1"
        mb['B'].attrs['standard_name'] = "Mean specific Mass balance 2000-2020"
        mb_df = mb.to_dataframe()

        cor_df = pd.merge(mb_df, hugo_df, on='rgi_id',
                          suffixes=('_oggm', '_hugo'))

        cor_ds = cor_df.to_xarray()
        correlation = xr.corr(cor_ds['B_oggm'], cor_ds['B_hugo']).round(2)

        # cor_ds = xr.merge([mb_ds, hugo_ds], dim='rgi_id')
        if m == 0 and member == 0:
            plt.plot(cor_ds['B_hugo'], cor_ds['B_hugo'], color='k',
                     label="correlation: 1", zorder=(sum(members)+1), alpha=1)
            plt.scatter(cor_ds['B_hugo'], cor_ds['B_oggm'], s=5, color=colors[model]
                        [member], label=f'{sample_id}, correlation: {correlation.values}', alpha=0.3, zorder=1)
        else:
            plt.scatter(cor_ds['B_hugo'], cor_ds['B_oggm'], s=5, color=colors[model]
                        [member], label=f'{sample_id}, correlation: {correlation.values}', alpha=0.3, zorder=0)

        # plt.text(-1.4,0.2, f'Correlation coefficient: {correlation.values}')

        plt.ylabel('Modelled mass balance  (m w.e. yr$^{-1}$)')
        plt.xlabel('Geodetic mass balance (m w.e. yr$^{-1}$)')
plt.legend()

fig_folder = os.path.join(fig_path, "03. Mass Balance", "Correlation")
os.makedirs(fig_folder, exist_ok=True)
plt.savefig(f"{fig_folder}/Correlation_curve_IPSL.png")
# plt.savefig(f"{fig_folder}/Correlation_curve.png")
fig.tight_layout()

plt.show()
# #


# %% Cell 8: map plot


# Configuration
names = "Code"
subregion_blocks = False
region_colors = {13: 'blue', 14: 'crimson', 15: 'orange'}

# Load datasets
df = pd.read_csv(
    f"{wd_path}masters/master_gdirs_r3_a5_rgi_date_A_V_RGIreg_B.csv")
# df = pd.read_csv(
#     f"{wd_path}masters/master_gdirs_r3_a5_rgi_date_A_V_RGIreg_B_hugo.csv")
master_ds = df[(~df['sample_id'].str.endswith('0')) |  # Exclude all the model averages ending with 0 except for IPSL
               (df['sample_id'].str.startswith('IPSL'))]

# Normalize values
# divide all the B values with 1000 to transform to m w.e. average over 30 yrs
master_ds[['B_noirr', 'B_irr', 'B_delta']] /= 1000

master_ds = master_ds[['rgi_id', 'rgi_region', 'rgi_subregion', 'full_name', 'cenlon', 'cenlat', 'rgi_date',
                       'rgi_area_km2', 'rgi_volume_km3', 'sample_id', 'B_noirr', 'B_irr', 'B_delta']]

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
    'B_delta': 'mean',
    'sample_id': 'first'
}
master_ds_avg = master_ds.groupby(['rgi_id'], as_index=False).agg({
    'B_delta': 'mean',
    'B_noirr': 'mean',
    # lamda is anonmous functions, returns 11 member average
    'sample_id': lambda _: "11 member average",
    # take first value for all columns that are not in the list
    **{col: 'first' for col in master_ds.columns if col not in ['B_noirr', 'B_delta', 'sample_id']}
})

# Aggregate data for scatter plot
master_ds_avg['grid_lon'] = np.floor(master_ds_avg['cenlon'])
master_ds_avg['grid_lat'] = np.floor(master_ds_avg['cenlat'])

# Aggregate dataset, area-weighted BDelta Birr and Bnoirr, Sample id is replaced by 11 member average
aggregated_ds = master_ds_avg.groupby(['grid_lon', 'grid_lat'], as_index=False).agg({
    'B_delta': lambda x: (x * master_ds.loc[x.index, 'rgi_area_km2']).sum() / master_ds.loc[x.index, 'rgi_area_km2'].sum(),
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
gdf = gpd.GeoDataFrame(aggregated_ds, geometry=gpd.points_from_xy(
    aggregated_ds['lon'] + 0.5, aggregated_ds['lat'] + 0.5))

# Colormap and scatter plot
custom_cmap = LinearSegmentedColormap.from_list(
    'red_white_blue', [(1, 0, 0), (1, 1, 1), (0, 0, 1)], N=256)
# norm = TwoSlopeNorm(vmin=gdf['B_irr'].min(
# ), vcenter=0, vmax=gdf['B_irr'].max())
norm = TwoSlopeNorm(vmin=-0.7, vcenter=0, vmax=0.2)

# Plot setup and plot shapefile
fig, ax = plt.subplots(figsize=(10, 7), subplot_kw={
                       'projection': ccrs.PlateCarree()})
ax.set_extent([45, 120, 13, 55], crs=ccrs.PlateCarree())
# # Load shapefiles
shapefile_path = '/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/01. Input files/03. Shapefile/Karakoram/Pan-Tibetan Highlands/Pan-Tibetan Highlands (Liu et al._2022)/Shapefile/Pan-Tibetan Highlands (Liu et al._2022)_P.shp'
shp = gpd.read_file(shapefile_path).to_crs('EPSG:4326')
shp.plot(ax=ax, edgecolor='red', linewidth=0, facecolor='lightgrey')
subregions_path = "/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/02. QGIS/RGI outlines/GTN-G_O2regions_selected.shp"
subregions = gpd.read_file(subregions_path).to_crs('EPSG:4326')

scatter = ax.scatter(gdf.geometry.x, gdf.geometry.y,
                     s=np.sqrt(gdf['rgi_area_km2'])*3, c=gdf['B_noirr'], cmap=custom_cmap, norm=norm, edgecolor='k', alpha=1)

# # Add labels, ticks, and colorbar
ax.set(xlabel='Longitude', ylabel='Latitude', xticks=np.arange(
    45, 120, 5), yticks=np.arange(13, 55, 5))
ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)

cbar_ax = fig.add_axes([0.08, -0.15, 0.28, 0.03])
cbar = plt.colorbar(plt.cm.ScalarMappable(
    cmap=custom_cmap, norm=norm), cax=cbar_ax, orientation='horizontal')
cbar.ax.tick_params(labelsize=12)
cbar.set_label(
    # '$\Delta$ $B_{Irr}$ - $B_{NoIrr}$ (m w.e. yr$^{-1}$)', fontsize=12) #for delta
    '$B_{Irr}$ (m w.e. yr$^{-1}$)', fontsize=12)  # for Birr
# '$\$B_{Irr}$ (m w.e. yr$^{-1}$)', fontsize=12) #for Bnoi

# Add volume legend
# Define the custom sizes (in data units) for the legend
# custom_sizes = [5, 50, 500]  # Example sizes for the legend - volume
custom_sizes = [200, 500, 1000, 2000]  # Example sizes for the legend -area
# Create labels for these sizes
size_labels = [f"{size:.0f}" for size in custom_sizes]

# Create legend handles using matplotlib Patch, simulating the size of scatter points
legend_handles = [plt.scatter([], [], s=np.sqrt(size)*3, edgecolor='k', facecolor='none')
                  for size in custom_sizes]  # Adjust size factor if needed

# Create the custom legend with the defined sizes
fig.legend(legend_handles, size_labels, loc="lower center", title="Total Area (km$^2$)", title_fontsize=12,
           bbox_to_anchor=(0.22, -0.1), ncol=5, fontsize=12)


# Subregion plotting
centroids = subregions.geometry.centroid

# define movements for the annotation of subregions
movements = {
    '13-01': [1.2, 0.8],
    '13-02': [6.4, 2.8],
    '13-03': [7, 0.8],
    '13-04': [-15, 1.5],
    '13-05': [-4, 0.5],
    '13-06': [13.5, -3.5],
    '13-07': [2, 1],
    '13-08': [-9.5, -5.2],
    '13-09': [-4.5, 3],
    '14-01': [1, 2],
    '14-02': [0, -5.5],
    '14-03': [1, 5],
    '15-01': [-6, 4],
    '15-02': [3, 2.5],
    '15-03': [-1, 4.5],
}
# annotate subregions
for attribute, subregion in subregions.groupby('o2region'):

    facecolor = region_colors.get(
        float(attribute[:2]), "none") if subregion_blocks else "none"

    subregion.plot(ax=ax, edgecolor="black", linewidth=2,
                   facecolor=facecolor, alpha=0.4)  # Plot the subregion
    # Get the boundary of the subregion instead of the centroid
    boundary = subregion.geometry.boundary.iloc[0]

    boundary_coords = list(boundary.coords)
    boundary_x, boundary_y = boundary_coords[0]  # First point on the boundary
    boundary_x -= movements[attribute][0]
    boundary_y -= movements[attribute][1]
    # Annotate or place text near the boundary
    if names == "Code":
        ax.text(boundary_x, boundary_y, f"{subregion['o2region'].iloc[0]}",
                horizontalalignment='center', fontsize=10, color='black', fontweight='bold')
    else:
        ax.text(boundary_x, boundary_y, f"{subregion['o2region'].iloc[0]}\n{subregion['full_name'].iloc[0]}",
                horizontalalignment='center', fontsize=10, color='black', fontweight='bold')

    centroid = subregion.geometry.centroid.iloc[0]
    centroid_x, centroid_y = centroid.x, centroid.y
    if attribute == "14-02":
        ax.plot([centroid_x, boundary_x-1.5], [centroid_y,
                boundary_y-0.3], color='black', linewidth=1)
    if attribute == "14-03":
        ax.plot([centroid_x, boundary_x], [centroid_y,
                boundary_y+0.5], color='black', linewidth=1)


# Define and iterate over grid layout
layout = [["13.01", "13.03", "13.04", "13.05", "13.06"], ["13.02", "", "", "", "13.07"], [
    "14.01", "14.02", "", "", "13.08"], ["14.03", "15.01", "15.02", "15.03", "13.09"]]
grid_positions = [[0.12 + col * (0.14 + 0.04), 0.82 - (0.14 + 0.07) * row - 0.05, 0.13, 0.14]
                  if layout[row][col] else None for row in range(4) for col in range(5)]
# Plot subregion time series
# for idx, pos in enumerate(grid_positions):
#     if pos:
#         ax_callout = fig.add_axes(pos)
#         region_id = layout[idx // 5][idx % 5]
#         print(region_id)
#         subregion_ds = master_ds_avg[master_ds_avg['rgi_subregion'].str.contains(f"{region_id}")]

#         # Baseline and model plotting
#         baseline_path = os.path.join(
#             wd_path, "summary", f"climate_run_output_baseline_W5E5.000.nc")
#         baseline = xr.open_dataset(baseline_path)
#         # Check if there are any matching rgi_id values
#         # Ensure that rgi_id values exist in both datasets
#         rgi_ids_in_baseline = baseline['rgi_id'].values
#         matching_rgi_ids = np.intersect1d(
#             rgi_ids_in_baseline, subregion_ds.rgi_id.values)
#         baseline_filtered = baseline.sel(rgi_id=matching_rgi_ids)

#         # Plot model member data
#         ax_callout.plot(baseline_filtered["time"].values, baseline_filtered['volume'].sum(dim="rgi_id") * 1e-9,
#                         label="W5E5.000", color="black", linewidth=2, zorder=15)
#         filtered_member_data = []
#         for m, model in enumerate(models_shortlist):
#             for i in range(members_averages[m]):
#                 sample_id = f"{model}.00{i + 1}" if members_averages[m] > 1 else f"{model}.000"
#                 climate_run_output = xr.open_dataset(os.path.join(
#                     sum_dir, f'climate_run_output_perturbed_{sample_id}.nc'))
#                 climate_run_output = climate_run_output.where(
#                     climate_run_output['rgi_id'].isin(subregion_ds.rgi_id.values), drop=True)
#                 ax_callout.plot(climate_run_output["time"].values, climate_run_output["volume"].sum(
#                     dim="rgi_id") * 1e-9, label=sample_id, color="grey", linewidth=2, linestyle="dotted")
#                 filtered_member_data.append(
#                     climate_run_output["volume"].sum(dim="rgi_id").values * 1e-9)

#         # Mean and range plotting
#         mean_values = np.mean(filtered_member_data, axis=0).flatten()
#         min_values = np.min(filtered_member_data, axis=0).flatten()
#         max_values = np.max(filtered_member_data, axis=0).flatten()
#         ax_callout.plot(climate_run_output["time"].values, mean_values,
#                         color="blue", linestyle='dashed', lw=2, label="11-member average")
#         ax_callout.fill_between(
#             climate_run_output["time"].values, min_values, max_values, color="lightblue", alpha=0.3)

#         # Subplot formatting
#         ax_callout.set_title(region_id, fontweight="bold", bbox=dict(
#             facecolor='white', edgecolor='none', pad=1))
#         # Count the number of glaciers (assuming each 'rgi_id' represents a glacier)
#         glacier_count = subregion_ds['rgi_id'].nunique()
#         # Add number of glaciers as a text annotation in the lower left corner
#         ax_callout.text(0.05, 0.05, f"{glacier_count}",
#                         transform=ax_callout.transAxes, fontsize=12, verticalalignment='bottom', fontstyle='italic')
#         # ax_callout.set_xlim(-3, 3)
#         # ax_callout.set_ylim(0, 20)
#         if idx < len(grid_positions) - 5:
#             ax_callout.tick_params(axis='x', labelbottom=False)
#         # if idx % 5 != 0:
#             ax_callout.tick_params(axis='y', labelleft=False)

# Sample data for the example plot (volume vs. time)
time = np.linspace(1985, 2015, 5)  # Simulated time points
volume_irr = [30, 28, 27, 25, 24]  # Simulated volume data for Irr
volume_noirr = [30, 27, 25, 23, 20]  # Simulated volume data for NoIrr
volume_members1 = [31, 28, 26, 24, 22]  # Individual members
volume_members2 = [29, 26, 24, 22, 18]  # Individual members

# Create a new figure for the small legend plot
fig_legend = fig.add_axes([0.5, -0.18, 0.13, 0.14])  # Small plot size

# Plot the sample data
fig_legend.plot(time, volume_irr, label='Irr (W5E5)',
                color='black', linewidth=2)  # Black line for Irr
fig_legend.plot(time, volume_noirr, label='NoIrr (11-member average)',
                color='blue', linestyle='-', linewidth=2)  # Blue line for NoIrr average
fig_legend.plot(time, volume_members1, label='NoIrr (individual member)', color='grey',
                linestyle='dotted', linewidth=1)  # Dotted grey for individual members
fig_legend.plot(time, volume_members2, label='', color='grey',
                linestyle='dotted', linewidth=1)  # Dotted grey for individual members

# Shade for the range
fig_legend.fill_between(time, volume_members2, volume_members1,
                        color='lightblue', alpha=0.3, label='NoIrr 11-member range')  # Shading for range

# Annotate number of glaciers (just for example)
fig_legend.text(0.05, 0.05, '# of glaciers', transform=fig_legend.transAxes,
                fontsize=12, verticalalignment='bottom', horizontalalignment='left',
                bbox=dict(facecolor='white', edgecolor='none', pad=2))

fig_legend.legend(loc='center right', bbox_to_anchor=(3.5, 0.5), ncol=1, fontsize=12,
                  frameon=False)

# Set labels for axes
fig_legend.set_xlabel('Time', fontsize=12)
fig_legend.set_ylabel('Volume [km$^3$]', fontsize=12)
fig_legend.set_title('RGI subregion', fontsize=12, fontweight='bold')

# Remove tick marks but keep the tick labels
fig_legend.tick_params(axis='both', which='major', length=0)
fig_legend.set_xticklabels([])  # Removes x-axis tick labels
fig_legend.set_yticklabels([])  # Removes y-axis tick labels


fig.tight_layout()
fig_folder = os.path.join(fig_path, "04. Map")
os.makedirs(fig_folder, exist_ok=True)
plt.savefig(f"{fig_folder}/Map_Plot_sA_cB_boxV_NOI.png")
fig.show()


# %% CEll 9: Create a table of total area per region included

# Sample dictionary with colors
region_colors = {
    13: 'blue',    # Blue for region 13
    14: 'crimson',    # Pink for region 14
    15: 'orange'   # Orange for region 15
}

# Group by 'rgi_subregion' and sum the values
grouped_area = ds.groupby('rgi_subregion').sum()

# Extract the first two characters (region number) and convert to integer
regions = grouped_area.index.str[:2].astype(int)

# Get the corresponding color for each region
# Use 'gray' if region not in region_colors
colors_ = [region_colors.get(region, 'gray') for region in regions]

# Plot the bar chart with the assigned colors
plt.bar(grouped_area.index, grouped_area['rgi_area_km2'], color=colors_)

# Add labels and title
plt.xlabel('RGI Subregion')
plt.ylabel('Area (km²)')
plt.title('Total Area by RGI Subregion')
plt.xticks(rotation=90)  # Rotate x-axis labels for readability
plt.show()

# %% Cell 10: Show the area by subregion

# Sample dictionary with colors
region_colors = {
    13: 'blue',    # Blue for region 13
    14: 'crimson',    # Pink for region 14
    15: 'orange'   # Orange for region 15
}


# Group by 'rgi_subregion' and sum the values
grouped_area = ds.groupby('rgi_subregion').sum()

# Number of subregions
subregions = grouped_area.index

# Create a 3x5 subplot grid
fig, axes = plt.subplots(5, 3, figsize=(
    10, 7), constrained_layout=True, sharey=True)

# Flatten axes for easy iteration
axes = axes.flatten()

# Iterate over each subregion and corresponding subplot
for i, subregion in enumerate(subregions):
    ax = axes[i]

    # Extract the region number (first 2 characters of subregion)
    region = int(subregion[:2])

    # Assign color based on the region
    color = region_colors.get(region, 'gray')

    # Get the area distribution for this subregion
    areas = ds[ds['rgi_subregion'] == subregion]  # ['rgi_area_km2']
    sorted_areas = areas.sort_values(by='rgi_area_km2', ascending=False)
    volume = ds[ds['rgi_subregion'] == subregion]  # ['rgi_volume_km3']
    sorted_volume = volume.sort_values(by='rgi_volume_km3', ascending=False)
    # Calculate the cumulative volume
    sorted_volume['cumulative_volume_km3'] = sorted_volume['rgi_volume_km3'].cumsum()

    colors_ = [region_colors.get(region, 'gray') for region in regions]

    # # Plot a histogram of glacier areas for this subregion
    # ax.hist(areas['rgi_area_km2'], bins=len(areas['rgi_area_km2']), color=colors[i], edgecolor='none')

    # Set title for each subplot
    ax.set_title(f'{subregion}', fontweight="bold")

    # Plot glacier area as a line chart
    if i == 6:
        ax.set_ylabel('Area (km²)', color='black')
    if i >= 12:
        ax.set_xlabel('# of glaciers', color='black')
    ax.set_yscale('log')
    # ax.set_ylim(0,125)
    # Plot the glacier area as a bar chart with conditional colors
    bars = ax.bar(range(1, len(sorted_areas['rgi_area_km2']) + 1),
                  sorted_areas['rgi_area_km2'], color=colors_[i], alpha=0.6, width=1)

    axes2 = ax.twinx()
    axes2.set_xlabel('Number of Glaciers')
    # Change label color to black
    if i == 8:
        axes2.set_ylabel('Cumulative Volume (km³)', color='black')
    axes2.plot(range(1, len(sorted_volume) + 1),
               sorted_volume['cumulative_volume_km3'], color='black', label='Cumulative Volume (km³)')
    axes2.tick_params(axis='y', labelcolor='black')
    # axes2.set_ylim(0,2000)

    total_glaciers = len(areas)

    # Place the annotation in the plot
    ax.annotate(total_glaciers, xy=(0.05, 0.8), xycoords='axes fraction',
                fontsize=10, fontstyle='italic')
