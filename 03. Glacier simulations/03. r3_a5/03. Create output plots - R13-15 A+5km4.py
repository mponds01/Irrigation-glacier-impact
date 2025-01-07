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
import mpl_axes_aligner
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
from matplotlib.colors import ListedColormap, BoundaryNorm
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

mb_members_noi = []
mb_members_cf = []

# Initialize plot

fig, ax = plt.subplots(1, 1, figsize=(8, 5), sharex=True, sharey=True)

# Load baseline data
i_path_base = os.path.join(sum_dir, 'specific_massbalance_mean_W5E5.000.csv')
mb_base = pd.read_csv(i_path_base, index_col=0).to_xarray()
bin_size = np.linspace(mb_base.B.min(), mb_base.B.max(),
                       int(len(mb_base.B) / 10))

# Plot baseline once if not using subplots
baseline_plotted = True
plot_gaussian(ax, mb_base, bin_size, "black", "AllForcings",
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

        # provide the path to datafile from the noi runn
        i_path_noi = os.path.join(
            sum_dir, f'specific_massbalance_mean_{sample_id}.csv')
        mb_noi = pd.read_csv(
            i_path_noi, index_col=0).to_xarray()  # open datafile
        mb_members_noi.append(mb_noi.B.values)
        plot_gaussian(ax, mb_noi, bin_size, colors["noirr"][1], label=None,  # sample_id",
                      zorder=members[m] + member + 1, linestyle=linestyle, gaussian_only=gaussian_only)
        # provide the path to datafile from the cf run
        i_path_cf = os.path.join(
            sum_dir, f'specific_massbalance_mean_{sample_id}_counterfactual.csv')
        mb_cf = pd.read_csv(
            i_path_cf, index_col=0).to_xarray()  # open datafile
        mb_members_cf.append(mb_cf.B.values)
        plot_gaussian(ax, mb_cf, bin_size, colors["cf"][1], label=None,  # sample_id",
                      zorder=members[m] + member + 1, linestyle=linestyle, gaussian_only=gaussian_only)

    # Add annotations and labels
    if m == 3:  # only for the last model plot
        # define number of glaciers for annotation
        total_glaciers = len(mb_noi.B.values)
        ax.annotate(f"Total # of glaciers: {total_glaciers}", xy=(
            0.75, 0.98), xycoords='axes fraction', fontsize=10, verticalalignment='top')  # annotate amount of glaciers
        ax.annotate("Time period: 1985-2014", xy=(0.75, 0.92), xycoords='axes fraction',
                    fontsize=10, verticalalignment='top')  # annotate time period


# calculate the 14-member average
mb_member_average_noi = np.mean(mb_members_noi, axis=0)
mb_member_average_cf = np.mean(mb_members_cf, axis=0)
# load dataframe structure in order to plot gaussian
mb_ds_members_noi = mb_noi.copy().to_dataframe()
mb_ds_members_cf = mb_cf.copy().to_dataframe()

mb_ds_members_noi['B'] = mb_member_average_noi  # add the data to the dataframe
plot_gaussian(ax, mb_ds_members_noi, bin_size, colors["noirr"][0], f"NoIrr ({sum(members_averages)}-member average)",
              zorder=members[m] + member + 1, linestyle="--", gaussian_only=gaussian_only)
mb_ds_members_cf['B'] = mb_member_average_cf  # add the data to the dataframe
plot_gaussian(ax, mb_ds_members_cf, bin_size, colors["cf"][0], f"NoForcings ({sum(members_averages)}-member average)",
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


alpha_set = 0.8
regions = [13, 14, 15]
subregions = [9, 3, 3]

n_rows = 5
n_cols = 3
subplots = True

if subplots == True:
    gaussian_only = False  # Flag to use Gaussian fit only

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 10), sharex=True, sharey=False, gridspec_kw={
        'hspace': 0.15, 'wspace': 0.25})  # create a new figure
    axes = axes.flatten()
else:
    gaussian_only = True  # Flag to use Gaussian fit only
    fig, axes = plt.subplots(figsize=(8, 5), sharex=True, sharey=False, gridspec_kw={
        'hspace': 0.15, 'wspace': 0.25})  # create a new figure

plot_index = 0
ds = pd.read_csv(
    f"{wd_path}masters/master_gdirs_r3_a5_rgi_date_A_V_RGIreg_B.csv")
master_ds = ds[(~ds['sample_id'].str.endswith('0')) |
               (ds['sample_id'] == 'IPSL-CM6.000')]
master_ds = master_ds[['rgi_id', 'rgi_region', 'rgi_subregion', 'full_name', 'cenlon', 'cenlat', 'rgi_date',
                       'rgi_area_km2', 'rgi_volume_km3', 'sample_id', 'B_noirr', 'B_irr', 'B_delta_irr', 'B_cf', 'B_delta_cf']]

master_ds_avg = master_ds.groupby(['rgi_id'], as_index=False).agg({  # calculate the 11 member average for noirr and delta, for irr the first value can be taken as they are all the same
    'B_delta_irr': 'mean',
    'B_noirr': 'mean',
    'B_delta_cf': 'mean',
    'B_cf': 'mean',
    # lamda is anonmous functions, returns 11 member average
    'sample_id': lambda _: f"{sum(members_averages)}-member average",
    # take first value for all columns that are not in the list
    **{col: 'first' for col in master_ds.columns if col not in ['B_noirr', 'B_delta', 'sample_id']}
})

# Aggregate dataset, area-weighted BDelta Birr and Bnoirr,
master_ds_area_weighted = master_ds_avg.groupby(['rgi_subregion'], as_index=False).agg({
    'B_delta_irr': lambda x: (x * master_ds.loc[x.index, 'rgi_area_km2']).sum() / master_ds.loc[x.index, 'rgi_area_km2'].sum(),
    'B_irr': lambda x: (x * master_ds.loc[x.index, 'rgi_area_km2']).sum() / master_ds.loc[x.index, 'rgi_area_km2'].sum(),
    'B_delta_cf': lambda x: (x * master_ds.loc[x.index, 'rgi_area_km2']).sum() / master_ds.loc[x.index, 'rgi_area_km2'].sum(),
    'B_cf': lambda x: (x * master_ds.loc[x.index, 'rgi_area_km2']).sum() / master_ds.loc[x.index, 'rgi_area_km2'].sum(),
    'B_noirr': lambda x: (x * master_ds.loc[x.index, 'rgi_area_km2']).sum() / master_ds.loc[x.index, 'rgi_area_km2'].sum(),
    'rgi_area_km2': 'sum',  # Sum for area
    'rgi_volume_km3': 'sum',  # Sum for volume
    # 'sample_id': lambda _: "11 member average",
    # take first value for all columns that are not in the list
    **{col: 'first' for col in master_ds.columns if col not in ['B_noirr', 'B_irr', 'B_delta_cf', 'B_cf', 'B_delta_cf', 'sample_id', 'rgi_area_km2', 'rgi_volume_km3']}
})


cumulative_index = 0
for r, region in enumerate(regions):
    for sub in range(subregions[r]):
        region_id = f"{region}.0{sub+1}"
        print(region_id)
        subregion_ds = master_ds_avg[master_ds_avg['rgi_subregion'].str.contains(
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
        plot_gaussian(ax, mb_base, bin_size, "black", "AllForcings",
                      zorder=1, gaussian_only=False)

        linestyle = '-'
        filtered_mb_noi = subregion_ds[['rgi_id', 'rgi_area_km2', 'B_noirr']].rename(columns={
            'B_noirr': 'B'})
        filtered_mb_cf = subregion_ds[['rgi_id', 'rgi_area_km2', 'B_cf']].rename(columns={
                                                                                 'B_cf': 'B'})

        # Plot data by subregion
        plot_gaussian(ax, filtered_mb_noi, bin_size, colors['noirr'][1], f"NoIrr ({sum(members_averages)}-member average)",  # colors[model][member]
                      zorder=members[m] + member + 1, linestyle=linestyle, gaussian_only=gaussian_only)
        plot_gaussian(ax, filtered_mb_cf, bin_size, colors['cf'][1], f"NoForcings ({sum(members_averages)}-member average)",  # colors[model][member]
                      zorder=members[m] + member + 1, linestyle=linestyle, gaussian_only=gaussian_only)

        area_weighted_noirr = master_ds_area_weighted['B_noirr'][cumulative_index]
        area_weighted_cf = master_ds_area_weighted['B_cf'][cumulative_index]
        area_weighted_irr = master_ds_area_weighted['B_irr'][cumulative_index]

        # add vertical line at x=0
        ax.axvline(0, color='k', linestyle="dashed", lw=1, zorder=20)

        # Add annotations and labels
        if subplots == True:
            total_glaciers = len(filtered_mb_noi.B.values)
            ax.annotate(f"{region_id}", xy=(
                0.02, 0.95), xycoords='axes fraction', fontsize=10, verticalalignment='top', fontweight='bold')  # annotate amount of glaciers
            ax.annotate(f"{total_glaciers}", xy=(
                0.95, 0.95), xycoords='axes fraction', fontsize=10, verticalalignment='top', horizontalalignment='right', fontstyle='normal')  # annotate amount of glaciers
            ax.annotate(f"$\\overline{{B}}_{{\\text{{noirr}}}}$\n {round(area_weighted_noirr,1)}", xy=(
                0.02, 0.75), xycoords='axes fraction', fontsize=10, verticalalignment='top', fontstyle='normal', color=colors['noirr'][0])  # annotate amount of glaciers
            ax.annotate(f"$\\overline{{B}}_{{\\text{{noforcings}}}}$\n {round(area_weighted_cf,1)}", xy=(
                0.02, 0.35), xycoords='axes fraction', fontsize=10, verticalalignment='top', fontstyle='normal', color=colors['cf'][0])  # annotate amount of glaciers
            ax.annotate(f"$\\overline{{B}}_{{\\text{{irr}}}}$ \n {round(area_weighted_irr,1)}", xy=(
                0.95, 0.75), xycoords='axes fraction', fontsize=10, verticalalignment='top', horizontalalignment='right', fontstyle='normal', color=colors['irr'][0])  # annotate amount of glaciers
        elif subplots == False:
            total_glaciers = len(ds.B_irr.values)
            ax.annotate(f"Region 13,14,15 A > 5 km$^2$", xy=(
                0.02, 0.95), xycoords='axes fraction', fontsize=10, verticalalignment='top', fontweight='bold')  # annotate amount of glaciers
            ax.annotate(f"{total_glaciers}", xy=(
                0.85, 0.95), xycoords='axes fraction', fontsize=10, verticalalignment='top', fontstyle='normal')  # annotate amount of glaciers

        plot_index += 1
        cumulative_index += 1

# format the plot
# if subplots==True:
fig.text(0.32, 0.04,
         "Mean specific mass balance [mm w.e. yr$^{-1}$]", va='center', rotation='horizontal', fontsize=12)
fig.text(0.04, 0.5, "Area [km$^{2}$]",
         va='center', rotation='vertical', fontsize=12)

# Create a legend for the regions and distinguish irr/noirr
region_legend_patches = [
    mpatches.Patch(
        color=colors['noirr'][0], label=f"NoIrr ({sum(members_averages)}-member average)"),
    mpatches.Patch(
        color=colors['cf'][0], label=f"NoForcings ({sum(members_averages)}-member average)"),
    mpatches.Patch(color='black', label='AllForcings (W5E5.000)'),

    # Create custom lines with specific line styles
    mlines.Line2D([], [], color='grey', linestyle='-',
                  label=f'NoIrr - {sum(members_averages)}-member average'),
    mlines.Line2D([], [], color='grey', linestyle=':',
                  label='NoIrr - Individual member'),
    mlines.Line2D([], [], color='black', linestyle='-',
                  label='AllForcings (W5E5.000)'),

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

    ax.text(median_value + 100, position, f'{median_value:.1f}',  # Place above for Irr
            va='center', ha='center', fontsize=10, color='black', fontweight='bold',
            # bbox=dict(boxstyle='round,pad=0.001',
            #           facecolor='white', edgecolor='none'),
            zorder=3)

    return bp


# Initialize plot
fig, ax = plt.subplots(figsize=(10, 10))

df = pd.read_csv(
    f"{wd_path}masters/master_gdirs_r3_a5_rgi_date_A_V_RGIreg_B.csv")
# df = pd.read_csv(
#     f"{wd_path}masters/master_gdirs_r3_a5_rgi_date_A_V_RGIreg_B_hugo.csv")
master_ds = df[(~df['sample_id'].str.endswith('0')) |  # Exclude all the model averages ending with 0 except for IPSL
               (df['sample_id'].str.startswith('IPSL'))]

# Normalize values
# divide all the B values with 1000 to transform to m w.e. average over 30 yrs
# master_ds[['B_noirr', 'B_irr', 'B_delta']] /= 1000

master_ds = master_ds[['rgi_id', 'rgi_region', 'rgi_subregion', 'full_name', 'cenlon', 'cenlat', 'rgi_date',
                       'rgi_area_km2', 'rgi_volume_km3', 'sample_id', 'B_noirr', 'B_irr', 'B_delta_irr', 'B_cf', 'B_delta_cf']]

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
master_ds_avg = master_ds.groupby(['rgi_id'], as_index=False).agg({  # calculate the 11 member average for noirr and delta, for irr the first value can be taken as they are all the same
    'B_delta_irr': 'mean',
    'B_delta_cf': 'mean',
    'B_noirr': 'mean',
    'B_cf': 'mean',
    # lamda is anonmous functions, returns 11 member average
    'sample_id': lambda _: "11 member average",
    # take first value for all columns that are not in the list
    **{col: 'first' for col in master_ds.columns if col not in ['B_noirr', 'B_delta', 'sample_id']}
})


# Aggregate dataset, area-weighted BDelta Birr and Bnoirr,
master_ds_area_weighted = master_ds_avg.groupby(['rgi_subregion'], as_index=False).agg({
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

use_weights = True
# Define regions and subregions to loop through
regions = [13, 14, 15]
subregions = [9, 3, 3]  # Subregions count for each region
region_colors = {13: 'blue', 14: 'crimson', 15: 'orange'}
v_space_noi = 0.8  # Vertical space between irr and noirr boxplots
v_space_irr = 1.6  # Vertical space between irr and noirr boxplots

# Storage for combined data
all_noirr, all_irr = [], []
position_counter = 1


cumulative_index = 14
# Example usage: Main loop through regions and subregions
for r, region in enumerate(reversed(regions)):
    for sub in reversed(range(list(reversed(subregions))[r])):
        # Filter subregion-specific data

        subregion_ds = master_ds_avg[master_ds_avg['rgi_subregion'].str.contains(
            f"{region}.0{sub+1}")]

        # Calculate mean values for Noirr and Irr
        noirr_mean = master_ds_area_weighted['B_noirr'][cumulative_index]
        irr_mean = master_ds_area_weighted['B_irr'][cumulative_index]

        # Set color and label based on the region
        # color = region_colors[region]
        label = [f"{region}-0{sub+1}"]

        # Plot Noirr and Irr boxplots
        box_cf = plot_boxplot(
            subregion_ds['B_cf'], position_counter, "", color=colors['cf'][0], alpha=0.5, is_noirr=False)
        box_noirr = plot_boxplot(
            subregion_ds['B_noirr'], position_counter + v_space_noi, label, color=colors['noirr'][0], alpha=0.5, is_noirr=True)
        box_irr = plot_boxplot(
            subregion_ds['B_irr'], position_counter + v_space_irr, label="", color=colors['irr'][0], alpha=0.5, is_noirr=False)

        # Annotate the number of glaciers and delta between the two columns
        num_glaciers = len(subregion_ds)
        delta = noirr_mean - irr_mean

        # Display number of glaciers and delta
        plt.text(850, position_counter + (v_space_noi / 2),
                 f'{num_glaciers} ',  # \nΔ = {delta:.2f}',
                 va='center', ha='center', fontsize=10, color='black',
                 backgroundcolor="white", fontstyle="italic", zorder=1)

        # Increment position counter for the next subregion
        position_counter += 4
        cumulative_index -= 1

overall_area_weighted_mean = {
    'B_delta_irr': (master_ds_avg['B_delta_irr'] * master_ds_avg['rgi_area_km2']).sum() / master_ds_avg['rgi_area_km2'].sum(),
    'B_irr': (master_ds_avg['B_irr'] * master_ds_avg['rgi_area_km2']).sum() / master_ds_avg['rgi_area_km2'].sum(),
    'B_noirr': (master_ds_avg['B_noirr'] * master_ds_avg['rgi_area_km2']).sum() / master_ds_avg['rgi_area_km2'].sum(),
    'B_delta_cf': (master_ds_avg['B_delta_cf'] * master_ds_avg['rgi_area_km2']).sum() / master_ds_avg['rgi_area_km2'].sum(),
    'B_cf': (master_ds_avg['B_irr'] * master_ds_avg['rgi_area_km2']).sum() / master_ds_avg['rgi_area_km2'].sum(),
    'total_area_km2': master_ds_avg['rgi_area_km2'].sum(),  # Total area sum
    # Total volume sum
    'total_volume_km3': master_ds_avg['rgi_volume_km3'].sum()
}

# Plot overall average boxplots for Irr and Noirr
avg_cf = plot_boxplot(master_ds_avg['B_cf'], position_counter, [
    "Average"], color=colors['cf'][1], alpha=1, is_noirr=False)
avg_noi = plot_boxplot(master_ds_avg['B_noirr'], position_counter + v_space_noi, [
    "Average"], color=colors['noirr'][1], alpha=1, is_noirr=True)
avg_irr = plot_boxplot(master_ds_avg['B_irr'],  position_counter +
                       v_space_irr, "", color=colors['irr'][1], alpha=1, is_noirr=False)


# Annotate the number of glaciers for the overall average
plt.text(850, position_counter + (v_space_noi / 2), f'{len(master_ds_avg)}',  # Display total number of glaciers
         va='center', ha='center', fontsize=10, color='black',
         fontstyle='italic', backgroundcolor="white", zorder=1)
plt.text(850, position_counter + 2, "# of glaciers",  # Plot number of glaciers title
         va='center', ha='center', fontsize=10, color='black', fontstyle='italic', backgroundcolor="white", zorder=1)

# Create custom legend elements for mean (dot) and median (stripe)
mean_dot = Line2D([0], [0], marker='o', color='w',
                  label='Area-weighted Mean (dot)', markerfacecolor='black', markersize=10)
median_stripe = Line2D([0], [0], color='black', lw=2, label='Median (stripe)')

# Add a legend for regions, mean (dot), and median (stripe)
region_legend_patches = [mpatches.Patch(color=colors['irr'][1], label='AllForcings'),
                         mpatches.Patch(
                             color=colors['noirr'][1], label='NoIrr'),
                         mpatches.Patch(
                             color=colors['cf'][1], label='NoForcings'),
                         ]

# Append the custom legend items for the mean dot and median stripe
region_legend_patches += [median_stripe]

# Create the legend with the updated patches
fig.legend(handles=region_legend_patches, loc='center right',
           bbox_to_anchor=(1.25, 0.5), ncols=1)
# ax.legend(handles=region_legend_patches, loc='lower center',
#           bbox_to_anchor=(0.5, -0.15), ncols=3)

ax.axvline(x=0, color='black', linestyle='--', linewidth=1, zorder=0)
ax.set_xlabel('Mean B_noirr and B_irr')
ax.set_ylabel('Regions and Subregions')
ax.set_xlim(-1350, 1000)

# Extend the ylims for more padding
y_min, y_max = ax.get_ylim()  # Get current y-axis limits
padding = 1  # Adjust this value as needed
ax.set_ylim(y_min - padding, y_max + padding)

# Adjust layout and display the plot
plt.tight_layout()
fig_folder = os.path.join(fig_path, "03. Mass Balance", "Boxplot")
os.makedirs(fig_folder, exist_ok=True)
plt.savefig(
    f"{fig_folder}/Gaussian_distribution_total_region_by_region_median.png")
plt.show()


# %% Cell 4b: Plot the specific mass balances using boxplots, different subsets, weighted by region - print area weighted mean
# Initialize plot
fig, ax = plt.subplots(figsize=(10, 10))

df = pd.read_csv(
    f"{wd_path}masters/master_gdirs_r3_a5_rgi_date_A_V_RGIreg_B.csv")
# df = pd.read_csv(
#     f"{wd_path}masters/master_gdirs_r3_a5_rgi_date_A_V_RGIreg_B_hugo.csv")
master_ds = df[(~df['sample_id'].str.endswith('0')) |  # Exclude all the model averages ending with 0 except for IPSL
               (df['sample_id'].str.startswith('IPSL'))]

# Normalize values
# divide all the B values with 1000 to transform to m w.e. average over 30 yrs
# master_ds[['B_noirr', 'B_irr', 'B_delta']] /= 1000

master_ds = master_ds[['rgi_id', 'rgi_region', 'rgi_subregion', 'full_name', 'cenlon', 'cenlat', 'rgi_date',
                       'rgi_area_km2', 'rgi_volume_km3', 'sample_id', 'B_noirr', 'B_irr', 'B_delta_irr', 'B_cf', 'B_delta_cf']]

master_ds_avg = master_ds.groupby(['rgi_id'], as_index=False).agg({  # calculate the 11 member average for noirr and delta, for irr the first value can be taken as they are all the same
    'B_delta_irr': 'mean',
    'B_noirr': 'mean',
    # lamda is anonmous functions, returns 11 member average
    'sample_id': lambda _: "11 member average",
    # take first value for all columns that are not in the list
    **{col: 'first' for col in master_ds.columns if col not in ['B_noirr', 'B_delta', 'sample_id']}
})


# Aggregate dataset, area-weighted BDelta Birr and Bnoirr,
master_ds_area_weighted = master_ds_avg.groupby(['rgi_subregion'], as_index=False).agg({
    'B_delta_irr': lambda x: (x * master_ds.loc[x.index, 'rgi_area_km2']).sum() / master_ds.loc[x.index, 'rgi_area_km2'].sum(),
    'B_irr': lambda x: (x * master_ds.loc[x.index, 'rgi_area_km2']).sum() / master_ds.loc[x.index, 'rgi_area_km2'].sum(),
    'B_delta_cf': lambda x: (x * master_ds.loc[x.index, 'rgi_area_km2']).sum() / master_ds.loc[x.index, 'rgi_area_km2'].sum(),
    'B_cf': lambda x: (x * master_ds.loc[x.index, 'rgi_area_km2']).sum() / master_ds.loc[x.index, 'rgi_area_km2'].sum(),
    'B_noirr': lambda x: (x * master_ds.loc[x.index, 'rgi_area_km2']).sum() / master_ds.loc[x.index, 'rgi_area_km2'].sum(),
    'rgi_area_km2': 'sum',  # Sum for area
    'rgi_volume_km3': 'sum',  # Sum for volume
    # 'sample_id': lambda _: "11 member average",
    # take first value for all columns that are not in the list
    **{col: 'first' for col in master_ds.columns if col not in ['B_noirr', 'B_irr', 'B_delta_irr', 'B_cf', 'B_delta_cf', 'sample_id', 'rgi_area_km2', 'rgi_volume_km3']}
})

use_weights = True
# Define regions and subregions to loop through
regions = [13, 14, 15]
subregions = [9, 3, 3]  # Subregions count for each region
region_colors = {13: 'blue', 14: 'crimson', 15: 'orange'}
v_space_noirr = 0.75  # Vertical space between irr and noirr boxplots
v_space_irr = 1.5  # Vertical space between irr and noirr boxplots

# Storage for combined data
all_noirr, all_irr = [], []
position_counter = 1


def plot_boxplot_awm(data, mean_value, position, label, color, alpha, is_noirr):
    # Create the boxplot
    box = ax.boxplot(data, patch_artist=True, labels=label if is_noirr else [""],  # Only set labels for Noirr
                     vert=False, widths=0.7,
                     boxprops=dict(facecolor=color, alpha=alpha,
                                   edgecolor='none'),
                     medianprops=dict(color='black'),
                     positions=[position], showfliers=False, zorder=2)

    # Plot black dot at the mean position
    # 'ko' for black dot ('k' = black, 'o' = dot)
    ax.plot(mean_value, position, 'ko', zorder=3)

    # Annotate text: above the boxplot for Irr, below for Noirr

    plt.text(mean_value+100, position, f'{mean_value:.1f}',  # Place above for Irr
             va='center', ha='center', fontsize=10, color='black', fontweight="bold",
             # bbox=dict(boxstyle='round,pad=0.001',
             #           facecolor='white', edgecolor='none'),
             zorder=3)

    return box


cumulative_index = 14
# Example usage: Main loop through regions and subregions
for r, region in enumerate(reversed(regions)):
    for sub in reversed(range(list(reversed(subregions))[r])):
        # Filter subregion-specific data
        print(region)
        print(sub)
        # print(sub+subregions[r]-1)

        subregion_ds = master_ds_avg[master_ds_avg['rgi_subregion'].str.contains(
            f"{region}.0{sub+1}")]

        # Calculate mean values for Noirr and Irr
        noirr_mean = master_ds_area_weighted['B_noirr'][cumulative_index]
        irr_mean = master_ds_area_weighted['B_irr'][cumulative_index]
        cf_mean = master_ds_area_weighted['B_cf'][cumulative_index]

        # Set color and label based on the region
        color = region_colors[region]
        label = [f"{region}-0{sub+1}"]

        # Plot Noirr and Irr boxplots
        box_cf = plot_boxplot_awm(
            subregion_ds['B_cf'].values, cf_mean, position_counter, label, color=colors["cf"][1], alpha=1, is_noirr=False)
        box_noirr = plot_boxplot_awm(
            subregion_ds['B_noirr'].values, noirr_mean, position_counter + v_space_noirr, label, color=colors["noirr"][1], alpha=1, is_noirr=True)
        box_irr = plot_boxplot_awm(
            subregion_ds['B_irr'], irr_mean, position_counter + v_space_irr, "", color=colors["irr"][1], alpha=1.0, is_noirr=False)

        # Annotate the number of glaciers and delta between the two columns
        num_glaciers = len(subregion_ds)
        delta = noirr_mean - irr_mean

        # Display number of glaciers and delta
        plt.text(850, position_counter + (v_space_noirr / 2),
                 f'{num_glaciers} ',  # \nΔ = {delta:.2f}',
                 va='center', ha='center', fontsize=10, color='black',
                 backgroundcolor="white", fontstyle="italic", zorder=1)

        # Increment position counter for the next subregion
        position_counter += 3
        cumulative_index -= 1

overall_area_weighted_mean = {
    'B_delta_irr': (master_ds_avg['B_delta_irr'] * master_ds_avg['rgi_area_km2']).sum() / master_ds_avg['rgi_area_km2'].sum(),
    'B_irr': (master_ds_avg['B_irr'] * master_ds_avg['rgi_area_km2']).sum() / master_ds_avg['rgi_area_km2'].sum(),
    'B_noirr': (master_ds_avg['B_noirr'] * master_ds_avg['rgi_area_km2']).sum() / master_ds_avg['rgi_area_km2'].sum(),
    'B_delta_cf': (master_ds_avg['B_delta_cf'] * master_ds_avg['rgi_area_km2']).sum() / master_ds_avg['rgi_area_km2'].sum(),
    'B_cf': (master_ds_avg['B_cf'] * master_ds_avg['rgi_area_km2']).sum() / master_ds_avg['rgi_area_km2'].sum(),
    'total_area_km2': master_ds_avg['rgi_area_km2'].sum(),  # Total area sum
    # Total volume sum
    'total_volume_km3': master_ds_avg['rgi_volume_km3'].sum()
}

# Plot overall average boxplots for Irr and Noirr

avg_cf = plot_boxplot_awm(master_ds_avg['B_cf'], overall_area_weighted_mean['B_cf'], position_counter +
                          v_space_noirr, ["Average"], color=colors['cf'][1], alpha=1, is_noirr=True)
avg_noirr = plot_boxplot_awm(master_ds_avg['B_noirr'], overall_area_weighted_mean['B_noirr'], position_counter, [
    ""], color=colors['noirr'][1], alpha=1, is_noirr=False)
avg_irr = plot_boxplot_awm(master_ds_avg['B_irr'], overall_area_weighted_mean['B_irr'], position_counter +
                           v_space_irr, "", color=colors['irr'][1], alpha=1.0, is_noirr=False)


# Annotate the number of glaciers for the overall average
plt.text(850, position_counter + (v_space_noirr / 2), f'{len(master_ds_avg)}',  # Display total number of glaciers
         va='center', ha='center', fontsize=10, color='black',
         fontstyle='italic', backgroundcolor="white", zorder=1)

plt.text(850, position_counter + 2, "# of glaciers",  # Plot number of glaciers title
         va='center', ha='center', fontsize=10, color='black', fontstyle='italic', backgroundcolor="white", zorder=1)

# Create custom legend elements for mean (dot) and median (stripe)
mean_dot = Line2D([0], [0], marker='o', color='w',
                  label='Area-weighted Mean (dot)', markerfacecolor='black', markersize=10)
median_stripe = Line2D([0], [0], color='black', lw=2, label='Median (stripe)')

# Add a legend for regions, mean (dot), and median (stripe)
region_legend_patches = [mpatches.Patch(color=colors['irr'][1], label='AllForcings'),
                         mpatches.Patch(
                             color=colors['noirr'][1], label='NoIrr'),
                         mpatches.Patch(
                             color=colors['cf'][1], label='NoForcings'),
                         ]

# Append the custom legend items for the mean dot and median stripe
region_legend_patches += [median_stripe]

# Create the legend with the updated patches
fig.legend(handles=region_legend_patches, loc='center right',
           bbox_to_anchor=(1.25, 0.5), ncols=1)
# ax.legend(handles=region_legend_patches, loc='lower center',
#           bbox_to_anchor=(0.5, -0.15), ncols=3)

ax.axvline(x=0, color='black', linestyle='--', linewidth=1, zorder=0)
ax.set_xlabel('Mean B [mm yr$^{-1}$]')
ax.set_ylabel('Regions and Subregions')
ax.set_xlim(-1350, 1000)

# Extend the ylims for more padding
y_min, y_max = ax.get_ylim()  # Get current y-axis limits
padding = 1  # Adjust this value as needed
ax.set_ylim(y_min - padding, y_max + padding)

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
    fig, ax = plt.subplots(figsize=(7, 4))  # create a new figure

    # create a timeseries for all the model members to add the data, needed to calculate averages later
    member_data = []
    member_data_cf = []

    # load and plot the baseline data
    baseline_path = os.path.join(
        wd_path, "summary", f"climate_run_output_baseline_W5E5.000.nc")
    baseline = xr.open_dataset(baseline_path)
    ax.plot(baseline["time"], baseline[var].sum(dim="rgi_id") * factors[v],
            label="AllForcings", color=colors["irr"][0], linewidth=2, zorder=15)
    mean_values_irr = (baseline[var].sum(
        dim="rgi_id") * factors[v]).values

    # loop through all the different model x member combinations
    for m, model in enumerate(models_shortlist):
        for i in range(members_averages[m]):

            # make sure the counter for sample ids starts with 001, 000 are averages of all members by model
            # IPSL-CM6 only has 1 member, so the sample_id must end with 000
            if members_averages[m] > 1:
                i += 1
                label = None
            else:
                label = "GCM member"

            sample_id = f"{model}.00{i}"

            # load and plot the data from the climate output run and counterfactual
            climate_run_opath = os.path.join(
                sum_dir, f'climate_run_output_perturbed_{sample_id}.nc')
            climate_run_opath_cf = os.path.join(
                sum_dir, f'climate_run_output_perturbed_{sample_id}_counterfactual.nc')
            climate_run_output = xr.open_dataset(climate_run_opath)
            climate_run_output_cf = xr.open_dataset(climate_run_opath_cf)
            ax.plot(climate_run_output["time"], climate_run_output[var].sum(dim="rgi_id") * factors[v],
                    label=label, color=colors["noirr"][0], linewidth=2, linestyle="dotted")
            ax.plot(climate_run_output_cf["time"], climate_run_output_cf[var].sum(dim="rgi_id") * factors[v],
                    label=None, color=colors["cf"][0], linewidth=2, linestyle="dotted")

            # add all the summed volumes/areas to the member list, so a multi-member average can be calculated
            member_data.append(climate_run_output[var].sum(
                dim="rgi_id").values * factors[v])

            member_data_cf.append(climate_run_output_cf[var].sum(
                dim="rgi_id").values * factors[v])
    # stack the member data
    stacked_member_data = np.stack(member_data)
    stacked_member_data_cf = np.stack(member_data_cf)

    # calculate and plot volume/area 10-member mean
    mean_values_noirr = np.median(stacked_member_data, axis=0).flatten()
    mean_values_cf = np.median(stacked_member_data_cf, axis=0).flatten()
    # mean_values_noirr = np.mean(stacked_member_data, axis=0).flatten()
    # mean_values_cf = np.mean(stacked_member_data_cf, axis=0).flatten()
    ax.plot(climate_run_output["time"].values, mean_values_noirr,
            color=colors["noirr"][0], linestyle='solid', lw=2, label=f"NoIrr ({sum(members_averages)}-member avg)")
    ax.plot(climate_run_output_cf["time"].values, mean_values_cf,
            color=colors["cf"][0], linestyle='solid', lw=2, label="NoForcings ({sum(members_averages)}-member avg)")

    # calculate the volume loss by scenario: irr - noirr and counterfactual
    volume_loss_percentage_noirr = (
        (mean_values_noirr - mean_values_irr[0]) / mean_values_irr[0]) * 100
    volume_loss_percentage_cf = (
        (mean_values_cf - mean_values_irr[0]) / mean_values_irr[0]) * 100
    volume_loss_percentage_irr = (
        (mean_values_irr - mean_values_irr[0]) / mean_values_irr[0]) * 100
    # #create a dataframe with the volume loss percentages and absolute values
    # loss_df_subregion = pd.DataFrame({
    #     'time': climate_run_output["time"].values,
    #     'subregion': np.repeat(region_id, len(climate_run_output["time"])),
    #     'volume_irr': mean_values_irr,
    #     'volume_noirr': mean_values_noirr,
    #     'volume_cf': mean_values_cf,
    #     'volume_loss_percentage_irr': volume_loss_percentage_irr,
    #     'volume_loss_percentage_noirr': volume_loss_percentage_noirr,
    #     'volume_loss_percentage_cf': volume_loss_percentage_cf
    # })

    ax2 = ax.twinx()
    # create a bar chart to show the volume loss by dataset
    ax2.bar(climate_run_output["time"].values[-1]+2, volume_loss_percentage_noirr[-1],
            color=colors['noirr'][0], label="Volume Loss NoIrr(%)", alpha=0.6, zorder=0)

    ax2.bar(climate_run_output["time"].values[-1]+2, volume_loss_percentage_irr[-1],
            color=colors['irr'][0],  label="Volume Loss Irr (%)", alpha=0.6, zorder=2)

    ax2.bar(climate_run_output["time"].values[-1]+2, volume_loss_percentage_cf[-1],
            color=colors['cf'][0],  label="Volume Loss NoForcings (%)", alpha=0.6, zorder=1)

    ax2.axhline(0, color='black', linestyle='--',
                linewidth=1.5, zorder=1)  # Dashed line at 0

    # Step 1: Set the value on ax1 where we want 0 on ax2 to align
    # e.g., the first value of data1
    align_value = (baseline[var].sum(dim="rgi_id") * factors[v])[0].values

    # Step 2: Calculate the offset required for ax2 limits
    mpl_axes_aligner.align.yaxes(ax, align_value, ax2, 0)

    # calculate and plot volume/area 10-member min and max for ribbon
    min_values = np.min(stacked_member_data, axis=0).flatten()
    max_values = np.max(stacked_member_data, axis=0).flatten()
    ax.fill_between(climate_run_output["time"].values, min_values, max_values,
                    color=colors["noirr"][1], alpha=0.3, label=f"NoIrr ({sum(members_averages)}-member range)", zorder=16)

    min_values_cf = np.min(stacked_member_data_cf, axis=0).flatten()
    max_values_cf = np.max(stacked_member_data_cf, axis=0).flatten()
    ax.fill_between(climate_run_output_cf["time"].values, min_values_cf, max_values_cf,
                    color=colors["cf"][1], alpha=0.3, label=f"NoForcings ({sum(members_averages)}-member range)", zorder=16)

    # Set labels and title for the combined plot
    ax.set_ylabel(variable_axes[v])
    ax2.set_ylabel(f"{variable_names[v]} [% of 1985-Irr]")
    ax.set_xlabel("Time [year]")
    ax.set_title(f"Summed {variable_names[v]}, RGI 13-15, A >5 km$^2$")

    # Adjust the legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
               bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.tight_layout()

    # specify and create a folder for saving the data (if it doesn't exists already) and save the plot
    o_folder_data = f"/Users/magaliponds/OneDrive - Vrije Universiteit Brussel/1. VUB/02. Coding/01. IRRMIP/04. Figures/02. OGGM simulations/01. Modelled output/3r_a5/0{v + 1}. {variable_names[v]}/00. Combined"
    os.makedirs(o_folder_data, exist_ok=True)
    o_file_name = f"{o_folder_data}/1985_2014.{timeframe}.delta.{variable_names[v]}.combined.png"
    # plt.savefig(o_file_name, bbox_inches='tight')


# %% Check climate run otuput for area - why not same starting value for cf and normal

# Select the first timestep for 'area' in each dataset
area_first_cf = climate_run_output_cf['area'].isel(time=0)
area_first = climate_run_output['area'].isel(time=0)

differences_first_timestep = area_first_cf != area_first
differing_rgi_ids = climate_run_output_cf['rgi_id'].where(
    differences_first_timestep, drop=True)
print(differing_rgi_ids)
print(area_first_cf.sel(rgi_id=differing_rgi_ids).values)

print(area_first.sel(rgi_id=differing_rgi_ids).values*10**-6)


# %% Cell 6b: Create output plots for Area and volume - per region

ds = pd.read_csv(
    f"{wd_path}masters/master_gdirs_r3_a5_rgi_date_A_V_RGIreg_B_hugo.csv")

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
    'B_delta_irr': 'mean',
    'B_cf': 'mean',
    'B_delta_cf': 'mean',
    'sample_id': 'first'
}

# Step 1: Group by 'rgi_id', apply custom aggregation
grouped_ds = ds.groupby('rgi_id').agg(aggregation_functions).reset_index()

# Step 2: Replace the 'sample_id' column with "11-member average"
grouped_ds['sample_id'] = f'{sum(members_averages)}-member average'

# Step 3: Keep only the required columns
grouped_ds = grouped_ds[['rgi_id', 'rgi_region', 'rgi_subregion', 'full_name', 'cenlon', 'cenlat', 'rgi_date',
                         'rgi_area_km2', 'rgi_volume_km3', 'sample_id', 'B_noirr', 'B_irr', 'B_delta_irr', 'B_cf', 'B_delta_cf']]

# Overwrite the original dataset with the grouped one
ds = grouped_ds

# define the variables for p;lotting
variables = ["volume"]  # , "area"]
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


volume_bars = True

# repeat the loop for area and volume
for v, var in enumerate(variables):
    print(var)
    # create a new plot for both area and volume
    n_rows = 5
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12), sharex=True, sharey=False, gridspec_kw={
                             'hspace': 0.15, 'wspace': 0.35})  # create a new figure
    plot_index = 0
    axes = axes.flatten()
    all_loss_dfs = []
    all_member_data = []
    member_data_df_noi = pd.DataFrame(
        columns=[f'member_{i+1}' for i in range(sum(members_averages))])
    member_data_df_noi.index.name = 'subregion'
    member_data_df_cf = pd.DataFrame(
        columns=[f'member_{i+1}' for i in range(sum(members_averages))])
    member_data_df_cf.index.name = 'subregion'

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
            filtered_member_data_cf = []
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
                    label="AllForcings", color=colors["irr"][0], linewidth=2, zorder=15)
            mean_values_irr = (filtered_baseline[var].sum(
                dim="rgi_id") * factors[v]).values
            ax2 = ax.twinx()

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
                    climate_run_opath_cf = os.path.join(
                        sum_dir, f'climate_run_output_perturbed_{sample_id}_counterfactual.nc')
                    climate_run_output = xr.open_dataset(climate_run_opath)
                    climate_run_output_cf = xr.open_dataset(
                        climate_run_opath_cf)
                    filtered_climate_run_output = climate_run_output.where(
                        climate_run_output['rgi_id'].isin(subregion_ds.rgi_id.values), drop=True)
                    filtered_climate_run_output_cf = climate_run_output_cf.where(
                        climate_run_output_cf['rgi_id'].isin(subregion_ds.rgi_id.values), drop=True)
                    if i == 1 and m == 1:
                        label = "GCM member"
                    else:
                        label = "_nolegend_"
                    ax.plot(filtered_climate_run_output["time"], filtered_climate_run_output[var].sum(dim="rgi_id") * factors[v],
                            label=label, color=colors["noirr"][0], linewidth=2, linestyle="dotted", zorder=3)
                    ax.plot(filtered_climate_run_output_cf["time"], filtered_climate_run_output_cf[var].sum(dim="rgi_id") * factors[v],
                            label=None, color=colors["cf"][0], linewidth=2, linestyle="dotted", zorder=3)

                    # add all the summed volumes/areas to the member list, so a multi-member average can be calculated
                    filtered_member_data.append(filtered_climate_run_output[var].sum(
                        dim="rgi_id").values * factors[v])
                    filtered_member_data_cf.append(filtered_climate_run_output_cf[var].sum(
                        dim="rgi_id").values * factors[v])

            # stack the member data
            stacked_member_data = np.stack(filtered_member_data)
            stacked_member_data_cf = np.stack(filtered_member_data_cf)
            # calculate and plot volume/area 10-member mean
            # mean_values_noirr = np.mean(stacked_member_data, axis=0).flatten()
            # mean_values_cf = np.mean(stacked_member_data_cf, axis=0).flatten()
            mean_values_noirr = np.median(
                stacked_member_data, axis=0).flatten()
            mean_values_cf = np.median(
                stacked_member_data_cf, axis=0).flatten()

            # calculate the volume loss by scenario: irr - noirr and counterfactual
            volume_loss_percentage_noirr = (
                (mean_values_noirr - mean_values_irr[0]) / mean_values_irr[0]) * 100
            volume_loss_percentage_cf = (
                (mean_values_cf - mean_values_irr[0]) / mean_values_irr[0]) * 100
            volume_loss_percentage_irr = (
                (mean_values_irr - mean_values_irr[0]) / mean_values_irr[0]) * 100
            # create a dataframe with the volume loss percentages and absolute values
            loss_df_subregion = pd.DataFrame({
                'time': climate_run_output["time"].values,
                'subregion': np.repeat(region_id, len(climate_run_output["time"])),
                'volume_irr': mean_values_irr,
                'volume_noirr': mean_values_noirr,
                'volume_cf': mean_values_cf,
                'volume_loss_percentage_irr': volume_loss_percentage_irr,
                'volume_loss_percentage_noirr': volume_loss_percentage_noirr,
                'volume_loss_percentage_cf': volume_loss_percentage_cf
            })

            uncertainty_data_noi = (stacked_member_data[:, -1]).reshape(1, -1)
            uncertainty_data_cf = (
                stacked_member_data_cf[:, -1]).reshape(1, -1)
            # create a dataframe with the member data of 2014, to use for uncertainty analysis
            row_noi = pd.DataFrame(
                uncertainty_data_noi, columns=member_data_df_noi.columns, index=[region_id])
            row_cf = pd.DataFrame(
                uncertainty_data_cf, columns=member_data_df_cf.columns, index=[region_id])
            # Concatenate the new row to the DataFrame
            member_data_df_noi = pd.concat([member_data_df_noi, row_noi])
            member_data_df_cf = pd.concat([member_data_df_cf, row_cf])

            all_loss_dfs.append(loss_df_subregion)

            # create a bar chart to show the volume loss by dataset
            ax2.bar(climate_run_output["time"].values[-1]+2, volume_loss_percentage_noirr[-1],
                    color=colors['noirr'][0], label="Volume Loss NoIrr(%)", alpha=0.6, zorder=0)

            ax2.bar(climate_run_output["time"].values[-1]+2, volume_loss_percentage_irr[-1],
                    color=colors['irr'][0],  label="Volume Loss Irr (%)", alpha=0.6, zorder=2)

            ax2.bar(climate_run_output["time"].values[-1]+2, volume_loss_percentage_cf[-1],
                    color=colors['cf'][0],  label="Volume Loss NoForcings (%)", alpha=0.6, zorder=1)

            ax2.axhline(0, color='black', linestyle='--',
                        linewidth=1.5, zorder=1)  # Dashed line at 0

            ax.plot(climate_run_output["time"].values, mean_values_noirr,
                    color=colors["noirr"][0], linestyle='solid', lw=2, label=f"NoIrr ({sum(members_averages)}-member avg)", zorder=3)
            ax.plot(climate_run_output_cf["time"].values, mean_values_cf,
                    color=colors["cf"][0], linestyle='solid', lw=2, label="fNoForcings ({sum(members_averages)}-member avg)", zorder=3)

            # calculate and plot volume/area 10-member min and max for ribbon
            min_values = np.min(stacked_member_data, axis=0).flatten()
            max_values = np.max(stacked_member_data, axis=0).flatten()
            ax.fill_between(climate_run_output["time"].values, min_values, max_values,
                            color=colors["noirr"][1], alpha=0.3, label=f"NoIrr ({sum(members_averages)}-member range)", zorder=3)

            min_values_cf = np.min(stacked_member_data_cf, axis=0).flatten()
            max_values_cf = np.max(stacked_member_data_cf, axis=0).flatten()
            ax.fill_between(climate_run_output_cf["time"].values, min_values_cf, max_values_cf,
                            color=colors["cf"][1], alpha=0.3, label=f"NoForcings ({sum(members_averages)}-member range)", zorder=3)

            # Determine row and column indices
            row = plot_index // n_cols
            col = plot_index % n_cols
            plot_index += 1

            # Add y label if it's the first column or bottom row
            if col == 0 and row == 2:
                # Set labels and title for the combined plot
                ax.set_ylabel(variable_axes[v])

            # Step 1: Set the value on ax1 where we want 0 on ax2 to align
            # e.g., the first value of data1
            align_value = (filtered_baseline[var].sum(
                dim="rgi_id") * factors[v])[0].values

            # Step 2: Calculate the offset required for ax2 limits
            mpl_axes_aligner.align.yaxes(ax, align_value, ax2, 0)

            # Adjust the y-label for the secondary y-axis
            if col == 2 and row == 2:
                ax2.set_ylabel("Volume Loss [%]", color='black')

            # Annotate region ID in the lower-left corner
            ax.text(0.05, 0.8, f'{region_id}', transform=ax.transAxes,
                    fontsize=12, verticalalignment='bottom', horizontalalignment='left', fontweight='bold')
            num_glaciers = len(filtered_baseline.rgi_id.values)
            ax.text(0.05, 0.05, f'{num_glaciers}', transform=ax.transAxes,
                    fontsize=12, verticalalignment='bottom', horizontalalignment='left')

    all_loss_dfs_combined = pd.concat(all_loss_dfs, ignore_index=True)
    o_folder_data = f"/Users/magaliponds/OneDrive - Vrije Universiteit Brussel/1. VUB/02. Coding/01. IRRMIP/03. Data/03. Output Files/02. OGGM/02. Volume Area simulations"
    o_file_data = f"{o_folder_data}/1985_2014.{timeframe}.delta.{variable_names[v]}.subregions.csv"
    o_file_data_uncertainties_noi = f"{o_folder_data}/1985_2014.{timeframe}.delta.{variable_names[v]}.subregions.uncertainties.noi.csv"
    o_file_data_uncertainties_cf = f"{o_folder_data}/1985_2014.{timeframe}.delta.{variable_names[v]}.subregions.uncertainties.cf.csv"
    all_loss_dfs_combined.to_csv(o_file_data)
    member_data_df_noi.to_csv(o_file_data_uncertainties_noi)
    member_data_df_cf.to_csv(o_file_data_uncertainties_cf)
    # Adjust the legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center',
               bbox_to_anchor=(0.5, 0.05), ncol=3)
    # ax.set_title(f"{region_id}")
    plt.tight_layout()
    plt.show()

    # specify and create a folder for saving the data (if it doesn't exists already) and save the plot
    o_folder_fig = f"/Users/magaliponds/OneDrive - Vrije Universiteit Brussel/1. VUB/02. Coding/01. IRRMIP/04. Figures/02. OGGM simulations/01. Modelled output/3r_a5/0{v + 1}. {variable_names[v]}/01. Subregions"
    os.makedirs(o_folder_data, exist_ok=True)
    o_file_name = f"{o_folder_fig}/1985_2014.{timeframe}.delta.{variable_names[v]}.subregions.png"
    plt.savefig(o_file_name, bbox_inches='tight')

# %% Cell 6c create output table


# Specify Paths
o_folder_data = (
    "/Users/magaliponds/OneDrive - Vrije Universiteit Brussel/1. VUB/02. Coding/01. IRRMIP/03. Data/03. Output Files/02. OGGM/02. Volume Area simulations"
)
o_file_data = f"{o_folder_data}/1985_2014.{timeframe}.delta.Volume.subregions.csv"
o_file_data_processed = f"{o_folder_data}/1985_2014.{timeframe}.delta.Volume.subregions_processed.csv"

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
    "volume_loss_percentage_cf": ((ds_total.loc[2014.0, "volume_cf"] - ds_total.loc[1985.0, "volume_irr"]) / ds_total.loc[1985.0, "volume_irr"]) * 100,
    "volume_irr": ds_total.loc[2014.0, "volume_irr"] - ds_total.loc[1985.0, "volume_irr"],
    "volume_noirr": ds_total.loc[2014.0, "volume_noirr"] - ds_total.loc[1985.0, "volume_irr"],
    "volume_cf": ds_total.loc[2014.0, "volume_cf"] - ds_total.loc[1985.0, "volume_irr"]
})

# Combine subregions with total
ds_all_losses = pd.concat([ds_subregions, df_total], ignore_index=True)

# Calculate deltas
ds_all_losses["delta_irr"] = ds_all_losses["volume_loss_percentage_noirr"] - \
    ds_all_losses["volume_loss_percentage_irr"]
ds_all_losses["delta_cf"] = ds_all_losses["volume_loss_percentage_cf"] - \
    ds_all_losses["volume_loss_percentage_irr"]

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
for scenario in ["noi", "cf"]:
    o_file_data_uncertainties = f"{o_folder_data}/1985_2014.{timeframe}.delta.Volume.subregions.uncertainties.{scenario}.csv"

    # Load uncertainties - lowest and heightest modelled values
    ds_uncertainties = pd.read_csv(o_file_data_uncertainties, index_col=0)
    ds_uncertainties.index.name = "subregion"

    # Calculate confidence intervals
    mean_values = ds_uncertainties.mean(axis=1)
    sem_values = ds_uncertainties.sem(axis=1)
    confidence_level = 0.90
    degrees_of_freedom = ds_uncertainties.shape[1] - 1
    critical_value = stats.t.ppf(
        (1 + confidence_level) / 2, degrees_of_freedom)
    margin_of_error_abs = critical_value * sem_values

    # Create confidence intervals DataFrame
    confidence_intervals = pd.DataFrame({
        f"error_margin_{scenario}_abs": margin_of_error_abs,
    }).reset_index()

    # Calculate total uncertainties
    ds_uncertainties_total = ds_uncertainties.sum().round(2)
    mean_values_total = ds_uncertainties_total.mean()
    sem_values_total = ds_uncertainties_total.sem()
    critical_value_total = stats.t.ppf(
        (1 + confidence_level) / 2, degrees_of_freedom)
    margin_of_error_total_abs = critical_value_total * sem_values_total

    df_uncertainties_total = pd.DataFrame({
        "subregion": ["total"],
        f"error_margin_{scenario}_abs": [margin_of_error_total_abs],
    })

    # Combine with total
    confidence_intervals = pd.concat(
        [confidence_intervals, df_uncertainties_total], ignore_index=True)

    # Merge confidence intervals into ds_all_losses
    ds_all_losses = ds_all_losses.merge(
        confidence_intervals, on="subregion", how="left")

# Merge RGI info into ds_all_losses
ds_all_losses['subregion'] = areas["subregion"]
ds_all_losses = ds_all_losses.merge(areas, on="subregion", how="left")
ds_all_losses = ds_all_losses.merge(volume, on="subregion", how="left")
ds_all_losses = ds_all_losses.merge(nr_glaciers, on="subregion", how="left")

ds_all_losses['error_margin_noi_rel'] = ds_all_losses['error_margin_noi_abs'] / \
    ds_all_losses['volume_noirr']*100
ds_all_losses['error_margin_cf_rel'] = ds_all_losses['error_margin_cf_abs'] / \
    ds_all_losses['volume_cf']*100
# Final selection of columns and save to CSV
ds_all_losses = ds_all_losses[[
    "subregion", "nr_glaciers", "area", "volume",
    "volume_loss_percentage_irr", "volume_irr",
    "volume_loss_percentage_noirr", "volume_noirr", "error_margin_noi_rel", "error_margin_noi_abs", "delta_irr",
    "volume_loss_percentage_cf", "volume_cf", "error_margin_cf_rel", "error_margin_cf_abs", "delta_cf",

]].round(2)

ds_all_losses.to_csv(o_file_data_processed)

# %% Process all volume losses and uncertainties to csv

o_folder_data = f"/Users/magaliponds/OneDrive - Vrije Universiteit Brussel/1. VUB/02. Coding/01. IRRMIP/03. Data/03. Output Files/02. OGGM/02. Volume Area simulations"
o_file_data = f"{o_folder_data}/1985_2014.{timeframe}.delta.Volume.subregions.csv"
o_file_data_processed = f"{o_folder_data}/1985_2014.{timeframe}.delta.Volume.subregions_processed.csv"
# Load your main dataset and process as required
ds = pd.read_csv(o_file_data)
ds_subregions = ds[ds.time == 2014.0][["subregion", 'volume_loss_percentage_irr',
                                       "volume_loss_percentage_noirr", "volume_loss_percentage_cf",
                                       'volume_irr', 'volume_noirr', 'volume_cf']]
# Process total volumes
ds_total = ds.groupby('time').sum()[
    ['volume_irr', 'volume_noirr', "volume_cf"]].round(2)
df_total = pd.DataFrame({
    "subregion": ["total"],
    "volume_loss_percentage_irr": ((ds_total.loc[2014.0, "volume_irr"] - ds_total.loc[1985.0, "volume_irr"]) / ds_total.loc[1985.0, "volume_irr"]) * 100,
    "volume_loss_percentage_noirr": ((ds_total.loc[2014.0, "volume_noirr"] - ds_total.loc[1985.0, "volume_irr"]) / ds_total.loc[1985.0, "volume_irr"]) * 100,
    "volume_loss_percentage_cf": ((ds_total.loc[2014.0, "volume_cf"] - ds_total.loc[1985.0, "volume_irr"]) / ds_total.loc[1985.0, "volume_irr"]) * 100,
    "volume_irr": ds_total.loc[2014.0, "volume_irr"] - ds_total.loc[1985.0, "volume_irr"],
    "volume_noirr": ds_total.loc[2014.0, "volume_noirr"] - ds_total.loc[1985.0, "volume_irr"],
    "volume_cf": ds_total.loc[2014.0, "volume_cf"] - ds_total.loc[1985.0, "volume_irr"]
})

# Concatenate subregions with totals
ds_all_losses = pd.concat([ds_subregions, df_total], ignore_index=True)
ds_all_losses["delta_irr"] = ds_all_losses["volume_loss_percentage_noirr"] - \
    ds_all_losses["volume_loss_percentage_irr"]
ds_all_losses["delta_cf"] = ds_all_losses["volume_loss_percentage_cf"] - \
    ds_all_losses["volume_loss_percentage_irr"]


# Load RGI data
df = pd.read_csv(
    f"{wd_path}masters/master_gdirs_r3_a5_rgi_date_A_V_RGIreg.csv")

# Standardize 'subregion' formatting in both DataFrames
df['subregion'] = df['rgi_subregion'].str.replace(
    '-', '.').str.strip().str.lower()

# Aggregate and rename columns in RGI data
areas = df.groupby('subregion').rgi_area_km2.sum().reset_index(name='area')
volume = df.groupby(
    'subregion').rgi_volume_km3.sum().reset_index(name='volume')
nr_glaciers = df.groupby(
    'subregion').rgi_id.count().reset_index(name='nr_glaciers')

areas_total = pd.DataFrame(
    {'subregion': ['total'], 'area': [areas['area'].sum()]})
volume_total = pd.DataFrame(
    {'subregion': ['total'], 'volume': [volume['volume'].sum()]})
nr_glaciers_total = pd.DataFrame({'subregion': ['total'], 'nr_glaciers': [
                                 nr_glaciers['nr_glaciers'].sum()]})

# Append the total row using pd.concat
areas = pd.concat([areas, areas_total], ignore_index=True)
volume = pd.concat([volume, volume_total], ignore_index=True)
nr_glaciers = pd.concat([nr_glaciers, nr_glaciers_total], ignore_index=True)

# this needs to be fixed!!!
ds_all_losses['subregion'] = areas['subregion']

# Merge RGI info into ds_all_losses
ds_all_losses = ds_all_losses.merge(areas, on='subregion', how='left')
ds_all_losses = ds_all_losses.merge(volume, on='subregion', how='left')
ds_all_losses = ds_all_losses.merge(nr_glaciers, on='subregion', how='left')

for scenario in ["noi", "cf"]:
    o_file_data_uncertainties = f"{o_folder_data}/1985_2014.{timeframe}.delta.Volume.subregions.uncertainties.{scenario}.csv"

    # Include the uncertainties by category
    ds_uncertainties = pd.read_csv(o_file_data_uncertainties, index_col=0)
    ds_uncertainties.index.name = 'subregion'
    print(ds_uncertainties)

    # Step 1: Calculate the mean for each subregion
    mean_values = ds_uncertainties.mean(axis=1)

    # Step 2: Calculate the standard error of the mean (SEM) for each subregion
    sem_values = ds_uncertainties.sem(axis=1)

    # Step 3: Determine the critical T-value for 90% confidence level
    confidence_level = 0.90
    # 11 members, so df = 11 - 1 = 10
    degrees_of_freedom = ds_uncertainties.shape[1] - 1
    critical_value = stats.t.ppf(
        (1 + confidence_level) / 2, degrees_of_freedom)

    # Step 4: Calculate the margin of error
    margin_of_error = critical_value * sem_values
    confidence_intervals = pd.DataFrame({
        f'error_margin_{scenario}_abs': margin_of_error
    }).reset_index()
    confidence_intervals['subregion'] = confidence_intervals['subregion'].astype(
        object)

    ds_uncertainties_total = ds_uncertainties.sum().round(2)
    mean_values_total = ds_uncertainties_total.mean()
    sem_values_total = ds_uncertainties_total.sem()
    # 11 members, so df = 11 - 1 = 10
    degrees_of_freedom_total = ds_uncertainties_total.shape[0] - 1
    critical_value_total = stats.t.ppf(
        (1 + confidence_level) / 2, degrees_of_freedom_total)
    margin_of_error_total = critical_value_total * sem_values_total

    df_uncertainties_total = pd.DataFrame({
        "subregion": ["total"],
        f"error_margin_{scenario}_abs": margin_of_error_total})

    # Concatenate subregions with totals
    confidence_intervals = pd.concat(
        [confidence_intervals, df_uncertainties_total], ignore_index=True)
    confidence_intervals['subregion'] = areas['subregion']
    print(confidence_intervals)

    # # Step 5: Calculate the confidence intervals
    # lower_bounds = mean_values - margin_of_error
    # upper_bounds = mean_values + margin_of_error

    ds_all_losses = ds_all_losses.merge(
        confidence_intervals, on='subregion', how='left')


# # Check results
ds_all_losses = ds_all_losses[['subregion', 'nr_glaciers', 'area', 'volume',
                               'volume_loss_percentage_irr', 'volume_irr',
                               'volume_loss_percentage_noirr', 'volume_noirr', 'delta_irr',
                               'error_margin_noi_abs',
                               'volume_loss_percentage_cf', 'volume_cf',  'delta_cf',
                               'error_margin_cf_abs',
                               ]].round(2)
ds_all_losses.to_csv(o_file_data_processed)
# print(ds_all_losses[['subregion', 'area', 'volume', 'nr_glaciers']].head())


# %% Cell 7: Correlation curve

hugo_ds = pd.read_csv(
    f"{wd_path}masters/master_gdirs_r3_a5_rgi_date_A_V_B_hugo.csv")

hugo_df = hugo_ds[['rgi_id', 'B_hugo']]

members = [1, 1, 1, 1, 1]
models = ["W5E5", "E3SM", "CESM2", "CNRM", "IPSL-CM6"]

plt.figure(figsize=(10, 6))

hugo_df = hugo_df.rename(columns={'B_hugo': 'B'})
# plt.scatter(mb_base.rgi_id, mb_base.B,color=colors["irr"][0])
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
gdf = gpd.GeoDataFrame(aggregated_ds, geometry=gpd.points_from_xy(
    aggregated_ds['lon'] + 0.5, aggregated_ds['lat'] + 0.5))

# Colormap and scatter plot
# custom_cmap = LinearSegmentedColormap.from_list(
#     'red_white_blue', [(1, 0, 0), (1, 1, 1), (0, 0, 1)], N=256)

# norm = TwoSlopeNorm(vmin=gdf['B_irr'].min(
# ), vcenter=0, vmax=gdf['B_irr'].max())
# norm = TwoSlopeNorm(vmin=-0.7, vcenter=0, vmax=0.2)
# Define the boundaries for each color block, switching every 0.1
boundaries = [-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2]

# Create the BoundaryNorm with the defined boundaries
# Define a list of colors for each range, making sure white is around zero
listed_colors = [
    (0.3, 0, 0),     # Very dark red
    (0.6, 0, 0),    # Darker red for values below -0.7
    (1, 0, 0),      # Red for values between -0.7 and -0.6
    (1, 0.2, 0.2),  # Lighter red for values between -0.6 and -0.5
    (1, 0.4, 0.4),  # Lighter red for values between -0.5 and -0.45
    (1, 0.6, 0.6),  # Lighter red for values between -0.45 and -0.4
    (1, 0.8, 0.8),  # Lightest red for values above -0.4
    (1, 1, 1),      # White for -0.2 to -0.1
    # (0.9, 0.9, 1),  # Light blue for -0.1 to 0 (this is distinct)
    (0.8, 0.8, 1),  # Light blue for values 0 to 0.1
    (0.5, 0.5, 1),  # Medium blue for 0.1 to 0.2
    (0, 0, 0.7),     # Darker blue for values above 0.3
    (0, 0, 0.4)     # Darker blue for values above 0.3
]


# Plot setup and plot shapefile
fig, ax = plt.subplots(figsize=(13, 10), subplot_kw={
                       'projection': ccrs.PlateCarree()})
# ax.set_extent([63, 107, 23, 48], crs=ccrs.PlateCarree())
# # Load shapefiles
shapefile_path = '/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/01. Input files/03. Shapefile/Karakoram/Pan-Tibetan Highlands/Pan-Tibetan Highlands (Liu et al._2022)/Shapefile/Pan-Tibetan Highlands (Liu et al._2022)_P.shp'
shp = gpd.read_file(shapefile_path).to_crs('EPSG:4326')
shp.plot(ax=ax, edgecolor='red', linewidth=0, facecolor='lightgrey')

subregions_path = "/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/02. QGIS/RGI outlines/GTN-G_O2regions_selected_clipped.shp"
subregions = gpd.read_file(subregions_path).to_crs('EPSG:4326')
ax.spines['geo'].set_visible(False)

# Optionally, remove gridlines
ax.gridlines().set_visible(False)

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

    highlighted_subregions = ["14-02", "14-01", "13-05", "13-02", "13-01"]

    alpha = 0.4  # if subregion.o2region.values in highlighted_subregions else 0.4
    facecolor = "none"  # if subregion.o2region.values in highlighted_subregions else "none"
    linecolor = "black"  # if subregion.o2region.values in highlighted_subregions else "black"
    subregion.plot(ax=ax, edgecolor=linecolor, linewidth=2,
                   facecolor=facecolor, alpha=alpha)  # Plot the subregion
    # Get the boundary of the subregion instead of the centroid
    boundary = subregion.geometry.boundary.iloc[0]

    # boundary_coords = list(boundary.coords)
    # boundary_x, boundary_y = boundary_coords[0]  # First point on the boundary
    # boundary_x -= movements[attribute][0]
    # boundary_y -= movements[attribute][1]
    # # Annotate or place text near the boundary
    # if names == "Code":
    #     ax.text(boundary_x, boundary_y, f"{subregion['o2region'].iloc[0]}",
    #             horizontalalignment='center', fontsize=10, color='black', fontweight='bold')
    # else:
    #     ax.text(boundary_x, boundary_y, f"{subregion['o2region'].iloc[0]}\n{subregion['full_name'].iloc[0]}",
    #             horizontalalignment='center', fontsize=10, color='black', fontweight='bold')

    # centroid = subregion.geometry.centroid.iloc[0]
    # centroid_x, centroid_y = centroid.x, centroid.y
    # if attribute == "14-02":
    #     ax.plot([centroid_x, boundary_x-1.5], [centroid_y,
    #             boundary_y-0.3], color='black', linewidth=1)
    # if attribute == "14-03":
    #     ax.plot([centroid_x, boundary_x], [centroid_y,
    #             boundary_y+0.5], color='black', linewidth=1)


# Create a ListedColormap with some colors (example)
custom_cmap = clrs.ListedColormap(listed_colors)
boundaries = [-0.75, -0.65, -0.55, -0.45, -0.35, -
              0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35]  # Define the boundaries for each color block
# Adjust the boundaries_ticks to match the boundaries
boundaries_ticks = [-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3]
# Create the BoundaryNorm with the defined boundaries
norm = clrs.BoundaryNorm(boundaries, custom_cmap.N, clip=False)
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm),
                    ax=ax, boundaries=boundaries, ticks=boundaries_ticks)  # Create the colorbar
cbar.set_label('$B_{Irr}$ (m w.e. yr$^{-1}$)',
               fontsize=12)  # Label for the colorbar
# Adjust labels to 1 decimal place
cbar.ax.set_yticklabels([f'{b:.1f}' for b in boundaries_ticks])
scatter = ax.scatter(gdf.geometry.x, gdf.geometry.y,
                     s=np.sqrt(gdf['rgi_area_km2'])*3, c=gdf['B_irr'], cmap=custom_cmap, norm=norm, edgecolor='k', alpha=1)

# # Add labels, ticks, and colorbar
ax.set(xlabel='Longitude', ylabel='Latitude', xticks=np.arange(
    45, 120, 5), yticks=np.arange(13, 55, 5))
ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)

# Add volume legend
custom_sizes = [200, 500, 1000, 2000]  # Example sizes for the legend -area
# Create labels for these sizes
size_labels = [f"{size:.0f}" for size in custom_sizes]

# Create legend handles using matplotlib Patch, simulating the size of scatter points
legend_handles = [plt.scatter([], [], s=np.sqrt(size)*3, edgecolor='k', facecolor='none')
                  for size in custom_sizes]  # Adjust size factor if needed

# Create the custom legend with the defined sizes
fig.legend(legend_handles, size_labels, loc="lower center", title="Total Area (km$^2$)", title_fontsize=12,
           bbox_to_anchor=(0.22, -0.1), ncol=5, fontsize=12)

# Define and iterate over grid layout
layout = [["13.01", "13.03", "13.04", "13.05", "13.06"], ["13.02", "", "", "", "13.07"], [
    "14.01", "14.02", "", "", "13.08"], ["14.03", "15.01", "15.02", "15.03", "13.09"]]
# grid_positions = [[-0.0 + col * (0.16 + 0.02), 0.8 - (0.1 + 0.05) * row - 0.05, 0.13, 0.14]
#                   if layout[row][col] else None for row in range(4) for col in range(5)]
movements_sp = {
    '13-01': [4.0, 3.0],   # Move further to the top-right
    '13-02': [10.0, 4.0],  # Move further to the top-right
    '13-03': [10.0, 3.0],  # Move further to the right
    '13-04': [-20, 3.0],   # Move further to the left
    '13-05': [-8, 2.0],    # Move further to the top-left
    '13-06': [18.0, -6.0],  # Move further down and to the right
    '13-07': [6.0, 3.0],   # Move further up and to the right
    '13-08': [-15.0, -8.0],  # Move further down and to the left
    '13-09': [-8.0, 5.0],  # Move further to the top-left
    '14-01': [3.0, 4.0],   # Move further to the top-right
    '14-02': [0.0, -10.0],  # Move further down
    '14-03': [3.0, 8.0],   # Move further to the top-right
    '15-01': [-10.0, 7.0],  # Move further to the top-left
    '15-02': [6.0, 4.5],   # Move further to the top-right
    '15-03': [-3.0, 8.0],  # Move further to the top-left
}

grid_positions = [[0.12 + col * (0.14 + 0.04), 0.82 - (0.14 + 0.07) * row - 0.05, 0.13, 0.14]
                  if layout[row][col] else None for row in range(4) for col in range(5)]

for idx, pos in enumerate(grid_positions):
    if pos:
        ax_callout = fig.add_axes(pos)
        region_id = layout[idx // 5][idx % 5]
        print(region_id)
        subregion_ds = master_ds_avg[master_ds_avg['rgi_subregion'].str.contains(
            f"{region_id}")]

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

        # Plot model member data
        ax_callout.plot(baseline_filtered["time"].values, baseline_filtered['volume'].sum(dim="rgi_id") * 1e-9,
                        label="W5E5.000", color="black", linewidth=2, zorder=15)
        filtered_member_data = []
        for m, model in enumerate(models_shortlist):
            for i in range(members_averages[m]):
                sample_id = f"{model}.00{i + 1}" if members_averages[m] > 1 else f"{model}.000"
                climate_run_output = xr.open_dataset(os.path.join(
                    sum_dir, f'climate_run_output_perturbed_{sample_id}.nc'))
                climate_run_output = climate_run_output.where(
                    climate_run_output['rgi_id'].isin(subregion_ds.rgi_id.values), drop=True)
                ax_callout.plot(climate_run_output["time"].values, climate_run_output["volume"].sum(
                    dim="rgi_id") * 1e-9, label=sample_id, color="grey", linewidth=2, linestyle="dotted")
                filtered_member_data.append(
                    climate_run_output["volume"].sum(dim="rgi_id").values * 1e-9)

        # Mean and range plotting
        mean_values = np.mean(filtered_member_data, axis=0).flatten()
        min_values = np.min(filtered_member_data, axis=0).flatten()
        max_values = np.max(filtered_member_data, axis=0).flatten()
        ax_callout.plot(climate_run_output["time"].values, mean_values,
                        color="blue", linestyle='dashed', lw=2, label=f"{sum(members_averages)}-member average")
        ax_callout.fill_between(
            climate_run_output["time"].values, min_values, max_values, color="lightblue", alpha=0.3)

        # Subplot formatting
        ax_callout.set_title(region_id, fontweight="bold", bbox=dict(
            facecolor='white', edgecolor='none', pad=1))
        # Count the number of glaciers (assuming each 'rgi_id' represents a glacier)
        glacier_count = subregion_ds['rgi_id'].nunique()
        # Add number of glaciers as a text annotation in the lower left corner
        ax_callout.text(0.05, 0.05, f"{glacier_count}",
                        transform=ax_callout.transAxes, fontsize=12, verticalalignment='bottom', fontstyle='italic')
        # ax_callout.set_xlim(-3, 3)
        # ax_callout.set_ylim(0, 20)
        if idx < len(grid_positions) - 5:
            ax_callout.tick_params(axis='x', labelbottom=False)
        # if idx % 5 != 0:
            ax_callout.tick_params(axis='y', labelleft=False)

# Sample data for the example plot (volume vs. time)
time = np.linspace(1985, 2015, 5)  # Simulated time points
volume_irr = [30, 28, 27, 25, 24]  # Simulated volume data for Irr
volume_noirr = [30, 27, 25, 23, 20]  # Simulated volume data for NoIrr
volume_members1 = [31, 28, 26, 24, 22]  # Individual members
volume_members2 = [29, 26, 24, 22, 18]  # Individual members

# Create a new figure for the small legend plot
fig_legend = fig.add_axes([0.5, -0.18, 0.13, 0.14])  # Small plot size

# Plot the sample data
fig_legend.plot(time, volume_irr, label='AllForcings (W5E5)',
                color='black', linewidth=2)  # Black line for Irr
fig_legend.plot(time, volume_noirr, label='fNoIrr ({sum(members_averages)}-member average)',
                color='blue', linestyle='-', linewidth=2)  # Blue line for NoIrr average
fig_legend.plot(time, volume_members1, label='NoIrr (individual member)', color='grey',
                linestyle='dotted', linewidth=1)  # Dotted grey for individual members
fig_legend.plot(time, volume_members2, label='', color='grey',
                linestyle='dotted', linewidth=1)  # Dotted grey for individual members

# Shade for the range
fig_legend.fill_between(time, volume_members2, volume_members1,
                        color='lightblue', alpha=0.3, label=f'NoIrr {sum(members_averages)}-member range')  # Shading for range

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
plt.savefig(f"{fig_folder}/Map_Plot_sA_cB_boxV_IRR_2000_2014.png")
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

# %% Cell 11a: Create new way nan mask


members = [3, 4, 6, 4, 1, 1]
models = ["E3SM", "CESM2", "CNRM", "NorESM", "W5E5", "IPSL-CM6"]

overview_df = pd.DataFrame()

for m, model in enumerate(models):
    for member in range(members[m]):
        df_tot = pd.DataFrame()
        sample_id = f"{model}.00{member}"

        for f, filepath in enumerate([f"climate_run_output_baseline_W5E5.000.nc", f"climate_run_output_baseline_W5E5.000_comitted_random.nc", f"climate_run_output_baseline_W5E5.000_comitted_random.nc"]):
            calendar_year = 2014
            if model != "W5E5":
                filepath = [f'climate_run_output_perturbed_{sample_id}.nc',
                            f'climate_run_output_perturbed_{sample_id}_comitted_random.nc',
                            f'climate_run_output_perturbed_{sample_id}_comitted_random_counterfactual.nc'][f]
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
            if f == 2:
                # Merge based on rgi_id
                df_tot = pd.merge(
                    df_tot,
                    df_individual[["rgi_id", "volume", "calendar_year"]],
                    on="rgi_id",
                    how="outer",
                    suffixes=("", "_noforcing")
                )
        df_zero_volume = df_tot[
            pd.isna(df_tot["volume_noirr"]) | pd.isna(
                df_tot["volume_noforcing"]) | pd.isna(df_tot["volume"])
        ]
    overview_df = pd.concat([overview_df, df_zero_volume], ignore_index=True)


unique_rgi_ids = overview_df['rgi_id'].unique()
print(len(unique_rgi_ids))
unique_rgi_ids = pd.DataFrame(unique_rgi_ids, columns=['rgi_ids'])
unique_rgi_ids.to_csv(os.path.join(
    wd_path, 'masters', 'nan_mask_comitted_random.csv'))

# Save the result to a CSV
output_path = os.path.join(
    wd_path, "masters", "nan_mask_all_models_volume_comitted_random.csv")
unique_rgi_ids.to_csv(output_path, index=False)


# %% CEll 11b: Create output plots for Area and volume Comitted - 2 axis
members_averages = [2, 3,  3,  5, 1]
models_shortlist = ["E3SM", "CESM2",  "NorESM",  "CNRM", "IPSL-CM6"]

# define the variables for p;lotting
variables = ["volume", "area"]
factors = [10**-9, 10**-6]

variable_names = ["Volume", "Area"]
variable_axes = ["Volume [km$^3$]", "Area [km$^2$]"]
use_multiprocessing = False

output_csv_path = os.path.join(wd_path, "masters", f"error_rgi_ids.csv")
error_ids = pd.read_csv(output_csv_path)['rgi_id'].tolist()
subset_gdirs = gdirs_3r_a5[:100]

nan_mask = pd.read_csv(os.path.join(
    wd_path, "masters", "nan_mask_all_models_volume.csv")).rgi_ids
# Remove duplicates if needed
nan_mask = set(pd.DataFrame({'rgi_id': nan_mask.unique()}).rgi_id.to_numpy())
rgi_ids_test = []
for gdir in subset_gdirs:
    rgi_ids_test.append(gdir.rgi_id)
rgi_ids_test = [rgi_id for rgi_id in rgi_ids_test if rgi_id not in nan_mask]
print(len(rgi_ids_test))


# repeat the loop for area and volume
for v, var in enumerate(variables):
    print(var)
    # create a new plot for both area and volume
    fig, ax = plt.subplots(figsize=(7, 4))  # create a new figure

    # create a timeseries for all the model members to add the data, needed to calculate averages later

    linestyles = ['solid', 'solid']
    # for f, filepath in enumerate([f"climate_run_output_baseline_W5E5.000.nc", f"climate_run_output_baseline_W5E5.000_comitted_random.nc"]):
    for f, filepath in enumerate([f"climate_run_output_baseline_W5E5.000.nc", f"climate_run_output_baseline_W5E5.000_comitted_cst_test.nc"]):
        if f == 1:
            color_id = "_com"
            legend_id = "committed"
            bar_values = 50
        else:
            color_id = ""
            legend_id = ""
            bar_values = 20
            ax2 = ax.twinx()

        print(f)
        # load and plot the baseline data
        baseline_path = os.path.join(
            wd_path, "summary", filepath)
        baseline = xr.open_dataset(baseline_path)
        # if f == 0:
        # rgi_ids_test = baseline.rgi_id.values[:10]
        baseline = baseline.where(
            baseline.rgi_id.isin(rgi_ids_test), drop=True)
        # print(baseline[var].sum(
        # dim="rgi_id").values * factors[v])
        # print(len(baseline.rgi_id))

        ax.plot(baseline["time"], baseline[var].sum(dim="rgi_id") * factors[v],
                label=f"W5E5.000 {legend_id}", color=colors[f"irr{color_id}"][0], linewidth=2, zorder=15, linestyle=linestyles[f])
        print("baseline time from:", baseline["time"][0].values)
        print("baseline time to:", baseline["time"][-1].values)

        mean_values_irr = (baseline[var].sum(
            dim="rgi_id") * factors[v]).values
        # print(baseline[var].sum(
        # dim="rgi_id").values * factors[v])

        # print(len(baseline[var][0].values))

        # # loop through all the different model x member combinations

    # for f in range(2):
        member_data_noirr = []
        nan_runs_noirr = []

        for m, model in enumerate(models_shortlist):
            for i in range(members_averages[m]):
                # make sure the counter for sample ids starts with 001, 000 are averages of all members by model
                # IPSL-CM6 only has 1 member, so the sample_id must end with 000
                if members_averages[m] > 1:
                    i += 1
                    label = None
                else:
                    label = "GCM member"

                sample_id = f"{model}.00{i}"
                filepath = [f'climate_run_output_perturbed_{sample_id}.nc',
                            # f'climate_run_output_perturbed_{sample_id}_comitted_random.nc'][f]
                            f'climate_run_output_perturbed_{sample_id}_comitted_cst_test.nc'][f]
                print(sample_id)
                # load and plot the data from the climate output run and counterfactual
                climate_run_opath_noirr = os.path.join(
                    sum_dir, filepath)  # f'climate_run_output_perturbed_{sample_id}_comitted.nc')
                climate_run_output_noirr = xr.open_dataset(
                    climate_run_opath_noirr)
                # if f == 0:
                # rgi_ids_test=baseline.rgi_id[:10]
                climate_run_output_noirr = climate_run_output_noirr.where(
                    climate_run_output_noirr.rgi_id.isin(rgi_ids_test), drop=True)
                # print(len(climate_run_output_noirr.rgi_id))
                # print(climate_run_output_noirr[var].sum(
                # dim="rgi_id").values * factors[v])
                # nan_mask = 0
                # nan_mask = climate_run_output_noirr['volume'].isnull()
                # nan_runs = climate_run_output_noirr.where(nan_mask, drop=True)
                # # print(nan_runs)
                # nan_runs_noirr.extend(nan_runs['rgi_id'].values)
                ax.plot(climate_run_output_noirr["time"], climate_run_output_noirr[var].sum(dim="rgi_id") * factors[v],
                        label=None, color=colors[f"noirr{color_id}"][1], linewidth=1, linestyle=linestyles[f])
                print("climate run output time from:",
                      climate_run_output_noirr["time"][0].values)
                print("climate run output time to:",
                      climate_run_output_noirr["time"][-1].values)

                # add all the summed volumes/areas to the member list, so a multi-member average can be calculated
                member_data_noirr.append(climate_run_output_noirr[var].sum(
                    dim="rgi_id").values * factors[v])
                # print(len(climate_run_output_noirr[var][0].values))

        # stack the member data
        stacked_member_data = np.stack(member_data_noirr)
        all_nan_rgi_ids = np.unique(nan_runs_noirr)
        df_nan_rgi_ids = pd.DataFrame({'rgi_id': all_nan_rgi_ids})
        output_csv_path = os.path.join(
            wd_path, "masters", f"error_rgi_ids.csv")
        df_nan_rgi_ids.to_csv(output_csv_path, index=False)

        # calculate and plot volume/area 10-member mean
        # mean_values_noirr = np.mean(stacked_member_data, axis=0).flatten()
        mean_values_noirr = np.median(stacked_member_data, axis=0).flatten()
        ax.plot(climate_run_output_noirr["time"].values, mean_values_noirr,
                color=colors[f"noirr{color_id}"][0], linestyle=linestyles[f], lw=2, label=f"NoIrr ({sum(members_averages)}-member avg) {legend_id}")

        # calculate and plot volume/area 10-member min and max for ribbon
        min_values_noirr = np.min(stacked_member_data, axis=0).flatten()
        max_values_noirr = np.max(stacked_member_data, axis=0).flatten()
        ax.fill_between(climate_run_output_noirr["time"].values, min_values_noirr, max_values_noirr,
                        color=colors[f"noirr{color_id}"][1], alpha=0.3, label=f"NoIrr ({sum(members_averages)}-member range) {legend_id}", zorder=16)

        if f == 0:
            var_value_1985 = mean_values_irr[0]

        # calculate the volume loss by scenario: irr - noirrrr and counterfactual
        volume_loss_percentage_noirr = (
            (mean_values_noirr - var_value_1985) / var_value_1985) * 100
        volume_loss_percentage_irr = (
            (mean_values_irr - var_value_1985) / var_value_1985) * 100
        # #create a dataframe with the volume loss percentages and absolute values

        loss_df_subregion = pd.DataFrame({
            'time': climate_run_output_noirr["time"].values,
            # 'subregion': np.repeat(region_id, len(climate_run_output_noirr["time"])),
            'volume_irr': mean_values_irr,
            'volume_noirr': mean_values_noirr,
            'volume_loss_percentage_irr': volume_loss_percentage_irr,
            'volume_loss_percentage_noirr': volume_loss_percentage_noirr,
        })

        # o_folder_name = f"{wd_path}masters/master_gdirs_r3_a5_volume_evolution{color_id}.csv"
        # loss_df_subregion.to_csv(o_folder_name)

        # # create a bar chart to show the volume loss by dataset
        # ax2.bar(climate_run_output_noirr["time"].values[-1]+bar_values, volume_loss_percentage_noirr[-1],
        #         color=colors[f'noirr{color_id}'][0], label="Volume Loss NoIrr(%)", alpha=1, zorder=0, width=15)

        # ax2.bar(climate_run_output_noirr["time"].values[-1]+bar_values, volume_loss_percentage_irr[-1],
        #         color=colors[f'irr{color_id}'][0],  label="Volume Loss Irr (%)", alpha=1, zorder=2, width=15)

        # if f==0:
        # ax.axvline(climate_run_output_noirr["time"].values[-1], color="black", lw=1, zorder=100, ls="--")

    ax2.axhline(0, color='black', linestyle='--',
                linewidth=1, zorder=1)  # Dashed line at 0
    if var == "area":
        # Adjust this range to make the secondary axis extend larger
        ax2.set_ylim(-6, 5)
    if var == "volume":
        print(var)
        ax2.set_ylim(-12, 5)

    # Step 1: Set the value on ax1 where we want 0 on ax2 to align
    # e.g., the first value of data1
    # align_value = (baseline[var].sum(dim="rgi_id") * factors[v])[0].values
    align_value = var_value_1985

    # Step 2: Calculate the offset required for ax2 limits
    mpl_axes_aligner.align.yaxes(ax, align_value, ax2, 0)

    # Set labels and title for the combined plot
    ax.set_ylabel(variable_axes[v])
    ax2.set_ylabel(f"{variable_names[v]} [% of 1985-Irr]")
    ax.set_xlabel("Time [year]")
    ax.set_title(f"Summed {variable_names[v]}, RGI 13-15, A >5 km$^2$")

    # Adjust the legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
               bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.tight_layout()

    # specify and create a folder for saving the data (if it doesn't exists already) and save the plot
    o_folder_data = f"/Users/magaliponds/OneDrive - Vrije Universiteit Brussel/1. VUB/02. Coding/01. IRRMIP/04. Figures/02. OGGM simulations/01. Modelled output/3r_a5/0{v + 1}. {variable_names[v]}/00. Combined"
    os.makedirs(o_folder_data, exist_ok=True)
    # o_file_name = f"{o_folder_data}/1985_2014.{timeframe}.delta.{variable_names[v]}.comitted_random.png"
    o_file_name = f"{o_folder_data}/1985_2014.{timeframe}.delta.{variable_names[v]}.comitted_cst_test.png"
    # plt.savefig(o_file_name, bbox_inches='tight')

# %% plot comitted data - % axis

members_averages = [2, 3,  3,  5, 1]
models_shortlist = ["E3SM", "CESM2",  "NorESM",  "CNRM", "IPSL-CM6"]

# define the variables for p;lotting
variables = ["volume", "area"]
factors = [10**-9, 10**-6]

variable_names = ["Volume", "Area"]
variable_axes = ["Volume compared to 1985 Irr-scenario [%]",
                 "Area compared to 1985 Irr-scenario [%]"]
use_multiprocessing = False

output_csv_path = os.path.join(wd_path, "masters", f"error_rgi_ids.csv")
error_ids = pd.read_csv(output_csv_path)['rgi_id'].tolist()
subset_gdirs = gdirs_3r_a5[:100]

nan_mask = pd.read_csv(os.path.join(
    wd_path, "masters", "nan_mask_all_models_volume_comitted_random.csv")).rgi_ids
# Remove duplicates if needed
nan_mask = set(pd.DataFrame({'rgi_id': nan_mask.unique()}).rgi_id.to_numpy())
rgi_ids_test = []
for gdir in subset_gdirs:
    rgi_ids_test.append(gdir.rgi_id)
rgi_ids_test = [rgi_id for rgi_id in rgi_ids_test if rgi_id not in nan_mask]
print(len(rgi_ids_test))


# repeat the loop for area and volume
for v, var in enumerate(variables):
    print(var)
    # create a new plot for both area and volume
    fig, ax = plt.subplots(figsize=(7, 4))  # create a new figure

    # create a timeseries for all the model members to add the data, needed to calculate averages later

    linestyles = ['solid', 'solid', 'solid']
    for f, filepath in enumerate([f"climate_run_output_baseline_W5E5.000.nc", f"climate_run_output_baseline_W5E5.000_comitted_random.nc"]):
        # for f, filepath in enumerate([f"climate_run_output_baseline_W5E5.000.nc", f"climate_run_output_baseline_W5E5.000_comitted_cst_test.nc"]):
        if f != 0:
            run_type = "noirr"
            color_id = "_com"
            legend_id = "committed"
            bar_values = 50
            run_label = "NoIrr"

        # if f == 2:
        #     run_type = "cf"
        #     color_id = "_com"
        #     legend_id = "committed"
        #     bar_values = 50
        #     run_label= "NoForcings"
        else:
            run_type = "irr"
            color_id = ""
            legend_id = ""
            bar_values = 20

        # load and plot the baseline data
        baseline_path = os.path.join(
            wd_path, "summary", filepath)
        baseline = xr.open_dataset(baseline_path)
        # if f == 0:
        # rgi_ids_test = baseline.rgi_id.values[:10]
        baseline = baseline.where(
            baseline.rgi_id.isin(rgi_ids_test), drop=True)
        # print(baseline[var].sum(
        # dim="rgi_id").values * factors[v])
        # print(len(baseline.rgi_id))
        if f == 0:
            resp_value = baseline[var].sum(dim="rgi_id")[0].values * factors[v]
        ax.plot(baseline["time"], (baseline[var].sum(dim="rgi_id") * factors[v])/resp_value*100,
                label=f"AllForcings {legend_id}", color=colors[f"irr{color_id}"][0], linewidth=2, zorder=15, linestyle=linestyles[f])
        print("baseline time from:", baseline["time"][0].values)
        print("baseline time to:", baseline["time"][-1].values)

        mean_values_irr = (baseline[var].sum(
            dim="rgi_id") * factors[v]).values

        for r in range(2):
            member_data_noirr = []
            nan_runs_noirr = []
            if r == 1:
                run_type = "noirr"
                color_id = "_com"
                legend_id = "committed"
                bar_values = 50
                run_label = "NoIrr"
            else:
                run_type = "cf"
                color_id = "_com"
                legend_id = "committed"
                bar_values = 50
                run_label = "NoForcings"

            for m, model in enumerate(models_shortlist):
                for i in range(members_averages[m]):
                    sample_id = f"{model}.00{i}"

                    # make sure the counter for sample ids starts with 001, 000 are averages of all members by model
                    # IPSL-CM6 only has 1 member, so the sample_id must end with 000
                    if members_averages[m] > 1:
                        i += 1
                        label = None
                    else:
                        label = "GCM member"
                    if r == 0:
                        filepath = [f'climate_run_output_perturbed_{sample_id}_counterfactual.nc',
                                    f'climate_run_output_perturbed_{sample_id}_comitted_random_counterfactual.nc',
                                    ][f]
                    else:
                        filepath = [f'climate_run_output_perturbed_{sample_id}.nc',
                                    f'climate_run_output_perturbed_{sample_id}_comitted_random.nc',
                                    ][f]

                    # f'climate_run_output_perturbed_{sample_id}_comitted_cst_test.nc'][f]
                    print(sample_id)
                    # load and plot the data from the climate output run and counterfactual
                    climate_run_opath_noirr = os.path.join(
                        sum_dir, filepath)  # f'climate_run_output_perturbed_{sample_id}_comitted.nc')
                    climate_run_output_noirr = xr.open_dataset(
                        climate_run_opath_noirr)
                    # if f == 0:
                    # rgi_ids_test=baseline.rgi_id[:10]
                    climate_run_output_noirr = climate_run_output_noirr.where(
                        climate_run_output_noirr.rgi_id.isin(rgi_ids_test), drop=True)
                    ax.plot(climate_run_output_noirr["time"], (climate_run_output_noirr[var].sum(dim="rgi_id") * factors[v])/resp_value*100,
                            label=None, color=colors[f"{run_type}{color_id}"][f], linewidth=1, linestyle="dotted", zorder=10)

                    # add all the summed volumes/areas to the member list, so a multi-member average can be calculated
                    member_data_noirr.append((climate_run_output_noirr[var].sum(
                        dim="rgi_id").values/resp_value*100 * factors[v]))
                    print(len(member_data_noirr))
                    # print(len(climate_run_output_noirr[var][0].values))

            # stack the member data
            stacked_member_data = np.stack(member_data_noirr)
            all_nan_rgi_ids = np.unique(nan_runs_noirr)
            df_nan_rgi_ids = pd.DataFrame({'rgi_id': all_nan_rgi_ids})
            output_csv_path = os.path.join(
                wd_path, "masters", f"error_rgi_ids.csv")
            df_nan_rgi_ids.to_csv(output_csv_path, index=False)

            # calculate and plot volume/area 10-member mean
            mean_values_noirr = np.median(
                stacked_member_data, axis=0).flatten()
            # mean_values_noirr = np.mean(stacked_member_data, axis=0).flatten()

            ax.plot(climate_run_output_noirr["time"].values, mean_values_noirr,
                    color=colors[f"{run_type}{color_id}"][f], linestyle=linestyles[f], lw=2, label=f"{run_label} ({sum(members_averages)}-member avg) {legend_id}", zorder=5)

            # calculate and plot volume/area 10-member min and max for ribbon
            min_values_noirr = np.min(stacked_member_data, axis=0).flatten()
            max_values_noirr = np.max(stacked_member_data, axis=0).flatten()
            ax.fill_between(climate_run_output_noirr["time"].values, min_values_noirr, max_values_noirr,
                            color=colors[f"{run_type}{color_id}"][f], alpha=0.3, label=f"{run_label} ({sum(members_averages)}-member range) {legend_id}", zorder=16)

        if f == 0:
            var_value_1985 = mean_values_irr[0]

    # Set labels and title for the combined plot
    ax.set_ylabel(variable_axes[v])
    ax.set_xlabel("Time [year]")
    ax.set_title(f"Summed {variable_names[v]}, RGI 13-15, A >5 km$^2$")

    # Adjust the legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
               bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.tight_layout()

    # specify and create a folder for saving the data (if it doesn't exists already) and save the plot
    o_folder_data = f"/Users/magaliponds/OneDrive - Vrije Universiteit Brussel/1. VUB/02. Coding/01. IRRMIP/04. Figures/02. OGGM simulations/01. Modelled output/3r_a5/0{v + 1}. {variable_names[v]}/00. Combined"
    os.makedirs(o_folder_data, exist_ok=True)
    o_file_name = f"{o_folder_data}/1985_2014.{timeframe}.delta.{variable_names[v]}.comitted_random.png"
    # o_file_name = f"{o_folder_data}/1985_2014.{timeframe}.delta.{variable_names[v]}.comitted_cst_test.png"
    # plt.savefig(o_file_name, bbox_inches='tight')

# %% Plot comitted data by region

# Define constants
members_averages = [2, 3, 3, 5, 1]
models_shortlist = ["E3SM", "CESM2", "NorESM", "CNRM", "IPSL-CM6"]
variables = ["volume"]
factors = [10**-9]
variable_names = ["Volume"]
variable_axes = ["Volume [km$^3$]"]
regions = [13, 14, 15]
subregions = [9, 3, 3]
region_colors = {
    13: 'blue',    # Blue for region 13
    14: 'crimson',  # Crimson for region 14
    15: 'orange'   # Orange for region 15
}

chunck_size = 1000

nan_mask = pd.read_csv(os.path.join(
    wd_path, "masters", "nan_mask_all_models_volume_comitted_random.csv")).rgi_ids
# Remove duplicates if needed
nan_mask = set(pd.DataFrame({'rgi_id': nan_mask.unique()}).rgi_id.to_numpy())


# Function to load and filter dataset
def load_filtered_dataset(filepath, rgi_ids, nan_mask):
    ds = xr.open_dataset(filepath)
    ds = ds.where(ds['rgi_id'].isin(rgi_ids), drop=True)
    ds = ds.where(~ds['rgi_id'].isin(nan_mask), drop=True)

    return ds[['volume', 'time', 'rgi_id']]

# Main plotting function


def plot_by_subregion(sum_dir, wd_path):
    for v, var in enumerate(variables):
        legend_id = "NoIrr"
        print(f"Processing variable: {var}")
        n_rows, n_cols = 5, 3
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(
            12, 12), sharex=True, sharey=True, gridspec_kw={'hspace': 0.15, 'wspace': 0.35})
        axes = axes.flatten()
        plot_index = 0

        for r, region in enumerate(regions):
            for subregion_idx in range(subregions[r]):
                ax = axes[plot_index]

                # Define region and subregion IDs
                subregion_idx += 1
                region_id = f"{region}.0{subregion_idx}"
                print(f"Processing subregion: {region_id}")
                # subregion_ds = ds[ds['rgi_subregion'].str.contains(f"{region_id}")]
                # subregion_ds = subregion_ds.rgi_id
                # subregion_ds.to_csv(os.path.join(wd_path,'masters',f"rgi_ids_{region_id}.csv"))

                ds_path = f"{wd_path}/masters/rgi_ids_{region_id}.csv"
                subregion_ds = pd.read_csv(ds_path)

                for f, filepaths in enumerate([f'climate_run_output_baseline_W5E5.000.nc', f'climate_run_output_baseline_W5E5.000_comitted_random.nc']):
                    filepath = os.path.join(sum_dir, filepaths)
                    # Load baseline data
                    baseline_path = os.path.join(wd_path, "summary", filepath)
                    baseline = load_filtered_dataset(
                        baseline_path, subregion_ds, nan_mask)  # .rgi_id.values)
                    if f == 0:
                        resp_value = baseline[var].sum(dim="rgi_id")[0].values

                    # Plot baseline data
                    ax.plot(baseline["time"], (baseline[var].sum(dim="rgi_id")/resp_value*100),
                            label="AllForcings", color=colors["irr"][f], linewidth=2, zorder=3)

                    member_data_noirr_com = []
                    member_data_noirr = []

                    for m, model in enumerate(models_shortlist):
                        for i in range(members_averages[m]):
                            if members_averages[m] > 1:
                                i += 1
                            sample_id = f"{model}.00{i}"
                            print(sample_id)

                            # Loop through models and members
                            filepaths = [
                                f'climate_run_output_perturbed_{sample_id}.nc', f'climate_run_output_perturbed_{sample_id}_comitted_random.nc']
                            filepath = os.path.join(sum_dir, filepaths[f])

                            climate_run_output = load_filtered_dataset(
                                filepath, subregion_ds, nan_mask)  # .rgi_id.values)

                            # Plot model member data
                            ax.plot(climate_run_output["time"], (climate_run_output[var].sum(dim="rgi_id") / resp_value*100),
                                    label=None, color=colors["noirr"][f], linewidth=0.5, linestyle="solid", zorder=2)

                            # Annotate region ID and glacier count

                            if f == 0:
                                member_data_noirr.append(
                                    (climate_run_output[var].sum(dim="rgi_id") / resp_value*100))
                            else:
                                member_data_noirr_com.append(
                                    (climate_run_output[var].sum(dim="rgi_id") / resp_value*100))

                    if f == 0:
                        stacked_member_data = np.stack(member_data_noirr)
                    else:
                        stacked_member_data = np.stack(member_data_noirr_com)

                    mean_values_noirr = np.median(
                        stacked_member_data, axis=0).flatten()
                    ax.plot(climate_run_output["time"].values, mean_values_noirr,
                            color=colors[f"noirr"][f], linestyle='solid', lw=2, label=f"NoIrr ({sum(members_averages)}-member avg) {legend_id}", zorder=3)

                    # calculate and plot volume/area 10-member min and max for ribbon
                    min_values_noirr = np.min(
                        stacked_member_data, axis=0).flatten()
                    max_values_noirr = np.max(
                        stacked_member_data, axis=0).flatten()
                    ax.fill_between(climate_run_output["time"].values, min_values_noirr, max_values_noirr,
                                    color=colors[f"noirr"][f], alpha=0.3, label=f"NoIrr ({sum(members_averages)}-member range) {legend_id}", zorder=0)

                    ax.text(0.05, 0.8, f'{region_id}', transform=ax.transAxes,
                            fontsize=12, fontweight='bold', zorder=20)
                    num_glaciers = len(baseline.rgi_id.values)
                    ax.text(0.05, 0.05, f'{num_glaciers} glaciers',
                            transform=ax.transAxes, fontsize=10, zorder=20)
                    if region_id == "15.02":
                        plt.xlabel("Time [yrs]")
                    if region_id == "13.07":
                        plt.ylabel("Volume compared to 1985-All Forcings [%]")
                plot_index += 1

        # Adjust layout and save figure
        plt.tight_layout()
        plt.show()

# Example usage


plot_by_subregion(sum_dir, wd_path)


# %% Plot individual model connection


# define the variables for p;lotting
variables = ["volume"]  # , "area"]
factors = [10**-9, 10**-6]

variable_names = ["Volume", "Area"]
variable_axes = ["Volume [km$^3$]", "Area [km$^2$]"]
use_multiprocessing = False

subset_gdirs = gdirs_3r_a5  # [:100]

rgi_ids_test = []
for gdir in subset_gdirs:
    rgi_ids_test.append(gdir.rgi_id)
# nan_mask = ['RGI60-13.22268', 'RGI60-13.53308',
#             'RGI60-13.36584', 'RGI60-13.54076', 'RGI60-14.14436']

# nan_mask = ['RGI60-13.22268', 'RGI60-13.53308',
#             'RGI60-13.36584', 'RGI60-13.54076', 'RGI60-14.14436']

rgi_ids_test_2 = []  # [rgi_id for rgi_id in rgi_ids_test if rgi_id not in nan_mask]
print(len(rgi_ids_test))
# repeat the loop for area and volume
for v, var in enumerate(variables):
    print(var)
    # create a new plot for both area and volume

    values_baseline = []
    values_baseline_2 = []
    values_noirr = []
    values_noirr_2 = []

    linestyles = ['solid', 'solid']

    nan_runs_noirr = []
    members_averages = [1, 2, 3, 5, 3]  # 1
    # "E3SM", "CESM2", "CNRM", "NorESM"]#"IPSL-CM6",
    # "E3SM", "CESM2", "CNRM", "NorESM"]
    models_shortlist = ["IPSL-CM6", "E3SM", "CESM2", "CNRM", "NorESM"]
    fig, ax = plt.subplots(figsize=(7, 4))  # create a new figure

    for m, model in enumerate(models_shortlist):

        for i in range(members_averages[m]):
            if members_averages[m] > 1:
                i += 1
                label = None
            else:
                label = "GCM member"

            sample_id = f"{model}.00{i}"
            for f, filepath in enumerate([f"climate_run_output_baseline_W5E5.000.nc", f"climate_run_output_baseline_W5E5.000_comitted_random.nc"]):
                member_data_noirr = []
                if f == 1:
                    color_id = "_com"
                    legend_id = "committed"
                    bar_values = 50
                    end = 0
                else:
                    color_id = ""
                    legend_id = ""
                    bar_values = 20
                    # ax2 = ax.twinx()
                    end = -1

                print(f)
                # load and plot the baseline data
                baseline_path = os.path.join(
                    wd_path, "summary", filepath)
                baseline = xr.open_dataset(baseline_path)
                # if f == 0:
                # rgi_ids_test = baseline.rgi_id.values[:10]
                # baseline = baseline.where(
                #     baseline.rgi_id.isin(rgi_ids_test), drop=True)
                # baseline = baseline.where(
                #     baseline.rgi_id.isin(rgi_ids_test_2), drop=True)

                ax.plot(baseline["time"][end], baseline[var].sum(dim="rgi_id")[end] * factors[v],
                        label=f"W5E5.000 {legend_id}", color=colors[f"irr{color_id}"][0], linewidth=2, zorder=15, linestyle=linestyles[f])
                mean_values_irr = (baseline[var].sum(
                    dim="rgi_id") * factors[v]).values
                print("baseline:", baseline["time"][end].values, baseline[var].sum(
                    dim="rgi_id")[end].values * factors[v])
                # print(len(baseline[var][0].values))

                # # loop through all the different model x member combinations
                # make sure the counter for sample ids starts with 001, 000 are averages of all members by model
                # IPSL-CM6 only has 1 member, so the sample_id must end with 000

                filepath = [f'climate_run_output_perturbed_{sample_id}.nc',
                            f'climate_run_output_perturbed_{sample_id}_comitted_random.nc'][f]

                # load and plot the data from the climate output run and counterfactual
                climate_run_opath_noirr = os.path.join(
                    sum_dir, filepath)  # f'climate_run_output_perturbed_{sample_id}_comitted.nc')
                climate_run_output_noirr = xr.open_dataset(
                    climate_run_opath_noirr)
                # if f == 0:
                # rgi_ids_test=baseline.rgi_id[:10]
                # climate_run_output_noirr = climate_run_output_noirr.where(
                #     climate_run_output_noirr.rgi_id.isin(rgi_ids_test_2), drop=True)

                # nan_mask = 0
                # nan_mask = climate_run_output_noirr['volume'].isnull()
                # nan_runs = climate_run_output_noirr.where(nan_mask, drop=True)
                # # print(nan_runs)
                # nan_runs_noirr.extend(nan_runs['rgi_id'].values)
                ax.plot(climate_run_output_noirr["time"][end], climate_run_output_noirr[var].sum(dim="rgi_id")[end] * factors[v],
                        label=None, color=colors[f"noirr{color_id}"][1], linewidth=1, linestyle=linestyles[f])
                print(f"noirr member {sample_id}:", climate_run_output_noirr["time"][end].values, climate_run_output_noirr[var].sum(
                    dim="rgi_id")[end].values * factors[v])

                # add all the summed volumes/areas to the member list, so a multi-member average can be calculated
                member_data_noirr.append(climate_run_output_noirr['volume'].sum(
                    dim="rgi_id").values * factors[v])
                # print(len(climate_run_output_noirr[var][0].values))

                if f == 0:
                    for rgi_id, baseline_volume in zip(baseline['rgi_id'].values, baseline['volume'].values[-1]):
                        values_baseline.append(
                            (sample_id, rgi_id, baseline_volume, "base"))
                    for rgi_id, volume in zip(climate_run_output_noirr['rgi_id'].values, climate_run_output_noirr['volume'].values[0]):
                        values_noirr.append(
                            (sample_id, rgi_id, volume, "noirr"))
                else:
                    for rgi_id, baseline_volume in zip(baseline['rgi_id'].values, baseline['volume'].values[-1]):
                        values_baseline_2.append(
                            (sample_id, rgi_id, baseline_volume, "base"))
                    for rgi_id, volume in zip(climate_run_output_noirr['rgi_id'].values, climate_run_output_noirr['volume'].values[0]):
                        values_noirr_2.append(
                            (sample_id, rgi_id, volume, "noirr"))

            # stack the member data
    stacked_member_data = np.stack(member_data_noirr)
    all_nan_rgi_ids = np.unique(nan_runs_noirr)
    df_nan_rgi_ids = pd.DataFrame({'rgi_id': all_nan_rgi_ids})
    output_csv_path = os.path.join(
        wd_path, "masters", f"error_rgi_ids.csv")
    df_nan_rgi_ids.to_csv(output_csv_path, index=False)

    # calculate and plot volume/area 10-member mean
    mean_values_noirr = np.median(stacked_member_data, axis=0).flatten()
    # mean_values_noirr = np.mean(stacked_member_data, axis=0).flatten()
    ax.plot(climate_run_output_noirr["time"][end].values, mean_values_noirr[end],
            color=colors[f"noirr{color_id}"][0], linestyle=linestyles[f], lw=2, label=f"NoIrr ({sum(members_averages)}-member avg) {legend_id}")
    print("noirr mean:",
          climate_run_output_noirr["time"][end].values, mean_values_noirr[end])
    # calculate and plot volume/area 10-member min and max for ribbon
    # min_values_noirr = np.min(stacked_member_data, axis=0).flatten()
    # max_values_noirr = np.max(stacked_member_data, axis=0).flatten()

    # if f == 0:
    #     var_value_1985 = mean_values_irr[0]

    # # calculate the volume loss by scenario: irr - noirrrr and counterfactual
    # volume_loss_percentage_noirr = (
    #     (mean_values_noirr - var_value_1985) / var_value_1985) * 100
    # volume_loss_percentage_irr = (
    #     (mean_values_irr - var_value_1985) / var_value_1985) * 100
    # # #create a dataframe with the volume loss percentages and absolute values

    # loss_df_subregion = pd.DataFrame({
    #     'time': climate_run_output_noirr["time"].values,
    #     # 'subregion': np.repeat(region_id, len(climate_run_output_noirr["time"])),
    #     'volume_irr': mean_values_irr,
    #     'volume_noirr': mean_values_noirr,
    #     'volume_loss_percentage_irr': volume_loss_percentage_irr,
    #     'volume_loss_percentage_noirr': volume_loss_percentage_noirr,
    # })

    # o_folder_name = f"{wd_path}masters/master_gdirs_r3_a5_volume_evolution{color_id}.csv"
    # loss_df_subregion.to_csv(o_folder_name)

    # # create a bar chart to show the volume loss by dataset
    # ax2.bar(climate_run_output_noirr["time"].values[-1]+bar_values, volume_loss_percentage_noirr[-1],
    #         color=colors[f'noirr{color_id}'][0], label="Volume Loss NoIrr(%)", alpha=1, zorder=0, width=15)

    # ax2.bar(climate_run_output_noirr["time"].values[-1]+bar_values, volume_loss_percentage_irr[-1],
    #         color=colors[f'irr{color_id}'][0],  label="Volume Loss Irr (%)", alpha=1, zorder=2, width=15)

    # # if f==0:
    # # ax.axvline(climate_run_output_noirr["time"].values[-1], color="black", lw=1, zorder=100, ls="--")

    # ax2.axhline(0, color='black', linestyle='--',
    #             linewidth=1, zorder=1)  # Dashed line at 0
    # if var == "area":
    #     # Adjust this range to make the secondary axis extend larger
    #     ax2.set_ylim(-6, 5)
    # if var == "volume":
    #     print(var)
    #     ax2.set_ylim(-6, 5)

    # # Step 1: Set the value on ax1 where we want 0 on ax2 to align
    # # e.g., the first value of data1
    # # align_value = (baseline[var].sum(dim="rgi_id") * factors[v])[0].values
    # align_value = var_value_1985

    # # Step 2: Calculate the offset required for ax2 limits
    # mpl_axes_aligner.align.yaxes(ax, align_value, ax2, 0)

    # # Set labels and title for the combined plot
    # ax.set_ylabel(variable_axes[v])
    # ax2.set_ylabel(f"{variable_names[v]} [% of 1985-Irr]")
    # ax.set_xlabel("Time [year]")
    # ax.set_title(f"Summed {variable_names[v]}, {sample_id}")

    # Adjust the legend
    # handles, labels = ax.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='lower center',
    #             bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.tight_layout()
plt.show()

# specify and create a folder for saving the data (if it doesn't exists already) and save the plot
# o_folder_data = f"/Users/magaliponds/OneDrive - Vrije Universiteit Brussel/1. VUB/02. Coding/01. IRRMIP/04. Figures/02. OGGM simulations/01. Modelled output/3r_a5/0{v + 1}. {variable_names[v]}/0§. Committed/Test/"
# os.makedirs(o_folder_data, exist_ok=True)
# o_file_name = f"{o_folder_data}/1985_2014.{timeframe}.delta.{variable_names[v]}.comitted.{sample_id}.png"
# plt.savefig(o_file_name, bbox_inches='tight')


# Create DataFrames
df_noirr = pd.DataFrame(values_noirr, columns=[
                        "sample_id", "rgi_id", "volume_1", "marker"])
df_baseline = pd.DataFrame(values_baseline, columns=[
                           "sample_id", "rgi_id", "volume_1", "marker"])
df_noirr_2 = pd.DataFrame(values_noirr_2, columns=[
                          "sample_id", "rgi_id", "volume_2", "marker"])
df_baseline_2 = pd.DataFrame(values_baseline_2, columns=[
                             "sample_id", "rgi_id", "volume_2", "marker"])

# # Merge DataFrames
df_noirr_combined = pd.merge(df_noirr, df_noirr_2, on=[
                             "sample_id", "rgi_id", "marker"])
df_base_combined = pd.merge(df_baseline, df_baseline_2, on=[
                            "sample_id", "rgi_id", "marker"])
# Save combined DataFrame to CSV
output_csv_path = os.path.join(
    wd_path, "log", "comparison_start_end_data_perturbed.csv")
df_noirr_combined.to_csv(output_csv_path, index=False)
output_csv_path = os.path.join(
    wd_path, "log", "comparison_start_end_data_base.csv")
df_base_combined.to_csv(output_csv_path, index=False)


# %% Create nan mask for errorids - outdated

members = [1, 3, 4, 6, 4, 1]
models = ["IPSL-CM6", "E3SM", "CESM2", "CNRM", "NorESM", "W5E5"]

overview_df = pd.DataFrame()

for m, model in enumerate(models):
    for member in range(members[m]):
        if model in ["IPSL-CM6", "W5E5"] or member != 0:
            sample_id = f"{model}.00{member}"
            ds = pd.read_csv(os.path.join(
                # log_dir, f"stats_perturbed_{sample_id}_climate_run_test.csv"))
                log_dir, f"stats_perturbed_{sample_id}_comitted_cst_test.csv"))
            filtered_data = ds[ds['error_msg'].str.contains(
                "Glacier exceeds domain boundaries", na=False)]
            print(len(filtered_data))
            if len(filtered_data) > 0:
                # Add sample_id as a new column
                filtered_data['sample_id'] = sample_id
                # Append filtered data to the overview dataframe
                overview_df = pd.concat(
                    [overview_df, filtered_data], ignore_index=True)
            overview_df = pd.concat(
                [overview_df, filtered_data], ignore_index=True)
            print(overview_df.rgi_id.values)

overview_df = overview_df[['rgi_id', 'error_task', 'error_msg', 'sample_id']]
overview_df.to_csv(os.path.join(wd_path, 'masters',
                   'error_overview_comitted_cst.csv'))

unique_rgi_ids = overview_df['rgi_id'].unique()
unique_rgi_ids = pd.DataFrame(unique_rgi_ids, columns=['rgi_ids'])
unique_rgi_ids.to_csv(os.path.join(
    wd_path, 'masters', 'nan_mask_comitted_cst_test.csv'))


# %% Create run with hydro output plot

subset_gdirs = gdirs_3r_a5[:10]

member_data = []
f, ax = plt.subplots(figsize=(12, 5), sharex=True)
members = [3, 4, 6, 4, 1]  # 1
members_averages = [1, 2, 3, 5, 5]  # 1
models = ["E3SM", "CESM2", "CNRM", "NorESM", "W5E5"]  # "IPSL-CM6",
for m, model in enumerate(models):
    for member in range(members[m]):
        df_tot = pd.DataFrame()
        if model in ["IPSL-CM6", "W5E5"] or member > 0:
            sample_id = f"{model}.00{member}"
            print(sample_id)
            if model == "W5E5":
                cid = 0
                model_type = 'irr'
                file_id = '_hydro_baseline_AllForcings'
                label = "Irr"
            else:
                cid = 1
                if model == "E3SM" and member == 1:
                    label = "NoIrr (individual members)"
                else:
                    label = ""
                model_type = 'noirr'
                file_id = f'_hydro_perturbed_{sample_id}'

            for i, gdir in enumerate(subset_gdirs):
                with xr.open_dataset(gdir.get_filepath('model_diagnostics', filesuffix=file_id)) as ds:
                    # Load the data into a dataframe
                    ds = ds.isel(time=slice(0, -1)).load()

                # Select annual variables
                sel_vars = [
                    v for v in ds.variables if 'month_2d' not in ds[v].dims]
                # And create a dataframe
                df_annual = ds[sel_vars].to_dataframe()
                df_tot = pd.concat([df_tot, df_annual], ignore_index=True)

            # print(df_tot)
            # Select the variables relevant for runoff.
            runoff_vars = ['melt_off_glacier', 'melt_on_glacier',
                           'liq_prcp_off_glacier', 'liq_prcp_on_glacier']

            for i in range(len(runoff_vars)):

                df_monthly_totals = df_tot.groupby(['calendar_year'])[
                    runoff_vars].sum()  # sum over all rgi_ids

           # Convert to mega tonnes instead of kg.
            df_runoff = df_monthly_totals[runoff_vars].clip(0) * 1e-9
            # Sum the variables each year "axis=1", take the 11 year rolling mean and plot it.
            # .rolling(window=11, center=True).mean()
            df_roll = df_runoff.sum(axis=1)

            df_roll.plot(ax=ax, label=label, color=colors[model_type][cid])

            if model != "W5E5":
                member_data.append(df_roll)
stacked_member_data = np.stack(member_data)
mean_values = np.median(stacked_member_data, axis=0).flatten()
# mean_values = np.mean(stacked_member_data, axis=0).flatten()
plt.plot(df_roll.index.values, mean_values,
         label=f"NoIrr ({sum(members_averages)}-member average", color=colors["noirr"][0])
min_values = np.min(stacked_member_data, axis=0).flatten()
max_values = np.max(stacked_member_data, axis=0).flatten()
ax.fill_between(df_roll.index.values, min_values, max_values,
                color=colors["noirr"][1], alpha=0.3, label=f"NoIrr ({sum(members_averages)}-member range)", zorder=16)

plt.ylabel('Annual runoff (Mt)')
plt.xlabel('Year')
plt.legend()
#     wd_path, "masters", f"{rgi_sel}_differences.csv")
# output_df.to_csv(output_csv_path, index=False)
# Calculate difference
# dif = climate_run_output_noirr_clim - climate_run_output_noirr_com

# # Extract data where difference is NaN
# nan_mask = np.isnan(dif)
# rgi_ids_nan = climate_run_output_noirr_clim.rgi_id.where(nan_mask, drop=True)
# climate_clim_nan = climate_run_output_noirr_clim.where(nan_mask, drop=True)
# climate_com_nan = climate_run_output_noirr_com.where(nan_mask, drop=True)

# Store the data in the list
#         for rgi_id, clim_val, com_val in zip(rgi_ids_nan.values, climate_clim_nan.values, climate_com_nan.values):
#             data_rows.append({
#                 "sample_id": sample_id,
#                 "rgi_id": rgi_id,
#                 "dif_value": np.nan,
#                 "noirr_clim_value": clim_val,
#                 "noirr_com_value": com_val
#             })

# # Create a DataFrame from the collected rows
# output_df = pd.DataFrame(data_rows)

# # Save or display the resulting DataFrame
# output_csv_path = os.path.join(wd_path, "masters", "climate_nan_differences.csv")
# output_df.to_csv(output_csv_path, index=False)

# Display the DataFrame to the user
# import ace_tools as tools; tools.display_dataframe_to_user(name="Climate NaN Differences Dataset", dataframe=output_df)
