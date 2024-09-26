#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 15:36:52 2024

@author: magaliponds

This code runs through the following operations:
    
SCOPE: Plot histograms for larger sets, comparing OGGM modelled to Hugonnet data
    Cell 1a: Load the Hugonnet data
    Cell 1b: Load RGI dataset and reformat RGI-notation
    Cell 2: Filter the Hugonnet data according to availability in the RGI   
    Cell 3a: Plot histogram from Hugonnet input data (# glaciers vs B)
    Cell 3b: Plot histogram from Hugonnet input data (# glaciers vs B)
    Cell 4a: Gdirs for the region (based on Hugonnet sample glacier dataset)
    Cell 4b: Save gdirs from OGGM
    Cell 4c: Load gdirs from OGGM (saved in 4b)
    Cell 5: Run the MB model for the glaciers 
    Cell 6: Create a histogram with the MB data
    Cell 7: Analyse glacier statistics
      
"""
# -*- coding: utf-8 -*-import oggm
# from OGGM_data_processing import process_perturbation_data
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
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
function_directory = "/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/src/03. Glacier simulations"
sys.path.append(function_directory)


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

# %% Cell 0: Initialize OGGM with the preferred model parameter set up
folder_path = '/Users/magaliponds/Documents/00. Programming'
wd_path = f'{folder_path}/02. Modelled perturbation-glacier interactions - regional analysis/'
os.makedirs(wd_path, exist_ok=True)
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

cfg.initialize(logging_level='WARNING')
cfg.PARAMS['baseline_climate'] = "GSWP3-W5E5"


cfg.PARAMS['use_multiprocessing'] = False
cfg.PARAMS['store_model_geometry'] = True
y0_clim = 1985
ye_clim = 2014


# %% Cell 1a: Load Hugonnet data - 2000-2020 year average

# Hugonnet data source: Harry via Teams
Hugonnet_path = '/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/04. Reference/01. Climate data/Hugonnet/aggregated_2000_2020/13_mb_glspec.dat'

df_raw = pd.read_csv(Hugonnet_path)


# Convert the pandas DataFrame to an Xarray Dataset
ds = xr.Dataset.from_dataframe(df_raw)
# Load the data variables, as they are listed in the first row
var_index = ds.coords['index'].values
header_str = var_index[1]  # Set the var_index as the header string
data_rows_str = var_index[2:]  # Set the datarows
# # print(header_str, data_rows_str)

# # split data input in such a way that it is loaded as dataframe with columns headers and data in table below
header = header_str.split()

# # Transform the data type from string values to integers for the relevant columns
data_rows = [row.split() for row in data_rows_str]


def str_to_float(value):
    return float(value)


df = pd.DataFrame(data_rows, columns=header)
for col in df.columns:
    if col != 'RGI-ID':  # Exclude column 'B'
        df[col] = df[col].apply(str_to_float)

# # # create a dataset from the transformed data in order to select the required glaciers
df.rename(columns={'RGI-ID': 'rgi_id'})
ds = xr.Dataset.from_dataframe(df)
# # df.to_csv("/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/04. Reference/Hugonnet/aggregated_2000_2020/13_mb_glspec-edited.csv")


# %% Cell 1b: Load RGI dataset and reformat RGI-notation
# downloaded from GLIMS
RGI_data = "/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/01. Input files/03. Shapefile/RGI/General Area/RGI2000-v7.0-C-13_central_asia/RGI2000-v7.0-C-13_central_asia.shp"

rgis = gpd.read_file(RGI_data)
# transform identifier to right format


def transform_identifier(identifier):
    num = int(identifier.split('-')[-1])
    return f'RGI60-13.{str(num).zfill(5)}'


# reformat the rgi_id column such that it is standard annotation
rgis['rgi_id'] = rgis['rgi_id'].apply(transform_identifier)


# %% Cell 2: Filter the Hugonnet data according to availability in the RGI

hugo_filtered = ds.where(ds['RGI-ID'].isin(rgis['rgi_id']), drop=True)


# %% Cell 2b Create master dataset containing all info from Hugonnet and RGI

# reset the indices for both dataset so that they can be merged using xarray
hugo_ds = hugo_filtered.rename({'RGI-ID': 'rgi_id'})
hugo_ds = hugo_ds.to_dataframe()
hugo_ds = hugo_ds.set_index('rgi_id').to_xarray()

rgis_copy = rgis
rgis_copy = rgis_copy.set_index('rgi_id')
rgis_ds = rgis_copy.to_xarray()
rgis_ds['rgi_id_rgis'] = rgis_ds['rgi_id']
hugo_ds['rgi_id_hugo'] = hugo_ds['rgi_id']


# hugo_ds = hugo_ds.reset_index('index', drop=True)

master_ds = xr.merge([rgis_ds, hugo_ds], join='inner')
master_ds['area_dif'] = master_ds['Area']-master_ds['area_km2']

plt.scatter(np.arange(0, len(master_ds.area_dif), 1), master_ds.area_dif)
plt.ylabel("$\Delta$ Area (Hugonnet - RGI)")
plt.xlabel("Glacier index #")
wd_path = '/Users/magaliponds/Documents/00. Programming/02. Modelled perturbation-glacier interactions - regional analysis/'
# master_ds.to_dataframe().to_csv(f"{wd_path}/region13_RGI_master_rgi_hugo.csv")

# %% Cell 3a: Plot histogram from Hugonnet input data (# glaciers A>10km2 vs B)
""" Find glaciers with most postive MB and other conditions"""

# create a copy of hugo_ds, where the non RGI glaciers have been filtered out
hugo_ds_filtered = master_ds
conditions = {
    # 'B': ds_filtered['B'] >= 0,
    # 'errB': ds_filtered['errB'] < 0.2,
    # 'area_km2': hugo_ds_filtered['area_km2'] > 10 #filter for rgis area is larger than 10 km2
    # filter for Hugonnet area is larger than 10 km2
    'Area': hugo_ds_filtered['Area'] > 10
}

for var, condition in conditions.items():
    hugo_ds_filtered = hugo_ds_filtered.where(condition, drop=True)


# create a pandas dataframe of the filtered glaciers in order to sort them on Area
hugo_df_filtered = hugo_ds_filtered.to_dataframe().reset_index()
hugo_df_filtered = hugo_df_filtered.sort_values(by='B', ascending=False)
# df_filtered_subset=df_filtered[1:11]
plt.figure(figsize=(10, 5))
n, bins, patches = plt.hist(hugo_df_filtered['B'], bins=np.linspace(hugo_df_filtered['B'].min(
), hugo_df_filtered['B'].max(), int(len(hugo_df_filtered['B'])/7)), align='left', rwidth=0.8, color='green')
# plt.xlim((0,1.25))

plt.ylabel("# of glaciers [-]")
plt.xlabel("B")

# Color the bins
for patch, bin_edge in zip(patches, bins):
    if bin_edge < 0:
        patch.set_facecolor('red')
    else:
        patch.set_facecolor('green')

plt.xlim((-2, 2))
plt.ylabel("# of glaciers [-]")
plt.xlabel("B (m w.e.)")


def gaussian(x, mu, sigma, A):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))


bin_centers = (bins[:-1] + bins[1:]) / 2

# Fit the Gaussian curve to the histogram data
params_hugo, covariance_hugo = curve_fit(gaussian, bin_centers, n, p0=[np.mean(
    hugo_df_filtered['B']), np.std(hugo_df_filtered['B']), np.max(n)])
x = np.linspace(hugo_df_filtered['B'].min(), hugo_df_filtered['B'].max(), 100)
plt.plot(x, gaussian(x, *params_hugo), color='black',
         label='Fitted Gaussian Hugonnet')
# plt.plot(x, gaussian(x, *params), color='grey', label='Fitted Gaussian OGGM')
# plt.title('Hugonnet derived B')
total_glaciers = len(hugo_df_filtered['B'].values)
plt.annotate(f"Total number of glaciers: {total_glaciers}", xy=(
    0.05, 0.95), xycoords='axes fraction', fontsize=12, verticalalignment='top')
plt.annotate(f"Std: {round(np.std(hugo_df_filtered['B']),2)}", xy=(
    0.05, 0.9), xycoords='axes fraction', fontsize=12, verticalalignment='top')

plt.legend()


wd_path = '/Users/magaliponds/Documents/00. Programming/02. Modelled perturbation-glacier interactions - regional analysis/'
hugo_df_filtered.to_csv(
    f"{wd_path}/region13_RGI_master_rgi_hugo_filtered_area.csv")


# %% Cell 3B: Compare the percentage of Area and volume of the hugo_ds with A>10km2

wd_path = '/Users/magaliponds/Documents/00. Programming/02. Modelled perturbation-glacier interactions - regional analysis/'

hugo_ds_filtered = pd.read_csv(
    f"{wd_path}/region13_RGI_master_rgi_hugo_filtered_area.csv")
hugo_ds = pd.read_csv(
    f"{wd_path}/region13_RGI_master_rgi_hugo_filtered_area.csv")

area_share = hugo_ds_filtered.Area.sum()/hugo_ds.Area.sum()
print(area_share*100)
share = len(hugo_ds_filtered.rgi_id)/len(hugo_ds.rgi_id)
print(share*100)


# %% Cell 4a: ONLY DOWNLOAD ONCE (takes a long time) Get OGGM gdirs for the based on Hugonnet sample glacier dataset

download_gdirs = False

if download_gdirs == True:
    rgi_id_sel = hugo_ds_filtered['rgi_id'].values

    # OGGM options
    oggm.cfg.initialize(logging_level='WARNING')
    oggm.cfg.PATHS['working_dir'] = utils.mkdir(wd_path, reset=False)
    oggm.cfg.PARAMS['store_model_geometry'] = True
    cfg.PARAMS['use_multiprocessing'] = True

    gdirs = []
    with tqdm(total=len(rgi_id_sel), desc="Initializing Glacier Directories", mininterval=1.0) as pbar:
        for rgi_id in rgi_id_sel:
            # Perform your processing here
            gdir = workflow.init_glacier_directories(
                [rgi_id],
                prepro_base_url=DEFAULT_BASE_URL,
                from_prepro_level=4,
                prepro_border=80
            )
            gdirs.extend(gdir)

            # Update tqdm progress bar
            pbar.update(1)

# %% Cell 4b: Save gdirs from OGGM as a pkl file (Save at the end of every working session)

# Save each gdir individually
for gdir in gdirs:
    gdir_path = os.path.join(pkls, f'{gdir.rgi_id}.pkl')
    with open(gdir_path, 'wb') as f:
        pickle.dump(gdir, f)

# %% Cell 4c: Load gdirs from pkl (fastest way to get started)

wd_path_pkls = f'{wd_path}/pkls/'

gdirs = []
for filename in os.listdir(wd_path_pkls):
    if filename.endswith('.pkl'):
        # f'{gdir.rgi_id}.pkl')
        file_path = os.path.join(wd_path_pkls, filename)
        with open(file_path, 'rb') as f:
            gdir = pickle.load(f)
            gdirs.append(gdir)

# # print(gdirs)

# %% Cell 5: Process the Irr climate perturbations for all the gdirs and compile


members = [1, 3, 4, 6, 1]
models = ["IPSL-CM6", "E3SM", "CESM2", "CNRM"]
timeframe = "monthly"

opath_climate = os.path.join(sum_dir, 'climate_historical.nc')
utils.compile_climate_input(
    gdirs, path=opath_climate, filename='climate_historical')

# if you get a long error log saying that "columns" can not be renamed it is often related to multiprocessing
cfg.PARAMS['use_multiprocessing'] = False
for m, model in enumerate(models):
    for member in range(members[m]):

        # Provide the path to the perturbation dataset
        i_folder_ptb = f"/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/03. Output files/01. Climate data/07. OGGM Perturbation input files/{timeframe}/{model}/{member}"
        ds_path = f"{i_folder_ptb}/{model}.00{member}.{y0_clim}_{ye_clim}.{timeframe}.perturbation.input.%.degC.nc"

        # Provide the sample ID to provide the processed pertrubations with the correct output suffix
        sample_id = f"{model}.00{member}"
        print(sample_id)

        workflow.execute_entity_task(process_perturbation_data, gdirs,
                                     ds_path=ds_path,
                                     y0=None, y1=None,
                                     output_filesuffix=f'_perturbation_{sample_id}')

        opath_perturbations = os.path.join(
            sum_dir, f'climate_historical_perturbation_{sample_id}.nc')
        utils.compile_climate_input(gdirs, path=opath_perturbations, filename='climate_historical',
                                    input_filesuffix=f'_perturbation_{sample_id}',
                                    use_compression=True)


# %% Cell 6: Perturb the climate historical with the processed Irr-perturbations, output is gcm file

# gdir = gdirs[0]
for gdir in gdirs:
    # tasks.init_present_time_glacier(gdir)
    with xr.open_dataset(gdir.get_filepath('climate_historical')) as ds:
        ds = ds.load()
    for m, model in enumerate(models_shortlist):
        for member in range(members[m]):
            sample_id = f"{model}.00{member}"
            # make a copy of the historical climate
            clim_ptb = ds.copy()
            # open the perturbation dataset and add the perturbations
            with xr.open_dataset(gdir.get_filepath('climate_historical', filesuffix='_perturbation_{}'.format(sample_id))) as ds_ptb:
                ds_ptb = ds_ptb.load()
            clim_ptb['temp'] = clim_ptb.temp - ds_ptb.temp
            clim_ptb['prcp'] = clim_ptb.prcp - clim_ptb.prcp * ds_ptb.prcp
            clim_ptb = clim_ptb.dropna('time')
            clim_ptb.to_netcdf(gdir.get_filepath(
                'gcm_data', filesuffix='_perturbed_{}'.format(sample_id)))


# %% Cell 7: Run Mass Balance model for the glaciers (V = A*B)

for m, model in enumerate(models):
    for member in range(members[m]):

        sample_id = f"{model}.00{member}"
        mb_ts_mean = []
        mb_ts_all = []
        mb_ts_mean_ext = []
        mb_ts_all_ext = []
        error_ids = []
        count = 0

        for (g, gdir) in enumerate(gdirs):
            try:
                count += 1
                # print(f"Processing {count} of {len(gdirs)}")
                # , filesuffix='_dyn_melt_f_calib')
                fls = gdir.read_pickle('model_flowlines')
                years = np.arange(1985, 2014)

                if model == "W5E5":
                    # extend range to see match w. geodetic mass balance
                    years_ext = np.arange(2000, 2020)
                    # Get the calibrated mass-balance model - the default is to use OGGM's "MonthlyTIModel" and compute specific mb
                    mbmod = massbalance.MonthlyTIModel(
                        gdir, filename='climate_historical')
                    mb_ts = mbmod.get_specific_mb(fls=fls, year=years)
                    mb_ts_ext = mbmod.get_specific_mb(fls=fls, year=years_ext)
                else:
                    mbmod = massbalance.MonthlyTIModel(
                        gdir, filename='gcm_data', input_filesuffix='_perturbed_{}'.format(sample_id))
                    mb_ts = mbmod.get_specific_mb(fls=fls, year=years)

                # Append all time series data to mb_ts_all
                for year, mb in zip(years, mb_ts):
                    mb_ts_all.append((gdir.rgi_id, year, mb))
                for year, mb in zip(years_ext, mb_ts_ext):
                    mb_ts_all_ext.append((gdir.rgi_id, year, mb))

            except Exception as e:
                # Handle the error and continue
                print(
                    f"Error processing {gdir.rgi_id} with model {model} and member {member}: {e}")
                # found error: RGI60-13.36875 no flowlines --> 542 to 541 glaciers in selected gdirs
                error_ids.append((gdir.rgi_id, model, member))
                continue
            mean_mb = np.mean(mb_ts)
            mb_ts_mean.append((gdir.rgi_id, mean_mb))

            if model == "W5E5":
                mean_mb_ext = np.mean(mb_ts_ext)
                mb_ts_mean_ext.append((gdir.rgi_id, mean_mb_ext))

        mb_df_mean = pd.DataFrame(mb_ts_mean, columns=['rgi_id', 'B'])
        mb_df_mean.to_csv(os.path.join(
            sum_dir, f'specific_massbalance_mean_{sample_id}.csv'), index=False)
        # Create a DataFrame from the collected data
        mb_ts_df = pd.DataFrame(
            mb_ts_all, columns=['rgi_id', 'Year', 'Mass_Balance'])
        # Save to a single CSV file
        mb_ts_df.to_csv(os.path.join(
            sum_dir, f'specific_massbalance_timeseries_{sample_id}.csv'), index=False)

        if model == "W5E5":
            mb_df_mean_ext = pd.DataFrame(
                mb_ts_mean_ext, columns=['rgi_id', 'B'])
            mb_df_mean_ext.to_csv(os.path.join(
                sum_dir, f'specific_massbalance_mean_extended_{sample_id}.csv'), index=False)
            mb_ts_df_ext = pd.DataFrame(
                mb_ts_all_ext, columns=['rgi_id', 'Year', 'Mass_Balance'])
            mb_ts_df_ext.to_csv(os.path.join(
                sum_dir, f'specific_massbalance_timeseries_extended_{sample_id}.csv'), index=False)

        # Optionally save the list of error cases to a CSV for later review
        if error_ids:
            error_df = pd.DataFrame(
                error_ids, columns=['rgi_id', 'Model', 'Member'])
            error_df.to_csv(os.path.join(
                log_dir, 'Error_Log.csv'), index=False)

# %% Cell 80: Update the master dataset with B, A, RGI ID and location for all the different sample ids (541x15=8115)

data = []
rgi_ids = []
labels = []


# Iterate over models and members, collecting data for boxplots
# only take the model shortlist as members are handled separately
for m, model in enumerate(models_shortlist):
    for member in range(members[m]):
        sample_id = f"{model}.0{member:02d}"  # Ensure leading zeros
        i_path = os.path.join(
            sum_dir, f'specific_massbalance_mean_{sample_id}.csv')

        # Load the CSV file into a DataFrame and convert to xarray
        mb = pd.read_csv(i_path, index_col=0).to_xarray()

        # Collect B values for each model and member
        data.append(mb.B.values)

        # Store RGI IDs only for the first model/member
        if m == 0 and member == 0:
            rgi_ids.append(mb.rgi_id.values)

        labels.append(sample_id)

i_path_base = os.path.join(sum_dir, 'specific_massbalance_mean_W5E5.000.csv')
mb_base = pd.read_csv(i_path_base, index_col=0)
base_array = np.array(mb_base)

# Convert the list of data into a NumPy array and transpose it
data_array = np.array(data)
# Shape: (number of B values, number of models * members)
reshaped_data = data_array.T
# Create a DataFrame for the reshaped data
df = pd.DataFrame(reshaped_data,  index=rgi_ids, columns=np.repeat(labels, 1))
df['B_irr'] = base_array

df.rename_axis("rgi_id", inplace=True)
df.reset_index(drop=False, inplace=True)

# Step 1: Melt the DataFrame to get B_noirr values
df_melted = pd.melt(df, id_vars='rgi_id', value_vars=[col for col in df.columns if col != 'B_irr'],
                    var_name='sample_id', value_name='B_noirr')

# Step 2: Create a DataFrame with repeated B_irr values
# Keep only the rgi_id and B_irr columns
b_irr_repeated = df[['rgi_id', 'B_irr']].copy()
b_irr_repeated = b_irr_repeated.merge(
    df_melted[['rgi_id', 'sample_id']], on='rgi_id')  # Ensure all combinations

# Now merge B_irr with melted DataFrame
df_complete = pd.merge(df_melted, b_irr_repeated, on=['rgi_id', 'sample_id'])
df_complete['B_delta'] = df_complete.B_irr-df_complete.B_noirr

# Merge with rgis_complete
master_df = master_ds.to_dataframe()
df_complete = pd.merge(df_complete, master_df, on='rgi_id', how='inner')

# reorder the dataset
df_complete = df_complete[[
    'rgi_id', 'rgi_id_rgis', 'rgi_id_hugo', 'sample_id',
    'B_noirr', 'B_irr', 'B_delta', 'B',
    'errB', 'cenlon', 'cenlat', 'lon',
    'lat', 'area_km2', 'Area', 'area_dif'
]]


df_complete.to_csv(
    f"{wd_path}/region13_RGI_master_rgi_hugo_filtered_area_B.csv")
# Display the final DataFrame
print(df_complete)


# %% Cell 8a plot mass balance data in histograms and gaussians (subplots and one plot for all member options, boxplot and gaussian only option)

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
subplots = False
alpha_set = 0.8


# Initialize plot
if subplots:
    fig, axes = plt.subplots(2, 2, figsize=(15, 8), sharex=True, sharey=True)
    axes = axes.flatten()
else:
    fig, axes = plt.subplots(1, 1, figsize=(10, 6), sharex=True, sharey=True)

# Load baseline data
i_path_base = os.path.join(sum_dir, 'specific_massbalance_mean_W5E5.000.csv')
mb_base = pd.read_csv(i_path_base, index_col=0).to_xarray()
bin_size = np.linspace(mb_base.B.min(), mb_base.B.max(),
                       int(len(mb_base.B) / 7))

# Plot baseline once if not using subplots
baseline_plotted = False
if not subplots:
    plot_gaussian(axes, mb_base, bin_size, "black", "W5E5.000",
                  zorder=1, gaussian_only=gaussian_only)
    baseline_plotted = True

# Iterate over models and members
for m, model in enumerate(models_shortlist):  # only perturbed models
    for member in range(members[m]):
        if member == 0:
            linestyle = '-'
        else:
            linestyle = ":"
        if subplots:
            ax = axes[m]
        else:
            ax = axes  # Single plot case

        sample_id = f"{model}.00{member}"
        i_path = os.path.join(
            sum_dir, f'specific_massbalance_mean_{sample_id}.csv')
        mb = pd.read_csv(i_path, index_col=0).to_xarray()

        # Plot baseline only for subplots or first member if not already plotted
        if subplots and member == 0:
            plot_gaussian(ax, mb_base, bin_size, "grey", "W5E5.000",
                          zorder=1, gaussian_only=gaussian_only)
            if not subplots:
                baseline_plotted = True

        # Plot ensemble member
        plot_gaussian(ax, mb, bin_size, colors[model][member], sample_id,
                      zorder=members[m] + member + 1, linestyle=linestyle, gaussian_only=gaussian_only)

        total_glaciers = len(mb.B.values)

    # Add annotations and labels
    if m == 3:
        ax.annotate(f"Total # of glaciers: {total_glaciers}", xy=(
            0.65, 0.98), xycoords='axes fraction', fontsize=10, verticalalignment='top')
        ax.annotate("Time period: 1985-2014", xy=(0.65, 0.92),
                    xycoords='axes fraction', fontsize=10, verticalalignment='top')

    if m in [0, 2]:
        ax.set_ylabel("# of glaciers [-]")
    if m in [2, 3]:
        ax.set_xlabel("Mean specific mass balance [mm w.e. yr$^{-1}$]")

    ax.legend(loc="upper left", bbox_to_anchor=(0, 1))
    ax.set_xlim(-1250, 1000)
    ax.axvline(0, color='k', linestyle="dashed", lw=1, zorder=20)

plt.tight_layout()
plt.show()


# %% Cell 8B: Plot the specific mass balances using boxplots

# Configuration variables
models = ["E3SM", "CESM2", "CNRM", "IPSL-CM6"]
members = [3, 4, 6, 1]  # Number of members for each model

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
alpha_set = 0.8

# Initialize plot
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Load baseline data
i_path_base = os.path.join(sum_dir, 'specific_massbalance_mean_W5E5.000.csv')
mb_base = pd.read_csv(i_path_base, index_col=0).to_xarray()

# Data structure for boxplot
data = []
labels = []
colors_list = []  # Store colors corresponding to each member for correct ordering

# Iterate over models and members, collect data for boxplots
for m, model in enumerate(models):
    for member in range(members[m]):
        sample_id = f"{model}.0{member:02d}"  # Ensure leading zeros
        i_path = os.path.join(
            sum_dir, f'specific_massbalance_mean_{sample_id}.csv')
        mb = pd.read_csv(i_path, index_col=0).to_xarray()

        # Collect B values for each model and member
        data.append(mb.B.values)
        labels.append(sample_id)
        # Store the color for this member
        colors_list.append(colors[model][member])


# Adding baseline data for W5E5
data.append(mb_base.B.values)
labels.append("W5E5.000")
colors_list.append('darkgrey')  # Assign W5E5 a default color (darkgrey)

# Reverse the order of the data, labels, and colors except for "W5E5.000"
data = data[:-1][::-1] + [data[-1]]
labels = labels[:-1][::-1] + [labels[-1]]
colors_list = colors_list[:-1][::-1] + [colors_list[-1]]

# Plotting the main boxplot
box = ax.boxplot(data, patch_artist=True, labels=labels,
                 vert=False, boxprops=dict(edgecolor='none'))
# plt.show()
# Color the boxes
for patch, color in zip(box['boxes'], colors_list):
    # rgba_color = clrs.to_rgba(color, alpha=0.5) #option to change alpha of face color, potential to use with colored median bar
    patch.set_facecolor(color)


# Change the color of the median line to black
for median, color in zip(box['medians'], colors_list):
    median.set_color('black')  # option to change to colors from color list

# Plot the W5E5 dashed outline on top of all other boxplots
# This is achieved by plotting only the W5E5 data again, but with dashed lines
box_w5e5 = ax.boxplot([mb_base.B.values] * len(data), patch_artist=False, vert=False,
                      positions=np.arange(len(data), 0, -1),
                      boxprops=dict(linestyle='--', color='black'),
                      whiskerprops=dict(linestyle='--', color='black'),
                      capprops=dict(linestyle='--', color='black'),
                      showfliers=False)

# Set the median lines for the W5E5 boxplot to dashed black
for median in box_w5e5['medians']:
    median.set_linestyle('--')
    median.set_color('black')

# Identify the index of the W5E5 label (assuming it's the last one)
y_labels = ax.get_yticklabels()
y_labels = y_labels[:15]  # Set the W5E5 label to an empty string
y_ticks = ax.get_yticks()
y_ticks = y_ticks[:15]  # Set the W5E5 label to an empty string
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels)

# Set labels and titlei have over
ax.set_xlabel("Spread (Mean specific mass balance [mm w.e. yr$^{-1}$])")
ax.set_ylabel("Models and Members")
# Reduce font size for y-axis labels if needed
ax.set_xlim(-1250, 1000)

# Display the plot
plt.tight_layout(pad=2.0)
plt.show()


# %% Cell 8C: Plot the specific mass balances using boxplots, different subsets

# Configuration variables
models = ["E3SM", "CESM2", "CNRM", "IPSL-CM6"]
members = [3, 4, 6, 1]  # Number of members for each model

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
alpha_set = 0.8

# Initialize plot
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Load baseline data
i_path_base = os.path.join(sum_dir, 'specific_massbalance_mean_W5E5.000.csv')
mb_base = pd.read_csv(i_path_base, index_col=0).to_xarray()

ds = pd.read_csv(f"{wd_path}/region13_RGI_B_master.csv", index_col=0, header=0)
mean_noirr = ds.groupby('rgi_id')['B_noirr'].mean().reset_index()
mean_irr = ds.groupby('rgi_id')['B_irr'].mean().reset_index()

# make a subset for all the glaciers with a specific area
mean_noirr_area = ds.where(ds.area_km2 > 1).groupby('rgi_id')[
    'B_noirr'].mean().reset_index()
mean_noirr_delta_all = ds.groupby('rgi_id')[['B_noirr', 'B_delta']].mean(
).reset_index()
mean_noirr_delta = mean_noirr_delta_all.nlargest(54, 'B_delta')

# make a ssubset for the glaciers with the largest change
labels = ["NoIrr - 14 member avg: largest 10% $\Delta$"] + \
    ["NoIrr - 14 member avg: A>1km$^2$"] + \
    ["NoIrr - 14 member avg: all"] + ["Irr (W5E5.000)"]
colors_list = ["orange"] + ["lightgreen"] + ["lightblue"] + ["darkgrey"]

# # Plotting the main boxplot
box = ax.boxplot([mean_noirr_delta.B_noirr.values, mean_noirr_area.B_noirr.values, mean_noirr.B_noirr.values, mean_irr.B_irr.values], patch_artist=True, labels=labels,
                 vert=False, boxprops=dict(edgecolor='none'))
# # plt.show()
# # Color the boxes
for patch, color in zip(box['boxes'], colors_list):
    # rgba_color = clrs.to_rgba(color, alpha=0.5) #option to change alpha of face color, potential to use with colored median bar
    patch.set_facecolor(color)


# # Change the color of the median line to black
for median, color in zip(box['medians'], colors_list):
    median.set_color('black')  # option to change to colors from color list

# # Plot the W5E5 dashed outline on top of all other boxplots
# # This is achieved by plotting only the W5E5 data again, but with dashed lines
box_w5e5 = ax.boxplot([mean_irr.B_irr.values]*len(labels), patch_artist=False, vert=False,
                      positions=np.arange(len(labels), 0, -1),
                      boxprops=dict(linestyle='--', color='black'),
                      whiskerprops=dict(linestyle='--', color='black'),
                      capprops=dict(linestyle='--', color='black'),
                      showfliers=False)

# # Set the median lines for the W5E5 boxplot to dashed black
for median in box_w5e5['medians']:
    median.set_linestyle('--')
    median.set_color('black')

# # Identify the index of the W5E5 label (assuming it's the last one)
y_labels = ax.get_yticklabels()
y_labels = y_labels[:4]  # Set the W5E5 label to an empty string
y_ticks = ax.get_yticks()
y_ticks = y_ticks[:4]  # Set the W5E5 label to an empty string
# Wrap labels using textwrap
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels)

# # Set labels and titlei have over
# ax.set_xlabel("Spread (Mean specific mass balance [mm w.e. yr$^{-1}$])")
# ax.set_ylabel("Models and Members")
# # Reduce font size for y-axis labels if needed
# ax.set_xlim(-1250, 1000)

# # Display the plot
# plt.tight_layout(pad=2.0)
# plt.show()

# %% Cell 8D: Link the largest change glaciers in a plot

# start from subset and match these to the cenlon, cenlats from the rgis
rgis_complete = rgis.drop(columns=['rgi_id']).rename(
    columns={'rgi_id_format': 'rgi_id'})
df_delta = pd.merge(mean_noirr_delta_all, rgis_complete,
                    on='rgi_id', how='inner')
# df_delta = pd.merge(mean_noirr_delta, rgis_complete, on='rgi_id', how='inner') #for only 10% changing largest glaciers

df_irr = pd.merge(mean_irr, rgis_complete, on='rgi_id', how='inner')

datasets = [df_delta, df_irr]

shapefile_path = '/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/01. Input files/03. Shapefile/Karakoram/Pan-Tibetan Highlands/Pan-Tibetan Highlands (Liu et al._2022)/Shapefile/Pan-Tibetan Highlands (Liu et al._2022)_P.shp'
shp = gpd.read_file(shapefile_path)
target_crs = 'EPSG:4326'
shp = shp.to_crs(target_crs)


fig, ax = plt.subplots(figsize=(10, 7), subplot_kw={
                       'projection': ccrs.PlateCarree()})
ax.set_extent([60, 110, 20, 50], crs=ccrs.PlateCarree())

# Add country borders
ax.add_feature(cfeature.BORDERS, linestyle='solid', color='grey')
ax.add_feature(cfeature.COASTLINE)

# Include HMA shapefile
shp.plot(ax=ax, edgecolor='red', linewidth=0, facecolor='bisque')

# Create a custom diverging colormap: red for negative, white for zero, blue for positive
colors = [(1, 0, 0), (1, 1, 1), (0, 0, 1)]  # Red -> White -> Blue
custom_cmap = LinearSegmentedColormap.from_list(
    'red_white_blue', colors, N=256)

# Normalize with zero centered (TwoSlopeNorm sets the midpoint to zero)
norm = TwoSlopeNorm(vmin=df_delta['B_delta'].min(
), vcenter=0, vmax=df_delta['B_delta'].max())

# Add scatter plot for the locations
sc = ax.scatter(df_delta['cenlon'], df_delta['cenlat'], s=df_delta['area_km2']*20,
                c=df_delta['B_delta'], cmap=custom_cmap, norm=norm, edgecolor='k', alpha=0.7)

# Add a colorbar for the B_delta values
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('$\Delta$ B (Irr - NoIrr)')


# Create a size legend manually
# Choose example areas to represent in the legend (you can adjust these)
area_legend_sizes = [0.05, 0.2, 5]  # These are \in km^2
markers = [plt.scatter([], [], s=size * 20, edgecolor='k', facecolor='none', label=f'{size} km²')
           for size in area_legend_sizes]

# Add the size legend
size_legend = ax.legend(handles=markers, title='Area (km²)', loc='upper left')

# Add the size legend to the plot
ax.add_artist(size_legend)

# Set plot labels
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# plt.title('Locations, Size and $\Delta$ of Glaciers Experiencing the Top 10% Most Change')
plt.title('Locations, Size and $\Delta$ of Glaciers in RGI reg. 13 (A>10km$^2$)')


plt.show()


# %% Cell 9: Run the climate model - Save pkl after running is done , as running takes quite a while

cfg.PARAMS['continue_on_error'] = True

# gdir = gdirs[0]
y0_clim = 1985
ye_clim = 2014

members = [6]  # 1, 3, 4, 6, 1]
models = ["CNRM"]  # "IPSL-CM6", "E3SM", "CESM2", "CNRM"]
for m, model in enumerate(models):
    for member in range(members[m]):
        sample_id = f"{model}.00{member}"
        workflow.execute_entity_task(tasks.init_present_time_glacier, gdirs)
        out_id = f'_perturbed_{sample_id}'
        opath = os.path.join(
            sum_dir, f'climate_run_output_perturbed_{sample_id}.nc')
        workflow.execute_entity_task(tasks.run_from_climate_data, gdirs,
                                     ys=y0_clim, ye=ye_clim,  # min_ys=None,
                                     max_ys=None, fixed_geometry_spinup_yr=None,
                                     store_monthly_step=False, store_model_geometry=None,
                                     store_fl_diagnostics=True, climate_filename='gcm_data',
                                     # mb_model=None, mb_model_class=<class 'oggm.core.massbalance.MonthlyTIModel'>,
                                     climate_input_filesuffix='_perturbed_{}'.format(
                                         sample_id),
                                     output_filesuffix=out_id,
                                     # init_model_filesuffix='_historical', #in case you want to start from a previous model state
                                     # init_model_yr=2015, init_model_fls=None,
                                     zero_initial_glacier=False, bias=0,
                                     temperature_bias=None, precipitation_factor=None)

        ds_ptb = utils.compile_run_output(
            gdirs, input_filesuffix=out_id, path=opath)  # compile the run output


# And run the climate model with reference data
workflow.execute_entity_task(tasks.run_from_climate_data, gdirs,
                             ys=y0_clim, ye=ye_clim,
                             output_filesuffix='_baseline_W5E5.000')
opath_base = os.path.join(sum_dir, 'climate_run_output_baseline_W5E5.000.nc')
ds_base = utils.compile_run_output(
    gdirs, input_filesuffix='_baseline_W5E5.000', path=opath_base)

# %% Cell 10: Create output plots for Area and volume


# def plot_volume_area_evolution_over_time_combined(averaged):

# timeframe = "monthly"
# members = [3, 4, 6, 1, 1]
models_test = ["E3SM", "CESM2",  "IPSL-CM6"]
variables = ["volume", "area"]
factors = [10**-9, 10**-6]

variable_names = ["Volume", "Area"]
variable_axes = ["Volume [km$^3$]", "Area [km$^2$]"]


# repeat the loop for area and volume
for v, var in enumerate(variables):
    print(var)
    # create a new plot for both area and volume
    fig, ax = plt.subplots(figsize=(10, 6))

    # create a timeseries for all the models
    member_data = []

    baseline_path = os.path.join(
        wd_path, "summary", f"climate_run_output_baseline_W5E5.000.nc")
    baseline = xr.open_dataset(baseline_path)
    ax.plot(baseline["time"], baseline[var].sum(dim="rgi_id") *
            factors[v], label="W5E5.000", color=colors["W5E5"][0], linewidth=2, zorder=15)

    for m, model in enumerate(models_shortlist):
        # if averaged == True:
        #     # if averaged only include the sample ids on ending 000
        #     members = [1, 1, 1, 1]
        # else:
        # if averaged only include the sample ids on ending 000
        # members_test = [2, 3, 1]  # , 5]
        for i in range(members_averages[m]):

            if members_averages[m] > 1:
                i += 1

            sample_id = f"{model}.00{i}"
            print(sample_id)

            climate_run_opath = os.path.join(
                sum_dir, f'climate_run_output_perturbed_{sample_id}.nc')
            # open the dataset with the climate run output from OGGM, using all the gdirs
            climate_run_output = xr.open_dataset(climate_run_opath)

            ax.plot(climate_run_output["time"], climate_run_output[var].sum(dim="rgi_id") * factors[v],
                    label=sample_id, color=colors[model][i], linewidth=2, linestyle="dotted")

            # total reg 13 change volume over time by member
            member_data.append(climate_run_output[var].sum(
                dim="rgi_id").values * factors[v])
            print(climate_run_output[var].sum(
                dim="rgi_id").values[1] * factors[v])

    if len(member_data) > 0:
        stacked_member_data = np.stack(member_data)

        # average over the 10 members
        mean_values = np.mean(stacked_member_data, axis=0).flatten()
        ax.plot(climate_run_output["time"].values, mean_values,
                color="black", linestyle='dashed', lw=2, label="14-member average")
        min_values = np.min(stacked_member_data, axis=0).flatten()
        max_values = np.max(stacked_member_data, axis=0).flatten()
        ax.fill_between(climate_run_output["time"].values, min_values, max_values,
                        color="lightblue", alpha=0.3, label=f"14-member range", zorder=16)

    # Set labels and title for the combined plot
    ax.set_ylabel(variable_axes[v])
    ax.set_xlabel("Time [year]")
    ax.set_title(f"Summed {variable_names[v]}, RGI 13, A >10 km$^2$")

    # Adjust the legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
               bbox_to_anchor=(0.5, -0.15), ncol=4)
    plt.tight_layout()

    o_folder_data = f"/Users/magaliponds/OneDrive - Vrije Universiteit Brussel/1. VUB/02. Coding/01. IRRMIP/04. Figures/02. OGGM simulations/01. Modelled output/0{v + 1}. {variable_names[v]}/00. Combined"
    os.makedirs(o_folder_data, exist_ok=True)
    o_file_name = f"{o_folder_data}/1985_2014.{timeframe}.delta.{variable_names[v]}.combined.png"
    plt.savefig(o_file_name, bbox_inches='tight')

    # return


# Call the function with appropriate parameters
# plot_volume_area_evolution_over_time_combined(False)


# %% Load the hugonnet data from oggm

rgi_id_sel = hugo_ds_filtered['RGI-ID'].values
folder_path = '/Users/magaliponds/Documents/00. Programming'
wd_path = f'{folder_path}/02. Modelled perturbation-glacier interactions - regional analysis/'

# OGGM options
oggm.cfg.initialize(logging_level='WARNING')
oggm.cfg.PARAMS['store_model_geometry'] = False
cfg.PARAMS['use_multiprocessing'] = False
rgi_ids = rgis['rgi_id_format']

# members = [1, 1, 1, 1, 1]
mb_geo = utils.get_geodetic_mb_dataframe()
mb_geo = xr.Dataset.from_dataframe(mb_geo)

# filter all the glacier ids that are larger than 10 km2 and have the right rgi_id format
conditions = {'area': mb_geo['area'] > 10}
for var, condition in conditions.items():
    mb_geo = mb_geo.where(condition, drop=True)

mb_geo = mb_geo.sel(mb_geo.rgiid.isin(rgis['rgi_id_format']), drop=True)
# check the length of this file

# %% Compare W5E5 to geodetic distribution

# mb_geo['B'] = mb_geo.B*1000

# mb_geo = hugo_ds


# i_path_base = os.path.join(
#     sum_dir, 'specific_massbalance_mean_extended_W5E5.000.csv')
# mb_base = pd.read_csv(i_path_base, index_col=0).to_xarray()
# bin_size = np.linspace(mb_geo.B.min(), mb_geo.B.max(), int(len(mb_geo.B)/10))


# # fig,axes=plt.subplots(1,1, figsize=(15,8))
# plt.figure(figsize=(10, 6))
# alpha_set = 0.8

# # bin_size = np.linspace(mb_base.B.min(), mb_base.B.max(), int(len(mb_base.B)/7))
# n, bins, patches = plt.hist(mb_base.B, bins=bin_size, align='left',
#                             rwidth=0.8, facecolor="grey", alpha=0.3, edgecolor="none", zorder=2, label="W5E5.000 modelled Mass Balance")

# for b, patch in zip(bins, patches):
#     if b < 0:
#         # Set color to red if count is greater than zero
#         patch.set_facecolor('red')
#     else:
#         patch.set_facecolor('green')

# n_geo, bins_geo, patches_geo = plt.hist(mb_geo.B, bins=bin_size, align='left', rwidth=0.8,
#                                         facecolor="none", alpha=1, edgecolor="k", zorder=1, label="Geodetic Mass Balance")


# bin_centers = (bins[:-1] + bins[1:]) / 2
# params, covariance = curve_fit(gaussian, bin_centers, n, p0=[
#                                np.mean(mb_base.B), np.std(mb_base.B), np.max(n)])
# params_geo, covariance_geo = curve_fit(gaussian, bin_centers, n, p0=[
#                                        np.mean(mb_geo.B), np.std(mb_geo.B), np.max(n)])

# x = np.linspace(mb_base.B.min(), mb_base.B.max(), 100)
# x_geo = np.linspace(mb_geo.B.min(), mb_geo.B.max(), 100)
# plt.plot(x, gaussian(x, *params),
#          color=colors["W5E5"][0], label="Gaussian fit W5E5.000", zorder=(members[m]+1))
# plt.plot(x, gaussian(x_geo, *params_geo),
#          color=colors["W5E5"][0], label="Gaussian fit Geodetic MB", zorder=(members[m]+1), linestyle="dashed")

# plt.legend(loc="upper left", bbox_to_anchor=(0, 1), ncols=1)
# plt.xlabel("Mean specific mass balance[mm w.e. yr$^{-1}$]")
# plt.ylabel("# of glaciers [-]")
# plt.axvline(0, color='k', linestyle="dotted")
# plt.annotate(f"Total # of glaciers: {total_glaciers}", xy=(
#     0.01, 0.74), xycoords='axes fraction', fontsize=10, verticalalignment='top')
# plt.annotate(f"Time period = 2000-2020", xy=(
#     0.01, 0.70), xycoords='axes fraction', fontsize=10, verticalalignment='top')

# plt.show()
# %% Make scatter plot with RGI data
members = [1, 3, 4, 6, 1, 1]
# members = [1, 1, 1, 1, 1]

# %% Correlation curve

members = [1, 1, 1, 1]
models = ["W5E5", "E3SM", "CESM2", "CNRM"]  # , "IPSL-CM6"]

plt.figure(figsize=(8, 4))

# plt.scatter(mb_base.rgi_id, mb_base.B,color=colors["W5E5"][0])
for m, model in enumerate(models):
    for member in range(members[m]):
        print(model)

        sample_id = f"{model}.00{member}"

        if model == "W5E5":
            i_path_base = os.path.join(
                sum_dir, 'specific_massbalance_mean_W5E5.000.csv')
            mb = pd.read_csv(i_path_base, index_col=0).to_xarray()
        else:
            i_path = os.path.join(
                sum_dir, f'specific_massbalance_mean_{sample_id}.csv')
            mb = pd.read_csv(i_path, index_col=0).to_xarray()

        mb['B'].attrs['units'] = "m w.e. yr-1"
        mb['B'].attrs['standard_name'] = "Mean specific Mass balance 2000-2020"

        hugo_ds = hugo_ds_filtered[['RGI-ID', 'B']].set_index(index='RGI-ID')
        hugo_ds = hugo_ds.rename({'index': 'rgi_id'})

        hugo_ds = hugo_ds.where(hugo_ds.rgi_id.isin(mb.rgi_id), drop=True)
        hugo_ds['B'] = hugo_ds.B*1000

        hugo_df = hugo_ds.to_dataframe()
        mb_df = mb.to_dataframe()
        cor_df = pd.merge(mb_df, hugo_df, on='rgi_id',
                          suffixes=('_oggm', '_hugo'))

        cor_ds = cor_df.to_xarray()
        correlation = xr.corr(cor_ds['B_oggm'], cor_ds['B_hugo']).round(2)
        rmse = np.sqrt(np.mean((cor_ds['B_oggm'] - cor_ds['B_hugo']) ** 2))

        # cor_ds = xr.merge([mb_ds, hugo_ds], dim='rgi_id')
        if m == 0 and member == 0:
            plt.plot(cor_ds['B_hugo'], cor_ds['B_hugo'], color='k',
                     label="correlation: 1", zorder=(sum(members)+1))

        plt.scatter(cor_ds['B_hugo'], cor_ds['B_oggm'], s=5, color=colors[model]
                    [member], label=f'{sample_id}, correlation: {correlation.values}, rmse: {np.round(rmse.values,2)}', alpha=0.5)

        # plt.text(-1.4,0.2, f'Correlation coefficient: {correlation.values}')

        plt.ylabel('Modelled mass balance  (mm w.e. yr$^{-1}$)')
        plt.xlabel('Geodetic mass balance (mm w.e. yr$^{-1}$)')


plt.legend()

# plt.scatter(mb_base.rgi_id, mb_base.B, color=colors[model][member])

plt.show()

# %% Extended W5E% to geodetic

# %% Correlation curve

members = [1, 1, 1, 1]
models = ["W5E5"]  # , "E3SM", "CESM2", "CNRM"]  # , "IPSL-CM6"]

plt.figure(figsize=(7, 4))

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

        mb['B'].attrs['units'] = "m w.e. yr-1"
        mb['B'].attrs['standard_name'] = "Mean specific Mass balance 2000-2020"

        hugo_ds = hugo_ds_filtered[['RGI-ID', 'B']].set_index(index='RGI-ID')
        hugo_ds = hugo_ds.rename({'index': 'rgi_id'})

        hugo_ds = hugo_ds.where(hugo_ds.rgi_id.isin(mb.rgi_id), drop=True)
        hugo_ds['B'] = hugo_ds.B*1000

        hugo_df = hugo_ds.to_dataframe()
        mb_df = mb.to_dataframe()
        cor_df = pd.merge(mb_df, hugo_df, on='rgi_id',
                          suffixes=('_oggm', '_hugo'))

        cor_ds = cor_df.to_xarray()
        correlation = xr.corr(cor_ds['B_oggm'], cor_ds['B_hugo']).round(2)

        # cor_ds = xr.merge([mb_ds, hugo_ds], dim='rgi_id')
        if m == 0 and member == 0:
            plt.plot(cor_ds['B_hugo'], cor_ds['B_hugo'], color='k',
                     label="correlation: 1", zorder=(sum(members)+1))

        plt.scatter(cor_ds['B_hugo'], cor_ds['B_oggm'], s=5, color=colors[model]
                    [member], label=f'{sample_id}, correlation: {correlation.values}', alpha=0.5)

        # plt.text(-1.4,0.2, f'Correlation coefficient: {correlation.values}')

        plt.ylabel('Modelled mass balance  (m w.e. yr$^{-1}$)')
        plt.xlabel('Geodetic mass balance (m w.e. yr$^{-1}$)')
plt.legend()

# plt.scatter(mb_base.rgi_id, mb_base.B, color=colors[model][member])

plt.show()
#
