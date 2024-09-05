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
from OGGM_data_processing import process_perturbation_data
import oggm
from oggm import utils, cfg, workflow, tasks, DEFAULT_BASE_URL, graphics, global_tasks
from oggm.core import massbalance, flowline
from oggm.utils import floatyear_to_date, hydrodate_to_calendardate
from oggm.sandbox import distribute_2d
from oggm.sandbox.edu import run_constant_climate_with_bias
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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

# %% Cell 1a: Load Hugonnet data
Hugonnet_path = '/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/04. Reference/01. Climate data/Hugonnet/aggregated_2000_2020/13_mb_glspec.dat'

df = pd.read_csv(Hugonnet_path, delimiter=',')

# Convert the pandas DataFrame to an Xarray Dataset
ds = xr.Dataset.from_dataframe(df)
var_index = ds.coords['index'].values
header_str = var_index[1]
data_rows_str = var_index[2:]
# print(header_str, data_rows_str)

# split data input in such a way that it is loaded as dataframe with columns headers and data in table below
header = header_str.split()

# Transform the data type from string values to integers for the relevant columns
data_rows = [row.split() for row in data_rows_str]


def str_to_float(value):
    return float(value)


df = pd.DataFrame(data_rows, columns=header)
for col in df.columns:
    if col != 'RGI-ID':  # Exclude column 'B'
        df[col] = df[col].apply(str_to_float)

# create a dataset from the transformed data in order to select the required glaciers
ds = xr.Dataset.from_dataframe(df)
# df.to_csv("/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/04. Reference/Hugonnet/aggregated_2000_2020/13_mb_glspec-edited.csv")

# %% Cell 1b: Load RGI dataset and reformat RGI-notation
RGI_data = "/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/01. Input files/03. Shapefile/RGI/General Area/RGI2000-v7.0-C-13_central_asia/RGI2000-v7.0-C-13_central_asia.shp"

rgis = gpd.read_file(RGI_data)
# transform identifier to right format


def transform_identifier(identifier):
    num = int(identifier.split('-')[-1])
    return f'RGI60-13.{str(num).zfill(5)}'


rgis['rgi_id_format'] = rgis['rgi_id'].apply(transform_identifier)


# %%% Cell 1c: Load the WGMS data - not used
WGMS_info_path = "/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/04. Reference/01. Climate data/WGMS-MB/data/glacier.csv"
WGMS_B_path = "/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/04. Reference/01. Climate data/WGMS-MB/data/mass_balance.csv"


WGMS_B = pd.read_csv(WGMS_B_path)
WGMS_B = xr.Dataset.from_dataframe(WGMS_B)
WGMS_I = pd.read_csv(WGMS_info_path)
WGMS_I = xr.Dataset.from_dataframe(WGMS_I)

condition = (WGMS_I.GLACIER_REGION_CODE == 'ASC') & (
    (WGMS_I.GLACIER_SUBREGION_CODE == 'ASC-02') | (WGMS_I.GLACIER_SUBREGION_CODE == 'ASC-09'))
WGMS_I_13 = WGMS_I.where(condition, drop=True)
WGMS_I_13['WGMS_ID'] = WGMS_I_13['WGMS_ID'].astype(int)
# print(WGMS_I_13)

objs = [WGMS_I_13, WGMS_B]
# merge data based on selected WGMS-I IDs, all data that do not match dont result in the dataset (hence len WGMS_I should be equal to length of merged dataset)
WGMS = xr.merge(objs, compat="override", join="inner")

condition2 = WGMS['ANNUAL_BALANCE'].notnull()
WGMS = WGMS.where(condition, drop=True)


# %% Cell 2: Filter the Hugonnet data according to availability in the RGI
""" Include only glaciers that are in RGI in the Hugonnet dataset"""

hugo_ds = ds.where(ds['RGI-ID'].isin(rgis['rgi_id_format']), drop=True)
# hugo_ds_excluded = ds.where(~ds['RGI-ID'].str[-5:].isin(rgis),drop=True)


# %% Cell 3a: Plot histogram from Hugonnet input data (# glaciers vs B)
""" Find glaciers with most postive MB and other conditions"""

# create a copy of hugo_ds, where the non RGI glaciers have been filtered out
hugo_ds_filtered = hugo_ds
conditions = {
    # 'B': ds_filtered['B'] >= 0,
    # 'errB': ds_filtered['errB'] < 0.2,
    'Area': hugo_ds_filtered['Area'] > 10
}

for var_name, condition in conditions.items():
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
# plt.figure()

# plt.scatter(df_filtered['errB'], df_filtered['Area'])
# # plt.xlim(0,2.5)
# # plt.ylim(0,150)
# plt.ylabel('Area')
# plt.xlabel('errB')
# plt.legend()

# plt.figure()
# plt.scatter(df_filtered['B'], df_filtered['Area'])
# # plt.xlim(0,2.5)
# # plt.ylim(0,150)
# plt.ylabel('Area')
# plt.xlabel('B (m w.e.)')
# plt.legend()

# Save to CSV
# df_filtered.to_csv('/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/04. Output files/02. OGGM/00. Glacier Selection/Hugonnet_Most_Positive_Screened_2000_2020.csv', index=False)


# %% Cell 4a: ONLY DOWNLOAD ONCE (takes a long time) Get gdirs for the based on Hugonnet sample glacier dataset

download_gdirs = False

if download_gdirs == True:
    rgi_id_sel = hugo_ds_filtered['RGI-ID'].values
    folder_path = '/Users/magaliponds/Documents/00. Programming'
    wd_path = f'{folder_path}/02. Modelled perturbation-glacier interactions - regional analysis/'

    # OGGM options
    oggm.cfg.initialize(logging_level='WARNING')
    oggm.cfg.PATHS['working_dir'] = utils.mkdir(wd_path, reset=False)
    oggm.cfg.PARAMS['store_model_geometry'] = True
    cfg.PARAMS['use_multiprocessing'] = False

    gdirs = []
    with tqdm(total=len(rgi_id_sel), desc="Initializing Glacier Directories", mininterval=1.0) as pbar:
        for rgi_id in rgi_id_sel:
            # Perform your processing here
            gdir = workflow.init_glacier_directories(
                [rgi_id],
                prepro_base_url=DEFAULT_BASE_URL,
                from_prepro_level=5,
                prepro_border=80
            )
            gdirs.extend(gdir)

            # Update tqdm progress bar
            pbar.update(1)

# %% Cell 4b: Save gdirs from OGGM as a pkl file

# Directory to save gdirs
folder_path = '/Users/magaliponds/Documents/00. Programming'
wd_path = f'{folder_path}/02. Modelled perturbation-glacier interactions - regional analysis/'

os.makedirs(wd_path, exist_ok=True)

# Save each gdir individually
for gdir in gdirs:
    gdir_path = os.path.join(wd_path, f'{gdir.rgi_id}.pkl')
    with open(gdir_path, 'wb') as f:
        pickle.dump(gdir, f)
# %% Cell 4c: Load gdirs from pkl
folder_path = '/Users/magaliponds/Documents/00. Programming'
wd_path = f'{folder_path}/02. Modelled perturbation-glacier interactions - regional analysis/'

gdirs = []
for filename in os.listdir(wd_path):
    if filename.endswith('.pkl'):
        file_path = os.path.join(save_dir, filename)
        with open(file_path, 'rb') as f:
            gdir = pickle.load(f)
            gdirs.append(gdir)

# %% Cell 5: Run the climate model for every glacier in our working directory

sum_dir = os.path.join(wd_path, 'summary')
os.makedirs(sum_dir, exist_ok=True)
cfg.PARAMS['use_multiprocessing'] = False
y0_clim = 1985
ye_clim = 2014

function_directory = "/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/src/03. Glacier simulations"
sys.path.append(function_directory)


members = [1, 3, 4, 6, 1]
models = ["IPSL-CM6", "E3SM", "CESM2", "CNRM"]
timeframe = "monthly"

opath_climate = os.path.join(sum_dir, 'climate_historical.nc')
utils.compile_climate_input(
    gdirs, path=opath_climate, filename='climate_historical')

for m, model in enumerate(models):
    for member in range(members[m]):

        # Provide the path to the perturbation dataset
        i_folder_ptb = f"/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/03. Output files/01. Climate data/07. OGGM Perturbation input files/{timeframe}/{model}/{member}"
        ds_path = f"{i_folder_ptb}/{model}.00{member}.{y0_clim}_{ye_clim}.{timeframe}.perturbation.input.%.degC.nc"

        # Provide the sample ID to provide the processed pertrubations with the correct output suffix
        sample_id = f"{model}.00{member}"

        workflow.execute_entity_task(process_perturbation_data, gdirs,
                                     ds_path=ds_path,
                                     y0=None, y1=None,
                                     output_filesuffix=f'_perturbation_{sample_id}')

        opath_perturbations = os.path.join(
            sum_dir, f'climate_historical_perturbation_{sample_id}.nc')
        utils.compile_climate_input(gdirs, path=opath_perturbations, filename='climate_historical',
                                    input_filesuffix=f'_perturbation_{sample_id}',
                                    use_compression=True)


# %% Cell 6: Perturb the climate data in a gcm file

# gdir = gdirs[0]
for gdir in gdirs:
    # tasks.init_present_time_glacier(gdir)
    with xr.open_dataset(gdir.get_filepath('climate_historical')) as ds:
        ds = ds.load()

    members = [1, 3, 4, 6, 1]
    models = ["IPSL-CM6", "E3SM", "CESM2", "CNRM"]
    for m, model in enumerate(models):
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


# %% Cell 7: Run Mass Balance model for the glaciers

log_dir = os.path.join(wd_path, "log")
os.makedirs(log_dir, exist_ok=True)


# gdirs_sel=gdirs[1:5]


members = [1, 3, 4, 6]  # , 1]
models = ["W5E5", "E3SM", "CESM2", "CNRM"]  # , "IPSL-CM6"]

for m, model in enumerate(models):
    for member in range(members[m]):
        sample_id = f"{model}.00{member}"
        mb_ts_mean = []
        mb_ts_all = []
        error_ids = []
        count = 0

        for (g, gdir) in enumerate(gdirs):
            try:
                count += 1
                # print(f"Processing {count} of {len(gdirs)}")
                fls = gdir.read_pickle('model_flowlines')
                years = np.arange(1985, 2014)

                if model == "W5E5":
                    # Get the calibrated mass-balance model - the default is to use OGGM's "MonthlyTIModel" and compute specific mb
                    mbmod = massbalance.MonthlyTIModel(gdir)
                    mb_ts = mbmod.get_specific_mb(fls=fls, year=years)
                else:
                    mbmod = massbalance.MonthlyTIModel(
                        gdir, filename='gcm_data', input_filesuffix='_perturbed_{}'.format(sample_id))
                    mb_ts = mbmod.get_specific_mb(fls=fls, year=years)

                # Append all time series data to mb_ts_all
                for year, mb in zip(years, mb_ts):
                    mb_ts_all.append((gdir.rgi_id, year, mb))

            except Exception as e:
                # Handle the error and continue
                print(
                    f"Error processing {gdir.rgi_id} with model {model} and member {member}: {e}")
                error_ids.append((gdir.rgi_id, model, member))
                continue
            mean_mb = np.mean(mb_ts)
            mb_ts_mean.append((gdir.rgi_id, mean_mb))

        mb_df_mean = pd.DataFrame(mb_ts_mean, columns=['rgi_id', 'B'])
        mb_df_mean.to_csv(os.path.join(
            sum_dir, f'specific_massbalance_mean_{sample_id}.csv'), index=False)

        # Create a DataFrame from the collected data
        mb_ts_df = pd.DataFrame(
            mb_ts_all, columns=['rgi_id', 'Year', 'Mass_Balance'])
        # Save to a single CSV file
        mb_ts_df.to_csv(os.path.join(
            sum_dir, f'specific_massbalance_timeseries_{sample_id}.csv'), index=False)

        # Optionally save the list of error cases to a CSV for later review
        if error_ids:
            error_df = pd.DataFrame(
                error_ids, columns=['rgi_id', 'Model', 'Member'])
            error_df.to_csv(os.path.join(
                log_dir, 'Error_Log.csv'), index=False)


# %%Cell 8:  Open the mass balance files and plot
def gaussian(x, mu, sigma, A):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))


members = [3, 4, 6, 1]
# members = [1, 1, 1, 1, 1]

models = ["E3SM", "CESM2", "CNRM", "IPSL-CM6",]  # , "IPSL-CM6"]
fig, axes = plt.subplots(2, 2, figsize=(15, 8), sharex=True, sharey=True)
axes = axes.flatten()
alpha_set = 0.8

i_path_base = os.path.join(sum_dir, 'specific_massbalance_mean_W5E5.000.csv')
mb_base = pd.read_csv(i_path_base, index_col=0).to_xarray()

bin_size = np.linspace(mb_base.B.min(), mb_base.B.max(), int(len(mb_base.B)/7))
linestyle = ["solid", "dotted", "dashed",
             "densely dashdotted", "dash dot dotted"]
for m, model in enumerate(models):
    for member in range(members[m]):

        member_reverse = members[m] - member - 1
        sample_id = f"{model}.00{member}"

        i_path = os.path.join(
            sum_dir, f'specific_massbalance_mean_{sample_id}.csv')
        mb = pd.read_csv(i_path, index_col=0).to_xarray()

        n, bins, patches = axes[m].hist(mb_base.B, bins=bin_size, align='left',
                                        rwidth=0.8, facecolor="grey", alpha=0.5, edgecolor="k", zorder=1)
        if member == 0:
            n, bins, patches = axes[m].hist(mb.B, bins=bin_size, align='left',
                                            rwidth=0.8, facecolor=colors[model][member], alpha=0.8, zorder=members[m])
        else:
            "Only use this line if you wish to plot for every member"

            n, bins, patches = axes[m].hist(
                mb.B, bins=bin_size, align='left',  rwidth=0.8, facecolor=colors[model][member], alpha=0.5, zorder=member)

        # for custom number of bars use: np.linspace(mb.B.min(), mb.B.max(), int(len(mb.B)/10))
        # Fit the Gaussian curve to the histogram data
        bin_centers = (bins[:-1] + bins[1:]) / 2
        params, covariance = curve_fit(gaussian, bin_centers, n, p0=[
                                       np.mean(mb.B), np.std(mb.B), np.max(n)])
        x = np.linspace(mb.B.min(), mb.B.max(), 100)

        params, covariance = curve_fit(gaussian, bin_centers, n, p0=[
                                       np.mean(mb.B), np.std(mb.B), np.max(n)])
        x_base = np.linspace(mb_base.B.min(), mb_base.B.max(), 100)

        if member == 0:
            axes[m].plot(x_base, gaussian(x, *params), color=colors["W5E5"]
                         [0], label="W5E5.000", zorder=(members[m]+1))
            axes[m].plot(x, gaussian(x, *params), color=colors[model]
                         [member], label=f'{sample_id}', zorder=(2*members[m]+1))
        else:
            axes[m].plot(x, gaussian(x, *params), color=colors[model][member],
                         linesteyle=":", label=f'{sample_id}', zorder=(members[m]+member+1))
            total_glaciers = len(mb.B.values)
        # plt.annotate(f"Total number of glaciers: {total_glaciers}", xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, verticalalignment='top')
        # plt.annotate(f"Std: {np.round(np.std(mb.B),2)}", xy=(0.05, 0.9), xycoords='axes fraction', fontsize=12, verticalalignment='top')

        if m in [0, 2]:
            axes[m].set_ylabel("Number of glaciers [#]")
        if m in [2, 3]:
            axes[m].set_xlabel(
                "Mean specific mass balance 2000-2020 [mm w.e. yr$^-1$]")
        axes[m].legend(loc="upper left", bbox_to_anchor=(0, 1), ncols=2)
        axes[m].set_xlim(-1250, 1000)
        axes[m].axvline(0, color='k')
plt.show()

# %% Make scatter plot with RGI data
members = [1, 3, 4, 6, 1, 1]
# members = [1, 1, 1, 1, 1]

models = ["W5E5", "E3SM", "CESM2", "CNRM"]  # , "IPSL-CM6"]

plt.figure(figsize=(15, 8))


# plt.scatter(mb_base.rgi_id, mb_base.B,color=colors["W5E5"][0])
for m, model in enumerate(models):
    for member in range(members[m]):

        sample_id = f"{model}.00{member}"

        if model == "W5E5":
            i_path_base = os.path.join(
                sum_dir, 'specific_massbalance_mean_W5E5.000.csv')
            mb = pd.read_csv(i_path_base, index_col=0).to_xarray()
        else:
            i_path = os.path.join(
                sum_dir, f'specific_massbalance_mean_{sample_id}.csv')
            mb = pd.read_csv(i_path, index_col=0).to_xarray()

        mb['B'] = mb.B/1000
        mb['B'].attrs['units'] = "m w.e. yr-1"
        mb['B'].attrs['standard_name'] = "Mean specific Mass balance 2000-2020"

        hugo_ds = hugo_ds_filtered[['RGI-ID', 'B']].set_index(index='RGI-ID')
        hugo_ds = hugo_ds.rename({'index': 'rgi_id'})

        hugo_ds = hugo_ds.where(hugo_ds.rgi_id.isin(mb.rgi_id), drop=True)

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


# %% Cell5 b: Convert MB data for plotting - checking if the same are in hugo ds
# mb_df_f = mb_df[mb_df['rgi_id'].isin(hugo_ds_filtered['RGI-ID'])]
# print(mb_df)
mb_df['B'] = mb_df['B']/1000
print(mb_df)

# %% Calculate correlation graph

# load the OGGM modelled B data
mb_df = pd.read_csv('/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/04. Output files/02. OGGM/00. Glacier Selection/OGGM_Positive_B_2000_2020.csv')
mb_df['B'] = mb_df['B']/1000
mb_ds = mb_df.set_index('rgi_id').to_xarray()
hugo_ds = hugo_ds_filtered[['RGI-ID', 'B']]
hugo_ds = hugo_ds.set_index(index='RGI-ID')
hugo_ds = hugo_ds.rename({'index': 'rgi_id'})

hugo_ds = hugo_ds.where(hugo_ds.rgi_id.isin(mb_ds.rgi_id), drop=True)

mb_df = mb_ds.to_dataframe()
hugo_df = hugo_ds.to_dataframe()
cor_df = pd.merge(mb_df, hugo_df, on='rgi_id', suffixes=('_oggm', '_hugo'))

cor_ds = cor_df.to_xarray()
correlation = xr.corr(cor_ds['B_oggm'], cor_ds['B_hugo']).round(2)

# cor_ds = xr.merge([mb_ds, hugo_ds], dim='rgi_id')
plt.scatter(cor_ds['B_hugo'], cor_ds['B_oggm'], s=5, color='lightblue')
plt.plot(cor_ds['B_hugo'], cor_ds['B_hugo'], color='k')
plt.text(-1.4, 0.2, f'Correlation coefficient: {correlation.values}')

plt.xlabel('B OGGM (mm w.e.)')
plt.ylabel('B hugo (mm w.e.)')
