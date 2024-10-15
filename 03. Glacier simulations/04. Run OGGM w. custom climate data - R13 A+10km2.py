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
import matplotlib.patches as mpatches
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


# %% Cell 1a: ONLY DOWNLOAD ONCE (takes a long time) Get OGGM gdirs for the based on Hugonnet sample glacier dataset

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

# %% Cell 1b: Save gdirs from OGGM as a pkl file (Save at the end of every working session)

# Save each gdir individually
for gdir in gdirs:
    gdir_path = os.path.join(pkls, f'{gdir.rgi_id}.pkl')
    with open(gdir_path, 'wb') as f:
        pickle.dump(gdir, f)

# %% Cell 1c: Load gdirs from pkl (fastest way to get started)

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

# %% Cell 2: Read the inversion output to add to the master
master_df = pd.read_csv(
    f"{wd_path}/region13_RGI_master_rgi_hugo.csv").drop(columns=['Unnamed: 0'])
print(master_df.columns)
# %% Cell 3: Create a dataset that contains Gdir Area, volume, rgi date and rgi ID

# Initialize lists to store output
rgi_ids = []
rgi_dates = []
rgi_areas = []
rgi_volumes = []

# Iterate through each glacier directory
for gdir in gdirs:
    try:
        # Load data
        rgi_ids.append(gdir.rgi_id)  # RGI ID
        rgi_dates.append(gdir.rgi_date)  # Validity date
        rgi_areas.append(gdir.rgi_area_km2)  # Area corresponding to RGI date

        # Calculate the number of steps from RGI date to 2020
        # steps = 2020 - gdir.rgi_date + 1

        # Load the model diagnostics
        with xr.open_dataset(gdir.get_filepath('model_diagnostics', filesuffix="_historical")) as ds:
            # [-steps]  # Use the most relevant volume data
            vol = ds.volume_m3[1]
            rgi_volumes.append(vol.values * 10**-9)  # Convert to km³

    except Exception as e:
        # Print error message for debugging
        print(f"Error processing {gdir.rgi_id}: {e}")

        # Remove the last added values from all lists if file couldn't be opened
        if rgi_ids:  # Check if lists are non-empty
            rgi_ids.pop()      # Remove last RGI ID
            rgi_dates.pop()    # Remove last validity date
            rgi_areas.pop()    # Remove last area

# Create a DataFrame only if lists are non-empty
if rgi_ids:
    data = {
        'rgi_id': rgi_ids,
        'rgi_date': rgi_dates,
        'rgi_area_km2': rgi_areas,
        'rgi_volume_km3': rgi_volumes
    }

    # Create the DataFrame
    df = pd.DataFrame(data)
else:
    # Empty DataFrame if no data
    df = pd.DataFrame(columns=['rgi_id', 'rgi_date',
                      'rgi_area_km2', 'rgi_volume_km3'])

# Output the DataFrame
print(df)
# ssave the dataframe in the working directory
df.to_csv(f"{wd_path}/master_gdir_rgi_date_A_V.csv")


# %% Cell 4: Plot total cumulative volume vs area (ascending)


# Sort the DataFrame by area from large to small
df_sorted = df.sort_values(by='rgi_area_km2', ascending=True)

# Calculate cumulative volume
df_sorted['cumulative_volume_km3'] = df_sorted['rgi_volume_km3'].cumsum()
df_sorted = df_sorted.sort_values(by='rgi_area_km2', ascending=True)

# Create the plot
plt.figure(figsize=(10, 6))

# Normalize the volumes for color mapping
norm = plt.Normalize(df_sorted['rgi_volume_km3'].min(),
                     df_sorted['rgi_volume_km3'].max())
cmap = cm.viridis  # You can choose any colormap you like

# Plot points with color scale based on individual volume
sc = plt.scatter(df_sorted['rgi_area_km2'], df_sorted['cumulative_volume_km3'],
                 c=df_sorted['rgi_volume_km3'], cmap=cmap, norm=norm, alpha=0.7)

plt.grid(False)  # turn off the gridlines
# Set titles and labels
plt.title('Cumulative Glacier Volume vs Area', fontsize=16)
plt.suptitle(f'Total Glaciers: {len(df_sorted)}', fontsize=12)
plt.xlabel('Glacier Area (km²)', fontsize=12)
plt.ylabel('Cumulative Glacier Volume (m³)', fontsize=12)

# Invert the y-axis to have cumulative volume decrease as area decreases
# plt.gca().invert_yaxis()

# Create a colorbar to indicate volume scale
cbar = plt.colorbar(sc)
cbar.set_label('Individual Glacier Volume (km³)', fontsize=12)

# Optional: Calculate and display mean and median volumes
mean_volume = df_sorted['rgi_volume_km3'].mean()
median_volume = df_sorted['rgi_volume_km3'].median()
plt.text(0.05, 0.95, f'Mean Volume: {mean_volume:.2f} km³\nMedian Volume: {median_volume:.2f} km³',
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor='white', facecolor='white'))

# Show the plot
plt.show()


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

        # create a sample id for all the model x member combinations
        sample_id = f"{model}.00{member}"

        # create lists to store the model output
        mb_ts_mean = []
        mb_ts_all = []
        mb_ts_mean_ext = []
        mb_ts_all_ext = []
        error_ids = []

        # load the gdirs
        for (g, gdir) in enumerate(gdirs):
            try:
                # provide the model flowlines and years for the mbmod
                fls = gdir.read_pickle('model_flowlines')
                years = np.arange(1985, 2015)
                # years = np.arange(2000, 2010) #test for all datasets in same timeframe

                if model == "W5E5":
                    # extend range for w5e5, to see match w. geodetic mass balance
                    years_ext = np.arange(2000, 2020)
                    # Get the calibrated mass-balance model - the default is to use OGGM's "MonthlyTIModel" and compute specific mb
                    mbmod = massbalance.MonthlyTIModel(
                        gdir, filename='climate_historical')
                    mb_ts = mbmod.get_specific_mb(fls=fls, year=years)
                    # also run the model for the extended years and save
                    mb_ts_ext = mbmod.get_specific_mb(fls=fls, year=years_ext)
                    for year, mb in zip(years, mb_ts):
                        mb_ts_all.append((gdir.rgi_id, year, mb))
                else:
                    mbmod = massbalance.MonthlyTIModel(
                        gdir, filename='gcm_data', input_filesuffix='_perturbed_{}'.format(sample_id))
                    mb_ts = mbmod.get_specific_mb(fls=fls, year=years)
                # Append all time series data to mb_ts_all
                for year, mb in zip(years, mb_ts):
                    mb_ts_all.append((gdir.rgi_id, year, mb))

            # include an exception so the model will continue running on error and provide the error
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

        # create a dataframe with the mass balance data of all gdirs
        mb_df_mean = pd.DataFrame(mb_ts_mean, columns=['rgi_id', 'B'])
        mb_df_mean.to_csv(os.path.join(
            sum_dir, f'specific_massbalance_mean_{sample_id}.csv'), index=False)
        # mb_df_mean.to_csv(os.path.join(
        #     sum_dir, f'specific_massbalance_mean_{sample_id}_2000_2010.csv'), index=False) #for testing with same timeseries
        mb_ts_df = pd.DataFrame(
            mb_ts_all, columns=['rgi_id', 'Year', 'Mass_Balance'])
        mb_ts_df.to_csv(os.path.join(
            sum_dir, f'specific_massbalance_timeseries_{sample_id}.csv'), index=False)
        # mb_ts_df.to_csv(os.path.join(
        #     sum_dir, f'specific_massbalance_timeseries_{sample_id}_2000_2010.csv'), index=False) # for testing with same timeframe

        if model == "W5E5":  # only for W5E5 also create a dataframe for the extended timeseries
            mb_df_mean_ext = pd.DataFrame(
                mb_ts_mean_ext, columns=['rgi_id', 'B'])
            mb_df_mean_ext.to_csv(os.path.join(
                sum_dir, f'specific_massbalance_mean_extended_{sample_id}.csv'), index=False)
            # mb_df_mean_ext.to_csv(os.path.join(
            #     sum_dir, f'specific_massbalance_mean_extended_{sample_id}_2000_2020.csv'), index=False) #for testing with the same timeframes
            mb_ts_df_ext = pd.DataFrame(mb_ts_all_ext, columns=[
                                        'rgi_id', 'Year', 'Mass_Balance'])
            mb_ts_df_ext.to_csv(os.path.join(
                sum_dir, f'specific_massbalance_timeseries_extended_{sample_id}.csv'), index=False)
            # mb_ts_df_ext.to_csv(os.path.join(
            #     sum_dir, f'specific_massbalance_timeseries_extended_{sample_id}_2000_2020.csv'), index=False) #for testing with the same timeframes

        # Optionally save the list of error cases to a CSV for later review
        if error_ids:
            error_df = pd.DataFrame(
                error_ids, columns=['rgi_id', 'Model', 'Member'])
            error_df.to_csv(os.path.join(
                log_dir, 'Error_Log.csv'), index=False)

# %% Cell 80: Update the master dataset with B, A, RGI ID and location for all the different sample ids (541x15=8115)

master_df = pd.read_csv(f"{wd_path}/region13_RGI_master_rgi_hugo.csv")
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
# master_df = master_ds.to_dataframe()
df_complete = pd.merge(df_complete, master_df, on='rgi_id', how='inner')

# reorder the dataset
df_complete = df_complete[[
    'rgi_id', 'rgi_id_rgis', 'rgi_id_hugo', 'sample_id',
    'B_noirr', 'B_irr', 'B_delta', 'B',
    'errB', 'cenlon', 'cenlat', 'lon',
    'lat', 'area_km2', 'Area', 'area_dif', 'Volume Change'
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
alpha_set = 0.8

mb_members = []

# Initialize plot

fig, ax = plt.subplots(1, 1, figsize=(10, 6), sharex=True, sharey=True)

# Load baseline data
i_path_base = os.path.join(sum_dir, 'specific_massbalance_mean_W5E5.000.csv')
mb_base = pd.read_csv(i_path_base, index_col=0).to_xarray()
bin_size = np.linspace(mb_base.B.min(), mb_base.B.max(),
                       int(len(mb_base.B) / 7))

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


# calculate the 11-member average
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
plt.show()


# %% Cell 8B: Plot the specific mass balances using boxplots

# Configuration variables

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
for m, model in enumerate(models_shortlist):
    for member in range(members_averages[m]):
        if members_averages[m] > 1:
            member += 1
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
data.append(np.mean(data, axis=0))
labels.append("11-member average")
colors_list.append("lightblue")

data.append(mb_base.B.values)
labels.append("W5E5.000")
colors_list.append('darkgrey')  # Assign W5E5 a default color (darkgrey)

# Reverse the order of the data, labels, and colors except for "W5E5.000"
# data[:-1] all but the last data point, reverse it and add it
data = data[:-2][::-1] + [data[-2]] + [data[-1]]
labels = labels[:-2][::-1] + [labels[-2]] + [labels[-1]]
colors_list = colors_list[:-2][::-1] + [colors_list[-2]] + [colors_list[-1]]

# Plotting the main boxplot
box = ax.boxplot(data, patch_artist=True, labels=labels,
                 vert=False, boxprops=dict(edgecolor='none'))  # , color=colors_list))
# plt.show()
# Color the boxes
for patch, color in zip(box['boxes'], colors_list):
    # option to change alpha of face color, potential to use with colored median bar
    rgba_color = clrs.to_rgba(color, alpha=0.2)
    patch.set_facecolor(color)


# Change the color of the median line to black
for median, color in zip(box['medians'], colors_list):
    median.set_color('black')  # option to change to colors from color list

# Plot the W5E5 dashed outline on top of all other boxplots
# This is achieved by plotting only the W5E5 data again, but with dashed lines
box_w5e5 = ax.boxplot([mb_base.B.values] * len(data), patch_artist=False, vert=False,
                      positions=np.arange(len(data), 0, -1),
                      boxprops=dict(linestyle=':', color='grey'),
                      whiskerprops=dict(linestyle=':', color='none'),
                      capprops=dict(linestyle=':', color='none'),
                      showfliers=False)

# Set the median lines for the W5E5 boxplot to dashed black
for median in box_w5e5['medians']:
    median.set_linestyle(':')
    median.set_color('grey')

# Identify the index of the W5E5 label (assuming it's the last one)
y_labels = ax.get_yticklabels()
y_labels = y_labels[:13]  # Set the W5E5 label to an empty string
y_ticks = ax.get_yticks()
y_ticks = y_ticks[:13]  # Set the W5E5 label to an empty string
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

ds = pd.read_csv(
    f"{wd_path}/region13_RGI_master_rgi_hugo_filtered_area_B.csv", index_col=0, header=0)
mean_noirr = ds.groupby('rgi_id')['B_noirr'].mean().reset_index()
mean_irr = ds.groupby('rgi_id')['B_irr'].mean().reset_index()

# make a subset for all the glaciers with a specific area
mean_noirr_area = ds.where(ds.area_km2 > 1).groupby('rgi_id')[
    'B_noirr'].mean().reset_index()
mean_noirr_delta_all = ds.groupby('rgi_id')[['B_noirr', 'B_delta']].mean(
).reset_index()
mean_noirr_delta = mean_noirr_delta_all.nlargest(54, 'B_delta')

# make a ssubset for the glaciers with the largest change
labels = ["NoIrr - 11 member avg: largest 10% $\Delta$"] + \
    ["NoIrr - 11 member avg: A>1km$^2$"] + \
    ["NoIrr - 11 member avg: all"] + ["Irr (W5E5.000)"]
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
# rgis_complete = rgis#.drop(columns=['rgi_id']).rename(columns={'rgi_id_format': 'rgi_id'})
# df_delta = pd.merge(mean_noirr_delta_all, rgis_complete,
#                     on='rgi_id', how='inner')
# df_delta = pd.merge(mean_noirr_delta, rgis_complete, on='rgi_id', how='inner') #for only 10% changing largest glaciers

master_df = pd.read_csv(
    f"{wd_path}/region13_RGI_master_rgi_hugo_filtered_area_B.csv")
threshold = master_df['B_delta'].quantile(
    0.95)  # only top 10% largest changes in B
master_df = master_df[master_df['B_delta'] >= threshold]
# df_irr = pd.merge(mean_irr, rgis_complete, on='rgi_id', how='inner')

# datasets = [df_delta, df_irr]

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
shp.plot(ax=ax, edgecolor='black', linewidth=0, facecolor='bisque')

# Create a custom diverging colormap: red for negative, white for zero, blue for positive
colors = [(1, 1, 1), (0, 0, 1)]  # Red -> White -> Blue
custom_cmap = LinearSegmentedColormap.from_list(
    'white_blues', colors, N=256)

# Normalize with zero centered (TwoSlopeNorm sets the midpoint to zero)
vmin = 0  # master_df['B_delta'].min()
vmax = master_df['B_delta'].quantile(0.9)
norm = TwoSlopeNorm(vmin=vmin, vcenter=(vmin+vmax)/2, vmax=vmax)

# Add scatter plot for the locations
sc = ax.scatter(master_df['cenlon'], master_df['cenlat'], s=master_df['B_irr'],
                c=master_df['B_delta'], cmap=custom_cmap, norm=norm, edgecolor='k', alpha=0.7)

# Add a colorbar for the B_delta values
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('$\Delta$ B (Irr-NoIrr)')


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

master_ds = pd.read_csv(
    f"{wd_path}/region13_RGI_master_rgi_hugo_filtered_area.csv")
hugo_area = master_ds.Area.sum()

# define the variables for plotting
variables = ["volume", "area"]
factors = [10**-9, 10**-6]

variable_names = ["Volume", "Area"]
variable_axes = ["Volume [km$^3$]", "Area [km$^2$]"]


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

    # calculate and plot volume/area 11-member mean
    mean_values = np.mean(stacked_member_data, axis=0).flatten()
    ax.plot(climate_run_output["time"].values, mean_values,
            color="black", linestyle='dashed', lw=2, label="11-member average")
    # if var=="area":
    #     ax.scatter([1985.], [hugo_area], s=20, marker="x", color="black",label="Hugonnet Area")

    # calculate and plot volume/area 11-member min and max for ribbon
    min_values = np.min(stacked_member_data, axis=0).flatten()
    max_values = np.max(stacked_member_data, axis=0).flatten()
    ax.fill_between(climate_run_output["time"].values, min_values, max_values,
                    color="lightblue", alpha=0.3, label=f"11-member range", zorder=16)

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
    o_folder_data = f"/Users/magaliponds/OneDrive - Vrije Universiteit Brussel/1. VUB/02. Coding/01. IRRMIP/04. Figures/02. OGGM simulations/01. Modelled output/0{v + 1}. {variable_names[v]}/00. Combined"
    os.makedirs(o_folder_data, exist_ok=True)
    o_file_name = f"{o_folder_data}/1985_2014.{timeframe}.delta.{variable_names[v]}.combined.png"
    plt.savefig(o_file_name, bbox_inches='tight')


# %% Load the hugonnet data from oggm

rgi_id_sel = hugo_ds_filtered['RGI-ID'].values

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
#                                 np.mean(mb_base.B), np.std(mb_base.B), np.max(n)])
# params_geo, covariance_geo = curve_fit(gaussian, bin_centers, n, p0=[
#                                         np.mean(mb_geo.B), np.std(mb_geo.B), np.max(n)])

# x = np.linspace(mb_base.B.min(), mb_base.B.max(), 100)
# x_geo = np.linspace(mb_geo.B.min(), mb_geo.B.max(), 100)
# plt.plot(x, gaussian(x, *params),
#           color=colors["W5E5"][0], label="Gaussian fit W5E5.000", zorder=(members[m]+1))
# plt.plot(x, gaussian(x_geo, *params_geo),
#           color=colors["W5E5"][0], label="Gaussian fit Geodetic MB", zorder=(members[m]+1), linestyle="dashed")

# plt.legend(loc="upper left", bbox_to_anchor=(0, 1), ncols=1)
# plt.xlabel("Mean specific mass balance[mm w.e. yr$^{-1}$]")
# plt.ylabel("# of glaciers [-]")
# plt.axvline(0, color='k', linestyle="dotted")
# plt.annotate(f"Total # of glaciers: {total_glaciers}", xy=(
#     0.01, 0.74), xycoords='axes fraction', fontsize=10, verticalalignment='top')
# plt.annotate(f"Time period = 2000-2020", xy=(
#     0.01, 0.70), xycoords='axes fraction', fontsize=10, verticalalignment='top')

# plt.show()

# %% Correlation curve

members = [1, 1, 1, 1, 1]
models = ["W5E5", "E3SM", "CESM2", "CNRM", "IPSL-CM6"]

hugo_ds_filtered = pd.read_csv(
    f"{wd_path}/region13_RGI_master_rgi_hugo_filtered_area.csv")

plt.figure(figsize=(10, 6))

# plt.scatter(mb_base.rgi_id, mb_base.B,color=colors["W5E5"][0])
for m, model in enumerate(models):
    for member in range(members[m]):
        print(model)

        sample_id = f"{model}.00{member}"

        if model == "W5E5":
            i_path_base = os.path.join(
                sum_dir, 'specific_massbalance_mean_extended_W5E5.000_2000_2020.csv')
            mb = pd.read_csv(i_path_base, index_col=0).to_xarray()

        else:
            i_path = os.path.join(
                sum_dir, f'specific_massbalance_mean_{sample_id}.csv')
            mb = pd.read_csv(i_path, index_col=0).to_xarray()

        mb['B'] = mb['B']/1000
        mb['B'].attrs['units'] = "m w.e. yr-1"
        mb['B'].attrs['standard_name'] = "Mean specific Mass balance 2000-2020"

        hugo_ds = hugo_ds_filtered[['rgi_id', 'B']
                                   ].set_index('rgi_id').to_xarray()

        hugo_ds = hugo_ds.where(hugo_ds.rgi_id.isin(mb.rgi_id), drop=True)
        hugo_ds['B'] = hugo_ds.B

        hugo_df = hugo_ds.to_dataframe()
        mb_df = mb.to_dataframe()
        cor_df = pd.merge(mb_df, hugo_df, on='rgi_id',
                          suffixes=('_oggm', '_hugo'))

        cor_ds = cor_df.to_xarray()
        correlation = xr.corr(cor_ds['B_oggm'], cor_ds['B_hugo']).round(2)
        rmse = np.sqrt(
            np.mean((cor_ds['B_oggm'] - cor_ds['B_hugo']) ** 2)).values.round(2)

        # cor_ds = xr.merge([mb_ds, hugo_ds], dim='rgi_id')
        if m == 0 and member == 0:
            plt.plot(cor_ds['B_hugo'], cor_ds['B_hugo'], color='k',
                     label="correlation: 1", zorder=(sum(members)+1))

        if model == "W5E5":
            plt.scatter([cor_ds['B_hugo']], [cor_ds['B_oggm']], s=5, color=colors[model][member],
                        label=f'{sample_id} (2000-2020), correlation: {correlation.values}, rmse: {rmse}', alpha=1, zorder=5)
        else:
            plt.scatter([cor_ds['B_hugo']], [cor_ds['B_oggm']], s=5, color=colors[model][member],
                        label=f'{sample_id}  (1985-2014), correlation: {correlation.values}, rmse: {rmse}', alpha=0.3)

        # plt.text(-1.4,0.2, f'Correlation coefficient: {correlation.values}')

        plt.ylabel('Modelled mass balance  (m w.e. yr$^{-1}$)')
        plt.xlabel('Geodetic mass balance (2000-2020) (m w.e. yr$^{-1}$)')
plt.legend()

# plt.scatter(mb_base.rgi_id, mb_base.B, color=colors[model][member])

plt.show()
#

# %% Correlation curve - for 2000-2010 - double plot for W5E5

members = [1, 1, 1, 1, 1]
models = ["W5E5", "E3SM", "CESM2", "CNRM"]  # , "IPSL-CM6"]

hugo_ds_filtered = pd.read_csv(
    f"{wd_path}/region13_RGI_master_rgi_hugo_filtered_area.csv")

plt.figure(figsize=(10, 6))

# plt.scatter(mb_base.rgi_id, mb_base.B,color=colors["W5E5"][0])
for m, model in enumerate(models):
    for member in range(members[m]):
        print(model)

        sample_id = f"{model}.00{member}"

        if model == "W5E5":
            i_path_base = os.path.join(
                sum_dir, 'specific_massbalance_mean_extended_W5E5.000_2000_2020.csv')
            mb_ext = pd.read_csv(i_path_base, index_col=0).to_xarray()/1000

        # else:
        i_path = os.path.join(
            sum_dir, f'specific_massbalance_mean_{sample_id}_2000_2010.csv')
        mb = pd.read_csv(i_path, index_col=0).to_xarray()

        mb['B'] = mb['B']/1000
        mb['B'].attrs['units'] = "m w.e. yr-1"
        mb['B'].attrs['standard_name'] = "Mean specific Mass balance 2000-2020"

        hugo_ds = hugo_ds_filtered[['rgi_id', 'B']
                                   ].set_index('rgi_id').to_xarray()

        hugo_ds = hugo_ds.where(hugo_ds.rgi_id.isin(mb.rgi_id), drop=True)
        hugo_ds['B'] = hugo_ds.B

        hugo_df = hugo_ds.to_dataframe()
        mb_df = mb.to_dataframe()
        cor_df = pd.merge(mb_df, hugo_df, on='rgi_id',
                          suffixes=('_oggm', '_hugo'))

        mb_df_ext = mb_ext.to_dataframe()
        mb_ext_df = mb_ext.to_dataframe()
        cor_df_ext = pd.merge(mb_df_ext, hugo_df, on='rgi_id',
                              suffixes=('_oggm', '_hugo'))
        cor_ds_ext = cor_df_ext.to_xarray()
        correlation_ext = xr.corr(
            cor_ds_ext['B_oggm'], cor_ds_ext['B_hugo']).round(2)
        rmse_ext = np.sqrt(
            np.mean((cor_ds_ext['B_oggm'] - cor_ds_ext['B_hugo']) ** 2)).values.round(2)

        cor_ds = cor_df.to_xarray()
        correlation = xr.corr(cor_ds['B_oggm'], cor_ds['B_hugo']).round(2)
        rmse = np.sqrt(
            np.mean((cor_ds['B_oggm'] - cor_ds['B_hugo']) ** 2)).values.round(2)

        # cor_ds = xr.merge([mb_ds, hugo_ds], dim='rgi_id')
        if m == 0 and member == 0:
            plt.plot(cor_ds['B_hugo'], cor_ds['B_hugo'], color='grey',
                     label="correlation: 1", zorder=(sum(members)+1))

        if model == "W5E5":
            plt.scatter([cor_ds['B_hugo']], [cor_ds['B_oggm']], s=5, color="grey",
                        label=f'{sample_id} (2000-2020), correlation: {correlation_ext.values}, rmse: {rmse_ext}', alpha=1, zorder=5)
            plt.scatter([cor_ds_ext['B_hugo']], [cor_ds_ext['B_oggm']], s=5, color=colors[model][member],
                        label=f'{sample_id} (2000-2010), correlation: {correlation.values}, rmse: {rmse}', alpha=1, zorder=5)
        else:
            plt.scatter([cor_ds['B_hugo']], [cor_ds['B_oggm']], s=5, color=colors[model][member],
                        label=f'{sample_id}  (1985-2014), correlation: {correlation.values}, rmse: {rmse}', alpha=0.3)

        # plt.text(-1.4,0.2, f'Correlation coefficient: {correlation.values}')

        plt.ylabel('Modelled mass balance  (m w.e. yr$^{-1}$)')
        plt.xlabel('Geodetic mass balance (2000-2010) (m w.e. yr$^{-1}$)')
plt.legend()

# plt.scatter(mb_base.rgi_id, mb_base.B, color=colors[model][member])

plt.show()
#

# %% Plot timeseries of mass balance agains hugonnet data

modelled_perturbed_mean = []
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# loop through all the different models and members
for m, model in enumerate(models):
    for member in range(members[m]):
        # creat a sample id for each modelxmember combi to access the data
        sample_id = f"{model}.00{member}"
        print(sample_id)
        mb_timeseries = pd.read_csv(os.path.join(
            sum_dir, f'specific_massbalance_timeseries_{sample_id}.csv'))  # load the mb timeseries (1985-2014 for all models, except W5E5 (1985-2020))
        pivot_df = mb_timeseries.pivot_table(
            index='rgi_id', columns='Year', values='Mass_Balance')  # pivot the datasets such that years are now columns
        years = np.arange(
            float(mb_timeseries['Year'].min()), mb_timeseries['Year'].max()+1, 1)  # create a years list based on min/max years in dataset

        # set model parameters depending on the model
        if model in ("W5E5", "IPSL-CM6") or member >= 1:
            if model == "W5E5":
                linestyle = "-"
                color = colors[model][member]
                linewidth = 2
                label = f"Modelled Irr (W5E5)"

            else:
                linestyle = ":"
                # add values of all individual members to create one mean value
                modelled_perturbed_mean.append(pivot_df.mean().values/1000)
                color = "grey"
                linewidth = 1.5
                if sample_id == "IPSL-CM6.000":
                    label = "Modelled NoIrr, individual member"
                else:
                    label = None
            # plot the timeseries for all members and W5E5, including labels, colors and lw as stated above
            ax.plot(years, pivot_df.mean()/1000, color=color,  # colors[model][member],
                    linewidth=linewidth, zorder=10, label=label, linestyle=linestyle)

        # reload dataset for W5E% - in case extended
        if model == "W5E5":
            mb_timeseries = pd.read_csv(os.path.join(
                sum_dir, f'specific_massbalance_timeseries_{sample_id}.csv'))

            # Keep only columns with year names greater than 2000 and calculate mean mb over this period
            mb_2020 = pivot_df.loc[:, pivot_df.columns.astype(int) > 2000]
            mean_mb_2020 = mb_2020.mean(axis=0).mean()/1000
            # plot W5E5-baseline average
            ax.plot([2000, 2020], [mean_mb_2020, mean_mb_2020], color=colors["W5E5"]
                    [0], linestyle='dashed', label='Modelled Irr (W5E5): mean 2000-2020')

stacked_member_data = np.stack(modelled_perturbed_mean)
mean_values = np.mean(stacked_member_data, axis=0).flatten()
min_values = np.min(stacked_member_data, axis=0).flatten()
max_values = np.max(stacked_member_data, axis=0).flatten()

ax.plot(years, mean_values, "grey", linewidth=2, zorder=10,
        label=f"Modelled NoIrr, 11-member average", linestyle=linestyle)
ax.fill_between(years, min_values, max_values, color="lightgrey",
                alpha=0.6, label=f"Modelled NoIrr, 11-member range", zorder=0)


# load the geodetic mass balance
mb_geodetic = utils.get_geodetic_mb_dataframe()
sel = mb_geodetic.loc[mb_geodetic.index.isin(
    mb_timeseries.rgi_id)].set_index('period')
# download the mb and error for Hugonnet
_mb, _err = sel.loc['2000-01-01_2020-01-01'][['dmdtda', 'err_dmdtda']].mean()

ax.plot([2000, 2020], [_mb, _mb], color='mediumslateblue',
        label='Geodetic (Hugonnet): mean 2000-2020', zorder=0)
ax.fill_between([2000, 2020], [_mb-_err, _mb-_err],
                [_mb+_err, _mb+_err], alpha=0.5, color='mediumslateblue', zorder=0, label='Geodetic (Hugonnet): error bounds $\pm$1$\sigma$ 2000-2020')

ax.set_ylabel("Specific Mass Balance (m w.e. yr$^{-1}$")
ax.set_xlabel("Year")
ax.set_title("RGI reg. 13, regional average Specific Mass Balance")
plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.3), ncols=3)
