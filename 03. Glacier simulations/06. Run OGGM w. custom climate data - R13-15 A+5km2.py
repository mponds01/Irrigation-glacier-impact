# %% Cell 0: Load data packages
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
    Cell 4a: gdirs_3r for the region (based on Hugonnet sample glacier dataset)
    Cell 4b: Save gdirs_3r from OGGM
    Cell 4c: Load gdirs_3r from OGGM (saved in 4b)
    Cell 5: Run the MB model for the glaciers 
    Cell 6: Create a histogram with the MB data
    Cell 7: Analyse glacier statistics
      
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
wd_path = f'{folder_path}/03. Modelled perturbation-glacier interactions - R13-15 A+5km2/'
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

# %% Download all the regional shapefiles

regions = [13, 14, 15]  # The RGI regions you want to process

rgi_files = []
for region in regions:
    rgi_file = utils.get_rgi_region_file(region=region)
    rgi_files.append(rgi_file)

rgi_ids = []
for fr in rgi_files:
    gdf = gpd.read_file(fr)
    rgi_ids.append(gdf['RGIId'].values)

# %% Flatten the list of all the rgi_ids
# Flatten the list of arrays and convert all elements to strings
rgi_ids_flattened = [item for sublist in rgi_ids for item in sublist]

print(rgi_ids_flattened)


# %% Cell 1a: ONLY DOWNLOAD ONCE (takes a long time) Get OGGM gdirs_3r for the based on Hugonnet sample glacier dataset

download_gdirs = False

if download_gdirs == True:

    # OGGM options
    oggm.cfg.PATHS['working_dir'] = utils.mkdir(wd_path, reset=False)
    oggm.cfg.PARAMS['store_model_geometry'] = True
    cfg.PARAMS['use_multiprocessing'] = True

    gdirs = []
    with tqdm(total=len(rgi_ids_flattened), desc="Initializing Glacier Directories", mininterval=1.0) as pbar:
        for r, rgi_id in enumerate(rgi_ids_flattened):
            # Print RGI ID to check its correctness
            print(f"Processing RGI ID: {rgi_id}")
            # Perform your processing here
            new_gdirs = workflow.init_glacier_directories(
                [rgi_id],
                prepro_base_url=DEFAULT_BASE_URL,
                from_prepro_level=4,
                prepro_border=80
            )
            gdirs.extend(new_gdirs)

            # Update tqdm progress bar
            pbar.update(1)

# rename gdir file to distinguish from other scripts
gdirs_3r = gdirs
# %% Cell 1b: Save gdirs_3r from OGGM as a pkl file (Save at the end of every working session)

# Save each gdir individually
for gdir in gdirs:
    gdir_path = os.path.join(pkls, f'{gdir.rgi_id}.pkl')
    with open(gdir_path, 'wb') as f:
        pickle.dump(gdir, f)

# %% Cell 1c: Load gdirs_3r from pkl (fastest way to get started)

wd_path_pkls = f'{wd_path}/pkls/'

gdirs_3r = []
for filename in os.listdir(wd_path_pkls):
    if filename.endswith('.pkl'):
        # f'{gdir.rgi_id}.pkl')
        file_path = os.path.join(wd_path_pkls, filename)
        with open(file_path, 'rb') as f:
            gdir = pickle.load(f)
            gdirs_3r.append(gdir)

# # print(gdirs)
# %% Cell 2a: Filter the gdirs list where the area is greater than 5 km^2

# Pre-load all areas into memory first with a progress bar
areas = [gdir.rgi_area_km2 for gdir in tqdm(
    gdirs_3r, desc="Loading glacier areas")]

# Filter glaciers with area > 5 km², with a progress bar and total set
gdirs_3r_a5 = [gdir for gdir, area in tqdm(zip(gdirs_3r, areas), total=len(
    gdirs_3r), desc="Filtering glaciers") if area > 5]
# %% Cell 2b: Save the subset of gdirs with area larger than 5km2
pkls_subset = os.path.join(wd_path, "pkls_subset")
os.makedirs(pkls_subset, exist_ok=True)

# Save each gdir individually
for gdir in gdirs_3r_a5:
    gdir_path = os.path.join(pkls_subset, f'{gdir.rgi_id}.pkl')
    with open(gdir_path, 'wb') as f:
        pickle.dump(gdir, f)


# %% Cell 3: Create a dataset that contains Gdir Area, volume, rgi date and rgi ID

# only for glaciers with an area larger than 5
# Initialize lists to store output
rgi_data = []
count = 0
# Iterate through each glacier directory
for gdir in gdirs_3r:
    count += 1
    print((count*100)/len(gdirs_3r))  # show progress in percentage
    try:
        # Create a temporary dictionary to hold data for this glacier
        temp_data = {
            'rgi_id': gdir.rgi_id,  # RGI ID
            'rgi_region': gdir.rgi_region,  # RGI region
            'rgi_subregion': gdir.rgi_subregion,  # RGI subregion
            'cenlon': gdir.cenlon,
            'cenlat': gdir.cenlat,
            'rgi_date': gdir.rgi_date,  # Validity date
            'rgi_area_km2': gdir.rgi_area_km2}  # Area corresponding to RGI date

        # Load the model diagnostics
        with xr.open_dataset(gdir.get_filepath('model_diagnostics', filesuffix="_historical")) as ds:
            vol = ds.volume_m3[1]  # Use the most relevant volume data
            temp_data['rgi_volume_km3'] = vol.values * 10**-9  # Convert to km³

        # If no error, append the temp_data to the final list
        rgi_data.append(temp_data)

    except Exception as e:
        # Print error message for debugging
        print(f"Error processing {gdir.rgi_id}: {e}")
        # Do not append the incomplete data

# Create DataFrame if data was collected
if rgi_data:
    df_3r = pd.DataFrame(rgi_data)
else:
    # Empty DataFrame if no data
    df_3r = pd.DataFrame(columns=['rgi_id', 'rgi_region', 'rgi_subregion', 'cenlon', 'cenlat',
                                  'rgi_date', 'rgi_area_km2', 'rgi_volume_km3'])

# Output the DataFrame

df_3r.to_csv(f"{wd_path}/master_gdirs_r3_rgi_date_A_V.csv")
print(df_3r.head)

# %% Cell 3b: Create a dataset that contains Gdir Area, volume, rgi date and rgi ID - parallel processing

# Initialize variables
rgi_data = []
count = 0
total_glaciers = len(gdirs_3r)
save_interval = 100  # Save every 100 glaciers processed
temp_save_path = f"{wd_path}/temp_rgi_data.csv"

# Function to process a single glacier


def process_glacier(gdir):
    try:
        temp_data = {
            'rgi_id': gdir.rgi_id,
            'rgi_region': gdir.rgi_region,
            'rgi_subregion': gdir.rgi_subregion,
            'cenlon': gdir.cenlon,
            'cenlat': gdir.cenlat,
            'rgi_date': gdir.rgi_date,
            'rgi_area_km2': gdir.rgi_area_km2
        }

        # Load model diagnostics
        with xr.open_dataset(gdir.get_filepath('model_diagnostics', filesuffix="_historical")) as ds:
            vol = ds.volume_m3[1]
            temp_data['rgi_volume_km3'] = vol.values * 10**-9

        return temp_data
    except Exception as e:
        print(f"Error processing {gdir.rgi_id}: {e}")
        return None

# Function to save data periodically


def save_data(data, path):
    df = pd.DataFrame(data)
    if os.path.exists(path):
        df.to_csv(path, mode='a', header=False,
                  index=False)  # Append if file exists
    else:
        df.to_csv(path, index=False)  # Create new file if not exists
    print(f"Progress saved to {path}")

# Function to update progress


def show_progress(future):
    global count, rgi_data
    count += 1
    result = future.result()
    if result:
        rgi_data.append(result)

    # Save periodically
    if count % save_interval == 0:
        save_data(rgi_data, temp_save_path)
        rgi_data = []  # Clear memory after saving
    # Check if we should print progress (every 1%)
    if count % (total_glaciers // 100) == 0:
        print(f"Progress: {(count / total_glaciers) * 100:.2f}%")


# Use ThreadPoolExecutor for parallel processing
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = {executor.submit(process_glacier, gdir)
                               : gdir for gdir in gdirs_3r}
    for future in concurrent.futures.as_completed(futures):
        show_progress(future)

# Final save at the end of processing
if rgi_data:  # Save any remaining data
    save_data(rgi_data, temp_save_path)


# %% Cell 3: Plot total cumulative volume vs area (ascending)

# Sort the dataframe by area in descending order and rename to df_sorted
df_sorted = df_3r.sort_values(by='rgi_area_km2', ascending=False)

# Calculate the cumulative volume
df_sorted['cumulative_volume_km3'] = df_sorted['rgi_volume_km3'].cumsum()

# Calculate the total volume and area
total_area = df_sorted['rgi_area_km2'].sum()
total_volume = df_sorted['rgi_volume_km3'].sum()
total_glaciers = len(df_sorted)

# Filter glaciers with area > 10 km²
glaciers_over_5km2 = df_sorted[df_sorted['rgi_area_km2'] > 5]

# Calculate the percentage of glaciers with area > 10 km²
percentage_glaciers_over_5km2 = len(
    glaciers_over_5km2) / len(df_sorted) * 100

# Calculate the percentage of total area and total volume for glaciers with area > 10 km²
percentage_area_over_5km2 = glaciers_over_5km2['rgi_area_km2'].sum(
) / total_area * 100
percentage_volume_over_5km2 = glaciers_over_5km2['rgi_volume_km3'].sum(
) / total_volume * 100

# Create the figure and axis
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot glacier area as a line chart
ax1.set_ylabel('Area (km²)', color='black')
ax1.set_xlabel('# of glaciers', color='black')
ax1.set_yscale('log')

# Create color conditions: orange if area < 10 km², blue otherwise
colors = ['tab:orange' if area <
          5 else 'tab:blue' for area in df_sorted['rgi_area_km2']]

# Plot the glacier area as a bar chart with conditional colors
bars = ax1.bar(range(1, len(df_sorted) + 1),
               df_sorted['rgi_area_km2'], color=colors, alpha=0.6, width=1)

# Plot cumulative volume on the first y-axis (line chart)
# Add second y-axis for area (bar chart)
ax2 = ax1.twinx()
ax2.set_xlabel('Number of Glaciers')
# Change label color to black
ax2.set_ylabel('Cumulative Volume (km³)', color='black')
ax2.plot(range(1, len(df_sorted) + 1),
         df_sorted['cumulative_volume_km3'], color='black', label='Cumulative Volume (km³)')
ax2.tick_params(axis='y', labelcolor='black')

# Add a legend for the bar colors
legend_elements = [
    Line2D([0], [0], color='orange', lw=4, label='Area < 5 km²'),
    Line2D([0], [0], color='blue', lw=4, label='Area ≥ 5 km²'),
    Line2D([0], [0], color='black', lw=2, label='Cumulative Volume (km³)')
]

ax1.legend(handles=legend_elements, loc='lower center',
           bbox_to_anchor=(0.5, -0.2), ncols=3)

# Add annotations for the share of glaciers and their corresponding area and volume percentages
annotation_text = (
    f"Total Number of Glaciers: {total_glaciers}\n"
    f"# Glaciers A > 5 km²: {percentage_glaciers_over_5km2:.2f}%\n"
    f"Area Share: {percentage_area_over_5km2:.2f}% of Regional Area ({total_area:.0f} km²)\n"
    f"Total Volume: {percentage_volume_over_5km2:.2f}% of Regional Volume ({total_volume:.0f} km³)"
)
# Place the annotation in the plot
ax1.annotate(annotation_text, xy=(0.5, 0.5), xycoords='axes fraction',
             fontsize=10, bbox=dict(boxstyle="round,pad=0.3", edgecolor='white', facecolor='white'))

# # Set limits for both axes to start at zero
# ax1.set_xlim(left=-1)  # Set x-axis to start at 0
ax2.set_ylim(bottom=0)  # Set y-axis for cumulative volume to start at 0
# Set y-axis for area to start just above 0 (log scale, cannot be 0)
ax1.set_ylim(bottom=0.01)

# Add title and layout adjustments
plt.title('Cumulative Glacier Volume and Glacier Area in RGI region 13-15 A>5km$^2$')
fig.tight_layout()

# Show plot
plt.show()


# %% Cell 4: Process the Irr climate perturbations for all the gdirs and compile


members = [1, 3, 4, 6, 1]
models = ["IPSL-CM6", "E3SM", "CESM2", "CNRM"]
timeframe = "monthly"

opath_climate = os.path.join(sum_dir, 'climate_historical.nc')
utils.compile_climate_input(
    gdirs_3r_a5, path=opath_climate, filename='climate_historical')

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

        workflow.execute_entity_task(process_perturbation_data, gdirs_3r_a5,
                                     ds_path=ds_path,
                                     y0=None, y1=None,
                                     output_filesuffix=f'_perturbation_{sample_id}')

        opath_perturbations = os.path.join(
            sum_dir, f'climate_historical_perturbation_{sample_id}.nc')
        utils.compile_climate_input(gdirs_3r_a5, path=opath_perturbations, filename='climate_historical',
                                    input_filesuffix=f'_perturbation_{sample_id}',
                                    use_compression=True)


# %% Cell 6: Perturb the climate historical with the processed Irr-perturbations, output is gcm file
cfg.PARAMS['use_multiprocessing'] = True
count = 0
# gdir = gdirs_3r_a5[0]
for gdir in gdirs_3r_a5:
    count += 1
    print(round((count*100)/len(gdirs_3r_a5), 2))
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
        print(sample_id)

        # create lists to store the model output
        mb_ts_mean = []
        mb_ts_all = []
        mb_ts_mean_ext = []
        mb_ts_all_ext = []
        error_ids = []

        # load the gdirs_3r_a5
        for (g, gdir) in enumerate(gdirs_3r_a5):
            try:
                # provide the model flowlines and years for the mbmod
                fls = gdir.read_pickle('model_flowlines')
                years = np.arange(1985, 2014)

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
                        mb_ts_all.append((gdir.rgi_id, years_ext, mb))
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
                # found error: RGI60-13.36875 no flowlines --> 542 to 541 glaciers in selected gdirs_3r_a5
                error_ids.append((gdir.rgi_id, model, member))
                continue
            mean_mb = np.mean(mb_ts)
            mb_ts_mean.append((gdir.rgi_id, mean_mb))

            if model == "W5E5":
                mean_mb_ext = np.mean(mb_ts_ext)
                mb_ts_mean_ext.append((gdir.rgi_id, mean_mb_ext))

        # create a dataframe with the mass balance data of all gdirs_3r_a5
        mb_df_mean = pd.DataFrame(mb_ts_mean, columns=['rgi_id', 'B'])
        mb_df_mean.to_csv(os.path.join(
            sum_dir, f'specific_massbalance_mean_{sample_id}.csv'), index=False)
        mb_ts_df = pd.DataFrame(
            mb_ts_all, columns=['rgi_id', 'Year', 'Mass_Balance'])
        mb_ts_df.to_csv(os.path.join(
            sum_dir, f'specific_massbalance_timeseries_{sample_id}.csv'), index=False)

        if model == "W5E5":  # only for W5E5 also create a dataframe for the extended timeseries
            mb_df_mean_ext = pd.DataFrame(
                mb_ts_mean_ext, columns=['rgi_id', 'B'])
            mb_df_mean_ext.to_csv(os.path.join(
                sum_dir, f'specific_massbalance_mean_extended_{sample_id}.csv'), index=False)
            mb_ts_df_ext = pd.DataFrame(mb_ts_all_ext, columns=[
                                        'rgi_id', 'Year', 'Mass_Balance'])
            mb_ts_df_ext.to_csv(os.path.join(
                sum_dir, f'specific_massbalance_timeseries_extended_{sample_id}.csv'), index=False)

        # Optionally save the list of error cases to a CSV for later review
        if error_ids:
            error_df = pd.DataFrame(
                error_ids, columns=['rgi_id', 'Model', 'Member'])
            error_df.to_csv(os.path.join(
                log_dir, 'Error_Log.csv'), index=False)

# %% Cell 80: Update the master dataset with B, A, RGI ID and location for all the different sample ids (541x15=8115) and the region name

master_df = pd.read_csv(f"{wd_path}/master_gdirs_r3_a5_rgi_date_A_V.csv")
data = []
rgi_ids = []
labels = []

members = [1, 3, 4, 6, 1]
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

subregions_path = "/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/02. QGIS/RGI outlines/GTN-G_O2regions_selected.shp"
subregions = gpd.read_file(subregions_path)
subregions = subregions[['o2region', 'full_name']]
subregions = subregions.rename(columns={'o2region': 'rgi_subregion'})
df_complete_2 = pd.merge(df_complete, subregions, on='rgi_subregion')

new_order = ['rgi_id', 'rgi_region', 'rgi_subregion', 'full_name', 'cenlon', 'cenlat', 'rgi_date',
             'rgi_area_km2', 'rgi_volume_km3', 'sample_id', 'B_noirr', 'B_irr', 'B_delta']

df_complete_2 = df_complete_2[new_order]
df_complete_2.to_csv(f"{wd_path}/master_gdirs_r3_a5_rgi_date_A_V_B.csv")
# Display the final DataFrame
print(df_complete_2)


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
plot_gaussian(ax, mb_ds_members, bin_size, "black", "10-member average",
              zorder=members[m] + member + 1, linestyle="--", gaussian_only=gaussian_only)

# format the plot
ax.set_ylabel("# of glaciers [-]")
ax.set_xlabel("Mean specific mass balance [mm w.e. yr$^{-1}$]")

ax.legend(loc="upper left", bbox_to_anchor=(0, 1))
ax.set_xlim(-1250, 1000)
ax.axvline(0, color='k', linestyle="dashed", lw=1, zorder=20)

plt.tight_layout()
plt.show()

# %% Cell 8a plot mass balance data in histograms and gaussians - by subregion (subplots and one plot for all member options, boxplot and gaussian only option)

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
subplots = False

if subplots == True:
    gaussian_only = False  # Flag to use Gaussian fit only

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 6), sharex=True, sharey=False, gridspec_kw={
        'hspace': 0.15, 'wspace': 0.25})  # create a new figure
    axes = axes.flatten()
else:
    gaussian_only = True  # Flag to use Gaussian fit only
    fig, axes = plt.subplots(figsize=(10, 6), sharex=True, sharey=False, gridspec_kw={
        'hspace': 0.15, 'wspace': 0.25})  # create a new figure


# Define distinct colors for regions
region_colors = {
    13: 'blue',    # Blue for region 13
    14: 'crimson',    # Pink for region 14
    15: 'orange'   # Orange for region 15
}

plot_index = 0


# Load dataset
ds = pd.read_csv(f"{wd_path}master_gdirs_r3_a5_rgi_date_A_V_B_hugo.csv")

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

plt.legend(handles=region_legend_patches, loc="center",
           bbox_to_anchor=(0.5, -0.3), ncols=3)
# plt.legend(handles=region_legend_patches, loc="center",
#                     bbox_to_anchor=(-0.8, -1.3), ncols=3)

ax.set_xlim(-1500, 1000)

plt.tight_layout()
plt.show()


# %% Cell 8B: Plot the specific mass balances using boxplots, different subsets, weighted by region - print median
# Initialize plot
fig, ax = plt.subplots(figsize=(10, 10))

# Load dataset
ds = pd.read_csv(f"{wd_path}master_gdirs_r3_a5_rgi_date_A_V_B_hugo.csv")

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
plt.show()


# %% Cell 8B: Plot the specific mass balances using boxplots, different subsets, weighted by region - print area weighted mean
# Initialize plot
fig, ax = plt.subplots(figsize=(10, 10))

# Load dataset
ds = pd.read_csv(f"{wd_path}master_gdirs_r3_a5_rgi_date_A_V_B_hugo.csv")

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
          bbox_to_anchor=(0.5, -0.15), ncols=3)

ax.axvline(x=0, color='black', linestyle='--', linewidth=1, zorder=0)
ax.set_xlabel('Mean B_noirr and B_irr')
ax.set_ylabel('Regions and Subregions')
ax.set_xlim(-1350, 1000)

# Adjust layout and display the plot
plt.tight_layout()
plt.show()


# %% Cell 9: Run the climate model - Save pkl after running is done , as running takes quite a while

cfg.PARAMS['continue_on_error'] = True
cfg.PARAMS['use_multiprocessing'] = False

# gdir = gdirs_3r_a5[0]
y0_clim = 1985
ye_clim = 2014

members = [1, 3, 4, 6]
models = ["IPSL-CM6", "E3SM", "CESM2", "CNRM"]
for m, model in enumerate(models):
    for member in range(members[m]):
        sample_id = f"{model}.00{member}"
        print(sample_id)
        workflow.execute_entity_task(
            tasks.init_present_time_glacier, gdirs_3r_a5)
        out_id = f'_perturbed_{sample_id}'
        opath = os.path.join(
            sum_dir, f'climate_run_output_perturbed_{sample_id}.nc')
        workflow.execute_entity_task(tasks.run_from_climate_data, gdirs_3r_a5,
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
            gdirs_3r_a5, input_filesuffix=out_id, path=opath)  # compile the run output


# And run the climate model with reference data
workflow.execute_entity_task(tasks.run_from_climate_data, gdirs_3r_a5,
                             ys=y0_clim, ye=ye_clim,
                             output_filesuffix='_baseline_W5E5.000')
opath_base = os.path.join(sum_dir, 'climate_run_output_baseline_W5E5.000.nc')
ds_base = utils.compile_run_output(
    gdirs_3r_a5, input_filesuffix='_baseline_W5E5.000', path=opath_base)

# %% Cell 10: Create output plots for Area and volume

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

# %% Cell 11: Create output plots for Area and volume - per region

ds = pd.read_csv(f"{wd_path}master_gdirs_r3_a5_rgi_date_A_V_B_hugo.csv")

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


# %% Cell 12: Open all hugonnet data and align with rgi_ids from r3_a5


def str_to_float(value):
    return float(value)


all_data = pd.DataFrame(columns=['B', 'rgi_id'])
regions = [13, 14, 15]
for region in regions:
    Hugonnet_path = f"/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/04. Reference/01. Climate data/Hugonnet/aggregated_2000_2020/{region}_mb_glspec.dat"
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

    hugo_ds = pd.DataFrame(data_rows, columns=header)
    for col in hugo_ds.columns:
        if col != 'RGI-ID':  # Exclude column 'B'
            hugo_ds[col] = hugo_ds[col].apply(str_to_float)

    # # # create a dataset from the transformed data in order to select the required glaciers
    hugo_ds = hugo_ds.rename(columns={'RGI-ID': 'rgi_id'})

    new_data = pd.DataFrame({
        # Repeat the rgi_id for each B value
        'rgi_id': hugo_ds['rgi_id'].values,
        # Assuming this is a 1D array or single value
        'B': hugo_ds['B'].values

    })

    # Concatenate the new data into the main DataFrame
    all_data = pd.concat([all_data, new_data], ignore_index=True)

master = pd.read_csv(f"{wd_path}master_gdirs_r3_a5_rgi_date_A_V_B.csv")

hugo_ds = pd.merge(master, all_data, on='rgi_id', how='left')
hugo_ds = hugo_ds.rename(columns={'B': 'B_hugo'})
hugo_ds.to_csv(f"{wd_path}master_gdirs_r3_a5_rgi_date_A_V_B_hugo.csv")

hugo_df = hugo_ds[['rgi_id', 'B_hugo']]


# %% Cell 13: Correlation curve

members = [1, 1, 1, 1]  # , 1]
models = ["W5E5", "E3SM", "CESM2", "CNRM"]  # , "IPSL-CM6"]

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

# plt.scatter(mb_base.rgi_id, mb_base.B, color=colors[model][member])

plt.show()
# #


# %% Cell 14: map plot

# to do:
# -  move legend
# - change location of annotation per subregion
# - change data in subplots
# - decide on country outlines (evt with white cover)
# - color coding per subregion


# Configuration
names = "Code"
subregion_blocks = False
region_colors = {13: 'blue', 14: 'crimson', 15: 'orange'}

# Load datasets
df = pd.read_csv(f"{wd_path}master_gdirs_r3_a5_rgi_date_A_V_B_hugo.csv")
master_ds = df[(~df['sample_id'].str.endswith('0')) |
               (df['sample_id'].str.startswith('IPSL'))]

# Normalize values
master_ds[['B_noirr', 'B_irr', 'B_delta']] /= 1000

# Aggregate dataset, area-weighted
master_ds_avg = master_ds.groupby('rgi_id').agg({
    'B_delta': lambda x: (x * master_ds.loc[x.index, 'rgi_area_km2']).sum() / master_ds.loc[x.index, 'rgi_area_km2'].sum(),
    'sample_id': lambda _: "11 member average",
    **{col: 'first' for col in master_ds.columns if col not in ['B_noirr', 'sample_id', 'area']}
})

ds = pd.read_csv(f"{wd_path}master_gdirs_r3_a5_rgi_date_A_V_B_hugo.csv")

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

# Load shapefiles
shapefile_path = '/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/01. Input files/03. Shapefile/Karakoram/Pan-Tibetan Highlands/Pan-Tibetan Highlands (Liu et al._2022)/Shapefile/Pan-Tibetan Highlands (Liu et al._2022)_P.shp'
subregions_path = "/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/02. QGIS/RGI outlines/GTN-G_O2regions_selected.shp"
shp = gpd.read_file(shapefile_path).to_crs('EPSG:4326')
subregions = gpd.read_file(subregions_path).to_crs('EPSG:4326')

# Plot setup
fig, ax = plt.subplots(figsize=(10, 7), subplot_kw={
                       'projection': ccrs.PlateCarree()})
ax.set_extent([45, 120, 13, 55], crs=ccrs.PlateCarree())
shp.plot(ax=ax, edgecolor='red', linewidth=0, facecolor='lightgrey')

# Subregion plotting
centroids = subregions.geometry.centroid

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

# Aggregate data for scatter plot
master_ds_avg['grid_lon'] = np.floor(master_ds_avg['cenlon'])
master_ds_avg['grid_lat'] = np.floor(master_ds_avg['cenlat'])
aggregated_ds = master_ds_avg.groupby(['grid_lon', 'grid_lat'], as_index=False).agg(
    B_delta_aggregated=('B_delta', 'sum'), V_aggregated=('rgi_volume_km3', 'sum')
)
aggregated_ds.rename(
    columns={'grid_lon': 'lon', 'grid_lat': 'lat'}, inplace=True)
gdf = gpd.GeoDataFrame(aggregated_ds, geometry=gpd.points_from_xy(
    aggregated_ds['lon'] + 0.5, aggregated_ds['lat'] + 0.5))

# Colormap and scatter plot
custom_cmap = LinearSegmentedColormap.from_list(
    'red_white_blue', [(1, 0, 0), (1, 1, 1), (0, 0, 1)], N=256)
norm = TwoSlopeNorm(vmin=gdf['B_delta_aggregated'].min(
), vcenter=0, vmax=gdf['B_delta_aggregated'].max())
scatter = ax.scatter(gdf.geometry.x, gdf.geometry.y,
                     s=np.log(gdf['V_aggregated'])*10, c=gdf['B_delta_aggregated'], cmap=custom_cmap, norm=norm, edgecolor='k', alpha=1)

# Add labels, ticks, and colorbar
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
    '$\Delta$ $B_{Irr}$ - $B_{NoIrr}$ (m w.e. yr$^{-1}$)', fontsize=12)

# Add volume legend
# custom_sizes = scatter.legend_elements(
#     prop="sizes", alpha=0.6, func=lambda s: np.exp(s / 20))
# Define the custom sizes (in data units) for the legend
custom_sizes = [5, 50, 500]  # Example sizes for the legend
# Create labels for these sizes
size_labels = [f"{size:.0f}" for size in custom_sizes]

# Create legend handles using matplotlib Patch, simulating the size of scatter points
legend_handles = [plt.scatter([], [], s=np.log(size)*10, edgecolor='k', facecolor='none')
                  for size in custom_sizes]  # Adjust size factor if needed

# Create the custom legend with the defined sizes
fig.legend(legend_handles, size_labels, loc="lower center", title="Total Volume (km$^3$)", title_fontsize=12,
           bbox_to_anchor=(0.22, -0.1), ncol=3, fontsize=12)


# Define and iterate over grid layout
layout = [["13.01", "13.03", "13.04", "13.05", "13.06"], ["13.02", "", "", "", "13.07"], [
    "14.01", "14.02", "", "", "13.08"], ["14.03", "15.01", "15.02", "15.03", "13.09"]]
grid_positions = [[0.12 + col * (0.14 + 0.04), 0.82 - (0.14 + 0.07) * row - 0.05, 0.13, 0.14]
                  if layout[row][col] else None for row in range(4) for col in range(5)]
# Plot subregion time series
for idx, pos in enumerate(grid_positions):
    if pos:
        ax_callout = fig.add_axes(pos)
        region_id = layout[idx // 5][idx % 5]
        print(region_id)
        subregion_ds = ds[ds['rgi_subregion'].str.contains(f"{region_id}")]

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
                        color="blue", linestyle='dashed', lw=2, label="11-member average")
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
fig.show()

# %% Create a table of total area per region included

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
colors = [region_colors.get(region, 'gray') for region in regions]

# Plot the bar chart with the assigned colors
plt.bar(grouped_area.index, grouped_area['rgi_area_km2'], color=colors)

# Add labels and title
plt.xlabel('RGI Subregion')
plt.ylabel('Area (km²)')
plt.title('Total Area by RGI Subregion')
plt.xticks(rotation=90)  # Rotate x-axis labels for readability
plt.show()

# %% Show the area by subregion

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

    colors = [region_colors.get(region, 'gray') for region in regions]

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
                  sorted_areas['rgi_area_km2'], color=colors[i], alpha=0.6, width=1)

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
