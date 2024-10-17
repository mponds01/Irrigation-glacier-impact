# %% Cell 0: Load data packages
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 15:36:52 2024

@author: magaliponds

This script creates a master dataset from OGGM operations and pre-existing data (Hugonnet, RGI)
      
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
# %% Cell 1b: Load gdirs_3r from pkl (fastest way to get started)
wd_path_pkls = f'{wd_path}/pkls_subset/'

gdirs_3r_a5 = []
for filename in os.listdir(wd_path_pkls):
    if filename.endswith('.pkl'):
        # f'{gdir.rgi_id}.pkl')
        file_path = os.path.join(wd_path_pkls, filename)
        with open(file_path, 'rb') as f:
            gdir = pickle.load(f)
            gdirs_3r_a5.append(gdir)
# # print(gdirs)

# %% Cell 2: Create a dataset that contains Gdir Area, volume, rgi date and rgi ID

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

df_3r.to_csv(f"{wd_path}/masters/master_gdirs_r3_rgi_date_A_V.csv")
print(df_3r.head)


# %% Cell 3: Update the master dataset with RGI id

df_3r = pd.read_csv(f"{wd_path}/masters/master_gdirs_r3_rgi_date_A_V.csv")

subregions_path = "/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/02. QGIS/RGI outlines/GTN-G_O2regions_selected.shp"
subregions = gpd.read_file(subregions_path)
subregions = subregions[['o2region', 'full_name']]
subregions = subregions.rename(columns={'o2region': 'rgi_subregion'})
df_complete = pd.merge(df_3r, subregions, on='rgi_subregion')

new_order = ['rgi_id', 'rgi_region', 'rgi_subregion', 'full_name', 'cenlon', 'cenlat', 'rgi_date',
             'rgi_area_km2', 'rgi_volume_km3']


df_complete = df_complete[new_order]
df_complete.to_csv(
    f"{wd_path}/masters/master_gdirs_r3_a5_rgi_date_A_V_RGIreg.csv")


# %% Cell 4: Update the master dataset with B, A, RGI ID and location for all the different sample ids (541x15=8115) and the region name

master_df = pd.read_csv(
    f"{wd_path}/masters/master_gdirs_r3_a5_rgi_date_A_V_RGIreg.csv")
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


new_order = ['rgi_id', 'rgi_region', 'rgi_subregion', 'full_name', 'cenlon', 'cenlat', 'rgi_date',
             'rgi_area_km2', 'rgi_volume_km3', 'sample_id', 'B_noirr', 'B_irr', 'B_delta']

df_complete = df_complete[new_order]
df_complete.to_csv(
    f"{wd_path}/masters/master_gdirs_r3_a5_rgi_date_A_V_RGIreg_B.csv")

# %% TEST Cell 4: Update the master dataset with B, A, RGI ID and location for all the different sample ids (541x15=8115) and the region name

master_df = pd.read_csv(
    f"{wd_path}/masters/master_gdirs_r3_a5_rgi_date_A_V_RGIreg.csv")
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
            sum_dir, f'specific_massbalance_mean_{sample_id}_2000_2014.csv')

        # Load the CSV file into a DataFrame and convert to xarray
        mb = pd.read_csv(i_path, index_col=0).to_xarray()

        # Collect B values for each model and member
        data.append(mb.B.values)

        # Store RGI IDs only for the first model/member
        if m == 0 and member == 0:
            rgi_ids.append(mb.rgi_id.values)

        labels.append(sample_id)

i_path_base = os.path.join(
    sum_dir, 'specific_massbalance_mean_W5E5.000_2000_2014.csv')
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


new_order = ['rgi_id', 'rgi_region', 'rgi_subregion', 'full_name', 'cenlon', 'cenlat', 'rgi_date',
             'rgi_area_km2', 'rgi_volume_km3', 'sample_id', 'B_noirr', 'B_irr', 'B_delta']

df_complete = df_complete[new_order]
df_complete.to_csv(
    f"{wd_path}/masters/master_gdirs_r3_a5_rgi_date_A_V_RGIreg_B_2000_2014.csv")


# %% Cell 5: Open all hugonnet data and align with rgi_ids from r3_a5


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

master = pd.read_csv(
    f"{wd_path}masters/master_gdirs_r3_a5_rgi_date_A_V_RGIreg_B.csv")

hugo_ds = pd.merge(master, all_data, on='rgi_id', how='left')
hugo_ds = hugo_ds.rename(columns={'B': 'B_hugo'})
hugo_ds.to_csv(
    f"{wd_path}masters/master_gdirs_r3_a5_rgi_date_A_V_RGIreg_B_hugo.csv")

hugo_df = hugo_ds[['rgi_id', 'B_hugo']]
