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
    "irr": ["#000000", "#555555"],  # Black and dark gray
    # Darker brown and golden yellow for contrast
    "noirr": ["#f5bf03","#fbeaac"],#["#8B5A00", "#D4A017"], #fdde6c
    "noirr_com": ["#E3C565", "#F6E3B0"],  # Lighter, distinguishable tan shades #
    "irr_com": ["#B5B5B5", "#D0D0D0"],  # Light gray, no change
    "cf": ["#004C4C", "#40E0D0"],
    "cf_com": ["#008B8B", "#40E0D0"],
    "cline": ["#e17701", '#ff9408']
}

region_colors = {13: 'blue', 14: 'crimson', 15: 'orange'}


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
wd_path_pkls = f'{wd_path}/pkls_subset_success/'

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
for gdir in gdirs_3r_a5:
    count += 1
    print((count*100)/len(gdirs_3r_a5))  # show progress in percentage
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
    df_3r_a5 = pd.DataFrame(rgi_data)
else:
    # Empty DataFrame if no data
    df_3r_a5 = pd.DataFrame(columns=['rgi_id', 'rgi_region', 'rgi_subregion', 'cenlon', 'cenlat',
                                  'rgi_date', 'rgi_area_km2', 'rgi_volume_km3'])

# Output the DataFrame

df_3r_a5.to_csv(f"{wd_path}/masters/master_gdirs_r3_a5_rgi_date_A_V.csv")
print(df_3r_a5.head)


# %% Cell 3: Update the master dataset with RGI id

df_3r_a5 = pd.read_csv(f"{wd_path}/masters/master_gdirs_r3_a5_rgi_date_A_V.csv")

subregions_path = "/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/03. Data/02. QGIS/RGI outlines/GTN-G_O2regions_selected.shp"
subregions = gpd.read_file(subregions_path)
subregions = subregions[['o2region', 'full_name']]
subregions = subregions.rename(columns={'o2region': 'rgi_subregion'})
df_complete = pd.merge(df_3r_a5, subregions, on='rgi_subregion')

new_order = ['rgi_id', 'rgi_region', 'rgi_subregion', 'full_name', 'cenlon', 'cenlat', 'rgi_date',
             'rgi_area_km2', 'rgi_volume_km3']


df_complete = df_complete[new_order]
df_complete.to_csv(
    f"{wd_path}/masters/master_gdirs_r3_a5_rgi_date_A_V_RGIreg.csv")


# %% Cell 4: Update the master dataset with B, A, RGI ID and location for all the different sample ids (541x15=8115) and the region name for only PT

master_df = pd.read_csv(
    f"{wd_path}/masters/master_gdirs_r3_a5_rgi_date_A_V_RGIreg.csv")


members = [1, 3, 4, 6, 4]
models_shortlist = ["IPSL-CM6", "E3SM", "CESM2", "CNRM", "NorESM"]

for var in ["P","T"]:
    
    data = []
    data_cf = []
    rgi_ids = []
    labels = []
    rgi_ids_sel=[]
    
    # Iterate over models and members, collecting data for boxplots
    # only take the model shortlist as members are handled separately
    for m, model in enumerate(models_shortlist):
        for member in range(members[m]):
            sample_id = f"{model}.0{member:02d}"  # Ensure leading zeros
            i_path = os.path.join(
                sum_dir, f'specific_massbalance_mean_{sample_id}_{pt}_only.csv')
    
            # Load the CSV file into a DataFrame and convert to xarray
            mb = pd.read_csv(i_path, index_col=0).to_xarray()
            print(sample_id, mb)
            # Collect B values for each model and member
            data.append(mb.B.values)            
    
            # Store RGI IDs only for the first model/member
            if m == 0 and member == 0:
                rgi_ids.append(mb.rgi_id.values)
    
            labels.append(sample_id)
    i_path_base = os.path.join(sum_dir, 'specific_massbalance_mean_W5E5.000.csv')
    mb_base = pd.read_csv(i_path_base)
    mb_base = mb_base[mb_base['rgi_id'].isin(rgi_ids[0])]
    base_array = np.array(mb_base.B)
    # Convert the list of data into a NumPy array and transpose it
    data_array = np.array(data)
    # Shape: (number of B values, number of models * members)
    reshaped_data = data_array.T
    # Create a DataFrame for the reshaped data
    df = pd.DataFrame(reshaped_data, index=rgi_ids, columns=np.repeat(labels, 1))
    
    df['B_irr'] = base_array
    # df['B_cf'] = reshaped_data_cf
    
    df.rename_axis(index='rgi_id', inplace=True)
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
    df_complete['B_delta_irr'] = df_complete.B_irr-df_complete.B_noirr
    
    # Merge with rgis_complete
    # master_df = master_ds.to_dataframe()
    df_complete = pd.merge(df_complete, master_df, on='rgi_id', how='inner')
    
    
    new_order = ['rgi_id', 'rgi_region', 'rgi_subregion', 'full_name', 'cenlon', 'cenlat', 'rgi_date',
                 'rgi_area_km2', 'rgi_volume_km3', 'sample_id', 'B_noirr', 'B_irr', 'B_delta_irr']
    
    df_complete = df_complete[new_order]
    # print(df_complete[1:5])
    df_complete.to_csv(
        f"{wd_path}/masters/master_gdirs_r3_a5_rgi_date_A_V_RGIreg_B_{pt}_only.csv")









#%% NOT NEEDED


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
        'Area': hugo_ds['Area'].values,
        # Assuming this is a 1D array or single value
        'B': hugo_ds['B'].values,
        'errB': hugo_ds['errB'].values

    })

    # Concatenate the new data into the main DataFrame
    all_data = pd.concat([all_data, new_data], ignore_index=True)

all_data.to_csv(
    f"{wd_path}masters/master_hugo_only_all_glaciers_13_14_15.csv")
master = pd.read_csv(
    f"{wd_path}masters/master_gdirs_r3_a5_rgi_date_A_V_RGIreg_B.csv")

hugo_ds_tot = pd.merge(master, all_data, on='rgi_id', how='left')
hugo_ds_tot = hugo_ds_tot.rename(columns={'B': 'B_hugo','errB': 'errB_hugo'})
hugo_ds_tot.to_csv(
    f"{wd_path}masters/master_gdirs_r3_a5_rgi_date_A_V_RGIreg_B_hugo.csv")

hugo_df = hugo_ds_tot[['rgi_id', 'B_hugo']]


#%% Add Comitted mass loss for boxplots new master ds


df = pd.read_csv(
    f"{wd_path}masters/master_gdirs_r3_a5_rgi_date_A_V_RGIreg_B_hugo.csv")
# df = pd.read_csv(
#     f"{wd_path}masters/master_gdirs_r3_a5_rgi_date_A_V_RGIreg_B_hugo.csv")
master_ds = df[(~df['sample_id'].str.endswith('0')) |  # Exclude all the model averages ending with 0 except for IPSL
               (df['sample_id'].str.startswith('IPSL'))]

members_averages = [2, 3,  3 ,5, 1 ]
models_shortlist = ["E3SM", "CESM2",  "NorESM",  "CNRM", "IPSL-CM6"]
filepath=f"climate_run_output_baseline_W5E5.000_comitted_random.nc"

# filepath_start=f"climate_run_output_baseline_W5E5.000.nc"

#open and add comitted mass loss 
baseline_path = os.path.join(
    wd_path, "summary", filepath)
baseline = xr.open_dataset(baseline_path)
baseline_end=baseline.sel(time=2264) #use final year
print(len(baseline_end.volume))

com = baseline_end[['volume']].to_dataframe().reset_index()[['rgi_id', 'volume']]
com = com.rename(columns={'volume': 'V_2264_irr'})


merged = pd.merge(master_ds, com, on=['rgi_id'], how='left')

start_baseline = xr.open_dataset(os.path.join( wd_path, "summary", f"climate_run_output_baseline_W5E5.000.nc"))

for time_sel in [1985,2014]:
    #open and add initial simulation volume 
    baseline_start=start_baseline.sel(time=time_sel) #use initial year
    start = baseline_start[['volume']].to_dataframe().reset_index()[['rgi_id', 'volume']]
    start = start.rename(columns={'volume': f'V_{time_sel}_irr'})
    merged = pd.merge(merged, start, on=['rgi_id'], how='left')

all_com_noi = []
for m, model in enumerate(models_shortlist): #Load the data for other models and calculate respective loss for each 
    for j in range(members_averages[m]):
        sample_id = f"{model}.00{j + 1}" if members_averages[m] > 1 else f"{model}.000"
        filepath = f'climate_run_output_perturbed_{sample_id}_comitted_random.nc'
        
        noirr_path = os.path.join(
            wd_path, "summary", filepath)
        noirr_all = xr.open_dataset(noirr_path)
        noirr_all=noirr_all.sel(time=2264) #use final year
        print(len(noirr_all.volume))

        com_noi = noirr_all[['volume']].to_dataframe().reset_index()[['rgi_id', 'volume']]
        com_noi = com_noi.rename(columns={'volume': 'V_2264_noirr'})
        com_noi['sample_id']=sample_id
        all_com_noi.append(com_noi) 
    
all_com_noi_df = pd.concat(all_com_noi, ignore_index=True)

merged = pd.merge(merged, all_com_noi_df, on=['rgi_id', 'sample_id'], how='left')
merged.to_csv(
    f"{wd_path}masters/master_gdirs_r3_a5_rgi_date_A_V_RGIreg_B_hugo_Vcom.csv")

for time_sel in [1985,2014]:
    all_com_noi = []
    for m, model in enumerate(models_shortlist): #Load the data for other models and calculate respective loss for each 
        for j in range(members_averages[m]):
            sample_id = f"{model}.00{j + 1}" if members_averages[m] > 1 else f"{model}.000"
            filepath = f'climate_run_output_perturbed_{sample_id}.nc'
            noirr_path = os.path.join(
                wd_path, "summary", filepath)
            noirr_all = xr.open_dataset(noirr_path)
            noirr_all=noirr_all.sel(time=time_sel) #use final year
            print(len(noirr_all.volume))
            com_noi = noirr_all[['volume']].to_dataframe().reset_index()[['rgi_id', 'volume']]
            com_noi = com_noi.rename(columns={'volume': f'V_{time_sel}_noirr'})
            com_noi['sample_id']=sample_id
            all_com_noi.append(com_noi) 
        
    all_com_noi_df = pd.concat(all_com_noi, ignore_index=True)
    
    merged = pd.merge(merged, all_com_noi_df, on=['rgi_id', 'sample_id'], how='left')
merged.to_csv(
    f"{wd_path}masters/master_gdirs_r3_a5_rgi_date_A_V_RGIreg_B_hugo_Vcom.csv")

#%% 

#%% Add Comitted mass loss for boxplots new master ds


df = pd.read_csv(
    f"{wd_path}masters/master_gdirs_r3_a5_rgi_date_A_V_RGIreg_B_hugo.csv")
# df = pd.read_csv(
#     f"{wd_path}masters/master_gdirs_r3_a5_rgi_date_A_V_RGIreg_B_hugo.csv")
master_ds = df[(~df['sample_id'].str.endswith('0')) |  # Exclude all the model averages ending with 0 except for IPSL
               (df['sample_id'].str.startswith('IPSL'))]

members_averages = [2, 3,  3 ,5]
models_shortlist = ["E3SM", "CESM2",  "NorESM",  "CNRM"]
filepath=f"climate_run_output_baseline_W5E5.000_comitted_random.nc"

# filepath_start=f"climate_run_output_baseline_W5E5.000.nc"

#open and add comitted mass loss 
baseline_path = os.path.join(
    wd_path, "summary", filepath)
baseline = xr.open_dataset(baseline_path)
baseline_end=baseline.sel(time=2264) #use final year
print(len(baseline_end.volume))

com = baseline_end[['volume']].to_dataframe().reset_index()[['rgi_id', 'volume']]
com = com.rename(columns={'volume': 'V_2264_irr'})


merged = pd.merge(master_ds, com, on=['rgi_id'], how='left')

start_baseline = xr.open_dataset(os.path.join( wd_path, "summary", f"climate_run_output_baseline_W5E5.000.nc"))

for time_sel in [1985,2014]:
    #open and add initial simulation volume 
    baseline_start=start_baseline.sel(time=time_sel) #use initial year
    start = baseline_start[['volume']].to_dataframe().reset_index()[['rgi_id', 'volume']]
    start = start.rename(columns={'volume': f'V_{time_sel}_irr'})
    merged = pd.merge(merged, start, on=['rgi_id'], how='left')

all_com_noi = []
for m, model in enumerate(models_shortlist): #Load the data for other models and calculate respective loss for each 
    for j in range(members_averages[m]):
        sample_id = f"{model}.00{j + 1}" if members_averages[m] > 1 else f"{model}.000"
        filepath = f'climate_run_output_perturbed_{sample_id}_comitted_random.nc'
        
        noirr_path = os.path.join(
            wd_path, "summary", filepath)
        noirr_all = xr.open_dataset(noirr_path)
        noirr_all=noirr_all.sel(time=2264) #use final year
        print(len(noirr_all.volume))

        com_noi = noirr_all[['volume']].to_dataframe().reset_index()[['rgi_id', 'volume']]
        com_noi = com_noi.rename(columns={'volume': 'V_2264_noirr'})
        com_noi['sample_id']=sample_id
        all_com_noi.append(com_noi) 
    
all_com_noi_df = pd.concat(all_com_noi, ignore_index=True)

merged = pd.merge(merged, all_com_noi_df, on=['rgi_id', 'sample_id'], how='left')
merged.to_csv(
    f"{wd_path}masters/master_gdirs_r3_a5_rgi_date_A_V_RGIreg_B_hugo_Vcom.csv")

for time_sel in [1985,2014]:
    all_com_noi = []
    for m, model in enumerate(models_shortlist): #Load the data for other models and calculate respective loss for each 
        for j in range(members_averages[m]):
            sample_id = f"{model}.00{j + 1}" if members_averages[m] > 1 else f"{model}.000"
            filepath = f'climate_run_output_perturbed_{sample_id}.nc'
            noirr_path = os.path.join(
                wd_path, "summary", filepath)
            noirr_all = xr.open_dataset(noirr_path)
            noirr_all=noirr_all.sel(time=time_sel) #use final year
            print(len(noirr_all.volume))
            com_noi = noirr_all[['volume']].to_dataframe().reset_index()[['rgi_id', 'volume']]
            com_noi = com_noi.rename(columns={'volume': f'V_{time_sel}_noirr'})
            com_noi['sample_id']=sample_id
            all_com_noi.append(com_noi) 
        
    all_com_noi_df = pd.concat(all_com_noi, ignore_index=True)
    
    merged = pd.merge(merged, all_com_noi_df, on=['rgi_id', 'sample_id'], how='left')
merged.to_csv(
    f"{wd_path}masters/master_gdirs_r3_a5_rgi_date_A_V_RGIreg_B_hugo_Vcom_noIPSL.csv")


# %% Area and volume  evolution- per region

members = [1, 3, 4, 6, 4, 1]
members_averages = [1, 2, 3, 5, 3]
models = ["IPSL-CM6", "E3SM", "CESM2", "CNRM", "NorESM", "W5E5"]
models_shortlist = ["IPSL-CM6", "E3SM", "CESM2", "CNRM", "NorESM"]

for pt in ["T"]: #"P"
    ds = pd.read_csv(
        f"{wd_path}masters/master_gdirs_r3_a5_rgi_date_A_V_RGIreg_B_{pt}_only.csv")
    
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
        'sample_id': 'first'
    }
    
    # Step 1: Group by 'rgi_id', apply custom aggregation
    grouped_ds = ds.groupby('rgi_id').agg(aggregation_functions).reset_index()
    
    # Step 2: Replace the 'sample_id' column with "11-member average"
    grouped_ds['sample_id'] = f'{sum(members_averages)}-member average'
    
    # Step 3: Keep only the required columns
    grouped_ds = grouped_ds[['rgi_id', 'rgi_region', 'rgi_subregion', 'full_name', 'cenlon', 'cenlat', 'rgi_date',
                             'rgi_area_km2', 'rgi_volume_km3', 'sample_id', 'B_noirr', 'B_irr', 'B_delta_irr']]
    
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
                            sum_dir, f'climate_run_output_perturbed_{sample_id}_{pt}_only.nc')
                        climate_run_output = xr.open_dataset(climate_run_opath)
                        filtered_climate_run_output = climate_run_output.where(
                            climate_run_output['rgi_id'].isin(subregion_ds.rgi_id.values), drop=True)
                        if i == 1 and m == 1:
                            label = "GCM member"
                        else:
                            label = "_nolegend_"
                        ax.plot(filtered_climate_run_output["time"], filtered_climate_run_output[var].sum(dim="rgi_id") * factors[v],
                                label=label, color=colors["noirr"][0], linewidth=2, linestyle="dotted", zorder=3)
    
                        # add all the summed volumes/areas to the member list, so a multi-member average can be calculated
                        filtered_member_data.append(filtered_climate_run_output[var].sum(
                            dim="rgi_id").values * factors[v])
    
                # stack the member data
                stacked_member_data = np.stack(filtered_member_data)
                # calculate and plot volume/area 10-member mean
                # mean_values_noirr = np.mean(stacked_member_data, axis=0).flatten()
                # mean_values_cf = np.mean(stacked_member_data_cf, axis=0).flatten()
                mean_values_noirr = np.median(
                    stacked_member_data, axis=0).flatten()
    
                # calculate the volume loss by scenario: irr - noirr and counterfactual
                volume_loss_percentage_noirr = (
                    (mean_values_noirr - mean_values_irr[0]) / mean_values_irr[0]) * 100
                volume_loss_percentage_irr = (
                    (mean_values_irr - mean_values_irr[0]) / mean_values_irr[0]) * 100
                # create a dataframe with the volume loss percentages and absolute values
                loss_df_subregion = pd.DataFrame({
                    'time': climate_run_output["time"].values,
                    'subregion': np.repeat(region_id, len(climate_run_output["time"])),
                    'volume_irr': mean_values_irr,
                    'volume_noirr': mean_values_noirr,
                    'volume_loss_percentage_irr': volume_loss_percentage_irr,
                    'volume_loss_percentage_noirr': volume_loss_percentage_noirr,
                })
    
                uncertainty_data_noi = (stacked_member_data[:, -1]).reshape(1, -1)
                # create a dataframe with the member data of 2014, to use for uncertainty analysis
                row_noi = pd.DataFrame(
                    uncertainty_data_noi, columns=member_data_df_noi.columns, index=[region_id])
                # Concatenate the new row to the DataFrame
                member_data_df_noi = pd.concat([member_data_df_noi, row_noi])
    
                all_loss_dfs.append(loss_df_subregion)
    
                # create a bar chart to show the volume loss by dataset
                ax2.bar(climate_run_output["time"].values[-1]+2, volume_loss_percentage_noirr[-1],
                        color=colors['noirr'][0], label="Volume Loss NoIrr(%)", alpha=0.6, zorder=0)
    
                ax2.bar(climate_run_output["time"].values[-1]+2, volume_loss_percentage_irr[-1],
                        color=colors['irr'][0],  label="Volume Loss Irr (%)", alpha=0.6, zorder=2)
    
    
                ax2.axhline(0, color='black', linestyle='--',
                            linewidth=1.5, zorder=1)  # Dashed line at 0
    
                ax.plot(climate_run_output["time"].values, mean_values_noirr,
                        color=colors["noirr"][0], linestyle='solid', lw=2, label=f"NoIrr ({sum(members_averages)}-member avg)", zorder=3)
    
                # calculate and plot volume/area 10-member min and max for ribbon
                min_values = np.min(stacked_member_data, axis=0).flatten()
                max_values = np.max(stacked_member_data, axis=0).flatten()
                ax.fill_between(climate_run_output["time"].values, min_values, max_values,
                                color=colors["noirr"][1], alpha=0.3, label=f"NoIrr ({sum(members_averages)}-member range)", zorder=3)
    
    
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
        o_file_data = f"{o_folder_data}/1985_2014.{timeframe}.delta.{variable_names[v]}.subregions_{pt}_only.csv"
        o_file_data_uncertainties_noi = f"{o_folder_data}/1985_2014.{timeframe}.delta.{variable_names[v]}.subregions.uncertainties.noi_{pt}_only.csv"
        o_file_data_uncertainties_cf = f"{o_folder_data}/1985_2014.{timeframe}.delta.{variable_names[v]}.subregions.uncertainties.cf_{pt}_only.csv"
        all_loss_dfs_combined.to_csv(o_file_data)
        member_data_df_noi.to_csv(o_file_data_uncertainties_noi)
        member_data_df_cf.to_csv(o_file_data_uncertainties_cf)
        # Adjust the legend
    
