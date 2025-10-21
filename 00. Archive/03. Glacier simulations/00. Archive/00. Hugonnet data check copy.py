#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 15:36:52 2024

@author: magaliponds
"""
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from tqdm import tqdm
import sys

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
#%% Cell 1
Hugonnet_path = '/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/05. Reference/Hugonnet/aggregated_2000_2020/13_mb_glspec.dat'
RGI_data="/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/01. Input files/Shapefile/RGI/General Area/RGI2000-v7.0-C-13_central_asia/RGI2000-v7.0-C-13_central_asia.shp"

df = pd.read_csv(Hugonnet_path, delimiter=',')
# Convert the pandas DataFrame to an Xarray Dataset
ds = xr.Dataset.from_dataframe(df)

rgis=gpd.read_file(RGI_data)
print(rgis)
# rgis=set(rgis.rgi_id.values)
# rgis = [rgi[-5:] for rgi in rgis]

# Now you can use the Xarray object for further analysis
# rgi_ids=['RGI60-13.37574', 'RGI60-13.37682', 'RGI60-13.37753',  'RGI60-13.53223', 'RGI60-13.53720']
rgi_ids=['RGI60-13.01337', 'RGI60-13.37824', 'RGI60-13.00967',  'RGI60-13.38774', 'RGI60-13.40982']

rgi_subset = rgis[rgis['area_km2']>10]

#%% Cell 2
""" Prepare dataset for investigation """

var_index = ds.coords['index'].values
header_str = var_index[1]
data_rows_str = var_index[2:]
# print(header_str, data_rows_str)

#split data input in such a way that it is loaded as dataframe with columns headers and data in table below
header = header_str.split()

# Transform the data type from string values to integers for the relevant columns
data_rows = [row.split() for row in data_rows_str]
def str_to_float(value):
    return float(value)
df = pd.DataFrame(data_rows, columns=header)
for col in df.columns:
    if col != 'RGI-ID':  # Exclude column 'B'
        df[col] = df[col].apply(str_to_float)

#create a dataset from the transformed data in order to select the required glaciers
ds = xr.Dataset.from_dataframe(df)
# df.to_csv("/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/05. Reference/Hugonnet/aggregated_2000_2020/13_mb_glspec-edited.csv")
   
#%% Cell 3     
""" Include only glaciers that are in RGI"""

rgi_ds = ds.where(ds['RGI-ID'].str[-5:].isin(rgis), drop=True)
rgi_ds_excluded = ds.where(~ds['RGI-ID'].str[-5:].isin(rgis),drop=True)

#%% Cell 4: Plot histogram from Hugonnet input data (# glaciers vs B)
""" Find glaciers with most postive MB"""
# print(df.info)

#create a copy of rgi_ds, where the non RGI glaciers have been filtered out
ds_filtered = rgi_ds
conditions = {
    'B': ds_filtered['B'] > 0,
    'errB': ds_filtered['errB'] < 0.2,
    'Area': ds_filtered['Area'] < 10
}

for var_name, condition in conditions.items():
    ds_filtered = ds_filtered.where(condition, drop=True)
    
df_filtered = ds_filtered.to_dataframe().reset_index()

# Sort the DataFrame based on the values column
df_filtered = df_filtered.sort_values(by='B', ascending=False)

plt.hist(df_filtered['B'], bins=np.linspace(df_filtered['B'].min(), df_filtered['B'].max(), 1000 + 1), align='left', rwidth=0.8, color='green')
plt.xlim((0,1.25))

plt.ylabel("# of glaciers [-]")
plt.xlabel("B")
plt.figure()
n, bins, patches = plt.hist(df['B'], bins=np.linspace(df['B'].min(), df['B'].max(), 1000 + 1), align='left', rwidth=0.8)

# Color the bins
for patch, bin_edge in zip(patches, bins):
    if bin_edge < 0:
        patch.set_facecolor('red')
    else:
        patch.set_facecolor('green')
        
plt.xlim((-2,2))
plt.ylabel("# of glaciers [-]")
plt.xlabel("B")

plt.figure()
plt.scatter(df_filtered['errB'], df_filtered['Area'])
# plt.xlim(0,2.5)
# plt.ylim(0,150)
plt.ylabel('Area')
plt.xlabel('errB')
plt.legend()

plt.figure()
plt.scatter(df_filtered['B'], df_filtered['Area'])
# plt.xlim(0,2.5)
# plt.ylim(0,150)
plt.ylabel('Area')
plt.xlabel('B')
plt.legend()

# Save to CSV
# df_filtered.to_csv('/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/04. Output files/02. OGGM/00. Glacier Selection/Hugonnet_Most_Positive_Screened_2000_2020.csv', index=False)

#%% Cell 5: Load data to compare Hugonnet histogram to OGGM results

""" Load the data """
folder_path='/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/06. OGGM Working Directories'
wd_path=f'{folder_path}/01. MB Control Run'
# sns.set_context('notebook') #plot defaults
base_url = 'https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/2023.3/elev_bands/W5E5/'
#configure OGGM with standard warning level, predefined working directory and multi_processing on
oggm.cfg.initialize(logging_level='WARNING')
oggm.cfg.PATHS['working_dir']=utils.mkdir(wd_path, reset=False)
oggm.cfg.PARAMS['store_model_geometry']=True
cfg.PARAMS['use_multiprocessing'] = True


#Create ids for all the glaciers in the region
numbers = np.arange(1, 53387, 1)
# Format each number to be five digits long and prepend the prefix
rgi_ids = [f'RGI60-13.{str(num).zfill(5)}' for num in numbers]
# Print the first few IDs to check
# print(glacier_ids[:10])


#%% Cell 6: Reformat dataset RGI column

def transform_identifier(identifier):
    num = int(identifier.split('-')[-1])
    return f'RGI60-13.{str(num).zfill(5)}'

rgi_subset['rgi_id_format'] = rgi_subset['rgi_id'].apply(transform_identifier)

#%% Cell 7: Gdirs for the region (or part of it to speed up)
rgi_id_sel = rgi_subset.rgi_id_format.values

# for rgi_id in tqdm(rgi_id_sel, desc="Initializing Glacier Directories"):
#     sys.stdout.flush() #flush standard output to ensure immediate display of timer
#     gdir = workflow.init_glacier_directories(
#         [rgi_id],
#         prepro_base_url=DEFAULT_BASE_URL,
#         from_prepro_level=5,
#         prepro_border=80
#     )
#     gdirs.extend(gdir)
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

 #%% Cell 7: Runn Mass Balance model for the glaciers
''' Derive mass balance model and compute specific mass balance for this model '''

#only mb plot
mb_ts_mean = []
for (i,gdir) in enumerate(gdirs):
    
    #calibrated mass balance model - default is to use the OGGM's MonthlyTIModel
    mbmod = massbalance.MultipleFlowlineMassBalance(gdir) 
    fls = gdir.read_pickle('model_flowlines') 
    years = np.arange(2000, 2015) 
    
    # Get specific mass balance time series and compute the mean
    mb_ts = mbmod.get_specific_mb(fls=fls, year=years)
    mean_mb = np.mean(mb_ts)
    mb_ts_mean.append((gdir.rgi_id, mean_mb))
    
    
    if i==1:
        plt.plot(years, mb_ts, color=colors[i], label=gdir.rgi_id)
        plt.ylabel('Specific MB (mm w.e.)');
        
# print(mb_ts_mean)
mb_df = pd.DataFrame(mb_ts_mean, columns=['RGI_ID', 'B'])
print(mb_df)

mb_df['B']=mb_df['B']/100
print(mb_df)


plt.figure()
#creating a histogram with evenly spaced bins, ranging between min and max B values, for a total of 1000 bins
n, bins, patches = plt.hist(mb_df['B'], bins=np.linspace(mb_df['B'].min(), mb_df['B'].max(),len(rgi_ids[1:50])), align='left', rwidth=0.8)

# Color the bins
for patch, bin_edge in zip(patches, bins):
    if bin_edge < 0:
        patch.set_facecolor('red')
    else:
        patch.set_facecolor('green')
plt.ylabel("# of glaciers [-]")
plt.xlabel("B (m w.e.)")
    
    