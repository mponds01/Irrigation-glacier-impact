#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 15:36:52 2024

@author: magaliponds

This code runs through the following operations:
    
SCOPE: Selecting glacier sample for test investigations (varying in Area, w. successful OGGM spinup, based on Hugonnet dataset)
    Cell 1a: Load the Hugonnet data
    Cell 1b: Load RGI dataset and reformat RGI-notation
    Cell 2: Filter the Hugonnet data according to availability in the RGI   
    Cell 3: Plot histogram from Hugonnet input data (# glaciers vs B)
    Cell 4a: Load Glacier IDs in OGGM according to specified conditions (A>10, errB<0.2, B>0)
    Cell 4b: Find glaciers with succesfull OGGM spinup (for small and large files)
    Cell 4c: Get info for the selected glacier subset (using test-glacier IDs)
    Cell 5: Plot selected subglaciers on a map
--> End of glacier subset selection, resulting in glacier subset. 
--> Test glacier subset is further ellaborated using selected glaciers in 01. WGMS observation availability glacier selection (2)
    
"""
# -*- coding: utf-8 -*-import oggm
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

cmap=cm.get_cmap('bone')
colors=[cmap(i / 5) for i in range(6)]
#%% Cell 1a: Load Hugonnet data
Hugonnet_path = '/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/05. Reference/Hugonnet/aggregated_2000_2020/13_mb_glspec.dat'

df = pd.read_csv(Hugonnet_path, delimiter=',')

# Convert the pandas DataFrame to an Xarray Dataset
ds = xr.Dataset.from_dataframe(df)
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
   
#%% Cell 1b: Load RGI dataset and reformat RGI-notation
RGI_data="/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/01. Input files/Shapefile/RGI/General Area/RGI2000-v7.0-C-13_central_asia/RGI2000-v7.0-C-13_central_asia.shp"

rgis=gpd.read_file(RGI_data)
#transform identifier to right format
def transform_identifier(identifier):
    num = int(identifier.split('-')[-1])
    return f'RGI60-13.{str(num).zfill(5)}'

rgis['rgi_id_format'] = rgis['rgi_id'].apply(transform_identifier)

# Now you can use the Xarray object for further analysis

rgi_subset = rgis[rgis['area_km2']>10]

#%%% Cell 1c: Load the WGMS data - not used 
WGMS_info_path ="/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/05. Reference/WGMS-MB/data/glacier.csv"
WGMS_B_path="/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/05. Reference/WGMS-MB/data/mass_balance.csv"


WGMS_B = pd.read_csv(WGMS_B_path)
WGMS_B = xr.Dataset.from_dataframe(WGMS_B)
WGMS_I = pd.read_csv(WGMS_info_path)
WGMS_I = xr.Dataset.from_dataframe(WGMS_I)

condition = (WGMS_I.GLACIER_REGION_CODE == 'ASC') & ((WGMS_I.GLACIER_SUBREGION_CODE == 'ASC-02') | (WGMS_I.GLACIER_SUBREGION_CODE == 'ASC-09'))
WGMS_I_13 = WGMS_I.where(condition, drop=True)
WGMS_I_13['WGMS_ID'] = WGMS_I_13['WGMS_ID'].astype(int)
# print(WGMS_I_13)

objs=[WGMS_I_13,WGMS_B]
#merge data based on selected WGMS-I IDs, all data that do not match dont result in the dataset (hence len WGMS_I should be equal to length of merged dataset)
WGMS = xr.merge(objs, compat="override", join="inner")

condition2 = WGMS['ANNUAL_BALANCE'].notnull()
WGMS = WGMS.where(condition, drop=True)


#%% Cell 2: Filter the Hugonnet data according to availability in the RGI    
""" Include only glaciers that are in RGI"""

rgi_ds = ds.where(ds['RGI-ID'].isin(rgis['rgi_id_format']), drop=True)
# rgi_ds_excluded = ds.where(~ds['RGI-ID'].str[-5:].isin(rgis),drop=True)

#%% Cell 3: Plot histogram from Hugonnet input data (# glaciers vs B)
""" Find glaciers with most postive MB"""

#create a copy of rgi_ds, where the non RGI glaciers have been filtered out
ds_filtered = rgi_ds
conditions = {
    'B': ds_filtered['B'] >= 0,
    'errB': ds_filtered['errB'] < 0.2,
    'Area': ds_filtered['Area']>10
}

for var_name, condition in conditions.items():
    ds_filtered = ds_filtered.where(condition, drop=True)

#create a pandas dataframe of the filtered glaciers in order to sort them on Area
df_filtered = ds_filtered.to_dataframe().reset_index()
df_filtered = df_filtered.sort_values(by='B', ascending=False)
df_filtered_subset=df_filtered[1:11]

plt.hist(df_filtered['B'], bins=np.linspace(df_filtered['B'].min(), df_filtered['B'].max(), 10), align='left', rwidth=0.8, color='green')
plt.xlim((0,1.25))

plt.ylabel("# of glaciers [-]")
plt.xlabel("B")

#%% Cell 4a Load Glacier IDs in OGGM according to specified conditions (A>10, errB<0.2, B>0)

#specify filterd rgis according to conditions B>0 and errB<0.2, A>10 or A<10 epending on size selected above
filtered_rgis = df_filtered['RGI-ID'].values

#specify working directory for OGGM to store glacier statistics
folder_path='/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/06. OGGM Working Directories'
# for small glaciers A<10
# wd_path=f'{folder_path}/03. Glacier Subset/02. Hugonnet_Small/'
#for large glaciers A>10
wd_path=f'{folder_path}/03. Glacier Subset/01. Hugonnet/'


#OGGM options
oggm.cfg.initialize(logging_level='WARNING')
oggm.cfg.PATHS['working_dir']=utils.mkdir(wd_path, reset=False)
oggm.cfg.PARAMS['store_model_geometry']=True
cfg.PARAMS['use_multiprocessing'] = False

gdirs = workflow.init_glacier_directories(
    filtered_rgis, #using RGI IDS specified before
    prepro_base_url=DEFAULT_BASE_URL, #using base_url with W5E5 data
    from_prepro_level=5, #using pre-pro level 5, indicating the model has been calibrated, and uses a pre-computed model run from the RGI outline date to the last possible date given by historical climate data
    prepro_border=80 #ndicates the number of grid points which we’d like to add to each side of the glacier for the local map
    )

#%% Cell 4b Find glaciers with succesfull OGGM spinup (for small and large files)

# utils.compile_glacier_statistics(gdirs)
stats_path = f'{wd_path}glacier_statistics.csv'
glacier_stats = pd.read_csv(stats_path)
glacier_stats=ds = xr.Dataset.from_dataframe(glacier_stats)

spinup_succes = glacier_stats.where(glacier_stats.run_dynamic_spinup_success==True, drop=True).rgi_id.values
ds_filtered = ds_filtered.where(ds_filtered['RGI-ID'].isin(spinup_succes))

df_filtered = ds_filtered.to_dataframe().reset_index()
df_filtered = df_filtered.sort_values(by='B', ascending=False)
# df_filtered.to_csv(f'{wd_path}Hugonnet_A<10_B>0_errB<0.2.csv')
df_filtered.to_csv(f'{wd_path}Hugonnet_A+10_B-0_errB-0.2.csv')

# print(df_filtered.head())
#selected IDs: RGI60-13.00967, RGI60-13.40982

#%% Cell 4c: Open loaded csv files with glacier statistics for smaller and larger glaciers
wd_path_large=f'{folder_path}/03. Glacier Subset/01. Hugonnet/'
wd_path_small=f'{folder_path}/03. Glacier Subset/02. Hugonnet_Small/'

df_filtered_small = f'{wd_path_small}Hugonnet_A-10_B+0_errB-0.2.csv'
df_filtered_large = f'{wd_path_large}Hugonnet_A+10_B+0_errB-0.2.csv'

df_filtered_small = pd.read_csv(df_filtered_small)
ds_filtered_small = xr.Dataset.from_dataframe(df_filtered_small)

df_filtered_large = pd.read_csv(df_filtered_large)
ds_filtered_large = xr.Dataset.from_dataframe(df_filtered_large)

# df_filtered_large = df_filtered_large.sort_values(by='Area', ascending=False)
# df_filtered_large_subset = df_filtered_large[1:10]


#%%% Cell 4d: Find selected WGMS IDS: RGI60-13.41891,RGI60-13.23659

#For areas with WGMS data
selected_ids = ['RGI60-13.41891', 'RGI60-13.23659']

# Use .isin() to create a boolean mask and apply it with .where()
WGMS_selection = rgi_ds.where(rgi_ds['RGI-ID'].isin(selected_ids), drop=True)

wd_path=f'{folder_path}/03. Glacier Subset/00. WGNS/'

#recreate the working directory, if not already available and download the glacier directoreis
# oggm.cfg.PATHS['working_dir']=utils.mkdir(wd_path, reset=False)

# gdirs_WGMS = workflow.init_glacier_directories(
#     selected_ids, #using RGI IDS specified before
#     prepro_base_url=DEFAULT_BASE_URL, #using base_url with W5E5 data
#     from_prepro_level=5, #using pre-pro level 5, indicating the model has been calibrated, and uses a pre-computed model run from the RGI outline date to the last possible date given by historical climate data
#     prepro_border=80 #ndicates the number of grid points which we’d like to add to each side of the glacier for the local map
#     )
# utils.compile_glacier_statistics(gdirs_WGMS)

#compile the glacier statistics & check the spinup success rate
stats_path_WGMS = f'{wd_path}glacier_statistics.csv'
glacier_stats_WGMS = pd.read_csv(stats_path_WGMS)
glacier_stats_WGMS=ds = xr.Dataset.from_dataframe(glacier_stats_WGMS)
spinup_succes_WGMS = glacier_stats_WGMS.where(glacier_stats_WGMS.run_dynamic_spinup_success==True, drop=True).rgi_id.values

#%% Cell 4e: Get info for the selected glacier subset
    
#Define the glacier subset
rgi_subset = ["RGI60-13.40102", # Area >10km2
  "RGI60-13.39195", # Area >10km2
  "RGI60-13.36881", # Area >10km2
  "RGI60-13.38969", # Area >10km2
  "RGI60-13.37184", # Area >10km2
  "RGI60-13.00967", # Area <10km2
  "RGI60-13.40982", # Area <10km2
  "RGI60-13.41891", # WGMS observation
  "RGI60-13.23659", # WGMS observation
]

#provide info on the glacier subset according to Hugonnet dataset
rgi_subset = rgi_ds.where(rgi_ds['RGI-ID'].isin(rgi_subset), drop=True)
rgi_subset_df = rgi_subset.to_dataframe()
rgi_subset_df.to_csv(f'{folder_path}/03. Glacier Subset/Glacier_subset_info.csv')
#%% Cell 4e: load subset info

rgi_subset_df = pd.read_csv(f'{folder_path}/03. Glacier Subset/Glacier_subset_info.csv')
rgi_subset_ds = xr.Dataset.from_dataframe(rgi_subset_df)
print(rgi_subset_ds)
#%% Cell 4d: Plot the selected glaciers on a map
shapefile_path ='/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/01. Input files/Shapefile/Karakoram/Pan-Tibetan Highlands/Pan-Tibetan Highlands (Liu et al._2022)/Shapefile/Pan-Tibetan Highlands (Liu et al._2022)_P.shp'
shp = gpd.read_file(shapefile_path)
target_crs='EPSG:4326'
shp = shp.to_crs(target_crs)

fig, ax = plt.subplots(figsize=(10, 7), subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_extent([60, 110, 20, 50], crs=ccrs.PlateCarree())

# Add country borders
ax.add_feature(cfeature.BORDERS, linestyle='solid', color='grey')
ax.add_feature(cfeature.COASTLINE)

#Include HMA shapefile
shp.plot(ax=ax, edgecolor='red', linewidth=0, facecolor='bisque')

# Add scatter plot for the locations
sc = ax.scatter(rgi_subset_df['lon'], rgi_subset_df['lat'], s=rgi_subset_df['Area']*15, c=rgi_subset_df['B'], cmap='Blues', edgecolor='k', alpha=0.7)

xytext=[(30,20),(-40,-20),(10,10)]
    

# for i, row in rgi_subset_df.iterrows():
#     if row['lon']>76 and row['lon']<90:
#         if i % 2 == 0:
#             j = 0
#         else:
#             j = 1
#     else:
#         j=2
#     print(j)
#     print(xytext[j])
#     ax.annotate(row['RGI-ID'], (row['lon'], row['lat']), xytext=xytext[j], textcoords='offset points',
#             arrowprops=dict(arrowstyle='-', color='grey', shrinkA=0, shrinkB=5))


# Add color bar
cbar = plt.colorbar(sc, ax=ax, orientation='horizontal', pad=0.05, aspect=50, label='B (m w.e.)')

sizes = [1, 5, 10, 20]  # example area sizes
handles = [plt.scatter([], [], s=size*10, edgecolor='k', alpha=0.7, color='white', label=f'{size} km²') for size in sizes]
ax.legend(handles=handles, title='Area Size')

# Set titles and labels
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# Add gridlines
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.gridlines(draw_labels=True)

# Show plot
plt.savefig(f'{folder_path}/glacier_subset_map_bg.png', bbox_inches='tight', pad_inches=0, transparent=False)
plt.show()


#%% Cell 5: Analysie glacier statistics

#Load previously created data file with calibration stats
stats_path = "/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/06. OGGM Working Directories/01. MB Control Run/glacier_statistics.csv"
stats = pd.read_csv(stats_path)
stats= stats.to_xarray()
spinup_stats = stats.run_dynamic_spinup_success
spinup_success = spinup_stats.where(spinup_stats==True,drop=True)
spinup_failure = spinup_stats.where(spinup_stats==False,drop=True)
# plt.hist(dmdtda_not_nan, np.linspace(min(dmdtda_not_nan), max(dmdtda_not_nan), int(len(dmdtda_not_nan)/7)))
print(len(spinup_success))
print(len(spinup_failure))

