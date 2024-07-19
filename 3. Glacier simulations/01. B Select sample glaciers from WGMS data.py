#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 13:48:31 2024

@author: magaliponds
"""

import pandas as pd
import xarray as xr
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import oggm
from oggm import utils, cfg, workflow, tasks, DEFAULT_BASE_URL, graphics, global_tasks
#%%% Cell 1: Load the data

#Mass balacne overview; overview of surveys, alternatively also point B and B per elevation band are submitted
WGMS_info_path ="/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/05. Reference/WGMS-MB/data/glacier.csv"
WGMS_id_path ="/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/05. Reference/WGMS-MB/data/glacier_id_lut.csv"
WGMS_B_path="/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/05. Reference/WGMS-MB/data/mass_balance.csv"


WGMS_B = pd.read_csv(WGMS_B_path)
WGMS_B_master = xr.Dataset.from_dataframe(WGMS_B)
WGMS_I = pd.read_csv(WGMS_info_path)
WGMS_I = xr.Dataset.from_dataframe(WGMS_I)
WGMS_ID = pd.read_csv(WGMS_id_path)
WGMS_ID = xr.Dataset.from_dataframe(WGMS_ID)
WGMS_ID = WGMS_ID[['RGI60_ID','WGMS_ID']]
WGMS_I = WGMS_I[['WGMS_ID', 'GLACIER_REGION_CODE', 'GLACIER_SUBREGION_CODE']]

condition = (~WGMS_B['SUMMER_BALANCE'].isnull() | ~WGMS_B['WINTER_BALANCE'].isnull() | ~WGMS_B['ANNUAL_BALANCE'].isnull())
WGMS_B = WGMS_B[condition]
WGMS_I = WGMS_I.set_index(index='WGMS_ID')
WGMS_ID = WGMS_ID.set_index(index='WGMS_ID')
WGMS_B_master = WGMS_B_master.set_index(index='WGMS_ID')

# Use `drop duplicates` to select data with unique indices
WGMS_B = WGMS_B_master.drop_duplicates(dim='index')

#Drop nan values in the RGI dataset
WGMS_ID = WGMS_ID.where(~WGMS_ID['RGI60_ID'].isnull(), drop=True)
# Select the desired glacier sub region within Central Asia (ASC) according to RGI
# 13-01: Hissar Alay
# 13-02: Pamir (Safed Khirs / West Tarim)
# 13-03: West Tien Shan
# 13-04: East Tien Shan (Dzhungaria)
# 13-05: West Kun Lun
# 13-06: East Kun Lun (Altyn Tagh)
# 13-07: Qilian Shan
# 13-08: Inner Tibet
# 13-09: Southeast Tibet

#%% Cell 2: Select only relevant data and link it to the MB observations
condition = (WGMS_I.GLACIER_REGION_CODE == 'ASC') #& ((WGMS_I.GLACIER_SUBREGION_CODE == 'ASC-02') | (WGMS_I.GLACIER_SUBREGION_CODE == 'ASC-09'))
WGMS_I_13 = WGMS_I.where(condition, drop=True)

objs=[WGMS_I_13,WGMS_B, WGMS_ID]

#merge data based on selected WGMS-I IDs, all data that do not match dont result in the dataset (hence len WGMS_I should be equal to length of merged dataset)
WGMS = xr.merge(objs, join="inner")
WGMS = WGMS.set_index(index='RGI60_ID')


# condition_2 = WGMS['ANNUAL_BALANCE'].notnull()
# WGMS = WGMS.where(condition_2, drop=True)

#%% Cell 2a: Compare dataset with RGI dataset
RGI_data="/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/01. Input files/Shapefile/RGI/General Area/RGI2000-v7.0-C-13_central_asia/RGI2000-v7.0-C-13_central_asia.shp"

rgis=gpd.read_file(RGI_data)
#transform identifier to right format
def transform_identifier(identifier):
    num = int(identifier.split('-')[-1])
    return f'RGI60-13.{str(num).zfill(5)}'

rgis['rgi_id_format'] = rgis['rgi_id'].apply(transform_identifier)

#%% Cell 2b: Load Hugonnet data

Hugonnet_path = '/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/05. Reference/Hugonnet/aggregated_2000_2020/13_mb_glspec.dat'
RGI_data="/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/01. Input files/Shapefile/RGI/General Area/RGI2000-v7.0-C-13_central_asia/RGI2000-v7.0-C-13_central_asia.shp"

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

#%% Cell 3: include only glaciers that are in the rgi and that have a very positive MB
rgi_ds = ds.where(ds['RGI-ID'].isin(rgis.rgi_id_format), drop=True)
rgi_ds = rgi_ds.set_index(index='RGI-ID')

#%%% CEll 4: Merge Hugonnet dataset with the WGMS observations to see which ones are overlapping

objs_2 =[WGMS, rgi_ds]
Master = xr.merge(objs_2, join="inner")

#%% Cell 4: Plot Hugonnet Mass Balance for data with observations

plt.figure(figsize=(3,2))
n, bins, patches = plt.hist(Master.B, bins=np.linspace(min(Master.B), max(Master.B), int(len(Master.B)/1.5)) )
for patch, bin_edge in zip(patches, bins):
    if bin_edge < 0:
        patch.set_facecolor('red')
    else:
        patch.set_facecolor('green')
plt.xlabel('B (m w.e.)')
plt.ylabel('# of glaciers')
        
subset = Master.where(Master.B>=0, drop=True)
# print(subset)

B_master = WGMS_B_master.where(WGMS_B_master.NAME.isin(subset.NAME),drop=True)
B_m = B_master.to_dataframe()

# Export DataFrame to CSV
B_m.to_csv('/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/04. Output files/02. OGGM/00. Glacier Selection/WGMS_Hugo_B_data_region_13_2000_2020.csv', index=False) 

# print(subset.index)
#%% Cell 5: Test spinup performance of selected glaciers:
    
#specify filterd rgis according to WGMS availability

rgi_id_sel = ['RGI60-13.41891' 'RGI60-13.23659']

#specify working directory for OGGM to store glacier statistics
folder_path='/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/06. OGGM Working Directories'
wd_path=f'{folder_path}/03. Glacier Subset/00. WGMS'

#OGGM options
oggm.cfg.initialize(logging_level='WARNING')
oggm.cfg.PATHS['working_dir']=utils.mkdir(wd_path, reset=False)
oggm.cfg.PARAMS['store_model_geometry']=True
cfg.PARAMS['use_multiprocessing'] = False

gdirs_WGMS = workflow.init_glacier_directories(
    rgi_id_sel, #using RGI IDS specified before
    prepro_base_url=DEFAULT_BASE_URL, #using base_url with W5E5 data
    from_prepro_level=5, #using pre-pro level 5, indicating the model has been calibrated, and uses a pre-computed model run from the RGI outline date to the last possible date given by historical climate data
    prepro_border=80 #ndicates the number of grid points which we’d like to add to each side of the glacier for the local map
    )

utils.compile_glacier_statistics(gdirs_WGMS)