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
""" Include only glaciers that are in RGI in the Hugonnet dataset"""

hugo_ds = ds.where(ds['RGI-ID'].isin(rgis['rgi_id_format']), drop=True)
# hugo_ds_excluded = ds.where(~ds['RGI-ID'].str[-5:].isin(rgis),drop=True)


#%% Cell 3a: Plot histogram from Hugonnet input data (# glaciers vs B)
""" Find glaciers with most postive MB and other conditions"""

#create a copy of hugo_ds, where the non RGI glaciers have been filtered out
hugo_ds_filtered = hugo_ds
conditions = {
    # 'B': ds_filtered['B'] >= 0,
    # 'errB': ds_filtered['errB'] < 0.2,
    'Area': hugo_ds_filtered['Area']>10
}

for var_name, condition in conditions.items():
    hugo_ds_filtered = hugo_ds_filtered.where(condition, drop=True)

#create a pandas dataframe of the filtered glaciers in order to sort them on Area
hugo_df_filtered = hugo_ds_filtered.to_dataframe().reset_index()
hugo_df_filtered = hugo_df_filtered.sort_values(by='B', ascending=False)
# df_filtered_subset=df_filtered[1:11]
plt.figure(figsize=(10,5))
n, bins, patches = plt.hist(hugo_df_filtered['B'], bins=np.linspace(hugo_df_filtered['B'].min(), hugo_df_filtered['B'].max(), int(len(hugo_df_filtered['B'])/7)), align='left', rwidth=0.8, color='green')
# plt.xlim((0,1.25))

plt.ylabel("# of glaciers [-]")
plt.xlabel("B")

# Color the bins
for patch, bin_edge in zip(patches, bins):
    if bin_edge < 0:
        patch.set_facecolor('red')
    else:
        patch.set_facecolor('green')
        
plt.xlim((-2,2))
plt.ylabel("# of glaciers [-]")
plt.xlabel("B (m w.e.)")

def gaussian(x, mu, sigma, A):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

bin_centers = (bins[:-1] + bins[1:]) / 2

# Fit the Gaussian curve to the histogram data
params_hugo, covariance_hugo = curve_fit(gaussian, bin_centers, n, p0=[np.mean(hugo_df_filtered['B']), np.std(hugo_df_filtered['B']), np.max(n)])
x = np.linspace(hugo_df_filtered['B'].min(), hugo_df_filtered['B'].max(), 100)
plt.plot(x, gaussian(x, *params_hugo), color='black', label='Fitted Gaussian Hugonnet')
plt.plot(x, gaussian(x, *params), color='grey', label='Fitted Gaussian OGGM')
# plt.title('Hugonnet derived B')
total_glaciers = len(hugo_df_filtered['B'].values)
plt.annotate(f"Total number of glaciers: {total_glaciers}", xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, verticalalignment='top')
plt.annotate(f"Std: {round(np.std(hugo_df_filtered['B']),2)}", xy=(0.05, 0.9), xycoords='axes fraction', fontsize=12, verticalalignment='top')

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
# df_filtered.to_csv('/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/04. Output files/02. OGGM/00. Glacier Selection/Hugonnet_Most_Positive_Screened_2000_2020.csv', index=False)



#%% Cell 4a: Gdirs for the based on Hugonnet sample glacier dataset
rgi_id_sel = hugo_ds_filtered['RGI-ID'].values
folder_path='/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/06. OGGM Working Directories'
wd_path=f'{folder_path}/02. Saved gdirs (+10km2)/'


#OGGM options
oggm.cfg.initialize(logging_level='WARNING')
oggm.cfg.PATHS['working_dir']=utils.mkdir(wd_path, reset=False)
oggm.cfg.PARAMS['store_model_geometry']=True
cfg.PARAMS['use_multiprocessing'] = False

gdirs=[]
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
        
#%%% Cell 4b: Save gdirs from OGGM

# Directory to save gdirs
save_dir = '/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/06. OGGM Working Directories/02. Saved gdirs (+10km2)/Saved'
os.makedirs(save_dir, exist_ok=True)

# Save each gdir individually
for gdir in gdirs:
    gdir_path = os.path.join(save_dir, f'{gdir.rgi_id}.pkl')
    with open(gdir_path, 'wb') as f:
        pickle.dump(gdir, f)
#%% Cell 4c: Load gdirs from OGGM
save_dir = '/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/06. OGGM Working Directories/02. Saved gdirs (+10km2)/Saved/' 

gdirs = []
for filename in os.listdir(save_dir):
    if filename.endswith('.pkl'):
        file_path = os.path.join(save_dir, filename)
        with open(file_path, 'rb') as f:
            gdir = pickle.load(f)
            gdirs.append(gdir)

#%% Cell 5: Runn Mass Balance model for the glaciers
''' Derive mass balance model and compute specific mass balance for this model '''

#only mb plot
mb_ts_mean = []
count = 0 
# gdirs_sel=gdirs[1:5]
error_ids=[]
for (i,gdir) in enumerate(gdirs):
    count +=1
    print(count/len(gdirs))
    # 

    #calibrated mass balance model - default is to use the OGGM's MonthlyTIModel
    try:
        mbmod = massbalance.MultipleFlowlineMassBalance(gdir) 
        fls = gdir.read_pickle('model_flowlines') 
        years = np.arange(2000, 2015) 
 
        # Get specific mass balance time series and compute the mean
        mb_ts = mbmod.get_specific_mb(fls=fls, year=years)

        mean_mb = np.mean(mb_ts)
        mb_ts_mean.append((gdir.rgi_id, mean_mb))
    
        if i==1:
            plt.plot(years, mb_ts, label=gdir.rgi_id)
            plt.ylabel('Specific MB (mm w.e.)');
    except Exception as e:
        #report the error
        print(f"Error reading data for glacier {gdir.rgi_id}: {str(e)}")
        error_ids.append((gdir.rgi_id, e))
        # Optionally, you can choose to skip the glacier and continue
        continue

        
# print(mb_ts_mean)
mb_df = pd.DataFrame(mb_ts_mean, columns=['rgi_id', 'B'])
print(mb_df)
mb_df.to_csv('/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/04. Output files/02. OGGM/00. Glacier Selection/OGGM_Positive_B_2000_2020.csv', index=False)


#%% Cell5 b: Convert MB data for plotting - checking if the same are in hugo ds
# mb_df_f = mb_df[mb_df['rgi_id'].isin(hugo_ds_filtered['RGI-ID'])]
# print(mb_df)
mb_df['B']=mb_df['B']/1000
print(mb_df)

#%% Cell 6: Create figure with MB data

plt.figure(figsize=(10,5))
#creating a histogram with evenly spaced bins, ranging between min and max B values, for a total of 1000 bins
n, bins, patches = plt.hist(mb_df['B'], bins=np.linspace(mb_df['B'].min(), mb_df['B'].max(),int(len(mb_df['B'])/7)), align='left', rwidth=0.8)

# Color the bins
for patch, bin_edge in zip(patches, bins):
    if bin_edge < 0:
        patch.set_facecolor('red')
    else:
        patch.set_facecolor('green')
plt.xlim((-2,2))
plt.ylabel("# of glaciers [-]")
plt.xlabel("B (m w.e.)")

def gaussian(x, mu, sigma, A):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

bin_centers = (bins[:-1] + bins[1:]) / 2

# Fit the Gaussian curve to the histogram data

params, covariance = curve_fit(gaussian, bin_centers, n, p0=[np.mean(mb_df['B']), np.std(mb_df['B']), np.max(n)])

# Plot the fitted curve
x = np.linspace(mb_df['B'].min(), mb_df['B'].max(), 100)
plt.plot(x, gaussian(x, *params_hugo), color='black', label='Fitted Gaussian Hugonnet')
plt.plot(x, gaussian(x, *params), color='grey', label='Fitted Gaussian OGGM')
total_glaciers = len(mb_df['B'].values)
plt.annotate(f"Total number of glaciers: {total_glaciers}", xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, verticalalignment='top')
plt.annotate(f"Std: {round(np.std(mb_df['B']),2)}", xy=(0.05, 0.9), xycoords='axes fraction', fontsize=12, verticalalignment='top')

plt.legend()
plt.show()


#%% Cell 7: Analyse glacier statistics

# utils.compile_glacier_statistics(gdirs)
#Load previously created data file with calibration stats
stats_path = "/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/06. OGGM Working Directories/02. Saved gdirs (+10km2)/glacier_statistics.csv"
stats = pd.read_csv(stats_path)
stats= stats.to_xarray()
spinup_stats = stats.run_dynamic_spinup_success
spinup_success = spinup_stats.where(spinup_stats==True,drop=True)
spinup_failure = spinup_stats.where(spinup_stats==False,drop=True)
# plt.hist(dmdtda_not_nan, np.linspace(min(dmdtda_not_nan), max(dmdtda_not_nan), int(len(dmdtda_not_nan)/7)))
print(len(spinup_success))
print(len(spinup_failure))

#%% Calculate correlation graph

#load the OGGM modelled B data
mb_df =pd.read_csv('/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/04. Output files/02. OGGM/00. Glacier Selection/OGGM_Positive_B_2000_2020.csv')
mb_df['B']=mb_df['B']/1000
mb_ds = mb_df.set_index('rgi_id').to_xarray()
hugo_ds=hugo_ds_filtered[['RGI-ID', 'B']]
hugo_ds=hugo_ds.set_index(index='RGI-ID')
hugo_ds=hugo_ds.rename({'index': 'rgi_id'})

hugo_ds = hugo_ds.where(hugo_ds.rgi_id.isin(mb_ds.rgi_id), drop=True)

mb_df = mb_ds.to_dataframe()
hugo_df = hugo_ds.to_dataframe()
cor_df = pd.merge(mb_df, hugo_df, on='rgi_id', suffixes=('_oggm', '_hugo'))

cor_ds = cor_df.to_xarray()
correlation = xr.corr(cor_ds['B_oggm'], cor_ds['B_hugo']).round(2)

# cor_ds = xr.merge([mb_ds, hugo_ds], dim='rgi_id')
plt.scatter(cor_ds['B_hugo'], cor_ds['B_oggm'], s=5, color='lightblue')
plt.plot(cor_ds['B_hugo'], cor_ds['B_hugo'], color='k')
plt.text(-1.4,0.2, f'Correlation coefficient: {correlation.values}')

plt.xlabel('B OGGM (mm w.e.)')
plt.ylabel('B hugo (mm w.e.)')
