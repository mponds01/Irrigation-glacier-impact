# -*- coding: utf-8 -*-
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
import geopandas as gpd
import pandas as pd
import numpy as np
from matplotlib import animation
from IPython.display import HTML, display
from scipy import stats
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


folder_path='/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/06. OGGM Working Directories'
wd_path=f'{folder_path}/OGGM-GettingStarted-Beginner-7. Inversion'
print(wd_path)
#OGGM option
cfg.initialize(logging_level='WARNING')
cfg.PARAMS['continue_on_error']=True
cfg.PARAMS['use_multiprocessing']=False
cfg.PARAMS['border']=80
working_dir = utils.mkdir(wd_path, reset=False)
cfg.PATHS['working_dir'] = working_dir

cmap=cm.get_cmap('bone')
colors=[cmap(i / 5) for i in range(5)]
aubergine ='#660066'

#define the glaciers we will play with
rgi_ids=['RGI60-13.37574', 'RGI60-13.37682', 'RGI60-13.37753',  'RGI60-13.53223', 'RGI60-13.53720']

#preparing the glacier data
base_url = 'https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/2023.3/centerlines/W5E5/'

# cfg.PARAMS['melt_f'], cfg.PARAMS['ice_density'], cfg.PARAMS['continue_on_error']

gdirs = workflow.init_glacier_directories(rgi_ids, from_prepro_level=3, prepro_base_url=base_url, prepro_border=80)

# """ Run"""

rgi_region='13'
path = utils.get_rgi_region_file(rgi_region)
# rgidf = gpd.read_file(path)
# rgidf_37 = rgidf.loc[rgidf['02Region']]=='37'
# rgidf_53 = rgidf.loc[rgidf['02Region']]=='53'

#default paramaters: deformation and slidnig parameter
glen_a=2.4e-24
fs = 5.7e-20

gdirs = workflow.init_glacier_directories(rgi_ids, from_prepro_level=3, prepro_base_url=base_url)

cfg.PARAMS['store_model_geometry']=True
workflow.execute_entity_task(tasks.prepare_for_inversion,gdirs)

# with utils.DisableLogger():  # this scraps some output - to use with caution!!!
    
#     # Correction factors
#     factors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
#     factors += [1.1, 1.2, 1.3, 1.5, 1.7, 2, 2.5, 3, 4, 5]
#     factors += [6, 7, 8, 9, 10]

#     # Run the inversions tasks with the given factors
#     for f in factors:
#         # Without sliding
#         suf = '_{:03d}_without_fs'.format(int(f * 10))
#         workflow.execute_entity_task(tasks.mass_conservation_inversion, gdirs,
#                                       glen_a=glen_a*f, fs=0)
#         # Store the results of the inversion only
#         utils.compile_glacier_statistics(gdirs, filesuffix=suf,
#                                           inversion_only=True)

#         # With sliding
#         suf = '_{:03d}_with_fs'.format(int(f * 10))
#         workflow.execute_entity_task(tasks.mass_conservation_inversion, gdirs,
#                                       glen_a=glen_a*f, fs=fs)
#         # Store the results of the inversion only
#         utils.compile_glacier_statistics(gdirs, filesuffix=suf,
#                                           inversion_only=True)

# df = pd.read_csv(os.path.join(working_dir, 'glacier_statistics_011_without_fs.csv'), index_col=0)
# # print(df.index)
# fig, ax = plt.subplots()
# # colors = ['blue', 'green', 'red', 'cyan', 'magenta']  # Customize this as needed
# markers = ['o', 's', '^', 'v', '*']
# for i, (idx, row) in enumerate(df.iterrows()):
#     ax.scatter(row['rgi_area_km2'], row['inv_volume_km3'], color=colors[i % len(colors)], marker=markers[i], label=rgi_ids[i])
# ax.set_xlabel('Area [m$^2$]')
# ax.set_ylabel('Volume [m$^3$]')

# ax.semilogx()
# ax.semilogy()

# # Add a legend
# ax.legend()

# xlim, ylim = [1, 01e3], [1e-1, 1e2]
# ax.set_xlim(xlim); ax.set_ylim(ylim);

# # Fit in log space 
# dfl = np.log(df[['inv_volume_km3', 'rgi_area_km2']])
# slope, intercept, r_value, p_value, std_err = stats.linregress(dfl.rgi_area_km2.values, dfl.inv_volume_km3.values)

# # ax = df.plot(kind='scatter', x='rgi_area_km2', y='inv_volume_km3', label='OGGM glaciers')
# ax.plot(xlim, np.exp(intercept) * (xlim ** slope), color=aubergine, label='Fitted line', ls='dashed')
# ax.semilogx(); ax.semilogy()
# ax.set_xlim(xlim); ax.set_ylim(ylim);
# ax.legend();


"""Sensitivity analysis --> high sensitivity for variations in A """ 

# dftot = pd.DataFrame(index=factors)
# for f in factors:
#     # Without sliding
#     suf = '_{:03d}_without_fs'.format(int(f * 10))
#     fpath = os.path.join(working_dir, 'glacier_statistics{}.csv'.format(suf))
#     _df = pd.read_csv(fpath, index_col=0, low_memory=False)
#     dftot.loc[f, 'without_sliding'] = _df.inv_volume_km3.sum()
    
#     # With sliding
#     suf = '_{:03d}_with_fs'.format(int(f * 10))
#     fpath = os.path.join(working_dir, 'glacier_statistics{}.csv'.format(suf))
#     _df = pd.read_csv(fpath, index_col=0, low_memory=False)
#     dftot.loc[f, 'with_sliding'] = _df.inv_volume_km3.sum()
    
# dftot.plot();
# plt.xlabel('Factor of Glen A (default 1)'); plt.ylabel('Regional volume (km$^3$)');

""" Calibrate to match consensus estimate """ 

# when we use all glaciers, no Glen A could be found within the range [0.1,10] that would match the consensus estimate
# usually, this is applied on larger regions where this error should not occur ! 
# cdf = workflow.calibrate_inversion_from_consensus(gdirs[1:], filter_inversion_output=False)
# cdf.sum()
# cdf.iloc[:5]

""" Distributed ice thickness """ 
workflow.execute_entity_task(tasks.distribute_thickness_per_altitude, gdirs);

# fig, axes = plt.subplots(1,5, sharey=True, figsize=(30,10))
# axes.flatten()
# for i, (ax, rgi_id) in enumerate(zip(axes,rgi_ids)):
#     gdir = gdirs[i]
#     ds = xr.open_dataset(gdir.get_filepath('gridded_data'))
#     f = ds.distributed_thickness.plot(add_colorbar=False, ax=ax);
#     ax.set_title(rgi_id)
# ax.set_ylim(3.89e6, 3.94e6)
# norm = Normalize(vmin=0, vmax=500)  # Adjust according to your data
# sm = ScalarMappable(norm=norm, cmap='viridis')
# sm.set_array([])  # Required for ScalarMappable to work without data

# # Add a colorbar for all subplots
# cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.05, pad=0.1)
# cbar.set_label ('Distributed ice thickenss [-]')

""" PLot multiple glaciers on one map"""

sel_gdirs = [gdir for gdir in gdirs if gdir.rgi_id in rgi_ids]
# fig,ax=plt.subplots() 

# graphics.plot_googlemap(sel_gdirs, ax=ax)
# # x_limits = ax.get_xlim()

# # you might need to install motionless if it is not yet in your environment
# fig,ax=plt.subplots(figsize=(14,4)) 
# graphics.plot_distributed_thickness(sel_gdirs, ax=ax) 
# # plt.tight_layout()

# graphics.plot_inversion(sel_gdirs, extend_plot_limit=True)

""" Using salem """

# Make a grid covering the desired map extent
g = salem.mercator_grid(center_ll=(81, 35.35), extent=(70000, 50000))
# Create a map out of it
smap = salem.Map(g, countries=False)
# Add the glaciers outlines
for gdir in sel_gdirs:
    crs = gdir.grid.center_grid
    geom = gdir.read_pickle('geometries')
    poly_pix = geom['polygon_pix']
    smap.set_geometry(poly_pix, crs=crs, fc='none', zorder=2, linewidth=.2)
    for l in poly_pix.interiors:
        smap.set_geometry(l, crs=crs, color='black', linewidth=0.5)
f, ax = plt.subplots(figsize=(6, 6))
smap.visualize();

# Now add the thickness data
for gdir in sel_gdirs:
    grids_file = gdir.get_filepath('gridded_data')
    with utils.ncDataset(grids_file) as nc:
        vn = 'distributed_thickness'
        thick = nc.variables[vn][:]
        mask = nc.variables['glacier_mask'][:]
    thick = np.where(mask, thick, np.NaN)
    # The "overplot=True" is key here
    # this needs a recent version of salem to run properly
    smap.set_data(thick, crs=gdir.grid.center_grid, overplot=True)

# Set colorscale and other things
smap.set_cmap(graphics.OGGM_CMAPS['glacier_thickness'])
smap.set_plot_params(nlevels=256)
# Plot
f, ax = plt.subplots(figsize=(6, 6))
smap.visualize(ax=ax, cbar_title='Glacier thickness (m)');
