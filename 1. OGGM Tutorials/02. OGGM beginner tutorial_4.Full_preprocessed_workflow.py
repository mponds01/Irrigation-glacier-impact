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
import rioxarray as rioxr

import pandas as pd
import numpy as np
from matplotlib import animation
from IPython.display import HTML, display

cmap=cm.get_cmap('bone')
colors=[cmap(i / 5) for i in range(6)]


folder_path='/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/06. OGGM Working Directories'
wd_path=f'{folder_path}/OGGM-GettingStarted-Beginner-6.OGGM shop'
print(wd_path)
#OGGM option
cfg.initialize(logging_level='WARNING')
cfg.PARAMS['continue_on_error']=True
cfg.PARAMS['use_multiprocessing']=False
cfg.PARAMS['border']=80
cfg.PATHS['working_dir'] = utils.mkdir(wd_path, reset=False)
print(cfg.PATHS['working_dir'] )


#define the glaciers we will play with
rgi_ids=['RGI60-13.37574', 'RGI60-13.37682', 'RGI60-13.37753',  'RGI60-13.53223', 'RGI60-13.53720']

#preparing the glacier data

# Where to fetch the pre-processed directories
base_url ='https://cluster.klima.uni-bremen.de/data/gdirs/dems_v2/default'
# gdirs = workflow.init_glacier_directories(rgi_ids, from_prepro_level=3, prepro_base_url=base_url, prepro_border=80)

# gdir.get_filepath('dem')

gdirs = workflow.init_glacier_directories(rgi_ids, from_prepro_level=1, 
                                          prepro_base_url=base_url,
                                          prepro_rgi_version=62,
                                          prepro_border=10)

# f,axes=plt.subplots(1,5,figsize=(18,4), sharey=True, sharex=True)
# axes.flatten()

# for i, (ax,rgi_id) in enumerate(zip(axes,rgi_ids)):
#     gdir=gdirs[i]
#     graphics.plot_googlemap(gdir, figsize=(8, 7), ax=ax)
#     ax.set_title(rgi_id)

# sources = [src for src in os.listdir(gdir.dir) if src in utils.DEM_SOURCES]
# print('Available DEM sources:', sources)

# ods = xr.Dataset()
# for src in sources:
#     demfile = os.path.join(gdir.dir, src) + '/dem.tif'
#     with rioxr.open_rasterio(demfile) as ds:
#         data = ds.sel(band=1).load() * 1.
#         ods[src] = data.where(data > -100, np.NaN)

#     sy, sx = np.gradient(ods[src], gdir.grid.dx, gdir.grid.dx)
#     ods[src + '_slope'] = ('y', 'x'),  np.arctan(np.sqrt(sy**2 + sx**2))

# with rioxr.open_rasterio(gdir.get_filepath('glacier_mask')) as ds:
#     ods['mask'] = ds.sel(band=1).load()
    
# Decide on the number of plots and figure size
# ns = len(sources)
# n_col = 3
# x_size = 12
# n_rows = -(-ns // n_col)
# y_size = x_size / n_col * n_rows

# from mpl_toolkits.axes_grid1 import AxesGrid
# import salem
# import matplotlib.pyplot as plt

# smap = salem.graphics.Map(gdir.grid, countries=False)
# smap.set_shapefile(gdir.read_shapefile('outlines'))
# smap.set_plot_params(cmap='topo')
# smap.set_lonlat_contours(add_tick_labels=False)
# smap.set_plot_params(vmin=np.nanquantile([ods[s].min() for s in sources], 0.25),
#                      vmax=np.nanquantile([ods[s].max() for s in sources], 0.75))

# fig = plt.figure(figsize=(x_size, y_size))
# grid = AxesGrid(fig, 111,
#                 nrows_ncols=(n_rows, n_col),
#                 axes_pad=0.7,
#                 cbar_mode='each',
#                 cbar_location='right',
#                 cbar_pad=0.1
#                 )

# for i, s in enumerate(sources):
#     data = ods[s]
#     smap.set_data(data)
#     ax = grid[i]
#     smap.visualize(ax=ax, addcbar=False, title=s)
#     if np.isnan(data).all():
#         grid[i].cax.remove()
#         continue
#     cax = grid.cbar_axes[i]
#     smap.colorbarbase(cax)

# # take care of uneven grids
# if ax != grid[-1]:
#     grid[-1].remove()
#     grid[-1].cax.remove()

""" Downloading velocity fields"""

# this will download severals large dataset (2 times a few 100s of MB)
from oggm.shop import its_live, rgitopo
workflow.execute_entity_task(rgitopo.select_dem_from_dir, gdirs, dem_source='COPDEM90', keep_dem_folders=True);
workflow.execute_entity_task(tasks.glacier_masks, gdirs);
workflow.execute_entity_task(its_live.velocity_to_gdir, gdirs);

f,axes=plt.subplots(1,5,figsize=(40,20), sharex=True)
axes.flatten()

for i, (ax,rgi_id) in enumerate(zip(axes,rgi_ids)):
    gdir=gdirs[i]
    with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
        ds = ds.load()
    ds
    
    # plot the salem map background, make countries in grey
    smap = ds.salem.get_map(countries=False)
    smap.set_shapefile(gdir.read_shapefile('outlines'))
    smap.set_topography(ds.topo.data);
        
    # get the velocity data
    u = ds.itslive_vx.where(ds.glacier_mask == 1)
    v = ds.itslive_vy.where(ds.glacier_mask == 1)
    ws = (u**2 + v**2)**0.5
    
    # get the axes ready    
    # Quiver only every N grid point
    us = u[1::3, 1::3]
    vs = v[1::3, 1::3]
    
    smap.set_data(ws)
    smap.set_cmap('Blues')
    smap.plot(ax=ax)
    smap.append_colorbar(ax=ax, label='ice velocity (m yr$^{-1}$)')
    
    # transform their coordinates to the map reference system and plot the arrows
    xx, yy = smap.grid.transform(us.x.values, us.y.values, crs=gdir.grid.proj)
    xx, yy = np.meshgrid(xx, yy)
    qu = ax.quiver(xx, yy, us.values, vs.values)
    qk = ax.quiverkey(qu, 0.82, 0.97, 1000, '1000 m yr$^{-1}$',
                      labelpos='E', coordinates='axes')
    ax.set_title(rgi_id);
    plt.tight_layout()
