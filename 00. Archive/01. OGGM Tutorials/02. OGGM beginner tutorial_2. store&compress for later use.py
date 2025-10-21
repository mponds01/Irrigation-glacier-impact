# -*- coding: utf-8 -*-

""" 1. Working with RGI files """
from oggm import utils, cfg, workflow, tasks, DEFAULT_BASE_URL, graphics, global_tasks
from oggm.core import massbalance, flowline
from oggm.utils import floatyear_to_date, hydrodate_to_calendardate
from oggm.sandbox import distribute_2d
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import xarray as xr
import os

import pandas as pd
import numpy as np
from matplotlib import animation
from IPython.display import HTML, display

cmap=cm.get_cmap('bone')
colors=[cmap(i / 5) for i in range(6)]


folder_path='/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/06. OGGM Working Directories'
wd_path=f'{folder_path}/OGGM-GettingStarted-Beginner-2. Glacier Area and Thickness'

# cfg.PATHS['working_dir']  = utils.mkdir(wd_path, reset=True)
rgi_ids=['RGI60-13.37574', 'RGI60-13.37682', 'RGI60-13.37753',  'RGI60-13.53223', 'RGI60-13.53720']

cfg.initialize(logging_level='WARNING')
cfg.PATHS['working_dir']  = utils.mkdir(wd_path, reset=True)

gdirs = workflow.init_glacier_directories(rgi_ids, prepro_base_url=DEFAULT_BASE_URL, from_prepro_level=4, prepro_=80)

""" Plot initial thickness """
# f, axes = plt.subplots(1, 5, figsize=(20, 4))
# axes = axes.flatten()

# for i, (ax, rgi_id) in enumerate(zip(axes, rgi_ids)):
    
#     gdir = gdirs[i]

#     tasks.run_random_climate(gdir, ys=2020, ye=2100,
#                               y0=2009, halfsize=10, #random climate 1999-2019,
#                               seed=1, #rtandom number generator seed
#                               temperature_bias=1.5, #additional warming - change for other scenarios
#                               store_fl_diagnostics=True, #important! This will be needed for the redistrubtion
#                               init_model_filesuffix='_spinup_historical', #start from the spinup
#                               output_filesuffix='_random_s1',
#                               )

    
#     distribute_2d.add_smoothed_glacier_topo(gdir) #to add new topography to the file
#     tasks.distribute_thickness_per_altitude(gdir) #to get the bed map at the start of the simulation
#     distribute_2d.assign_points_to_band(gdir) #to prepare the glacier direcotry for the interpolation (needs to be done only once)
       
#     with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
#         ds.load()
#         im = ds.distributed_thickness.plot(ax=ax, add_colorbar=False)
#     ax.axis('equal')
#     ax.set_title(rgi_id)
#     ax.legend([rgi_id])
    
#     # Set y-label only for the first subplot
#     if i != 0:
#         ax.set_ylabel('')
        
#     # Which points belongs to which band, and then within one band which are the first to melt

# # Add a single colorbar to the last subplot
# cbar = f.colorbar(im, ax=axes[-1], orientation='vertical')
# cbar.set_label('Thickness')

# # Adjust layout to make room for the colorbar and titles
# plt.tight_layout()
# plt.show()

""" redistribute area volume RGI date simulation """
# f, axes = plt.subplots(1, 5, figsize=(20, 4))
# axes = axes.flatten()
# for i, (ax, rgi_id) in enumerate(zip(axes, rgi_ids)):
#     gdir = gdirs[i]

#     tasks.run_random_climate(gdir, ys=2020, ye=2100,
#                               y0=2009, halfsize=10, #random climate 1999-2019,
#                               seed=1, #rtandom number generator seed
#                               temperature_bias=1.5, #additional warming - change for other scenarios
#                               store_fl_diagnostics=True, #important! This will be needed for the redistrubtion
#                               init_model_filesuffix='_spinup_historical', #start from the spinup
#                               output_filesuffix='_random_s1',
#                               )
#     distribute_2d.add_smoothed_glacier_topo(gdir) #to add new topography to the file
#     tasks.distribute_thickness_per_altitude(gdir) #to get the bed map at the start of the simulation
#     distribute_2d.assign_points_to_band(gdir) #to prepare the glacier direcotry for the interpolation (needs to be done only once)
    
#     ds = distribute_2d.distribute_thickness_from_simulation(gdir, 
#                                                             input_filesuffix='_random_s1',
#                                                             concat_input_filesuffix='_spinup_historical',
#                                                             )
    
#     with xr.open_dataset(gdir.get_filepath('gridded_simulation', filesuffix='_random_s1')) as ds: 
#         ds = ds.load()
#         #select only the dates in rgi timeframe, these are valid

#         ds = ds.sel(time=slice(gdir.rgi_date, None))
    
     
#     #plotting area development over time
#     # f, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(7,2))
#     # ds.distributed_thickness.sel(time=2000).plot(ax=ax1,vmax=400)
#     # ds.distributed_thickness.sel(time=2050).plot(ax=ax2,vmax=400)
#     # ds.distributed_thickness.sel(time=2100).plot(ax=ax3,vmax=400)
#     # ax1.set_title('2000'); ax2.set_title('2050'); ax3.set_title('2100');
#     # ax1.axis('equal'); ax2.axis('equal'); ax3.axis('equal'); plt.tight_layout()
    
#     #plotting area continuous over time
#     # area = (ds.distributed_thickness > 0).sum(dim=['x', 'y']) * gdir.grid.dx**2 * 1e-6
#     # area.plot(ax=ax, label='Distributed area', color=colors[i]);
#     # ax.hlines(gdir.rgi_area_km2, gdir.rgi_date,2100, color=colors[i], linestyles='--', label='RGI Area');
#     # print(i) 
#     # if i==0:
#     #     ax.set_ylabel('Area [km2]');
#     #     ax.legend(loc='lower left'); 
#     # else:
#     #     ax.set_ylabel('')
    
#     #plotting volume conintuous over time
#     vol = ds.distributed_thickness.sum(dim=['x', 'y']) * gdir.grid.dx**2 * 1e-9
#     vol.plot(ax=ax, label='Distributed volume', color=colors[i]); 
#     if i==0: 
#         ax.set_ylabel('Distributed volume [km3]');
#         ax.legend(loc='lower left'); 
#     else:
#         ax.set_ylabel('')
    
#     ax.set_title(rgi_id)

""" Creating an animation in OGGM & finetuning the visualisation"""


# axes = axes.flatten()
# for (i, rgi_id) in enumerate(rgi_ids):
#     gdir = gdirs[i]
#     fig, ax = plt.subplots()
    
#     tasks.run_random_climate(gdir, ys=2020, ye=2100,
#                               y0=2009, halfsize=10, #random climate 1999-2019,
#                               seed=1, #rtandom number generator seed
#                               temperature_bias=1.5, #additional warming - change for other scenarios
#                               store_fl_diagnostics=True, #important! This will be needed for the redistrubtion
#                               init_model_filesuffix='_spinup_historical', #start from the spinup
#                               output_filesuffix='_random_s1',
#                               )
#     distribute_2d.add_smoothed_glacier_topo(gdir) #to add new topography to the file
#     tasks.distribute_thickness_per_altitude(gdir) #to get the bed map at the start of the simulation
#     distribute_2d.assign_points_to_band(gdir) #to prepare the glacier direcotry for the interpolation (needs to be done only once)
    
#     ds = distribute_2d.distribute_thickness_from_simulation(gdir, 
#                                                             input_filesuffix='_random_s1',
#                                                             concat_input_filesuffix='_spinup_historical',
#                                                             )
    
#     with xr.open_dataset(gdir.get_filepath('gridded_simulation', filesuffix='_random_s1')) as ds: 
#         ds = ds.load()
#         #select only the dates in rgi timeframe, these are valid
#         ds = ds.sel(time=slice(gdir.rgi_date, None))
#     thk = ds['distributed_thickness']
    
#     cax = thk.isel(time=0).plot(ax=ax,
#                                 add_colorbar=True,
#                                 cmap='viridis',
#                                 vmin=0, vmax=350,
#                                 cbar_kwargs={
#                                     'extend':'neither'
#                                 }
#                             )
    
#     # if i==0:
#     #     ax.set_ylabel('Distributed volume [km3]');
#     #     ax.legend(loc='lower left'); 
#     # else:
#     #     ax.set_ylabel('')                    
        
#     ax.axis('equal')
#     def animate(frame):
#         ax.set_title(f'{rgi_id} Year {int(thk.time[frame])}')
        
#         cax.set_array(thk.values[frame,:].flatten())
#     ani_glacier = animation.FuncAnimation(fig, animate, frames=len(thk.time), interval=100)
#     HTML(ani_glacier.to_jshtml())
    
    # Write to mp4?
    # FFwriter = animation.FFMpegWriter(fps=10)
    # ani_glacier.save(f'{folder_path}/01. Results/animation_{rgi_id}.mp4', writer=FFwriter)

# f, axes = plt.subplots(1, 5, figsize=(20, 4))
# axes = axes.flatten()
# for i, (ax, rgi_id) in enumerate(zip(axes, rgi_ids)):
    
# for (i, rgi_id) in enumerate(rgi_ids):

#     gdir = gdirs[i]
#     fig, ax = plt.subplots()
    
#     tasks.run_random_climate(gdir, ys=2020, ye=2100,
#                               y0=2009, halfsize=10, #random climate 1999-2019,
#                               seed=1, #rtandom number generator seed
#                               temperature_bias=1.5, #additional warming - change for other scenarios
#                               store_fl_diagnostics=True, #important! This will be needed for the redistrubtion
#                               init_model_filesuffix='_spinup_historical', #start from the spinup
#                               output_filesuffix='_random_s1',
#                               )
#     distribute_2d.add_smoothed_glacier_topo(gdir) #to add new topography to the file
#     tasks.distribute_thickness_per_altitude(gdir) #to get the bed map at the start of the simulation
#     distribute_2d.assign_points_to_band(gdir) #to prepare the glacier direcotry for the interpolation (needs to be done only once)
    
#     # ds = distribute_2d.distribute_thickness_from_simulation(gdir,
#     #                                                            input_filesuffix='_random_s1',
#     #                                                            concat_input_filesuffix='_spinup_historical', 
#     #                                                            )
    
#     #workaround to make more smooth --> rollling mean smoothing
#     ds_smooth = distribute_2d.distribute_thickness_from_simulation(gdir,
#                                                                input_filesuffix='_random_s1',
#                                                                concat_input_filesuffix='_spinup_historical', 
#                                                                ys=2003, ye=2100,  # make the output smaller
#                                                                output_filesuffix='_random_s1_smooth',  # do not overwrite the previous file (optional) 
#                                                                # add_monthly=True,  # more frames! (12 times more - we comment for the demo, but recommend it)
#                                                                rolling_mean_smoothing=7,  # smooth the area time series
#                                                                fl_thickness_threshold=1,  # avoid snow patches to be nisclassified
#                                                                )
    
#     with xr.open_dataset(gdir.get_filepath('gridded_simulation', filesuffix='_random_s1_smooth')) as ds_smooth: 
#         # ds = ds.load()
#         # ds = ds.sel(time=slice(gdir.rgi_date, None))
#         ds_smooth = ds_smooth.load()
#         ds_smooth = ds_smooth.sel(time=slice(gdir.rgi_date, None))
    
#     # area = (ds.distributed_thickness > 0).sum(dim=['x', 'y']) * gdir.grid.dx**2 * 1e-6
#     # area.plot(ax=ax, label='Distributed area (raw)', color=colors[i]);
#     # area = (ds_smooth.distributed_thickness > 0).sum(dim=['x', 'y']) * gdir.grid.dx**2 * 1e-6
#     # area.plot(ax=ax, label='Distributed area (smooth)', color='r', ls='dotted');
#     # plt.legend(loc='lower left'); 
#     # ax.set_title(rgi_id)
    
#     # if i==0:
#     #     ax.set_ylabel('Area [km2]');
#     # else:
#     #     ax.set_ylabel('')

#     thk = ds_smooth['distributed_thickness']
    
#     cax = thk.isel(time=0).plot(ax=ax,
#                                 add_colorbar=True,
#                                 cmap='viridis',
#                                 vmin=0, vmax=350,
#                                 cbar_kwargs={
#                                     'extend':'neither'
#                                 }
#                             )
    
#     # if i==0:
#     #     ax.set_ylabel('Distributed volume [km3]');
#     #     ax.legend(loc='lower left'); 
#     # else:
#     #     ax.set_ylabel('')                    
        
#     ax.axis('equal')
#     def animate(frame):
#         ax.set_title(f'{rgi_id} Year {int(thk.time[frame])} SM')
        
#         cax.set_array(thk.values[frame,:].flatten())
#     ani_glacier_smooth = animation.FuncAnimation(fig, animate, frames=len(thk.time), interval=100)
#     HTML(ani_glacier.to_jshtml())
    
#     # Write to mp4?
#     # FFwriter = animation.FFMpegWriter(fps=10)
#     # ani_glacier_smooth.save(f'{folder_path}/01. Results/smoothened_animation_{rgi_id}.mp4', writer=FFwriter)
    
from IPython.display import Video
Video("../../img/mittelbergferner.mp4", embed=True, width=700)