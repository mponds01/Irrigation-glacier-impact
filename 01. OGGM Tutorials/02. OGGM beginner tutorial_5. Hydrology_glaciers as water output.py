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

import pandas as pd
import numpy as np
from matplotlib import animation
from IPython.display import HTML, display

cmap=cm.get_cmap('bone')
colors=[cmap(i / 5) for i in range(6)]


folder_path='/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/06. OGGM Working Directories'
wd_path=f'{folder_path}/OGGM-GettingStarted-Beginner-3. Water Resources'

sns.set_context('notebook') #plot defaults

#OGGM options
oggm.cfg.initialize(logging_level='WARNING')
oggm.cfg.PATHS['working_dir']=utils.mkdir(wd_path, reset=False)
oggm.cfg.PARAMS['min_ice_thick_for_lenght']=1 #a glacier is defined when ice is thicker than 1 m
oggm.cfg.PARAMS['store_model_geometry']=True

#define the glaciers we will play with
rgi_ids=['RGI60-13.37574', 'RGI60-13.37682', 'RGI60-13.37753',  'RGI60-13.53223', 'RGI60-13.53720']

#preparing the glacier data

base_url = 'https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/2023.3/elev_bands/W5E5'

# f, axes = plt.subplots(1, 5, figsize=(30, 4)) 
# axes=axes.flatten()
# for i, (ax,rgi_id) in enumerate(zip(axes,rgi_ids)):
#     print(rgi_id)
#     gdir = workflow.init_glacier_directories([rgi_id], from_prepro_level=5, prepro_border=80, prepro_base_url=base_url)[0]
    
#     """ Interactive glacier map """
#     #a first glimpse of the glacier of interest, one interactve plot below requires Bokeh
    
#     try: 
#         import holoview as hv
#         hv.extension('bokeh')
#         import geoviews as gv
#         import geoviews.tile_sources as gts
        
#         sh = salem.transform_geopandas(gdir.read_shapefile('outlines'))
#         out = (gv.Polygons(sh).opts(fill_color=None, color_index=None) *
#                 gts.tile_sources['EsriImagery'] *
#                 gts.tile_sources['StamenLabels']).opts(width=800, height=500, active_tools=['pan', 'wheel_zoom'])
#     except:
#         #the rest of the notebook works without this dependency
#         out=None
    
#     fls=gdir.read_pickle('model_flowlines')
#     graphics.plot_modeloutput_section(fls, ax=ax)
#     if i==4:
#         ax.set_ylabel('Altitude (m')
#     else:
#         ax.set_ylabel('')
    
#     ax.set_title(rgi_id)
#     ax.legend(loc='upper center')


#for OGGM, glaciers are 1.5 dimensional along their flowline

""" Generating a glacier in equilibrium with climate """

#lets prepare a run with the run_constant_climate_with_bias tasks from the oggm_edu package

years=np.arange(400)
temp_bias_ts=pd.Series(years * 0. -2, index=years)
# temp_bias_ts.plot(); plt.xlabel('Year'); plt.ylabel('Temperature bias (°C)')

#the temp bias will be described to the standard climate

file_id='_spin_up'
""" Linear climate input"""

ny_s=50 #start
ny_t=150 #trend
ny_e=102 #stabilisation   

temp_bias_ts = np.concatenate([np.full(ny_s, -2.), np.linspace(-2, 0.5, ny_t), np.full(ny_e, 0.5)])
temp_bias_ts = pd.Series(temp_bias_ts, index=np.arange(ny_s + ny_t + ny_e))
temp_bias_ts.plot(); plt.xlabel('Year'); plt.ylabel('Temperature bias (°C)');    
    
file_id2='_lin_temp' 

# we are using hte task run_with_hydro to store hydrological outputs
# along with usual glaciological outputs

f, ax = plt.subplots(2,5,figsize=(15, 7), sharex=True) 
# f, ax = plt.subplots(1,5,figsize=(20, 7)) 

ax=ax.flatten()
for (i, rgi_id) in enumerate(rgi_ids):
    gdir = workflow.init_glacier_directories([rgi_id], from_prepro_level=5, prepro_border=80, prepro_base_url=base_url)[0]
    tasks.run_with_hydro(gdir,
                      temp_bias_ts=temp_bias_ts,
                      run_task=run_constant_climate_with_bias, #which climate
                      y0=2009, halfsize=10, #period which we will average and constantly repeat
                      store_monthly_hydro=True,
                      output_filesuffix=file_id)
    

    #we run the model for 400 yeras, as defined by our control temperature series
    # the model runs with a constant climate averaged over 21 years (2 times halfsize +1) for the period of 1999-2019
    # the model runs with a disequilibrium bias of -2°C
    
    with xr.open_dataset(gdir.get_filepath('model_diagnostics', filesuffix=file_id)) as ds:
        #the last step of the hydrological output is Nan (we cant compute it for this year)
        ds = ds.isel(time=slice(0,-1)).load()
    
    # print(ds)
    
    
    #plot annual evolution of the volume and lenght of the glacier
    # ds.volume_m3.plot(ax=ax[i], color=colors[i])

    # ds.length_m.plot(ax=ax[i+len(rgi_ids)], color=colors[i])
    # plt.tight_layout()
    # # ax[0].set_xlabel(""), ax[0].set_title(rgi_id); ax[1].set_ylabel('Years')
    # if i<5:
    #     ax[i].set_ylim(2.55e10,7e10)
    # else:
    #     ax[i+4].set_ylim(1.9e4,4e4)
        
    # if i<5:
    #     ax[i].set_title(rgi_id)
    
    tasks.run_with_hydro(gdir,
                      temp_bias_ts=temp_bias_ts,  # the temperature bias timeseries we just created
                      run_task=run_constant_climate_with_bias,  # which climate scenario? See following notebook for other examples
                      y0=2009, halfsize=10,  # Period which we will average and constantly repeat
                      store_monthly_hydro=True,  # Monthly ouptuts provide additional information
                      init_model_filesuffix='_spin_up',  # We want to start from the glacier in equibrium we created earlier
                      output_filesuffix=file_id2);  # an identifier for the output file, to read it later
    
    with xr.open_dataset(gdir.get_filepath('model_diagnostics', filesuffix=file_id)) as ds:
    # The last step of hydrological output is NaN (we can't compute it for this year)
        ds = ds.isel(time=slice(0, -1)).load()
    
    # ds.volume_m3.plot(ax=ax[i], color=colors[i])

    # ds.length_m.plot(ax=ax[i+len(rgi_ids)], color=colors[i])
    # plt.tight_layout()
    # if i<5:
    #     ax[i].set_ylim(2.55e10,5e10)
    # else:
    #     ax[i+4].set_ylim(1.9e4,4e4)
       
    # if i<5:
    #     ax[i].set_title(rgi_id)

    """ Annual ronoff: OGGM simulation outcomes for hydrology"""

    # sel_vars = [v for v in ds.variables if 'month_2d' not in ds[v].dims]
    # df_annual = ds[sel_vars].to_dataframe()
    # # Select only the runoff variables
    # runoff_vars = ['melt_off_glacier', 'melt_on_glacier', 'liq_prcp_off_glacier', 'liq_prcp_on_glacier']
    # # Convert them to megatonnes (instead of kg)
    # df_runoff = df_annual[runoff_vars] * 1e-9
    # # We smooth the output, which is otherwize noisy because of area discretization
    # df_runoff = df_runoff.rolling(31, center=True, min_periods=1).mean()
    # # fig, ax = plt.subplots(figsize=(10, 3.5), sharex=True)
    # df_runoff.sum(axis=1).plot(ax=ax[i], color=colors[i]);
    # df_runoff.plot.area(ax=ax[i+5], color=sns.color_palette('rocket'))
    # plt.xlabel('Years')
    # plt.legend()
    # if i==0:
    #     # ax[i].set_ylim(2.55e10,7e10)
    #     ax[i].set_ylabel('Total annual runoff [m3')
    #     ax[i+5].set_ylabel('Runoff (Mt)')
    #     # ax[i+4].set_ylim(1.9e4,4e4)
    # else:
    #     ax[i+5].set_ylabel('')
    #     ax[i].set_ylabel('')

    
    
        
        
    # if i<5:
    #     ax[i].set_title(rgi_id)
    
    """ Monthly runoff"""
    
    # # Select only the runoff variables and convert them to megatonnes (instead of kg)
    # monthly_runoff = ds['melt_off_glacier_monthly'] + ds['melt_on_glacier_monthly'] + ds['liq_prcp_off_glacier_monthly'] + ds['liq_prcp_on_glacier_monthly']
    # monthly_runoff = monthly_runoff.rolling(time=31, center=True, min_periods=1).mean() * 1e-9
    # # monthly_runoff.clip(0).plot(cmap='Blues', cbar_kwargs={'label': 'Runoff (Mt)'}, ax=ax[i]); ax[i].xlabel('Months'); plt.ylabel('Years');
    # # ax[i].set_title(rgi_id)
    # # plt.tight_layout()
    
    # # Compute total precipitation (Snow + Liquid)
    # tot_precip = ds['liq_prcp_off_glacier_monthly'] + ds['liq_prcp_on_glacier_monthly'] + ds['snowfall_off_glacier_monthly'] + ds['snowfall_on_glacier_monthly']
    # tot_precip *= 1e-9  # in Mt
    # yr = 0
    # r = monthly_runoff.sel(time=yr)
    # p = tot_precip.sel(time=yr)
    
    
    # f, ax = plt.subplots(figsize=(10, 6));
    # r.plot(ax=ax[i], color='C3', label='Monthly runoff', linewidth=3);
    # p.plot(ax=ax[i], color=colors[i], label='Monthly precipitation', linewidth=3);
    # ax[i].fill_between(r.month_2d, r, p, where=(p >= r), facecolor=colors[i], interpolate=True, alpha=0.5)
    # ax[i].fill_between(r.month_2d, r, p, where=(r > p), facecolor='C3', interpolate=True, alpha=0.5)
    # ax[i].set_ylabel('Mt yr$^{-1}$'); ax[i].legend(loc='upper center');
    # ax[i].set_xlabel('Month'); ax[i].set_title(rgi_id);
    # ax[i].set_xlim(1,12)
    # plt.tight_layout()
    
    # cmap = sns.color_palette('magma', 3)
    # for j, yr in enumerate([0, 120, 300]):
    #     monthly_runoff.sel(time=yr).plot(ax=ax[i], color=cmap[j], label=f'Year {yr}')
    # ax[i].set_ylabel('Mt yr$^{-1}$'); plt.legend(loc='best');
    # ax[i].set_xlabel('Month'); ax[i].set_title(rgi_id)
    # plt.tight_layout()
    