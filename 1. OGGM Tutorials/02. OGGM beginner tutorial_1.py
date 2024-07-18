# -*- coding: utf-8 -*-

""" 1. Working with RGI files """
from oggm import utils, cfg, workflow, tasks, DEFAULT_BASE_URL, graphics, global_tasks
from oggm.core import massbalance, flowline
from oggm.utils import floatyear_to_date, hydrodate_to_calendardate
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import xarray as xr
import os

import pandas as pd
import numpy as np

cmap=cm.get_cmap('bone')
colors=[cmap(i / 5) for i in range(6)]

# utils.get_rgi_dir(version='62')

# fr_14 = utils.get_rgi_region_file(14, version='62')
# fr_13 = utils.get_rgi_region_file(13, version='62')

# gdf_14 = gpd.read_file(fr_14)
# gdf_13 = gpd.read_file(fr_13)


#the geodataframe has almost the same functionalities as pandas DataFrames
#gdf.head

# gdf_14[['Area']].plot(kind='hist', bins=100, logy=True);

#selecting glaciers per attribute
# gdf_14_2 = gdf_14.loc[gdf_14.O2Region == '2'] #karakoram
# gdf_13_2 = gdf_13.loc[gdf_13.O2Region =='2'] #Parmir
# gdf_13_5 = gdf_13.loc[gdf_13.O2Region =='5'] #West Kunlun
# gdf_13_6 = gdf_13.loc[gdf_13.O2Region =='6'] #East Kunlun


# fig, ax = plt.subplots(1,1, figsize=(14,5))
# gdf_14_2.plot(ax=ax, color='red', label='Karakoram', legend=True)
# gdf_13_2.plot(ax=ax, color='orange', label='Pamir', legend=True)
# gdf_13_5.plot(ax=ax, color='pink', label='West Kunlun', legend=True)
# gdf_13_6.plot(ax=ax, color='purple', label='East Kunlun', legend=True)
# ax.legend()
# plt.title('Selected glaciers from RGI')

# glacier entities can also be selectd by their ID



#Use the RGI files for an OGGM run

folder_path='/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/06. OGGM Working Directories'
wd_path=f'{folder_path}/OGGM-GettingStarted-Beginner-1. RGI'

# cfg.PATHS['working_dir']  = utils.mkdir(wd_path, reset=True)

cfg.initialize(logging_level='WARNING')
cfg.PARAMS['continue_on_error']=True
cfg.PARAMS['use_multiprocessing']=False
cfg.PARAMS['border']=80
# cfg.PARAMS['working_dir']=True
# cfg.PATHS['working_dir']  = utils.mkdir(wd_path, reset=True)

#Go - get the pre-processed glacier directories
# gdirs_13_5 = workflow.init_glacier_directories(gdf_13_5, prepro_base_url=DEFAULT_BASE_URL, from_prepro_level=5)
# gdirs_13_2 = workflow.init_glacier_directories(gdf_13_2, prepro_base_url=DEFAULT_BASE_URL, from_prepro_level=5)
# gdirs_14_2 = workflow.init_glacier_directories(gdf_14_2, prepro_base_url=DEFAULT_BASE_URL, from_prepro_level=5)



    
rgi_ids=['RGI60-13.37574', 'RGI60-13.37682', 'RGI60-13.37753',  'RGI60-13.53223', 'RGI60-13.53720']
# for rgi_id in rgi_ids:
#     gdir_sel = utils.get_rgi_glacier_entities([rgi_id], version='62')
#     gdir = workflow.init_glacier_directories(gdir_sel, prepro_base_url=DEFAULT_BASE_URL, from_prepro_level=5)    
#     workflow.execute_entity_task(tasks.run_random_climate, gdir, nyears=100,
#                                  y0=2009, halfsize=10, output_filesuffix='_2000')
#     ds2000 = utils.compile_run_output(gdir, input_filesuffix='_2000')
#     ds2000.sum(dim='rgi_id').volume.plot(label=rgi_id)

""" 2. Plot MB"""
folder_path='/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/06. OGGM Working Directories'
wd_path=f'{folder_path}/OGGM-GettingStarted-Beginner-2. Plot MB'

cfg.PATHS['working_dir']=utils.mkdir(wd_path, reset=True)
cfg.PARAMS['store_model_geometry']=True

base_url = ('https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/'
            'L3-L5_files/2023.1/centerlines/W5E5/')
gdirs = workflow.init_glacier_directories(rgi_ids, from_prepro_level=4,
                                          prepro_base_url=DEFAULT_BASE_URL)
# when using centerlines, the new default `SemiImplicit` scheme does not work at the moment,
# we have to use the `FluxBased` scheme instead:
cfg.PARAMS['evolution_model'] = 'FluxBased'

""" Plotting measured specific MB"""
# for (i,rgi_id) in enumerate(rgi_ids):
#     gdir = gdirs[i]
#     tasks.init_present_time_glacier(gdir)

#     #with a statis geometry
#     # Get the calibrated mass-balance model - the default is to use OGGM's "MonthlyTIModel"
#     mbmod = massbalance.MultipleFlowlineMassBalance(gdir)
    
#     # Compute the specific MB for this glacier
#     fls = gdir.read_pickle('model_flowlines')
#     years = np.arange(1985, 2019)
#     mb_ts = mbmod.get_specific_mb(fls=fls, year=years)
    
#     plt.plot(years, mb_ts, label=rgi_id, color=colors[i]);
#     plt.ylabel('Specific MB (mm w.e.)');
#     plt.xlabel('Year') 
# plt.legend(loc='lower center', ncols=3, bbox_to_anchor=(0.5,-0.35))
# plt.title('Measured specific MB')

    

#sometimes also measured data can be plotted --> but not for the glaciers we selected in this script
# Get the reference mass-balance from the WGMS
# ref_df = gdir.get_ref_mb_data()
# # Get OGGM MB at the same dates
# ref_df['OGGM'] = mbmod.get_specific_mb(fls=fls, year=ref_df.index.values)
# # Plot
# plt.plot(ref_df.index, ref_df.OGGM, label='OGGM');
# plt.plot(ref_df.index, ref_df.ANNUAL_BALANCE, label='WGMS');
# plt.ylabel('Specific MB (mm w.e.)'); plt.legend();
# plt.xlabel('Year');
    
# for flmod in mbmod.flowline_mb_models:
#     print(flmod.melt_f, f'{flmod.temp_bias:.2f}', f'{flmod.prcp_fac:.2f}')
    
#here we have the same mass-balance model parameters for each flowline (that’s frequently the case). 
#To get the mass-balance as a function of height we have several possibilities:
    
#specific mass balance is specific at for one area


""" Plot modelled MB at any height"""

plt.figure()
# for (i,rgi_id) in enumerate(rgi_ids):
#     # if i==1:
#         """ Plotting modelled mass balance at any height"""
#         gdir = gdirs[i]
#         # print(gdir)
#         tasks.init_present_time_glacier(gdir)
    
#         #with a static geometry
#         # Get the calibrated mass-balance model - the default is to use OGGM's "MonthlyTIModel"
#         mbmod = massbalance.MultipleFlowlineMassBalance(gdir)
        
#         # Compute the specific MB for this glacier
#         fls = gdir.read_pickle('model_flowlines')
#         # Calculate the MB where the flowlines are:
#         heights, widths, mb = mbmod.get_annual_mb_on_flowlines(fls, year=2000)
#         # and convert units kg ice per second to mm w.e. per year:
#         mb = mb * cfg.SEC_IN_YEAR * cfg.PARAMS['ice_density'] 
        
#         # Plot
#         # plt.plot(mb, heights, '*', label='2000');
        
#         # Another year:
#         heights, widths, mb = mbmod.get_annual_mb_on_flowlines(fls, year=2001)
#         # units kg ice per second to mm w.e. per year:
#         mb = mb * cfg.SEC_IN_YEAR * cfg.PARAMS['ice_density'] 
#         # Plot
#         # plt.plot(mb, heights, '*', label='2001');
#         plt.ylabel('Elevation (m a.s.l.)'); plt.xlabel('MB (mm w.e. yr$^{-1}$)'); plt.legend();
        
        
#         #Now we calculate the MB at any height, arranging the height, either picking one single flowline MB model or mulitple one
#         # plt.figure()
#         heights = np.arange(2500, 4000)
#         # Method 1: for all flowlines
#         mbmod1 = mbmod.flowline_mb_models[0]
#         mb = mbmod1.get_annual_mb(heights, year=1985) * cfg.SEC_IN_YEAR * cfg.PARAMS['ice_density'] 
#         # Method 2: for multiple flowlines
#         mb2 = mbmod.get_annual_mb(heights, year=1985, fl_id=0) * cfg.SEC_IN_YEAR * cfg.PARAMS['ice_density'] 
#         np.testing.assert_allclose(mb, mb2)
#         plt.plot(mb, heights, label='2001', color=colors[i]);
#         plt.plot(mb2, heights, label='2001 mb2', color=colors[i], ls='dashed');

#         plt.ylabel('Elevation (m a.s.l.)'); plt.xlabel('MB (mm w.e. yr$^{-1}$)'); plt.legend(); 
        
        #for this glacier the MB is similar for both glaciers, but for some it might change from one flowline to another

""" Plot volume evolution change """
plt.figure()
# for (i,rgi_id) in enumerate(rgi_ids):    
#         gdir = gdirs[i]
#          1print(i)
#         tasks.init_present_time_glacier(gdir)
#         
        
#         ''' Modelled volume evolution change '''
#         #we want to use these flowline files for the AAR
#         cfg.PARAMS['store_fl_diagnostics'] = True
#         #run from the outline inventory year (2003 for HEF) to 2017 (end of CRU data in hydro year convention)
#         tasks.run_from_climate_data(gdir, ys=1985, ye=2020)
        
#         ds_diag = xr.open_dataset(gdir.get_filepath('model_diagnostics'))
#         ds_diag.volume_m3.plot(label=rgi_id, color=colors[i], linestyle='dashed'); #plt.title(f'Volume evolution of glaciers in the Karakoram Area')
#         # print(ds_diag.head)
#         plt.ylim(2.35e10, 3.8e10)
#         plt.xlabel('Year') 
#         plt.ylabel('Total glacier volume [m^3]')
#         plt.legend(loc='lower center', ncols=3, bbox_to_anchor=(0.5,-0.35))
#         plt.title('Glacier volume evolution change ')
        

''' Model timestamps in OGGM - using hydrological years'''

print('Hydro dates')
print(floatyear_to_date(2003.0))
print(floatyear_to_date(2003.99))

print('Calendar dates')
print(hydrodate_to_calendardate(*floatyear_to_date(2003.0), start_month=10))
print(hydrodate_to_calendardate(*floatyear_to_date(2003.99), start_month=10))

plt.figure()
# for (i,rgi_id) in enumerate(rgi_ids):    
    # gdir = gdirs[i]
    # print(i)
    # tasks.init_present_time_glacier(gdir)
            
            
    # ''' Modelled volume evolution change '''
    # #we want to use these flowline files for the AAR
    # cfg.PARAMS['store_fl_diagnostics'] = True
    # #run from the outline inventory year (2003 for HEF) to 2017 (end of CRU data in hydro year convention)
    # tasks.run_from_climate_data(gdir, ys=1985, ye=2020)
        
    # ds_diag = xr.open_dataset(gdir.get_filepath('model_diagnostics'))
    # print('rgi_id: ',rgi_id, ds_diag.time.values[0], ds_diag.calendar_year.values[0], ds_diag.calendar_month.values[0])
    # # hydro year and month
    # print(ds_diag.time.values[0], ds_diag.hydro_year.values[0], ds_diag.hydro_month.values[0])
    # """The simulations are almost in accordance with the rgi inventory date"""
    # gdir.rgi_date

""" Calculationg specific mass balacne from volume change time series
calculated using delta volume devidded by the glacier area and then compare it to the observations and mb with fixed geometry"""

# plt.figure()
# for (i,rgi_id) in enumerate(rgi_ids):    
#     gdir = gdirs[i]
#     print(i)
#     tasks.init_present_time_glacier(gdir)
            
            
#     ''' Modelled volume evolution change '''
#     #we want to use these flowline files for the AAR
#     cfg.PARAMS['store_fl_diagnostics'] = True
#     #run from the outline inventory year (2003 for HEF) to 2017 (end of CRU data in hydro year convention)
#     tasks.run_from_climate_data(gdir, ys=1985, ye=2020)
        
#     ds_diag = xr.open_dataset(gdir.get_filepath('model_diagnostics'))
#     smb = (ds_diag.volume_m3.values[1:]-ds_diag.volume_m3.values[:-1])/ds_diag.area_m2.values[1:]
#     smb = smb*cfg.PARAMS['ice_density']
#     plt.plot(ds_diag.time[:-1], smb, color=colors[i], label=rgi_id)
#     #smb from the WGMS and fixed geometry we already did before
#     # ref_df = gdir.get_ref_mb_data()
#     # plt.plot(ref_df.loc[2004:].index, ref_df.loc[2004:].OGGM, label='OGGM (fixed geom)')
#     # plt.plot(ref_df.loc[2004:].index, ref_df.loc[2004:].ANNUAL_BALANCE, label='WGMS')
#     plt.xlabel('Year') 
#     plt.ylabel('OGGM dynamic Mass balance [mm w.e.]')
#     plt.legend(loc='lower center', ncols=3, bbox_to_anchor=(0.5,-0.35))
#     plt.title('Mass balance over time')

""" Retrospective mass-balance form output - updating geometry whilst recomputing MB"""

# for (i,rgi_id) in enumerate(rgi_ids):  
#     plt.figure()

#     gdir = gdirs[i]
#     print(i)    
#     dyn_mod = flowline.FileModel(gdir.get_filepath('model_geometry'))
#     mbmod = massbalance.MultipleFlowlineMassBalance(gdir)
    
   
#     # we can loop through years and read the flowlines for each year
#     for year in range(2004,2017):
#         dyn_mod.run_until(year)
#         h=[]
#         smb=[]
#         for fl_id, fl in enumerate(dyn_mod.fls):
#             h = np.append(h,fl.surface_h)
#             mb = mbmod.get_annual_mb(fl.surface_h, year=year, fl_id=fl_id)
#             smb = np.append(smb, mb*cfg.SEC_IN_YEAR * cfg.PARAMS['ice_density'])
#         plt.plot(smb,h,'.', label=year)
#     plt.legend(title='year', loc='upper left')
#     plt.ylabel('Elevation (m a.s.l.)'); plt.xlabel('MB (mm w.e. yr$^{-1}$')
#     plt.title(rgi_id)

    
""" Calculating the ELA with oggm"""
# fig,ax=plt.subplots(1,1)
# global_tasks.compile_ela(gdirs, ys=1985, ye=2014)
# ela_df = pd.read_hdf(os.path.join(cfg.PATHS['working_dir'], 'ELA.hdf'))
# ela_df.plot(ax=ax, color=colors)
# # plt.xlabel('year'); plt.ylabel('ELA [m]')

# areas=[gd.rgi_area_km2 for gd in gdirs]
# avg = ela_df.mean(axis=1).to_frame(name='mean')
# # avg['weighed average'] = np.average(ela_df, axis=1, weights=areas)
# # avg['median'] = np.median(ela_df, axis=1)

# avg.plot(ax=ax, ls='dashed', color='red')
# plt.xlabel('year', plt.ylabel('ELA [m]'))

# for (i,rgi_id) in enumerate(rgi_ids):  
#     plt.figure()

#     gdir = gdirs[i]

""" What would the ELA look like in a 1degree warmer climate """

# global_tasks.compile_ela(gdirs, ys=1985, ye=2014)
# ela_df = pd.read_hdf(os.path.join(cfg.PATHS['working_dir'], 'ELA.hdf'))
# for (i,rgi_id) in enumerate(rgi_ids):  
#     plt.figure()

#     gdir = gdirs[i]
#     mbmod = massbalance.MonthlyTIModel(gdir, filename='climate_historical')
#     # print(mbmod)
#     corr_temp_bias = gdir.read_json('mb_calib')['temp_bias']
#     corr_prcp_fac = gdir.read_json('mb_calib')['prcp_fac']    
#     print(corr_prcp_fac)
     
#     """ you can change the precipitation factor in order to change the precipitation"""

#     # print(corr_temp_bias)
    
#     global_tasks.compile_ela(gdir, ys=1985, ye=2014, temperature_bias=corr_temp_bias+1, filesuffix='_t1')
#     global_tasks.compile_ela(gdir, ys=1985, ye=2014, precipitation_factor=corr_prcp_fac+10, filesuffix='_p10')
#     ela_df_t1=pd.read_hdf(os.path.join(cfg.PATHS['working_dir'], 'ELA_t1.hdf'))
#     ela_df_p10=pd.read_hdf(os.path.join(cfg.PATHS['working_dir'], 'ELA_p10.hdf'))

#     plt.plot(ela_df[[ela_df.columns[i]]].mean(axis=1), label='default climate', color=colors[i])
#     plt.plot(ela_df_t1.mean(axis=1), label=('default climate +1 $\degree$C'), ls='dashed', color=colors[i])
#     plt.plot(ela_df_p10.mean(axis=1), label=('default climate +10% precipitation'), ls='dotted', color=colors[i])

    
#     plt.xlabel('year CE'); plt.ylabel('ELA [m]'); plt.legend()
#     plt.title(rgi_id)
    

""" Lets look at longer timeseries"""

# for (i,rgi_id) in enumerate(rgi_ids):  
#     fig, ax = plt.subplots(1,1)
#     gdir = gdirs[i]
#     global_tasks.compile_ela(gdir,ys=1902,ye=2019, filesuffix='_1901_2019', csv=True)
#     ela_df_long=pd.read_csv(os.path.join(cfg.PATHS['working_dir'], 'ELA_1901_2019.csv'))
#     ela_df_long.plot(x=ela_df_long.columns[0], y=ela_df_long.columns[1], ax=ax, color=colors[i], ls='dashed')
#     ela_df_long.rolling(5).mean().plot(ela_df_long.rolling(5).mean().columns[0], ela_df_long.rolling(5).mean().columns[1],ax=ax, lw=2, color=colors[i], label='')
#     plt.xlabel('year CE'); plt.ylabel('ELA [m]'); ax.legend(["Annual", "5_yr average"])
#     plt.title(rgi_id)

""" Calculating ELA for specific years""" 
# yrs =np.arange(1985,2015,1)
# ela_yrs=massbalance.compute_ela(gdirs[0], years=yrs)
# print(yrs, ela_yrs)
    
"""Calculating accumation area ratio """ 
# rgi_id = 'RGI60-11.00897'

global_tasks.compile_ela(gdirs, ys=1985, ye=2014)
ela_df = pd.read_hdf(os.path.join(cfg.PATHS['working_dir'], 'ELA.hdf'))

for (i,rgi_id) in enumerate(rgi_ids):  
    fig, ax = plt.subplots(1,1)
    gdir = gdirs[i]
    tot_area = 0
    tot_area_above = 0
    fls = gdir.read_pickle('model_flowlines')
    for fl in fls:
        # The area is constant
        tot_area += fl.area_m2
        # The area above a certain limit is not - we make a (nx, nt) array
        # This is a little bit of dimensional juggling (broadcasting)
        is_above = (fl.surface_h[:, np.newaxis, ] > ela_df[rgi_id].values)
        # Here too - but the time dim is constant
        area = fl.bin_area_m2[:, np.newaxis] * np.ones(len(ela_df))
        tot_area_above += (area * is_above).sum(axis=0)
    
    # Write the output
    fixed_aar = pd.DataFrame(index=ela_df.index)
    fixed_aar['AAR'] = tot_area_above / tot_area
    
    # Plot ELA and AAR on the same plot
    fixed_aar['ELA'] = ela_df[rgi_id]
    fixed_aar.plot(ax=ax, secondary_y='ELA'); plt.tilte(rgi_id)
        