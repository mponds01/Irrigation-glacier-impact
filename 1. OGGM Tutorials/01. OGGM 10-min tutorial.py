# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#General start to OGGM script
from oggm import cfg, utils, workflow, tasks, DEFAULT_BASE_URL, graphics
import xarray as xr
import matplotlib.pyplot as plt
import os
from oggm.shop import gcm_climate

""" TUTORIAL 1: A PRE-PROCESSED DIRECOTRY"""

cfg.initialize(logging_level='INFO')

#cfg initialize will read the default parameter files, 
#and makes them available to all other OGGM tools via the cfg.PARAMS dictionary
# print(cfg.PARAMS['melt_f'], cfg.PARAMS['ice_density'], cfg.PARAMS['continue_on_error'])

# You can try with or without multiprocessing: with two glaciers, OGGM could run on two processors
cfg.PARAMS['use_multiprocessing'] = False

"""Working directories"""
#working directory needs to be specified for each run, can be a temporary folder, like below
folder_path='/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/06. OGGM Working Directories'
wd_path=f'{folder_path}/OGGM-GettingStarted-10m'
# cfg.PATHS['working_dir'] = utils.gettempdir(dirname=wd_path, reset=True)
# print(cfg.PATHS['working_dir'])

# a persistent working directory is made with the following commands
cfg.PATHS['working_dir']  = utils.mkdir(wd_path, reset=True)
# print(path)

"""Define the glaciers for the run"""
rgi_ids=['RGI60-13.53720', 'RGI60-13.37753', 'RGI60-13.37682', 'RGI60-13.53223', 'RGI60-13.37574']

"""Glacier directories"""
#print default base url, where glacier directory is located and where preloaded data can be found
DEFAULT_BASE_URL

# #lets use oggm to download glacier directories for the selected glaciers

# gdirs = workflow.init_glacier_directories(
#     rgi_ids,
#     prepro_base_url=DEFAULT_BASE_URL,
#     from_prepro_level=4,
#     prepro_border=80
#     )

# #read output and attributes
# #type(gdirs), type(gdirs[0])


# gdir = gdirs[2] #take the first glacier = Zhongfeng Glacier
# # print('Path to the DEM:', gdir.get_filepath('dem'))
# # print(gdir.rgi_date)
# # graphics.plot_domain(gdirs,figsize=(6,5))

# """ Accessing data in the preprocessed directories"""
# gdir.get_filepath('climate_historical')

# with xr.open_dataset(gdir.get_filepath('climate_historical')) as ds:
#     ds=ds.load()
# #now plot the data
# # ds.temp.resample(time='AS').mean().plot(label=f'Annual temperature at {int(ds.ref_hgt)}m a.s.l.');
# # ds.temp.resample(time='AS').mean().rolling(time=31, center=True, min_periods=15).mean().plot(label='30yr average');
# # plt.legend()

# #list files
# # print(os.listdir(gdir.dir))

# """ perform tasks with OGGM"""
# # run the compute_inversion_velocities task on all gdirs
# workflow.execute_entity_task(tasks.compute_inversion_velocities, gdirs)

# inversion_output = gdir.read_pickle('inversion_output')  # The task above wrote the data to a pickle file - but we write to plenty of other files!

# # Take the first flowline

# fl = inversion_output[0]
# # print(len(fl))
# # the last grid points often have irrealistic velocities
# # because of the equilibrium assumption
# vel = fl['u_surface'][:-1]  
# plt.plot(vel, label='Velocity along flowline (m yr-1)'); plt.legend();



""" TUTORIAL 2: A GLACIER CHANGE PROJECTION WITH GCM DATA"""

# wd_path=f'{folder_path}/OGGM-GettingStarted-10m-2. GCM Data'
# cfg.initialize(logging_level='WARNING')
# cfg.PATHS['working_dir']  = utils.mkdir(wd_path, reset=True)

# #rgi_ids remain similar for this run
# gdirs = workflow.init_glacier_directories(
#     rgi_ids,
#     prepro_base_url=DEFAULT_BASE_URL,
#     from_prepro_level=5
#     )

# #prepro level 5 comes witha pre-computed model frun from the RGI outline date to the last possible data given by the historical climate data. 
# #In the case of the new default climate dataset W5E5, this is until the end of 2019, so the volume is computed until January first 2020
# #These files are stored in a "_spinup_historical" suffic

# # plt.figure()
# # ds = utils.compile_run_output(gdirs, input_filesuffix='_spinup_historical')
# # (ds.volume /ds.volume.sel(time=2000)*100).plot(hue='rgi_id');
# # plt.ylabel('Volume (%, reference 2000)')

# # # this is the inventory data, The glacier volume and area changes before that date are highly uncertain and serve the purpose of spinup only!
# # gdirs[0].rgi_date, gdirs[1].rgi_date, gdirs[2].rgi_date, gdirs[3].rgi_date, gdirs[4].rgi_date

# # #use ISIMIP GCM data, 5 different GCMs:
# # # 'gfdl-esm4_r1i1p1f1', 'mpi-esm1-2-hr_r1i1p1f1', 'mri-esm2-0_r1i1p1f1' ("low sensitivity" models, within typical ranges from AR6)
# # # 'ipsl-cm6a-lr_r1i1p1f1', 'ukesm1-0-ll_r1i1p1f2' ("hotter" models, especially ukesm1-0-ll)

# #set member and ssp scenario
# member = 'mri-esm2-0_r1i1p1f1' 
# for ssp in ['ssp126', 'ssp370','ssp585']:
#     # bias correct them on monthly isimip data - isimip is already corrected by the ISIMIP consortium, so no need to further bias correct
#     workflow.execute_entity_task(gcm_climate.process_monthly_isimip_data, gdirs, 
#                                  ssp = ssp,
#                                  # gcm member -> you can choose another one
#                                  member=member,
#                                  # recognize the climate file for later
#                                  output_filesuffix=f'_ISIMIP3b_{member}_{ssp}'
#                                  );
# #get info on what historical data you are using  
# gdirs[0].get_climate_info()

# #run entity task to insert climate model data on the glaciers and see volume evolution over time
# #climate filename& suffix together construct the file which is used as input to force the file
# #it starts from glaciers (input model filesuffix = spinup historical)
# for ssp in ['ssp126', 'ssp370', 'ssp585']:
#     rid = f'_ISIMIP3b_{member}_{ssp}'
#     workflow.execute_entity_task(tasks.run_from_climate_data, gdirs,
#                                  climate_filename='gcm_data',  # use gcm_data, not climate_historical
#                                  climate_input_filesuffix=rid,  # use the chosen scenario
#                                  init_model_filesuffix='_spinup_historical',  # this is important! Start from 2020 glacier
#                                  output_filesuffix=rid,  # recognize the run for later
#                                 );
    
# #Plotting model output
# f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5, figsize=(14,4))
# color_dict={'ssp126':'blue', 'ssp370':'orange', 'ssp585':'red'} #set colors
# for ssp in ['ssp126', 'ssp370', 'ssp585']:
#     rid =f'_ISIMIP3b_{member}_{ssp}'
#     #compile output in 1 file
#     ds=utils.compile_run_output(gdirs, input_filesuffix=rid)
#     #create plots
#     ds.isel(rgi_id=0).volume.plot(ax=ax1, label=ssp, c=color_dict[ssp])
#     ds.isel(rgi_id=1).volume.plot(ax=ax2, label=ssp, c=color_dict[ssp])
#     ds.isel(rgi_id=2).volume.plot(ax=ax3, label=ssp, c=color_dict[ssp])
#     ds.isel(rgi_id=3).volume.plot(ax=ax4, label=ssp, c=color_dict[ssp])
#     ds.isel(rgi_id=4).volume.plot(ax=ax5, label=ssp, c=color_dict[ssp])
# plt.legend();

""" TUTORIAL 3: UNDERSTAND THE NEW DYNAMICAL SPINUP IN OGGM V1.6"""

wd_path=f'{folder_path}/OGGM-GettingStarted-10m-3. New dynamical spinup'
cfg.initialize(logging_level='WARNING')
cfg.PATHS['working_dir']  = utils.mkdir(wd_path, reset=True)
rgi_ids=rgi_ids[0]

#fetch preprocessed directories
gdirs = workflow.init_glacier_directories(rgi_ids, from_prepro_level=5, prepro_base_url=DEFAULT_BASE_URL)

#lets compare to the old run without dynamical spinup, influencing calibration parameters (especially melt factor)

# open the new historical run including a dynamic spinup
ds_spinup = utils.compile_run_output(gdirs, input_filesuffix='_spinup_historical')

# open the old historical run without a spinup
ds_historical = utils.compile_run_output(gdirs, input_filesuffix='_historical')

# compare area and volume evolution
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

# Area
# ds_spinup.area.plot(ax=ax1, label='dynamic spinup')
# ds_historical.area.plot(ax=ax1, label='no spinup')
# ax1.set_ylim(2.25e8,2.5e8)
# ax1.set_title('Area [m2]')

# # Volume
# ds_spinup.volume.plot(ax=ax2, label='dynamic spinup')
# ds_historical.volume.plot(ax=ax2, label='no spinup')
# ax2.set_ylim(3.5e10,3.9e10)
# ax2.set_title('Volume [m3]')

# plt.legend();

#--> Dynamical spinup tis much longer (40 instead of 10 years) 
# area and volume are both larger
# melt factor is dynamically recalibrated

# Old spinup had some troubles re assuming a steady state at start: 
    #This could lead to artifacts (mainly in the glacier length and area, as well as velocities) during the first few years of the simulation.

gdir = gdirs[0]

# period of geodetic mass balance
ref_period = cfg.PARAMS['geodetic_mb_period']

# open the observation with uncertainty
df_ref_dmdtda = utils.get_geodetic_mb_dataframe().loc[gdir.rgi_id]  # get the data from Hugonnet et al., 2021
df_ref_dmdtda = df_ref_dmdtda.loc[df_ref_dmdtda['period'] == ref_period]  # only select the desired period
dmdtda_reference = df_ref_dmdtda['dmdtda'].values[0] * 1000  # get the reference dmdtda and convert into kg m-2 yr-1
dmdtda_reference_error = df_ref_dmdtda['err_dmdtda'].values[0] * 1000  # corresponding uncertainty

# calculate dynamic geodetic mass balance
def get_dmdtda(ds):
    yr0_ref_mb, yr1_ref_mb = ref_period.split('_')
    yr0_ref_mb = int(yr0_ref_mb.split('-')[0])
    yr1_ref_mb = int(yr1_ref_mb.split('-')[0])

    return ((ds.volume.loc[yr1_ref_mb].values[0] -
             ds.volume.loc[yr0_ref_mb].values[0]) /
            gdir.rgi_area_m2 /
            (yr1_ref_mb - yr0_ref_mb) *
            cfg.PARAMS['ice_density'])

print(f'Reference dmdtda 2000 to 2020 (Hugonnet 2021): {dmdtda_reference:.2f} +/- {dmdtda_reference_error:6.2f} kg m-2 yr-1')
print(f'Dynamic spinup dmdtda 2000 to 2020:            {float(get_dmdtda(ds_spinup)):.2f}            kg m-2 yr-1')
print(f"Dynamically calibrated melt_f:                 {gdir.read_json('mb_calib')['melt_f']:.1f}                 kg m-2 day-1 °C-1")

#OGGM tries to match the observations within 20% error

# to check the importance of the spinup it helps to take a look at the glacier velocities
f = gdir.get_filepath('fl_diagnostics', filesuffix='_historical')
with xr.open_dataset(f, group=f'fl_0') as dg:
    dgno = dg.load()
f = gdir.get_filepath('fl_diagnostics', filesuffix='_spinup_historical')
with xr.open_dataset(f, group=f'fl_0') as dg:
    dgspin = dg.load()


year = 2012
dgno.ice_velocity_myr.sel(time=year).plot(label='No spinup');
dgspin.ice_velocity_myr.sel(time=year).plot(label='With spinup');
plt.title(f'Velocity along the flowline at year {year}'); plt.legend();

#Ice velocities in the spinup case are considerably lower 
#because they take into account the current retreat and past history of the glacier, 
#while the blue line is the velocity of a glacier just getting out of steady state.



