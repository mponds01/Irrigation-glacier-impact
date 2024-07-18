# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-import oggm

""" 
This script cruns the OGGM step by step guide to a preprocessed directory
"""


#%% Cell 1: Load the required directories

from oggm import cfg, utils, workflow, tasks, DEFAULT_BASE_URL

import geopandas as gpd
import numpy as np
import os
from oggm.shop import rgitopo


#%% Cell 2: Initialize and define the working directory

# we always need to initialzie and define a working directory
cfg.initialize(logging_level='WARNING')
cfg.PATHS['working_dir'] = utils.gettempdir(dirname='OGGM-full_prepro_elevation_bands', reset=True)

# Our example glacier
rgi_ids = ["RGI60-13.40102", # Area >10km2
  "RGI60-13.39195", # Area >10km2
  "RGI60-13.36881", # Area >10km2
  "RGI60-13.38969", # Area >10km2
  "RGI60-13.37184", # Area >10km2
  "RGI60-13.00967", # Area <10km2
  "RGI60-13.40982", # Area <10km2
  "RGI60-13.41891", # WGMS observation
  "RGI60-13.23659", # WGMS observation
] 

rgi_region = '13'  # this must fit to example glacier(s), if starting from level 0

#%% Cell 3: Define if to load from pre_pro level

# This section is only for future developments of the tutorial (e.g. updateing for new OGGM releases)
# Test if prepro_base_url valid for both flowline_type_to_use, see level 2.
# In total four complete executions of the notebook:
# (load_from_prepro_base_url=False/True and flowline_type_to_use = 'elevation_band'/'centerline')
load_from_prepro_base_url = False

#%% Cell 4: LEVEL 0

""" Tasks:

Define the rgi_id for your glacier directory gdir.
Define the map projection of the glacier directory
Add an outline of the glacier.
Optionally add intersects to other outlines. 

"""


# load all RGI outlines for our region and extract the example glaciers
rgidf = gpd.read_file(utils.get_rgi_region_file(rgi_region, version='62'))
rgidf = rgidf[np.isin(rgidf.RGIId, rgi_ids)]

# set the used projection used for gdir, options 'tmerc' or 'utm'
cfg.PARAMS['map_proj'] = cfg.PARAMS['map_proj']  # default is 'tmerc'

gdirs = workflow.init_glacier_directories(rgidf, reset=True, force=True)

#%% CELL 4b: Instruction for beginning with existing OGGM preprocessed directories

# Instruction for beginning with existing OGGM's preprocessed directories
if load_from_prepro_base_url:
    # to start from level 0 you can do
    prepro_base_url_L0 = 'https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L1-L2_files/elev_bands/'
    gdirs = workflow.init_glacier_directories(rgi_ids,
                                              from_prepro_level=0,
                                              prepro_base_url=prepro_base_url_L0,
                                              prepro_border=80,  # could be 10, 80, 160 or 240
                                              reset=True,
                                              force=True,
                                             )
    
#%% Cell 5: LEVEL 1

""" Tasks:

Define the border around the outline.
Define the local grid resolution, which will also set the resolution for the flowlines.
Add the digital elevation model DEM.
Set up a local grid for each gdir."""


# define the border, we keep the default here
cfg.PARAMS['border'] = cfg.PARAMS['border']

# set the method for determining the local grid resolution
cfg.PARAMS['grid_dx_method'] = cfg.PARAMS['grid_dx_method']  # The default method is 'square', which determines the grid spacing (dx) based on the glacier's outline area.
cfg.PARAMS['fixed_dx'] = cfg.PARAMS['fixed_dx']  # This allows setting a specific resolution in meters. It's applicable only when grid_dx_method is set to 'fixed'.

# set the DEM source to use
source = None  # we stick with the OGGM default

# this task adds the DEM and defines the local grid
workflow.execute_entity_task(tasks.define_glacier_region, gdirs,
                             source=source);

#%% Cell 5b: Load prepro base URL

    # Instruction for beginning with existing OGGM's preprocessed directories
if load_from_prepro_base_url:
    # to start from level 1 you can do
    prepro_base_url_L1 = 'https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L1-L2_files/elev_bands/'
    gdirs = workflow.init_glacier_directories(rgi_ids,
                                              from_prepro_level=1,
                                              prepro_base_url=prepro_base_url_L1,
                                              prepro_border=80,  # could be 10, 80, 160 or 240
                                              reset=True,
                                              force=True,
                                             )
#%% Cell 6: LEVEL 2

""" Tasks:

Choose the type of flowline to use.
Create the flowlines surface structure, including surface height and width.
Create the downstream flowline, which starts from the glacier’s terminus and extends downstream.
Optionally you can bring in extra data from the OGGM-shop and bin it to the elevation band flowline. """

flowline_type_to_use = 'elevation_band'  # you can also select 'centerline' here

if flowline_type_to_use == 'elevation_band': #type of flowline to use
    elevation_band_task_list = [
        tasks.simple_glacier_masks, #mask glacier area
        tasks.elevation_band_flowline, #creating flowline surface structure 
        tasks.fixed_dx_elevation_band_flowline,
        tasks.compute_downstream_line, #creating downstream flowline
        tasks.compute_downstream_bedshape,
    ]

    for task in elevation_band_task_list:
        workflow.execute_entity_task(task, gdirs);

elif flowline_type_to_use == 'centerline':
    # for centerline we can use parabola downstream line
    cfg.PARAMS['downstream_line_shape'] = 'parabola'

    centerline_task_list = [
        tasks.glacier_masks,
        tasks.compute_centerlines,
        tasks.initialize_flowlines,
        tasks.catchment_area,
        tasks.catchment_intersections,
        tasks.catchment_width_geom,
        tasks.catchment_width_correction,
        tasks.compute_downstream_line,
        tasks.compute_downstream_bedshape,
    ]

    for task in centerline_task_list:
        workflow.execute_entity_task(task, gdirs);
    
else:
    raise ValueError(f"Unknown flowline type '{flowline_type_to_use}'! Select 'elevation_band' or 'centerline'!")

#%% Cell 6b: LEVEL 2 - pre-processed
# Instruction for beginning with existing OGGM's preprocessed directories
if load_from_prepro_base_url:
    # to start from level 2 we need to distinguish between the flowline types
    if flowline_type_to_use == 'elevation_band':
        prepro_base_url_L2 = 'https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L1-L2_files/2023.2/elev_bands_w_data/'
    elif flowline_type_to_use == 'centerline':
        prepro_base_url_L2 = 'https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L1-L2_files/centerlines/'
    else:
        raise ValueError(f"Unknown flowline type '{flowline_type_to_use}'! Select 'elevation_band' or 'centerline'!")

    gdirs = workflow.init_glacier_directories(rgi_ids,
                                              from_prepro_level=2,
                                              prepro_base_url=prepro_base_url_L2,
                                              prepro_border=80,  # could be 10, 80, 160 or 240
                                              reset=True,
                                              force=True,
                                             )
    
#%% Cell 7: LEVEL 3

"""Tasks:

Add baseline climate data to gdir.
Calibrate the mass balance model statically (without considering glacier dynamics) using geodetic observations. This involves the calibration of melt_f, prcp_fac and temp_bias.
Conduct an inversion for the glacier’s bed topography. Including the calibration of glen_a and fs by matching to the total volume estimate.
Create the dynamic flowline for dynamic simulation runs."""

# define the climate data to use, we keep the default
cfg.PARAMS['baseline_climate'] = cfg.PARAMS['baseline_climate'] #default baseline climate is GSWP3_W5E5

# add climate data to gdir
workflow.execute_entity_task(tasks.process_climate_data, gdirs);

# the default mb calibration
workflow.execute_entity_task(tasks.mb_calibration_from_geodetic_mb, #geodetic mb calibration is to Hugonnet data
                             gdirs,
                             informed_threestep=True,  # only available for 'GSWP3_W5E5'
                            );

# glacier bed inversion
workflow.execute_entity_task(tasks.apparent_mb_from_any_mb, gdirs);
#for calibration on inversion different approach if individual glaciers
#from consensus is based on total region consensus estimate, not per glacier. 
#Volume estimates are model based and not directly observed and less reliable for individual calibration --> here we do however
#volume estimate based on farinotti
workflow.calibrate_inversion_from_consensus(  
    gdirs,
    apply_fs_on_mismatch=True,
    error_on_mismatch=True,  # if you running many glaciers some might not work
    filter_inversion_output=True,  # this partly filters the overdeepening due to
    # the equilibrium assumption for retreating glaciers (see. Figure 5 of Maussion et al. 2019)
    volume_m3_reference=None,  # here you could provide your own total volume estimate in m3
);

# finally create the dynamic flowlines
workflow.execute_entity_task(tasks.init_present_time_glacier, gdirs);

#%% Cell 7b: Level 3 -  using ERA 5 data - calibration included

#Currently, OGGM supports a variety of baseline climates, including ‘CRU’, ‘HISTALP’, ‘W5E5’, ‘GSWP3_W5E5’ (the default), ‘ERA5’, ‘ERA5L’, ‘CERA’, ‘ERA5dr’, and ‘ERA5L-HMA’. 
#Although switching between these datasets is straightforward, calibrating the mass balance model according to each dataset is more complex. 
#For instance, you’ll need to choose a default precipitation factor that suits both your selected climate dataset and your specific region.

# define the baseline climate and add it
cfg.PARAMS['baseline_climate'] = 'ERA5'
workflow.execute_entity_task(tasks.process_climate_data, gdirs);

# define the default precipitation factor
cfg.PARAMS['prcp_fac'] = 1.6  # Note: This is not a universial value!
cfg.PARAMS['use_winter_prcp_fac'] = False  # This option is only available for 'GSWP3_W5E5'
cfg.PARAMS['use_temp_bias_from_file'] = False  # This option is only available for 'GSWP3_W5E5'

# an example of static calibration for mass balance, more options are available in the tutorial
workflow.execute_entity_task(tasks.mb_calibration_from_geodetic_mb,
                             gdirs,
                             calibrate_param1='melt_f',
                             calibrate_param2='prcp_fac',
                             calibrate_param3='temp_bias')

#%% Cell 7c - Level 3 using own climate data

cfg.PARAMS['baseline_climate'] = 'CUSTOM'
cfg.PATHS['climate_file'] = path_to_the_climate_file

workflow.execute_entity_task(tasks.process_climate_data, gdirs);

# proceed with defining the default precipitation factor and mass balance calibration as shown above

#%% Cell 7d: Level 3 - with pre-processed directories

# Instruction for beginning with existing OGGM's preprocessed directories
if load_from_prepro_base_url:
    # to start from level 3 you can do
    if flowline_type_to_use == 'elevation_band':
        prepro_base_url_L3 = 'https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/2023.3/elev_bands/W5E5/'
    elif flowline_type_to_use == 'centerline':
        prepro_base_url_L3 = 'https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/2023.3/centerlines/W5E5/'
    else:
        raise ValueError(f"Unknown flowline type '{flowline_type_to_use}'! Select 'elevation_band' or 'centerline'!")

    gdirs = workflow.init_glacier_directories(rgi_ids,
                                              from_prepro_level=3,
                                              prepro_base_url=prepro_base_url_L3,
                                              prepro_border=80,  # could be 80 or 160
                                              reset=True,
                                              force=True,
                                             )
    
#%% Cell 8: Level 4

""" 
Initialize the current state of the glacier without a dynamic spinup. 
This method, default until version 1.6., is mainly for comparison purposes and can often be skipped.

Initialize the current glacier state with a dynamic spinup. 
This process includes a dynamic calibration of the mass balance. 
It’s important to note that this option isn’t available for centerlines in the current OGGM preprocessed directories, 
meaning it hasn’t been tested or analyzed."""

# set the ice dynamic solver depending on the flowline-type
if flowline_type_to_use == 'elevation_band':
    cfg.PARAMS['evolution_model'] = 'SemiImplicit'
elif flowline_type_to_use == 'centerline':
    cfg.PARAMS['evolution_model'] = 'FluxBased'
else:
    raise ValueError(f"Unknown flowline type '{flowline_type_to_use}'! Select 'elevation_band' or 'centerline'!")

# get the start and end year of the selected baseline
y0 = gdirs[0].get_climate_info()['baseline_yr_0']
ye = gdirs[0].get_climate_info()['baseline_yr_1'] + 1  # run really to the end until 1.1.

# 'static' initialisation
workflow.execute_entity_task(tasks.run_from_climate_data, gdirs,
                             min_ys=y0, ye=ye,
                             fixed_geometry_spinup_yr=None,  # here you could add a static spinup if you want
                             output_filesuffix='_historical')

# 'dynamic' initialisation, including dynamic mb calibration
dynamic_spinup_start_year = 1979
minimise_for = 'area'  # other option would be 'volume'
workflow.execute_entity_task(
    tasks.run_dynamic_melt_f_calibration, gdirs,
    err_dmdtda_scaling_factor=0.2,  # by default we reduce the mass balance error for accounting for
    # corrleated uncertainties on a regional scale
    ys=dynamic_spinup_start_year, ye=ye,
    kwargs_run_function={'minimise_for': minimise_for},
    ignore_errors=True,
    kwargs_fallback_function={'minimise_for': minimise_for},
    output_filesuffix='_spinup_historical',
);



