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
wd_path=f'{folder_path}/OGGM-GettingStarted-Beginner-4. Full Preprocessed Directory'
print(wd_path)
#OGGM option
cfg.PATHS['working_dir'] = utils.gettempdir(wd_path, reset=False)

#define the glaciers we will play with
rgi_ids=['RGI60-13.37574', 'RGI60-13.37682', 'RGI60-13.37753',  'RGI60-13.53223', 'RGI60-13.53720']

#preparing the glacier data

base_url = 'https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/2023.3/elev_bands/W5E5'

cfg.initialize(logging_level='WARNING')
cfg.PARAMS['melt_f'], cfg.PARAMS['ice_density'], cfg.PARAMS['continue_on_error']

gdirs = workflow.init_glacier_directories(rgi_ids, from_prepro_level=3, prepro_base_url=base_url, prepro_border=80)
