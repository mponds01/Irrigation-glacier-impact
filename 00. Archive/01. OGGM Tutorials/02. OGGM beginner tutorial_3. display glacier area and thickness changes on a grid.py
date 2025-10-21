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
import shutil

import pandas as pd
import numpy as np
from matplotlib import animation
from IPython.display import HTML, display

cmap=cm.get_cmap('bone')
colors=[cmap(i / 5) for i in range(6)]


folder_path='/Users/magaliponds/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/1. VUB/02. Coding/01. IRRMIP/02. Data/06. OGGM Working Directories'
wd_path=f'{folder_path}/OGGM-GettingStarted-Beginner-3. Store directories for later use'

# cfg.PATHS['working_dir']  = utils.mkdir(wd_path, reset=True)
rgi_ids=['RGI60-13.37574', 'RGI60-13.37682', 'RGI60-13.37753',  'RGI60-13.53223', 'RGI60-13.53720']
rgi_ids = utils.get_rgi_glacier_entities(rgi_ids)


""" The structure of a working directory """ 
#Open a new worksflow for two glaciers

# cfg.initialize(logging_level='WARNING')
# cfg.PATHS['working_dir']  = utils.mkdir(wd_path, reset=True)


# #get preprocessed glacier directories
# base_url = ('https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/2023.3/elev_bands/W5E5/')

# gdirs = workflow.init_glacier_directories(rgi_ids, prepro_base_url=base_url, from_prepro_level=3, prepro_border=80)

def file_tree_print(prepro_dir=False):
    #just a utility function to show the dir structure and selected files
    print("cfg.PATHS['working_dir']/")
    tab = '   '
    for dirname, dirnames, filenames in os.walk(cfg.PATHS['working_dir']):
        for subdirname in dirnames:
            print(tab+subdirname + '/')
        for filename in filenames:
            if '.tar' in filename and 'RGI' in filename:
                print(tab + filename)
            tab +='  '

# #folders are organized in regional here
# #adding steps to the workflow, liek a spinup run

# # Our files are located in the final folders of this tree (not shown in the tree).
# #For example:
    
# # gdirs[0].get_filepath('dem').replace(wd_path, 'wd_path')

# #set the working dir correctly
# cfg.PATHS['working_dir'] = utils.get.mkdir(wd_path)
# workflow.execute_entity_task(tasks.run_from_climate_data, gdirs, 
#                                 output_filesuffix='_spinup',
#                                 );

"""Stop here and start from the same spot """ 
#set the working directory correctly
cfg.PATHS['working_dir'] = utils.gettempdir(wd_path)
#Go re-open the pre-processed glacier directories from what's there
# gdirs = workflow.init_glacier_directories()

# The step above can be quite slow (because OGGM has to parse quite some info from the directories). 
# Better is to start from the list of glaciers you want to work with:
gdirs = workflow.init_glacier_directories(rgi_ids) 

"""!!!CAREFUL!!! do not start from a preprocessed level (or from a tar file), 
or your local directories (which may contain new data) will be overwritten, 
i.e. workflow.init_glacier_directories(rgi_ids, from_prepro_level=3, prepro_base_url=base_url)
 will always start from the pre-processed, fresh state."""
 
"""Store the singel glacier directories into tar files""" 
#Writes the content of a glacier directory to a tar file.

# The tar file is located at the same location of the original directory.
# The glacier directory objects are useless if deleted!


# workflow.execute_entity_task(utils.gdir_to_tar,gdirs,delete=False);
# file_tree_print()
    
#or reconstruct gdirs from tar files
gdirs = workflow.init_glacier_directories(rgi_ids, from_tar=True, delete_tar=True)
file_tree_print()
    
"""Bundle of directories """ 
#easir way of bundling directories
workflow.execute_entity_task(utils.gdir_to_tar,gdirs,delete=True)
utils.base_dir_to_tar(wd_path, delete=True)
file_tree_print()

#while bunding it is still possible to select individual glaciers
gdirs = workflow.init_glacier_directories(rgi_ids, from_tar=True)
file_tree_print()
