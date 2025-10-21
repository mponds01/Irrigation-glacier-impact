# -*- coding: utf-8 -*-

import os
# from oggm import utils, cfg
# from oggm import GlacierDirectory

import glob
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

folder_path = '/Users/magaliponds/Documents/00. Programming'
base_dir = Path(f'{folder_path}/04. Modelled perturbation-glacier interactions - R13-15 A+1km2/per_glacier/')
files = list(base_dir.rglob("*_noi_bias_hydro.nc"))

def delete_one(p: Path):
    try:
        # print(p)
        p.unlink()
        return f"Deleted: {p}"
    except FileNotFoundError:
        return f"Already gone: {p}"
    except PermissionError as e:
        return f"Permission error: {p} ({e})"
    except OSError as e:
        return f"OS error: {p} ({e})"

# Start conservatively; bump if needed.
max_workers = 8

with ThreadPoolExecutor(max_workers=max_workers) as ex:
    futures = [ex.submit(delete_one, p) for p in files]
    for fut in as_completed(futures):
        print(fut.result())
        




# keywords_to_keep = ['CNRM', 'W5E5', 'IPSL-CM6', 'CESM2']




# rgi_regions=['13','14','15']
# rgi_totals={
#     '13':['54'],
#     '14':['27'],
#      '15':['13']}

# Loop through glacier directories
# for region_id in os.listdir(base_dir):
# region_path = os.path.join(base_dir, "RGI60-15")
# if not os.path.isdir(region_path) or region_id.startswith('.'):
    # continue
# for subregion_id in os.listdir(region_path):
#     subregion_path = os.path.join(region_path, subregion_id)
#     if not os.path.isdir(subregion_path) or subregion_id.startswith('.'):
#         continue
#     for glacier_id in os.listdir(subregion_path):
#         gdir_path = os.path.join(subregion_path, glacier_id)
#         if not os.path.isdir(gdir_path) or glacier_id.startswith('.'):
#             continue
    
#         contains_keyword = False
#         if os.path.isdir(gdir_path):
#             for fname in os.listdir(gdir_path):
#                 if any(k in fname for k in keywords_to_keep):
#                     contains_keyword = True
#                     # print("keyword set to true", contains_keyword)
#                     break
    
#         if not contains_keyword:
#             try:
#             # Load glacier directory
#                 gdir = GlacierDirectory(glacier_id, base_dir=base_dir)
#                 print(gdir)
#                 # Archive and remove the gdir folder
#                 print(f"Archiving: {glacier_id}")
#                 utils.gdir_to_tar(gdir)  # creates tar and deletes folder
#             except Exception as e:
#                 print(f"Failed to process {glacier_id}: {e}")
#         else:
#             print(f"Keeping: {glacier_id} (contains climate data)")