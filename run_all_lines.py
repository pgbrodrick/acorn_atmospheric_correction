

import pandas as pd
import numpy as np
import subprocess
import os


OBS_DIR = '/lustre/scratch/pbrodrick/colorado/20190607/all_obs/'
GLT_DIR = '/lustre/scratch/pbrodrick/colorado/20190607/all_glts/'
IGM_DIR = '/lustre/scratch/pbrodrick/colorado/20190607/all_igm/'

OUTPUT_DIR = 'acorn_mid_variable_vis'
if (os.path.isdir(OUTPUT_DIR) == False):
    os.mkdir(OUTPUT_DIR)

rad_files = np.squeeze(np.array(pd.read_csv('all_radiance_files.txt', header=None))).tolist()
obs_files = [os.path.join(OBS_DIR, os.path.basename(x).replace('rdn', 'rdn_obs'))
             for x in rad_files]
igm_files = [os.path.join(IGM_DIR, os.path.basename(
    x).replace('rdn', 'rdn_ort_igm')) for x in rad_files]
refl_files = [os.path.join(OUTPUT_DIR, os.path.basename(
    x).replace('rdn', 'acorn_autovis_refl')) for x in rad_files]
set_files = [os.path.join(OUTPUT_DIR, os.path.basename(x).replace(
    'rdn', 'acorn_autovis_settings.csv')) for x in rad_files]


for _i in range(0, len(rad_files)):
    cmd_str = 'sbatch single_line_acorn.csh ' + \
              rad_files[_i] + ' ' + \
              rad_files[_i] + ' ' + \
              obs_files[_i] + ' ' + \
              igm_files[_i] + ' ' + \
              refl_files[_i] + ' ' + \
              set_files[_i]
    if (os.path.isfile(refl_files[_i] + '_ciacorn_rgb.tif') == False):
        print(cmd_str)
        subprocess.call(cmd_str, shell=True)
