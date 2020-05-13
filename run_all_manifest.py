
import pandas as pd
import numpy as np
import subprocess
import os


# Example for cloud masks
GLT_DIR = '/lustre/scratch/pbrodrick/colorado/20190607/all_glts/'
OUTPUT_DIR = '../shade-ray-trace/cloud_masks'
ORTHO_DIR = 'cloud_ortho_lines'
MANIFEST_DIR = 'manifests'

rad_files = np.squeeze(np.array(pd.read_csv('all_radiance_files.txt', header=None))).tolist()
rawspace_files = [os.path.join(OUTPUT_DIR, os.path.basename(
    x).replace('rdn', 'cloud_mask')) for x in rad_files]
glt_files = [os.path.join(GLT_DIR, os.path.basename(
    x).replace('rdn', 'rdn_ort_glt')) for x in rad_files]
orthod_files = [os.path.join(ORTHO_DIR, os.path.basename(
    x).replace('rdn', 'cloud_ort')) for x in rad_files]
manifest_files = [os.path.join(MANIFEST_DIR, os.path.basename(
    x).replace('rdn', 'cloud_ort'), 'band_0.tif') for x in rad_files]
line_name = 'cloud'


for _i in range(len(rad_files)):

    cmd_str = 'python apply_glt.py {} {} {} -f ENVI -co INTERLEAVE=BIL -c 100 -a_nodata 10; python manifest_generator.py {} -ee_line_name {}'.format(
        rawspace_files[_i], glt_files[_i], orthod_files[_i], orthod_files[_i], line_name)
    subprocess.call(
        'sbatch -o logs/o_{} -e logs/e_{} -n 1 --mem=30000 -p DGE --wrap="{}"'.format(_i, _i, cmd_str), shell=True)
