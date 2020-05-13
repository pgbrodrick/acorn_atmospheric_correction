import gdal
import numpy as np
import subprocess
import pandas as pd
import multiprocessing
import argparse
import math
import time
import os
from osgeo import osr

"""
This code requires a substantial utilization of globals, defined dynamically throughout the script.  While not
perhaps the most elegant, it significantly decreases parameter passing.  A refactor could fix this if desired.
"""


GAIN_FILE = os.path.join(os.getcwd(), 'support/neon_426_gains.txt')
FWHM_FILE = os.path.join(os.getcwd(), 'support/neon_426_bands_wv_fwhm.txt')
OFFSET_FILE = os.path.join(os.getcwd(), 'support/neon_offset_426.txt')


def run_cmd(command):
    subprocess.call(command, shell=True)

############ IO #############################


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser(
    description='Run a parallel calculation of surface reflectance with tile-specific parameters generated from the obs and igm files')
parser.add_argument('radiance_file')
parser.add_argument('igm_file')
parser.add_argument('obs_file')
parser.add_argument('template_iacorn_file')
parser.add_argument('final_refl_file')
parser.add_argument('epsg', type=int)
parser.add_argument('-shade_mask_file')
parser.add_argument('-cloud_mask_file')
parser.add_argument('-final_wtrl_file')
parser.add_argument('-final_wtrv_file')
parser.add_argument('-final_vis_file')
parser.add_argument('-final_log_file')
parser.add_argument('-scratch_folder')
parser.add_argument('-copy_to_dest', type=boolean_string, default=False)
parser.add_argument('-n_proc', type=int, default=23)
parser.add_argument('-year_month_day')
parser.add_argument('-visibility', type=int, default=50)  # IACORN - 100
parser.add_argument('-subline_obs', type=bool, default=False)
parser.add_argument('-nodata_value', type=int, default=0)
parser.add_argument('-max_acorn_runs', type=int, default=96)
parser.add_argument('-output_bands', type=int, default=426)
parser.add_argument('-_420_band', type=int, default=9)
args = parser.parse_args()

flight_line = os.path.basename(args.radiance_file)

if args.final_wtrl_file is None:
    args.final_wtrl_file = os.path.join(os.path.dirname(
        args.final_refl_file), os.path.basename(args.final_refl_file) + '_wtrl')
if args.final_wtrv_file is None:
    args.final_wtrv_file = os.path.join(os.path.dirname(
        args.final_refl_file), os.path.basename(args.final_refl_file) + '_wtrv')
if args.final_vis_file is None:
    args.final_vis_file = os.path.join(os.path.dirname(
        args.final_refl_file), os.path.basename(args.final_refl_file) + '_vis')
if args.final_log_file is None:
    args.final_log_file = os.path.join(os.path.dirname(
        args.final_refl_file), os.path.basename(args.final_refl_file) + '_logs.txt')
if args.scratch_folder is None:
    args.scratch_folder = '/tmp/' + flight_line
if args.year_month_day is None:
    args.year_month_day = flight_line[6:14]  # TODO - don't hardcode this....

run_cmd('mkdir ' + args.scratch_folder)

if (args.copy_to_dest):
    active_radiance_file = os.path.join(args.scratch_folder, os.path.basename(args.radiance_file))
    run_cmd('cp ' + args.radiance_file + ' ' + active_radiance_file)
    run_cmd('cp ' + args.radiance_file + '.hdr ' + active_radiance_file + '.hdr')
else:
    active_radiance_file = args.radiance_file


def append_to_output(st):
    run_cmd('echo \"' + st + '\" >> ' + args.final_log_file)


run_cmd('echo \" \" > ' + args.final_log_file)

append_to_output('Radiance File: ' + args.radiance_file)
append_to_output('Igm File: ' + args.igm_file)
append_to_output('Obs File: ' + args.obs_file)
append_to_output('Settings Template File: ' + args.template_iacorn_file)
append_to_output('Output Reflectance File: ' + args.final_refl_file)
if (args.shade_mask_file is None):
    append_to_output('Shade Mask File: None')
else:
    append_to_output('Shade Mask File: ' + args.shade_mask_file)
if (args.cloud_mask_file is None):
    append_to_output('Cloud Mask File: None')
else:
    append_to_output('Cloud Mask File: ' + args.cloud_mask_file)
append_to_output('Output WTRL File: ' + args.final_wtrl_file)
append_to_output('Output WTRV File: ' + args.final_wtrv_file)
append_to_output('Output Visibility File: ' + args.final_vis_file)
append_to_output('Output Log File: ' + args.final_log_file)
append_to_output('Scratch Folder: ' + args.scratch_folder)
append_to_output('year_month_day: ' + args.year_month_day)
append_to_output('Specified Visibility: ' + str(args.visibility))
append_to_output('Specified EPSG: ' + str(args.epsg))

if (not args.shade_mask_file is None):
    shade_set = gdal.Open(args.shade_mask_file, gdal.GA_ReadOnly)
if (not args.cloud_mask_file is None):
    cloud_set = gdal.Open(args.cloud_mask_file, gdal.GA_ReadOnly)

# prepare the settings file with universal parameters
settings_base = pd.read_csv(args.template_iacorn_file, header=None, sep=',')
header = settings_base.iloc[4, :].tolist()
settings_base.iloc[5, header.index('Date (day, month, year)')] = args.year_month_day[6:]
settings_base.iloc[5, header.index('Date (day, month, year)')+1] = args.year_month_day[5:6]
settings_base.iloc[5, header.index('Date (day, month, year)')+2] = args.year_month_day[:4]

rad_set = gdal.Open(args.radiance_file, gdal.GA_ReadOnly)
n_bands = rad_set.RasterCount
if (n_bands == 214):
    append_to_output('using default band characteristics')
elif (n_bands == 426):
    append_to_output('n bands == 426')
    settings_base.iloc[5, header.index('Gain File')] = GAIN_FILE
    settings_base.iloc[5, header.index('FWHM or SRF File')] = FWHM_FILE
    settings_base.iloc[5, header.index('Offset File')] = OFFSET_FILE
    settings_base.iloc[5, header.index('Bands')] = n_bands
else:
    append_to_output('different n bands ' + str(n_bands))
    Exception('Number of bands: ' + str(n_bands) + '.  Only 214 and 426 supported')

dataset = gdal.Open(active_radiance_file, gdal.GA_ReadOnly)
x_max = dataset.RasterXSize
y_max = dataset.RasterYSize
# y_max=400


# set up the coordinate system transformation (for writing lat / long in settings file)
igm_coordinate_system = args.epsg
coord_a = osr.SpatialReference()
coord_a.ImportFromEPSG(igm_coordinate_system)
coord_b = osr.SpatialReference()
coord_b.ImportFromEPSG(4326)
coordinate_transform = osr.CoordinateTransformation(coord_a, coord_b)

# batch information
refl_x_size = 200
refl_y_size = 200

# convert decimal degrees to hours, minutes, seconds


def dd_dms(dd):
    out = []
    out.append(math.floor(dd))
    out.append(math.floor((dd - math.floor(dd)) * 60.))
    out.append(((dd - math.floor(dd)) * 60. - math.floor((dd - math.floor(dd)) * 60.))*60)
    return out


# write the settings file in acorn style
def write_settings_file(df, xsize, ysize, inname, vis, igm, alt, utc_time):
    df.iloc[5, header.index('Samples')] = xsize
    df.iloc[5, header.index('Lines')] = ysize
    df.iloc[5, header.index('Output Reflectance Image Filename')] = inname + 'refl'
    df.iloc[5, header.index('Input Radiance Image')] = inname

    if igm is not None:
        df.iloc[5, header.index('Mean Elev (m)')] = igm[2]

        coord = coordinate_transform.TransformPoint(igm[0], igm[1])

        dms = dd_dms(coord[1])
        df.iloc[5, header.index('Latitude (deg, min, sec)  (+N,-S)')] = dms[0]
        df.iloc[5, header.index('Latitude (deg, min, sec)  (+N,-S)')+1] = dms[1]
        df.iloc[5, header.index('Latitude (deg, min, sec)  (+N,-S)')+2] = dms[2]

        dms = dd_dms(coord[0])
        df.iloc[5, header.index('Longitude (+E, -W) (deg, min, sec)')] = dms[0]
        df.iloc[5, header.index('Longitude (+E, -W) (deg, min, sec)')+1] = dms[1]
        df.iloc[5, header.index('Longitude (+E, -W) (deg, min, sec)')+2] = dms[2]

    if alt is not None:
        df.iloc[5, header.index('Mean flight alt (km)')] = alt

    if utc_time is not None:
        lt = dd_dms(utc_time)
        df.iloc[5, header.index('Average Time (UTC) (hr, min, sec)')] = lt[0]
        df.iloc[5, header.index('Average Time (UTC) (hr, min, sec)')+1] = lt[1]
        df.iloc[5, header.index('Average Time (UTC) (hr, min, sec)')+2] = lt[2]

    if (vis is not None):
        df.iloc[5, header.index('Visibility (5 to 250 km)')] = vis

    df.to_csv(inname + '.csv', sep=',', index=False, header=False)


# execute a single acorn iteration
def single_cell(df, rx, ry, rxsize, rysize, igm_file, obs_file, vis_file, refl_file, wtrl_file, wtrv_file, inname_append=''):

    inname = os.path.join(args.scratch_folder, str(rx) + '_' + str(ry) + '_' +
                          str(rxsize) + '_' + str(rysize) + '_' + inname_append)
    translate_cmd_str = 'gdal_translate -of ENVI --config GDAL_CACHEMAX 64 -co INTERLEAVE=BIL ' + \
        active_radiance_file + ' ' + inname + ' -srcwin ' + \
        str(rx) + ' ' + str(ry) + ' ' + str(rxsize) + ' ' + str(rysize)
    run_cmd(translate_cmd_str)
    run_cmd('rm ' + inname + '.hdr')

    igm_set = gdal.Open(igm_file, gdal.GA_ReadOnly)
    obs_set = gdal.Open(obs_file, gdal.GA_ReadOnly)
    vis_set = gdal.Open(vis_file, gdal.GA_ReadOnly)

    igm = igm_set.ReadAsArray(rx, ry, rxsize, rysize)
    valid = igm[0, ...] != -9999

    igm = [np.mean(igm[_i, ...][valid]) for _i in range(igm.shape[0])]
    utc_time = np.mean(obs_set.GetRasterBand(10).ReadAsArray(rx, ry, rxsize, rysize)[valid])
    alt = np.mean((np.squeeze(igm_set.GetRasterBand(3).ReadAsArray(rx, ry, rxsize, rysize)[
                  valid]) + np.squeeze(obs_set.GetRasterBand(1).ReadAsArray(rx, ry, rxsize, rysize)[valid]))/1000.)
    calc_vis = np.mean(vis_set.GetRasterBand(1).ReadAsArray(rx, ry, rxsize, rysize)[valid])

    del valid

    # write settings file
    write_settings_file(df, rxsize, rysize, inname, calc_vis, igm, alt, utc_time)

    # execute acorn
    acorn_cmd_str = '/absolute_path_to_acorn/acorn5lx/bin/acorn6lx -USERKEY' + inname + '.csv'
    run_cmd(acorn_cmd_str)

    try:
        loc_refl = gdal.Open(inname + 'refl', gdal.GA_ReadOnly).ReadAsArray().astype(np.float32)
        loc_wtrl = gdal.Open(inname + 'refl_wtrl',
                             gdal.GA_ReadOnly).ReadAsArray().astype(np.float32)
        loc_wtrv = gdal.Open(inname + 'refl_wtrv',
                             gdal.GA_ReadOnly).ReadAsArray().astype(np.float32)
    except:
        # try rerun
        run_cmd('rm ' + inname + '*')
        run_cmd(translate_cmd_str)
        run_cmd('rm ' + inname + '.hdr')

        if (args.visibility == -1 or args.visibility == -2):
            calc_vis = np.mean(vis_set.ReadAsArray(rx, ry, rxsize, rysize))
        else:
            calc_vis = args.visibility

        # write settings file
        write_settings_file(df, rxsize, rysize, inname, calc_vis, igm, alt, utc_time)

        # execute acorn
        run_cmd(acorn_cmd_str)

        try:
            loc_refl = gdal.Open(inname + 'refl', gdal.GA_ReadOnly).ReadAsArray().astype(np.float32)
            loc_wtrl = gdal.Open(inname + 'refl_wtrl',
                                 gdal.GA_ReadOnly).ReadAsArray().astype(np.float32)
            loc_wtrv = gdal.Open(inname + 'refl_wtrv',
                                 gdal.GA_ReadOnly).ReadAsArray().astype(np.float32)
        except:
            loc_refl = np.zeros((args.output_bands, rysize, rxsize))-1
            loc_wtrl = np.zeros((rysize, rxsize))-1
            loc_wtrv = np.zeros((rysize, rxsize))-1

    run_cmd('rm ' + inname + '*')

    write_lock.acquire()
    output_refl_binary_file = np.memmap(
        args.final_refl_file, mode='r+', shape=(y_max, args.output_bands, x_max), dtype=np.float32)
    output_refl_binary_file[ry:ry+rysize, :, rx:rx+rxsize] = np.swapaxes(loc_refl, 0, 1)
    del loc_refl

    output_wtrl_binary_file = np.memmap(
        args.final_wtrl_file, mode='r+', shape=(y_max, x_max), dtype=np.float32)
    output_wtrl_binary_file[ry:ry+rysize, rx:rx+rxsize] = loc_wtrl
    del loc_wtrl

    output_wtrv_binary_file = np.memmap(
        args.final_wtrv_file, mode='r+', shape=(y_max, x_max), dtype=np.float32)
    output_wtrv_binary_file[ry:ry+rysize, rx:rx+rxsize] = loc_wtrv
    del loc_wtrv
    write_lock.release()

    return rx, ry, rxsize, rysize


######################## Reflectance line processing #############################
print('creating datasets')
driver = gdal.GetDriverByName('ENVI')
driver.Register()
# create the final output images (for active writing throughout processing)
reflDataset = driver.Create(args.final_refl_file, x_max, y_max,
                            args.output_bands, gdal.GDT_Float32, options=['INTERLEAVE=BIL'])
reflDataset.SetProjection(dataset.GetProjection())
reflDataset.SetGeoTransform(dataset.GetGeoTransform())
del reflDataset

wtrlDataset = driver.Create(args.final_wtrl_file, x_max, y_max, 1, gdal.GDT_Float32)
wtrlDataset.SetProjection(dataset.GetProjection())
wtrlDataset.SetGeoTransform(dataset.GetGeoTransform())
del wtrlDataset

wtrvDataset = driver.Create(args.final_wtrv_file, x_max, y_max, 1, gdal.GDT_Float32)
wtrvDataset.SetProjection(dataset.GetProjection())
wtrvDataset.SetGeoTransform(dataset.GetGeoTransform())
del wtrvDataset

write_lock = multiprocessing.Lock()
pool = multiprocessing.Pool(processes=args.n_proc)

st = time.time()
results = []
last_ry = 0
for _ry in range(0, y_max, refl_y_size):
    rys = min(refl_y_size, y_max-_ry)
    for _rx in range(0, x_max, refl_x_size):
        rxs = min(refl_x_size, x_max-_rx)

        results.append(pool.apply_async(single_cell, args=(settings_base, _rx, _ry, rxs, rys, args.igm_file,
                                                           args.obs_file, args.final_vis_file, args.final_refl_file, args.final_wtrl_file, args.final_wtrv_file,)))


results = [p.get() for p in results]
pool.close()
pool.join()

append_to_output('total_seconds: {}'.format(time.time()-st))
append_to_output('lines / second: {}'.format(y_max / float(time.time()-st)))
