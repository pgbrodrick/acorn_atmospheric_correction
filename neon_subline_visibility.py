

import gdal
import numpy as np
import numpy.matlib
import subprocess
import os
import sys
import pandas as pd
import multiprocessing
import argparse
import math
from scipy.interpolate import griddata
import scipy.ndimage
import os
from osgeo import osr

GAIN_FILE = os.path.join(os.getcwd(), 'support/neon_426_gains.txt')
FWHM_FILE = os.path.join(os.getcwd(), 'support/neon_426_bands_wv_fwhm.txt')
OFFSET_FILE = os.path.join(os.getcwd(), 'support/neon_offset_426.txt')


def run_cmd(command):
    # print(command)
    subprocess.call(command, shell=True)

############ IO #############################


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser(description='Run a parallel calculation of surface reflectance with tile-specific '
                                             'parameters generated from the obs and igm files')
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
    args.year_month_day = flight_line[6:14]

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

igm = gdal.Open(args.igm_file, gdal.GA_ReadOnly).ReadAsArray()
obs = gdal.Open(args.obs_file, gdal.GA_ReadOnly).ReadAsArray()
utc_time = np.squeeze(obs[9, :, :])
alt = (np.squeeze(igm[2, :, :]) + np.squeeze(obs[0, :, :]))/1000.
del obs

if (not args.shade_mask_file is None):
    shade = np.squeeze(gdal.Open(args.shade_mask_file, gdal.GA_ReadOnly).ReadAsArray())
if (not args.cloud_mask_file is None):
    cloud = np.squeeze(gdal.Open(args.cloud_mask_file, gdal.GA_ReadOnly).ReadAsArray())

############ END IO #############################


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
def single_cell(df, rx, ry, rxsize, rysize, igm, alt, utc_time, calc_vis, inname_append=''):

    inname = os.path.join(args.scratch_folder, str(rx) + '_' + str(ry) + '_' +
                          str(rxsize) + '_' + str(rysize) + '_' + inname_append)
    translate_cmd_str = 'gdal_translate -of ENVI --config GDAL_CACHEMAX 64 -co INTERLEAVE=BIL ' + \
        active_radiance_file + ' ' + inname + ' -srcwin ' + \
        str(rx) + ' ' + str(ry) + ' ' + str(rxsize) + ' ' + str(rysize)
    run_cmd(translate_cmd_str)
    run_cmd('rm ' + inname + '.hdr')

    # write settings file
    write_settings_file(df, rxsize, rysize, inname, calc_vis, igm, alt, utc_time)

    # execute acorn
    acorn_cmd_str = '/acorn_basepath/bin/acorn6lx -USERKEY' + inname + '.csv'
    run_cmd(acorn_cmd_str)

    try:
        loc_refl = gdal.Open(inname + 'refl', gdal.GA_ReadOnly).ReadAsArray().astype(np.float32)
    except:
        # try rerun
        run_cmd('rm ' + inname + '*')
        run_cmd(translate_cmd_str)
        run_cmd('rm ' + inname + '.hdr')

        # write settings file
        write_settings_file(df, rxsize, rysize, inname, calc_vis, igm, alt, utc_time)

        # execute acorn
        run_cmd(acorn_cmd_str)

        if (os.path.isfile(inname + 'refl')):
            loc_refl = gdal.Open(inname + 'refl', gdal.GA_ReadOnly).ReadAsArray().astype(np.float32)
        # if still a failure, populate with null
        else:
            loc_refl = np.zeros((args.output_bands, rysize, rxsize))-1
    if (os.path.isfile(inname + 'refl_wtrl')):
        loc_wtrl = gdal.Open(inname + 'refl_wtrl',
                             gdal.GA_ReadOnly).ReadAsArray().astype(np.float32)
    else:
        loc_wtrl = np.zeros((rysize, rxsize))-1
    if (os.path.isfile(inname + 'refl_wtrv')):
        loc_wtrv = gdal.Open(inname + 'refl_wtrv',
                             gdal.GA_ReadOnly).ReadAsArray().astype(np.float32)
    else:
        loc_wtrv = np.zeros((rysize, rxsize))-1

    run_cmd('rm ' + inname + '*')

    return rx, ry, rxsize, rysize, loc_refl, loc_wtrl, loc_wtrv


def internal_iacorn(df, x, y, xsize, ysize, good_px, l_igm, l_alt, l_utc_time):
    vis_range = [100, 80, 60, 40, 35, 30, 25, 20]
    _420_out = []
    for _vis in range(0, len(vis_range)):
        aaa, bbb, ccc, ddd, loc_refl, loc_wtrl, loc_wtrv = single_cell(
            df, x, y, xsize, ysize, l_igm, l_alt, l_utc_time, vis_range[_vis], str(vis_range[_vis]))
        del aaa, bbb, ccc, ddd, loc_wtrl, loc_wtrv

        bmult = int(args.output_bands/213)
        ndvi = (loc_refl[42*bmult, :, :] - loc_refl[30*bmult, :, :]) / \
            (loc_refl[42*bmult, :, :] + loc_refl[30*bmult, :, :])
        _420 = np.squeeze(loc_refl[args._420_band, :, :])

        if (_vis == 0):
            good_px[ndvi < 0.8] = False
            good_px[np.isnan(ndvi)] = False
            good_px[np.isinf(ndvi)] = False
            good_px[np.all(loc_refl == 0, axis=0)] = False
            good_px[np.any(np.isnan(loc_refl), axis=0)] = False

            if (np.sum(good_px) < 2000):
                append_to_output('found only {} px'.format(np.sum(good_px)))
                #run_cmd('rm ' + inname + '*')
                return x, y, np.nan
        del loc_refl
        del ndvi

        _420_out.append(np.nanmean(_420[good_px]))
        if (_420_out[-1] < 100.):
            break
        else:
            append_to_output('latest vis: {}, {}'.format(vis_range[_vis], _420_out[-1]))

    append_to_output('########### iacorn printout ############')
    for i in range(0, len(_420_out)):
        append_to_output('Vis range, value: {}, {}'.format(vis_range[i], _420_out[i]))
    vis_range = np.array(vis_range)
    _420_out = np.array(_420_out)
    vis_range = vis_range[:len(_420_out)]
    vis_range = vis_range[np.isnan(_420_out) == False]
    _420_out = _420_out[np.isnan(_420_out) == False]

    order = np.argsort(_420_out)
    if (np.any(np.isnan(_420_out))):
        return x, y, np.nan
    else:
        selected_vis = np.interp(100., _420_out[order], vis_range[order])
        append_to_output('selected vis: {}'.format(selected_vis))
        return x, y, selected_vis


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

# load info from the radiance file
dataset = gdal.Open(active_radiance_file, gdal.GA_ReadOnly)
x_max = dataset.RasterXSize
y_max = dataset.RasterYSize

driver = gdal.GetDriverByName('ENVI')
driver.Register()
pool = multiprocessing.Pool(processes=args.n_proc)


vis_y_size = 200
vis_x_size = 200
iacorn_y_corners = list(range(0, y_max-vis_y_size, 400))
iacorn_y_corners.append(y_max-vis_y_size)
iacorn_y_corners = np.array(iacorn_y_corners).astype(int)
iacorn_x_corners = np.array([0, x_max-vis_x_size]).astype(int)

l_igm = None
l_alt = None
l_utc_time = None


####################### IACORN visibility calculation ############################
if (args.visibility == -1):
    visDataset = driver.Create(args.final_vis_file, x_max, y_max, 1, gdal.GDT_Float32)
    visDataset.SetProjection(dataset.GetProjection())
    visDataset.SetGeoTransform(dataset.GetGeoTransform())

    interp_x = []
    interp_y = []
    interp_vis = []

    results = []
    for _y in iacorn_y_corners:
        ys = int(min(vis_y_size, y_max-_y))
        for _x in iacorn_x_corners:
            xs = int(min(vis_x_size, x_max-_x))
            _x = int(_x)
            _y = int(_y)
            if (args.subline_obs):
                l_igm = [np.mean(igm[iii, _y:_y+ys, _x:_x+xs]) for iii in range(0, 3)]
                l_alt = np.mean(alt[_y:_y+ys, _x:_x+xs])
                l_utc_time = np.mean(utc_time[_y:_y+ys, _x:_x+xs])

            valid = igm[0, _y:_y+ys, _x:_x+xs] != -9999
            if (not args.shade_mask_file is None):
                valid[shade[_y:_y+ys, _x:_x+xs] != 255] = False
            if (not args.cloud_mask_file is None):
                valid[cloud[_y:_y+ys, _x:_x+xs] != 255] = False
            if (np.sum(valid) > 0):
                results.append(pool.apply_async(internal_iacorn, args=(
                    settings_base, _x, _y, vis_x_size, vis_y_size, valid, l_igm, l_alt, l_utc_time,)))

                if (len(results) > 24):
                    output = np.array([p.get() for p in results])
                    for o in output:
                        _y = o[1]
                        _x = o[0]
                        if (np.isnan(o[2]) == False):
                            interp_x.append(o[0])
                            interp_y.append(o[1])
                            interp_vis.append(o[2])
                            append_to_output(str(interp_x[-1]) + ' ' +
                                             str(interp_y[-1]) + ' ' + str(interp_vis[-1]))

                    append_to_output('starting interpolation')
                    results.clear()
                    results = []

    if (len(results) > 0):
        output = np.array([p.get() for p in results])
        for o in output:
            _y = o[1]
            _x = o[0]
            if (np.isnan(o[2]) == False):
                interp_x.append(o[0])
                interp_y.append(o[1])
                interp_vis.append(o[2])
                append_to_output(str(interp_x[-1]) + ' ' +
                                 str(interp_y[-1]) + ' ' + str(interp_vis[-1]))

        append_to_output('starting interpolation')
        results.clear()
        results = []

    if (len(interp_vis) > 1):
        x_arr = np.matlib.repmat(np.arange(0, x_max).reshape(
            1, x_max), y_max, 1).flatten().astype(float)
        y_arr = np.matlib.repmat(np.arange(0, y_max).reshape(
            y_max, 1), 1, x_max).flatten().astype(float)

        calc_vis = np.zeros(y_max*x_max)
        chunk_step = 200
        for _chunk in range(0, len(calc_vis), chunk_step):
            calc_vis[_chunk:_chunk+chunk_step] = griddata(np.transpose(np.vstack([np.array(interp_x), np.array(interp_y)])), np.array(
                interp_vis), (x_arr[_chunk:_chunk+chunk_step], y_arr[_chunk:_chunk+chunk_step]), method='nearest')

        calc_vis = np.reshape(calc_vis, (y_max, x_max))
        calc_vis = scipy.ndimage.filters.gaussian_filter(calc_vis, (vis_y_size, vis_x_size))
        del x_arr, y_arr
    else:
        calc_vis = np.ones((y_max, x_max)) * 100
    visDataset.GetRasterBand(1).WriteArray(calc_vis, 0, 0)
    visDataset.FlushCache()
    del visDataset
