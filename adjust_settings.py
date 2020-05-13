
import gdal
import numpy as np
import argparse
import pandas as pd
import subprocess
import os
import sys
from osgeo import osr
import math


parser = argparse.ArgumentParser(description='Build an ACORN style settings file')
parser.add_argument('radiance_file')
parser.add_argument('obs_file')
parser.add_argument('igm_file')
parser.add_argument('reflectance_file')
parser.add_argument('output_settings_file')
parser.add_argument('-template_iacorn_file', default='support/template_iacorn_settings.csv')
parser.add_argument('-visibility', type=int, default=100)
parser.add_argument('-epsg', type=int, default=32613)
parser.add_argument('-year_month_day')
args = parser.parse_args()


# convert decimal degrees to hours, minutes, seconds
def dd_dms(dd):
    out = []
    out.append(math.floor(dd))
    out.append(math.floor((dd - math.floor(dd)) * 60.))
    out.append(((dd - math.floor(dd)) * 60. - math.floor((dd - math.floor(dd)) * 60.))*60)
    return out


igm_set = gdal.Open(args.igm_file, gdal.GA_ReadOnly)
ground_elev = igm_set.GetRasterBand(3).ReadAsArray()

obs_set = gdal.Open(args.obs_file, gdal.GA_ReadOnly)
alt = (ground_elev + obs_set.GetRasterBand(1).ReadAsArray())/1000.
alt[ground_elev == -9999] = -9999

igm_coordinate_system = args.epsg
coord_a = osr.SpatialReference()
coord_a.ImportFromEPSG(igm_coordinate_system)
coord_b = osr.SpatialReference()
coord_b.ImportFromEPSG(4326)
coordinate_transform = osr.CoordinateTransformation(coord_a, coord_b)


flight_line = os.path.basename(args.radiance_file)
if args.year_month_day is None:
    args.year_month_day = flight_line[18:26]  # TODO - don't hardcode this....
settings_base = pd.read_csv(args.template_iacorn_file, header=None, sep=',')
header = settings_base.iloc[4, :].tolist()
settings_base.iloc[5, header.index('Date (day, month, year)')] = args.year_month_day[6:]
settings_base.iloc[5, header.index('Date (day, month, year)')+1] = args.year_month_day[5:6]
settings_base.iloc[5, header.index('Date (day, month, year)')+2] = args.year_month_day[:4]

settings_base.iloc[5, header.index('Samples')] = igm_set.RasterXSize
settings_base.iloc[5, header.index('Lines')] = igm_set.RasterYSize
settings_base.iloc[5, header.index('Mean Elev (m)')] = np.median(ground_elev[ground_elev != -9999])
settings_base.iloc[5, header.index('Mean flight alt (km)')] = np.median(alt[alt != -9999])

transform = obs_set.GetGeoTransform()
coord = coordinate_transform.TransformPoint(transform[0], transform[3])

dms = dd_dms(coord[1])
settings_base.iloc[5, header.index('Latitude (deg, min, sec)  (+N,-S)')] = dms[0]
settings_base.iloc[5, header.index('Latitude (deg, min, sec)  (+N,-S)')+1] = dms[1]
settings_base.iloc[5, header.index('Latitude (deg, min, sec)  (+N,-S)')+2] = dms[2]

dms = dd_dms(coord[0])
settings_base.iloc[5, header.index('Longitude (+E, -W) (deg, min, sec)')] = dms[0]
settings_base.iloc[5, header.index('Longitude (+E, -W) (deg, min, sec)')+1] = dms[1]
settings_base.iloc[5, header.index('Longitude (+E, -W) (deg, min, sec)')+2] = dms[2]

utc_time = obs_set.GetRasterBand(9).ReadAsArray()
lt = dd_dms(np.median(utc_time[utc_time != -9999]))
settings_base.iloc[5, header.index('Average Time (UTC) (hr, min, sec)')] = lt[0]
settings_base.iloc[5, header.index('Average Time (UTC) (hr, min, sec)')+1] = lt[1]
settings_base.iloc[5, header.index('Average Time (UTC) (hr, min, sec)')+2] = lt[2]

settings_base.iloc[5, header.index('Visibility (5 to 250 km)')] = args.visibility


settings_base.iloc[5, header.index('Output Reflectance Image Filename')] = args.reflectance_file
settings_base.iloc[5, header.index('Input Radiance Image')] = args.radiance_file


rad_set = gdal.Open(args.radiance_file, gdal.GA_ReadOnly)
n_bands = rad_set.RasterCount
if (n_bands == 214):
    print('using default band characteristics')
elif (n_bands == 426):
    print('n bands == 426')
    settings_base.iloc[5, header.index(
        'Gain File')] = '/lustre/scratch/pbrodrick/colorado/raw_lines/support/neon_426_gains.txt'
    settings_base.iloc[5, header.index(
        'FWHM or SRF File')] = '/lustre/scratch/pbrodrick/colorado/raw_lines/support/neon_426_bands_wv_fwhm.txt'
    settings_base.iloc[5, header.index(
        'Offset File')] = '/lustre/scratch/pbrodrick/colorado/raw_lines/support/neon_offset_426.txt'
    settings_base.iloc[5, header.index('Bands')] = n_bands
else:
    print('different n bands ' + str(n_bands))
    Exception('Number of bands: ' + str(n_bands) + '.  Only 214 and 426 supported')


settings_base.to_csv(args.output_settings_file, sep=',', index=False, header=False)
