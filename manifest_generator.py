

import gdal
import numpy as np
import argparse
import os
import sys
from tqdm import tqdm
import json


def main():
    parser = argparse.ArgumentParser(description='Prepare files for GEE manifest upload')
    parser.add_argument('input_file')
    parser.add_argument('-output_base', default='manifests', type=str)
    parser.add_argument('-gs_bucket_base', default='gs://mosaics', type=str)
    parser.add_argument('-ee_line_name', default='ciacorn', type=str)
    parser.add_argument('-manifest_only', default=0, type=int)
    args = parser.parse_args()

    if args.manifest_only == 1:
        args.manifest_only = True
    else:
        args.manifest_only = False

    if (os.path.isdir(args.output_base) is False):
        os.mkdir(args.output_base)

    output_dir = os.path.join(args.output_base, os.path.splitext(
        os.path.basename(args.input_file))[0])
    if (os.path.isdir(output_dir) is False):
        os.mkdir(output_dir)

    dataset = gdal.Open(args.input_file, gdal.GA_ReadOnly)

    driver = gdal.GetDriverByName('ENVI')
    driver.Register()

    output_filenames = []
    for _b in range(dataset.RasterCount):
        fn = os.path.join(output_dir, 'band_' + str(_b))
        if args.manifest_only is False:
            od = driver.Create(fn, dataset.RasterXSize, dataset.RasterYSize, 1, gdal.GDT_Float32)
            od.SetGeoTransform(dataset.GetGeoTransform())
            od.SetProjection(dataset.GetProjection())
            del od
        output_filenames.append(fn)

    manifest = {}
    manifest['name'] = os.path.join('projects/earthengine-legacy/assets/users/pgbrodrick/SFA/lines',
                                    args.ee_line_name, os.path.splitext(os.path.basename(args.input_file))[0])
    manifest['tilesets'] = []

    uris = []
    ids = []
    for _b in range(dataset.RasterCount):
        uris.append(args.gs_bucket_base +
                    output_filenames[_b].replace(args.output_base, '') + '.tif')
        ids.append('tileset_for_band{}'.format(_b))

    for _b in range(dataset.RasterCount):
        tdict = {'id': ids[_b], 'sources': [{'uris': [uris[_b]]}]}
        manifest['tilesets'].append(tdict)

    wv = np.genfromtxt('support/neon_426_wavelengths.txt', delimiter='\n', dtype=str)
    wavelengths = [float(x[:-1]) for x in wv[2:427]]
    wavelengths.append(float(wv[427][:-2]))

    manifest['bands'] = []
    for _b in range(dataset.RasterCount):
        manifest['bands'].append({'id': 'B{}'.format(_b), 'tileset_id': ids[_b]})

    manifest['missing_data'] = {'values': [10]}

    with open(os.path.join(output_dir, 'manifest.json'), 'w') as outfile:
        json.dump(manifest, outfile, indent=4)

    if args.manifest_only:
        quit()

    y_size = dataset.RasterYSize
    x_size = dataset.RasterXSize

    for _line in tqdm(range(y_size), ncols=80):
        line = dataset.ReadAsArray(0, int(_line), dataset.RasterXSize, 1)
        if (len(line.shape) == 2):
            line = np.reshape(line, (1, line.shape[0], line.shape[1]))

        for _b in range(dataset.RasterCount):
            out_memmap = np.memmap(
                output_filenames[_b], mode='r+', shape=(y_size, x_size), dtype=np.float32)
            out_memmap[_line:_line+1, ...] = line[_b, :, :]
            del out_memmap

    for _b in tqdm(range(dataset.RasterCount), ncols=80):
        gdal.Translate(output_filenames[_b] + '.tif', gdal.Open(output_filenames[_b],
                                                                gdal.GA_ReadOnly), options=' -co COMPRESS=DEFLATE -co TILED=YES')
        os.remove(output_filenames[_b])
        os.remove(output_filenames[_b] + '.hdr')


if __name__ == "__main__":
    main()
