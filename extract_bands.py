
import numpy as np
import gdal
import argparse

import h5py
from osgeo import osr
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description='Convert select bands from h5 files to raster format')
    parser.add_argument('input_file')
    parser.add_argument('output_file')
    parser.add_argument('-hdf5_key', default='CRBU/Reflectance/Reflectance_Data')
    parser.add_argument('-b', nargs='*', default=[-1], type=int)
    parser.add_argument('-write_bil', default=False, type=bool)
    parser.add_argument('-of', default='GTiff')
    args = parser.parse_args()

    f = h5py.File(args.input_file, 'r')
    dat = f[args.hdf5_key]

    if (args.b[0] == -1):
        n_bands = dat.shape[2]
        args.b = np.arange(0, n_bands).astype(int).tolist()
    else:
        n_bands = len(args.b)

    driver = gdal.GetDriverByName(args.of)
    driver.Register()

    opt = []
    if (args.write_bil == True):
        opt = ['INTERLEAVE=BIL']
    outDataset = driver.Create(
        args.output_file, dat.shape[1], dat.shape[0], n_bands, gdal.GDT_Float32, options=opt)

    name_base = args.hdf5_key.split('/')[0] + '/' + args.hdf5_key.split('/')[1]
    epsg_number = int(f[name_base + '/Metadata/Coordinate_System/EPSG Code'].value)
    proj = osr.SpatialReference()
    proj.ImportFromEPSG(epsg_number)
    proj = str(proj)
    outDataset.SetProjection(proj)

    map_info = str(f[name_base + '/Metadata/Coordinate_System/Map_Info'].value)
    x_ul = float(map_info.split(',')[3])
    y_ul = float(map_info.split(',')[4])
    px_w = float(map_info.split(',')[5])
    px_h = float(map_info.split(',')[6])

    out_trans = [x_ul, px_w, 0, y_ul, 0, -px_h]
    outDataset.SetGeoTransform(out_trans)

    if (args.write_bil == False):
        for b in tqdm(range(0, n_bands), ncols=80):
            outDataset.GetRasterBand(b+1).WriteArray(np.squeeze(dat[:, :, int(args.b[b])]), 0, 0)
            outDataset.FlushCache()
        del outDataset
    else:
        del outDataset
        output_rad_file = open(args.output_file, 'wb', 0)
        for l in np.arange(0, dat.shape[0], 1).astype(int):
            ld = np.array(dat[l, :, :], dtype='float32')
            np.reshape(ld, (1, ld.shape[0], ld.shape[1])).tofile(output_rad_file)
            del ld
        output_rad_file.close()


if __name__ == "__main__":
    main()
