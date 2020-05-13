
import numpy as np
import pandas as pd
import argparse
import gdal
from tqdm import tqdm
import numpy.matlib


def main():

    parser = argparse.ArgumentParser(description='Apply gains with BIL IO')
    parser.add_argument('input_file')
    parser.add_argument('gain_file')
    parser.add_argument('output_file')
    parser.add_argument('-chunk_size', default=1, type=int)
    args = parser.parse_args()

    gains = np.squeeze(np.array(pd.read_csv(args.gain_file, header=None)))

    dataset = gdal.Open(args.input_file, gdal.GA_ReadOnly)
    data_trans = dataset.GetGeoTransform()
    max_y = dataset.RasterYSize
    max_x = dataset.RasterXSize
    n_bands = dataset.RasterCount

    driver = gdal.GetDriverByName('ENVI')
    driver.Register()
    outDataset = driver.Create(args.output_file, max_x, max_y, n_bands,
                               gdal.GDT_Int16, options=['INTERLEAVE=BIL'])
    outDataset.SetGeoTransform(data_trans)
    outDataset.SetProjection(dataset.GetProjection())
    del outDataset

    gain_mat = np.zeros((n_bands, args.chunk_size, max_x))
    for n in range(0, args.chunk_size):
        gain_mat[:, n, :] = np.matlib.repmat(gains, gain_mat.shape[2], 1).T

    scale_factor = 10
    # Writing format = y,b,x
    # Input format = b,y,x
    for l in tqdm(np.arange(0, max_y, args.chunk_size).astype(int), ncols=80):
        dat = dataset.ReadAsArray(0, int(l), int(max_x), int(min(max_y-l, args.chunk_size)))
        bad_mask = np.all(dat == -9999, axis=0)
        dat = dat / gain_mat[:, :int(min(max_y-l, args.chunk_size)), :] * scale_factor
        dat[:, bad_mask] = -9999
        dat = np.round(np.swapaxes(dat, 0, 1)).astype(np.int16)
        wf = 'a'
        if (l == 0):
            wf = 'w'
        with open(args.output_file, wf) as out_file:
            dat.tofile(out_file)
        del out_file


if __name__ == "__main__":
    main()
