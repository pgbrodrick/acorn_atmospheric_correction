#!/bin/bash
#SBATCH --job-name=iacornp
#SBATCH -N 1
#SBATCH -c 24
#SBATCH --mem=128000
#SBATCH -p DGE
#SBATCH -o logs/op
#SBATCH -e logs/ep
## The above are SLURM parameters, safely ingnorable if running in bash on single machine

# Inputs
HDF5_FILE=${1}
RAD_FILE=${2}
OBS_FILE=${3}
IGM_FILE=${4}
REFL_FILE=${5}
SET_FILE=${6}

#### Load modules necessary for system - below are legacy from initial runs
#source deactivate gdal
#module purge
#module load python/3.6.0
#source activate gdal

# Set necessary evironment variabls
export GDAL_CACHEMAX=1
export ACORNDIR=/system_acorn_reference/acorn5lx/bin
gain_file=${PWD}/support/neon_426_gains.txt

rm ${RAD_FILE}
rm ${IGM_FILE}
rm ${OBS_FILE}

# Extract bands from NEON cube
python extract_bands.py ${HDF5_FILE} ${RAD_FILE}_bsq -b -1 -hdf5_key CRBU/Radiance/Radiance_Data -of ENVI

# Generate BIL formatted file
rm ${RAD_FILE}
gdal_translate ${RAD_FILE}_bsq ${RAD_FILE} -of ENVI -co INTERLEAVE=BIL
rm ${RAD_FILE}_bsq*
python extract_bands.py ${HDF5_FILE} ${RAD_FILE} -b -1 -hdf5_key CRBU/Radiance/Radiance_Data -write_bil True -of ENVI
python extract_bands.py ${HDF5_FILE} ${IGM_FILE} -b -1 -hdf5_key CRBU/Radiance/Metadata/Ancillary_Rasters/IGM_Data -write_bil True -of ENVI
python extract_bands.py ${HDF5_FILE} ${OBS_FILE} -b -1 -hdf5_key CRBU/Radiance/Metadata/Ancillary_Rasters/OBS_Data -write_bil True -of ENVI

python apply_gain.py ${RAD_FILE} ${gain_file} ${RAD_FILE}_int
cat support/neon_426_wavelengths.txt >> ${RAD_FILE}_int.hdr

rm /tmp/NIS* -r
python adjust_settings.py ${RAD_FILE}_int ${OBS_FILE} ${IGM_FILE} ${REFL_FILE} ${SET_FILE}

rm ${REFL_FILE}_ciacorn*
python neon_subline_visibility.py ${RAD_FILE}_int ${IGM_FILE} ${OBS_FILE} ${SET_FILE} ${REFL_FILE}_ciacorn 32613 -visibility -1 -max_acorn_runs 69 -subline_obs True
python neon_subline_iacorn_par.py ${RAD_FILE}_int ${IGM_FILE} ${OBS_FILE} ${SET_FILE} ${REFL_FILE}_ciacorn 32613 -visibility -2 -max_acorn_runs 69 -subline_obs True 
cat support/neon_426_wavelengths.txt >> ${REFL_FILE}_ciacorn.hdr

gdal_translate ${REFL_FILE}_ciacorn ${REFL_FILE}_ciacorn_rgb.tif -of ENVI -co COMPRESS=LZW -b 52 -b 34 -b 16
rm /tmp/NIS* -r


