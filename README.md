# Wrapper to run tiled ACORN atmospheric correction with local aerosol estimates

This repository runs the ACORN atmospheric correction algorithm in a tiled fashion. Initial iterative runs are used to 
estimate aerosols (dynamically throughout a flightline) by standardizing low-wavelength reflectance values from vegetation.
These aerosol estimates are then used in a final set of runs that span the full flightline, with each run executing on a 
local set of reflectance data in order to capture locally changing environmental conditions (path length, elevation, aerosols,
view angle geometry, etc.).
This code was created as part of an effort to generate foliar trait maps throughout the Department of Energy (DOE) Watershed Function Scientific Focus Area (WF-SFA) site in Crested Butte, CO in association with NEON's Assignable Asset program. <br>


Main files:

run_all_lines.py - perform atmospheric correction on all lines, batching slurm-style submission<br>
single_line_acorn.csh - run atmospheric correction on a single binary cube.  Works as bash or slurm script <br>
run_all_manifest.py - prepare outputs for upload to Google Earth Engine<br>


A full description of the effort can be found at:

> K. Dana Chadwick, Philip Brodrick, Kathleen Grant, Tristan Goulden, Amanda Henderson, Nicola Falco, Haruko Wainwright, Kenneth H. Williams, Markus Bill, Ian Breckheimer, Eoin L. Brodie, Heidi Steltzer, C. F. Rick Williams, Benjamin Blonder, Jiancong Chen, Baptiste Dafflon, Joan Damerow, Matt Hancher, Aizah Khurram, Jack Lamb, Corey Lawrence, Maeve McCormick. John Musinsky, Samuel Pierce, Alexander Polussa, Maceo Hastings Porro, Andea Scott, Hans Wu Singh, Patrick O. Sorensen, Charuleka Varadharajan, Bizuayehu Whitney, Katharine Maher. Integrating airborne remote sensing and field campaigns for ecology and Earth system science. <i>In Review</i>, 2020.

and use of this code should cite that manuscript.

### Visualization code in GEE for all products in this project can be found here: 
https://code.earthengine.google.com/5c96bbc96ffd50e3c8b1433b34a0bb86
<br>

### Assets generated from this process are available on GEE: 
Custom Reflectance: https://code.earthengine.google.com/?asset=users/pgbrodrick/SFA/collections/ciacorn_priority <br> 
OBS data: https://code.earthengine.google.com/?asset=users/pgbrodrick/SFA/collections/obs_priority <br>
Water vapor estimates: https://code.earthengine.google.com/?asset=users/pgbrodrick/SFA/collections/wtrv_priority <br>
Canopy water content: https://code.earthengine.google.com/?asset=users/pgbrodrick/SFA/collections/wtrl_priority <br>
<br>
and are included as part of the data package: 
> Brodrick P ; Goulden T ; Chadwick K D (2020): Custom NEON AOP reflectance mosaics and maps of shade masks, canopy water content. Watershed Function SFA. DOI: 10.15485/1618131
<br>
<br>


## All repositories relevant to this project:

### Atmospheric correction wrapper: 
https://github.com/pgbrodrick/acorn_atmospheric_correction

### Shade ray tracing: 
https://github.com/pgbrodrick/shade-ray-trace

### Conifer Modeling:
https://github.com/pgbrodrick/conifer_modeling

### Trait Model Generation:
https://github.com/kdchadwick/east_river_trait_modeling

### PLSR Ensembling:
https://github.com/pgbrodrick/ensemblePLSR
