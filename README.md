[![DOI](https://zenodo.org/badge/512200960.svg)](https://zenodo.org/badge/latestdoi/512200960)
# irminger-proj

## Licence
This code is published under a BSD-3-C licence. Although you're very free to do what you like with the code, please acknowledge the code appropriately if you use it in your own work; and an email saying that you've used it would be appreciated. Citations for works which use this code can be found at the end of this readme.

The MITgcm setups may be useful; however, the post processing code is quite messy. It's included for completeness and if performing similar studies you may be best of looking at it to see how certain analyses are implemented rather than trying to use it directly.

## Directory structure
The `src` directory contains code for:
- A 2D and 3D model MITgcm model
- Generating input files for the model
- Analysing the model

### 2d-mitgcm-models \& 3d-mitgcm-moodels
Both these folders have a standard MITgcm type model directory. You'll need to obtain the MITgcm code (I use checkpoint68i) and build the model using the mods in the directories starting with `code` and corresponding to the model you want (2D 25m, 2D 200m or 3D 200m). You can then use the namelists in the input folders to run the model with your executable. You will need to generate initial conditions files first, however.

### initial_condition_generation
This folder has a notebook for generating ICs for the 2D and 3D models. They're fairly self explanatory.

### post_processing
This folder has scripts for post processing the data. It contains the following python scripts and notebooks:
- `combine_ensemble.py`: combines the raw data from the ensemble runs into a single zarr dataset
- `compress-3d.ipynb`: compresses the raw data from the 3D run into a zarr dataset
- `egu-figures.ipynb`: a notebook where I make some figures I used in my 2023 EGU presentation
- `ensemble_plotting.py`: a script which creates the ensemble plots that appear in my thesis
- `ERA5.ipynb`: a notebook where we estimate the transformation induced by down-front wind events over the course of a season
- `figure-6-7.ipynb`: a notebook where we create figures 6 and 7 as used in the paper
- `mld_calculations.py`: a script to calculate the change in mixed layer depth for the ensemble from the zarr
- `standard_model_plotting.py`: where we did the standard plots used in my thesis
- `tau_int-sketch.ipynb`: notebook for schematic of integrated wind stress in short vs long wind events.
- `transformation-fits.ipynb`: calcuate the MLD power law and WMT scalings
- `wmt_calculations.py`: where we do the WMT calcs
- `wmt_exploration.ipynb`: where we plot some WMT calcs and do some eploring
- `wmt-3d.ipynb`: Actually maybe this is where we do the WMT calcs? It's where we do a lot of the plotting for the paper.


## Related publications
Goldsworth, Fraser William. 2022. ‘Symmetric Instability in the Atlantic Meridional Overturning Circulation’. Oxford: University of Oxford. https://doi.org/10.5287/ora-xogpmrvzd.

Goldsworth, F.W., Johnson, H.L., Marshall, D.P., Le Bras, I.A. 2023. ‘Saturation of destratifying and restratifying instabilities during down front wind events: a case study in the Irminger Sea’. Submitted: JGR Oceans.

Goldsworth, Fraser. 2023. ‘Fraserwg/Irminger-Proj: V1.1’. Zenodo. https://doi.org/10.5281/zenodo.8233578.

Goldsworth, F.W., I.A. Le Bras, H.L. Johnson, and D.P. Marshall. 2023. ‘Data for “Saturation of Destratifying and Restratifying Instabilities during down Front Wind Events: A Case Study in the Irminger Sea”’. Zenodo. https://doi.org/10.5281/zenodo.8232682.

