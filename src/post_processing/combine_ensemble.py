"""combine_ensemble.py

This script is for combining the data from the individual runs in the ensemble
in to a single zarr file, with a "run dimension" 
"""

import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO
                   )

logging.info('Importing standard python libraries')
from pathlib import Path
from itertools import product

logging.info('Importing third party python libraries')
import numpy as np
import dask
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
import xarray as xr
import zarr
import xmitgcm
import f90nml

logging.info('Setting paths')
base_path = Path('/work/n01/n01/fwg/irminger-proj')

log_path = base_path / 'src/post_processing/.tmp/slurm-out'
dask_worker_path = base_path / 'src/post_processing/.tmp/dask-worker-space'

env_path = base_path / 'irminger-proj/bin/activate'
ensemble_path = base_path / 'data/raw/2d-models'
processed_path = base_path / 'data/processed'
interim_path = base_path / 'data/interim'

# Check paths exist etc.
if not log_path.exists(): log_path.mkdir(parents=True)
if not processed_path.exists(): processed_path.mkdir()
assert ensemble_path.is_dir()
assert env_path.exists()

#logging.info('Initialising the dask cluster')
# Set up the dask cluster
scluster = SLURMCluster(queue='standard',
                        project="n01-siAMOC",
                        job_cpu=256,
                        log_directory=log_path,
                        local_directory=dask_worker_path,
                        cores=128,
                        processes=32,  # Can change this
                        memory="256 GiB",
                        header_skip= ['#SBATCH --mem='],  
                        walltime="00:20:00",
                        death_timeout=60,
                        interface='hsn0',
                        job_extra=['--qos=short', '--reservation=shortqos'],
                        env_extra=['module load cray-python',
                                   'source {}'.format(str(env_path.absolute()))]
                       )

client = Client(scluster)
scluster.scale(jobs=4)

logging.info(client)
logging.info(scluster)

run_labels = list(np.arange(0, 50))
duplicate_runs = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45]
run_labels = [label for label in run_labels if label not in duplicate_runs]

# Wind stress parameters
logging.info('Check the wind stress parameters are correct.')
days = 24 * 60 * 60
tau_max_arr = np.linspace(0, -0.75, 10)
sigmat_arr = np.linspace(1e-9, 5 * days, 5)

wind_params = np.array(list(product(tau_max_arr, sigmat_arr)))
wind_stress = wind_params[run_labels][:, 0]
duration = wind_params[run_labels][:, 1]

def create_encoding_for_ds(ds, clevel):
    compressor = zarr.Blosc(cname="zstd", clevel=clevel, shuffle=2)
    enc = {x: {"compressor": compressor} for x in ds}
    return enc

def open_dataset(run_label, suffix):
    run_path = ensemble_path / f'run{run_label}_{suffix}'
    logging.info(f'Opening {run_path}')
    
    logging.info('Reading in model parameters from the namelist')
    data_nml = f90nml.read(ensemble_path / f'input_data_files_{suffix}/data')

    delta_t = data_nml['parm03']['deltat']
    f0 = data_nml['parm01']['f0']
    beta = data_nml['parm01']['beta']
    no_slip_bottom = data_nml['parm01']['no_slip_bottom']
    no_slip_sides = data_nml['parm01']['no_slip_sides']


    logging.info('Reading in the model dataset')
    ds = xmitgcm.open_mdsdataset(run_path,
                                prefix=['ZLevelVars', 'IntLevelVars'],
                                delta_t=delta_t,
                                geometry='cartesian',
                                #chunks=300
                                )
    
    ds.attrs['f0'] = f0
    ds.attrs['beta'] = beta
    ds.attrs['no_slip_bottom'] = no_slip_bottom
    ds.attrs['no_slip_sides'] = no_slip_sides
    return ds

@dask.delayed
def combine_along_time(run_label):
    logging.info(f"Combining run {run_label}")
    run_suffs = ['a', 'b', 'c', 'd']
    ds_list = [open_dataset(run_label, suff) for suff in run_suffs]
    ds_combined = xr.concat(ds_list, 'time')
    return ds_combined


def combine_along_ensemble(run_label_list):
    logging.info("Combining ensemble")
    ds_list = [combine_along_time(run_label) for run_label in run_label_list]
    ds_list = dask.compute(*ds_list)
    ds_combined = xr.concat(ds_list, 'run')
    ds_combined['run'] = run_label_list
    return ds_combined

if __name__ == "__main__":
    ds_interim = combine_along_ensemble(run_labels)
    ds_interim['wind_stress'] = ('run', wind_stress, {'units': 'N m-2',
                                          'long_name': 'Minimum wind stress'})
    ds_interim['wind_duration'] = ('run', duration, {'units': 's',
                                         'long_name': 'Standard deviation of wind forcing'})
    
    enc = create_encoding_for_ds(ds_interim, 9)

    out_path = interim_path / "ensemble.zarr"
    logging.info(f"Saving to {out_path}")
    ds_interim.to_zarr(out_path, encoding=enc)
    
    logging.info("Combining complete")