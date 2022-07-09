import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO
                   )

logging.info('Importing standard python libraries')
from pathlib import Path

logging.info('Importing third party python libraries')
import numpy as np
import dask
from dask.distributed import Client, LocalCluster
from dask_jobqueue import SLURMCluster
import xarray as xr
import xmitgcm
from xgcm import Grid
import f90nml

logging.info('Importing custom python libraries')
import pvcalc


logging.info('Setting paths')
base_path = Path('/work/n01/n01/fwg/irminger-proj')

log_path = base_path / 'src/post_processing/.tmp/slurm-out'
dask_worker_path = base_path / 'src/post_processing/.tmp/dask-worker-space'

env_path = Path('/work/n01/n01/fwg/venvs/parallel-base/bin/activate')
run_path = base_path / 'data/raw/run'
processed_path = base_path / 'data/processed'

# Check paths exist etc.
if not log_path.exists(): log_path.mkdir(parents=True)
if not processed_path.exists(): processed_path.mkdir()
assert run_path.exists()
assert env_path.exists()


logging.info('Initialising the dask cluster')
# Set up the dask cluster
scluster = SLURMCluster(queue='standard',
                        project="n01-siAMOC",
                        job_cpu=128,
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
scluster.scale(jobs=1)


logging.info('Reading in model parameters from the namelist')
with open(run_path / 'data') as data:
    data_nml = f90nml.read(data)

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
                             chunks=300
                            )


logging.info('Calculating the potential vorticity')
grid = pvcalc.create_xgcm_grid(ds)
ds['drL'] = pvcalc.create_drL_from_dataset(ds)
ds['rho'] = pvcalc.calculate_density(ds['RHOAnoma'], ds['rhoRef'])
ds['b'] = pvcalc.calculate_buoyancy(ds['rho'])

ds['db_dx'], ds['db_dy'], ds['db_dz'] = pvcalc.calculate_grad_buoyancy(ds['b'], ds, grid)

ds['zeta_x'], ds['zeta_y'], ds['zeta_z'] = pvcalc.calculate_curl_velocity(ds['UVEL'],
                                                                          ds['VVEL'],
                                                                          ds['WVEL'],
                                                                          ds,
                                                                          grid,no_slip_bottom,
                                                                          no_slip_sides
                                                                         )

ds['Q'] = pvcalc.calculate_potential_vorticity(ds['zeta_x'],
                                               ds['zeta_y'],
                                               ds['zeta_z'],
                                               ds['db_dx'],
                                               ds['db_dy'],
                                               ds['db_dz'],
                                               ds,
                                               grid,
                                               beta,
                                               f0
                                              )

logging.info('Creating land masks')
ds['bool_land_mask'] = xr.where(-ds['Depth'] <= ds['Z'], 1, 0)
ds['nan_land_mask'] = xr.where(-ds['Depth'] <= ds['Z'], 1, np.nan)
slice_nan_land_mask = ds['nan_land_mask'].isel(YC=0).values

dst = ds.isel(time=-1)
xr.Dataset({'rho': dst['rho']}).to_zarr(processed_path / '../interim/test_rho')