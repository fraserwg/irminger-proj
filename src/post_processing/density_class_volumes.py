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
                             chunks=300,
                             iters=[71640]
                            )


logging.info('Creating a grid and calcualating the density')
ds['drL'] = pvcalc.create_drL_from_dataset(ds)
grid = pvcalc.create_xgcm_grid(ds)
ds['rho'] = pvcalc.calculate_density(ds['RHOAnoma'], ds['rhoRef'])

logging.info('Creating land masks')
ds['bool_land_mask'] = xr.where(-ds['Depth'] <= ds['Z'], 1, 0)
ds['nan_land_mask'] = xr.where(-ds['Depth'] <= ds['Z'], 1, np.nan)
slice_nan_land_mask = ds['nan_land_mask'].isel(YC=0).values

logging.info('Calculating volumes of density classes')
dVol = ds['dxF'] * ds['dyF'] * ds['drF'] * ds['bool_land_mask']
rho_t = grid.interp(ds['rho'].isel(time=-1), 'Z', boundary='fill', to='outer')
rho_t.name = 'rho_t'

#target_rho_levels = ds['rhoRef'].values
target_rho_levels = np.linspace(1026.1, 1027.4, 200)

dVol_rho_coords = grid.transform(dVol.chunk({'Z': -1}),
                                 'Z',
                                 target_rho_levels[::-1],
                                 target_data=rho_t.chunk({'Zp1': -1}),
                                 method='conservative'
                                 )

Volume = dVol_rho_coords.sel(XC=slice(0, 90e3)).cumsum('rho_t').sum('XC').transpose('rho_t', 'YC').compute()

scluster.close()
logging.info('Processing complete')

import matplotlib.pyplot as plt
import cmocean.cm as cmo
fig, ax = plt.subplots()
Volume.isel(YC=slice(300, -300)).plot.contourf(cmap=cmo.dense, levels=np.linspace(6, 8.2, 10) * 1e10)
ax.invert_yaxis()
ax.invert_xaxis()