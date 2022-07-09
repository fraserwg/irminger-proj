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

ds['bool_land_mask'].isel(YC=0).to_netcdf(processed_path / 'bool_land_mask.nc')


logging.info('Creating initial and boundary conditions')
V_init = ds['VVEL'].isel(YG=0, time=0) * slice_nan_land_mask
rho_init = ds['rhoRef']
ds_init = xr.Dataset({'V_init': V_init, 'rho_init': rho_init})
ds_init.to_netcdf(processed_path / 'init.nc')


logging.info('Creating overturning mechanism datasets')
rho_3 = ds['rho'].isel(time=slice(-10, -1)).sel(YC=-250e3, XC=90e3, method='nearest').mean('time')
zetay_3 = ds['zeta_y'].isel(time=slice(-10, -1)).sel(YC=-250e3, XG=90e3, method='nearest').mean('time')
ds_overturning = xr.Dataset({'zeta_y': zetay_3, 'rho': rho_3})
ds_overturning.to_netcdf(processed_path / 'overturning.nc')


logging.info('Creating stratification slice dataset')
ylats = [600e3, 800e3, 1000e3]
dbdz_slice = ds['db_dz'].isel(time=-1).sel(YC=ylats, method='nearest') * np.swapaxes(np.tile(slice_nan_land_mask, (len(ylats), 1, 1)), 0, 1)
dbdz_slice.to_netcdf(processed_path / 'dbdz_slice.nc')


logging.info('Creating meridional vorticity slice dataset')
zetay_slice = ds['zeta_y'].isel(time=slice(-10, -1)).sel(YC=800e3, method='nearest').mean('time') * slice_nan_land_mask
zetay_slice.to_netcdf(processed_path / 'zeta_y_slice.nc')


logging.info('Creating potential vorticity slice dataset')
ylats = [600e3, 800e3, 1000e3]
Q_slice = ds['Q'].isel(time=-1).sel(YC=ylats, method='ffill') * np.swapaxes(np.tile(slice_nan_land_mask, (len(ylats), 1, 1)), 0, 1)
Q_slice.to_netcdf(processed_path / 'Q_slice.nc')


logging.info('Calculating potential vorticity on a density level')
Q_t = ds['Q'].isel(time=-1)
rho_t = ds['rho'].isel(time=-1)
target_rho_levels = ds['rhoRef'].sel(Z=[-10], method='nearest').values  # -100 m is arbitrary

# Create a 2D mask which sets land points to nan
masked_rho = rho_t * ds['nan_land_mask']
land_point_zeros = xr.where(masked_rho >= target_rho_levels, 1, 0).sum('Z') # Where zero, land, else water
target_rho_mask = xr.where(land_point_zeros == 0, np.nan, 1)

Q_on_rho = grid.transform(Q_t.chunk({'Z': -1}),
                          'Z',
                          target_rho_levels,
                          target_data=rho_t.chunk({'Z': -1}),
                          method='linear'
                         ) * target_rho_mask

logging.info('Saving potential vorticity on a density level dataset')
Q_on_rho.to_netcdf(processed_path / 'Q_on_rho.nc')

scluster.close()
logging.info('Processing complete')