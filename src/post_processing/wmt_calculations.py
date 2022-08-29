import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO
                   )

logging.info('Importing standard python libraries')
from pathlib import Path

logging.info('Importing third party python libraries')
import numpy as np
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
import xarray as xr
import zarr
import f90nml

logging.info("Importing custom python libraries")
import pvcalc

logging.info('Setting paths')
base_path = Path('/work/n01/n01/fwg/irminger-proj')
raw_path = base_path / 'data/raw'
interim_path = base_path / 'data/interim'
processed_path = base_path / 'data/processed'
figure_path = base_path / 'figures'
out_path = processed_path / 'wmt.zarr'

log_path = base_path / 'src/post_processing/.tmp/slurm-out'
dask_worker_path = base_path / 'src/post_processing/.tmp/dask-worker-space'
env_path = base_path / 'irminger-proj/bin/activate'

logging.info("Opening ensemble")
run_path = interim_path / "ensemble.zarr"
assert run_path.exists()
assert not out_path.exists()
ds = xr.open_zarr(run_path, chunks={'Z': -1,
                                    'XC': -1,
                                    'time': 1,
                                    'run': -1})

logging.info("Extracting namelist data")
data_nml = f90nml.read(raw_path / '2d-models/input_data_files_a/data')
delta_t = data_nml['parm03']['deltat']
f0 = ds.attrs['f0']
beta = ds.attrs['beta']
no_slip_bottom = ds.attrs['no_slip_bottom']
no_slip_sides = ds.attrs['no_slip_sides']
rhonil = data_nml['parm01']['rhonil']
talpha = data_nml['parm01']['talpha']

logging.info("Creating grid and doing density calculations")
grid = pvcalc.create_xgcm_grid(ds)
ds['drL'] = pvcalc.create_drL_from_dataset(ds)
ds['rho'] = pvcalc.calculate_density(ds['RHOAnoma'], ds['rhoRef'])
ds['b'] = pvcalc.calculate_buoyancy(ds['rho'])
_, _, ds['db_dz'] = pvcalc.calculate_grad_buoyancy(ds['b'], ds, grid)

ds['ZuMask'] = xr.where(grid.interp(ds['maskC'], ['Z'], to={'Z': 'right'}, boundary='fill') == 0,
                        np.nan, 1)

ds['db_dz'] = ds['db_dz'] * ds['ZuMask']

ds['rhoTEND'] = - rhonil * talpha * ds['TOTTTEND']
ds['bTEND'] = pvcalc.calculate_buoyancy(ds['rhoTEND'])
ds['hTEND'] = ds['bTEND'] / grid.interp(ds['db_dz'], 'Z', boundary='fill')

logging.info("Creating wmt dataset")

density_class_boundaries = np.array([1026.92, 1026.98, 1027.05, 1027.1211])
assert np.all(np.diff(density_class_boundaries) > 0)


class_coords = {'classs': ('classs',
                           range(0, len(density_class_boundaries) + 1)),
                
                'rho_upper': ('classs',
                              np.insert(density_class_boundaries, 0, 0)),
                
                'rho_lower': ('classs',
                              np.append(density_class_boundaries, np.inf)),
                
                'XC': ds['XC'],
                'XG': ds['XG'],
                'YC': ds['YC'],
                'YG': ds['YG'],
                'Z': ds['Z'],
                'Zl': ds['Zl'],
                'Zu': ds['Zu'],
                'Zp1': ds['Zp1'],}

ds_class = xr.Dataset(coords=class_coords)
ds['NaNmaskC'] = xr.where(ds['maskC'] == 1, 1, np.NaN)
ds_class['rho'] = ds['rho'] * ds['NaNmaskC'] * xr.ones_like(ds_class['classs'])

ds_class['mask'] = xr.where(ds_class['rho'] <= ds_class['rho_lower'],
                            True,
                            False) * xr.where(ds_class['rho'] > ds_class['rho_upper'],
                                              True,
                                              False)

def invert_bool_mask(da):
    return xr.where(da == True, False, True)

def dVol_dt(db_dz, classMask, hTend):
    db_dz_mask = grid.interp(xr.where(db_dz == 0, 0, db_dz), 'Z',boundary='extend')
    hTends = classMask * (invert_bool_mask(classMask.shift({'Z': 1})) + -1 * invert_bool_mask(classMask.shift({'Z': -1}))) * hTend * db_dz_mask
    VolTend = (hTends.sum('Z') * hTend['dxF']).sum('XC')
    return VolTend

ds_class['volTEND'] = dVol_dt(ds['db_dz'], ds_class['mask'], ds['hTEND'])
ds_class['vol'] = (ds_class['mask'] * ds['drF']).sum(['Z', 'XC'])
ds_class['volANOM'] = ds_class['vol'] - ds_class['vol'].isel(time=0)

tdays = ds_class['time'].values.astype('float32') * 1e-9 / 24 / 60 / 60
ds_class = ds_class.assign_coords({'tdays': ('time',
                                             tdays,
                                             {'units': 'days'})}) 


ds_class = ds_class.drop(['rho', 'mask']).squeeze()

def create_encoding_for_ds(ds, clevel):
    compressor = zarr.Blosc(cname="zstd", clevel=clevel, shuffle=2)
    enc = {x: {"compressor": compressor} for x in ds}
    return enc

logging.info("Launching cluster")
scluster = SLURMCluster(queue='standard',
                        project="n01-siAMOC",
                        job_cpu=256,
                        log_directory=log_path,
                        local_directory=dask_worker_path,
                        cores=16,
                        processes=16,  # Can change this
                        memory="512 GiB",
                        header_skip= ['#SBATCH --mem='],  
                        walltime="00:20:00",
                        death_timeout=60,
                        interface='hsn0',
                        job_extra=["--qos=highmem", "--partition=highmem"],
                        env_extra=['module load cray-python',
                                   'source {}'.format(str(env_path.absolute()))]
                       )

client = Client(scluster)
scluster.scale(jobs=8)

logging.info("Rechunking dataset")
# ds_class = ds_class.chunk({'time': -1,
#                            'classs': -1,})

logging.info(f"Saving dataset to {out_path}")
enc = create_encoding_for_ds(ds_class, 9)
ds_class.to_zarr(out_path, encoding=enc)