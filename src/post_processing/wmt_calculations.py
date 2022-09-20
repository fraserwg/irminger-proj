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

logging.info("Calculations to be performed:")
wmt = True
mld = False
logging.info(f"water mass transformation: {wmt}")
logging.info(f"mixed layer depth: {mld}")

logging.info('Setting paths')
base_path = Path('/work/n01/n01/fwg/irminger-proj')
raw_path = base_path / 'data/raw'
interim_path = base_path / 'data/interim'
processed_path = base_path / 'data/processed'
figure_path = base_path / 'figures'
out_path = processed_path / 'enswmt.zarr'

log_path = base_path / 'src/post_processing/.tmp/slurm-out'
dask_worker_path = base_path / 'src/post_processing/.tmp/dask-worker-space'
env_path = base_path / 'irminger-proj/bin/activate'

logging.info("Opening ensemble")
run_path = interim_path / "ensemble.zarr"
assert run_path.exists()
#assert not out_path.exists()
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
db = grid.diff(ds['b'], 'Z', boundary='extend', to="right")

ds['hTEND2'] = - 1 / np.square(ds['db_dz']) * grid.diff(ds['bTEND'], 'Z', to="right", boundary="extend") / ds['drL'] * db

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
                            
ds_class['rhoL'] = grid.interp(ds['rho'], 'Z', to="right", boundary="extend")
ds_class['maskL'] = xr.where(ds_class['rhoL'] <= ds_class['rho_lower'],
                            True,
                            False) * xr.where(ds_class['rhoL'] > ds_class['rho_upper'],
                                              True,
                                              False)

def invert_bool_mask(da):
    return xr.where(da == True, False, True)

def dVol_dt(db_dz, classMask, hTend):
    db_dz_mask = xr.where(db_dz == 0, 0, 1)
    hTends = classMask * hTend * db_dz_mask
    # hTends = classMask * (invert_bool_mask(classMask.shift({'Z': 1})) + -1 * invert_bool_mask(classMask.shift({'Z': -1}))) * hTend * db_dz_mask
    VolTend = (hTends.sum('Zl') * hTend['dxF']).sum('XC')
    return VolTend

#ds_class['volTEND'] = dVol_dt(ds['db_dz'], ds_class['maskL'], ds['hTEND2'])
logging.info("Not currently calculating volume tendenct as method is flawed")

ds_class["vol"] = (ds_class["mask"] * ds["drF"] * ds["dxF"]).sum(["Z", "XC"])
ds_class["vol"].attrs = {"units": "m3 m-1"}

ds_class['volANOM'] = ds_class['vol'] - ds_class['vol'].isel(time=0)
ds_class["volANOM"].attrs = {"units": "m3 m-1"}


tdays = ds_class['time'].values.astype('float32') * 1e-9 / 24 / 60 / 60
ds_class = ds_class.assign_coords({'tdays': ('time',
                                             tdays,
                                             {'units': 'days'})}) 


ds_class = ds_class.drop(['rho', 'mask', 'rhoL', 'maskL']).squeeze()

def create_encoding_for_ds(ds, clevel):
    compressor = zarr.Blosc(cname="zstd", clevel=clevel, shuffle=2)
    enc = {x: {"compressor": compressor} for x in ds}
    return enc

if __name__=="__main__":
    if wmt:
        # scluster = SLURMCluster(queue='standard',
        #                         project="n01-siAMOC",
        #                         job_cpu=256,
        #                         log_directory=log_path,
        #                         local_directory=dask_worker_path,
        #                         cores=16,
        #                         processes=16,  # Can change this
        #                         memory="256 GiB",
        #                         header_skip= ['#SBATCH --mem='],  
        #                         walltime="00:15:00",
        #                         death_timeout=60,
        #                         interface='hsn0',
        #                         job_extra=["--qos=short", 
        #                                    "--reservation=shortqos"],
        #                         env_extra=['module load cray-python',
        #                                 'source {}'.format(str(env_path.absolute()))]
        #                     )

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
                                walltime="00:10:00",
                                death_timeout=60,
                                interface='hsn0',
                                job_extra=["--qos=highmem", "--partition=highmem"],
                                env_extra=['module load cray-python',
                                        'source {}'.format(str(env_path.absolute()))]
                            )


        client = Client(scluster)
        scluster.scale(jobs=8)

        logging.info(f"Saving dataset to {out_path}")
        enc = create_encoding_for_ds(ds_class, 9)
        ds_class.to_zarr(out_path, encoding=enc)

        scluster.close()




    if mld:  
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
                                walltime="00:15:00",
                                death_timeout=60,
                                interface='hsn0',
                                job_extra=["--qos=standard", "--reservation=short"],
                                env_extra=['module load cray-python',
                                        'source {}'.format(str(env_path.absolute()))]
                            )

        client = Client(scluster)
        scluster.scale(jobs=8)
        
        logging.info("Doing MLD calculations")
        logging.info("Isopycnal depths")
        # idxmax('Z') finds the first depth at which the condition is satisfied
        iso_depth = ds['rho'].where(lambda x: x <= ds_class['rho_lower']).idxmax('Z')
        iso_depth = iso_depth.compute()
        da_mn_iso_depth = iso_depth.mean('XC')
        da_max_iso_depth = iso_depth.max('XC')
        da_min_iso_depth = iso_depth.min('XC')

        logging.info("Delta rho drop depths")
        delta_rho = 0.05
        attrs = {'delta_rho': delta_rho,
                'delta_rho_units': 'kg m-3'}

        da_delta_rho_surf = ds['rho'] - ds['rho'].isel(Z=0)
        delta_rho_drop = da_delta_rho_surf.where(lambda x: x<= delta_rho).idxmax('Z')
        delta_rho_drop = delta_rho_drop.compute()
        da_mn_drop_depth = delta_rho_drop.mean('XC').assign_attrs(attrs)
        da_max_drop_depth = delta_rho_drop.max('XC').assign_attrs(attrs)
        da_min_drop_depth = delta_rho_drop.min('XC').assign_attrs(attrs)

        ds_mld = xr.Dataset({'mn_iso_depth': da_mn_iso_depth,
                            'min_iso_depth': da_min_iso_depth,
                            'max_iso_depth': da_max_iso_depth,
                            'mn_drop_depth': da_mn_drop_depth,
                            'min_drop_depth': da_min_drop_depth,
                            'max_drop_depth': da_max_drop_depth})

        mld_out_path = processed_path / "mld.zarr"
        enc = create_encoding_for_ds(ds_mld, 9)
        ds_mld.to_zarr(mld_out_path, encoding=enc)