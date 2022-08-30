import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO
                   )

logging.info('Importing standard python libraries')
from pathlib import Path

logging.info('Importing third party python libraries')
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
import xarray as xr
import zarr

logging.info("Importing custom python libraries")
import pvcalc

logging.info('Setting paths')
base_path = Path('/work/n01/n01/fwg/irminger-proj')
raw_path = base_path / 'data/raw'
interim_path = base_path / 'data/interim'
processed_path = base_path / 'data/processed'
figure_path = base_path / 'figures'
wmt_path = processed_path / 'wmt.zarr'
run_path = interim_path / 'ensemble.zarr'

log_path = base_path / 'src/post_processing/.tmp/slurm-out'
dask_worker_path = base_path / 'src/post_processing/.tmp/dask-worker-space'
env_path = base_path / 'irminger-proj/bin/activate'

mld_out_path = processed_path / "mld.zarr"
assert not mld_out_path.exists()

def create_encoding_for_ds(ds, clevel):
    compressor = zarr.Blosc(cname="zstd", clevel=clevel, shuffle=2)
    enc = {x: {"compressor": compressor} for x in ds}
    return enc


logging.info("Launching cluster")
scluster = SLURMCluster(queue="standard",
                        project="n01-siAMOC",
                        job_cpu=256,
                        log_directory=log_path,
                        local_directory=dask_worker_path,
                        cores=128,
                        processes=16,  # Can change this
                        memory="256 GiB",
                        header_skip= ["#SBATCH --mem="],  
                        walltime="00:20:00",
                        death_timeout=60,
                        interface="hsn0",
                        job_extra=["--qos=short", "--reservation=shortqos"],
                        env_extra=["module load cray-python",
                                   f"source {str(env_path.absolute())}"
                       )

client = Client(scluster)
scluster.scale(jobs=2)

ds_class = xr.open_zarr(wmt_path)

ds = xr.open_zarr(run_path, chunks={'Z': -1,
                                    'XC': -1,
                                    'time': 1,
                                    'run': -1})

ds['rho'] = pvcalc.calculate_density(ds["RHOAnoma"], ds["rhoRef"])

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


enc = create_encoding_for_ds(ds_mld, 9)
ds_mld.to_zarr(mld_out_path, encoding=enc)