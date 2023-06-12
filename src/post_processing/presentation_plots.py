import logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO
                   )

logging.info('Importing standard python libraries')
from pathlib import Path

logging.info('Importing third party python libraries')
import numpy as np
#import dask
#from dask.distributed import Client
#from dask_jobqueue import SLURMCluster
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from matplotlib import gridspec
from matplotlib.ticker import ScalarFormatter
import matplotlib.font_manager as fm
import cartopy.feature as cfeature
import cartopy.crs as ccrs
#from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

import cmocean.cm as cmo
import f90nml

logging.info("Importing custom python libraries")
import pvcalc

logging.info('Setting paths')
base_path = Path('/work/n01/n01/fwg/irminger-proj')
raw_path = base_path / 'data/raw'
interim_path = base_path / 'data/interim'
processed_path = base_path / 'data/processed'
figure_path = base_path / 'figures'
ensemble_path = interim_path / "ensemble.zarr"
mld_path = processed_path / "mld.zarr"
wmt_path = processed_path / "wmt.zarr"

logging.info("Opening ensemble")
run_path = interim_path / "ensemble.zarr"
assert run_path.exists()
ds = xr.open_zarr(run_path)

data_nml = f90nml.read(raw_path / '2d-models/input_data_files_a/data')
delta_t = data_nml['parm03']['deltat']
f0 = ds.attrs['f0']
beta = ds.attrs['beta']
no_slip_bottom = ds.attrs['no_slip_bottom']
no_slip_sides = ds.attrs['no_slip_sides']

std_run = 32
logging.info(f"Standard run = {std_run}")
ds_std = ds.sel(run=std_run)

def select_marker(wind_duration):
    if np.allclose(wind_duration, 1.08e5):
        marker, colour = "x", "blue"
    elif np.allclose(wind_duration, 2.16e5):
        marker, colour = "o", "orange"
    elif np.allclose(wind_duration, 3.24e5):
        marker, colour = "^", "green"
    elif np.allclose(wind_duration, 4.32e5):
        marker, colour = "s", "red"
    elif np.allclose(wind_duration, 0):
        marker, colour = "D", "black"
    else:
        raise ValueError("wind_duration not as expected")
    return marker, colour


def prep_for_wmt(ds, grid, rhonil, talpha):
    ds['rhoTEND'] = - rhonil * talpha * ds['TOTTTEND']
    ds['bTEND'] = pvcalc.calculate_buoyancy(ds['rhoTEND'])
    ds['hTEND'] = ds['bTEND'] / grid.interp(ds['db_dz'], 'Z', boundary='fill')
    return ds


def prep_for_pv(ds):
    grid = pvcalc.create_xgcm_grid(ds)
    ds['drL'] = pvcalc.create_drL_from_dataset(ds)
    ds['rho'] = pvcalc.calculate_density(ds['RHOAnoma'], ds['rhoRef'])
    ds['b'] = pvcalc.calculate_buoyancy(ds['rho'])

    ds['db_dx'], ds['db_dy'], ds['db_dz'] = pvcalc.calculate_grad_buoyancy(ds['b'], ds, grid)
    
    db_dz_mask = xr.where(grid.interp(ds['maskC'],
                                      ['Z'],
                                      to={'Z': 'right'},
                                      boundary='fill') == 0,
                          np.nan,
                          1)

    ds['db_dz'] = ds['db_dz'] * db_dz_mask

    ds['zeta_x'], ds['zeta_y'], ds['zeta_z'] = pvcalc.calculate_curl_velocity(ds['UVEL'],
                                                                          ds['VVEL'],
                                                                          ds['WVEL'],
                                                                          ds,
                                                                          grid,no_slip_bottom,
                                                                          no_slip_sides
                                                                         )

    
    ds['NaNmaskC'] = xr.where(ds['maskC'] == 1, 1, np.NaN)
    
    ds['maskQ'] = grid.interp(ds['maskW'],
                              ['Y', 'Z'],
                              to={'Z': 'right', 'Y': 'left'},
                              boundary='fill')
    
    ds['NaNmaskQ'] = xr.where(ds['maskQ'] == 0, np.nan, 1)
    
    ds['Q'] = pvcalc.calculate_C_potential_vorticity(ds['zeta_x'],
                                                    ds['zeta_y'],
                                                    ds['zeta_z'],
                                                    ds['b'],
                                                    ds,
                                                    grid,
                                                    beta,
                                                    f0
                                                    ) * ds['NaNmaskQ']

    
    
    
    return ds, grid


ds_std, grid = prep_for_pv(ds_std)
days = 24 * 60 * 60
rhonil, talpha = data_nml['parm01']['rhonil'], data_nml['parm01']['talpha']
ds_std = prep_for_wmt(ds_std, grid, rhonil, talpha)

logging.info('Setting plotting defaults')
# fonts
if Path('/System/Library/Fonts/Supplemental/PTSans.ttc').exists():
    fpath = Path('/System/Library/Fonts/Supplemental/PTSans.ttc')
elif Path('/home/n01/n01/fwg/.local/share/fonts/HelveticaNeue.ttf').exists():
    fpath = Path('/home/n01/n01/fwg/.local/share/fonts/HelveticaNeue.ttf')
else:
    fpath = None
if fpath != None:
    print(fpath)
    font_prop = fm.FontProperties(fname=fpath)
    plt.rcParams['font.family'] = font_prop.get_family()
    plt.rcParams['font.sans-serif'] = [font_prop.get_name()]

# font size
plt.rc('xtick', labelsize='14')
plt.rc('ytick', labelsize='14')
plt.rc('text', usetex=False)
plt.rcParams['axes.titlesize'] = 14

# output
dpi = 600



rho_levels = np.array([1026.92 - 0.06, 1026.92, 1026.98, 1027.05, 1027.1211])


plot_pv = False
if plot_pv:
    clim = 2e-9
    fig = plt.figure(figsize=[6, 6])
    gs = gridspec.GridSpec(2, 2, height_ratios=[20, 1])
    
    ax0 = fig.add_subplot(gs[0])
    
    nd = 7
    cax0 = ax0.pcolormesh(ds_std['XG'] * 1e-3,
                          -ds_std['Zl'],
                          ds_std['Q'].squeeze().sel(time=np.timedelta64(nd, 'D')),
                          cmap=cmo.curl, shading='nearest',
                          vmin=-clim, vmax=clim, rasterized=True)
    
    ax0.contour(ds_std['XC'] * 1e-3,
                -ds_std['Zl'],
                ds_std['rho'].squeeze().sel(time=np.timedelta64(nd, 'D')),
                cmap=cmo.dense,
                levels=rho_levels)
    
    ax1 = fig.add_subplot(gs[1])
    
    cax0 = ax1.pcolormesh(ds_std['XG'] * 1e-3,
                          -ds_std['Zl'],
                          ds_std['Q'].squeeze().sel(time=np.timedelta64(14, 'D')),
                          cmap=cmo.curl, shading='nearest',
                          vmin=-clim, vmax=clim, rasterized=True)
    
    ax1.contour(ds_std['XC'] * 1e-3,
                -ds_std['Zl'],
                ds_std['rho'].squeeze().sel(time=np.timedelta64(14, 'D')),
                cmap=cmo.dense,
                levels=rho_levels)

    ax0.set_ylim(200, 0)
    ax1.set_ylim(200, 0)
    
    ax0.set_facecolor('grey')
    ax1.set_facecolor('grey')
    
    
    ax0.set_xlim(0, 100)
    ax1.set_xlim(0, 100)
    
        
    ax0.set_ylabel('Depth (m)', fontsize=16)
    ax0.set_xlabel('Longitude (km)', fontsize=16)
    ax1.set_xlabel('Longitude (km)', fontsize=16)
    
    
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    cbax = fig.add_subplot(gs[1, :])
    cb = fig.colorbar(cax0, cax=cbax, orientation='horizontal', format=fmt)

    cb.set_label("$Q$ (s$^{-3}$)", fontsize=16)

    fig.tight_layout()
    
    figure_name = f'Pres{std_run}PV.pdf'
    fig.savefig(figure_path / figure_name, dpi=dpi)
    

plot_ics = False
if plot_ics:
    logging.info('Plotting initial conditions')

    # Open the datasats
    ds_init = ds_std.isel(time=0).squeeze()

    logging.info("Setting up the plot area")
    pad = 35
    fig = plt.figure(figsize=(4, 4))

    gs = gridspec.GridSpec(2, 1,
                           width_ratios=[1],
                           height_ratios=[1, 1/16]
                           )
    
    ax2 = fig.add_subplot(gs[0, 0])
    cmo.tempo_r.set_bad('grey')
                         
    cax = ax2.pcolormesh(ds_init['XC'] * 1e-3,
                         -ds_init['Z'],
                         ds_init['VVEL'] * ds_init['NaNmaskC'] * 1e2,
                         vmin=-20,
                         vmax=0,
                         shading='nearest',
                         cmap=cmo.tempo_r,
                         rasterized=True)
    
    ax2.contour(ds_init['XC'] * 1e-3,
                -ds_init['Z'],
                ds_init['rho'] * ds_init['NaNmaskC'],
                levels=rho_levels,
                cmap=cmo.dense,
                linewidths=2)
    
    cbax = fig.add_subplot(gs[1, 0])
    cb = plt.colorbar(cax,
                      cax=cbax,
                      orientation='horizontal')
    
    cb.set_label("Meridional velocity (cm$\,$s$^{-1}$)", fontsize=16)

    cb.formatter.set_useMathText(True)
    
    ax2.set_xlabel("Longitude (km)", fontsize=16)
    ax2.set_ylabel("Depth (m)", fontsize=16)
    
    ax2.set_ylim(500, 0)
    
    fig.tight_layout()
    fig.savefig(figure_path / "PresGeostrophy.pdf", dpi=dpi)

ensemble_mld = True
if ensemble_mld:
    ds_ensemble = xr.open_zarr(ensemble_path)
    ds_mld = xr.open_zarr(mld_path).squeeze()
    delta_mld = ds_mld["mn_drop_depth"].isel(time=0) - ds_mld["mn_drop_depth"].isel(time=-1)
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    for run in ds_mld['run']:
        marker, colour = select_marker(ds_ensemble["wind_duration"].sel(run=run))
        
        ax.scatter(-ds_ensemble["wind_stress"].sel(run=run),
                   delta_mld.sel(run=run),
                   marker=marker,
                   color=colour)   
    
    
    ax.scatter(None, None, marker="D", color="black", label=0)
    ax.scatter(None, None, marker="x", color="blue", label=1.08e5 / days)
    ax.scatter(None, None, marker="o", color="orange", label=2.16e5 / days)
    ax.scatter(None, None, marker="^", color="green", label=3.24e5 / days)
    ax.scatter(None, None, marker="s", color="red", label=4.32e5 / days)
        
    ax.set_xlabel("Wind stress (N$\\,$m$^{-2}$)", fontsize=16)
    ax.set_ylabel("$\Delta$ MLD (m)", fontsize=16)
    ax.legend(title="Wind duration (days)", fontsize=12, loc="upper left")
    fig.tight_layout()
    fig.savefig(figure_path / "PresDeltaMLD.pdf")