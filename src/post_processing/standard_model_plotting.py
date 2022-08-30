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

rhonil, talpha = data_nml['parm01']['rhonil'], data_nml['parm01']['talpha']
ds_std = prep_for_wmt(ds_std, grid, rhonil, talpha)


logging.info('Setting plotting defaults')
# fonts
fpath = Path('/System/Library/Fonts/Supplemental/PTSans.ttc')
if fpath.exists():
    font_prop = fm.FontProperties(fname=fpath)
    plt.rcParams['font.family'] = font_prop.get_family()
    plt.rcParams['font.sans-serif'] = [font_prop.get_name()]

# font size
plt.rc('xtick', labelsize='10')
plt.rc('ytick', labelsize='10')
plt.rc('text', usetex=False)
plt.rcParams['axes.titlesize'] = 12

# output
dpi = 600
cm = 1/2.54

rho_levels = np.array([1026.92, 1026.98, 1027.05, 1027.1211])


plot_pv = True
if plot_pv:
    clim = 2e-9
    fig = plt.figure(figsize=[6, 4])
    gs = gridspec.GridSpec(2, 3, height_ratios=[14, 1])
    
    ax0 = fig.add_subplot(gs[0, 0])
    
    cax0 = ax0.pcolormesh(ds_std['XG'] * 1e-3,
                          -ds_std['Zl'],
                          ds_std['Q'].squeeze().sel(time=np.timedelta64(7, 'D')),
                          cmap=cmo.curl, shading='nearest',
                          vmin=-clim, vmax=clim, rasterized=True)
    ax0.contour(ds_std['XC'] * 1e-3,
                -ds_std['Zl'],
                ds_std['rho'].squeeze().sel(time=np.timedelta64(7, 'D')),
                cmap=cmo.dense,
                levels=rho_levels)
    
    ax1 = fig.add_subplot(gs[0, 1])
    cax1 = ax1.pcolormesh(ds_std['XG'] * 1e-3,
                          -ds_std['Zl'],
                          ds_std['Q'].squeeze().sel(time=np.timedelta64(14, 'D')),
                          cmap=cmo.curl, shading='nearest',
                          vmin=-clim, vmax=clim, rasterized=True)
    
    ax1.contour(ds_std['XC'] * 1e-3,
                -ds_std['Zl'],
                ds_std['rho'].squeeze().sel(time=np.timedelta64(14, 'D')),
                cmap=cmo.dense,
                levels=rho_levels)

    ax2 = fig.add_subplot(gs[0, 2])
    cax2 = ax2.pcolormesh(ds_std['XG'] * 1e-3,
                          -ds_std['Zl'],
                          ds_std['Q'].squeeze().sel(time=np.timedelta64(21, 'D'), method='nearest'),
                          cmap=cmo.curl, shading='nearest',
                          vmin=-clim, vmax=clim, rasterized=True)

    ax2.contour(ds_std['XC'] * 1e-3,
                -ds_std['Zl'],
                ds_std['rho'].squeeze().sel(time=np.timedelta64(21, 'D'),
                                            method='nearest'),
                cmap=cmo.dense,
                levels=rho_levels)

    ax0.set_ylim(250, 0)
    ax1.set_ylim(250, 0)
    ax2.set_ylim(250, 0)

    ax0.set_facecolor('grey')
    ax1.set_facecolor('grey')
    ax2.set_facecolor('grey')

    ax0.set_xlim(0, 100)
    ax1.set_xlim(0, 100)
    ax2.set_xlim(0, 100)

    ax0.set_title('1 week')
    ax1.set_title('2 weeks')
    ax2.set_title('3 weeks')
    
    ax0.set_title('(a)', loc='left')
    ax1.set_title('(b)', loc='left')
    ax2.set_title('(c)', loc='left')

    fig.suptitle(f'Run {std_run}')

    ax0.set_ylabel('Depth (m)')
    ax1.set_xlabel('Longitude (km)')

    ax1.set_yticklabels([])
    ax2.set_yticklabels([])
    
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    cbax = fig.add_subplot(gs[1, :])
    cb = fig.colorbar(cax0, cax=cbax, orientation='horizontal',
                      label='$Q$ (s$^{-3}$)', format=fmt)

    fig.tight_layout()
    
    figure_name = f'run{std_run}PV.pdf'
    fig.savefig(figure_path / figure_name, dpi=dpi)
    

plot_ics = True
if plot_ics:
    logging.info('Plotting initial conditions')

    # Open the datasats
    logging.info("Opening GEBCO bathymetry data")
    gebco_path = raw_path / 'GEBCO-bathymetry-data/gebco_2022_n80.0_s50.0_w-55.0_e5.0.nc'
    assert gebco_path.exists()
    
    ds_bathymetry = xr.open_dataset(gebco_path)
    ds_bathymetry = ds_bathymetry.coarsen(lon=5, lat=5, boundary='trim').mean()
    ds_mooring_locs = xr.open_dataset(raw_path / 'CFall_finergrid_2m.nc')
    ds_init = ds_std.isel(time=0).squeeze()

    logging.info("Setting up the plot area")
    pad = 35
    fig = plt.figure(figsize=(6, 9 * cm))

    gs = gridspec.GridSpec(2, 2,
                           width_ratios=[1, 1],
                           height_ratios=[1, 1/16]
                           )

    logging.info("Setting up the projection")
    proj = ccrs.LambertConformal(central_longitude=-25,
                                 #cutoff=50,
                                 central_latitude=65)
    
    ax1 = fig.add_subplot(gs[0, 0], projection=proj)

    logging.info("Plotting the bathymetry")
    cax_bathy = ax1.pcolormesh(ds_bathymetry['lon'],
                               ds_bathymetry['lat'],
                               -ds_bathymetry['elevation'],
                               shading='nearest',
                               rasterized=True,
                               cmap=cmo.deep,
                               vmin=0,
                               vmax=6000,
                               transform=ccrs.PlateCarree()
                              )
    
    xlim = [-55, 5]
    ylim = [50, 80]
    lower_space = 5 # this needs to be manually increased if the lower arched is cut off by changing lon and lat lims

    rect = mpath.Path([[xlim[0], ylim[0]],
                    [xlim[1], ylim[0]],
                    [xlim[1], ylim[1]],
                    [xlim[0], ylim[1]],
                    [xlim[0], ylim[0]],
                    ]).interpolated(20)

    proj_to_data = ccrs.PlateCarree()._as_mpl_transform(ax1) - ax1.transData
    rect_in_target = proj_to_data.transform_path(rect)

    ax1.set_boundary(rect_in_target)
    ax1.set_extent([xlim[0], xlim[1], ylim[0] - lower_space, ylim[1]])

    # logging.info("Adding land features")
    ax1.add_feature(cfeature.NaturalEarthFeature('physical',
                                                 'land',
                                                 '110m',
                                                 edgecolor='face',
                                                 facecolor='grey'
                                                ))
    
    ax1.plot(ds_mooring_locs['lon'],
                ds_mooring_locs['lat'][:-1],
                transform=ccrs.PlateCarree(),
                color='red', lw=4)
    
    ax1.gridlines()
    ax1.set_title("The Sub-Polar Atlantic")
    ax1.set_title("(a)", loc='left')
    cbax1 = fig.add_subplot(gs[1, 0])
    cb1 = plt.colorbar(cax_bathy, cax=cbax1, label='Depth (m)', orientation='horizontal')
    
    ax2 = fig.add_subplot(gs[0, 1])
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
    
    cbax = fig.add_subplot(gs[1, 1])
    cb = plt.colorbar(cax,
                      cax=cbax,
                      label='Meridional velocity (cm$\,$s$^{-1}$)',
                      orientation='horizontal')

    cb.formatter.set_useMathText(True)
    
    ax2.set_xlabel("Longitude (km)")
    ax2.set_ylabel("Depth (m)")
    ax2.set_title("Initial conditions")
    ax2.set_title("(b)", loc='left')
    
    ax2.set_ylim(500, 0)
    
    fig.tight_layout()
    fig.savefig(figure_path / "EnsembleICs.pdf", dpi=dpi)



plot_wmts = True
if plot_wmts:
    wmt_path = processed_path / "wmt.zarr"
    logging.info(f"Loading {wmt_path}")
    ds_wmts = xr.open_zarr(wmt_path)
    
    ds_wmt_std = ds_wmts.sel(run=std_run)
    
    fig, axs = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
    
    fig.suptitle(f"Run {std_run}")
    axs[0].set_title("Wind forcing")
    axs[1].set_title("Water mass formation")
    axs[2].set_title("Volume anomaly")
    
    axs[0].set_title("(a)", loc="left")
    axs[1].set_title("(b)", loc="left")
    axs[2].set_title("(c)", loc="left")
    
    
    logging.info("Calculating the winds")
    stress = ds["wind_stress"].sel(run=std_run).values
    sigma = ds["wind_duration"].sel(run=std_run).values
    days = 24 * 60 * 60
    t = np.array([float(time) * 1e-9 for time in ds_wmt_std['time']])
    tau = stress * np.exp(-(t - 10.5 * days)**2 / 2 / sigma **2)
    
    axs[0].plot(t / days, tau, c='k')
    
    for classs in ds_wmt_std['classs']:
        axs[1].plot(t / days,
                    ds_wmt_std['volTEND'].sel(classs=classs) * 1e-6 * 1e3,
                    label=classs.values)
        
        axs[2].plot(t / days,
                    ds_wmt_std['volANOM'].sel(classs=classs),
                    label=classs.values)
        
    axs[2].legend()
    fig.tight_layout()
    fig.show()
    fig.savefig(figure_path / f"WMT{std_run}.pdf")
    
    
plot_mlds = True
if plot_mlds:
    mld_path = processed_path / 'mld.zarr'
    
    logging.info(f"Loading {mld_path}")
    ds_mlds = xr.open_zarr(mld_path)
    
    ds_mld_std = ds_mlds.sel(run=std_run)
    
    fig, axs = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
    
    fig.suptitle(f"Run {std_run}")
    axs[0].set_title("Wind forcing")
    axs[1].set_title("Mixed layer depths")
    axs[2].set_title("Isopycnal depths")
    
    axs[0].set_title("(a)", loc="left")
    axs[1].set_title("(b)", loc="left")
    axs[2].set_title("(c)", loc="left")
    
    
    logging.info("Calculating the winds")
    stress = ds["wind_stress"].sel(run=std_run).values
    sigma = ds["wind_duration"].sel(run=std_run).values
    days = 24 * 60 * 60
    t = np.array([float(time) * 1e-9 for time in ds_mld_std['time']])
    tau = stress * np.exp(-(t - 10.5 * days)**2 / 2 / sigma **2)
    
    axs[0].plot(t / days, tau, c='k')

    axs[1].plot(t / days, -ds_mld_std['min_drop_depth'], label='max', ls='--', c="k")
    axs[1].plot(t / days, -ds_mld_std['mn_drop_depth'], label='mean', ls='-', c="k")
    
    axs[1].invert_yaxis()
    
    ds_mld_std['colour'] = ('classs', ['blue', 'orange', 'green', 'red' ,'purple'])
    axs[1].legend()
    
    for classs in ds_mld_std['classs'][:-1]:        
        axs[2].plot(t / days,
                    -ds_mld_std['min_iso_depth'].sel(classs=classs),
                    color=ds_mld_std['colour'].sel(classs=classs).item(),
                    ls='--')

        axs[2].plot(t / days,
                    -ds_mld_std['mn_iso_depth'].sel(classs=classs),
                    label=classs.item(),
                    color=ds_mld_std['colour'].sel(classs=classs).item(),
                    ls='-')
        
    
    axs[2].invert_yaxis()    
    axs[2].legend()
    fig.tight_layout()
    fig.show()
    
    fig.savefig(figure_path / f"MLD{std_run}.pdf")