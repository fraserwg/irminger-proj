import logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO
                   )

logging.info('Importing standard python libraries')
from pathlib import Path

logging.info('Importing third party python libraries')
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from matplotlib import gridspec
from matplotlib.ticker import ScalarFormatter
import matplotlib.font_manager as fm
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import cmocean.cm as cmo
import cycler
import f90nml


logging.info("Importing custom python libraries")
import pvcalc

plot_mlds = False
plot_pv = False
plot_ics = False
plot_strat = False
plot_classes = False

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

logging.info("Setting plotting defaults")
logging.info('Setting plotting defaults')
fpath = Path('/home/n01/n01/fwg/.local/share/fonts/PTSans-Regular.ttf')
assert fpath.exists()
font_prop = fm.FontProperties(fname=fpath)
plt.rcParams['font.family'] = font_prop.get_family()
plt.rcParams['font.sans-serif'] = [font_prop.get_name()]

# font size
plt.rc('xtick', labelsize='8')
plt.rc('ytick', labelsize='8')
plt.rc('text', usetex=False)
plt.rcParams['axes.titlesize'] = 10

# output
dpi = 600
text_width = 6


n = 6
color = cmo.dense(np.linspace(0, 1,n))
plt.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)


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

    grad_b = pvcalc.calculate_grad_buoyancy(ds['b'], ds, grid)
    ds['db_dx'], ds['db_dy'], ds['db_dz'] = grad_b
    
    db_dz_mask = xr.where(grid.interp(ds['maskC'],
                                      ['Z'],
                                      to={'Z': 'right'},
                                      boundary='fill') == 0,
                          np.nan,
                          1)

    ds['db_dz'] = ds['db_dz'] * db_dz_mask

    curl_vel = pvcalc.calculate_curl_velocity(ds['UVEL'],
                                              ds['VVEL'],
                                              ds['WVEL'],
                                              ds,
                                              grid,no_slip_bottom,
                                              no_slip_sides)

    ds['zeta_x'], ds['zeta_y'], ds['zeta_z'] = curl_vel
    
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




rho_levels = np.array([1026.92, 1026.98, 1027.05, 1027.1211])


if plot_pv:
    logging.info("Plotting the potential vorticity")
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
                colors=color[1:],
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
                colors=color[1:],
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
                colors=color[1:],
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

    fig.suptitle(f'Potential vorticity')

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

if plot_strat:
    logging.info("Plotting the stratification")
    clim = 1e-5
    cmap = cmo.tarn
    fig = plt.figure(figsize=[6, 4])
    gs = gridspec.GridSpec(2, 3, height_ratios=[14, 1])
    
    ax0 = fig.add_subplot(gs[0, 0])
    
    cax0 = ax0.pcolormesh(ds_std['XG'] * 1e-3,
                          -ds_std['Zl'],
                          ds_std['db_dz'].squeeze().sel(time=np.timedelta64(7, 'D')),
                          cmap=cmap, shading='nearest',
                          vmin=-clim, vmax=clim, rasterized=True)

    
    ax1 = fig.add_subplot(gs[0, 1])
    cax1 = ax1.pcolormesh(ds_std['XG'] * 1e-3,
                          -ds_std['Zl'],
                          ds_std['db_dz'].squeeze().sel(time=np.timedelta64(14, 'D')),
                          cmap=cmap, shading='nearest',
                          vmin=-clim, vmax=clim, rasterized=True)
    

    ax2 = fig.add_subplot(gs[0, 2])
    cax2 = ax2.pcolormesh(ds_std['XG'] * 1e-3,
                          -ds_std['Zl'],
                          ds_std['db_dz'].squeeze().sel(time=np.timedelta64(21, 'D'), method='nearest'),
                          cmap=cmap, shading='nearest',
                          vmin=-clim, vmax=clim, rasterized=True)

    ax0.contour(ds_std['XC'] * 1e-3,
                -ds_std['Zl'],
                ds_std['rho'].squeeze().sel(time=np.timedelta64(7, 'D')),
                colors=color[1:],
                levels=rho_levels)
    
    ax1.contour(ds_std['XC'] * 1e-3,
            -ds_std['Zl'],
            ds_std['rho'].squeeze().sel(time=np.timedelta64(14, 'D')),
            colors=color[1:],
            levels=rho_levels)
    
    ax2.contour(ds_std['XC'] * 1e-3,
                -ds_std['Zl'],
                ds_std['rho'].squeeze().sel(time=np.timedelta64(21, 'D'),
                                            method='nearest'),
                colors=color[1:],
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

    fig.suptitle(f'Stratification')

    ax0.set_ylabel('Depth (m)')
    ax1.set_xlabel('Longitude (km)')

    ax1.set_yticklabels([])
    ax2.set_yticklabels([])
    
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    cbax = fig.add_subplot(gs[1, :])
    cb = fig.colorbar(cax0, cax=cbax, orientation='horizontal',
                      label='$\partial_z b$ (s$^{-2}$)', format=fmt)

    fig.tight_layout()
    
    figure_name = f'run{std_run}Strat.pdf'
    fig.savefig(figure_path / figure_name, dpi=dpi)

  
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
    fig = plt.figure(figsize=(6, 3.25))

    gs = gridspec.GridSpec(2, 2,
                           width_ratios=[1, 1],
                           height_ratios=[1, 1/16]
                           )

    logging.info("Setting up the projection")
    proj = ccrs.LambertConformal(central_longitude=-25,
                                 #cutoff=50,
                                 central_latitude=65)
    
    ax1 = fig.add_subplot(gs[0, 0], projection=proj)
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
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
    lower_space = 3 # this needs to be manually increased if the lower arch is cut off by changing lon and lat lims

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
                color='tab:red', lw=4)
    
    
    gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,#draw_labels={"bottom": "x", 
                                                #            "left": "y"},
                       linewidth=1, linestyle='-', x_inline=False, 
                       y_inline=False, rotate_labels=False,
                       xlocs=[-50, -35, -20, -5, 10], ylocs=[55, 65, 75])
    
    gl.top_labels = False
    #gl.right_labels = False
    

    # gl.xlocator = mticker.FixedLocator([-50, -35, -20, -5, 10])
    # gl.ylocator = mticker.FixedLocator([55, 65, 75])
    #gl.ylocator = LatitudeLocator()
    #gl.xformatter = LongitudeFormatter()
    #gl.yformatter = LatitudeFormatter()
    #gl.xlabel_style = {'size': 15, 'color': 'gray'}
    #gl.xlabel_style = {'color': 'red', 'weight': 'bold'}
    
    ax1.set_title("The Sub-Polar Atlantic")
    ax1.set_title("(a)", loc='left')
    cbax1 = fig.add_subplot(gs[1, 0])
    cb1 = plt.colorbar(cax_bathy,
                       cax=cbax1,
                       label='Depth (m)',
                       orientation='horizontal')
    
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
    
    n = 6
    color = cmo.dense(np.linspace(0, 1, n))
    plt.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)
    cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', ["C1", "C2", "C3", "C4", "C5"], 5)
    
    
    ax2.contour(ds_init['XC'] * 1e-3,
                -ds_init['Z'],
                ds_init['rho'] * ds_init['NaNmaskC'],
                levels=rho_levels,
                colors=["C1", "C2", "C3", "C4"],
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

if plot_mlds:
    mld_path = processed_path / 'mld.zarr'
    wmt_path = processed_path / "wmt.zarr"
    
    ds_wmts = xr.open_zarr(wmt_path)
    ds_wmt_std = ds_wmts#.sel(run=std_run)
    
    
    
    logging.info(f"Loading {mld_path}")
    ds_mlds = xr.open_zarr(mld_path)
    ds_wmt_std['volTEND2'] = ds_wmt_std["volANOM"].diff("time") / ds_wmt_std["time"].astype("float32").diff("time") / 1e-9 
    ds_mld_std = ds_mlds.sel(run=std_run)
    
    fig, axs = plt.subplots(4, 1, figsize=(6, 7), sharex=True)
    
    fig.suptitle(f"Reference integration")
    axs[0].set_title("Wind forcing")
    axs[1].set_title("Mixed layer and class boundary depths")
    axs[2].set_title("Volume anomaly")
    
    axs[0].set_title("(a)", loc="left")
    axs[1].set_title("(b)", loc="left")
    axs[2].set_title("(c)", loc="left")
    axs[3].set_title("(d)", loc="left")
    
    logging.info("Calculating the winds")
    stress = ds["wind_stress"].sel(run=std_run).values
    sigma = ds["wind_duration"].sel(run=std_run).values
    days = 24 * 60 * 60
    t = np.array([float(time) * 1e-9 for time in ds_mld_std['time']])
    tau = stress * np.exp(-(t - 10.5 * days)**2 / 2 / sigma **2)
    
    axs[0].plot(t / days, tau, c='k')
    axs[0].set_ylabel("$\\tau_y$ (N$\,$m$^{-2}$)")

    
    axs[1].invert_yaxis()
    
    ds_mld_std['colour'] = ('classs', ['C1', 'C2', 'C3', 'C4' ,'C5'])
    axs[1].set_ylabel("Depth (m)")
    
    for classs in ds_mld_std['classs'][:-1]:        
        axs[1].plot(t / days,
                    -ds_mld_std['min_iso_depth'].sel(classs=classs),
                    color=ds_mld_std['colour'].sel(classs=classs).item(),
                    ls='--')

        axs[1].plot(t / days,
                    -ds_mld_std['mn_iso_depth'].sel(classs=classs),
                    color=ds_mld_std['colour'].sel(classs=classs).item(),
                    ls='-')
    
    axs[1].plot(t / days, -ds_mld_std['min_drop_depth'], ls='--', c="tab:orange")
    axs[1].plot(t / days, -ds_mld_std['mn_drop_depth'], label='Mixed layer',ls='-', c="tab:orange")
    
    axs[1].plot(np.NaN, np.NaN, ls="-", c="k", label="Mean")
    axs[1].plot(np.NaN, np.NaN, ls="--", c="k", label="Maximum")
     

    for classs in ds_wmt_std['classs']:
        axs[2].plot(t / days,
                    ds_wmt_std['volANOM'].sel(classs=classs) * 1e3 * 1e-9,
                    label=f"Class {classs.item()}",
                    color=f"C{classs + 1}")
        
        axs[3].plot(t / days,
                    ds_wmt_std['volTEND2'].sel(classs=classs) * 1e-6 * 1e3 * 1e2,
                    color=f"C{classs + 1}")
        
        axs[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        
    axs[2].set_ylabel("$\Delta \mathcal{V}$ ($\\times$10$^{9}$ m$^3\,$km$^{-1}$)")
    
    fig.legend()

    axs[3].set_ylabel("$\partial_t \mathcal{V}$ ($\\times$10$^{-2}$ Sv$\,$km$^{-1}$)")
    axs[3].set_xlabel("Time (days)")
    axs[3].set_title("Smoothed formation rate")
    axs[3].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    
    axs[0].set_xlim(0, 21)
    fig.tight_layout()
    fig.show()
    
    fig_path = figure_path / f"MLD{std_run}.pdf"
    logging.info(f"saving figure to {fig_path}")
    fig.savefig(fig_path)
    

if plot_classes:
    ds_init = ds_std.isel(time=0).squeeze()
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
    ds_init['NaNmaskC'] = xr.where(ds_init['maskC'] == 1, 1, np.NaN)
    ds_class['rho'] = ds_init['rho'] * ds_init['NaNmaskC'] * xr.ones_like(ds_class['classs'])

    ds_class['mask'] = xr.where(ds_class['rho'] <= ds_class['rho_lower'],
                                ds_class["classs"],
                                False) * xr.where(ds_class['rho'] > ds_class['rho_upper'],
                                                True,
                                                False)
    
    ds_class["class_label"] = ds_class["mask"].sum("classs") \
                              * ds_init["NaNmaskC"]
    
    
    fig = plt.figure(figsize=(4, 3))

    gs = gridspec.GridSpec(1, 2,
                           width_ratios=[1, 1/16])
    
    
    n = 6
    color = cmo.dense(np.linspace(0, 1, n))
    plt.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)
    cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', ["C1", "C2", "C3", "C4", "C5"], 5)
    cmap.set_bad("grey")
    
    ax = fig.add_subplot(gs[0])
    
    
    cax = ax.contourf(ds_class['XC'] * 1e-3,
                     -ds_class['Z'],
                     ds_class["class_label"],
                     levels=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5],
                     cmap=cmap)
    ax.set_facecolor("grey")
    
    cbax = fig.add_subplot(gs[1])
    cb = plt.colorbar(cax,
                      cax=cbax,
                      label='Class',
                      orientation='vertical')
    
    cb.set_ticks([0, 1, 2, 3, 4])

    cb.formatter.set_useMathText(True)
    
    ax.set_xlabel("Longitude (km)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Water mass classes")

    
    ax.set_ylim(500, 0)
    fig.tight_layout()
    
    fig.savefig(figure_path / "WaterMassClass.pdf")