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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib import colors, rc
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cmocean.cm as cmo


figure1 = False
figure2 = False
thesiscover = False
figure3 = False
figure4 = True


logging.info('Setting paths')
base_path = Path('/work/n01/n01/fwg/irminger-proj')
raw_path = base_path / 'data/raw'
processed_path = base_path / 'data/processed'
figure_path = base_path / 'figures'


logging.info('Setting plotting defaults')
SMALL_SIZE = 8
MEDIUM_SIZE = 8
BIGGER_SIZE = 8
rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

cm = 1/2.54
dpi = 300


text_width = 5.5  # in inches

if figure1:
    logging.info('Plotting initial and boundary conditions')

    # Open the datasats
    ds_init = xr.open_dataset(processed_path / 'init.nc')
    ds_bathymetry = xr.open_dataset(raw_path / 'GEBCO-bathymetry-data/gebco_2021_n30.0_s-30.0_w-85.0_e-10.0.nc')
    ds_bathymetry = ds_bathymetry.coarsen(lon=5, lat=5, boundary='trim').mean()
    ds_climatological_gamma_n = xr.open_dataset(processed_path / 'climatological_gamman.nc', decode_times=False)

    # Set up the canvas
    fig = plt.figure(figsize=(text_width, 8.5 * cm))

    gs = gridspec.GridSpec(2, 2,
                           width_ratios=[1, 1],
                           height_ratios=[1, 1/16]
                           )

    ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())

    # Plot the bathymetry
    cax_bathy = ax1.pcolormesh(ds_bathymetry['lon'],
                               ds_bathymetry['lat'],
                               -ds_bathymetry['elevation'],
                               shading='nearest',
                               rasterized=True,
                               cmap=cmo.deep,
                               vmin=0
                              )

    # Add some land
    ax1.add_feature(cfeature.NaturalEarthFeature('physical',
                                                 'land',
                                                 '110m',
                                                 edgecolor='face',
                                                 facecolor='grey'
                                                ))
    
    y0 = ds_climatological_gamma_n['lat'].min()
    ywid = ds_climatological_gamma_n['lat'].max() - y0
    x0 = ds_climatological_gamma_n['lon'].min()
    xwid = ds_climatological_gamma_n['lon'].max() - x0
    ax1.add_patch(Rectangle((x0, y0), xwid, ywid, ec='red', fc='none'))

    # Axes limits, labels and features
    ax1.axhline(0, c='k', ls='--')

    ax1.set_ylim(-12, 30)
    ax1.set_xlim(-85, -25)

    ax1.set_xticks(np.arange(-85, -24, 10), crs=ccrs.PlateCarree())
    ax1.set_yticks(np.arange(-10, 31, 10), crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    ax1.xaxis.set_major_formatter(lon_formatter)
    ax1.yaxis.set_major_formatter(lat_formatter)

    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title('The Tropical Atlantic')
    ax1.set_title('(a)', loc='left')

    # Colorbars
    cbax1 = fig.add_subplot(gs[1, 0])
    cb1 = plt.colorbar(cax_bathy, cax=cbax1, label='Depth (m)', orientation='horizontal')

    # Initial condition plots
    ax2 = fig.add_subplot(gs[0, 1])
    cbax = fig.add_subplot(gs[1, 1])

    ax2.set_title('Initial conditions')
    ax2.set_title('(b)', loc='left')
    ax2.set_xlabel('Longitude (km)')
    ax2.set_ylabel('Depth (m)')

    cmo.tempo_r.set_bad('grey')
    cax = ax2.pcolormesh(ds_init['XC'] * 1e-3,
                         -ds_init['Z'],
                         ds_init['V_init'] * 1e2,
                         vmin=-20,
                         vmax=0,
                         shading='nearest',
                         cmap=cmo.tempo_r,
                         rasterized=True
                        )

    ax2.invert_yaxis()

    cb = plt.colorbar(cax, cax=cbax, label='Meridional velocity (cm$\,$s$^{-1}$)',
                      orientation='horizontal')

    cb.formatter.set_useMathText(True)

    axins = ax2.twiny()

    ln_id, = axins.plot(ds_init['rho_init'] - 1000, -ds_init['Z'], c='k', label='idealised')
    ln_clim, = axins.plot(ds_climatological_gamma_n['mn_gamma_n'],
                          -ds_climatological_gamma_n['depth'],
                          label='climatalogical',
                          c='k',
                          ls='--'
                         )

    axins.set_xlabel('$\\sigma$ (kg$\,$m$^{-3}$)', labelpad=3, loc='center')
    axins.set_xlim(20,29)
    axins.set_xticks(range(22, 29))

    ax2.legend([ln_id, ln_clim], ['Idealised', 'Climatological'], loc='lower center')
    ax2.set_ylim(-ds_init['Z'][-1] ,0)

    fig.tight_layout()
    fig.savefig(figure_path / 'Figure1.pdf', dpi=dpi)

    
if figure2:
    logging.info('Plotting stratification slices')

    da_dbdz = xr.open_dataarray(processed_path / 'dbdz_slice.nc')
    X, Z = da_dbdz['XC'] * 1e-3, -da_dbdz['Zl']
    
    fig = plt.figure(figsize=(text_width, 8.5 * cm))

    gs = gridspec.GridSpec(2, 3,
                           width_ratios=[1, 1, 1],
                           height_ratios=[15, 1]
                           )

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharey=ax1)
    ax3 = fig.add_subplot(gs[2], sharey=ax1)

    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.setp(ax3.get_yticklabels(), visible=False)

    cbax = plt.subplot(gs[3:])

    ax1.set_title('600 km North')
    ax2.set_title('800 km North')
    ax3.set_title('1,000 km North')

    ax1.set_title('(a)', loc='left')
    ax2.set_title('(b)', loc='left')
    ax3.set_title('(c)', loc='left')

    ax1.set_ylabel('Depth (m)')
    ax1.set_xlabel('Longitude (km)')
    ax2.set_xlabel('Longitude (km)')
    ax3.set_xlabel('Longitude (km)')

    cmo.matter.set_bad('grey')
    cax = ax1.pcolormesh(X, Z, da_dbdz.isel(YC=0), shading='nearest', cmap=cmo.matter, vmin=0, vmax=7.5e-6, rasterized=True)
    ax2.pcolormesh(X, Z, da_dbdz.isel(YC=1), shading='nearest', cmap=cmo.matter, vmin=0, vmax=7.5e-6, rasterized=True)
    ax3.pcolormesh(X, Z, da_dbdz.isel(YC=2), shading='nearest', cmap=cmo.matter, vmin=0, vmax=7.5e-6, rasterized=True)

    ax1.invert_yaxis()

    #ax1.set_xlim(0, 300)
    #ax2.set_xlim(0, 300)
    #ax3.set_xlim(0, 300)

    #ax2.axvline(90, c='magenta')

    cb = plt.colorbar(cax, cax=cbax, orientation='horizontal', label='$\partial_z$b (m$\,$s$^{-2})$')
    cb.formatter.set_useMathText(True)
    
    #yticks = [0, 1000, 2000, 3000, 4000]
    #ax1.set_yticks(yticks)

    fig.tight_layout()

    fig.savefig(figure_path / 'Figure2.pdf', dpi=dpi)


if figure4:
    logging.info('Plotting potential vorticity')
    
    cmo.curl.set_bad('grey')
    Qcmap = cmo.curl
    Qlim = 1e-9

    da_Q_on_rho = xr.open_dataarray(processed_path / 'Q_on_rho.nc')
    X_bigQ, Y_bigQ = da_Q_on_rho['XC'] * 1e-3, da_Q_on_rho['YC'] * 1e-3
    C_bigQ = da_Q_on_rho.values.squeeze()
    
    da_Q_slice = xr.open_dataarray(processed_path / 'Q_slice.nc')
    X, Z = da_Q_slice['XC'] * 1e-3, -da_Q_slice['Z']
    
    fig = plt.figure(figsize=(text_width, 23 * cm))

    gs = gridspec.GridSpec(4, 2,
                           width_ratios=[1, 1],
                           height_ratios=[1, 1, 1, 0.1]
                           )

    big_Q_ax = fig.add_subplot(gs[:3, 0])

    slice_ax3 = fig.add_subplot(gs[2, 1])
    slice_ax1 = fig.add_subplot(gs[0, 1], sharex=slice_ax3)
    slice_ax2 = fig.add_subplot(gs[1, 1], sharex=slice_ax3)

    plt.setp(slice_ax2.get_xticklabels(), visible=False)
    plt.setp(slice_ax1.get_xticklabels(), visible=False)

    slice_cbax = fig.add_subplot(gs[3, :])
    #big_cbax = fig.add_subplot(gs[0:, 0])

    big_Q_ax.set_title('$\sigma = what$')
    slice_ax1.set_title('600 km North')
    slice_ax2.set_title('800 km North')
    slice_ax3.set_title('1,000 km North')

    big_Q_ax.set_title('(a)', loc='left')
    slice_ax1.set_title('(b)', loc='left')
    slice_ax2.set_title('(c)', loc='left')
    slice_ax3.set_title('(d)', loc='left')


    big_Q_ax.set_xlabel('Longitude (km)')
    slice_ax3.set_xlabel('Longitude (km)')

    big_Q_ax.set_ylabel('Latitude (km)')
    slice_ax1.set_ylabel('Depth (m)')
    slice_ax2.set_ylabel('Depth (m)')
    slice_ax3.set_ylabel('Depth (m)')
    
    #yticks = [0, 1000, 2000, 3000, 4000]
    #slice_ax1.set_yticks(yticks)
    #slice_ax2.set_yticks(yticks)
    #slice_ax3.set_yticks(yticks)

    big_Q_cax = big_Q_ax.pcolormesh(X_bigQ, Y_bigQ, C_bigQ, cmap=Qcmap, shading='nearest',
                                    vmin=-Qlim, vmax=Qlim, rasterized=True)

    slice_ax1.pcolormesh(X, Z, da_Q_slice.isel(YC=0), cmap=Qcmap, shading='nearest', vmin=-Qlim, vmax=Qlim, rasterized=True)
    slice_ax2.pcolormesh(X, Z, da_Q_slice.isel(YC=1), cmap=Qcmap, shading='nearest', vmin=-Qlim, vmax=Qlim, rasterized=True)
    slice_ax3.pcolormesh(X, Z, da_Q_slice.isel(YC=2), cmap=Qcmap, shading='nearest', vmin=-Qlim, vmax=Qlim, rasterized=True)


    big_Q_ax.axhline(600, label='600 km North', ls='-', c='k')
    big_Q_ax.axhline(800, label='800 km North', ls=':', c='k')
    big_Q_ax.axhline(1000, label='1,000 North', ls='-.', c='k')
    #big_Q_ax.scatter(90, -250, label='Profile point', marker='o', c='magenta', linewidths=2)

    #slice_ax2.axvline(90, c='magenta')

    #slice_ax1.set_xlim(0, 300)
    #slice_ax2.set_xlim(0, 300)
    #slice_ax3.set_xlim(0, 300)

    big_Q_ax.set_aspect('equal')
    #big_Q_ax.set_ylim(-1800, 500)

    slice_ax1.invert_yaxis()
    slice_ax2.invert_yaxis()
    slice_ax3.invert_yaxis()

    slice_cb = plt.colorbar(big_Q_cax, cax=slice_cbax, orientation='horizontal', label='Q (s$^{-3}$)')
    slice_cb.formatter.set_useMathText(True)

    big_Q_ax.legend(loc='upper right')

    fig.tight_layout()

    fig.savefig(figure_path / 'Figure4.pdf', dpi=dpi)

    
if figure4 and thesiscover:
    logging.info('Plotting thesis cover image')
    
    fig, ax = plt.subplots(frameon=True, figsize=(12, 12))

    ax.pcolormesh(X_bigQ, Y_bigQ, C_bigQ, cmap=Qcmap, shading='nearest',vmin=-Qlim, vmax=Qlim, rasterized=True)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(50, 600)
    fig.tight_layout()
    fig.savefig(figure_path / 'ThesisCover.png', dpi=600)
    

if figure3:
    logging.info('Plotting overturning mechanisms')
    
    ds_overturning = xr.open_dataset(processed_path / 'overturning.nc')
    #rho_3, zetay_3 = xr.open_dataarray('rho3.nc'), xr.open_dataarray('zeta_y.nc')
    da_zeta_y_slice = xr.open_dataarray(processed_path / 'zeta_y_slice.nc')
    
    fig = plt.figure(figsize=(text_width, 8.5 * cm))

    gs = gridspec.GridSpec(2, 2,
                           width_ratios=[1, 1],
                           height_ratios=[1, 1/16]
                           )

    ax1 = fig.add_subplot(gs[:, 1])
    ax_overturn = fig.add_subplot(gs[0, 0])
    cbax = fig.add_subplot(gs[1, 0])
    ax2 = ax1.twiny()

    # Right hand panel with staircase and zeta_y
    ax1.plot(ds_overturning['rho'] - 1000, -ds_overturning['Z'], ls='-', c='k', label='$\\sigma(z)$')
    ax2.plot(ds_overturning['zeta_y'], -ds_overturning['Zl'], ls='-', c='grey', label='$\\zeta_y(z)$')
    ax2.axvline(0, ls=':', c='grey')

    ax1.invert_yaxis()
    #ax1.set_xlim(1027.9 - 1000, 1028.15 - 1000)
    #ax1.set_ylim((ds_overturning['Depth'] - 16), 1500)

    ax1.set_ylabel('Depth (m)')
    ax1.set_xlabel('$\\sigma$ (kg$\,$m$^{-3}$)')
    ax2.set_xlabel('$\\zeta_y$ (s$^{-1}$)', labelpad=10)
    ax2.ticklabel_format(axis='x', style='sci', scilimits=(0, 0), useMathText=True)
    ax2.set_title('(b)', loc='left')
    ax2.set_title('Staircases & overturning')

    # Left hand panel with zeta_y
    cmo.balance.set_bad('grey')
    zylim = 2.5e-4
    cax = ax_overturn.pcolormesh(da_zeta_y_slice['XG'] * 1e-3, -da_zeta_y_slice['Zl'], da_zeta_y_slice,
                           shading='nearest', cmap=cmo.balance, vmin=-zylim, vmax=zylim,
                           rasterized=True
                          )

    ax_overturn.invert_yaxis()
    ax_overturn.axvline(90, c='magenta')
    ax_overturn.set_xlim(0, 300)

    ax_overturn.set_xlabel('Longitude (km)')
    ax_overturn.set_ylabel('Depth (m)')
    ax_overturn.set_title('Meridional relative vorticity')
    ax_overturn.set_title('(a)', loc='left')

    # Colorbar
    cb = plt.colorbar(cax, cax=cbax, label='$\\zeta_y$ (s$^{-1}$)',
                      orientation='horizontal')
    cb.formatter.set_useMathText(True)
    cb.formatter.set_powerlimits((0, 0))

    # Figure stuff
    fig.legend(loc='upper right', bbox_to_anchor=(0.95, 0.79))
    #fig.suptitle('Staircases & overturning', weight='bold', y=0.97)

    fig.tight_layout()

    fig.savefig(figure_path / 'Figure3.pdf', dpi=dpi)
    
logging.info('Plotting complete')
