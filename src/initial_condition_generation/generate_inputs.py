import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO
                   )

logging.info('Importing standard python libraries')
from pathlib import Path
import os.path as osp

logging.info('Importing third party python libraries')
import xarray as xr
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.stats import linregress
from scipy.linalg import solve
import matplotlib.pyplot as plt
import numpy as np
import xgcm
import MITgcmutils.mds as mds
import cmocean.cm as cmo

logging.info('Importing custom python libraries')
import okapy.thermo as okt

plot = True

logging.info('Setting jet and model parameters')
# Thermodynamics
T_0 = 8
alpha_T = 2e-4
beta_S = None
S = 0
rho_0 = 1027


# Model domain
H_min = -1900
Lx = 600e3 ## Set to 600e3
Ly = 1800e3  # Large is 3000e3

dx = 1e3
dy = 1e3
dz = 4

nx = int(Lx / dx)
ny = int(Ly / dy)
nz = int(- H_min / dz)


# Jet parameters
V0L = -0.2
V0L = -0.4
xmL = 10e3
sigmaxL = 7e3
zmL = - 0
sigmazL = 750

V0R = -0.6
V0R = -0.4
xmR = 65e3
sigmaxR = 10e3
zmR = - 0
sigmazR = - H_min


# Physical parameters
g = 9.81
f = 1.26e-4
beta = None


logging.info('Setting paths')
base_path = Path('/work/n01/n01/fwg/irminger-proj')
input_pf = base_path / 'data/raw/run/input'
observations_path = base_path / 'data/raw/CFall_finergrid_2m.nc'
mean_obs_path = base_path / 'data/interim/mean_observations.nc'

precision = 'float32'

bathymetry_fn = 'bathymetry'
uvel_fn = 'UVEL'
vvel_fn = 'VVEL'
Tinit_fn = 'T_init'
Tref_fn = 'T_ref'
Sref_fn = 'S_ref'
Eta_fn = 'Eta'
nv_fn = 'VVEL_bound'
sv_fn = 'VVEL_bound'
T_bound_fn = 'T_bound'
umask_fn = 'gammaU'
vmask_fn = 'gammaV'
Tmask_fn = 'gammaT'
tauy_fn = 'merid_wind_stress'
taux_fn = 'zonal_wind_stress'

input_pf.mkdir(parents=True, exist_ok=True)

logging.info('Opening observational data')
def open_observational_data():
    # Open the original dataset
    ds_orig = xr.open_dataset(observations_path, chunks=-1)
    ds_orig = ds_orig.transpose('date', 'depth', 'distance', ...)
    ds_orig

    # Reformat the original dataset
    ds = ds_orig.copy(deep=True)

    ds['distance'] = (ds['distance'] - ds['distance'].isel(distance=0)) * 1e3
    ds['distance'].attrs = {'units': 'metres'}

    ds['depth'] = - ds['depth']
    ds['depth'].attrs = {'units': 'metres', 'convention': 'depth increases towards the surface'}

    return ds

if mean_obs_path.exists():
    ds_mean = xr.open_dataset(mean_obs_path)
else:
    ds_mean = open_observational_data().mean(dim='date').load()
    ds_mean.to_netcdf(mean_obs_path)

if plot:
    fig, ax = plt.subplots()

    cax0 = ds_mean['across track velocity'].plot(ax=ax)
    cax1 = ds_mean['potential density'].plot.contour(ax=ax, levels=np.arange(26.5, 28, 0.1), cmap=cmo.dense)

    fig.show()
    
logging.info('Creating the model grid')
def create_grids():
    drF = -dz * np.ones(nz)

    xg = np.linspace(-0.5 * dx, Lx - 0.5 * dx, nx)  # Coordinate of left, u points
    xc = xg + dx * 0.5  # Coordinate of right v, eta, rho and h points

    xg_reversed = xg[::-1]
    xc_reversed = xc[::-1]
    
    yg = np.linspace(-0.5 * dy, Ly - 0.5 * dy, ny)  # Coordinate of left, u points
    yc = yg + dy * 0.5  # Coordinate of right v, eta, rho and h points

    zu = np.cumsum(drF)  # This is the lower coordinate of the vertical cell faces, i.e. the w points
    zl = np.concatenate(([0], zu[:-1]))  # This is the upper coordinate of the vertical cell faces
    z = 0.5 * (zl + zu)  # Vertical coordiante of the velocity points


    ds_grid = xr.Dataset(coords={'XG': xg,
                                 'XC': xc,
                                 'YG': yg,
                                 'YC': yc,
                                 'Zu': zu,
                                 'Zl': zl,
                                 'Z': z})
    grid = xgcm.Grid(ds_grid,
                     periodic=['X'],
                     coords={'X': {'left': 'XG', 'center': 'XC'},
                             'Y': {'left': 'YG', 'center': 'YC'},
                             'Z': {'left': 'Zu', 'right': 'Zl', 'center': 'Z'}})
    
    ds_reversed_xgrid = xr.Dataset(coords={'XG': xg_reversed,
                                           'XC': xc_reversed,
                                           'YG': yg,
                                           'YC': yc,
                                           'Zu': zu,
                                           'Zl': zl,
                                           'Z': z})
    
    reversed_xgrid = xgcm.Grid(ds_reversed_xgrid,
                     periodic=['X'],
                     coords={'X': {'left': 'XG', 'center': 'XC'},
                             'Y': {'left': 'YG', 'center': 'YC'},
                             'Z': {'left': 'Zu', 'right': 'Zl', 'center': 'Z'}})
    
    
    ds_reversed_zgrid = xr.Dataset(coords={'XG': xg,
                                           'XC': xc,
                                           'YG': yg,
                                           'YC': yc,
                                           'Zu': zu[::-1],
                                           'Zl': zl[::-1],
                                           'Z': z[::-1]})
    
    reversed_zgrid = xgcm.Grid(ds_reversed_zgrid,
                     periodic=['X'],
                     coords={'X': {'left': 'XG', 'center': 'XC'},
                             'Y': {'left': 'YG', 'center': 'YC'},
                             'Z': {'left': 'Zu', 'right': 'Zl', 'center': 'Z'}})
    
    ds_grid['deltaX'] = ('XC', dx * np.ones(nx))
    ds_grid['deltaY'] = ('YC', dy * np.ones(ny))
    ds_grid['deltaZ'] = ('Z', dz * np.ones(nz))
    
    return ds_grid, grid, reversed_xgrid, reversed_zgrid

ds_grid, grid, reversed_xgrid, reversed_zgrid = create_grids()
ds_input = ds_grid.copy(deep=True)

logging.info('Creating the model bathymetry')
def produce_depth():
    ds_depth = xr.Dataset(coords={'distance': ds_mean['distance'],
                                  'depth': ds_mean['depth']})

    # Extract the depth data from the 2D field
    ds_depth['2d_depth'] = ds_depth['distance'] * ds_depth['depth']
    ds_depth['2d_depth'] = ds_depth['2d_depth'] + (0 * ds_mean['potential density'])
    depth_args = ds_depth['2d_depth'].argmax('depth', skipna=False)
    ds_depth['h'] = ds_depth['depth'].isel(depth=depth_args)
    ds_depth['h'].values[0] = 0  # Set the depth at the first grid cell to zero
    return ds_depth

ds_depth = produce_depth()
ds_depth

def depth_function(x, h0, lambdaa, xm, delta):
    h1 = - 0.5 * (H_min + h0)
    h_upper = h0 * (np.exp(-x * lambdaa) - 1)
    h_lower =  - h1 * (np.tanh((x - xm) / delta) + 1)
    return h_upper + h_lower

params, _ = curve_fit(depth_function, ds_depth['distance'], ds_depth['h'],
                      bounds=(0, (np.inf, 1/2e3, np.inf, np.inf)),
                      p0=(250, 1/10e3, 40e3, 30e3))

def plot_the_idealised_bathmetry():
    # Evaluate the idealised depth function
    X = np.linspace(0, 100e3, 400)
    idealised_depth = depth_function(X, *params)

    # Plot the observed depth and idealised depth together.
    fig, ax = plt.subplots()
    ds_depth['h'].plot(label='original', ax=ax)
    ax.plot(X, idealised_depth, label='idealised')
    ax.legend()
    ax.set_title("Comparison of idealised and original bathymetrys")
    fig.show()

    logging.info('h_0 = {:.2f}, lambda = {:.2e}, x_mid = {:.2f}, delta = {:.2f}'.format(*params))

if plot: plot_the_idealised_bathmetry()

# Make and plot the bathymetry masks
ds_input['bathymetry'] = xr.DataArray(depth_function(ds_input['XC'], *params))
ds_input['bathymetry'].plot(lw=3, c='m', ls='-.')

ds_input['bool_land_mask'] = xr.where(ds_input['bathymetry'] <= ds_input['Z'], 1, 0)
ds_input['nan_land_mask'] = xr.where(ds_input['bathymetry'] <= ds_input['Z'], 1, np.nan)

if plot: ds_input['bool_land_mask'].plot()

logging.info('Interpolating the observations to a smooth grid')
def calculate_bottom_indices():
    # Land masked depth
    depth_land_masked = ds_input['bool_land_mask'] * ds_input['Z']
    bottom_depth_indices = depth_land_masked.argmin('Z')    
    return bottom_depth_indices

bottom_depth_indices = calculate_bottom_indices()

east_boundary_vvel = xr.zeros_like(ds_input['Z'])
bottom_boundary_vvel = xr.zeros_like(ds_input['XC'])

east_boundary_density = ds_mean['potential density'].isel(distance=-1).interp({'depth': ds_input['Z']})
bottom_boundary_density = 'density'

def put_da_on_ideal_grid(da, east_boundary_values, bottom_boundary_values):
    if type(bottom_boundary_values) == str:
        if bottom_boundary_values == 'density':
            density = True
        else:
            raise ValueError("`bottom_boundary_values` must be 'density' or array like")
    else:
        density = False
    
    da_model_grid = da.interp({'distance': ds_input['XC'], 'depth': ds_input['Z']})
    
    # Set the array values at the bottom and eastern boundaries
    if density:
        da_model_grid[bottom_depth_indices] = xr.full_like(ds_input['XC'], np.nan)
    else:
        da_model_grid[bottom_depth_indices] = bottom_boundary_values

    da_model_grid[:, -1] = east_boundary_values
    
    if density:
        da_filled = da_model_grid[::-1].interpolate_na(dim='Z', method='linear', fill_value='extrapolate')[::-1]
        da_filled = da_filled.interpolate_na(dim='XC', method='linear')

    else:
        da_filled = da_model_grid[::-1].interpolate_na(dim='Z', method='linear')[::-1]
        da_filled = da_filled.interpolate_na(dim='XC', method='linear')
    
    # Remove the extra grid points
    da_on_ideal_grid = ds_input['nan_land_mask'] * da_filled
    da_on_ideal_grid = da_on_ideal_grid.fillna(0)
    return da_on_ideal_grid

ds_input['VVEL'] = put_da_on_ideal_grid(ds_mean['across track velocity'], east_boundary_vvel, bottom_boundary_vvel)
ds_input['sigma'] = put_da_on_ideal_grid(ds_mean['potential density'], east_boundary_density, bottom_boundary_density)

if plot:
    fig, ax = plt.subplots()
    ds_input['VVEL'].plot(ax=ax, cmap=cmo.tempo_r)
    ds_input['sigma'].plot.contour(levels=np.arange(26.5, 28, 0.1), ax=ax, cmap=cmo.dense)
    fig.show()
    
logging.info('Setting the input velocity')
def input_velocity(X, Z, mask=None):
    V0L = -0.2
    V0L = -0.4
    xmL = 10e3
    sigmaxL = 7e3
    zmL = - 0
    sigmazL = 750
    
    V0R = -0.6
    V0R = -0.4
    xmR = 65e3
    sigmaxR = 10e3
    zmR = - 0
    sigmazR = - H_min

    VL = gaussian_jet(X, Z, V0L, xmL, sigmaxL, zmL, sigmazL)
    VR = gaussian_jet(X, Z, V0R, xmR, sigmaxR, zmR, sigmazR)
    
    V = VL + VR
    if mask is not None:
        V = V * ds_input['nan_land_mask']
    return V


def gaussian_jet(X, Z, V0, Xmid, sigmaX, Zmid, sigmaZ):
    VZ = (Z - Zmid) / sigmaZ + 1
    VX = np.exp(-np.square(X - Xmid) / 2 / np.square(sigmaX))
    V = V0 * VZ * VX
    return V

ds_input['VVEL'] = input_velocity(ds_input['XC'], ds_input['Z'], mask=ds_input['nan_land_mask'])
if plot: ds_input['VVEL'].plot(cmap=cmo.balance, vmin=-0.6, vmax=0.6)
ds_input['UVEL'] = xr.zeros_like(ds_input['Z'] * ds_input['XG'])


logging.info('Creating a reference density profile')
def create_idealised_reference_density(observed_reference_density):
    # Set the depths at which to switch profiles
    z_therm = -200 + 1e-7 # Depth at which the linaar surface profile ends
    z_bound = -500 + 1e-7 # Depth at which the linear deep profile starts

    surf_rho = observed_reference_density.sel(Z=slice(0, z_therm))
    mid_rho = observed_reference_density.sel(Z=slice(z_therm, z_bound))
    deep_rho = observed_reference_density.sel(Z=slice(z_bound, H_min))

    # Use observations to linearly fit the deep profile
    k2, c2 = deep_rho.polyfit('Z', 1)['polyfit_coefficients']
    interp_deep_rho = k2 * deep_rho['Z'] + c2

    # Use observations to linearly fit the surface profile
    k1, c1 = surf_rho.polyfit('Z', 1)['polyfit_coefficients']
    interp_surf_rho = k1 * surf_rho['Z'] + c1

    # Fit a second order polynomial to the middle density profile
    # Match the density and its derivative at the boundary layer
    # Match the the density, but not its derivative at the the thermocline
    rho_therm = k1 * z_therm + c1
    rho_bound = k2 * z_bound + c2
    A = np.array([[z_therm ** 2 , z_therm, 1],[z_bound ** 2, z_bound, 1],[2 * z_bound, 1, 0]])
    bprime = np.array([rho_therm, rho_bound, k2])
    a, b, c = solve(A, bprime)  # rho = a * z ** 2 + b * z + c
    interp_mid_rho = a * mid_rho['Z'] * mid_rho['Z']  + b * mid_rho['Z'] + c
    
    ds_input['rho_ref'] = xr.concat([interp_surf_rho, interp_mid_rho, interp_deep_rho], dim='Z')

    rho_0 = ds_input['rho_ref'][0].values

    if plot:
        plt.plot(ds_input['rho_ref'], ds_input['Z'], c='k', ls='-', label='idealised neutral density')
        plt.plot(observed_reference_density, observed_reference_density['Z'], c='k', ls='-.', label='observed neutral density')


        plt.xlabel('Density  (kg m$^{-3}$)')
        plt.ylabel('Depth (m)')
        plt.grid()
        plt.legend()
        plt.show()

    logging.info('rho_0 = {}'.format(rho_0))
    
create_idealised_reference_density(ds_input['sigma'].isel(XC=-1) + 1000)

logging.info('Using thermal wind balance to calculate the density field')

def create_idealized_density(ds_input, rho_ref):
    V = input_velocity(ds_input['XG'], ds_input['Zl'])
    dv_dz = grid.diff(V, 'Z', boundary='extrapolate')
    drho_dx = - f * rho_0 / g * dv_dz

    rho = rho_ref + reversed_xgrid.cumsum(drho_dx[:, ::-1], 'X') * dx * 0.5
    rho = rho * ds_input['nan_land_mask']
    return rho

ds_input['rho'] = create_idealized_density(ds_input, ds_input['rho_ref'])
(ds_input['rho'] - 1000).plot(cmap=cmo.dense, levels=np.arange(26.5, 28, 0.1))

if xr.where(ds_input['rho'].diff('Z') < 0, 1, 0).sum() != 0:
    logging.info('stratification is unstable')
    print(xr.where(ds_input['rho'].diff('Z') < 0, 1, 0).sum().values)
else:
    logging.info('stratification is stable')


logging.info('Recalculating velocity field based on thermal wind balance')
db_dx = g / rho_0 * grid.diff(ds_input['rho'], 'X') / dx
db_dx_interp = grid.interp(db_dx, ['X', 'Z'], boundary='fill')

V = reversed_zgrid.cumsum(db_dx_interp[::-1], 'Z') * -dz / f
if plot:
    plt.figure()
    V.plot(vmin=-0.6, vmax=0.6, cmap=cmo.balance)

try:
    ds_input['VVEL_orig']
except:
    ds_input['VVEL_orig'] = ds_input['VVEL']

ds_input['VVEL'] = V


logging.info('Converting density into temperature')
ds_input['rho_ref'] = ds_input['rho'].isel(XC=-1).fillna(0)

ds_input['T_ref'] = okt.T_from_rho(ds_input['rho_ref'], rho_0, alpha_T) + T_0
ds_input['T_init'] = (okt.T_from_rho(ds_input['rho'], rho_0, alpha_T) + T_0)
ds_input['T_init'] = ds_input['T_init'].fillna(1)

ds_input['T_init'].plot.contourf()

ds_input['S_ref'] = xr.zeros_like(ds_input['Z'])
ds_input['S_init'] = ds_input['S_ref'].broadcast_like(ds_input['XC'])


logging.info('Setting the meridional wind stress')
tau_y = np.exp(-(ds_input['YC'] - (Ly - 950e3)) ** 2/ (75e3) ** 2 / 2)
tau_y = tau_y / tau_y.max() * -0.6

xstar = 300e3
wwidth = 25e3
amplitude = (-np.tanh((ds_input['XG'] - xstar) / wwidth) + 1) * 0.5
tau_y = (tau_y * amplitude)
ds_input['merid_wind_stress'] = tau_y


logging.info('Setting the zonal wind stress')
zonal_wind = 'zero'

if zonal_wind == 'zero':
    tau_x = xr.zeros_like(ds_input['UVEL'])

elif zonal_wind == 'divergence_free':
    dtaux_dx = -grid.diff(tau_y, 'Y', boundary='fill') / dy
    tau_x = grid.cumsum(dtaux_dx, 'X') * dx

    skip = 1
    fig, ax = plt.subplots(figsize=(15, 15))
    #plt.streamplot(ds_input['XC'].values, ds_input['YC'].values, tau_x[::skip, ::skip], tau_y[::skip, ::skip], )
    #plt.gca().set_aspect('equal')

elif zonal_wind == 'irrotational':
    ## Irrotational winds

    dtaux_dy = grid.diff(ds_input['merid_wind_stress'], 'X') / dx

    tau_x = grid.cumsum(dtaux_dy, 'Y', boundary='fill') * dy 
    tau_x = tau_x - tau_x.max() * 0.5
    #tau_x.plot(robust=True)


ds_input['zonal_wind_stress'] = tau_x

if plot:
    fig, axs = plt.subplots(1, 2)
    tau_y.plot(ax=axs[0])
    tau_x.plot(ax=axs[1])
    fig.tight_layout()
    fig.show()

    
logging.info('Setting up sponges')
def sponge_gamma():
    L_Nsponge = 300
    gammamax= 2e-5
    delta_Nsponge = 50
    mid_Nsponge = L_Nsponge / 2
    Nsponge = np.arange(0, L_Nsponge + 1)
    gamma_Nsponge = (np.tanh((Nsponge - mid_Nsponge) / delta_Nsponge) - np.tanh(- mid_Nsponge / delta_Nsponge)) * gammamax / np.tanh(mid_Nsponge / delta_Nsponge) / 2

    L_Ssponge = 300
    gammamax= 2e-5
    delta_Ssponge = 50
    mid_Ssponge = L_Ssponge / 2
    Ssponge = np.arange(0, L_Ssponge + 1)
    gamma_Ssponge = (np.tanh((Ssponge - mid_Ssponge) / delta_Ssponge) - np.tanh(- mid_Ssponge / delta_Ssponge)) * gammamax / np.tanh(mid_Ssponge / delta_Ssponge) / 2
    gamma_Ssponge = gamma_Ssponge[::-1]

    L_interior = int(Ly / dy - L_Ssponge - L_Nsponge - 2)
    gamma_interior = np.zeros(L_interior)
    gamma_whole = np.concatenate((gamma_Ssponge, gamma_interior, gamma_Nsponge))
    return gamma_whole

gamma_whole = sponge_gamma()

if plot:
    plt.figure()
    plt.plot(gamma_whole)
    plt.xlabel('Y grid point')
    plt.ylabel('Inverse relaxation time scale (Hz)')

ds_input['gammaV'] = xr.DataArray(gamma_whole, dims=('YG')).broadcast_like(ds_input['VVEL'])
ds_input['gammaU'] = xr.DataArray(gamma_whole, dims=('YC')).broadcast_like(ds_input['UVEL'])
ds_input['gammaT'] = xr.DataArray(gamma_whole, dims=('YC')).broadcast_like(ds_input['VVEL'])
ds_input['gammaS'] = xr.DataArray(gamma_whole, dims=('YC')).broadcast_like(ds_input['VVEL'])

logging.info('replacing nans with zeros')
for field in ds_input.keys():
    logging.info(field)
    if not field.startswith('gamma'):
        ds_input[field] = ds_input[field].fillna(0)
    

logging.info('Broadcasting to 3D arrays')
def broadcast_to_3d_vars(ds_input):
    ds_3d = xr.Dataset()
    
    ds_3d['VVEL'] = ds_input['VVEL'].broadcast_like(ds_input['YG'])
    ds_3d['UVEL'] = ds_input['UVEL'].broadcast_like(ds_input['YC'])
    ds_3d['VVEL_bound'] = ds_input['VVEL']
    
    ds_3d['bathymetry'] = ds_input['bathymetry'].broadcast_like(ds_input['YC'])
    ds_3d['deltaZ'] = ds_input['deltaZ']
    
    ds_3d['T_bound'] = ds_input['T_init']
    ds_3d['T_init'] = ds_input['T_init'].broadcast_like(ds_input['YC'])
    ds_3d['T_ref'] = ds_input['T_ref']
 
    ds_3d['S_init'] = ds_input['S_init'].broadcast_like(ds_input['YC'])
    ds_3d['S_ref'] = ds_input['S_ref']
    
    ds_3d['gammaU'] = ds_input['gammaU']
    ds_3d['gammaV'] = ds_input['gammaV']
    ds_3d['gammaT'] = ds_input['gammaT']
    ds_3d['gammaS'] = ds_input['gammaS']
    
    ds_3d['merid_wind_stress'] = ds_input['merid_wind_stress']
    ds_3d['zonal_wind_stress'] = ds_input['zonal_wind_stress']
    ds_3d = ds_3d.transpose('Z', 'Zl', 'Zu', 'Zp1', 'YG', 'YC', 'XG', 'XC', missing_dims='ignore')
    return ds_3d

ds_3d = broadcast_to_3d_vars(ds_input)
ds_3d

logging.info('Saving input files')

for field in ds_3d.keys():
    file_path = input_pf / field
    logging.info(file_path)
    mds.wrmds(str(file_path), ds_3d[field].values, dataprec=precision)