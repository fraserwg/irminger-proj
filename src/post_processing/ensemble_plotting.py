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
import matplotlib.font_manager as fm

logging.info('Setting paths')
base_path = Path('/work/n01/n01/fwg/irminger-proj')
raw_path = base_path / 'data/raw'
interim_path = base_path / 'data/interim'
processed_path = base_path / 'data/processed'
figure_path = base_path / 'figures'

logging.info('Setting plotting defaults')
# fonts
fpath = Path('/home/n01/n01/fwg/.local/share/fonts/PTSans-Regular.ttf')
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
days = 24 * 60 * 60

logging.info("Opening ensemble")
ensemble_path = interim_path / "ensemble.zarr"
mld_path = processed_path / "mld.zarr"
wmt_path = processed_path / "wmt.zarr"
assert mld_path.exists()


delta_mld = True

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

delta_mld = False
if delta_mld:
    ds_ensemble = xr.open_zarr(ensemble_path)
    ds_mld = xr.open_zarr(mld_path).squeeze()
    delta_mld = ds_mld["mn_drop_depth"].isel(time=0) - ds_mld["mn_drop_depth"].isel(time=-1)
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
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
        
    ax.set_xlabel("Wind stress (N$\\,$m$^{-2}$)")
    ax.set_ylabel("$\Delta$ MLD (m)")
    ax.set_title("Change in mixed layer depth")
    ax.legend(title="Wind duration (days)")
    fig.savefig(figure_path / "DeltaMLD.pdf")
    
vol_anom = True
if vol_anom:
    ds_ensemble = xr.open_zarr(ensemble_path)
    ds_wmt = xr.open_zarr(wmt_path).squeeze()
    fig, axs = plt.subplots(3, 2, figsize=(6, 7), sharex=True, sharey=True)
    axs = axs.flatten()
    
    for classs in ds_wmt["classs"]:
        for run in ds_wmt["run"]:
            marker, colour = select_marker(ds_ensemble["wind_duration"].sel(run=run))
            
            axs[classs].scatter(-ds_ensemble["wind_stress"].sel(run=run) * 1e-3,
                                ds_wmt["volANOM"].isel(time=-1).sel(run=run,
                                                                    classs=classs),
                                marker=marker,
                                color=colour)

    for i in range(5):
        axs[i].set_title(f"Class {i}")
        axs[i].ticklabel_format(axis="y", style="sci", scilimits=(6, 6), useMathText=True)
    
    ax=axs[5]    
    ax.scatter(None, None, marker="D", color="black", label=0)
    ax.scatter(None, None, marker="x", color="blue", label=1.08e5 / days)
    ax.scatter(None, None, marker="o", color="orange", label=2.16e5 / days)
    ax.scatter(None, None, marker="^", color="green", label=3.24e5 / days)
    ax.scatter(None, None, marker="s", color="red", label=4.32e5 / days)
    ax.axis("off")
    ax.legend(title="Wind duration (days)", loc="center")

    axs[2].set_ylabel("$\\Delta$V (m$^{3}\,$km$^{-1}$)")
    axs[4].set_xlabel("Wind stress (N$\\,$m$^{-2}$)")

    fig.suptitle("Volume anomaly")
    fig.tight_layout()
    
    fig.savefig(figure_path / "EnsembleVolAnom.pdf")
    
vol_tend = True
if vol_tend:
    ds_ensemble = xr.open_zarr(ensemble_path)
    ds_wmt = xr.open_zarr(wmt_path).squeeze()
    fig, axs = plt.subplots(3, 2, figsize=(6, 7), sharex=True, sharey=True)
    axs = axs.flatten()
    
    # We convert to Sv / km here too
    da_voltend_mn = ds_wmt["volTEND"].mean("time").compute() * 1e-6 * 1e3
    da_voltend_max = ds_wmt["volTEND"].max("time").compute() * 1e-6 * 1e3 - da_voltend_mn
    da_voltend_min = da_voltend_mn - ds_wmt["volTEND"].min("time").compute() * 1e-6 * 1e3
    
    for classs in ds_wmt["classs"]:
        for run in ds_wmt["run"]:
            marker, colour = select_marker(ds_ensemble["wind_duration"].sel(run=run))
            
            lower_err = da_voltend_min.sel(run=run, classs=classs).values
            upper_err = da_voltend_max.sel(run=run, classs=classs).values
            
            axs[classs].errorbar(-ds_ensemble["wind_stress"].sel(run=run),
                                da_voltend_mn.sel(run=run,
                                                  classs=classs),
                                yerr=np.row_stack([lower_err, upper_err]),
                                capsize=6,
                                marker=marker,
                                color=colour)


    for i in range(5):
        axs[i].set_title(f"Class {i}")
        axs[i].set_yscale("symlog", linthresh=0.0015)
    
    ax=axs[5]    
    ax.scatter(None, None, marker="D", color="black", label=0)
    ax.scatter(None, None, marker="x", color="blue", label=1.08e5 / days)
    ax.scatter(None, None, marker="o", color="orange", label=2.16e5 / days)
    ax.scatter(None, None, marker="^", color="green", label=3.24e5 / days)
    ax.scatter(None, None, marker="s", color="red", label=4.32e5 / days)
    ax.axis("off")
    ax.legend(title="Wind duration (days)", loc="center")

    axs[2].set_ylabel("Peak $V_{tend}$ (Sv$\\,$km$^{-1}$)")
    axs[4].set_xlabel("Wind stress (N$\\,$m$^{-2}$)")

    fig.suptitle("Peak Volume tendencies")
    fig.tight_layout()
    
    fig.savefig(figure_path / "EnsembelPeakFormation.pdf")