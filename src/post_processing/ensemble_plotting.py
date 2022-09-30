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
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import cmocean.cm as cmo
import cycler

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
matplotlib.use("pgf")
plt.rc('xtick', labelsize='8')
plt.rc('ytick', labelsize='8')
plt.rc('text', usetex=False)
plt.rcParams['axes.titlesize'] = 10
plt.rcParams["text.latex.preamble"] = "\\usepackage{euler} \\usepackage{paratype}  \\usepackage{mathfont} \\mathfont[digits]{PT Sans}"
plt.rcParams["pgf.preamble"] = plt.rcParams["text.latex.preamble"]
plt.rc('text', usetex=False)


n = 6
color = cmo.dense(np.linspace(0, 1,n))
plt.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)

# output
dpi = 600
cm = 1/2.54
days = 24 * 60 * 60

logging.info("Opening ensemble")
ensemble_path = interim_path / "ensemble.zarr"
mld_path = processed_path / "mld.zarr"
wmt_path = processed_path / "enswmt.zarr"
assert mld_path.exists()


def select_marker(wind_duration):
    if np.allclose(wind_duration, 1.08e5):
        marker, colour = "*", "tab:blue"
    elif np.allclose(wind_duration, 2.16e5):
        marker, colour = "o", "tab:orange"
    elif np.allclose(wind_duration, 3.24e5):
        marker, colour = "^", "tab:green"
    elif np.allclose(wind_duration, 4.32e5):
        marker, colour = "s", "tab:red"
    elif np.allclose(wind_duration, 0):
        marker, colour = "D", "black"
    else:
        raise ValueError("wind_duration not as expected")
    return marker, colour

delta_mld = True
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
    ax.scatter(None, None, marker="*", color="blue", label=1.08e5 / days)
    ax.scatter(None, None, marker="o", color="orange", label=2.16e5 / days)
    ax.scatter(None, None, marker="^", color="green", label=3.24e5 / days)
    ax.scatter(None, None, marker="s", color="red", label=4.32e5 / days)
        
    ax.set_xlabel("Wind stress (N$\\,$m$^{-2}$)")
    ax.set_ylabel("$\Delta$ MLD (m)")
    ax.set_title("Change in mixed layer depth")
    ax.legend(title="Wind event duration (days)")
    fig.savefig(figure_path / "DeltaMLD.pdf")
    
# vol_anom = False
# if vol_anom:
#     ds_ensemble = xr.open_zarr(ensemble_path)
#     ds_wmt = xr.open_zarr(wmt_path).squeeze()
#     ds_wmt = ds_wmt.assign_coords({"wind_stress": ds_ensemble["wind_stress"],
#                                    "wind_duration": ds_ensemble["wind_duration"]})
    
#     fig, axs = plt.subplots(3, 2, figsize=(6, 7), sharex=True, sharey=True)
#     axs = axs.flatten()
    
#     for classs in ds_wmt["classs"]:
#         for wind_duration in set(ds_wmt["wind_duration"].values):
#             marker, colour = select_marker(wind_duration)

#             vol_anom = ds_wmt["volANOM"].isel(time=-1).sel(classs=classs).where(ds_wmt["wind_duration"] == wind_duration)
#             vol_anom = np.array([vol_anom.values for vol_anom in vol_anom if not np.isnan(vol_anom)])

#             wind_stress = ds_wmt["wind_stress"].where(ds_wmt["wind_duration"] == wind_duration)
#             wind_stress = np.array([wind_stress.values for wind_stress in wind_stress if not np.isnan(wind_stress)])

#             axs[classs].plot(-wind_stress,
#                             vol_anom * 1e3,
#                             marker=marker,
#                             color=colour)

#     for i in range(5):
#         axs[i].set_title(f"Class {i}")
#         axs[i].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    
#     ax=axs[5]    
#     ax.scatter(None, None, marker="D", color="black", label=0)
#     ax.scatter(None, None, marker="*", color="tab:blue", label=1.08e5 / days)
#     ax.scatter(None, None, marker="o", color="tab:orange", label=2.16e5 / days)
#     ax.scatter(None, None, marker="^", color="tab:green", label=3.24e5 / days)
#     ax.scatter(None, None, marker="s", color="tab:red", label=4.32e5 / days)
#     ax.axis("off")
#     ax.legend(title="Wind event duration (days)", loc="center")

#     axs[2].set_ylabel("$\\Delta \\mathcal{V}$ (m$^{3}\,$km$^{-1}$)")
#     axs[4].set_xlabel("Wind stress (N$\\,$m$^{-2}$)")

#     fig.suptitle("Volume anomaly")
#     fig.tight_layout()
    
#     out_path = figure_path / "EnsembleVolAnom.pdf"
#     logging.info(f"Saving to {out_path}")
#     fig.savefig(out_path)
    
    
# vol_tend = False
# if vol_tend:
#     ds_ensemble = xr.open_zarr(ensemble_path)
#     ds_wmt = xr.open_zarr(wmt_path).squeeze()
#     ds_wmt = ds_wmt.assign_coords({"wind_stress": ds_ensemble["wind_stress"],
#                                    "wind_duration": ds_ensemble["wind_duration"]})

#     ds_wmt['volTEND2'] = ds_wmt["volANOM"].diff("time") / ds_wmt["time"].astype("float32").diff("time") / 1e-9 
    
#     # We convert to Sv / km here too
#     da_voltend_mn = ds_wmt["volTEND2"].mean("time").compute() * 1e-6 * 1e3
#     da_voltend_max = ds_wmt["volTEND2"].max("time").compute() * 1e-6 * 1e3
#     da_voltend_min = ds_wmt["volTEND2"].min("time").compute() * 1e-6 * 1e3
    
#     fig, axs = plt.subplots(3, 2, figsize=(6, 7), sharex=True, sharey=True)
#     axs = axs.flatten()

#     for classs in ds_wmt["classs"]:
#         for wind_duration in set(da_voltend_mn["wind_duration"].values):
#             marker, colour = select_marker(wind_duration)

            
#             upper = da_voltend_max.sel(classs=classs).where(da_voltend_max["wind_duration"] == wind_duration)
#             upper = np.array([upper.values for upper in upper if not np.isnan(upper)])

#             lower = da_voltend_min.sel(classs=classs).where(da_voltend_min["wind_duration"] == wind_duration)
#             lower = np.array([lower.values for lower in lower if not np.isnan(lower)])

#             wind_stress = da_voltend_mn["wind_stress"].where(da_voltend_min["wind_duration"] == wind_duration)
#             wind_stress = np.array([wind_stress.values for wind_stress in wind_stress if not np.isnan(wind_stress)])

#             # axs[classs].fill_between(wind_stress,
#             #                          lower, upper, alpha=0.2, color=colour)
            
#             axs[classs].plot(-wind_stress, lower, color=colour, marker=marker)
#             axs[classs].plot(-wind_stress, upper, color=colour, marker=marker)

#     for i in range(5):
#         axs[i].set_title(f"Class {i}")
#         #axs[i].set_yscale("symlog", linthresh=0.0015)
    
#     ax=axs[5]    
#     ax.scatter(None, None, marker="D", color="black", label=0)
#     ax.scatter(None, None, marker="*", color="tab:blue", label=1.08e5 / days)
#     ax.scatter(None, None, marker="o", color="tab:orange", label=2.16e5 / days)
#     ax.scatter(None, None, marker="^", color="tab:green", label=3.24e5 / days)
#     ax.scatter(None, None, marker="s", color="tab:red", label=4.32e5 / days)
#     ax.axis("off")
#     ax.legend(title="Wind event duration (days)", loc="center")

#     axs[2].set_ylabel("$\\partial_t\\mathcal{V}$ range (Sv$\\,$km$^{-1}$)")
#     axs[4].set_xlabel("Wind stress (N$\\,$m$^{-2}$)")

#     fig.suptitle("Maixma and minima of volume tendencies")
#     fig.tight_layout()
    
#     fig_out_path = figure_path / "EnsemblePeakFormation.pdf"
#     logging.info(f"Saving figure to {fig_out_path}")
#     fig.savefig(fig_out_path)
    

int_wind_stress = True
if int_wind_stress:
    from scipy.special import erf  
    ds_ensemble = xr.open_zarr(ensemble_path)
    ds_wmt = xr.open_zarr(wmt_path).squeeze()
    
    ds_wmt = ds_wmt.assign_coords({"wind_stress": ds_ensemble["wind_stress"],
                                   "wind_duration": ds_ensemble["wind_duration"]})

    tau_int = np.sqrt(2 * np.pi) \
              * ds_wmt["wind_stress"] * ds_wmt["wind_duration"] \
              * erf(10.5 * days / np.sqrt(2) / ds_wmt["wind_duration"]) * 1e-5


    ds_wmt['volTEND2'] = ds_wmt["volANOM"].diff("time") / ds_wmt["time"].astype("float32").diff("time") / 1e-9 
    
    # We convert to Sv / km here too
    da_voltend_mn = ds_wmt["volTEND2"].mean("time").compute() * 1e-6 * 1e3
    da_voltend_max = ds_wmt["volTEND2"].max("time").compute() * 1e-6 * 1e3
    da_voltend_min = ds_wmt["volTEND2"].min("time").compute() * 1e-6 * 1e3
    
    fig, axs = plt.subplots(3, 2, figsize=(6, 7), sharex=True, sharey=True)
    axs = axs.flatten()
    ds_mld = xr.open_zarr(mld_path).squeeze()
    for classs in ds_wmt["classs"]:
        for run in ds_mld['run']:
            marker, colour = select_marker(ds_ensemble["wind_duration"].sel(run=run))
            axs[classs].scatter(-tau_int.sel(run=run),
                                da_voltend_min.sel(classs=classs,run=run) * 1e2, 
                                color=colour, marker=marker, facecolors="none")
            axs[classs].scatter(-tau_int.sel(run=run),
                                da_voltend_max.sel(classs=classs, run=run) * 1e2,
                                color=colour, marker=marker)


    for i in range(5):
        axs[i].set_title(f"Class {i}")
        axs[i].ticklabel_format(axis="both", style="sci", scilimits=(0, 0), useMathText=True)
        #axs[i].set_yscale("symlog", linthresh=0.0015)
    
    ax=axs[5]    
    ax.scatter(None, None, marker="D", color="black", label=0)
    ax.scatter(None, None, marker="*", color="tab:blue", label=1.08e5 / days)
    ax.scatter(None, None, marker="o", color="tab:orange", label=2.16e5 / days)
    ax.scatter(None, None, marker="^", color="tab:green", label=3.24e5 / days)
    ax.scatter(None, None, marker="s", color="tab:red", label=4.32e5 / days)
    ax.axis("off")
    ax.legend(title="Wind event duration (days)", loc="center")

    axs[2].set_ylabel("$\\partial_t\\mathcal{V}$ ($\\times$10$^{-2}$ Sv$\\,$km$^{-1}$)", usetex=True)
    axs[4].set_xlabel("Integrated wind stress ($\\times$10$^{5}$ N$\\,$s$\\,$m$^{-2}$)", usetex=True)

    axs[0].set_title("(a)", loc="left")
    axs[1].set_title("(b)", loc="left")
    axs[2].set_title("(c)", loc="left")
    axs[3].set_title("(d)", loc="left")
    axs[4].set_title("(e)", loc="left")

    fig.suptitle("Formation rate maxima and minima")
    fig.tight_layout()
    
    fig_out_path = figure_path / "EnsemblePeakFormation.pdf"
    logging.info(f"Saving figure to {fig_out_path}")
    fig.savefig(fig_out_path)
    
int_wind_stress_anom = True
if int_wind_stress_anom:
    from scipy.special import erf  
    ds_ensemble = xr.open_zarr(ensemble_path)
    ds_wmt = xr.open_zarr(wmt_path).squeeze()
    
    ds_wmt = ds_wmt.assign_coords({"wind_stress": ds_ensemble["wind_stress"],
                                   "wind_duration": ds_ensemble["wind_duration"]})

    ds_wmt["tau_int"] = np.sqrt(2 * np.pi) \
                        * ds_wmt["wind_stress"] * ds_wmt["wind_duration"] \
                        * erf(10.5 * days / np.sqrt(2) \
                        / ds_wmt["wind_duration"]) * 1e-5
    
    fig, axs = plt.subplots(3, 2, figsize=(6, 7), sharex=True, sharey=True)
    axs = axs.flatten()

    for classs in ds_wmt["classs"]:
        for run in ds_mld['run']:
            marker, colour = select_marker(ds_ensemble["wind_duration"].sel(run=run))
        
            axs[classs].scatter(-ds_wmt["tau_int"].sel(run=run),
                                ds_wmt["volANOM"].isel(time=-1).sel(run=run,
                                                                    classs=classs) * 1e3 * 1e-9,
                                marker=marker,
                                color=colour)   

    for i in range(5):
        axs[i].set_title(f"Class {i}")
        axs[i].ticklabel_format(axis="both", style="sci", scilimits=(0, 0), useMathText=True)
    
    ax=axs[5]    
    ax.scatter(None, None, marker="D", color="black", label=0)
    ax.scatter(None, None, marker="*", color="tab:blue", label=1.08e5 / days)
    ax.scatter(None, None, marker="o", color="tab:orange", label=2.16e5 / days)
    ax.scatter(None, None, marker="^", color="tab:green", label=3.24e5 / days)
    ax.scatter(None, None, marker="s", color="tab:red", label=4.32e5 / days)
    ax.axis("off")
    ax.legend(title="Wind event duration (days)", loc="center")

    axs[2].set_ylabel("$\\Delta \\mathcal{V}$ ($\\times$10$^{9}$ m$^3\\,$km$^{-1}$)", usetex=True)
    axs[4].set_xlabel("Integrated wind stress ($\\times$10$^{5}$ N$\\,$s$\\,$m$^{-2}$)", usetex=True)

    axs[0].set_title("(a)", loc="left")
    axs[1].set_title("(b)", loc="left")
    axs[2].set_title("(c)", loc="left")
    axs[3].set_title("(d)", loc="left")
    axs[4].set_title("(e)", loc="left")
        

    fig.suptitle("Volume anomaly")
    fig.tight_layout()
    
    out_path = figure_path / "EnsembleVolAnom.pdf"
    logging.info(f"Saving to {out_path}")
    fig.savefig(out_path)   


tau_int_delta_mld = True
if tau_int_delta_mld:
    ds_ensemble = xr.open_zarr(ensemble_path)
    ds_mld = xr.open_zarr(mld_path).squeeze()
    ds_mld = ds_mld.assign_coords({"wind_stress": ds_ensemble["wind_stress"],
                                   "wind_duration": ds_ensemble["wind_duration"]})

    ds_mld["tau_int"] = np.sqrt(2 * np.pi) \
                        * ds_mld["wind_stress"] * ds_mld["wind_duration"] \
                        * erf(10.5 * days / np.sqrt(2) \
                        / ds_mld["wind_duration"]) * 1e-5
    
    delta_mld = ds_mld["mn_drop_depth"].isel(time=0) - ds_mld["mn_drop_depth"].isel(time=-1)
    
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    for run in ds_mld['run']:
        marker, colour = select_marker(ds_ensemble["wind_duration"].sel(run=run))
        
        ax.scatter(-ds_mld["tau_int"].sel(run=run),
                   delta_mld.sel(run=run),
                   marker=marker,
                   color=colour)   
    
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
    
    ax.scatter(None, None, marker="D", color="black", label=0)
    ax.scatter(None, None, marker="*", color="tab:blue", label=1.08e5 / days)
    ax.scatter(None, None, marker="o", color="tab:orange", label=2.16e5 / days)
    ax.scatter(None, None, marker="^", color="tab:green", label=3.24e5 / days)
    ax.scatter(None, None, marker="s", color="tab:red", label=4.32e5 / days)
        
    ax.set_xlabel("Integrated wind stress ($\\times$10$^{5}$ N$\\,$s$\\,$m$^{-2}$)", usetex=True)
    ax.set_ylabel("$\Delta$ MLD (m)", usetex=True)
    ax.set_title("Change in mixed layer depth")
    ax.legend(title="Wind event duration (days)")
    
    fig.tight_layout()
    fig.savefig(figure_path / "DeltaMLD.pdf")