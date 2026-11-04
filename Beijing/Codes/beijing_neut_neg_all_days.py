import xarray as xr
import numpy as np
import pandas as pd
import aerosol.functions as af
import matplotlib.pyplot as plt
import aerosol.plotting as ap
import glob
import matplotlib.dates as mdates

files = sorted(glob.glob("/home/kivekia/Summer Work 2026/beijing/*.nc"))
datasets = [xr.open_dataset(f) for f in files]

ds = xr.concat(datasets, dim='time')

negp_og=ds["neg_particles"]
negi_og=ds["neg_ions"]

negp_og=negp_og.copy()
negi_og=negi_og.copy()

negp_og2=negp_og.dropna(dim="diameter", how="all")
negp=negp_og2.dropna(dim="time", how="all")

negi_og2=negi_og.dropna(dim="diameter", how="all")
negi=negi_og2.dropna(dim="time", how="all")

neg_p=negp.to_pandas()
neg_i=negi.to_pandas()

neg_p.columns = neg_p.columns.astype(float)
neg_i.columns = neg_i.columns.astype(float)

neg_p.index = neg_p.index.tz_localize("UTC").tz_convert("Asia/Shanghai")
neg_i.index = neg_i.index.tz_localize("UTC").tz_convert("Asia/Shanghai")

ratio = neg_p / neg_i

df_30 = ratio.resample('30min', origin = "start_day").median()

df_30["clocktime"] = df_30.index.strftime('%H:%M')

median_diurnal = ( df_30.groupby("clocktime").median())

times = sorted(median_diurnal.index)

median_diurnal.index = pd.to_datetime( median_diurnal.index, format='%H:%M')

fig, ax = plt.subplots(figsize=(10,6))

im, cbar = ap.plot_aerosol_dist(
    median_diurnal,
    ax,
    norm='linear',
    clim=[1, 250]
)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter('%H:%M'))

cbar.set_ticks([50, 100, 150, 200])
cbar.set_ticklabels(["50", "100", "150", "200"])

cbar.set_label('Total Particles / Negative Ions')

plt.savefig("ratio_beijing_tot_neg_all_days.png", bbox_inches='tight', dpi=300)