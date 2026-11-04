import xarray as xr
import numpy as np
import pandas as pd
import aerosol.functions as af
import matplotlib.pyplot as plt
import aerosol.plotting as ap

import matplotlib
import matplotlib.cm as cm

if not hasattr(cm, "get_cmap"):
    cm.get_cmap = matplotlib.colormaps.get_cmap

ds = xr.open_dataset(r"C:\Users\kivekia\Downloads\Python\beijing_all.nc")

ds = ds.sortby('time')

negp_og=ds["neg_particles"]
negi_og=ds["neg_ions"]
posi_og=ds["pos_ions"]

negp_og=negp_og.copy()
negi_og=negi_og.copy()
posi_og=posi_og.copy()

negp_og2=negp_og.dropna(dim="diameter", how="all")
negp=negp_og2.dropna(dim="time", how="all")

negi_og2=negi_og.dropna(dim="diameter", how="all")
negi=negi_og2.dropna(dim="time", how="all")

posi_og2=posi_og.dropna(dim="diameter", how="all")
posi=posi_og2.dropna(dim="time", how="all")

neg_p=negp.to_pandas()
neg_i=negi.to_pandas()
pos_i=posi.to_pandas()

neg_p.index = neg_p.index.tz_localize("UTC").tz_convert("Asia/Shanghai")
neg_i.index = neg_i.index.tz_localize("UTC").tz_convert("Asia/Shanghai")
pos_i.index = pos_i.index.tz_localize("UTC").tz_convert("Asia/Shanghai")

#neg_p = neg_p.loc[:, neg_p.columns <= 11e-9]
#neg_i = neg_i.loc[:, neg_i.columns <= 11e-9]

#neg_p = neg_p.loc[:, neg_p.columns >= 2.5e-9]
#neg_i = neg_i.loc[:, neg_i.columns >= 2.5e-9]

#pos_i = pos_i.loc[:, pos_i.columns <= 11e-9]
#pos_i = pos_i.loc[:, pos_i.columns >= 2.5e-9]

df = pd.read_excel(r"C:\Users\kivekia\Downloads\Python\BUCT DMPS 2018-2022\BUCT DMPS 2018-2022\DMPS_BUCT_to_D.xlsx",
                    index_col=0,
                    parse_dates=True
                    )

dp_interp = neg_p.columns.to_numpy(dtype=float) * 1e9
dp_nm = neg_p.columns.to_numpy(dtype=float)

coagN = af.calc_coags(df, dp_nm)
coagN_median = coagN.median(axis=0)


concentrations = af.calc_conc_interp(neg_p, 2.5e-9, 5e-9).squeeze()

night = concentrations.between_time(
    "01:00",
    "02:59:59"
)

nightly_background = (
    night
    .resample("1D")
    .median()
)

nightly_background = nightly_background.dropna()

day = concentrations.index.normalize()

background_for_each_time = (
    pd.Series(
        nightly_background.values,
        index=nightly_background.index
    )
    .reindex(day)
)

background_for_each_time.index = concentrations.index

alpha = 1.5

threshold = alpha * background_for_each_time

above_threshold = concentrations > threshold

above_threshold = above_threshold.fillna(False)

duration = 3

event_binary = above_threshold.astype(int)

groups = event_binary.ne(event_binary.shift()).cumsum()

run_length = event_binary.groupby(groups).transform("sum")

mask = above_threshold & (run_length >= duration)

#mask = concentrations > threshold

event = neg_p.where(mask, np.nan)

event = event.dropna(how="all")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

df_30 = event.resample('30min').median()

df_30 = df_30.dropna(how="all")

df_30["clocktime"] = df_30.index.strftime('%H:%M')

median_diurnal = (
    df_30
    .groupby("clocktime")
    .median(numeric_only=True)
)

times = sorted(median_diurnal.index)

median_diurnal = median_diurnal.loc[times]

median_diurnal.index = pd.to_datetime(
    median_diurnal.index,
    format='%H:%M')

fig, ax = plt.subplots(figsize=(12,6))

im, cbar = ap.plot_aerosol_dist(
    median_diurnal,
    ax,
    norm='log', 
)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter('%H:%M'))
#clim=[0.28, 0.60]

#cbar.set_ticks([1.2e4, 1.5e4, 2e4, 3e4])
#cbar.set_ticklabels([""])
#cbar.set_label('Concentration')

plt.show()
#plt.savefig("concentrations_n_pos_masked.png", dpi=300, bbox_inches="tight")