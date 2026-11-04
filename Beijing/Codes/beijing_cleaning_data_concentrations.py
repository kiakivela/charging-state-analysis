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
#neg_i = neg_i.loc[:, neg_i.columns <= 7e-9]

#neg_p = neg_p.loc[:, neg_p.columns >= 2.5e-9]
#neg_i = neg_i.loc[:, neg_i.columns >= 5e-9]

#pos_i = pos_i.loc[:, pos_i.columns <= 11e-9]
#pos_i = pos_i.loc[:, pos_i.columns >= 2.5e-9]

concentrations = af.calc_conc_interp(neg_p, 2.5e-9, 5e-9).squeeze()
concentrations_2 = af.calc_conc_interp(neg_i, 2.5e-9, 5e-9).squeeze()
concentrations_3 = af.calc_conc_interp(pos_i, 2.5e-9, 5e-9).squeeze()

#t = np.percentile(concentrations_3, 60)

threshold = 2200
threshold_2 = 15
threshold_3 = 25

mask = concentrations > threshold
mask_2 = concentrations_2 > threshold_2
mask_3 = concentrations_3 > threshold_3

neg_p_masked = neg_p.copy()
neg_i_masked = neg_i.copy()
pos_i_masked = pos_i.copy()

neg_p_masked[~mask] = np.nan
neg_i_masked[~mask_2] = np.nan
pos_i_masked[~mask_3] = np.nan

daily_median = (
    neg_i.between_time("01:00", "03:00")
      .resample("1D")
      .median()
      .dropna(how="all")
)
#      .dropna(how="all")

#specific_time = daily_median.copy()
#specific_time = specific_time.loc["2020-07-01":"2021-02-20"]

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

fig, ax = plt.subplots(figsize=(12,6))

im, cbar = ap.plot_aerosol_dist(
    daily_median,
    ax,
    norm='log', 
)

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
#clim=[0.28, 0.60]

#cbar.set_ticks([1.2e4, 1.5e4, 2e4, 3e4])
#cbar.set_ticklabels([""])
#cbar.set_label('Concentration')

#plt.xticks(fontsize=18)
#plt.yticks(fontsize=18)
ax.set_title("Negative ions, median between 1:00-3:00, one value a day, NaN-rows removed", fontsize=16)

#plt.show()
plt.savefig("neg_i_one_a_day.png", dpi=300, bbox_inches="tight")