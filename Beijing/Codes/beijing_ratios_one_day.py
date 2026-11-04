import xarray as xr
import numpy as np
import pandas as pd
import aerosol.functions as af
import importlib.metadata
import matplotlib.pyplot as plt
import aerosol.plotting as ap
import glob
from pathlib import Path

ds = xr.open_dataset(r"C:\Users\Kia\Documents\beijing\NAIS_20230722.nc")

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

common_index = neg_p.index.intersection(neg_i.index)

neg_p = neg_p.loc[common_index]
neg_i = neg_i.loc[common_index]

neg_p.columns = neg_p.columns.astype(float)
neg_i.columns = neg_i.columns.astype(float)

#neg_p = neg_p.loc[:, neg_p.columns <= 11e-9]
#neg_i = neg_i.loc[:, neg_i.columns <= 11e-9]

#neg_p = neg_p.loc[:, neg_p.columns >= 2.5e-9]
#neg_i = neg_i.loc[:, neg_i.columns >= 2.5e-9]

neg_p.index = neg_p.index.tz_localize("UTC").tz_convert("Asia/Shanghai")
neg_i.index = neg_i.index.tz_localize("UTC").tz_convert("Asia/Shanghai")

concentrations = af.calc_conc_interp(neg_p, 2.5e-9, 5e-9).squeeze()

threshold = 4500

mask = concentrations > threshold

neg_p_masked = neg_p.copy()
neg_i_masked = neg_i.copy()

#neg_p_masked[~mask] = np.nan
#neg_i_masked[~mask] = np.nan

#df_events.loc[:, df_events.columns < 2.5e-9] = np.nan
#df_events.loc[:, df_events.columns > 5e-9] = np.nan

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# measured charging fraction
f_meas = neg_i_masked / neg_p_masked

f_eq = pd.Series(
    {
        diameter: af.eq_charge_frac(diameter, -1)
        for diameter in f_meas.columns
    }
)

overcharge = f_meas.divide(
    f_eq,
    axis=1
)
small = overcharge.copy()
small = small.loc[:, small.columns >= 4e-9]
small = small.loc[:, small.columns <= 5e-9]

large = overcharge.copy()
large = large.loc[:, large.columns >= 2.5e-9]
large = large.loc[:, large.columns <= 3e-9]

small_med = small.median(axis=1, numeric_only=True)
large_med = large.median(axis=1, numeric_only=True)

# ratio between size ranges
ratio_series = large_med / small_med

grouped = ratio_series.groupby(
    ratio_series.index.strftime('%H:%M')
)

p25 = grouped.quantile(0.25)
p50 = grouped.quantile(0.50)
p75 = grouped.quantile(0.75)

# convert to datetime
for s in [p25, p50, p75]:
    s.index = pd.to_datetime(s.index, format='%H:%M')

fig, ax = plt.subplots(figsize=(10,5))

ax.plot(p25.index, p25.values, label='25th percentile')
ax.plot(p50.index, p50.values, label='50th percentile')
ax.plot(p75.index, p75.values, label='75th percentile')

ax.set_xlabel("Time of day")
ax.set_ylabel("Ratio")

ax.set_ylim(0, 50)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter('%H:%M')
)

plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()