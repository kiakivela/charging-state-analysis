import xarray as xr
import numpy as np
import pandas as pd
import aerosol.functions as af
import matplotlib.pyplot as plt

ds = xr.open_dataset(r"C:\Users\kivekia\Downloads\Python\beijing_all.nc")

ds = ds.sortby('time')

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

neg_p.index = neg_p.index.tz_localize("UTC").tz_convert("Asia/Shanghai")
neg_i.index = neg_i.index.tz_localize("UTC").tz_convert("Asia/Shanghai")

conc_1 = af.calc_conc(neg_i, 0.8e-09, 1.2e-9).squeeze()

#c = conc_1.median()
f = conc_1/1e4

f_eq = af.eq_charge_frac(1e-9, -1)

seq = f/f_eq

concentrations = af.calc_conc(neg_p, 2.5e-9, 5e-9).squeeze()

threshold = 2000

mask = concentrations > threshold

neg_p_masked = neg_p.copy()
neg_i_masked = neg_i.copy()

neg_p_masked[~mask] = np.nan
neg_i_masked[~mask] = np.nan

d1=2.5e-9
d2=3e-9

d3=4e-9
d4=5e-9

G = 2e-9
a = 1.6e-6
N = af.calc_conc(neg_i_masked, 0.8e-9, 2e-9).squeeze()
S0 = seq
S0 = S0.to_numpy()[:, None]

N[N <= 0] = np.nan

K = ((a*N)/G)

d0 = 1e-9
dp_nm = np.array(neg_p_masked.columns, dtype=float)

# Range of dp values
#dp = np.linspace(dp_nm.min()*1e9, d0, 200)

K = K.to_numpy()                   # shape (nt,)
K = K[:, None]                     # shape (nt,1)

S = (
    1
     - 1/(K*dp_nm)
    + ((1 + (S0-1)*K*d0)/(K*dp_nm))
      * np.exp(-K*(dp_nm-d0))
)

S = pd.DataFrame(
    S,
    index=neg_p_masked.index,
    columns=dp_nm
)

#small_idx = np.argmin(np.abs(dp_nm - (d1+d2)/2))
#large_idx = np.argmin(np.abs(dp_nm - (d3+d4)/2))

#ratio_series = pd.Series(
   # S.iloc[:, small_idx] / S.iloc[:, large_idx],
  #  index=neg_p_masked.index
#)

small_mask = (dp_nm >= d1) & (dp_nm <= d2)
large_mask = (dp_nm >= d3) & (dp_nm <= d4)

small_S = S.loc[:, small_mask].mean(axis=1)
large_S = S.loc[:, large_mask].mean(axis=1)

ratio_series = small_S / large_S

grouped = ratio_series.groupby(
    ratio_series.index.strftime('%H:%M')
)

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

p25 = grouped.quantile(0.25)
p50 = grouped.quantile(0.50)
p75 = grouped.quantile(0.75)

# convert to datetime
for n in [p25, p50, p75]:
    n.index = pd.to_datetime(n.index, format='%H:%M')

fig, ax = plt.subplots(figsize=(10,5))

ax.plot(p25.index, p25.values, label='25th percentile')
ax.plot(p50.index, p50.values, label='50th percentile')
ax.plot(p75.index, p75.values, label='75th percentile')

ax.set_xlabel("Time of day")
ax.set_ylabel("Ratio")

#ax.set_ylim(2.0885, 2.0910)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter('%H:%M')
)

plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()