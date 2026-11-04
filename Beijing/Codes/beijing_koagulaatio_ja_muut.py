import xarray as xr
import numpy as np
import pandas as pd
import aerosol.functions as af
import matplotlib.pyplot as plt
import aerosol.plotting as ap
import matplotlib.dates as mdates
import matplotlib.cm as cm

if not hasattr(cm, "get_cmap"):
    cm.get_cmap = plt.get_cmap

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

neg_p.columns = neg_p.columns.astype(float)
neg_i.columns = neg_i.columns.astype(float)

neg_p.index = neg_p.index.tz_localize("UTC").tz_convert("Asia/Shanghai")
neg_i.index = neg_i.index.tz_localize("UTC").tz_convert("Asia/Shanghai")

neg_i[neg_i <= 0] = np.nan
neg_p[neg_p <= 0] = np.nan

common_index = neg_p.index.intersection(neg_i.index)

neg_p = neg_p.loc[common_index]
neg_i = neg_i.loc[common_index]

common_cols = neg_p.columns.intersection(neg_i.columns)

neg_p = neg_p.loc[:, common_cols]
neg_i = neg_i.loc[:, common_cols]

f = neg_i / neg_p
a = 1.6e-6
d0 = 1e-9
a0 = 0.006e-6
l = 1.2
dp_nm = np.array(neg_p.columns[0:39], dtype=float)
dp_nm_2 = dp_nm*1e9
N = 300
N = af.calc_conc(neg_i, 0.8e-9, 1.6e-9).squeeze()

coagN = af.calc_coags(neg_p, dp_nm).squeeze()
coagS = ((1+1.5/np.sqrt(dp_nm_2)-4/(dp_nm_2+2)**2)*coagN)

coagN = np.asarray(coagN)
coagS = np.asarray(coagS)

a_0 = a0*(dp_nm / d0)**l



f_eq_1 = pd.Series(
    {
        diameter: af.eq_charge_frac(diameter, -1)
        for diameter in neg_p.columns
    }
)

f_eq_2 = pd.Series(
    {
        diameter: (a0*(diameter / d0)**l)/a
        for diameter in neg_p.columns
    }
)

frac_1 = f/f_eq_1
frac_2 = f/f_eq_2
#frac_3 = f_coag/f_eq_2
#frac_4 = f_coag_2/f_eq_2

frac_1 = frac_1.replace([np.inf, -np.inf], np.nan)
frac_2 = frac_2.replace([np.inf, -np.inf], np.nan)
#frac_3 = frac_3.replace([np.inf, -np.inf], np.nan)
#frac_4 = frac_4.replace([np.inf, -np.inf], np.nan)

grouped_1 = frac_1.groupby(
    frac_1.index.strftime('%H:%M')
).median()

grouped_2 = frac_2.groupby(
    frac_2.index.strftime('%H:%M')
).median()

#grouped_3 = frac_3.groupby(
#    frac_3.index.strftime('%H:%M')
#).median()

#grouped_4 = frac_4.groupby(
#    frac_4.index.strftime('%H:%M')
#).median()

mean_1 = frac_1.mean(axis=0)
mean_2 = frac_2.mean(axis=0)
#mean_3 = frac_3.mean(axis=0)
#mean_4 = frac_4.mean(axis=0)

# Diameter in nm for plotting
diam_nm = np.array(neg_p.columns, dtype=float)

fig, ax1 = plt.subplots(figsize=(8, 5))

# Left axis
#line1 = ax1.plot(
#    diam_nm,
#    mean_1.to_numpy(),
#    label="S, Wiedensohler",
#)

#line2 = ax1.plot(
#    diam_nm,
#    mean_2.to_numpy(),
#    label="S, Vellu",
#)

#line3 = ax1.plot(
#    diam_nm,
#    mean_3.to_numpy(),
#    label="S, Vellu + koagulaatio",
#)

#line4 = ax1.plot(
#    diam_nm,
#    mean_4.to_numpy(),
#    label="S, Vellu + koagulaatio 2",
#)


ax1.set_xlabel("Diameter (m)")
ax1.set_ylabel("Charge fraction")
ax1.set_xscale("log")
#ax1.set_xlim(2e-9, 30e-9)
#ax1.set_ylim(0, 0.70)

# Combined legend
handles1, labels1 = ax1.get_legend_handles_labels()

ax1.legend(
    handles1,
    labels1,
    loc="best",
)

plt.tight_layout()
plt.show()