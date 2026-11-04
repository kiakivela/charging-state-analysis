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

a = 1.6e-6
d0 = 1e-9
a0 = 0.006e-6
l = 1.2
dp_nm = np.array(neg_p.columns[0:39], dtype=float)
dp_nm_2 = dp_nm*1e9

N = af.calc_conc(neg_i, 0.8e-9, 1.6e-9).squeeze().to_numpy()

N_med = np.nanmedian(N, axis=0)

g = (1+1.5/np.sqrt(dp_nm_2)-4/(dp_nm_2+2)**2)

dp = neg_p.columns[:39]

neg_p_calc = neg_p.loc[:, dp]

neg_i_calc = neg_i.reindex(columns=dp)

f = neg_i_calc / neg_p_calc

f_med = np.nanmedian(f, axis=0)

timestamp1 = pd.Timestamp("2023-12-31 02:00:00", tz="Asia/Shanghai")
t1 = neg_p.index.get_loc(timestamp1)

N_t1 = N[t1]           
f_t1 = f.iloc[t1, :].to_numpy()  

timestamp2 = pd.Timestamp("2023-12-31 12:00:00", tz="Asia/Shanghai")
t2 = neg_p.index.get_loc(timestamp2)

N_t2 = N[t2]             
f_t2 = f.iloc[t2, :].to_numpy()

timestamp3 = pd.Timestamp("2023-12-31 17:10:00", tz="Asia/Shanghai")
t3 = neg_p.index.get_loc(timestamp3)

N_t3 = N[t3]              # scalar
f_t3 = f.iloc[t3, :].to_numpy()

timestamp4 = pd.Timestamp("2023-12-31 18:00:00", tz="Asia/Shanghai")
t4 = neg_p.index.get_loc(timestamp4)

N_t4 = N[t4]              # scalar
f_t4 = f.iloc[t4, :].to_numpy()

def calculate_coags(dp, N, f, a, a0, d0, l, g):
    result = np.full(len(dp), np.nan)

    for i, diameter in enumerate(dp):
        A = a0 * (diameter / d0)**l

        denominator = (g[i] - 1) * f[i]

        if (
            np.isfinite(N)
            and np.isfinite(f[i])
            and np.isfinite(denominator)
            and denominator != 0
            and np.isfinite(g[i])
            and g[i] != 1
        ):
            result[i] = (
                (A * N) / denominator
                + (2 * A * N - a * N) / (g[i] - 1)
            )

    return result

f_eq_1 = calculate_coags(dp_nm, N_t1, f_t1, a, a0, d0, l, g)
f_eq_2 = calculate_coags(dp_nm, N_t2, f_t2, a, a0, d0, l, g)
f_eq_3 = calculate_coags(dp_nm, N_t3, f_t3, a, a0, d0, l, g)
f_eq_4 = calculate_coags(dp_nm, N_t4, f_t4, a, a0, d0, l, g)
f_eq_5 = calculate_coags(dp_nm, N_med, f_med, a, a0, d0, l, g)

f_eq_1[~np.isfinite(f_eq_1)] = np.nan
f_eq_2[~np.isfinite(f_eq_2)] = np.nan
f_eq_3[~np.isfinite(f_eq_3)] = np.nan
f_eq_4[~np.isfinite(f_eq_4)] = np.nan
f_eq_5[~np.isfinite(f_eq_5)] = np.nan

# Diameter in nm for plotting
diam_nm = np.array(neg_p.columns[0:39], dtype=float)

fig, ax1 = plt.subplots(figsize=(8, 5))

# Left axis
#line1 = ax1.plot(
#    diam_nm,
#    f_eq_1,
#    label="KoagulaatioS, klo 2",
#)

#line2 = ax1.plot(
#    diam_nm,
#    f_eq_2,
#    label="KoagulaatioS, klo 12",
#)

#line3 = ax1.plot(
#    diam_nm,
#    f_eq_3,
#    label="KoagulaatioS, klo 17",
#)

#line4 = ax1.plot(
#    diam_nm,
#    f_eq_4,
#    label="KoagulaatioS, klo 18",
#)

line5 = ax1.plot(
    diam_nm,
    f_eq_5,
    label="CoagS, mediaani",
)

ax1.set_xlabel("Diameter (m)")
ax1.set_ylabel("CoagS")
ax1.set_xscale("log")
#ax1.set_xlim(2e-9, 20e-9)
#ax1.set_ylim(0, 0.2)

# Combined legend
handles1, labels1 = ax1.get_legend_handles_labels()

ax1.legend(
    handles1,
    labels1,
    loc="best",
)

plt.tight_layout()
plt.show()
