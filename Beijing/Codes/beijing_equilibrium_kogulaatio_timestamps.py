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
N1=300

N = af.calc_conc(neg_i, 0.8e-9, 1.6e-9).squeeze().to_numpy()

coagN = af.calc_coags(neg_p, dp_nm).to_numpy()
coagS = (1 + 1.5/np.sqrt(dp_nm_2) - 4/(dp_nm_2 + 2)**2) * coagN

coagN1 = af.calc_coags(neg_p, dp_nm).squeeze()
coagS1 = ((1+1.5/np.sqrt(dp_nm_2)-4/(dp_nm_2+2)**2)*coagN)

coagN_med = np.nanmedian(coagN1, axis=0)
coagS_med = np.nanmedian(coagS1, axis=0)

timestamp1 = pd.Timestamp("2023-12-31 02:00:00", tz="Asia/Shanghai")
t1 = neg_p.index.get_loc(timestamp1)

N_t1 = N[t1]              # scalar
coagN_t1 = coagN[t1, :]   # length 39
coagS_t1 = coagS[t1, :]   # length 39

timestamp2 = pd.Timestamp("2023-12-31 12:00:00", tz="Asia/Shanghai")
t2 = neg_p.index.get_loc(timestamp2)

N_t2 = N[t2]              # scalar
coagN_t2 = coagN[t2, :]   # length 39
coagS_t2 = coagS[t2, :]   # length 39

timestamp3 = pd.Timestamp("2023-12-31 17:10:00", tz="Asia/Shanghai")
t3 = neg_p.index.get_loc(timestamp3)

N_t3 = N[t3]              # scalar
coagN_t3 = coagN[t3, :]   # length 39
coagS_t3 = coagS[t3, :]   # length 39

timestamp4 = pd.Timestamp("2023-12-31 18:00:00", tz="Asia/Shanghai")
t4 = neg_p.index.get_loc(timestamp4)

N_t4 = N[t4]              # scalar
coagN_t4 = coagN[t4, :]   # length 39
coagS_t4 = coagS[t4, :]   # length 39

f_eq_1 = np.empty(len(dp_nm))

#        + coagS_t1[i]
#        - coagN_t1[i]

for i, diameter in enumerate(dp_nm):
    A = a0 * (diameter/d0)**l

    f_eq_1[i] = (
        A * N1
    ) / (
        a*N1
        + coagS_med[i]
        - coagN_med[i]
        - 2*A*N1
    )

f_eq_2 = np.empty(len(dp_nm))

for i, diameter in enumerate(dp_nm):
    A = a0 * (diameter/d0)**l

    f_eq_2[i] = (
        A * N1
    ) / (
        a*N1
        + coagS_med[i]
        - coagN_med[i]
        - 2*A*N1
    )

f_eq_3 = np.empty(len(dp_nm))

for i, diameter in enumerate(dp_nm):
    A = a0 * (diameter/d0)**l

    f_eq_3[i] = (
        A * N1
    ) / (
        a*N1
        + coagS_med[i]
        - coagN_med[i]
        - 2*A*N1
    )

f_eq_4 = np.empty(len(dp_nm))

for i, diameter in enumerate(dp_nm):
    A = a0 * (diameter/d0)**l

    f_eq_4[i] = (
        A * N1
    ) / (
        a*N1
        + coagS_med[i]
        - coagN_med[i]
        - 2*A*N1
        + 2 / 3600
    )

f_eq_5 = np.empty(len(dp_nm))

for i, diameter in enumerate(dp_nm):
    A = a0 * (diameter/d0)**l

    f_eq_5[i] = (
        A * N1
    ) / (
        a*N1
        + coagS_med[i]
        - coagN_med[i]
        - 2*A*N1
        + 1 / 3600
    )

# Diameter in nm for plotting
diam_nm = np.array(neg_p.columns[0:39], dtype=float)

fig, ax1 = plt.subplots(figsize=(8, 5))

# Left axis
line1 = ax1.plot(
    diam_nm,
    f_eq_1,
    label=r"Equilibrium fraction, $\Delta$GR=0",
)

#line2 = ax1.plot(
#    diam_nm,
#    f_eq_2,
#    label="Equilibrium fraction, klo 12",
#)

#line3 = ax1.plot(
#    diam_nm,
#    f_eq_3,
#    label="Equilibrium fraction, klo 17",
#)

line4 = ax1.plot(
    diam_nm,
    f_eq_4,
    label=r"Equilibrium fraction, $\Delta$GR=2",
)

line5 = ax1.plot(
    diam_nm,
    f_eq_5,
    label=r"Equilibrium fraction, $\Delta$GR=1",
)

ax1.set_xlabel("Diameter (m)")
ax1.set_ylabel("Charge fraction")
ax1.set_xscale("log")
ax1.set_xlim(2e-9, 20e-9)
ax1.set_ylim(0, 0.2)

# Combined legend
handles1, labels1 = ax1.get_legend_handles_labels()

ax1.legend(
    handles1,
    labels1,
    loc="best",
)

plt.tight_layout()
plt.show()