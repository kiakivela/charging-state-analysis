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

a = 1.6e-6
d0 = 1e-9
a0 = 0.006e-6

timestamp = pd.Timestamp("2023-12-31 08:10:00", tz="Asia/Shanghai")
ti=neg_i.loc[timestamp]
tp = neg_p.loc[timestamp]    

common_cols = neg_p.columns.intersection(neg_i.columns)

tp = neg_p.loc[timestamp, common_cols]
ti = neg_i.loc[timestamp, common_cols]

frac = ti/tp

f_eq_1 = pd.Series(
    [af.eq_charge_frac(d, -1) for d in frac.index],
    index=frac.index
)

f1 = frac / f_eq_1

G = 2e-9
a = 1.6e-6
N = af.calc_conc(neg_i, 0.8e-9, 2e-9).squeeze()
x=2
l=1
S0 = seq
S0 = S0.to_numpy()[:, None]

N[N <= 0] = np.nan

K = ((a*N)/G)

d0 = 1e-9
dp_nm = np.array(neg_p.columns, dtype=float)

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
    index=neg_p.index,
    columns=dp_nm
)

f2=S.loc[timestamp]

#C = af.calc_coags(neg_p, neg_p.columns[0:39]).squeeze()
tp = neg_p.loc[[timestamp]]      # note the double brackets

C = af.calc_coags(tp, tp.columns[0:39]).squeeze()

ratio_list = []
time_list = []

Cmat = C.values
Nvec = N.reindex(C.index).values
N0 = N.loc[timestamp]

dp_nm = np.array(tp.columns[:-1], dtype=float)

valid = np.isfinite(Nvec)

n_times = len(C)
n_dp = len(dp_nm)

SS = np.zeros(len(dp_nm))
S0 = seq.loc[timestamp]

SS[-1] = S0

for i in range(n_dp-1, 0, -1):

    dpi = dp_nm[i]

    ddp = dp_nm[i-1] - dp_nm[i]

    rhs = (
        a*N0/G
        - (a*N0/G)*SS[i]
        + ((1-x)*Cmat[i]/G)*SS[i]
        - (l/dpi)*SS[i]
    )

    SS[i-1] = SS[i] + rhs*ddp

SS = pd.Series(
    SS,
    index=dp_nm,
    name=timestamp
)

f3=SS

# Diameter in nm for plotting
diam_nm = np.array(neg_p.columns, dtype=float)

fig, ax1 = plt.subplots(figsize=(8, 5))

# Left axis
line1 = ax1.plot(
    diam_nm,
    f1.values,
    label="S(dp), Wiedensohler",
)

line2 = ax1.plot(
    diam_nm,
    f2.values,
    label="S(dp), Vellu (analytical)",
)

line2 = ax1.plot(
    f3.index,
    f3.values,
    label="S(dp), Vellu (koagulaatio)",
)


ax1.set_xlabel("Diameter (m)")
ax1.set_ylabel("Charging State")
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