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

df = pd.read_excel(r"C:\Users\kivekia\Downloads\Python\BUCT DMPS 2018-2022\BUCT DMPS 2018-2022\DMPS_BUCT_to_D.xlsx",
                    index_col=0,
                    parse_dates=True
                    )

dp = df.columns[0:24]
dp_interp = neg_p.columns.to_numpy(dtype=float) * 1e9
dp_nm = neg_p.columns.to_numpy(dtype=float)

coagN = af.calc_coags(df, dp_nm)
print(coagN)
coagN_median = coagN.median(axis=0)

from scipy.interpolate import interp1d

f_interp = interp1d(
    dp,
    coagN_median,
    kind="linear",
    bounds_error=False,
    fill_value="extrapolate"
)


# New diameter grid every 1 nm

# Interpolated coagulation sink
coagN_interp = f_interp(dp_interp)

plt.figure(figsize=(8, 5))

plt.plot(
    dp,
    coagN_median,
    "o",
    label="Original"
)

plt.plot(
    dp_interp,
    coagN_interp,
    "-",
    label="Interpolated (1 nm)"
)

plt.xlabel("Diameter (nm)")
plt.ylabel("Coagulation sink (s$^{-1}$)")
plt.title("Coagulation sink vs diameter")

plt.xscale("log")
plt.grid(True, which="both", alpha=0.3)
plt.legend()

plt.show()