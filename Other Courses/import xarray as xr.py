import xarray as xr
import numpy as np
import pandas as pd
import aerosol.functions as af
import importlib.metadata

#print("aerosol version", importlib.metadata.version("aerosol.functions"))

ds = xr.open_dataset("/home/kivekia/Python/kumpula_1h_checked.nc")

da_original=ds["neg_particles"]
da_original=da_original.copy()
da2=da_original.dropna(dim="diameter", how="all")
da=da2.dropna(dim="time", how="any")

#da.encoding.pop("_FillValue", None)
#da.encoding.pop("missing_value", None)
#print("type da", type(da))
#print (da.dims)
#print (da.shape)

#print (da)

df=da.to_pandas()

#print("pandas: shape", getattr(df, 'shape', None))

#print (da)
#print (df)

#print (da.dtype)
#print (da.attrs)
#print (da.encoding)
#print ("All NaN?", np.isnan(da.values).all())
#print ("How many NaN?", np.isnan(df.values).sum())
#print ("How many finite?", np.isfinite(df.values).sum())
#print ("Any finite?", np.isfinite(da.values).any())
#print ("total elements", da.size)
#print ("NaNs total", np.isnan(da.values).sum())
#print (da.isnull().sum(dim="diameter").isel(time=slice(0, 10)))
#rint (da.isnull().sum(dim="time").isel(diameter=slice(0, 10)))

#from aerosol.functions import calc_conc_interp
dp_low = 2e-9
dp_high = 3e-9
neg_ion_conc_2_7 = af.calc_conc_interp(
 df,
 dp_low,
 dp_high)

#print ("calc.concent", neg_ion_conc_2_7)

diams = df.columns.astype(float).values

#print("min diameter:", np.min(diams))
#print("max diameter:", np.max(diams))

in_range_idx = np.where((diams > dp_low) & (diams < dp_high))[0]

expanded_idx = np.unique(np.concatenate([in_range_idx - 1, in_range_idx, in_range_idx + 1]))

    # Clip indices to stay within bounds of array
expanded_idx = expanded_idx[(expanded_idx >= 0) & (expanded_idx < len(diams))]

df_in_range = df.iloc[:,expanded_idx]

data_reso=1.0
nan_threshold=20.0
col_threshold=1

if np.any((df_in_range.isnull().mean(axis=0)>=nan_threshold).values):
    print(-999)

if np.any((df_in_range.isnull().mean(axis=1)>=col_threshold).values):
    print(-996) 

if (df.index.is_monotonic_increasing==False):
    print(-998)

if ((df.shape[0]/data_reso) < (23.0/data_reso)):
    print(-997) 

df_2024 = df.loc["2024"]
#df_small = df.dropna(axis=1, how="any")
#df_small = df_small.iloc[::100, :20]


#print(np.all(np.diff(df.columns.astype(float)) > 0))


gr_list = []

for date, df_day in df_2024.groupby(df_2024.index.date):

    # skip short days
    if len(df_day) < 24:
        continue

    gr = af.cross_corr_gr(
        df_day,
        dp_low,
        dp_high,
        smoothing_window=3.0,
        tau_window=23,
        col_threshold=1,
        row_threshold=1,
        number_of_divisions=1,
        data_reso=1.0,
        verbose=False
    )

    if isinstance(gr, dict):
        gr_list.append(gr["gr"])

#print (gr_list)

import matplotlib.pyplot as plt
gr_array = np.array(gr_list)
gr_array = gr_array[gr_array > 0]
gr_filtered = gr_array[gr_array >= 0.5]   # typical cutoff 0.3–0.5 nm/h


bins = np.logspace(
    np.log10(gr_array.min()),
    np.log10(gr_array.max()),
    50
)

plt.hist(gr_filtered, bins=bins, color="black")
plt.xscale("log")
plt.xlabel("GR (nm h$^{-1}$)")
plt.ylabel("Number of days")
plt.title("Daily growth rates 2024, Negative Particles at 3-7 nm")
plt.show()


plt.hist(gr_list, bins=20)
plt.xlabel("GR (nm/h)")
plt.ylabel("Number of days")
plt.title("Daily growth rates 2024")
#plt.show()
