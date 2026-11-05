import xarray as xr
import pandas as pd
import aerosol.functions as af

ds = xr.open_dataset("/home/kivekia/Python/kumpula_1h_checked.nc")

df_negative_ions = ds.pos_particles.to_pandas()
print (df_negative_ions)

from aerosol.functions import calc_conc_interp
dp_low = 2e-9
dp_high = 7e-9
neg_ion_conc_2_7 = calc_conc_interp(
 df_negative_ions,
 dp_low,
 dp_high)

print (neg_ion_conc_2_7)

# Remove diameter bins that contain NaNs anywhere
df_no_nan_cols = df_negative_ions.dropna(axis=1, how="any")

print("Original shape:", df_negative_ions.shape)
print("After dropping NaN columns:", df_no_nan_cols.shape)

gr = af.cross_corr_gr(
    df_no_nan_cols,
    1, 10,
    number_of_divisions=10,
    verbose=True
)

print(gr)

ds.close()

