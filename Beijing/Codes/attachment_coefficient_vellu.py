import xarray as xr
import numpy as np
import pandas as pd
import aerosol.functions as af
import matplotlib.pyplot as plt
import math

a = 1.6e-6
d0 = 1e-9
a0 = 0.006e-6
l = 1
l_2 = 1.2
l_3 = 1.5
dp_nm = np.arange(1, 101)*1e-9

f_eq_1 = pd.Series(
    {
        diameter: (a0*(diameter / d0)**l)
        for diameter in dp_nm
    }
)

f_eq_2 = pd.Series(
    {
        diameter: (a0*(diameter / d0)**l_2)
        for diameter in dp_nm
    }
)

f_eq_3 = pd.Series(
    {
        diameter: (a0*(diameter / d0)**l_3)
        for diameter in dp_nm
    }
)

f_eq_4 = pd.Series(
    {
        diameter: (10**(-0.0774*(math.log(diameter, 10))**3-1.7933*(math.log(diameter, 10)**2)-12.486*math.log(diameter, 10)-31.755))
        for diameter in dp_nm
    }
)

# Diameter in nm for plotting
diam_nm = np.arange(1, 101)

fig, ax1 = plt.subplots(figsize=(8, 5))

# Left axis
#line1 = ax1.plot(
#    diam_nm,
#    f_eq_1.values,
#    label="Attachment coefficient, l = 1"
#)

line2 = ax1.plot(
    diam_nm,
    f_eq_2.values,
    label="Attachment coefficient, l = 1.2"
)

#line3 = ax1.plot(
#    diam_nm,
#    f_eq_3.values,
#    label="Attachment coefficient, l = 1.5"
#)

line4 = ax1.plot(
    diam_nm,
    f_eq_4.values,
    label="Attachment coefficient, polynomial = 3"
)

ax1.set_xlabel("Diameter (nm)")
ax1.set_ylabel(r"Attachment Coefficient (cm$^{3}$ s$^{-1}$)")
ax1.set_xscale("log")
ax1.set_yscale("log")
#ax1.set_xlim(2e-9, 20e-9)
#ax1.set_ylim(-0.05, 0.9)

# Combined legend
handles1, labels1 = ax1.get_legend_handles_labels()

ax1.legend(
    handles1,
    labels1,
    loc="best",
)

plt.tight_layout()
plt.show()