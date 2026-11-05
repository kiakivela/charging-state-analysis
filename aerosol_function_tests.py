import xarray as xr
import numpy as np
import pandas as pd
import aerosol.functions as af
import importlib.metadata
import matplotlib.pyplot as plt
import aerosol.plotting as ap
import glob

f= af.eq_charge_frac(2e-9, -1)

print(f)