import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Load data
# ---------------------------
# Replace with your actual file names
background = np.loadtxt("/home/kivekia/Downloads/measurement5_hist.txt", skiprows=1)

background_plus_source = np.loadtxt("/home/kivekia/Downloads/measurement1_hist.txt", skiprows=1)

# Match lengths
min_len = min(len(background), len(background_plus_source))

background = background[:min_len]
background_plus_source = background_plus_source[:min_len]

source_only = background_plus_source - background

source_only[source_only < 0] = 0

# ---------------------------
# Plot results
# ---------------------------
plt.figure()
#plt.plot(background, label="Background")
#plt.plot(background_plus_source, label="Background + Source")
plt.plot(source_only, label="Source Only")
plt.legend()
plt.xlabel("Channel / Bin")
plt.ylabel("Counts (normalized)")
plt.title("Radiation Signal Separation")
plt.show()

# ---------------------------
# Save result
# ---------------------------
np.savetxt("source_only.txt", source_only)

