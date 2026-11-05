import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

#Vibration data = [Measurement 1, Measurement 2, Measurement 3]
vibration_data=[4865, 5011, 4886]
#Vibration errors (standard deviations) = [Measurement 1, Measurement 2, Measurement 3]
vib_error=[586.6, 618.4, 713.3]

#Calibration data, intensity = [Measurement 1, Measurement 2, Measurement 3]
calibrations=[np.array([5951, 5359, 4816, 4267, 4182, 3617, 3199, 2726, 2171, 1636]),
    np.array([8616, 8296, 7893, 7447, 6967, 6475, 5935, 5439, 4910, 4413, 3832]),
    np.array([8701, 8405, 8029, 7578, 7132, 6629, 6170, 5625, 5115, 4597, 4000, 3439])]

#Calibration data, intensity errors (standard deviations) = [Measurement 1, Measurement 2, Measurement 3]
calibration_err=[np.array([726.5, 623.5, 560.4, 489.9, 488.1, 432.8, 382.3, 324.9, 265.0, 194.9]),
    np.array([970.9, 933.6, 904.6, 856.1, 805.2, 769.9, 719.3, 639.3, 564.1, 512.6, 445.9]),
    np.array([972.0, 970.5, 920.5, 860.4, 842.6, 784.1, 727.4, 676.4, 603.3, 531.0, 479.7, 405.3])]

#Calibration data, displacement = [Measurement 1, Measurement 2, Measurement 3]
displacements=[np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]),
    np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]),
    np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1])]

#(a, m):
fit_params=[]  
#(sigma_a, sigma_m):      
fit_errors=[]    

vib_disp=[]
vib_disp_errors=[]

x_plot=[]
y_plot=[]

#Loop for calibration curves
for i in range(3):
    x=displacements[i]
    I=calibrations[i]
    sigma_I=calibration_err[i]

    #Weighted linear fit: I=a+mx
    p, cov=np.polyfit(x, I, 1, w=1/sigma_I, cov=True)
    m, a=p
    sigma_m=np.sqrt(cov[0,0])
    sigma_a=np.sqrt(cov[1,1])

    fit_params.append((a, m))
    fit_errors.append((sigma_a, sigma_m))

    #Smooth line for plotting
    x_smooth=np.linspace(x.min(), x.max(), 500)
    y_smooth=a+m*x_smooth
    x_plot.append(x_smooth)
    y_plot.append(y_smooth)

    #Displacement from vibration intensity
    I_vib=vibration_data[i]
    sigma_I_vib=vib_error[i]
    x_v=(I_vib-a)/m

    #Error propagation
    sigma_x=np.sqrt(
        (sigma_I_vib/m)**2+
        (sigma_a/m)**2+
        ((I_vib-a)*sigma_m/m**2)**2)

    vib_disp.append(x_v)
    vib_disp_errors.append(sigma_x)

#Print results
for i in range(3):
    print(f"Measurement {i+1}:")
    print(f"Displacement = {vib_disp[i]:.3f} ± {vib_disp_errors[i]:.3f} mm")

average_disp=sum(vib_disp)/3
average_error=np.sqrt(0.128**2+0.126**2+0.145**2)/3
print(f"Average: {average_disp:.3f} ± {average_error:.3f} mm")

#Plot the results
colors=['blue','green','red']
e_colors=['lightblue','lightgreen','pink']

labels=[
    'Calibration 1 ± σ',
    'Calibration 2 ± σ',
    'Calibration 3 ± σ',
    f'Vibration 1: d={vib_disp[0]:.2f} ± {vib_disp_errors[0]:.2f} mm',
    f'Vibration 2: d={vib_disp[1]:.2f} ± {vib_disp_errors[1]:.2f} mm',
    f'Vibration 3: d={vib_disp[2]:.2f} ± {vib_disp_errors[2]:.2f} mm']

fig, ax=plt.subplots(figsize=(12,8))

for i in range(3):
    #Calibration points
    ax.errorbar(displacements[i], calibrations[i],
                yerr=calibration_err[i],
                fmt='o', color=colors[i], ecolor=e_colors[i],
                capsize=4, markersize=7, label=labels[i])

    #Fit line
    ax.plot(x_plot[i], y_plot[i], '--', color=colors[i])

    #Vibration point
    ax.errorbar(vib_disp[i], vibration_data[i],
                xerr=vib_disp_errors[i],
                fmt='D', color=colors[i],
                markersize=9, capsize=4,
                label=labels[i+3])

ax.set_xlabel('Displacement (mm)', fontsize=14)
ax.set_ylabel('Intensity (lx)', fontsize=14)
ax.set_title('Tuning Fork', fontsize=15)
ax.legend()
plt.tight_layout()
plt.show()