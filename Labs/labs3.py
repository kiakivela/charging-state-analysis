import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

#Measurement 1
vib_1=4865 #Intensity of the vibration measurement
vib_1_std=586.6 #Uncertanty of intensity
calibration1=np.array([5951, 5359, 4816, 4267, 4182, 3617, 3199, 2726, 2171, 1636]) #Intensities from the calibration measurements
calibration1_std=np.array([726.5, 623.5, 560.4, 489.9, 488.1, 432.8, 382.3, 324.9, 265.0, 194.9]) #Calibration intensity uncertanties
displacement_1=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) #mm

#Create interpolation curves  using Scipy library's interpolate module
curve1=interpolate.interp1d(displacement_1, calibration1, kind='linear') 
inverse_curve1=interpolate.interp1d(calibration1, displacement_1, kind='linear') #Inverse interpolation
x1=np.linspace(displacement_1.min(), displacement_1.max(), 500) #Large arays for smooth curves
y1=curve1(x1)

#Measurement 2
vib_2=5011
vib_2_std=618.4
calibration2=np.array([8616, 8296, 7893, 7447, 6967, 6475, 5935, 5439, 4910, 4413, 3832])
calibration2_std=np.array([970.9, 933.6, 904.6, 856.1, 805.2, 769.9, 719.3, 639.3, 564.1, 512.6, 445.9])
displacement_2=np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

curve2=interpolate.interp1d(displacement_2, calibration2, kind='linear')
inverse_curve2=interpolate.interp1d(calibration2, displacement_2, kind='linear')
x2=np.linspace(displacement_2.min(), displacement_2.max(), 500)
y2=curve2(x2)

#Measurement 3
vib_3=4886
vib_3_std=713.3
calibration3=np.array([8701, 8405, 8029, 7578, 7132, 6629, 6170, 5625, 5115, 4597, 4000, 3439])
calibration3_std=np.array([972.0, 970.5, 920.5, 860.4, 842.6, 784.1, 727.4, 676.4, 603.3, 531.0, 479.7, 405.3])
displacement_3=np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])

curve3=interpolate.interp1d(displacement_3, calibration3, kind='linear')
inverse_curve3=interpolate.interp1d(calibration3, displacement_3, kind='linear')
x3=np.linspace(displacement_3.min(), displacement_3.max(), 500)
y3=curve3(x3)

#Gather all data into lists for easier plotting
calibrations=[calibration1, calibration2, calibration3]
displacements=[displacement_1, displacement_2, displacement_3]
vibration_data=[vib_1, vib_2, vib_3]
vib_error=[vib_1_std, vib_2_std, vib_3_std]
inv_extpol=[inverse_curve1(vib_1), inverse_curve2(vib_2), inverse_curve3(vib_3)]
x_data=[x1, x2, x3]
y_error=[calibration1_std, calibration2_std, calibration3_std]
y_data=[y1, y2, y3]
vib_disp_errors=[]

#Calculate displacement error of the vibration measurements using error propagation: σ_disp = |slope| × σ_intensity
for i in range(3):
    vib_intensity=vibration_data[i]
    vib_intensity_error=vib_error[i]

    cal=calibrations[i]
    disp=displacements[i]

    #Find where the vibration intensity falls in the calibration data
    index=np.searchsorted(cal, vib_intensity)
 
    #Handle edge cases
    if index==0:
        #Use first two points to calculate the slope
        slope=(disp[1] - disp[0]) / (cal[1] - cal[0])
    elif index >= len(cal):
        #Use last two points to calculate the slope
        slope=(disp[-1] - disp[-2]) / (cal[-1] - cal[-2])
    else:
        #Use surrounding points to calculate the slope
        slope=(disp[index] - disp[index-1]) / (cal[index] - cal[index-1])                                 
    
    #Calculate displacement error    
    disp_error=(1/abs(slope))*vib_intensity_error  
    vib_disp_errors.append(disp_error)

#Parameters for easier plotting
labels=['Calibration 1 data', 'Calibration 2 data', 'Calibration3 data']
line_formats=['--b', '--g', '--r']
colors=['blue', 'green', 'red']
e_colors=['lightblue', 'lightgreen', 'pink']
labels=['Calibration 1 data ± σ', 'Calibration 2 data ± σ', 'Calibration 3 data ± σ',
         f'Vibration 1 ± σ, displacement={inverse_curve1(vib_1):.2f} ± {vib_disp_errors[0]:.2f} mm',
         f'Vibration 2 ± σ, displacement={inverse_curve2(vib_2):.2f} ± {vib_disp_errors[1]:.2f} mm', 
         f'Vibration 3 ± σ, displacement={inverse_curve3(vib_3):.2f} ± {vib_disp_errors[2]:.2f} mm']

#Plot all the data using a loop for easy plotting
fig, ax = plt.subplots(figsize=(12,8))
for i in range(3):
    ax.errorbar(displacements[i], calibrations[i], yerr=y_error[i], fmt='o', color=colors[i], ecolor=e_colors[i], elinewidth=1.5, 
                capsize =4, capthick=1.5, markersize=8, markeredgecolor='white', zorder=0.6, 
                label=labels[i]) #Plot calibration data and its uncertanties
    ax.plot(x_data[i], y_data[i], line_formats[i]) #Plot interpolation curve
    ax.errorbar(inv_extpol[i], vibration_data[i], xerr=vib_disp_errors[i], fmt='D', color=colors[i], markeredgecolor='white',
                 markersize=10, capsize=4, label=labels[i+3]) #Plot vibration measurement data using extrapolation

ax.set_xlabel('Displacement (mm)', fontsize=14)
ax.set_ylabel('Intensity (lx)', fontsize=14)
ax.legend()
plt.title("Calibration curves and the bending of a vibrating tuning fork as displacement")
plt.tight_layout()
plt.show()