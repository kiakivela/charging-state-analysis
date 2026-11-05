import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

#Load data
background=np.loadtxt("/home/kivekia/Downloads/measurement5_hist.txt", skiprows=1)
measurement1=np.loadtxt("/home/kivekia/Downloads/measurement1_hist.txt", skiprows=1)
measurement2=np.loadtxt("/home/kivekia/Downloads/measurement2_hist.txt", skiprows=1)
measurement3=np.loadtxt("/home/kivekia/Downloads/measurement3_hist.txt", skiprows=1)
measurement4=np.loadtxt("/home/kivekia/Downloads/measurement4_hist.txt", skiprows=1)

#Function for filtering data points with count=0
def filter_data(data):
    filtered_data=data[data[:,1] !=0]  #Filter out rows where the count is 0
    return filtered_data

#Filter measurement data
measurement1_ftd=filter_data(measurement1)
measurement2_ftd=filter_data(measurement2)
measurement3_ftd=filter_data(measurement3)
measurement4_ftd=filter_data(measurement4)
background_ftd=filter_data(background)


#Function for removing background noice using direct substraction
def direct_substraction(source_data, background_data):
    #Create dictionaries of the datasets  for easier data handling
    source_dict={height: count for height, count in source_data}
    background_dict={height:count for height, count in background_data}

    result_tuples=[]
    #for each height in the source+background data search for the corresponding height entry 
    #from the background dictionary and substract the count rate of that entry from the source
    #dictionary count entry
    for height, count1 in source_dict.items():
        count2=background_dict.get(height, 0)
        new_count=count1-count2
        if new_count>0:
            result_tuples.append((height, new_count))
        
    #Return the result as an array of tuples    
    result=np.array(result_tuples)
    return result

#remove background noise from datasets
result1=direct_substraction(measurement1_ftd, background_ftd)
result2=direct_substraction(measurement2_ftd, background_ftd)
result3=direct_substraction(measurement3_ftd, background_ftd)
result4=direct_substraction(measurement4_ftd, background_ftd)

#Filter out unneccessary data for a cleaner graph (most likely background noise)
result1=result1[result1[:,0]<5]
result2=result2[result2[:,0]<5]
result3=result3[result3[:,0]<5]
result4=result4[result4[:,0]<4]

#Create lists for plotting using a loop
results=[result1, result2, result3, result4]
titles=['3 plates, background removed', '6 plates, background removed','9 plates, background removed','12 plates, background removed']

fig, axes =plt.subplots(2,2, figsize=(14,10))
axes=axes.flatten()

#Number of bins in the histogram
nbins=50

#Create histograms for each dataset using a loop
for ax, result, title in zip(axes, results, titles):
    #Plot the histograms
    n, bins, patches = ax.hist(result[:,0], weights=result[:,1], bins=nbins, edgecolor='black', alpha=0.6, label='Data', density=False)

    #Separate the heights and countrates
    heights=result[:,0]
    counts=result[:,1]

    #Create an expanded dataset with each height is repeated according to its frequency
    expanded_data=np.repeat(heights, counts.astype(int))
    #Calculate the standard deviation using the expanded dataset (ddof = divide by N-1)
    poisson_std=np.std(expanded_data, ddof=1)
    #Calculate total counts
    N=np.sum(counts)
    #calculate expectancy values
    expect_value = np.sum(result[:, 0]*result[:, 1]) / N
    #Calculate the standard error of the mean
    error_mean=poisson_std/np.sqrt(N)

    #Create an large smooth array starting from 0 or min height value - 1 and ending at max height value + 1
    x=np.linspace(max(0, result[:, 0].min() - 1), result[:, 0].max() + 1, 1000)
    #Determine histogram bin width
    bin_width=bins[1]-bins[0] if len(bins) >1 else 1

    #Create an array of integer values based on x for poisson curve
    x_int=np.arange(int(np.floor(x[0])), int(np.ceil(x[-1]))+1)
    #Create a poisson curve using the poisson probability mass function
    poisson_curve=poisson.pmf(x_int,expect_value )*N*bin_width
    #Plot the poisson curve 
    ax.plot(x_int, poisson_curve, '-r', linewidth=2, label=f'Poisson fit')
    #Plot a coloured range showing the span of the standard deviation
    ax.axvspan(expect_value - poisson_std, expect_value + poisson_std, alpha=0.2, color='gray', label=f'±1σ span, σ={poisson_std:.3f}')
    #Plot a vertical line showing the expectancy value
    ax.axvline(expect_value, color='green', linestyle='--', alpha=0.7, label=f'Expectancy value (λ)= {expect_value:.3f} \n Standard error of the mean σ / √N = {error_mean:.3f}')
   
    #Set graph details and labels
    ax.set_ylabel('Number of Counts')
    ax.set_xlabel('Height of the signal')
    ax.set_title(title, fontsize=11, pad=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    ax.set_xlim(left=0)

#fig.suptitle('Poisson Distribution of Photons Emitted from Radioactive Decay', fontsize=16)
plt.tight_layout()
plt.show()