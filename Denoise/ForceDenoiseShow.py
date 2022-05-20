from pylab import *
from Denoise.ForceDenoiseFun import *
from FileImport.ReadData import *

'''
Show the effect of force signal denoising by three denoising methods
filename: full file path + file name, for example: r'D:\\DataSet\\DenoiseDataShow\\Signal.csv' 
'''
def ForceDenoiseShow(filename):
    sampling_rate = 1000
    force,Raw_1,Envelope_1,Raw_2,Envelope_2=LoadDataSet(filename)
    
    force_low_filtering = ButterLowFilter(force, 5, sampling_rate)
    force_median_filtering = MedianFilter(force)
    force_slidingmean_filter = SlidingMeanFiltering(force, 51)
    
    figure(1)
    plt.title('without denoising')
    plt.plot(range(len(force)),force,'k-',label='force')
    plt.legend()
    
    figure(2)
    plt.title('force_denoise')
    plt.plot(range(len(force)),force,'k-',label='force')
    plt.plot(range(len(force_low_filtering)),force_low_filtering,'r-',label='low')
    plt.plot(range(len(force_median_filtering)),force_median_filtering,'g-',label='median')
    plt.plot(range(len(force_slidingmean_filter)),force_slidingmean_filter,'y-',label='slidingmean')
    plt.legend()
    
    plt.show()
    

'''
input: r'D:\\DataSet\\DenoiseDataShow\\Signal.csv' 
'''    
ForceDenoiseShow(r'D:\\DataSet\\DenoiseDataShow\\Signal.csv')