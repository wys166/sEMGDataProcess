import scipy
from scipy import signal
import numpy as np
import os
import math

'''
Function
'''

########Low pass filtering#########
'''
data: signal
cutoff: Cutoff frequency of the filter
sampling_rate: Sampling rate of signal
'''
def ButterLowFilter(data, cutoff, sampling_rate): 
    b, a = signal.butter(5, cutoff/(sampling_rate/2), 'low')
    Data = signal.filtfilt(b,a,data)
    
    return Data 
########Low pass filtering#########

########median filtering#########
def MedianFilter(data): 
    Data = signal.medfilt(data, kernel_size=5)
    
    return Data 
########median filtering#########

########Moving mean filtering#########
'''
window:length of moving window, it must be odd number
'''
def SlidingMeanFiltering(data, window):
    if window%2 == 0:
        print("window must be an odd number!")
        return 
    
    length=len(data)
    newdata=[]
    
    half_window = np.int((window-1)/2)
    i=0
    while i<length:
        if i<half_window:
            newdata.append(data[i])  
        elif i<length-half_window:
            newdata.append(np.mean(data[i - half_window:i + half_window]))    
        else:
            newdata.append(data[i]) 
        
        i=i+1
      
    return newdata
########Moving mean filtering######### 




    

