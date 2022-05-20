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

########band stop filtering#########
'''
data: signal
cutoff_low:  lower-cut-off frequency of the filter
cutoff_high: upper cut-off frequency of the filter
sampling_rate: Sampling rate of signal
'''
def ButterBandstopFilter(data, cutoff_low, cutoff_high, sampling_rate): 
    b, a = signal.butter(4,[cutoff_low/(sampling_rate/2), cutoff_high/(sampling_rate/2)], btype='bandstop')
    Data = signal.filtfilt(b,a,data)
    
    return Data 
########band stop filtering#########

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

########LMS filtering#########
'''
input:：
Xn: reference signal
Dn: signal before filtering
order：order of filter
u：step of filter
ouput:
Yn: estimated reference signal related to Xn
error：signal after filtering
weight：weight of filter
'''
def LMSFiltering(Xn, Dn, order, u):
    itr = len(Xn)
    error=np.array(np.zeros(itr)) 
    weight=np.matrix(np.zeros((order, itr)))    
    
    i=order
    while i<itr:
        x=Xn[i-order:i]
        x=np.matrix(x[::-1])
        y=x * weight[:,i-1]        
        error[i] = Dn[i] - y[0, 0] 
        weight[:,i]=weight[:,i-1] + 2*u*error[i]*np.transpose(x)
        
        i=i+1
            
    Yn=np.array(np.zeros(itr))
    i=order
    while i<itr:
        x=Xn[i-order:i]
        x=np.matrix(x[::-1])
        y=x*weight[:,-1]
        Yn[i]=y[0, 0]
        
        i=i+1
        
    return Yn, error, weight
########LMS filtering#########

########NLMS filtering#########
'''
input:：
Xn: reference signal
Dn: signal before filtering
order：order of filter
u：step of filter
ouput:
Yn: estimated reference signal related to Xn
error：signal after filtering
weight：weight of filter
'''
def NLMSFiltering(Xn, Dn, order, n):
    itr = np.min([int(len(Dn)), int(len(Xn))])
    error=np.array(np.zeros(itr)) 
    weight=np.matrix(np.zeros((order, itr)))  
    
    i=order
    while i<itr:
        x=Xn[i-order:i]
        x=np.matrix(x[::-1])
        y=x * weight[:,i-1] 
        error[i] = Dn[i] - y[0, 0] 
        u = n / (x * x.transpose() + 0.00000001)
        weight[:,i]=weight[:,i-1] + u[0, 0]*error[i]*np.transpose(x)
        
        i=i+1
            
    Yn=np.array(np.zeros(itr))
    i=order
    while i<itr:
        x=Xn[i-order:i]
        x=np.matrix(x[::-1])
        y=x*weight[:,-1]
        Yn[i]=y[0, 0]
        
        i=i+1
          
    return Yn, error, weight
########NLMS filtering#########

########Function#########
'''
get the power
'''
def GerPowerFre(data, sampling_rate):
    freqs, power_spectrum = signal.periodogram(data, fs=sampling_rate, window='boxcar', nfft=None,  scaling='spectrum')
    
    return sum(power_spectrum)
########Function#########

########Function######### 
'''
get the power for every iterate
'''
def Get_Power_yn(Xn, weight, order, weight_idex, sampling_rate):
    itr = len(Xn)
    Yn=np.array(np.zeros(itr))
    i=order
    while i<itr:
        x=Xn[i-order:i]
        x=np.matrix(x[::-1])
        y=x*weight[:,weight_idex]
        Yn[i]=y[0, 0]
        
        i=i+1
    
    P_yn = GerPowerFre(Yn, sampling_rate)
    return P_yn
########Function#########

########Function#########
'''
normalization
'''
def Normalization(data_array):
    length = len(data_array)
    
    data_array_nor = []
    data_max = max(data_array)
    data_min = min(data_array)
    
    i = 0
    while i<length:
        data_array_nor.append(round((data_array[i]-data_min)/(data_max-data_min), 5))
        
        i = i + 1
    
    return data_array_nor    
########Function#########

########RLS filtering#########
'''

input:：
Xn: reference signal
Dn: signal before filtering
order: order of filter
forget_factor: forgetting factor
Delta: adjustment parameter
ouput:
Yn: estimated reference signal related to Xn
error：signal after filtering
weight：weight of filter
'''
def RLSFiltering(Xn, Dn, order, forget_factor, Delta):
    itr = len(Xn)
    error=np.array(np.zeros(itr)) 
    weight=np.matrix(np.zeros((order, itr)))   
    T = np.eye(order)/Delta
    i=order
    while i<itr:
        x=Xn[i-order:i]
        x=np.matrix(x[::-1])
        x_t = x.transpose()
        z=T * x_t 
        k=z/(forget_factor + x * z) 
        error[i]=Dn[i] - x * weight[:,i-1] 
        weight[:,i] = weight[:,i-1] + error[i] * k 
        T1 = k * z.transpose()
        T = (T - T1)/ forget_factor 
        
        i=i+1  
                  
    Yn=np.array(np.zeros(itr))
    i=order
    while i<itr:
        x=Xn[i-order:i]
        x=np.matrix(x[::-1])
        y=x*weight[:,-1]
        Yn[i]=y[0, 0]
        
        i=i+1
        
    return Yn, error, weight
########RLS filtering#########


