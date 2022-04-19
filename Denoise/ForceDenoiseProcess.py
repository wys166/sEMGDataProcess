import scipy
from scipy import signal
import numpy as np
import os
import math

from FileImport.ReadData import *


########巴特沃斯低通滤波器#########
def ButterLowFilter(data, cutoff, sampling_rate): ####data傅里叶变换的原始数据,cutoff低通滤波器截止频率，sampling_rate采样频率
    b, a = signal.butter(5, cutoff/(sampling_rate/2), 'low')
    Data = signal.filtfilt(b,a,data)
    
    return Data #返回滤波后的数据
########巴特沃斯低通滤波器#########

########巴特沃斯带阻滤波器#########
def ButterBandstopFilter(data, cutoff_low, cutoff_high, sampling_rate): ####data傅里叶变换的原始数据,cutoff低通滤波器截止频率，sampling_rate采样频率
    b, a = signal.butter(4,[cutoff_low/(sampling_rate/2), cutoff_high/(sampling_rate/2)], btype='bandstop')###########限波范围为cutoff_low-cutoff_high的带阻滤波器阶数不能超过5阶
    Data = signal.filtfilt(b,a,data)
    
    return Data #返回滤波后的数据
########巴特沃斯带阻滤波器#########

########中值滤波#########
def MedianFilter(data): ####data傅里叶变换的原始数据,cutoff低通滤波器截止频率，sampling_rate采样频率
    Data = signal.medfilt(data, kernel_size=5)
    
    return Data #返回滤波后的数据
########中值滤波#########

########滑动平均滤波#########
'''
window:窗口长度，取奇数
'''
def SlidingMeanFiltering(data, window):
    if window%2 == 0:
        print("滑动均值滤波无效，窗口长度不是奇数！")
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
########滑动平均滤波######### 



def ForceDenoiseShow(filename):
    sampling_rate = 1000
    force,Raw_1,Envelope_1,Raw_2,Envelope_2=LoadDataSetAfterProcess(filename)
    
    force_low_filtering = ButterLowFilter(force, 5, sampling_rate)
    force_median_filtering = MedianFilter(force)
    force_slidingmean_filter = SlidingMeanFiltering(force, 51)
    
    
    figure(1)
    plt.title('force_denoise')
    plt.plot(range(len(force)),force,'k-',label='force')
    plt.plot(range(len(force_low_filtering)),force_low_filtering,'r-',label='low')
    plt.plot(range(len(force_median_filtering)),force_median_filtering,'g-',label='median')
    plt.plot(range(len(force_slidingmean_filter)),force_slidingmean_filter,'y-',label='slidingmean')
    plt.legend()
    
    plt.show()
    

