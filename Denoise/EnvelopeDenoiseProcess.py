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



def EnvelopeDenoiseShow(filename):
    sampling_rate = 1000
    force,Raw_1,Envelope_1,Raw_2,Envelope_2=LoadDataSetAfterProcess(filename)
    
    envelope_low_filtering = ButterLowFilter(Envelope_1, 10, sampling_rate)
    envelope_median_filtering = MedianFilter(Envelope_1)
    envelope_slidingmean_filter = SlidingMeanFiltering(Envelope_1, 51)
    
    
    ##############功率谱################
    sig_dn=Envelope_1[1000:7500]
    fft_size=len(sig_dn)
    freqs1=np.linspace(0, sampling_rate/2, int(fft_size/2+1))    
    xf_dn=np.fft.rfft(sig_dn)/fft_size
    xf_dn=abs(xf_dn)
    
    
    freqs, power_spectrum_envelope1  = signal.periodogram(Envelope_1[1000:7500], fs=sampling_rate, window='boxcar', nfft=None,  scaling='spectrum')
    freqs, power_spectrum_envelope1_filter  = signal.periodogram(envelope_slidingmean_filter[1000:7500], fs=sampling_rate, window='boxcar', nfft=None,  scaling='spectrum')
    
    ##############功率谱################
    
    figure(1)
    plt.title('Envelope_1')
    plt.plot(range(len(Envelope_1)),Envelope_1,'k-',label='Envelope_1')
    plt.plot(range(len(envelope_low_filtering)),envelope_low_filtering,'r-',label='low')
    plt.plot(range(len(envelope_median_filtering)),envelope_median_filtering,'g-',label='median')
    plt.plot(range(len(envelope_slidingmean_filter)),envelope_slidingmean_filter,'y-',label='slidingmean')
    plt.legend()
    
    
    envelope_slidingmean_filter2 = SlidingMeanFiltering(envelope_median_filtering, 51)
    
    figure(2)
    plt.title('Envelope_1')
    plt.plot(range(len(Envelope_1)),Envelope_1,'k-',label='Envelope_1')
    plt.plot(range(len(envelope_low_filtering)),envelope_low_filtering,'r-',label='low')
    plt.plot(range(len(envelope_median_filtering)),envelope_median_filtering,'g-',label='median')
    plt.plot(range(len(envelope_slidingmean_filter2)),envelope_slidingmean_filter2,'y-',label='slidingmean')
    plt.legend()
    
    figure(3)
    plt.title('Fre_Envelope_1')
    plt.plot(freqs[1:],power_spectrum_envelope1[1:],'k-',label='beforefilter')
    plt.legend()
    
    figure(4)
    plt.title('Fre_Envelope_1')
    plt.plot(freqs[1:],power_spectrum_envelope1_filter[1:],'r-',label='afterfilter')
    plt.legend()
    
    figure(5)
    plt.title('Fre_Envelope_1')
    plt.plot(freqs1[1:],xf_dn[1:],'k-',label='beforefilter')
    plt.legend()
    
    plt.show()
    