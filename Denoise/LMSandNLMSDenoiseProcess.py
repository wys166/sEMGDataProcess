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
    Data = signal.medfilt(data, kernel_size=None)
    
    return Data #返回滤波后的数据
########中值滤波#########

########滑动平均滤波#########
def SlidingMeanFiltering(data, window):
    length=len(data)
    newdata=[]
    
    i=0
    while i<length:
        if i<window-1:
            newdata.append(data[i])  
        else:
            newdata.append(np.mean(data[i+1-window:i+1]))      
        
        i=i+1
        
    return newdata
########滑动平均滤波#########

########自适应滤波噪声抵消算法LMS#########
'''
输入：
Xn:FIR滤波器输入参考信号，是在没有信号时的背景噪声
Dn:期望信号，噪声和有用信号叠加后的信号
order：滤波器的阶数
u：每次迭代的步长
输出：
Yn:最优滤波器的输出序列,即通过FIR滤波器输出的最接近信号噪声序列
error：误差输出序列
weight：权值
'''
def LMSFiltering(Xn, Dn, order, u):
    itr = len(Xn)
    error=array(np.zeros(itr)) #误差序列,error[k]表示第k次迭代时预期输出与实际输入的误差
#     print(size(error,0))
    weight=np.matrix(np.zeros((order, itr)))#权值    
    
    i=order
    while i<itr:#迭代计算
        x=Xn[i-order:i]
        x=np.matrix(x[::-1])#order阶滤波器抽头输入
        y=x * weight[:,i-1]#滤波器输出        
#         print(y[0, 0])
        error[i] = Dn[i] - y[0, 0] #计算误差
        weight[:,i]=weight[:,i-1] + 2*u*error[i]*np.transpose(x)#滤波器权值计算的迭代式
        
        i=i+1
            
    Yn=array(np.zeros(itr))
    i=order
    while i<itr:
        x=Xn[i-order:i]
        x=np.matrix(x[::-1])#order阶滤波器抽头输入
        y=x*weight[:,-1]
        Yn[i]=y[0, 0]
        
        i=i+1
        
    return Yn, error, weight
########自适应滤波噪声抵消算法LMS#########

########应用自适应滤波噪声抵消算法LMS，显示滤波效果#########
def LMSFilterShow(filename1, filename2):    
    sampling_rate=1000
    Force,Force_index,Raw_1,Envelope_1,Raw_2,Envelope_2=LoadRawDataSet(filename1)
    raw_1,envelope_1,raw_2,envelope_2 = LoadRawDataSetByChNum(filename2, CH=4)
    
    start_index = 0
    end_index = 100000  #59400    
    dn=Raw_1[start_index:end_index]
    xn=raw_1[0:end_index-start_index]
    print(len(dn))
    print(len(xn))
    
    #9阶，0.0000053的步长修正因子下，信噪比能达到45.0492
    #10阶，0.00000479的步长修正因子下，信噪比能达到53.5609
    #11阶，0.00000464的步长修正因子下，信噪比能达到53.9782
    yn, error, weight=LMSFiltering(xn, dn, 9, 0.0000053)   
    print(len(dn))
    print(len(error))
    noise = array(dn) - array(error)
    
    r_t, p_value_t = scipy.stats.pearsonr(xn, yn)
    print("时域内LMS算法下噪声相关性:{}".format(r_t))
    print("时域内LMS算法下p值:{}".format(p_value_t))
    
    ##############功率谱################
    freqs, power_spectrum_dn  = signal.periodogram(dn, fs=sampling_rate, window='boxcar', nfft=None,  scaling='spectrum')
    freqs, power_spectrum_xn  = signal.periodogram(xn, fs=sampling_rate, window='boxcar', nfft=None,  scaling='spectrum')
    freqs, power_spectrum_error  = signal.periodogram(error, fs=sampling_rate, window='boxcar', nfft=None,  scaling='spectrum')
    freqs, power_spectrum_yn  = signal.periodogram(yn, fs=sampling_rate, window='boxcar', nfft=None,  scaling='spectrum')
    ##############功率谱################
    
    r_f, p_value_f = scipy.stats.pearsonr(power_spectrum_xn, power_spectrum_yn)#原噪声与估计噪声之间的相关性
    print("频域内LMS算法下噪声相关性:{}".format(r_f))
    print("频域内LMS算法下p值:{}".format(p_value_f))
    
    figure(1)
    plt.plot(range(len(dn)),dn,'r-',label='dn')
    plt.legend()
    figure(2)
    plt.plot(range(len(xn)),xn,'r.-',label='xn')
    plt.legend()
    figure(3)
    plt.plot(range(len(yn)),yn,'r.-',label='yn')
    plt.legend()
    figure(4)
    plt.plot(range(len(error)),error,'b-',label='error')
    plt.legend()
    figure(5)
    plt.plot(range(len(dn)),dn,'r-',label='dn')
    plt.plot(range(len(error)),error,'b-',label='error')
    plt.plot(range(len(xn)),xn,'g-',label='xn')
    plt.plot(range(len(yn)),yn,'k-',label='yn')
    plt.legend()
    figure(6)
    plt.plot(range(len(noise)),noise,'r.-',label='noise')
    plt.legend()
    
    
    '''
    figure(0)
    plt.plot(range(len(dn)),dn,'r-',label='dn')
    plt.legend()    
    figure(1)
    plt.plot(freqs[0:-1],power_spectrum_xn[0:-1],'r.-',label='power_spectrum_xn')
    plt.legend()
    figure(2)
    plt.plot(freqs[0:],power_spectrum_dn[0:],'r.-',label='power_spectrum_dn')
    plt.legend()
    figure(3)
    plt.plot(freqs[0:],power_spectrum_error[0:],'r.-',label='power_spectrum_error')
    plt.legend()
    figure(4)
    plt.plot(freqs[0:],power_spectrum_yn[0:],'r.-',label='power_spectrum_yn')
    plt.legend()   
    '''
    
    plt.show()
########应用自适应滤波噪声抵消算法LMS，显示滤波效果#########


########自适应滤波噪声抵消算法NLMS#########
'''
输入：
Xn:FIR滤波器输入参考信号，是在没有信号时的背景噪声
Dn:期望信号，噪声和有用信号叠加后的信号
order：滤波器的阶数
n：修正的步长常量，0<n<2
输出：
Yn:最优滤波器的输出序列,即通过FIR滤波器输出的最接近信号噪声序列
error：误差输出序列
weight：权值
'''
def NLMSFiltering(Xn, Dn, order, n):
    itr = np.min([int(len(Dn)), int(len(Xn))])
    error=array(np.zeros(itr)) #误差序列,error[k]表示第k次迭代时预期输出与实际输入的误差
#     print(size(error,0))
    weight=np.matrix(np.zeros((order, itr)))#权值    
    
    i=order
    while i<itr:#迭代计算
        x=Xn[i-order:i]
        x=np.matrix(x[::-1])#order阶滤波器抽头输入
        y=x * weight[:,i-1]#滤波器输出 
      
#         print(y[0, 0])
        error[i] = Dn[i] - y[0, 0] #计算误差
        u = n / (x * x.transpose() + 0.00000001)
#         print(u[0, 0])
        weight[:,i]=weight[:,i-1] + u[0, 0]*error[i]*np.transpose(x)#滤波器权值计算的迭代式
        
        i=i+1
            
    Yn=array(np.zeros(itr))
    i=order
    while i<itr:
        x=Xn[i-order:i]
        x=np.matrix(x[::-1])#order阶滤波器抽头输入
        y=x*weight[:,-1]
        Yn[i]=y[0, 0]
        
        i=i+1
          
    return Yn, error, weight
########自适应滤波噪声抵消算法NLMS#########

########应用自适应滤波噪声抵消算法NLMS，显示滤波效果#########
def NLMSFilterShow(filename1, filename2):    
    sampling_rate=1000
    Force,Force_index,Raw_1,Envelope_1,Raw_2,Envelope_2=LoadRawDataSet(filename1)
    raw_1,envelope_1,raw_2,envelope_2 = LoadRawDataSetByChNum(filename2, CH=4)
        
    start_index = 0
    end_index = 100000    
    dn=Raw_1[start_index:end_index]
    xn=raw_1[0:end_index-start_index]
    
    #原信号信噪比为28.9798 
    #9阶，0.0052的步长修正因子下，信噪比能达到47.8067
    #10阶，0.00605的步长修正因子下，信噪比能达到53.4643
    #11阶，0.00716的步长修正因子下，信噪比能达到53.5039
    yn, error, weight=NLMSFiltering(xn, dn, 9, 0.0052)  
    noise = array(dn) - array(error)
    
    r_t, p_value = scipy.stats.pearsonr(xn, yn)
    print("时域内NLMS算法下噪声相关性:{}".format(r_t))
    
    ##############功率谱################
    freqs, power_spectrum_dn  = signal.periodogram(dn, fs=sampling_rate, window='boxcar', nfft=None,  scaling='spectrum')
    freqs, power_spectrum_xn  = signal.periodogram(xn, fs=sampling_rate, window='boxcar', nfft=None,  scaling='spectrum')
    freqs, power_spectrum_error  = signal.periodogram(error, fs=sampling_rate, window='boxcar', nfft=None,  scaling='spectrum')
    freqs, power_spectrum_yn  = signal.periodogram(yn, fs=sampling_rate, window='boxcar', nfft=None,  scaling='spectrum')
    ##############功率谱################
    
    r_f, p_value = scipy.stats.pearsonr(power_spectrum_xn, power_spectrum_yn)#原噪声与估计噪声之间的相关性
    print("频域内NLMS算法下噪声相关性:{}".format(r_f))
    
    
    '''
    figure(1)
    plt.plot(range(len(dn)),dn,'r-',label='dn')
    plt.legend()
    figure(2)
    plt.plot(range(len(xn)),xn,'r.-',label='xn')
    plt.legend()
    figure(3)
    plt.plot(range(len(yn)),yn,'r-',label='yn')
    plt.legend()
    figure(4)
    plt.plot(range(len(error)),error,'b-',label='error')
    plt.legend()
    figure(5)
    plt.plot(range(len(dn)),dn,'r-',label='dn')
    plt.plot(range(len(error)),error,'b-',label='error')
    plt.plot(range(len(xn)),xn,'g-',label='xn')
    plt.plot(range(len(yn)),yn,'k-',label='yn')
    plt.legend()
    figure(6)
    plt.plot(range(len(noise)),noise,'r.-',label='noise')
    plt.legend()
    '''
    
    
    figure(0)
    plt.plot(range(len(dn)),dn,'r-',label='dn')
    plt.legend()    
    figure(1)
    plt.plot(freqs[0:-1],power_spectrum_xn[0:-1],'r.-',label='power_spectrum_xn')
    plt.legend()
    figure(2)
    plt.plot(freqs[0:],power_spectrum_dn[0:],'r.-',label='power_spectrum_dn')
    plt.legend()
    figure(3)
    plt.plot(freqs[0:],power_spectrum_error[0:],'r.-',label='power_spectrum_error')
    plt.legend()
    figure(4)
    plt.plot(freqs[0:],power_spectrum_yn[0:],'r.-',label='power_spectrum_yn')
    plt.legend()
#     figure(5)
#     plt.plot(freqs[0:],power_spectrum_dn[0:],'r.',label='power_spectrum_dn')
#     plt.plot(freqs[0:],power_spectrum_error[0:],'b.',label='power_spectrum_error')
#     plt.legend()
    
    
    plt.show()
########应用自适应滤波噪声抵消算法NLMS，显示滤波效果#########
 
    
########LMS与NLMS两种算法下得到的噪声yn在频域内的相关性比较#########    
'''
FIR滤波器的阶数分别在9、10、11阶下
'''
def CorrelationShow_Compare_LMS_NLMS(filename1, filename2): 
    sampling_rate=1000
    Force,Force_index,Raw_1,Envelope_1,Raw_2,Envelope_2=LoadRawDataSet(filename1)
    raw_1,envelope_1,raw_2,envelope_2 = LoadRawDataSetByChNum(filename2, CH=4)
        
    start_index = 0
    end_index = 59400    
    dn=Raw_1[start_index:end_index]
    xn=raw_1[0:end_index-start_index]
    
    freqs, power_spectrum_dn  = signal.periodogram(dn, fs=sampling_rate, window='boxcar', nfft=None,  scaling='spectrum')
    freqs, power_spectrum_xn  = signal.periodogram(xn, fs=sampling_rate, window='boxcar', nfft=None,  scaling='spectrum')
    
    #############NLMS算法##############
    yn_11, error_11, weight=NLMSFiltering(xn, dn, 11, 0.00716)  
    yn_10, error_10, weight=NLMSFiltering(xn, dn, 10, 0.00605)
    yn_9, error_9, weight=NLMSFiltering(xn, dn, 9, 0.0052)

    freqs, power_spectrum_yn_11  = signal.periodogram(yn_11, fs=sampling_rate, window='boxcar', nfft=None,  scaling='spectrum')
    freqs, power_spectrum_yn_10  = signal.periodogram(yn_10, fs=sampling_rate, window='boxcar', nfft=None,  scaling='spectrum')
    freqs, power_spectrum_yn_9  = signal.periodogram(yn_9, fs=sampling_rate, window='boxcar', nfft=None,  scaling='spectrum')

    r_11, p_value = scipy.stats.pearsonr(power_spectrum_xn, power_spectrum_yn_11)
    r_10, p_value = scipy.stats.pearsonr(power_spectrum_xn, power_spectrum_yn_10)
    r_9, p_value = scipy.stats.pearsonr(power_spectrum_xn, power_spectrum_yn_9)
    
    r_NLMS = [r_9, r_10, r_11]
    print("频域内NLMS算法下噪声相关性:{}".format(r_NLMS))
    #############NLMS算法##############
    
    #############LMS算法##############
    yn_11, error_11, weight=LMSFiltering(xn, dn, 11, 0.00000464)  
    yn_10, error_10, weight=LMSFiltering(xn, dn, 10, 0.00000479)
    yn_9, error_9, weight=LMSFiltering(xn, dn, 9, 0.0000053)

    freqs, power_spectrum_yn_11  = signal.periodogram(yn_11, fs=sampling_rate, window='boxcar', nfft=None,  scaling='spectrum')
    freqs, power_spectrum_yn_10  = signal.periodogram(yn_10, fs=sampling_rate, window='boxcar', nfft=None,  scaling='spectrum')
    freqs, power_spectrum_yn_9  = signal.periodogram(yn_9, fs=sampling_rate, window='boxcar', nfft=None,  scaling='spectrum')

    r_11, p_value = scipy.stats.pearsonr(power_spectrum_xn, power_spectrum_yn_11)
    r_10, p_value = scipy.stats.pearsonr(power_spectrum_xn, power_spectrum_yn_10)
    r_9, p_value = scipy.stats.pearsonr(power_spectrum_xn, power_spectrum_yn_9)
    
    r_LMS = [r_9, r_10, r_11]
    print("频域内LMS算法下噪声相关性:{}".format(r_LMS))
    #############LMS算法##############
    
    width=0.3
    x_range = array([9, 10, 11])
    figure(1)
    plt.plot(x_range, r_LMS, 'r.-',label='r_LMS')
    plt.plot(x_range, r_NLMS, 'b.-',label='r_NLMS')
    plt.legend() 
    
    figure(2)
    plt.bar(x_range, r_LMS, width, label='r_LMS')
    plt.bar(x_range+width, r_NLMS, width, label='r_NLMS')
    plt.legend() 
    
    
    
    plt.show()   
  
########LMS与NLMS两种算法下得到的噪声yn在频域内的相关性比较#########       
    

########50HZ带阻滤波#########
'''
使用巴特沃斯带阻滤波器对表面肌电原始信号进行去噪
'''
def BandstopFilterShow(filename1, filename2):    
    sampling_rate=1000
    Force,Force_index,Raw_1,Envelope_1,Raw_2,Envelope_2=LoadRawDataSet(filename1)
    raw_1,envelope_1,raw_2,envelope_2 = LoadRawDataSetByChNum(filename2, CH=4)
    
    start_index = 0
    end_index = 100000   
    Raw = raw_1[start_index:end_index]    
    
#     dn=Envelope_1[:100000]
#     xn=envelope_1[:100000]
    
    Raw_Filter = ButterBandstopFilter(Raw, 49, 51, sampling_rate)
    
    freqs, power_spectrum_raw  = signal.periodogram(Raw, fs=sampling_rate, window='boxcar', nfft=None,  scaling='spectrum')
    freqs, power_spectrum_raw_filter  = signal.periodogram(Raw_Filter, fs=sampling_rate, window='boxcar', nfft=None,  scaling='spectrum')
    
    figure(0)
    plt.plot(range(len(Raw)),Raw,'r-',label='Raw')
    plt.legend()    
    figure(1)
    plt.plot(freqs[0:],power_spectrum_raw[0:],'r.-',label='power_spectrum_raw')
    plt.legend()
    figure(2)
    plt.plot(freqs[0:],power_spectrum_raw_filter[0:],'r.-',label='power_spectrum_raw_filter')
    plt.legend()  
    
    plt.show()  
########50HZ带阻滤波#########

########显示信号的功率表密度或功率谱#########
def PowerSpectralShow(filename):
    sampling_rate=1000
    Force,Force_index,Raw_1,Envelope_1,Raw_2,Envelope_2=LoadRawDataSet(filename)
#     Raw_2,Envelope_1 = LoadRawDataSetByChNum(filename, CH=2)
#     Raw_1,Envelope_1,Raw_2,Envelope_2 = LoadRawDataSetByChNum(filename, CH=4)
#     Raw_1=Envelope_1
    start_index = 0
    end_index = 2500   
    sig = Raw_1[start_index:end_index]
    
    #############################此过程等于signal.periodogram(sig, fs=sampling_rate, nfft=None,  scaling='spectrum')求功率谱
    fft_size=len(sig)
    freqs=np.linspace(0, sampling_rate/2, int(fft_size/2+1))   
    print(freqs[1])
     
    xf_Raw=np.fft.rfft(sig)/fft_size
    xf_Raw=(abs(xf_Raw)**2)*2
    xf_Raw[0]=0
    print(xf_Raw[1])
    ############################此过程等于signal.periodogram(sig, fs=sampling_rate, nfft=None,  scaling='spectrum')求功率谱
    
    freqs_spectrum_density, power_spectrum_density  = signal.periodogram(sig, fs=sampling_rate, nfft=None,  scaling='density')#能量谱，傅里叶变化的平方
    print(power_spectrum_density[1])
    freqs_spectrum, power_spectrum  = signal.periodogram(sig, fs=sampling_rate, nfft=None,  scaling='spectrum')#功率谱，傅里叶变换的平方除以时间长度
    print(power_spectrum[0])
    print(power_spectrum[1])
    print(power_spectrum[1]*(end_index - start_index)/sampling_rate)
    print(freqs_spectrum[1]) 
     
    figure(0)
    plt.plot(freqs[0:],xf_Raw[0:],'r.-',label='xf_Raw')
    plt.legend()
    
    figure(1)
    plt.plot(freqs_spectrum_density[0:],power_spectrum_density[0:],'r.-',label='power_spectrum_density')
    plt.legend()
    
    figure(2)
    plt.plot(freqs_spectrum[0:],power_spectrum[0:],'r.-',label='power_spectrum')
    plt.legend()
    
    plt.show()
########显示信号的功率表密度或功率谱#########



