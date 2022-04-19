import scipy
from scipy import signal
import numpy as np
import os
import math

from FileImport.ReadData import *
from scipy.sparse.linalg.eigen.arpack._arpack import dnaupd


'''
将data_array内的所有数据归一化
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


########自适应滤波噪声抵消算法RLS#########
'''
输入：
Xn:FIR滤波器输入参考信号，是在没有信号时的背景噪声
Dn:期望信号，噪声和有用信号叠加后的信号
order：滤波器的阶数
forget_factor：遗忘因子
Delta: 调整参数
输出：
Yn:最优滤波器的输出序列,即通过FIR滤波器输出的最接近信号噪声序列
error：误差输出序列
weight：权值
'''
def RLSFiltering(Xn, Dn, order, forget_factor, Delta):
    itr = len(Xn)
    error=array(np.zeros(itr)) #误差序列,error[k]表示第k次迭代时预期输出与实际输入的误差
    weight=np.matrix(np.zeros((order, itr)))#权值    
    T = np.eye(order)/Delta

    i=order
    while i<itr:#迭代计算
        x=Xn[i-order:i]
        x=np.matrix(x[::-1])#order阶滤波器抽头输入,倒序, x是行向量
        x_t = x.transpose()
        z=T * x_t #z是列向量
        k=z/(forget_factor + x * z) #列向量
        error[i]=Dn[i] - x * weight[:,i-1] #计算误差
        weight[:,i] = weight[:,i-1] + error[i] * k #更新权重
        T1 = k * z.transpose()
        T = (T - T1)/ forget_factor #更新T
        
        i=i+1
            
    Yn=array(np.zeros(itr))
    i=order
    while i<itr:
        x=Xn[i-order:i]
        x=np.matrix(x[::-1])#order阶滤波器抽头输入,行向量
        y=x*weight[:,-1]#滤波器的最优权重
        Yn[i]=y[0, 0]
        
        i=i+1
        
    return Yn, error, weight
########自适应滤波噪声抵消算法RLS#########

########应用自适应滤波噪声抵消算法RLS，显示滤波效果#########
def RLSFilterShow(filename1, filename2):    
    sampling_rate=1000
    Force,Force_index,Raw_1,Envelope_1,Raw_2,Envelope_2=LoadRawDataSet(filename1)
    raw_1,envelope_1,raw_2,envelope_2 = LoadRawDataSetByChNum(filename2, CH=4)
    
    start_index = 0
    end_index = 100000     
    dn=Raw_1[start_index:end_index]
    xn=raw_1[0:end_index-start_index]

    yn, error, weight= RLSFiltering(xn, dn, 10, 0.9996919, mean((array(dn))**2))  
 
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
    
    '''
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
    
    
    plt.show()
########应用自适应滤波噪声抵消算法RLS，显示滤波效果#########

########频域下计算信号平均功率#########
'''
频域下计算离散信号平均功率时，先求信号的功率谱，再将功率谱求和
'''
def GerPowerFre(data, sampling_rate):#计算功率
    freqs, power_spectrum = signal.periodogram(data, fs=sampling_rate, window='boxcar', nfft=None,  scaling='spectrum')
    
    return sum(power_spectrum)
    
########频域下计算信号平均功率#########

########滤波器最佳阶数不变的情况下，改变步长n计算信噪比######### 
'''
滤波器阶数固定时，改变步长因子，信噪比展示
'''
def SNRShow_Fre_Change_forgetfactor(filename1, filename2):
    sampling_rate = 1000
    Force,Force_index,Raw_1,Envelope_1,Raw_2,Envelope_2=LoadRawDataSet(filename1)
    raw_1,envelope_1,raw_2,envelope_2 = LoadRawDataSetByChNum(filename2, CH=4)
        
    start_index = 0
    end_index = 100000 
    sig=Raw_1[start_index:end_index]
    noise=raw_1[0:end_index-start_index]
    
    P_sig = GerPowerFre(sig, sampling_rate)
    P_noise = GerPowerFre(noise, sampling_rate)
    SNR_org = round(10* math.log((P_sig-P_noise)/P_noise, 10), 4) #原信号信噪比为28.9798 
    
    ####################RLS算法####################
    order = 11 #滤波器阶数  
    n = arange(0.99968, 0.99972, 0.000001)#RLS算法步长修正因子
    n = n[::-1]#倒序排列
    ####################RLS算法####################
    
    print("所需的运算次数：{}".format(len(n)))
    Delta = mean((array(sig))**2)
    
    SNR_filter = []
    SNR_dif = []
    i=0
    while i<len(n): 
        yn, sig_denoise, weight = RLSFiltering(noise, sig, order, n[i], Delta)  
        P_sig_denoise = GerPowerFre(sig_denoise, sampling_rate)
        P_yn = GerPowerFre(yn, sampling_rate)
        P_remain_noise = P_noise - P_yn
        try:
            SNR_filter.append(round(10 * math.log((P_sig_denoise-P_remain_noise)/P_remain_noise, 10), 4))            
        except:            
            print('最大步长修正系数n:{}'.format(n[i-1]))
            print("剩余噪声：{}".format(P_remain_noise))
            print("有用信号噪声-剩余噪声：{}".format(P_sig_denoise-P_remain_noise))
            break
        SNR_dif.append(SNR_filter[i] - SNR_org)
        print("信噪比之差:{}".format(SNR_dif[i]))
        print("第{}次".format(i))
        
        i=i+1 
    
    print("原来信噪比：{}" .format(SNR_org))
    print("现在信噪比：{}" .format(SNR_filter))
    print("信噪比之差：{}" .format(SNR_dif))
    
    name=str(order)
    figure(1)
    plt.title(name)
    plt.plot(range(len(SNR_filter)),SNR_filter,'r.-',label='SNR_filter')
    plt.legend()
    
    figure(2)
    plt.title(name)
    plt.plot(range(len(SNR_dif)),SNR_dif,'r.-',label='SNR_dif')
    plt.legend()
    
    plt.show()     
########滤波器最佳阶数不变的情况下，改变步长n计算信噪比######### 


def Get_Power_yn(Xn, weight, order, weight_idex, sampling_rate):
    itr = len(Xn)
    Yn=array(np.zeros(itr))
    i=order
    while i<itr:
        x=Xn[i-order:i]
        x=np.matrix(x[::-1])#order阶滤波器抽头输入
        y=x*weight[:,weight_idex]
        Yn[i]=y[0, 0]
        
        i=i+1
    
    P_yn = GerPowerFre(Yn, sampling_rate)
#     P_yn = GerPowerTime(Yn)
    return P_yn
'''
剩余噪声的变化曲线，查看收敛速度
'''   
def Remain_Noise_Power_Show(filename1, filename2):
    sampling_rate = 1000
    Force,Force_index,Raw_1,Envelope_1,Raw_2,Envelope_2=LoadRawDataSet(filename1)
    raw_1,envelope_1,raw_2,envelope_2 = LoadRawDataSetByChNum(filename2, CH=4)
        
    start_index = 0
    end_index = 100000 
    sig=Raw_1[start_index:end_index]
    noise=raw_1[0:end_index-start_index]
    
    P_noise = GerPowerFre(noise, sampling_rate)
#     P_noise = GerPowerTime(noise)
    Delta = mean((array(sig))**2)
    
    order = 9
    yn_RLS, sig_denoise_RLS, weight_RLS = RLSFiltering(noise, sig, order, 0.999705, Delta)
    
    P_Rest_RLS = []
    
    itr = len(noise)
    i = itr - 3000
    i_step = 5
    
    while i < itr:
        P_yn_RLS = Get_Power_yn(sig, weight_RLS, order, i, sampling_rate)
        P_remain_RLS = (P_noise - P_yn_RLS)**2  #剩余噪声的平方
        print("RLS剩余噪声：{}".format(P_remain_RLS))
        P_Rest_RLS.append(P_remain_RLS)        
        
        print("第{}次".format(i))
        i = i + i_step*order
        
    P_Rest_RLS_nor = Normalization(P_Rest_RLS) #误差平方归一化
    
    print("RLS算法标准差:{}".format(np.std(P_Rest_RLS_nor)))
    
    x_range = array(arange(0, 1000, round(1000/len(P_Rest_RLS_nor))))
    figure(1)
#     plt.plot(range(len(P_Rest_LMS_nor)),array(P_Rest_LMS_nor),'r-',label='P_Rest_LMS')
#     plt.plot(range(len(P_Rest_NLMS_nor)),array(P_Rest_NLMS_nor),'b-',label='P_Rest_NLMS')

    plt.plot(x_range,array(P_Rest_RLS_nor),'r-',label='P_Rest_RLS_nor')
    plt.xticks([0, 200, 400, 600, 800, 1000])
    plt.legend()
    
    plt.show()   
########查看剩余噪声的平均功率######### 

