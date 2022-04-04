import numpy as np
import math
from scipy import signal

###############提取力的积分值和均值#################
def FeatureExtractForForce(force):    
    force_integral = round(sum(force))
    force_mean = round(np.mean(force))
    
    return force_integral, force_mean
###############提取力的积分值和均值#################

###############从时域提取表面肌电原始信号特征#################   
def TDFeatureExtractForRaw(Raw):
    WAMP_TH = 20 #威利森幅值的阈值 Willison amplitude, WAMP
    WAMP = 0 #威利森幅值
    WL_sum = 0 #波形长度累积
    SSC = 0 #斜率符合变化
    SSC_TH = 18 #斜率符合变化的阈值
    ZC = 0 #过零率
    ZC_TH = 14 #过零率的阈值
    
    length = len(Raw)
    i = 0
    while i < length:
        if i>0:
            WL_sum = WL_sum + abs(Raw[i] - Raw[i-1])#波形长度
            if abs(Raw[i] - Raw[i-1])>WAMP_TH:#威利森幅值
                WAMP = WAMP + 1
            if i < length-1:
                if (Raw[i] - Raw[i-1])*(Raw[i] - Raw[i+1]) > SSC_TH:#斜率符合变化，SSC
                    SSC = SSC+1  
            if abs(Raw[i] - Raw[i-1])>ZC_TH and Raw[i]*Raw[i-1]<0: #过零率
                ZC = ZC + 1    
                
        
        i = i+1
        
    Raw_array = np.array(Raw) #将Raw转化为数组 
      
    MAV = round(np.mean(abs(Raw_array)), 5)        #平均绝对值
    RMS = round((np.mean(Raw_array**2))**0.5, 5)   #均方根
    Var = round(np.var(Raw_array), 5)              #方差
    IEMG = np.sum(abs(Raw_array))                  #积分肌电值
    WL = WL_sum                                    #波形长度
    WAMP = WAMP                                    #威利森幅值
    SSC = SSC                                      #斜率符合变化
    ZC = ZC                                        #过零率
    SampEnValue = SampEnMatrix(Raw, 1)             #样本熵
    
    return MAV, RMS, Var, IEMG, WL, WAMP, SSC, ZC, SampEnValue

########提取样本熵值########
'''
data：时间序列
m：维数
r：阈值大小，一般取r=0.1-0.5*Std(data)
'''
def SampEnFun(data, m, r):
    N = len(data)
    
    data_construction = []
    i=0
    while i<N-m+1:
        data_construction.append(data[i:i+m])#取m维行向量
        
        i=i+1
    
    D = np.zeros((N-m+1, m))#产生N-m+1 * m维的0元素数组
    i=0
    while i<N-m+1:
        j=0
        while j<N-m+1:
            if j!=i:
                D[j] = max(abs(np.array(data_construction[i]) - np.array(data_construction[j])))
            
            j=j+1
        
        i=i+1
        
    Ci = []
    i=0
    while i<N-m+1:
        D_row = D[i]
        Nm = np.sum(D_row<r)-1#对于每个Nm统计小于r的距离个数,每行中都有零以此要减去1
        Ci.append(round(Nm/(N-m), 5))
        
        i=i+1 
    
    return round(np.mean(Ci), 8) #返回Ci的均值
def SampEn(data, m):
    r = np.std(data) * 0.1      #r=0.1-0.25*Std(data)
    
    S1 = SampEnFun(data, m, r)
    S2 = SampEnFun(data, m+1, r)
    
    sampEn = round(-math.log(S2/S1, math.e), 5)
    
    return sampEn

'''
求样本熵，通过矩阵运算
'''
def SampEnMatrix(data, m):
    r = np.std(data) * 0.15      #r=0.1-0.25*Std(data)
    N = len(data)
    X1 = np.matrix(np.zeros((N-m+1, m)))#N-m+1 * m维矩阵
    X2 = np.matrix(np.zeros((N-m, m+1)))#N-m * m+1维矩阵,m=m+1
    
    i=0
    while i<N-m+1:
        if i<N-m+1:
            temp = data[i:i+m]
            X1[i, :] = np.matrix(temp)#N-m+1 * m维矩阵
            
        if i<N-m:#m=m+1
            temp = data[i:(i+m+1)]
            X2[i, :] = np.matrix(temp)#N-m+2 * m+1维矩阵     
        
        i=i+1
        
    Ci_1 = 0  
    Ci_2 = 0    
    One_Matrix1 = np.matrix(np.ones((N-m+1, 1)))
    One_Matrix2 = np.matrix(np.ones((N-m, 1)))
    i=0
    while i<N-m+1:
        if i<N-m+1:
            Xm_i = X1[i, :]#第i行
            temp_matrix = One_Matrix1 * Xm_i #N-m+1 * m维矩阵，每行都是Xm_i
            Di = np.transpose(np.max(abs(temp_matrix-X1), axis=1))#距离，1 * N-m+1维
            Nm = np.sum(Di<r)-1#对于每个Nm统计小于r的距离个数,每行中都有零以此要减去1
            Ci_1 = Ci_1 + round(Nm/(N-m), 5)
            
        if i<N-m:
            Xm_i = X2[i, :]#第i行
            temp_matrix = One_Matrix2 * Xm_i #N-m+1 * m维矩阵，每行都是Xm_i
            Di = np.transpose(np.max(abs(temp_matrix-X2), axis=1))#距离，1 * N-m+1维
            Nm = np.sum(Di<r)-1#对于每个Nm统计小于r的距离个数,每行中都有零以此要减去1
            Ci_2 = Ci_2 + round(Nm/(N-m-1), 5)
        
        i=i+1   
         
    Ci_1_mean = Ci_1 /(N-m+1)  
    Ci_2_mean = Ci_2 /(N-m)
      
    sampEn = round(-math.log(Ci_2_mean/Ci_1_mean, math.e), 5)    
    return sampEn
###############从时域提取表面肌电原始信号特征################# 

###############从频域提取表面肌电原始信号特征#################   
def FDFeatureExtractForRaw(Raw, sampling_rate):
    PKF = 0 #峰值频率
    MNF = 0 #平均频率
    MNF_sum = 0
    MDF = 0 #中值频率
    MDF_sum = 0
    
    freqs, power_spectrum  = signal.periodogram(Raw, fs=sampling_rate, window='boxcar', nfft=None,  scaling='spectrum')#功率谱密度，即能量谱除以时间
    freqs, power_spectrum_density  = signal.periodogram(Raw, fs=sampling_rate, window='boxcar', nfft=None,  scaling='density')#能量谱，也称为能量谱密度，傅里叶变换后的平方
    
    power_spectrum_sum = sum(power_spectrum)
    
    length = len(power_spectrum_density)
    i = 0
    while i < length:
        MNF_sum = MNF_sum + power_spectrum[i] * freqs[i]
        
        MDF_sum = MDF_sum + power_spectrum[i]
        if MDF_sum - power_spectrum[i] <= (power_spectrum_sum/2) and  MDF_sum >= (power_spectrum_sum/2):
            MDF = freqs[i] 
                
        
        i = i+1
    
    power_spectrum = list(power_spectrum)  
    PKF = round(freqs[power_spectrum.index(max(power_spectrum))], 5)        #峰值频率
    MNF = round(MNF_sum/power_spectrum_sum, 5)                              #平均功率
    MDF = round(MDF, 5)                                                     #中值频率
                                
    
    return PKF, MNF, MDF

###############从频域提取表面肌电原始信号特征################# 

###############从时域提取表面肌电包洛信号特征#################   
def TDFeatureExtractForEnvelope(Envelope):
    envelope_integral = np.sum(Envelope)
    envelope_mean = np.mean(Envelope)
    
    length = len(Envelope)
    i = 0
    while i < length:
        
        
        
        
        i = i+1
    
    envelope_integral = envelope_integral                        #积分值
    envelope_mean = round(envelope_mean, 5)                      #均值
    envelope_SampEnValue = SampEnMatrix(Envelope, 1)             #样本熵
    
    return envelope_integral, envelope_mean, envelope_SampEnValue

###############从时域提取表面肌电包洛信号特征################# 

########列表转置########
def List_Transpose(data):
    data = list(map(list, zip(*data)))    #转置
    return data
########列表转置########





