import numpy as np
import math
from scipy import signal

'''
Function
'''

########sample entropy (SampEn) feature extraction#######
def SampEnMatrix(data, m):
    r = np.std(data) * 0.15      #r=0.1-0.25*Std(data)
    N = len(data)
    X1 = np.matrix(np.zeros((N-m+1, m)))#N-m+1 * m
    X2 = np.matrix(np.zeros((N-m, m+1)))#N-m * m+1,m=m+1
    
    i=0
    while i<N-m+1:
        if i<N-m+1:
            temp = data[i:i+m]
            X1[i, :] = np.matrix(temp)#N-m+1 * m
            
        if i<N-m:#m=m+1
            temp = data[i:(i+m+1)]
            X2[i, :] = np.matrix(temp)#N-m+2 * m+1     
        
        i=i+1
        
    Ci_1 = 0  
    Ci_2 = 0    
    One_Matrix1 = np.matrix(np.ones((N-m+1, 1)))
    One_Matrix2 = np.matrix(np.ones((N-m, 1)))
    i=0
    while i<N-m+1:
        if i<N-m+1:
            Xm_i = X1[i, :]
            temp_matrix = One_Matrix1 * Xm_i #N-m+1 * m
            Di = np.transpose(np.max(abs(temp_matrix-X1), axis=1))
            Nm = np.sum(Di<r)-1
            Ci_1 = Ci_1 + round(Nm/(N-m), 5)
            
        if i<N-m:
            Xm_i = X2[i, :]
            temp_matrix = One_Matrix2 * Xm_i #N-m+1 * m
            Di = np.transpose(np.max(abs(temp_matrix-X2), axis=1))#1 * N-m+1
            Nm = np.sum(Di<r)-1
            Ci_2 = Ci_2 + round(Nm/(N-m-1), 5)
        
        i=i+1   
         
    Ci_1_mean = Ci_1 /(N-m+1)  
    Ci_2_mean = Ci_2 /(N-m)
      
    sampEn = round(-math.log(Ci_2_mean/Ci_1_mean, math.e), 5)    
    return sampEn
########sample entropy (SampEn) feature extraction####### 

###############IFORCE and Force mean#################
def FeatureExtractForForce(force):    
    force_integral = round(np.sum(force))
    force_mean = round(np.mean(force))
    
    return force_integral, force_mean
###############IFORCE and Force mean#################

###############Time Domain feature extraction for raw#################   
def TDFeatureExtractForRaw(Raw):
    WAMP_TH = 20 
    WAMP = 0 
    WL_sum = 0 
    SSC = 0 
    SSC_TH = 18 
    ZC = 0 
    ZC_TH = 14 
    
    length = len(Raw)
    i = 0
    while i < length:
        if i>0:
            WL_sum = WL_sum + abs(Raw[i] - Raw[i-1])
            if np.abs(Raw[i] - Raw[i-1])>WAMP_TH:
                WAMP = WAMP + 1
            if i < length-1:
                if (Raw[i] - Raw[i-1])*(Raw[i] - Raw[i+1]) > SSC_TH:
                    SSC = SSC+1  
            if np.abs(Raw[i] - Raw[i-1])>ZC_TH and Raw[i]*Raw[i-1]<0: 
                ZC = ZC + 1    
                        
        i = i+1
        
    Raw_array = np.array(Raw) 
      
    MAV = round(np.mean(abs(Raw_array)), 5)        #MAV
    RMS = round((np.mean(Raw_array**2))**0.5, 5)   #RMS
    Var = round(np.var(Raw_array), 5)              #VAR
    IEMG = np.sum(np.abs(Raw_array))               #IEMG
    WL = WL_sum                                    #WL
    WAMP = WAMP                                    #WAMP
    SSC = SSC                                      #SSC
    ZC = ZC                                        #ZC
    SampEnValue = SampEnMatrix(Raw, 1)             #SampEn
    
    return MAV, RMS, Var, IEMG, WL, WAMP, SSC, ZC, SampEnValue
########Time Domain feature extraction for raw########

########Frequency Domain feature extraction for raw######## 
def FDFeatureExtractForRaw(Raw, sampling_rate):
    PKF = 0 
    MNF = 0 
    MNF_sum = 0
    MDF = 0 
    MDF_sum = 0
    
    freqs, power_spectrum  = signal.periodogram(Raw, fs=sampling_rate, window='boxcar', nfft=None,  scaling='spectrum')#FFT
    power_spectrum_sum = sum(power_spectrum)
    
    length = len(power_spectrum)
    i = 0
    while i < length:
        MNF_sum = MNF_sum + power_spectrum[i] * freqs[i]
        
        MDF_sum = MDF_sum + power_spectrum[i]
        if MDF_sum - power_spectrum[i] <= (power_spectrum_sum/2) and  MDF_sum >= (power_spectrum_sum/2):
            MDF = freqs[i] 
                
        i = i+1
    
    power_spectrum = list(power_spectrum)  
    PKF = round(freqs[power_spectrum.index(max(power_spectrum))], 5)        #PKF
    MNF = round(MNF_sum/power_spectrum_sum, 5)                              #MNF
    MDF = round(MDF, 5)                                                     #MDF
                                
    return PKF, MNF, MDF
########Frequency Domain feature extraction for raw########  

########Time Domain feature extraction for envelope########  
def TDFeatureExtractForEnvelope(Envelope):
    envelope_integral = np.sum(Envelope)
    envelope_mean = np.mean(Envelope)
    
    envelope_integral = envelope_integral                        #IEMG
    envelope_mean = round(envelope_mean, 5)                      #MEAN
    envelope_SampEnValue = SampEnMatrix(Envelope, 1)             #SampEn
    
    return envelope_integral, envelope_mean, envelope_SampEnValue
########Time Domain feature extraction for envelope######## 

########List transpose########
def List_Transpose(data):
    data = list(map(list, zip(*data)))    
    return data
########List transpose########



