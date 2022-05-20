from Denoise.RawDenoiseFun import *
from FileImport.ReadData import *

'''
Show result of NLMS
filename1: r'D:\\DataSet\\DenoiseDataShow\\Signal.csv'
filename2: r'D:\\DataSet\\DenoiseDataShow\\Reference.csv'
'''
def NLMSFilterShow(filename1, filename2):    
    sampling_rate=1000
    Force,Raw_1,Envelope_1,Raw_2,Envelope_2=LoadDataSet(filename1)
    raw_1,envelope_1,raw_2,envelope_2 = LoadReferenceDataSetByChNum(filename2, CH=4)
        
    start_index = 0
    end_index = 98000 # 15 full period   
    dn=Raw_1[start_index:end_index]
    xn=raw_1[start_index:end_index]
    
    yn, error, weight=NLMSFiltering(xn, dn, 9, 0.0052)  
    
    ##############FFT################
    freqs, power_spectrum_dn  = signal.periodogram(dn, fs=sampling_rate, window='boxcar', nfft=None,  scaling='spectrum')
    freqs, power_spectrum_xn  = signal.periodogram(xn, fs=sampling_rate, window='boxcar', nfft=None,  scaling='spectrum')
    freqs, power_spectrum_error  = signal.periodogram(error, fs=sampling_rate, window='boxcar', nfft=None,  scaling='spectrum')
    freqs, power_spectrum_yn  = signal.periodogram(yn, fs=sampling_rate, window='boxcar', nfft=None,  scaling='spectrum')
    ##############FFT################
    
    Raw = dn
    Raw_Filter = ButterBandstopFilter(Raw, 49, 51, sampling_rate)
    freqs, power_spectrum_bandstop  = signal.periodogram(Raw_Filter, fs=sampling_rate, window='boxcar', nfft=None,  scaling='spectrum')
    
    figure(0)
    plt.title('raw signal with noise')
    plt.plot(range(len(dn)),dn,'k-',label='raw sEMG')  
    figure(1)
    plt.title('rest PSD')
    plt.plot(freqs[0:-1],power_spectrum_xn[0:-1],'k.-',label='power spectrum density')
    figure(2)
    plt.title('rest PSD')
    plt.plot(freqs[0:-1],power_spectrum_xn[0:-1],'k.-',label='power spectrum density')
    plt.ylim(0, 0.6)
    figure(3)
    plt.title('PSD signal with noise')
    plt.plot(freqs,power_spectrum_dn,'k.-',label='before preprocessing')
    figure(4)
    plt.title('PSD signal with noise')
    plt.plot(freqs,power_spectrum_dn,'k.-',label='before preprocessing') 
    plt.xlim(30, 70)
    plt.ylim(-1, 24)   
    figure(5)
    plt.title('NLMS PSD')
    plt.plot(freqs,power_spectrum_error,'k.-',label='NLMS denoise')
    figure(6)
    plt.title('NLMS PSD')
    plt.plot(freqs,power_spectrum_error,'k.-',label='NLMS denoise')
    plt.xlim(30, 70)
    plt.ylim(-1, 24)
    figure(7)
    plt.title('yn PSD')
    plt.plot(freqs,power_spectrum_yn,'r.-',label='power_spectrum_yn')
    figure(8)
    plt.title('resting signal')
    plt.plot(range(len(raw_1[start_index:end_index])),raw_1[start_index:end_index],'k-',label='raw sEMG')
    plt.ylim(-2100, 2100)
    figure(9)
    plt.title('bandstop PSD')
    plt.plot(freqs,power_spectrum_bandstop,'k.-',label='power_spectrum_bandstop')
    figure(10)
    plt.title('bandstop PSD')
    plt.plot(freqs,power_spectrum_bandstop,'k.-',label='power_spectrum_bandstop')
    plt.xlim(30, 70)
    plt.ylim(-1, 24)
    
    
    plt.show()
  
  
'''
Show NLMS result
input:
filename1=r'D:\\DataSet\\DenoiseDataShow\\Signal.csv'
filename2=r'D:\\DataSet\\DenoiseDataShow\\Reference.csv'
'''    
NLMSFilterShow(r'D:\\DataSet\\DenoiseDataShow\\Signal.csv', r'D:\\DataSet\\DenoiseDataShow\\Reference.csv')    
