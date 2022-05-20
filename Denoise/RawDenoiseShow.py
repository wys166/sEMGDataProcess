from Denoise.RawDenoiseFun import *
from FileImport.ReadData import *


#########################LMS Show#########################
'''
Show result of LMS
filename1: signal with noise, filename1=r'D:\\DataSet\\DenoiseDataShow\\Signal.csv'
filename2: reference signal, filename2=r'D:\\DataSet\\DenoiseDataShow\\Reference.csv'
'''
def LMSFilterShow(filename1, filename2):    
    sampling_rate=1000
    Force,Raw_1,Envelope_1,Raw_2,Envelope_2=LoadDataSet(filename1)
    raw_1,envelope_1,raw_2,envelope_2 = LoadReferenceDataSetByChNum(filename2, CH=4)
    
    start_index = 0
    end_index = 98000 # 15 full period     
    dn=Raw_1[start_index:end_index]
    xn=raw_1[start_index:end_index]

    yn, error, weight=LMSFiltering(xn, dn, 9, 0.0000053)   

    r_t, p_value_t = scipy.stats.pearsonr(xn, yn)
    print("Relevance r:{}".format(r_t))
    print("p value:{}".format(p_value_t))
    
    ##############FFT################
    freqs, power_spectrum_dn  = signal.periodogram(dn, fs=sampling_rate, window='boxcar', nfft=None,  scaling='spectrum')
    freqs, power_spectrum_xn  = signal.periodogram(xn, fs=sampling_rate, window='boxcar', nfft=None,  scaling='spectrum')
    freqs, power_spectrum_error  = signal.periodogram(error, fs=sampling_rate, window='boxcar', nfft=None,  scaling='spectrum')
    freqs, power_spectrum_yn  = signal.periodogram(yn, fs=sampling_rate, window='boxcar', nfft=None,  scaling='spectrum')
    ##############FFT################
    
    r_f, p_value_f = scipy.stats.pearsonr(power_spectrum_xn, power_spectrum_yn)
    print("Power Spectrum relevance r:{}".format(r_f))
    print("Power Spectrum p value:{}".format(p_value_f))
    
    figure(1)
    plt.plot(range(len(dn)),dn,'r-',label='dn')
    plt.legend()
    figure(2)
    plt.plot(range(len(xn)),xn,'r.-',label='xn')
    plt.legend()
    plt.ylim(-2000, 2000)
    figure(3)
    plt.plot(range(len(yn)),yn,'r.-',label='yn')
    plt.legend()
    plt.ylim(-2000, 2000)
    figure(4)
    plt.plot(range(len(error)),error,'b-',label='error')
    plt.legend()
    figure(5)
    plt.plot(range(len(dn)),dn,'r-',label='dn')
    plt.plot(range(len(error)),error,'b-',label='error')
    plt.plot(range(len(xn)),xn,'g-',label='xn')
    plt.plot(range(len(yn)),yn,'k-',label='yn')
    plt.legend()
    plt.ylim(-2000, 2000)
    figure(6)
    plt.plot(range(len(dn)),dn,'r-',label='dn')
    plt.legend()    
    figure(7)
    plt.plot(freqs[0:-1],power_spectrum_xn[0:-1],'r.-',label='power_spectrum_xn')
    plt.legend()
    figure(8)
    plt.plot(freqs[0:],power_spectrum_dn[0:],'r.-',label='power_spectrum_dn')
    plt.legend()
    figure(9)
    plt.plot(freqs[0:],power_spectrum_error[0:],'r.-',label='power_spectrum_error')
    plt.legend()
    figure(10)
    plt.plot(freqs[0:],power_spectrum_yn[0:],'r.-',label='power_spectrum_yn')
    plt.legend()   
    
    plt.show()
#########################LMS Show#########################

#########################NLMS Show#########################
'''
Show result of NLMS
filename1: signal with noise, filename1=r'D:\\DataSet\\DenoiseDataShow\\Signal.csv'
filename2: reference signal, filename2=r'D:\\DataSet\\DenoiseDataShow\\Reference.csv'
'''
def NLMSFilterShow(filename1, filename2):    
    sampling_rate=1000
    Force,Raw_1,Envelope_1,Raw_2,Envelope_2=LoadDataSet(filename1)
    raw_1,envelope_1,raw_2,envelope_2 = LoadReferenceDataSetByChNum(filename2, CH=4)
        
    start_index = 0
    end_index = 98000  # 15 full period     
    dn=Raw_1[start_index:end_index]
    xn=raw_1[start_index:end_index]
    
    yn, error, weight=NLMSFiltering(xn, dn, 9, 0.0052)  
    
    r_t, p_value = scipy.stats.pearsonr(xn, yn)
    print("Relevance r:{}".format(r_t))
    print("p value:{}".format(p_value))
    
    ##############FFT################
    freqs, power_spectrum_dn  = signal.periodogram(dn, fs=sampling_rate, window='boxcar', nfft=None,  scaling='spectrum')
    freqs, power_spectrum_xn  = signal.periodogram(xn, fs=sampling_rate, window='boxcar', nfft=None,  scaling='spectrum')
    freqs, power_spectrum_error  = signal.periodogram(error, fs=sampling_rate, window='boxcar', nfft=None,  scaling='spectrum')
    freqs, power_spectrum_yn  = signal.periodogram(yn, fs=sampling_rate, window='boxcar', nfft=None,  scaling='spectrum')
    ##############FFT################
    
    r_f, p_value_f = scipy.stats.pearsonr(power_spectrum_xn, power_spectrum_yn)
    print("Power Spectrum relevance r:{}".format(r_f))
    print("Power Spectrum p value:{}".format(p_value_f))
    
    figure(1)
    plt.plot(range(len(dn)),dn,'r-',label='dn')
    plt.legend()
    figure(2)
    plt.plot(range(len(xn)),xn,'r.-',label='xn')
    plt.legend()
    plt.ylim(-2000, 2000)
    figure(3)
    plt.plot(range(len(yn)),yn,'r-',label='yn')
    plt.legend()
    plt.ylim(-2000, 2000)
    figure(4)
    plt.plot(range(len(error)),error,'b-',label='error')
    plt.legend()
    figure(5)
    plt.plot(range(len(dn)),dn,'r-',label='dn')
    plt.plot(range(len(error)),error,'b-',label='error')
    plt.plot(range(len(xn)),xn,'g-',label='xn')
    plt.plot(range(len(yn)),yn,'k-',label='yn')
    plt.legend() 
    plt.ylim(-2000, 2000)
    figure(6)
    plt.plot(range(len(dn)),dn,'k-',label='raw sEMG')
    plt.legend()    
    figure(7)
    plt.plot(freqs[0:-1],power_spectrum_xn[0:-1],'k.-',label='power spectrum density')
    plt.legend()
    figure(8)
    plt.plot(freqs[0:],power_spectrum_dn[0:],'k.-',label='before preprocessing')
    plt.xticks([0, 250, 500])
    plt.legend()
    figure(9)
    plt.plot(freqs[0:],power_spectrum_error[0:],'k.-',label='NLMS')
    plt.xticks([0, 250, 500])
    plt.legend()
    figure(10)
    plt.plot(freqs[0:],power_spectrum_yn[0:],'r.-',label='power_spectrum_yn')
    plt.legend()
    figure(11)
    plt.plot(freqs[0:],power_spectrum_dn[0:],'r.',label='power_spectrum_dn')
    plt.plot(freqs[0:],power_spectrum_error[0:],'b.',label='power_spectrum_error')
    plt.legend()   
    figure(12)
    plt.plot(range(len(raw_1[start_index:end_index])),raw_1[start_index:end_index],'k-',label='resting raw sEMG')
    plt.legend() 
    plt.ylim(-2000, 2000)
    
    plt.show()
#########################NLMS Show#########################

#########################Relevance analysis######################### 
'''
Compare between LMS and NLMS
filename1: signal with noise, filename1=r'D:\\DataSet\\DenoiseDataShow\\Signal.csv'
filename2: reference signal, filename2=r'D:\\DataSet\\DenoiseDataShow\\Reference.csv'
'''
def RelevanceShow_LMS_NLMS(filename1, filename2): 
    sampling_rate=1000
    Force,Raw_1,Envelope_1,Raw_2,Envelope_2=LoadDataSet(filename1)
    raw_1,envelope_1,raw_2,envelope_2 = LoadReferenceDataSetByChNum(filename2, CH=4)
        
    start_index = 0
    end_index = 59400    
    dn=Raw_1[start_index:end_index]
    xn=raw_1[start_index:end_index]
    
    freqs, power_spectrum_dn  = signal.periodogram(dn, fs=sampling_rate, window='boxcar', nfft=None,  scaling='spectrum')
    freqs, power_spectrum_xn  = signal.periodogram(xn, fs=sampling_rate, window='boxcar', nfft=None,  scaling='spectrum')
    
    #############NLMS##############
    yn_11, error_11, weight=NLMSFiltering(xn, dn, 11, 0.00716)  
    yn_10, error_10, weight=NLMSFiltering(xn, dn, 10, 0.00605)
    yn_9, error_9, weight=NLMSFiltering(xn, dn, 9, 0.0052)

    freqs, power_spectrum_yn_11  = signal.periodogram(yn_11, fs=sampling_rate, window='boxcar', nfft=None,  scaling='spectrum')
    freqs, power_spectrum_yn_10  = signal.periodogram(yn_10, fs=sampling_rate, window='boxcar', nfft=None,  scaling='spectrum')
    freqs, power_spectrum_yn_9  = signal.periodogram(yn_9, fs=sampling_rate, window='boxcar', nfft=None,  scaling='spectrum')

    r_11, p_value = scipy.stats.pearsonr(power_spectrum_xn, power_spectrum_yn_11)
    r_11 = np.round(r_11, 3)
    r_10, p_value = scipy.stats.pearsonr(power_spectrum_xn, power_spectrum_yn_10)
    r_10 = np.round(r_10, 3)
    r_9, p_value = scipy.stats.pearsonr(power_spectrum_xn, power_spectrum_yn_9)
    r_9 = np.round(r_9, 3)
    r_NLMS = [r_9, r_10, r_11]
    print("NLMS relevance r:{}".format(r_NLMS))
    #############NLMS##############
    
    #############LMS##############
    yn_11, error_11, weight=LMSFiltering(xn, dn, 11, 0.00000464)  
    yn_10, error_10, weight=LMSFiltering(xn, dn, 10, 0.00000479)
    yn_9, error_9, weight=LMSFiltering(xn, dn, 9, 0.0000053)

    freqs, power_spectrum_yn_11  = signal.periodogram(yn_11, fs=sampling_rate, window='boxcar', nfft=None,  scaling='spectrum')
    freqs, power_spectrum_yn_10  = signal.periodogram(yn_10, fs=sampling_rate, window='boxcar', nfft=None,  scaling='spectrum')
    freqs, power_spectrum_yn_9  = signal.periodogram(yn_9, fs=sampling_rate, window='boxcar', nfft=None,  scaling='spectrum')

    r_11, p_value = scipy.stats.pearsonr(power_spectrum_xn, power_spectrum_yn_11)
    r_11 = np.round(r_11, 3)
    r_10, p_value = scipy.stats.pearsonr(power_spectrum_xn, power_spectrum_yn_10)
    r_10 = np.round(r_10, 3)
    r_9, p_value = scipy.stats.pearsonr(power_spectrum_xn, power_spectrum_yn_9)
    r_9 = np.round(r_9, 3)
    r_LMS = [r_9, r_10, r_11]
    print("LMS relevance r:{}".format(r_LMS))
    #############LMS##############
    
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
#########################Relevance analysis#########################   

#########################convergence rate show#########################
'''
compare convergence rate between LMS and NLMS
filename1: signal with noise, filename1=r'D:\\DataSet\\DenoiseDataShow\\Signal.csv'
filename2: reference signal, filename2=r'D:\\DataSet\\DenoiseDataShow\\Reference.csv'
'''   
def ConvergenceRateShow(filename1, filename2):
    sampling_rate = 1000
    Force,Raw_1,Envelope_1,Raw_2,Envelope_2=LoadDataSet(filename1)
    raw_1,envelope_1,raw_2,envelope_2 = LoadReferenceDataSetByChNum(filename2, CH=4)
         
    start_index = 0
    end_index = 59400 
    sig=Raw_1[start_index:end_index]
    noise=raw_1[start_index:end_index]
    P_noise = GerPowerFre(noise, sampling_rate)
 
    order = 9
    yn_NLMS, sig_denoise_NLMS, weight_NLMS = NLMSFiltering(noise, sig, order, 0.0052)
    yn_LMS, sig_denoise_LMS, weight_LMS = LMSFiltering(noise, sig, order, 0.0000053)   
     
    P_Rest_LMS = []
    P_Rest_NLMS = []   
     
    itr = len(noise)
    i = itr - 1000
    i_step = 5
     
    while i < itr:
        P_yn_LMS = Get_Power_yn(sig, weight_LMS, order, i, sampling_rate)
        P_remain_LMS = (P_noise - P_yn_LMS)**2  
        P_Rest_LMS.append(P_remain_LMS)
         
        P_yn_NLMS = Get_Power_yn(sig, weight_NLMS, order, i, sampling_rate)
        P_remain_NLMS = (P_noise - P_yn_NLMS)**2  
        P_Rest_NLMS.append(P_remain_NLMS)
         
        print("time:{}".format(i))
        i = i + i_step*order
         
    P_Rest_LMS_nor = Normalization(P_Rest_LMS) 
    P_Rest_NLMS_nor = Normalization(P_Rest_NLMS) 
     
    print("LMS Std:{}".format(np.std(P_Rest_LMS_nor)))
    print("NLMS Std:{}".format(np.std(P_Rest_NLMS_nor)))
    x_range = array(arange(0, 1000, round(1000/len(P_Rest_LMS_nor))))
    x_range = x_range[:len(P_Rest_LMS_nor)]
    figure(1)
    plt.plot(x_range,array(P_Rest_LMS_nor),'r-',label='P_Rest_LMS')
    plt.plot(x_range,array(P_Rest_NLMS_nor),'b-',label='P_Rest_NLMS')
    plt.xticks([0, 200, 400, 600, 800, 1000])
    plt.legend()
     
    plt.show()   
#########################convergence rate show#########################

#########################RLS Show#########################
'''
Show result of RLS
filename1: signal with noise, filename1=r'D:\\DataSet\\DenoiseDataShow\\Signal.csv'
filename2: reference signal, filename2=r'D:\\DataSet\\DenoiseDataShow\\Reference.csv'
''' 
def RLSFilterShow(filename1, filename2):    
    sampling_rate=1000
    Force,Raw_1,Envelope_1,Raw_2,Envelope_2=LoadDataSet(filename1)
    raw_1,envelope_1,raw_2,envelope_2 = LoadReferenceDataSetByChNum(filename2, CH=4)
    
    start_index = 0
    end_index = 98000      
    dn=Raw_1[start_index:end_index]
    xn=raw_1[start_index:end_index]
 
    yn, error, weight= RLSFiltering(xn, dn, 10, 0.9996919, mean((array(dn))**2))  
 
    r_t, p_value_t = scipy.stats.pearsonr(xn, yn)
    print("Relevance r:{}".format(r_t))
    print("p value:{}".format(p_value_t))    
    
    ##############FFT################
    freqs, power_spectrum_dn  = signal.periodogram(dn, fs=sampling_rate, window='boxcar', nfft=None,  scaling='spectrum')
    freqs, power_spectrum_xn  = signal.periodogram(xn, fs=sampling_rate, window='boxcar', nfft=None,  scaling='spectrum')
    freqs, power_spectrum_error  = signal.periodogram(error, fs=sampling_rate, window='boxcar', nfft=None,  scaling='spectrum')
    freqs, power_spectrum_yn  = signal.periodogram(yn, fs=sampling_rate, window='boxcar', nfft=None,  scaling='spectrum')
    ##############FFT################
    
    r_f, p_value_f = scipy.stats.pearsonr(power_spectrum_xn, power_spectrum_yn)
    print("Power Spectrum relevance r:{}".format(r_f))
    print("Power Spectrum p value:{}".format(p_value_f))    
    
    figure(1)
    plt.plot(range(len(dn)),dn,'r-',label='dn')
    plt.legend()
    figure(2)
    plt.plot(range(len(xn)),xn,'r.-',label='xn')
    plt.legend()
    plt.ylim(-2000, 2000)
    figure(3)
    plt.plot(range(len(yn)),yn,'r.-',label='yn')
    plt.legend()
    plt.ylim(-2000, 2000)
    figure(4)
    plt.plot(range(len(error)),error,'b-',label='error')
    plt.legend()
    figure(5)
    plt.plot(range(len(dn)),dn,'r-',label='dn')
    plt.plot(range(len(error)),error,'b-',label='error')
    plt.plot(range(len(xn)),xn,'g-',label='xn')
    plt.plot(range(len(yn)),yn,'k-',label='yn')
    plt.legend()
    plt.ylim(-2000, 2000)
    figure(6)
    plt.plot(range(len(dn)),dn,'r-',label='dn')
    plt.legend()    
    figure(7)
    plt.plot(freqs[0:-1],power_spectrum_xn[0:-1],'r.-',label='power_spectrum_xn')
    plt.legend()
    figure(8)
    plt.plot(freqs[0:],power_spectrum_dn[0:],'r.-',label='power_spectrum_dn')
    plt.legend()
    figure(9)
    plt.plot(freqs[0:],power_spectrum_error[0:],'r.-',label='power_spectrum_error')
    plt.legend()
    figure(10)
    plt.plot(freqs[0:],power_spectrum_yn[0:],'r.-',label='power_spectrum_yn')
    plt.legend()   
    
    
    plt.show()
#########################RLS Show#########################



# LMSFilterShow(r'D:\\DataSet\\DenoiseDataShow\\Signal.csv', r'D:\\DataSet\\DenoiseDataShow\\Reference.csv')

# NLMSFilterShow(r'D:\\DataSet\\DenoiseDataShow\\Signal.csv', r'D:\\DataSet\\DenoiseDataShow\\Reference.csv')

# RelevanceShow_LMS_NLMS(r'D:\\DataSet\\DenoiseDataShow\\Signal.csv', r'D:\\DataSet\\DenoiseDataShow\\Reference.csv')

# ConvergenceRateShow(r'D:\\DataSet\\DenoiseDataShow\\Signal.csv', r'D:\\DataSet\\DenoiseDataShow\\Reference.csv')

RLSFilterShow(r'D:\\DataSet\\DenoiseDataShow\\Signal.csv', r'D:\\DataSet\\DenoiseDataShow\\Reference.csv')


