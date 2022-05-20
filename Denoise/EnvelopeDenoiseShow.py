from pylab import *
from Denoise.EnvelopeDenoiseFun import *
from FileImport.ReadData import *

'''
Show the effect of sEMG envelope signal denoising by three denoising methods
filename: full file path + file name, for example: filename=r'D:\\DataSet\\DenoiseDataShow\\Signal.csv'
'''
def EnvelopeDenoiseShow(filename):
    sampling_rate = 1000
    force,Raw_1,Envelope_1,Raw_2,Envelope_2=LoadDataSet(filename)
    
    envelope_low_filtering = ButterLowFilter(Envelope_1, 10, sampling_rate)
    envelope_median_filtering = MedianFilter(Envelope_1)
    envelope_slidingmean_filter = SlidingMeanFiltering(Envelope_1, 51)
    
    ##############FFT################
    begin_index = 0
    end_index = 98000

    freqs, power_spectrum_envelope1  = signal.periodogram(Envelope_1[begin_index:end_index], fs=sampling_rate, window='boxcar', nfft=None,  scaling='spectrum')
    freqs, power_spectrum_envelope1_filter  = signal.periodogram(envelope_slidingmean_filter[begin_index:end_index], fs=sampling_rate, window='boxcar', nfft=None,  scaling='spectrum')
    ##############FFT################
    
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
    plt.ylim(-10, 500)
    
    figure(4)
    plt.title('Fre_Envelope_1')
    plt.plot(freqs[1:],power_spectrum_envelope1_filter[1:],'r-',label='afterfilter')
    plt.legend()
    plt.ylim(-10, 500)
    
  
    plt.show()
 
 
'''
input: r'D:\\DataSet\\DenoiseDataShow\\Signal.csv'
'''     
EnvelopeDenoiseShow(r'D:\\DataSet\\DenoiseDataShow\\Signal.csv')



