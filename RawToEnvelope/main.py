from FileImport.ReadData import *

############Function############
def Rectification(data):
    length=len(data)
    newdata=[]
    
    i=0
    while i<length:
        if data[i]<0:
            newdata.append(-data[i])
        else:
            newdata.append(data[i])
        
        i=i+1
    
    return newdata
############Function############

'''
Show difference of the raw sEMG signal and sEMG envelope signal
input: filename=r'D:\\DataSet\\DenoiseDataShow\\Signal.csv'
'''
def RawToEnvelopeShow(filename):    
    Force,Raw_1,Envelope_1,Raw_2,Envelope_2=LoadDataSet(filename)
        
    start = 13800
    end = 21300
    envelope=Envelope_1[start:end]# a full period
    raw=Raw_1[start:end]# a full period
      
    rectify = Rectification(raw)
           
    
    figure(1)
    plt.title('raw sEMG')
    plt.plot(range(len(raw)),raw,'b-',label='raw sEMG')
    plt.xticks(np.arange(0, 8001, 2000),fontproperties = 'Times New Roman', size = 10)
    plt.yticks(np.arange(-2000, 2500, 500),fontproperties = 'Times New Roman', size = 10)

    figure(2)
    plt.title('rectified raw sEMG')
    plt.plot(range(len(rectify)),rectify,'k-',label='rectified sEMG')
    plt.xticks(np.arange(0, 8001, 2000),fontproperties = 'Times New Roman', size = 10)
    plt.yticks(np.arange(0, 2500, 500),fontproperties = 'Times New Roman', size = 10)

    figure(3)
    plt.title('sEMG envelope')
    plt.plot(range(len(envelope)),envelope,'r-',label='sEMG envelope')
    plt.xticks(np.arange(0, 8001, 2000),fontproperties = 'Times New Roman', size = 10)
    plt.yticks(np.arange(0, 3500, 1000),fontproperties = 'Times New Roman', size = 10)


    plt.show()


'''
Show difference of the raw sEMG signal and sEMG envelope signal
input: r'D:\\DataSet\\DenoiseDataShow\\Signal.csv'
'''
RawToEnvelopeShow(r'D:\\DataSet\\DenoiseDataShow\\Signal.csv')

