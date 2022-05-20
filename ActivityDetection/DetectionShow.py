from Denoise.ForceDenoiseFun import *
from FileImport.ReadData import *
from ActivityDetection.DetectionFun import *


'''
Show the result of muscle contraction status detection
filename: full file path + file name, for example: filename=r'D:\\DataSet\\Subject20\\T5.csv'
'''
def MuscleContractionStatusDetectionShow(filename):
    sampling_rate = 1000
    force_speed = int(filename[-5])#identify the grip pattern
    subject_index = GetSubjectIndex(filename)
    print('subject ID:{}'.format(subject_index))    
    force,Raw_1,Envelope_1,Raw_2,Envelope_2=LoadDataSet(filename)
    force = ButterLowFilter(force, 5, sampling_rate)#force signal denoising
    force = np.rint(force)
    force = array(force)
    force[force<0] = 0
    force = list(force) 
    start_index,end_index,force_start_y,force_end_y,\
    raw1_start_y,raw1_end_y,raw2_start_y,raw2_end_y,\
    envelope1_start_y,envelope1_end_y,envelope2_start_y,envelope2_end_y = GetStartAndEndByForce(force, Raw_1, Envelope_1, Raw_2, Envelope_2, force_speed, subject_index)
     
    force_th_x = range(len(force))
    force_th_y = [force_th] * len(force)
     
    figure(1)
    plt.plot(range(len(force)),force,'k-',label='force')
    plt.plot(range(len(force_th_x)),force_th_y,'b-',label='force_th')
    plt.legend()
     
    figure(2)
    plt.plot(range(len(force)),force,'k.-',label='force')
    plt.plot(range(len(force_th_x)),force_th_y,'b-',label='force_th')
    plt.plot(start_index,force_start_y,'ro',label='start_point')
    plt.plot(end_index,force_end_y,'bs',label='end_point')    
    plt.legend()
     
  
    plt.show()


'''
Muscle contraction status detection show
input: full file path + file name
For example: filename=r'D:\\DataSet\\Subject20\\T5.csv' 
Please note: This function takes a lot of time to run
''' 
MuscleContractionStatusDetectionShow(r'D:\\DataSet\\Subject20\\T5.csv')

