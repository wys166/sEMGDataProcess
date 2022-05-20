from Denoise.ForceDenoiseFun import *
from FileImport.ReadData import *
from ActivityDetection.DetectionFun import *

#####################Function#####################
'''
Muscle contraction status detection function for signal
'''
def GetStartAndEndByForce(force, Raw_1, Envelope_1, Raw_2, Envelope_2, force_speed): 
    force_th = 8
    force_min_length_1 = 800 
    force_min_length_2 = 1500
    force_min_length_3 = 2500
    force_min_length_4 = 3500
    force_min_length_5 = 4500
      
    force_max_length_1 = 2000
    force_max_length_2 = 3300
    force_max_length_3 = 4000  
    force_max_length_4 = 5000
    force_max_length_5 = 6300   
      
    length=len(force)
    
    force_start_y=[]
    force_end_y=[]

    raw1_start_y=[]
    raw1_end_y=[]
    raw2_start_y=[]
    raw2_end_y=[]
    
    envelope1_start_y=[]
    envelope1_end_y=[]
    envelope2_start_y=[]
    envelope2_end_y=[]
    
    start_index=[]
    end_index=[]
    
    delete_fault_num_1 = 0
    delete_fault_num_2 = 0
    delete_fault_num_3 = 0
    delete_fault_num_4 = 0
    delete_fault_num_5 = 0
    
    find_flag=False
    
    i=0
    while i<length:        
        if find_flag==False and force[i]>=force_th:
            if np.abs(force[i] - force_th) > np.abs(force[i-1] - force_th):
                delay=i-1
            else:
                delay=i
                
            start_index.append(delay)
            force_start_y.append(force[delay])
            raw1_start_y.append(Raw_1[delay])
            raw2_start_y.append(Raw_2[delay])
            envelope1_start_y.append(Envelope_1[delay])
            envelope2_start_y.append(Envelope_2[delay])
            
            find_flag=True
            last_i = i
        elif find_flag==True and force[i]<=force_th and i-last_i>40:
            if np.abs(force[i] - force_th) > np.abs(force[i-1] - force_th):
                delay=i-1
            else:
                delay=i
                
            end_index.append(delay)
            force_end_y.append(force[delay])
            raw1_end_y.append(Raw_1[delay])
            raw2_end_y.append(Raw_2[delay])
            envelope1_end_y.append(Envelope_1[delay])
            envelope2_end_y.append(Envelope_2[delay])
            
            find_flag=False
         
        if force_speed == 1:       
            if len(end_index) > 0 and len(end_index) == len(start_index) and (end_index[-1] - start_index[-1] < force_min_length_1 or end_index[-1] - start_index[-1] > force_max_length_1):
                start_index,end_index,force_start_y,force_end_y,raw1_start_y,raw1_end_y,raw2_start_y,raw2_end_y,envelope1_start_y,envelope1_end_y,envelope2_start_y,envelope2_end_y = DeleteFaultSample(start_index,end_index,force_start_y,force_end_y,raw1_start_y,raw1_end_y,raw2_start_y,raw2_end_y,envelope1_start_y,envelope1_end_y,envelope2_start_y,envelope2_end_y)       
                delete_fault_num_1 = delete_fault_num_1+1
        elif force_speed == 2:    
            if len(end_index) > 0 and len(end_index) == len(start_index) and (end_index[-1] - start_index[-1] < force_min_length_2 or end_index[-1] - start_index[-1] > force_max_length_2):
                start_index,end_index,force_start_y,force_end_y,raw1_start_y,raw1_end_y,raw2_start_y,raw2_end_y,envelope1_start_y,envelope1_end_y,envelope2_start_y,envelope2_end_y = DeleteFaultSample(start_index,end_index,force_start_y,force_end_y,raw1_start_y,raw1_end_y,raw2_start_y,raw2_end_y,envelope1_start_y,envelope1_end_y,envelope2_start_y,envelope2_end_y)       
                delete_fault_num_2 = delete_fault_num_2+1
        elif force_speed == 3:
            if len(end_index) > 0 and len(end_index) == len(start_index) and (end_index[-1] - start_index[-1] < force_min_length_3 or end_index[-1] - start_index[-1] > force_max_length_3):
                start_index,end_index,force_start_y,force_end_y,raw1_start_y,raw1_end_y,raw2_start_y,raw2_end_y,envelope1_start_y,envelope1_end_y,envelope2_start_y,envelope2_end_y = DeleteFaultSample(start_index,end_index,force_start_y,force_end_y,raw1_start_y,raw1_end_y,raw2_start_y,raw2_end_y,envelope1_start_y,envelope1_end_y,envelope2_start_y,envelope2_end_y)       
                delete_fault_num_3 = delete_fault_num_3+1
        elif force_speed == 4:
            if len(end_index) > 0 and len(end_index) == len(start_index) and (end_index[-1] - start_index[-1] < force_min_length_4 or end_index[-1] - start_index[-1] > force_max_length_4):
                start_index,end_index,force_start_y,force_end_y,raw1_start_y,raw1_end_y,raw2_start_y,raw2_end_y,envelope1_start_y,envelope1_end_y,envelope2_start_y,envelope2_end_y = DeleteFaultSample(start_index,end_index,force_start_y,force_end_y,raw1_start_y,raw1_end_y,raw2_start_y,raw2_end_y,envelope1_start_y,envelope1_end_y,envelope2_start_y,envelope2_end_y)       
                delete_fault_num_4 = delete_fault_num_4+1
        elif force_speed == 5:
            if len(end_index) > 0 and len(end_index) == len(start_index) and (end_index[-1] - start_index[-1] < force_min_length_5 or end_index[-1] - start_index[-1] > force_max_length_5):
                start_index,end_index,force_start_y,force_end_y,raw1_start_y,raw1_end_y,raw2_start_y,raw2_end_y,envelope1_start_y,envelope1_end_y,envelope2_start_y,envelope2_end_y = DeleteFaultSample(start_index,end_index,force_start_y,force_end_y,raw1_start_y,raw1_end_y,raw2_start_y,raw2_end_y,envelope1_start_y,envelope1_end_y,envelope2_start_y,envelope2_end_y)       
                delete_fault_num_5 = delete_fault_num_5+1
        
        if len(end_index)>0 and len(end_index) == len(start_index) and np.mean(force[start_index[-1] : end_index[-1]])<30:
            del start_index[-1]
            del force_start_y[-1]
            del raw1_start_y[-1]
            del raw2_start_y[-1]
            del envelope1_start_y[-1]
            del envelope2_start_y[-1]
            
            del end_index[-1]
            del force_end_y[-1]
            del raw1_end_y[-1]
            del raw2_end_y[-1]
            del envelope1_end_y[-1]
            del envelope2_end_y[-1] 
            
    
        i=i+1
    
    if force_speed == 1:
        print('remove number of shake:{}'.format(delete_fault_num_1))
    elif force_speed == 2:    
        print('remove number of shake:{}'.format(delete_fault_num_2))
    elif force_speed == 3:    
        print('remove number of shake:{}'.format(delete_fault_num_3))  
    elif force_speed == 4:    
        print('remove number of shake:{}'.format(delete_fault_num_4)) 
    elif force_speed == 5:    
        print('remove number of shake:{}'.format(delete_fault_num_5)) 
    
    if find_flag==True and len(start_index)-len(end_index)==1: 
        del start_index[-1]
        del force_start_y[-1]
        del raw1_start_y[-1]
        del raw2_start_y[-1]
        del envelope1_start_y[-1]
        del envelope2_start_y[-1]
        
    print("Number of activity:{}".format(len(end_index)))
    return start_index,end_index,force_start_y,force_end_y,raw1_start_y,raw1_end_y,raw2_start_y,raw2_end_y,envelope1_start_y,envelope1_end_y,envelope2_start_y,envelope2_end_y      
#####################Function#####################

'''
input: filename=r'D:\\DataSet\\ActivityDetectDataShow\\Signal.csv'
'''
def MuscleContractionStatusDetectionShow(filename):
    sampling_rate = 1000
    force,Raw_1,Envelope_1,Raw_2,Envelope_2=LoadDataSet(filename)
    force = ButterLowFilter(force, 5, sampling_rate)
    force = np.rint(force)
    force = array(force)
    force[force<0] = 0
    force = list(force) 
    start_index,end_index,force_start_y,force_end_y,\
    raw1_start_y,raw1_end_y,raw2_start_y,raw2_end_y,\
    envelope1_start_y,envelope1_end_y,envelope2_start_y,envelope2_end_y = GetStartAndEndByForce(force, Raw_1, Envelope_1, Raw_2, Envelope_2, 4)
    
    force_th_x = range(len(force))
    force_th_y = [8] * len(force)
    
    figure(1)
    plt.title('detection by TH1 ')
    plt.plot(range(len(force)),force,'k-',label='grip force')
    plt.plot(range(len(force_th_x)),force_th_y,'b-',label='TH1')
    
    figure(2)
    plt.title('force detection')
    plt.plot(range(len(force)),force,'k-',label='grip force')
    plt.plot(range(len(force_th_x)),force_th_y,'b-',label='TH1')
    plt.plot(start_index,force_start_y,'ro',label='beginning')
    plt.plot(end_index,force_end_y,'bs',label='ending')    
    
    Raw = Raw_2
    Raw_beginning = raw2_start_y
    Raw_ending = raw2_end_y
    figure(3)
    plt.title('raw detection')
    plt.plot(range(len(Raw)),Raw,'k-',label='raw sEMG')
    plt.plot(start_index,Raw_beginning,'ro',label='beginning')
    plt.plot(end_index,Raw_ending,'bs',label='ending')
    
    Envelope_beginning = envelope2_start_y
    Envelope_ending = envelope2_end_y
    figure(4)
    plt.title('envelope detection')
    plt.plot(range(len(Envelope_2)),Envelope_2,'k-',label='sEMG envelope')
    plt.plot(start_index,Envelope_beginning,'ro', label='beginning')
    plt.plot(end_index,Envelope_ending,'bs',label='ending')

    
    plt.show()
    
'''
Muscle contraction status detection show
input: r'D:\\DataSet\\ActivityDetectDataShow\\Signal.csv'
'''     
MuscleContractionStatusDetectionShow(r'D:\\DataSet\\ActivityDetectDataShow\\Signal.csv')



