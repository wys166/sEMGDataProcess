import numpy as np
'''
Function and Parameter
'''


#####################Parameter#####################
'''
Set the thresholds, 5 grip patterns and 20 Subjects (The threshold can also vary slightly between subjects due to individual differences)
'''
force_th = 5 #TH1, about 2%MVC, force_th = 5 is suitable for 20 subjects, and it can also be adjusted according to the result of muscle contraction status detection.
TH2_1 = [800,  800,  800,  800,  800,  800,  800,  800,  800,  800,  800,  800,  800,  800,  800,  800,  800,  800,  800,  800]  #T1, TH2
TH2_2 = [1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500] #T2
TH2_3 = [2500, 2500, 2500, 2500, 2500, 2500, 2500, 2200, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500] #T3
TH2_4 = [3500, 3500, 3500, 3500, 3500, 3500, 3500, 3500, 3500, 3500, 3500, 3500, 3500, 3500, 3500, 3500, 3500, 3500, 3500, 3500] #T4
TH2_5 = [4700, 4500, 4500, 4500, 4500, 4500, 4500, 4500, 4500, 4500, 4500, 4500, 4500, 4500, 4500, 4500, 4500, 4500, 4500, 4500] #T5

TH3_1 = [2000, 2000, 2000, 2000, 2000, 2000, 2000, 1800, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000] #T1, TH3
TH3_2 = [3300, 3300, 3300, 3300, 3300, 3300, 3300, 2800, 3300, 3300, 3300, 3300, 3300, 3300, 3300, 3300, 3300, 3300, 3300, 3300] #T2
TH3_3 = [4100, 4000, 3500, 4000, 4000, 3500, 4000, 3800, 3500, 4000, 4000, 3500, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000] #T3
TH3_4 = [5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000] #T4
TH3_5 = [6500, 6300, 6300, 6300, 6300, 6300, 6300, 6300, 6300, 6300, 6300, 6300, 6300, 6300, 6300, 6300, 6300, 6300, 6300, 6300] #T5
#####################Parameter#####################

#####################Function#####################
'''
Get the ID of subject by filename
''' 
def GetSubjectIndex(filename):  
    if filename[-10] == 't':
        subject_index = int(filename[-9]) - 1
    else:
        subject_index =  int(int(filename[-9]) + int(filename[-10]) * 10 - 1)
        
    return subject_index 
#####################Function#####################

#####################Function#####################
'''
Delete Shake
'''
def DeleteFaultSample(start_index,end_index,force_start_y,force_end_y,raw1_start_y,raw1_end_y,raw2_start_y,raw2_end_y,envelope1_start_y,envelope1_end_y,envelope2_start_y,envelope2_end_y):
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
    
    return start_index,end_index,force_start_y,force_end_y,raw1_start_y,raw1_end_y,raw2_start_y,raw2_end_y,envelope1_start_y,envelope1_end_y,envelope2_start_y,envelope2_end_y
#####################Function#####################

#####################Function#####################
'''
Muscle contraction status detection function
force_speed: grip mode(1:T1,2:T2,3:T3,4:T4,5:T5)
subject_index: ID of subject (range:0-19)
'''
def GetStartAndEndByForce(force, Raw_1, Envelope_1, Raw_2, Envelope_2, force_speed, subject_index=0): 
    force_min_length_1 = TH2_1[subject_index]  
    force_min_length_2 = TH2_2[subject_index] 
    force_min_length_3 = TH2_3[subject_index] 
    force_min_length_4 = TH2_4[subject_index] 
    force_min_length_5 = TH2_5[subject_index] 
      
    force_max_length_1 = TH3_1[subject_index] 
    force_max_length_2 = TH3_2[subject_index]  
    force_max_length_3 = TH3_3[subject_index]  
    force_max_length_4 = TH3_4[subject_index]  
    force_max_length_5 = TH3_5[subject_index]   
       
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
 
   
        
    
    
