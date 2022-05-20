import csv
import numpy as np
'''
Function
'''

#########parameter#########
force_th = 5
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
#########parameter#########

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
Muscle contraction status detection function for force sumulation 
'''
def GetStartAndEndByForce(force, Raw_1, Envelope_1, Raw_2, Envelope_2, force_speed):       
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
     
    if find_flag==True and len(start_index)-len(end_index)==1: 
        del start_index[-1]
        del force_start_y[-1]
        del raw1_start_y[-1]
        del raw2_start_y[-1]
        del envelope1_start_y[-1]
        del envelope2_start_y[-1]
        
    return start_index,end_index,force_start_y,force_end_y,raw1_start_y,raw1_end_y,raw2_start_y,raw2_end_y,envelope1_start_y,envelope1_end_y,envelope2_start_y,envelope2_end_y      
#####################Function#####################

########Function#########
'''
list Transpose
'''
def List_Transpose(data):
    data = list(map(list, zip(*data)))    
    return data
########Function#########

########Function#########
def LoadResultAndSeparateAllClass(filename):    
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        data = dataset[1:]       
        
        x_line=np.size(data,0)  
        y_line=np.size(data,1)  
        i=0
        j=0
        while i<x_line:
            j=0
            while j<y_line:
                data[i][j]=np.float(data[i][j])
               
                j=j+1 
            i=i+1            
    csvfile.close()
    
    data = List_Transpose(data) 
    Regression_Result = data
    Label = Regression_Result[0]
    true_force_integral = Regression_Result[1]
    predict_force_integral = Regression_Result[2]
    ture_slope = Regression_Result[3]
    predict_slope = Regression_Result[4]
    
    sample_width = 40  
    label = [Label[0:1*sample_width], Label[1*sample_width:2*sample_width], Label[2*sample_width:3*sample_width], Label[3*sample_width:4*sample_width], Label[4*sample_width:5*sample_width]]  
    True_IFORCE = [true_force_integral[0:1*sample_width], true_force_integral[1*sample_width:2*sample_width], true_force_integral[2*sample_width:3*sample_width], true_force_integral[3*sample_width:4*sample_width], true_force_integral[4*sample_width:5*sample_width]]
    Predict_IFORCE = [predict_force_integral[0:1*sample_width], predict_force_integral[1*sample_width:2*sample_width], predict_force_integral[2*sample_width:3*sample_width], predict_force_integral[3*sample_width:4*sample_width], predict_force_integral[4*sample_width:5*sample_width]]    
    True_Slope = [ture_slope[0:1*sample_width], ture_slope[1*sample_width:2*sample_width], ture_slope[2*sample_width:3*sample_width], ture_slope[3*sample_width:4*sample_width], ture_slope[4*sample_width:5*sample_width]]
    Predict_Slope = [predict_slope[0:1*sample_width], predict_slope[1*sample_width:2*sample_width], predict_slope[2*sample_width:3*sample_width], predict_slope[3*sample_width:4*sample_width], predict_slope[4*sample_width:5*sample_width]]    
    
    return label, True_IFORCE, Predict_IFORCE, True_Slope, Predict_Slope
########Function#########

########Function#########
def GetSpecialIndex(data, specialvalue):
    length = len(data)
    
    value = []
    i=0
    while i<length:
        value.append(abs(data[i]-specialvalue))
                
        i=i+1
    value_min = min(value)
    
    return value.index(value_min)
########Function#########

########Function#########
def BuildTriangle(force, force_speed, start_index, end_index, Predict_IFORCE, True_IFORCE, force_index):
    ######################
    activity_force = list(force[start_index[2*force_index+1]:end_index[2*force_index+1]])
    activity_force_axis = range(len(activity_force))
    activity_force_max_value = np.max(activity_force)
    activity_force_max_value_index = activity_force.index(activity_force_max_value)+30
    activity_force_front = activity_force[:activity_force_max_value_index]
    activity_force_front_axis = activity_force_axis[:activity_force_max_value_index]
    p_front = np.polyfit(activity_force_front_axis, activity_force_front, 1)
    activity_force_front_fit = activity_force_front_axis*p_front[0] + p_front[1]
    activity_force_back = [activity_force_front_fit[-1], activity_force[-1]]
    activity_force_back_axis = [activity_force_front_axis[-1], activity_force_axis[-1]]
    activity_force_front_5_index = GetSpecialIndex(activity_force_front_fit, force_th)
    activity_force_front_fit_part = activity_force_front_fit[activity_force_front_5_index:]
    activity_force_front_axis_part = activity_force_front_axis[activity_force_front_5_index:]
    true_triangle_x = list(activity_force_front_axis_part) + list(activity_force_back_axis) 
    true_triangle_y = list(activity_force_front_fit_part) + list(activity_force_back) 
    High_true_triangle = activity_force_front_fit[-1] - force_th 
    Bottom_true_triangle = activity_force_back_axis[-1] - activity_force_front_axis_part[0]
    S_true_triangle = High_true_triangle * Bottom_true_triangle 
    ######################
    
    ###################### 
    predict_force_integral_class = Predict_IFORCE[int(force_speed)-1]
    predict_force_integral = predict_force_integral_class[force_index]
    true_force_integral_class = True_IFORCE[int(force_speed)-1]
    true_force_integral = true_force_integral_class[force_index]
    print('true IFORCE:{},predict IFORCE:{}'.format(true_force_integral, predict_force_integral))
    predict_force_integral = predict_force_integral - force_th*len(activity_force)
    true_force_integral = true_force_integral - force_th*len(activity_force)
    
    S_predict_triangle = S_true_triangle * (predict_force_integral/true_force_integral)
    High_predict_triangle = S_predict_triangle/Bottom_true_triangle
    print('true high:{},simulative high:{}'.format(High_true_triangle, High_predict_triangle))
    Heigh_predict_triangle_y = High_predict_triangle +force_th 
    triangle_predict_front_y_axis = [activity_force_front_fit_part[0], Heigh_predict_triangle_y] 
    triangle_predict_front_x_axis = [activity_force_front_axis_part[0], activity_force_front_axis_part[-1]]
    triangle_predict_back_y_axis = [Heigh_predict_triangle_y, force_th] 
    triangle_predict_back_x_axis = [activity_force_front_axis_part[-1], activity_force_back_axis[-1]]
    predict_triangle_x = list(triangle_predict_front_x_axis) + list(triangle_predict_back_x_axis) 
    predict_triangle_y = list(triangle_predict_front_y_axis) + list(triangle_predict_back_y_axis) 
    p_predict_front = np.polyfit(triangle_predict_front_x_axis, triangle_predict_front_y_axis, 1)
    ###################### 
    
    return activity_force_axis, activity_force, true_triangle_x, true_triangle_y, predict_triangle_x, predict_triangle_y   
########Function#########
