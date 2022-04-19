from FileImport.ReadData import *


#########################参数设置
#每个Subject握力大小不一,以及握的动作的标准程度有差异，因此以下参数可能不同，需要根据肌肉收缩活动段检测结果进行适当调整
force_th = 5 #大约2%MVC，不同的subject可能不同，根据MVC进行估计
force_min_length_1 = 800  #1秒力的数据长度最小值
force_min_length_2 = 1500 #2秒力的数据长度最小值
force_min_length_3 = 2500 #3秒力的数据长度最小值
force_min_length_4 = 3500 #4秒力的数据长度最小值
force_min_length_5 = 4500 #5秒力的数据长度最小值
  
force_max_length_1 = 2000 #1秒力的数据长度最大值
force_max_length_2 = 3300 #2秒力的数据长度最大值
force_max_length_3 = 4000 #3秒力的数据长度最大值
force_max_length_4 = 5000 #4秒力的数据长度最大值
force_max_length_5 = 6300 #5秒力的数据长度最大值
#########################参数设置

'''
把查找的干扰抖动去掉
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

'''
force:力数据
force_speed:取值为1/2/3/4/5，表示五种速度下的力
'''
def GetStartAndEndByForceUsingInterp1d(force, Raw_1, Envelope_1, Raw_2, Envelope_2, force_speed):    
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
    
    #################删除的错误活动段数量计数
    delete_fault_num_1 = 0
    delete_fault_num_2 = 0
    delete_fault_num_3 = 0
    delete_fault_num_4 = 0
    delete_fault_num_5 = 0
    #################删除的错误活动段数量计数
    
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
            
            print('活动段内的均值小于50，已删除！')
    
        i=i+1
    
    if force_speed == 1:
        print('删除的错误活动段数量:{}'.format(delete_fault_num_1))
    elif force_speed == 2:    
        print('删除的错误活动段数量:{}'.format(delete_fault_num_2))
    elif force_speed == 3:    
        print('删除的错误活动段数量:{}'.format(delete_fault_num_3))  
    elif force_speed == 4:    
        print('删除的错误活动段数量:{}'.format(delete_fault_num_4)) 
    elif force_speed == 5:    
        print('删除的错误活动段数量:{}'.format(delete_fault_num_5)) 
    
    if find_flag==True and len(start_index)-len(end_index)==1: #防止最后一个只找到了起始点，没有结束点
        del start_index[-1]
        del force_start_y[-1]
        del raw1_start_y[-1]
        del raw2_start_y[-1]
        del envelope1_start_y[-1]
        del envelope2_start_y[-1]
        
    print("活动段总数量为："+str(len(end_index)))
    return start_index,end_index,force_start_y,force_end_y,raw1_start_y,raw1_end_y,raw2_start_y,raw2_end_y,envelope1_start_y,envelope1_end_y,envelope2_start_y,envelope2_end_y      

    
