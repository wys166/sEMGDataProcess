from FileImport.ReadData import *
from Denoise.ForceDenoiseProcess import *

force_th = 5 #大约2%MVC，不同的subject可能不同，根据MVC进行估计

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
    
    if find_flag==True and len(start_index)-len(end_index)==1: #防止最后一个只找到了起始点，没有结束点
        del start_index[-1]
        del force_start_y[-1]
        del raw1_start_y[-1]
        del raw2_start_y[-1]
        del envelope1_start_y[-1]
        del envelope2_start_y[-1]
        
    print("活动段总数量为："+str(len(end_index)))
    return start_index,end_index,force_start_y,force_end_y,raw1_start_y,raw1_end_y,raw2_start_y,raw2_end_y,envelope1_start_y,envelope1_end_y,envelope2_start_y,envelope2_end_y      
    

def ForceActivity_Show(filename):
    sampling_rate = 1000
    force,Raw_1,Envelope_1,Raw_2,Envelope_2=LoadDataSetAfterProcess(filename)
    force = ButterLowFilter(force, 5, sampling_rate)#力去噪
    force = np.rint(force)#四舍五入取整
    force = array(force)
    force[force<0] = 0#小于0的数全部替换成0
    force = list(force) 
    start_index,end_index,force_start_y,force_end_y,\
    raw1_start_y,raw1_end_y,raw2_start_y,raw2_end_y,\
    envelope1_start_y,envelope1_end_y,envelope2_start_y,envelope2_end_y = GetStartAndEndByForceUsingInterp1d(force, Raw_1, Envelope_1, Raw_2, Envelope_2, 1)
    
    force_th_x = range(len(force))#力阈值x轴
    force_th_y = [force_th] * len(force)#力阈值y轴
    
    figure(1)
    plt.plot(range(len(force)),force,'k-',label='grip force')
    plt.plot(range(len(force_th_x)),force_th_y,'b-',label='TH1')
    plt.legend()
    figure(2)
    plt.plot(range(len(force)),force,'k-',label='grip force')
    plt.plot(range(len(force_th_x)),force_th_y,'b-',label='TH1')
    plt.plot(start_index,force_start_y,'ro',label='beginning')
    plt.plot(end_index,force_end_y,'bs',label='ending')    
    plt.legend()
    
    
    Raw = Raw_2
    Raw_beginning = raw2_start_y
    Raw_ending = raw2_end_y
    figure(3)
    plt.plot(range(len(Raw)),Raw,'k-',label='raw sEMG')
    plt.plot(start_index,Raw_beginning,'ro',label='beginning')
    plt.plot(end_index,Raw_ending,'bs',label='ending')
    plt.legend()
    
    Envelope = Envelope_2
    Envelope_beginning = envelope2_start_y
    Envelope_ending = envelope2_end_y
    figure(4)
    plt.plot(range(len(Envelope_2)),Envelope_2,'k-',label='sEMG envelope')
    plt.plot(start_index,Envelope_beginning,'ro', label='beginning')
    plt.plot(end_index,Envelope_ending,'bs',label='ending')
    plt.legend()
    
    
    plt.show()

