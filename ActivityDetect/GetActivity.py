from FileImport.ReadData import *

#####################力的短时能量阈值
force_energy_th_start = 8
force_energy_th_end = 5
activity_length = 800
#####################力的短时能量阈值


def GetActivity(Force):
    length=len(Force)
    
    window_width=100
    
    short_energy=[]
    sum=0
    
    i=0
    while i<length:
        if i<window_width-1:
            short_energy.append(0)
            sum=sum+Force[i]**2
        elif i==window_width-1:
            sum=sum+Force[i]**2
            short_energy.append(sum/(window_width*10))
        else:
            sum=sum-Force[i-window_width]**2
            sum=sum+Force[i]**2
            short_energy.append(sum/(window_width*10))
        
        i=i+1
    
    return short_energy


def GetStartAndEndByShortEnergy(short_energy, Force, Force_index, Raw_1,Envelope_1,Raw_2,Envelope_2):    
    length=len(short_energy)
    
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
        delay=i-0
        if find_flag==False and short_energy[i]>force_energy_th_start:
            start_index.append(Force_index[delay])
            force_start_y.append(Force[delay])
            raw1_start_y.append(Raw_1[Force_index[delay]])
            raw2_start_y.append(Raw_2[Force_index[delay]])
            envelope1_start_y.append(Envelope_1[Force_index[delay]])
            envelope2_start_y.append(Envelope_2[Force_index[delay]])
            
            find_flag=True
            
        if find_flag==True and short_energy[i]<force_energy_th_end:
            end_index.append(Force_index[i])
            force_end_y.append(Force[i])
            raw1_end_y.append(Raw_1[Force_index[i]])
            raw2_end_y.append(Raw_2[Force_index[i]])
            envelope1_end_y.append(Envelope_1[Force_index[i]])
            envelope2_end_y.append(Envelope_2[Force_index[i]])
            
            find_flag=False
            
        if len(end_index)>0 and end_index[-1]-start_index[-1] < activity_length:
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
    print("活动段个数为："+str(len(end_index)))
    return start_index,end_index,force_start_y,force_end_y,raw1_start_y,raw1_end_y,raw2_start_y,raw2_end_y,envelope1_start_y,envelope1_end_y,envelope2_start_y,envelope2_end_y      

def GetStartAndEndByShortEnergyUsingInterp1d(short_energy, force, Raw_1,Envelope_1,Raw_2,Envelope_2):    
    length=len(short_energy)
    
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
        if find_flag==False and short_energy[i]>force_energy_th_start:
            delay=i-250
            start_index.append(delay)
            force_start_y.append(force[delay])
            raw1_start_y.append(Raw_1[delay])
            raw2_start_y.append(Raw_2[delay])
            envelope1_start_y.append(Envelope_1[delay])
            envelope2_start_y.append(Envelope_2[delay])
            
            find_flag=True
            
        if find_flag==True and short_energy[i]<force_energy_th_end:
            delay=i+50
            end_index.append(delay)
            force_end_y.append(force[delay])
            raw1_end_y.append(Raw_1[delay])
            raw2_end_y.append(Raw_2[delay])
            envelope1_end_y.append(Envelope_1[delay])
            envelope2_end_y.append(Envelope_2[delay])
            
            find_flag=False
            
        if len(end_index) > 0 and len(end_index) == len(start_index) and end_index[-1] - start_index[-1] < activity_length + 300:
            print('删除掉的活动段长度：{}'.format(end_index[-1] - start_index[-1]))
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
    print("活动段个数为："+str(len(end_index)))
    return start_index,end_index,force_start_y,force_end_y,raw1_start_y,raw1_end_y,raw2_start_y,raw2_end_y,envelope1_start_y,envelope1_end_y,envelope2_start_y,envelope2_end_y      
    


def ActivityShow(filename):
    force,Raw_1,Envelope_1,Raw_2,Envelope_2=LoadDataSetAfterProcess(filename)
    short_energy=GetActivity(force)
    
    figure(1)
    plt.plot(range(len(force)),force,'r-',label='force')
    plt.legend()
    
    figure(2)
    plt.plot(range(len(short_energy)),short_energy,'r-',label='short_energy')
    plt.legend()
    
    plt.show()
    
    
 
    
