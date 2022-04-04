from ActivityDetect.ForceActivity import *
from Denoise.ForceDenoiseProcess import *


def ForceActivity_Show(filename):
    sampling_rate = 1000
    force_speed = int(filename[-5])#根据文件名称中的倒数第五位确定是哪种速度下的力
    force,Raw_1,Envelope_1,Raw_2,Envelope_2=LoadDataSetAfterProcess(filename)
    force = ButterLowFilter(force, 5, sampling_rate)#力去噪
    force = np.rint(force)#四舍五入取整
    force = array(force)
    force[force<0] = 0#小于0的数全部替换成0
    force = list(force) 
    start_index,end_index,force_start_y,force_end_y,\
    raw1_start_y,raw1_end_y,raw2_start_y,raw2_end_y,\
    envelope1_start_y,envelope1_end_y,envelope2_start_y,envelope2_end_y = GetStartAndEndByForceUsingInterp1d(force, Raw_1, Envelope_1, Raw_2, Envelope_2, force_speed)
    
    force_th_x = range(len(force))#力阈值x轴
    force_th_y = [force_th] * len(force)#力阈值y轴
    
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
    
    
    figure(3)
    plt.plot(range(len(Raw_1)),Raw_1,'k-',label='raw1')
    # plt.plot(range(len(Raw_2)),Raw_2,'b-',label='raw2')
    # plt.plot(range(len(force)),force,'y-',label='force')
    plt.plot(start_index,raw1_start_y,'ro',label='start_point')
    plt.plot(end_index,raw1_end_y,'bs',label='end_point')
    plt.legend()
    
    figure(4)
    plt.plot(range(len(Envelope_2)),Envelope_2,'k-',label='Envelope_2')
    plt.plot(start_index,envelope2_start_y,'ro', label='start_point')
    plt.plot(end_index,envelope2_end_y,'bs',label='end_point')
    plt.legend()
    
    
    plt.show()




