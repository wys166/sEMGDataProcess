from ActivityDetect.GetActivity import *


def StartAndEndPointShow(filename):   
    force, Raw_1,Envelope_1,Raw_2,Envelope_2 = LoadDataSetAfterProcess(filename)
    short_energy=GetActivity(force)
    start_index,end_index,force_start_y,force_end_y,\
    raw1_start_y,raw1_end_y,raw2_start_y,raw2_end_y,\
    envelope1_start_y,envelope1_end_y,envelope2_start_y,envelope2_end_y=GetStartAndEndByShortEnergyUsingInterp1d(short_energy, force, Raw_1,Envelope_1,Raw_2,Envelope_2)
     
    figure(1)
    plt.plot(range(len(Raw_1)),Raw_1,'r-',label='raw1')
    plt.plot(range(len(Raw_2)),Raw_2,'b-',label='raw2')
    plt.plot(range(len(force)),force,'y-',label='force')
    plt.plot(start_index,force_start_y,'ko',label='start_point')
    plt.plot(end_index,force_end_y,'cs',label='end_point')
    plt.legend()
    
    figure(2)
    plt.plot(range(len(Envelope_1)),Envelope_1,'g-',label='Envelope_1')
    plt.plot(range(len(Envelope_2)),Envelope_2,'k-',label='Envelope_2')
    plt.plot(range(len(force)),force,'y-',label='force')
    plt.plot(start_index,force_start_y,'ko',label='start_point')
    plt.plot(end_index,force_end_y,'cs',label='end_point')
    plt.legend()
     
    figure(3)
    plt.plot(range(len(Envelope_2)),Envelope_2,'b-',label='Envelope_1')
    plt.plot(start_index,envelope2_start_y,'ko', label='start_point')
    plt.plot(end_index,envelope2_end_y,'cs',label='end_point')
    plt.legend()
    
    figure(4)
    plt.plot(range(len(force)),force,'y-',label='force')
    plt.plot(start_index,force_start_y,'ko', label='start_point')
    plt.plot(end_index,force_end_y,'cs',label='end_point')
    plt.legend()

    plt.show()
    
def ActivityDetectShow(filename):
    force,Raw_1,Envelope_1,Raw_2,Envelope_2=LoadDataSetAfterProcess(filename)
    short_energy=GetActivity(force)
    start_index,end_index,force_start_y,force_end_y,\
    raw1_start_y,raw1_end_y,raw2_start_y,raw2_end_y,\
    envelope1_start_y,envelope1_end_y,envelope2_start_y,envelope2_end_y=GetStartAndEndByShortEnergyUsingInterp1d(short_energy, force, Raw_1,Envelope_1,Raw_2,Envelope_2)
    
    figure(1)
    plt.plot(range(len(Raw_1)),Raw_1,'k-',label='raw1')
    plt.plot(range(len(Raw_2)),Raw_2,'g-',label='raw2')
    plt.plot(range(len(force)),force,'y-',label='force')
    plt.plot(start_index,force_start_y,'ro',label='start_point')
    plt.plot(end_index,force_end_y,'bs',label='end_point')
    plt.legend()
    
    figure(2)
    plt.plot(range(len(Envelope_1)),Envelope_1,'k-',label='Envelope_1')
    plt.plot(range(len(Envelope_2)),Envelope_2,'g-',label='Envelope_2')
    plt.plot(range(len(force)),force,'y-',label='force')
    plt.plot(start_index,force_start_y,'ro',label='start_point')
    plt.plot(end_index,force_end_y,'bs',label='end_point')
    plt.legend()
    
    
    plt.show()
    
 
    
    
    
    