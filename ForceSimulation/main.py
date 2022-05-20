from ForceSimulation.ForceSimulationProcessFun import *
from FileImport.ReadData import *
from Denoise.ForceDenoiseFun import *

'''
Force sumulation result show
input: filepath=r'D:\\DataSet'
Please note: This function takes a lot of time to run
'''
def ForceSimulationShow(filepath):
    sampling_rate=1000
    
    filename = filepath + r'\\ForceDataSimulationShow\\MLPRegressionResult.csv'
    label, True_IFORCE, Predict_IFORCE, True_Slope, Predict_Slope = LoadResultAndSeparateAllClass(filename)
    
    Standard_Slope = [np.mean(True_Slope[0]), np.mean(True_Slope[1]), np.mean(True_Slope[2]), np.mean(True_Slope[3]), np.mean(True_Slope[4])]
    Mean_Predict_Slope = [np.mean(Predict_Slope[0]), np.mean(Predict_Slope[1]), np.mean(Predict_Slope[2]), np.mean(Predict_Slope[3]), np.mean(Predict_Slope[4])]
    
    ################T1################
    filename = filepath + r'\\ForceDataSimulationShow\\T1.csv'
    force,Raw_1,Envelope_1,Raw_2,Envelope_2=LoadDataSet(filename)
    force = ButterLowFilter(force, 5, sampling_rate)
    force = np.rint(force) 
    force_speed = int(filename[-5])
    start_index,end_index,force_start_y,force_end_y,\
    raw1_start_y,raw1_end_y,raw2_start_y,raw2_end_y,\
    envelope1_start_y,envelope1_end_y,envelope2_start_y,envelope2_end_y=GetStartAndEndByForce(force, Raw_1,Envelope_1,Raw_2,Envelope_2, force_speed)
     
    activity_force_axis_1, activity_force_1, true_triangle_x_1, true_triangle_y_1, predict_triangle_x_1, predict_triangle_y_1 = BuildTriangle(force, force_speed, start_index, end_index, Predict_IFORCE, True_IFORCE, 25)
     
    figure(1)
    plt.title('T'+str(force_speed))
    plt.plot(activity_force_axis_1, activity_force_1,'k-',label='actual')
    plt.plot(true_triangle_x_1, true_triangle_y_1,'b-',label='standard')
    plt.plot(predict_triangle_x_1, predict_triangle_y_1,'r--',label='simulative')
    plt.legend()
    ################T1################
    
    ################T2################
    filename = filepath + r'\\ForceDataSimulationShow\\T2.csv'
    force,Raw_1,Envelope_1,Raw_2,Envelope_2=LoadDataSet(filename)
    force = ButterLowFilter(force, 5, sampling_rate)
    force = np.rint(force)  
    force_speed = int(filename[-5])
    start_index,end_index,force_start_y,force_end_y,\
    raw1_start_y,raw1_end_y,raw2_start_y,raw2_end_y,\
    envelope1_start_y,envelope1_end_y,envelope2_start_y,envelope2_end_y=GetStartAndEndByForce(force, Raw_1,Envelope_1,Raw_2,Envelope_2, force_speed)
     
    activity_force_axis_2, activity_force_2, true_triangle_x_2, true_triangle_y_2, predict_triangle_x_2, predict_triangle_y_2 = BuildTriangle(force, force_speed, start_index, end_index, Predict_IFORCE, True_IFORCE, 4)
     
    figure(2)
    plt.title('T'+str(force_speed))
    plt.plot(activity_force_axis_2, activity_force_2,'k-',label='actual')
    plt.plot(true_triangle_x_2, true_triangle_y_2,'b-',label='standard')
    plt.plot(predict_triangle_x_2, predict_triangle_y_2,'r--',label='simulative')
    plt.legend()
    ################T2################
    
    ################T3################
    filename = filepath + r'\\ForceDataSimulationShow\\T3.csv'
    force,Raw_1,Envelope_1,Raw_2,Envelope_2=LoadDataSet(filename)
    force = ButterLowFilter(force, 5, sampling_rate)
    force = np.rint(force)  
    force_speed = int(filename[-5])
    start_index,end_index,force_start_y,force_end_y,\
    raw1_start_y,raw1_end_y,raw2_start_y,raw2_end_y,\
    envelope1_start_y,envelope1_end_y,envelope2_start_y,envelope2_end_y=GetStartAndEndByForce(force, Raw_1,Envelope_1,Raw_2,Envelope_2, force_speed)
 
    activity_force_axis_3, activity_force_3, true_triangle_x_3, true_triangle_y_3, predict_triangle_x_3, predict_triangle_y_3 = BuildTriangle(force, force_speed, start_index, end_index, Predict_IFORCE, True_IFORCE, 11)
      
    figure(3)
    plt.title('T'+str(force_speed))
    plt.plot(activity_force_axis_3, activity_force_3,'k-',label='actual')
    plt.plot(true_triangle_x_3, true_triangle_y_3,'b-',label='standard')
    plt.plot(predict_triangle_x_3, predict_triangle_y_3,'r--',label='simulative')
    plt.legend()
    ################T3################
    
    ################T4################
    filename = filepath + r'\\ForceDataSimulationShow\\T4.csv'
    force,Raw_1,Envelope_1,Raw_2,Envelope_2=LoadDataSet(filename)
    force = ButterLowFilter(force, 5, sampling_rate)
    force = np.rint(force)  
    force_speed = int(filename[-5])
    start_index,end_index,force_start_y,force_end_y,\
    raw1_start_y,raw1_end_y,raw2_start_y,raw2_end_y,\
    envelope1_start_y,envelope1_end_y,envelope2_start_y,envelope2_end_y=GetStartAndEndByForce(force, Raw_1,Envelope_1,Raw_2,Envelope_2, force_speed)
       
    activity_force_axis_4, activity_force_4, true_triangle_x_4, true_triangle_y_4, predict_triangle_x_4, predict_triangle_y_4 = BuildTriangle(force, force_speed, start_index, end_index, Predict_IFORCE, True_IFORCE, 16)
       
    figure(4)
    plt.title('T'+str(force_speed))
    plt.plot(activity_force_axis_4, activity_force_4,'k-',label='actual')
    plt.plot(true_triangle_x_4, true_triangle_y_4,'b-',label='standard')
    plt.plot(predict_triangle_x_4, predict_triangle_y_4,'r--',label='simulative')
    plt.legend()
    ################T4################
    
    ################T5################
    filename = filepath + r'\\ForceDataSimulationShow\\T5.csv'
    force,Raw_1,Envelope_1,Raw_2,Envelope_2=LoadDataSet(filename)
    force = ButterLowFilter(force, 5, sampling_rate)
    force = np.rint(force)  
    force_speed = int(filename[-5])
    start_index,end_index,force_start_y,force_end_y,\
    raw1_start_y,raw1_end_y,raw2_start_y,raw2_end_y,\
    envelope1_start_y,envelope1_end_y,envelope2_start_y,envelope2_end_y=GetStartAndEndByForce(force, Raw_1,Envelope_1,Raw_2,Envelope_2, force_speed)
        
    activity_force_axis_5, activity_force_5, true_triangle_x_5, true_triangle_y_5, predict_triangle_x_5, predict_triangle_y_5 = BuildTriangle(force, force_speed, start_index, end_index, Predict_IFORCE, True_IFORCE, 25)
        
    figure(5)
    plt.title('T'+str(force_speed))
    plt.plot(activity_force_axis_5, activity_force_5,'k-',label='actual')
    plt.plot(true_triangle_x_5, true_triangle_y_5,'b-',label='standard')
    plt.plot(predict_triangle_x_5, predict_triangle_y_5,'r--',label='simulative')
    plt.xticks([0, 2500, 5000])
    plt.legend()
    ################T5################

    ################Ratio################
    X_name=np.array([1, 2, 3, 4, 5])
    name_list = ['T1', 'T2', 'T3', 'T4', 'T5']
    figure(6)
    plt.title('Ratio')
    plt.plot(X_name, Standard_Slope,'b.-',label='standard')
    plt.plot(X_name, Mean_Predict_Slope,'r.--',label='simulative')
    plt.legend()
    plt.xticks(X_name, name_list)
    plt.yticks([0.05, 0.1, 0.15])
    ################Ratio################
    
    ################################
    figure(7)
    plt.subplot('231')
    plt.title('T1')
    plt.plot(activity_force_axis_1, activity_force_1,'k-',label='actual')
    plt.plot(true_triangle_x_1, true_triangle_y_1,'b-',label='standard')
    plt.plot(predict_triangle_x_1, predict_triangle_y_1,'r--',label='simulative')
    plt.subplot('232')
    plt.title('T2')
    plt.plot(activity_force_axis_2, activity_force_2,'k-',label='actual')
    plt.plot(true_triangle_x_2, true_triangle_y_2,'b-',label='standard')
    plt.plot(predict_triangle_x_2, predict_triangle_y_2,'r--',label='simulative')
    plt.subplot('233')
    plt.title('T3')
    plt.plot(activity_force_axis_3, activity_force_3,'k-',label='actual')
    plt.plot(true_triangle_x_3, true_triangle_y_3,'b-',label='standard')
    plt.plot(predict_triangle_x_3, predict_triangle_y_3,'r--',label='simulative')
    plt.subplot('234')
    plt.title('T4')
    plt.plot(activity_force_axis_4, activity_force_4,'k-',label='actual')
    plt.plot(true_triangle_x_4, true_triangle_y_4,'b-',label='standard')
    plt.plot(predict_triangle_x_4, predict_triangle_y_4,'r--',label='simulative')
    plt.subplot('235')
    plt.title('T5')
    plt.plot(activity_force_axis_5, activity_force_5,'k-',label='actual')
    plt.plot(true_triangle_x_5, true_triangle_y_5,'b-',label='standard')
    plt.plot(predict_triangle_x_5, predict_triangle_y_5,'r--',label='simulative')
    plt.xticks([0, 2500, 5000])
    plt.subplot('236')
    plt.title('Ratio')
    plt.plot(X_name, Standard_Slope,'b.-',label='standard')
    plt.plot(X_name, Mean_Predict_Slope,'r.--',label='simulative')
    plt.xticks(X_name, name_list)
    plt.yticks([0.05, 0.1, 0.15])
    ################################
    
    plt.show()
    
    

'''   
Show force sumulation result
input: r'D:\\DataSet' 
Please note: This function takes a lot of time to run
'''  
ForceSimulationShow(r'D:\\DataSet')
    
    