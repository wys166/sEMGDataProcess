from pylab import *
from FileImport.ReadData import *
from ActivityDetection.DetectionFun import *

'''
Function
'''

############################Function############################
'''
Select TrainSet and TestSet random
It is best to set it to mode=1 or mode=2 to ensure that the number of the two is equal.
mode=1:even:train, odd:test
mode=2:odd:train, even:test
mode=3:train probability:1/2, test probability:1/2.
The number of train samples maybe be not 40, and the number of test samples maybe be not 40, when mode=3. 
'''
def TrainAndTestSetSelect(featurefilename, mode=1):
    feature_value, feature_name, singleCh_feature_name = LoadFeatures(featurefilename)
    normalization_feature_value = FeatureNormalizationExceptForceIntegral(feature_value)
    y_label = normalization_feature_value[-1] 
    x = List_Transpose(normalization_feature_value[:-3]) 
    y = normalization_feature_value[-2] 
    sample_num = len(y_label)
    
    trainSet_x = []
    trainSet_y = []
    trainSet_y_label = []
    testSet_x = []
    testSet_y =[]
    testSet_y_label =[]
    
    ratio = 5#Probability: train:1/2, test:1/2
    
    i=0
    while i < sample_num:
        if mode == 1:            
            if i%2 == 0: #even
                trainSet_x.append(x[i])
                trainSet_y.append(y[i])
                trainSet_y_label.append(y_label[i])
            else:
                testSet_x.append(x[i])
                testSet_y.append(y[i])
                testSet_y_label.append(y_label[i])
        elif mode == 2:
            if i%2 == 1: #odd
                trainSet_x.append(x[i])
                trainSet_y.append(y[i])
                trainSet_y_label.append(y_label[i])
            else:
                testSet_x.append(x[i])
                testSet_y.append(y[i])
                testSet_y_label.append(y_label[i])
        elif mode == 3:
            if random.randint(0, 10) < ratio:
                trainSet_x.append(x[i])
                trainSet_y.append(y[i])
                trainSet_y_label.append(y_label[i])
            else:
                testSet_x.append(x[i])
                testSet_y.append(y[i])
                testSet_y_label.append(y_label[i])
        
        i=i+1
        
    return trainSet_x, trainSet_y, trainSet_y_label, testSet_x, testSet_y, testSet_y_label
############################Function############################

############################Function############################
def SingleClassSeparate(predict_y, true_y, true_y_label):
    y_label = [1, 2, 3, 4, 5]#label
    length = len(true_y_label)
    
    predict_y_class1 = [] 
    predict_y_class2 = [] 
    predict_y_class3 = [] 
    predict_y_class4 = [] 
    predict_y_class5 = [] 
    
    true_y_class1 = [] 
    true_y_class2 = [] 
    true_y_class3 = [] 
    true_y_class4 = [] 
    true_y_class5 = [] 
    
    i=0
    while i<length:
        if true_y_label[i] == y_label[0]:
            predict_y_class1.append(predict_y[i])  
            true_y_class1.append(true_y[i])
        elif true_y_label[i] == y_label[1]:   
            predict_y_class2.append(predict_y[i])  
            true_y_class2.append(true_y[i])
        elif true_y_label[i] == y_label[2]:   
            predict_y_class3.append(predict_y[i])  
            true_y_class3.append(true_y[i])
        elif true_y_label[i] == y_label[3]:   
            predict_y_class4.append(predict_y[i])  
            true_y_class4.append(true_y[i])
        elif true_y_label[i] == y_label[4]:   
            predict_y_class5.append(predict_y[i])  
            true_y_class5.append(true_y[i])    
                     
        i=i+1
        
    predict_y = [predict_y_class1, predict_y_class2, predict_y_class3, predict_y_class4, predict_y_class5]
    true_y = [true_y_class1, true_y_class2, true_y_class3, true_y_class4, true_y_class5]
    return predict_y, true_y
############################Function############################

############################Function############################
def GetStdForPredictAndTrue(predict_y, true_y):
    error_value = np.array(predict_y) - np.array(true_y)
    std_value = np.std(error_value)
    return error_value, round(std_value)
############################Function############################

############################Function############################
def GetSpecialIndex(data, specialvalue):
    length = len(data)
    
    value = []
    i=0
    while i<length:
        value.append(abs(data[i]-specialvalue))
                
        i=i+1
    value_min = min(value)
    
    return value.index(value_min)
############################Function############################

############################Function############################
def StructureTriangle(force, force_speed, start_index, end_index, predict_y, true_y, force_index):
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
    Heigh_true_triangle = activity_force_front_fit[-1] - force_th 
    Bottom_true_triangle = activity_force_back_axis[-1] - activity_force_front_axis_part[0]
    S_true_triangle = Heigh_true_triangle * Bottom_true_triangle 
    predict_force_integral_class = predict_y[int(force_speed)-1]
    predict_force_integral = predict_force_integral_class[force_index]
    true_force_integral_class = true_y[int(force_speed)-1]
    true_force_integral = true_force_integral_class[force_index]
    predict_force_integral = predict_force_integral - force_th*len(activity_force)
    true_force_integral = true_force_integral - force_th*len(activity_force)
    S_predict_triangle = S_true_triangle * (predict_force_integral/true_force_integral)
    Heigh_predict_triangle = S_predict_triangle/Bottom_true_triangle
    Heigh_predict_triangle_y = Heigh_predict_triangle +force_th 
    triangle_predict_front_y_axis = [activity_force_front_fit_part[0], Heigh_predict_triangle_y] 
    triangle_predict_front_x_axis = [activity_force_front_axis_part[0], activity_force_front_axis_part[-1]]
    triangle_predict_back_y_axis = [Heigh_predict_triangle_y, force_th] 
    triangle_predict_back_x_axis = [activity_force_front_axis_part[-1], activity_force_back_axis[-1]]    
    p_predict_front = np.polyfit(triangle_predict_front_x_axis, triangle_predict_front_y_axis, 1)
    k_true = round(p_front[0], 8)
    k_predict = round(p_predict_front[0], 8) 
    
    return k_predict, k_true
############################Function############################

########Low pass filtering#########
'''
data: signal
cutoff: Cutoff frequency of the filter
sampling_rate: Sampling rate of signal
'''
def ButterLowFilter(data, cutoff, sampling_rate): 
    b, a = signal.butter(5, cutoff/(sampling_rate/2), 'low')
    Data = signal.filtfilt(b,a,data)
    
    return Data 
########Low pass filtering#########

############################Function############################
def GetReggressionSlope(filename, testSet_y, predict_y, true_y):
    sampling_rate=1000
    force,Raw_1,Envelope_1,Raw_2,Envelope_2=LoadDataSet(filename)
    force = ButterLowFilter(force, 5, sampling_rate)
    force = np.rint(force) 
    ################muscle contraction status detection################
    force_speed = int(filename[-5])
    subject_index =  GetSubjectIndex(filename)
    start_index,end_index,force_start_y,force_end_y,\
    raw1_start_y,raw1_end_y,raw2_start_y,raw2_end_y,\
    envelope1_start_y,envelope1_end_y,envelope2_start_y,envelope2_end_y=GetStartAndEndByForce(force, Raw_1,Envelope_1,Raw_2,Envelope_2, force_speed, subject_index)
    ################muscle contraction status detection################
    
    per_class_num = 40 
    K_predict = []
    K_true = []
    i = 0
    while i<per_class_num:
        k_predict_, k_true_ = StructureTriangle(force, force_speed, start_index, end_index, predict_y, true_y, i)
        K_predict.append(k_predict_)
        K_true.append(k_true_)
        i=i+1
        
    return K_predict, K_true
############################Function############################

########Function#########
'''
Load Regression Result
y_label:label of class
force_integral_true: Ture IFORCE
force_integral_predict: Predicted IFORCE
slope_true:Ture Slope
slope_predict: Predicted Slope
'''
def LoadRegressionResult(RegressionResultFilename):
    y_label=[]
    force_integral_true=[]
    force_integral_predict=[]
    slope_true=[]
    slope_predict=[] 
        
    i=0
    j=0
    with open(RegressionResultFilename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        dataset = dataset[1:]
        
        x_line=size(dataset,0)  
        y_line=size(dataset,1)  
        while i<x_line:
            j=0
            while j<y_line:
                if j<3:
                    dataset[i][j]=np.int(np.float(dataset[i][j]))
                else:
                    dataset[i][j]=np.float(dataset[i][j])  
                    
                if j==0:
                    y_label.append(dataset[i][j])                                           
                elif j==1:
                    force_integral_true.append(dataset[i][j])
                elif j==2:
                    force_integral_predict.append(dataset[i][j])                    
                elif j==3:
                    slope_true.append(dataset[i][j])  
                elif j==4:
                    slope_predict.append(dataset[i][j])                   
                
                j=j+1 
            i=i+1            
    csvfile.close()
    print('Data Length:'+str(len(dataset))) 
    
    return y_label, force_integral_true, force_integral_predict, slope_true, slope_predict
########Function#########

########Function#########
def LoadRegressionResultAnalysisFile(filename):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        dataset = dataset[1:]
        
        force_integral_std = dataset[0]
        force_integral_std = force_integral_std[1:]
        
        slope_std = dataset[1]
        slope_std = slope_std[1:]
        
        i = 0
        while i<5:
            force_integral_std[i] = float(force_integral_std[i])
            slope_std[i] = float(slope_std[i])
            i=i+1            
    csvfile.close()
    
    return force_integral_std, slope_std
########Function#########

########Function#########
def LoadMLPRegressionResult(MLPRegressionResultFilename):    
    with open(MLPRegressionResultFilename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        data = dataset[1:]       
        
        x_line=size(data,0)  
        y_line=size(data,1)  
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
    return Regression_Result
########Function#########

########Function#########
def SingleClassSeparateForRegressionResult(true_force_integral, predict_force_integral, ture_slope, predict_slope):
    sample_width = 40    
    True_IFORCE = [true_force_integral[0:1*sample_width], true_force_integral[1*sample_width:2*sample_width], true_force_integral[2*sample_width:3*sample_width], true_force_integral[3*sample_width:4*sample_width], true_force_integral[4*sample_width:5*sample_width]]
    Predict_IFORCE = [predict_force_integral[0:1*sample_width], predict_force_integral[1*sample_width:2*sample_width], predict_force_integral[2*sample_width:3*sample_width], predict_force_integral[3*sample_width:4*sample_width], predict_force_integral[4*sample_width:5*sample_width]]    
    True_Slope = [ture_slope[0:1*sample_width], ture_slope[1*sample_width:2*sample_width], ture_slope[2*sample_width:3*sample_width], ture_slope[3*sample_width:4*sample_width], ture_slope[4*sample_width:5*sample_width]]
    Predict_Slope = [predict_slope[0:1*sample_width], predict_slope[1*sample_width:2*sample_width], predict_slope[2*sample_width:3*sample_width], predict_slope[3*sample_width:4*sample_width], predict_slope[4*sample_width:5*sample_width]]    
    
    return True_IFORCE, Predict_IFORCE, True_Slope, Predict_Slope
########Function#########

################Function################
def LoadRegressionResultMeanStd(RegressionResultMeanStdfile):
    with open(RegressionResultMeanStdfile, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        dataset = dataset[1:]
        
        force_integral_mean_std = dataset[0]
        force_integral_mean_std = force_integral_mean_std[1:]
        
        i = 0
        while i<5:
            force_integral_mean_std[i] = float(force_integral_mean_std[i])
            force_integral_mean_std[i] = round(force_integral_mean_std[i], 5)
            i=i+1            
    csvfile.close()
    
    return force_integral_mean_std
################Function################

################Function################
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.- 0.2, 1.03*height, '%s' % int(height))
################Function################

