from FileImport.ReadData import *
import random
from _operator import itemgetter
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
    normalization_feature_value = FeatureNormalization(feature_value)
    y = normalization_feature_value[-1] 
    x = List_Transpose(normalization_feature_value[:-3]) 
    sample_num = len(y)
    
    trainSet_x = []
    trainSet_y = []
    testSet_x = []
    testSet_y =[]
    
    ratio = 5#Probability: train:1/2, test:1/2
    
    i=0
    while i < sample_num:
        if mode == 1:            
            if i%2 == 0: #even
                trainSet_x.append(x[i])
                trainSet_y.append(y[i])
            else:
                testSet_x.append(x[i])
                testSet_y.append(y[i])
        elif mode == 2:
            if i%2 == 1: #odd
                trainSet_x.append(x[i])
                trainSet_y.append(y[i])
            else:
                testSet_x.append(x[i])
                testSet_y.append(y[i])
        elif mode == 3:
            if random.randint(0, 10) < ratio:
                trainSet_x.append(x[i])
                trainSet_y.append(y[i])
            else:
                testSet_x.append(x[i])
                testSet_y.append(y[i])
        
        i=i+1
        
    return trainSet_x, trainSet_y, testSet_x, testSet_y
############################Function############################

############################Function############################
'''
get mean accuracy of all samples
'''
def GetAllMeanAccuracy(predict_label, true_label):
    length = len(predict_label)
    
    accuracy_num = 0
    i=0
    while i<length:
        if predict_label[i] == true_label[i]:
            accuracy_num = accuracy_num+1  
                 
        print('Predict label: {}, Ture label: {}'.format(predict_label[i], true_label[i]))
        i=i+1
    
    accuracy = round(accuracy_num/length, 4) * 100
    
    print("Accuracy: {}%".format(accuracy))
    
    return round(accuracy, 2)
############################Function############################

############################Function############################
'''
get detailed accuracy
'''
def GetAccuracyForMultiClass(predict_label, true_label):    
    n_sample=len(true_label)
    
    class_actual_prediction={} #format：{class:[num_actual,num_prediction]}

    i=0
    while i<n_sample:
        if true_label[i] in class_actual_prediction:
            class_actual_prediction[true_label[i]][0]+=1
            if true_label[i]==predict_label[i]:
                class_actual_prediction[true_label[i]][1]+=1       
        else:
            class_actual_prediction[true_label[i]]=[1,0]
            if true_label[i]==predict_label[i]:
                class_actual_prediction[true_label[i]][1]=1
        i=i+1
        
    accuracy=[]
    sum=0
    for key in class_actual_prediction:
        accuracy.append([key,round((class_actual_prediction[key][1]*100/class_actual_prediction[key][0]), 2)])
        sum=sum+class_actual_prediction[key][1]
        
    accuracy=sorted(accuracy,key=itemgetter(0),reverse=False)
    accuracy.insert(0,['mean',round(sum*100/n_sample,2)])
    print('mean and every class accuracy: {}'.format(accuracy))
    accuracy=list(map(list,zip(*accuracy))) 
             
    return accuracy #accuracy[0]:label, accuracy[1]:accuracy(percentage) 
############################Function############################

############################Function############################
'''
Load single subject classification result
'''
def LoadSingleSubjectClassificationResult(filename):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        dataset = dataset[1:]
        
        LDA_Classer_Accuracy = []
        Ridge_Classer_Accuracy = []
        Log_Classer_Accuracy = []
        Knn_Classer_Accuracy = []
        MLP_Classer_Accuracy = []
        SVM_Classer_Accuracy = []

        x_line=size(dataset,0)  
        y_line=size(dataset,1)  
        i = 0
        while i<x_line:
            j=1
            while j<y_line:
                dataset[i][j]=np.float(dataset[i][j])
                if i==0:
                    LDA_Classer_Accuracy.append(dataset[i][j])    
                elif i==1:                                       
                    Ridge_Classer_Accuracy.append(dataset[i][j]) 
                elif i==2:
                    Log_Classer_Accuracy.append(dataset[i][j])
                elif i==3:
                    Knn_Classer_Accuracy.append(dataset[i][j])
                elif i==4:
                    MLP_Classer_Accuracy.append(dataset[i][j])
                elif i==5:
                    SVM_Classer_Accuracy.append(dataset[i][j])    
                      
                j=j+1 
            i=i+1            
    csvfile.close()
    
    return LDA_Classer_Accuracy, Ridge_Classer_Accuracy, Log_Classer_Accuracy, Knn_Classer_Accuracy, MLP_Classer_Accuracy, SVM_Classer_Accuracy
############################Function############################

############################Function############################
'''
Load overall classification result
'''
def LoadOverallClassificationResult(filename):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        result = dataset[1:] 
        result = List_Transpose(result)
        result = result[1:-1]
        result = List_Transpose(result)
        
        x_line=size(result,0)  
        y_line=size(result,1)  
        i=0
        while i<x_line:
            j=0
            while j<y_line:
                result[i][j]=np.float(result[i][j])
               
                j=j+1 
            i=i+1            
    csvfile.close()
    print('Data Length:'+str(len(dataset))) 
    
    accuracy_mean = result[0:6]
    accuracy_min = result[6:12]
    accuracy_max = result[12:18]
    
    return accuracy_mean, accuracy_min, accuracy_max   
############################Function############################

############################Function############################
def autolabel(rects):    
    for rect in rects:
        height = rect.get_height()
        if rect.get_x()==0.6 and height==91.89:
            plt.text(rect.get_x()+rect.get_width()/2.- 0.45, 1.1*height, '%s' % float(height), fontsize=8)
        else:
            plt.text(rect.get_x()+rect.get_width()/2.- 0.45, 1.03*height, '%s' % float(height), fontsize=8)
############################Function############################

############################Function############################
def allclassificationautolabel(rects):    
    for rect in rects:
        height = rect.get_height()
        if rect.get_x()==0.6 and height==91.89:
            plt.text(rect.get_x()+rect.get_width()/2.- 0.35, 1.1*height, '%s' % float(height), fontsize=9)
        else:
            plt.text(rect.get_x()+rect.get_width()/2.- 0.35, 1.03*height, '%s' % float(height), fontsize=9)
############################Function############################