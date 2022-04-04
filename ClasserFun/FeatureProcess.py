from FileImport.ReadData import *
import random
from _operator import itemgetter

def TrainAndTestSetSelect(featurefilename, mode=1):
    feature_value, feature_name, singleCh_feature_name = LoadFeatures(featurefilename)
    normalization_feature_value = FeatureNormalization(feature_value)
    y = normalization_feature_value[-1] #标签
    x = List_Transpose(normalization_feature_value[:-3]) #sEMG特征值
    sample_num = len(y)
    
    trainSet_x = []
    trainSet_y = []
    testSet_x = []
    testSet_y =[]
    
    ratio = 6#训练集所占的比例，ratio成
    
    i=0
    while i < sample_num:
        if mode == 1:            
            if i%2 == 0:
                trainSet_x.append(x[i])
                trainSet_y.append(y[i])
            else:
                testSet_x.append(x[i])
                testSet_y.append(y[i])
        elif mode == 2:
            if random.randint(0, 10) <=ratio:
                trainSet_x.append(x[i])
                trainSet_y.append(y[i])
            else:
                testSet_x.append(x[i])
                testSet_y.append(y[i])
        
        i=i+1
        
    return trainSet_x, trainSet_y, testSet_x, testSet_y

'''
根据预测值与实际值获得所有类别的平均准确率
'''
def GetAllMeanAccuracy(predict_label, true_label):
    length = len(predict_label)
    
    accuracy_num = 0
    i=0
    while i<length:
        if predict_label[i] == true_label[i]:
            accuracy_num = accuracy_num+1  
                 
        print('预测值：{}，实际值：{}'.format(predict_label[i], true_label[i]))
        i=i+1
    
    accuracy = round(accuracy_num/length, 4) * 100
    
    print("准确率：{}%".format(accuracy))
    
    return round(accuracy, 2)

'''
根据预测值与实际值获得每种类别的平均准确率
'''
def GetAccuracyForMultiClass(predict_label, true_label):
    
    n_sample=len(true_label)
    
    class_actual_prediction={} #形式：{class:[num_actual,num_prediction]}

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
    print('平均及各个动作识别率：{}'.format(accuracy))
    accuracy=list(map(list,zip(*accuracy))) 
             
    return accuracy #accuracy[0]是类别名称，accuracy[1]是各个类别的识别率，放大了100倍
#####################################################################计算识别精度
