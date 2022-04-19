from FileImport.ReadData import *
import random
from _operator import itemgetter

from ActivityDetect.ForceActivity import *


'''
对所有类进行训练集和测试集选择
mode=1:偶数作训练，奇数作测试
mode=2:按一定比例随机挑选测试集与训练集
'''
def TrainAndTestSetSelectForAllClass(featurefilename, mode=1):
    feature_value, feature_name, singleCh_feature_name = LoadFeatures(featurefilename)
    normalization_feature_value = FeatureNormalizationExceptForceIntegral(feature_value)#力随时间的积分值未进行归一化
    y_label = normalization_feature_value[-1] #类别标签
    y = normalization_feature_value[-2] #力随时间的积分值
    x = List_Transpose(normalization_feature_value[:-2]) #sEMG特征值
    sample_num = len(y)
    
    trainSet_x = []
    trainSet_y = []
    trainSet_y_label = []
    testSet_x = []
    testSet_y =[]
    testSet_y_label =[]
    
    ratio = 5#训练集所占的比例，ratio/10的比例
    
    i=0
    while i < sample_num:
        if mode == 1:            
            if i%2 == 1:
                trainSet_x.append(x[i])
                trainSet_y.append(y[i])
                trainSet_y_label.append(y_label[i])
            else:
                testSet_x.append(x[i])
                testSet_y.append(y[i])
                testSet_y_label.append(y_label[i])
        elif mode == 2:
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

'''
从获得的所有类中，将各类分开
predict_y:每一行代表一类的预测值
true_y:每一行代表一类的真实值
'''
def SingleClassSeparate(predict_y, true_y, true_y_label):
    y_label = [1, 2, 3, 4, 5]#五类样本的标签
    
    length = len(true_y_label)
    
    predict_y_class1 = [] #类别1
    predict_y_class2 = [] #类别2
    predict_y_class3 = [] #类别3
    predict_y_class4 = [] #类别4
    predict_y_class5 = [] #类别5
    
    true_y_class1 = [] #类别1
    true_y_class2 = [] #类别2
    true_y_class3 = [] #类别3
    true_y_class4 = [] #类别4
    true_y_class5 = [] #类别5
    
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



'''
将每一类样本分开
返回值中索引0表示第一类，索引1表示第二类......
'''
def EachClassSampleSeparate(featurefilename):
    i=0
    j=0
    with open(featurefilename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        feature_name = dataset[0]
        feature_value = dataset[1:] #特征值数值
        
        x_line=size(feature_value,0)  #####行，有多少种特征值
        y_line=size(feature_value,1)  #####列，一种特征值提取了多少个
        while i<x_line:
            j=0
            while j<y_line:
                feature_value[i][j]=np.float(feature_value[i][j])
               
                j=j+1 
            i=i+1            
    csvfile.close()
    print('数据总长度：'+str(len(dataset))) 
    
    feature_value = List_Transpose(feature_value)#特征值转置
    del feature_value[-3]#去掉力的均值
    feature_value = List_Transpose(feature_value)#特征值转置
    
    each_class_feature = []
    each_class_sample_num = 80#每一类有80个样本
    class_num = 5#一共有5类
    i=0
    while i<class_num:
        each_class_feature.append(feature_value[i*each_class_sample_num : (i+1)*each_class_sample_num])
        
        i=i+1
    
    return each_class_feature
     
'''
对单个类进行训练集和测试集选择
class_label:类别标签
mode=1:偶数作训练，奇数作测试
mode=2:按一定比例随机挑选测试集与训练集
'''
def TrainAndTestSetSelectForSingleClass(featurefilename, class_label=1, mode=1):  
    each_class_feature = EachClassSampleSeparate(featurefilename)
    feature_value = each_class_feature[class_label-1]
    feature_value = List_Transpose(feature_value)#特征值转置
    normalization_feature_value = FeatureNormalizationExceptForceIntegral(feature_value)#力随时间的积分值未进行归一化 
    y_label = normalization_feature_value[-1] #类别标签
    y = normalization_feature_value[-2] #力随时间的积分值
    x = List_Transpose(normalization_feature_value[:-2]) #sEMG特征值
    sample_num = len(y)
    
    trainSet_x = []
    trainSet_y = []
    trainSet_y_label = []
    testSet_x = []
    testSet_y =[]
    testSet_y_label =[]
    
    ratio = 5#训练集所占的比例，ratio成
    
    i=0
    while i < sample_num:
        if mode == 1:            
            if i%2 == 0:
                trainSet_x.append(x[i])
                trainSet_y.append(y[i])
                trainSet_y_label.append(y_label[i])
            else:
                testSet_x.append(x[i])
                testSet_y.append(y[i])
                testSet_y_label.append(y_label[i])
        elif mode == 2:
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

'''
求预测值与实际值的差值与标准差
'''
def GetStdForPredictAndTrue(predict_y, true_y):
    length = len(true_y)
    error_value = np.array(predict_y) - np.array(true_y)#预测值与真实值的误差
    std_value = np.std(error_value)#预测值与实际值的标准差
    return error_value, round(std_value)
    
        
        
        
    
    
    
    
    
    