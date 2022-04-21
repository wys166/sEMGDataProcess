# _*_ coding=utf-8 _*_
import numpy as np
from pylab import *
import csv
import random
import operator
import csv
import matplotlib
from scipy import signal
from scipy import interpolate


########函数#########
def LoadRawDataSetByChNum(filename, CH=2):   ####################力信号、两路包络信号和两路原始信号
    Envelope_1=[]
    Raw_1=[]
    Envelope_2=[]
    Raw_2=[]   
        
    i=0
    j=0
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        x_line=size(dataset,0)  #####行
        y_line=size(dataset,1)  #####列
        while i<x_line:
            j=0
            while j<y_line:
                dataset[i][j]=np.int(dataset[i][j])
                if CH==2:
                    if j==0:
                        Raw_1.append(dataset[i][j]-2076)                                           
                    elif j==1:
                        Envelope_1.append(dataset[i][j])
                elif CH==4:    
                    if j==0:
                        Raw_1.append(dataset[i][j]-2076)                                           
                    elif j==1:
                        Envelope_1.append(dataset[i][j])
                    elif j==2:
                        Raw_2.append(dataset[i][j]-2053)                    
                    elif j==3:
                        Envelope_2.append(dataset[i][j])                    
                    
                j=j+1 
            i=i+1            
    csvfile.close()
    print('数据总长度：'+str(len(dataset))) 
    if CH==2:
        return Raw_1,Envelope_1
    elif CH==4:
        return Raw_1,Envelope_1,Raw_2,Envelope_2
########函数#########

'''
filename：文件名
force/Raw/Envelope：力/表面肌电原始信号/表面肌电包络信号
'''
def LoadDataSetAfterProcess(filename):   ####################力信号、两路包络信号和两路原始信号
    force=[]
    Envelope_1=[]
    Raw_1=[]
    Envelope_2=[]
    Raw_2=[]
        
    i=0
    j=0
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        x_line=size(dataset,0)  #####行
        y_line=size(dataset,1)  #####列
        while i<x_line:
            j=0
            while j<y_line:
                dataset[i][j]=np.float(dataset[i][j])
                dataset[i][j] = np.int(dataset[i][j])
                if j==0:
                    force.append(dataset[i][j])                          
                elif j==1:
                    Raw_1.append(dataset[i][j])
                elif j==2:
                    Envelope_1.append(dataset[i][j])                    
                elif j==3:
                    Raw_2.append(dataset[i][j])
                elif j==4:
                    Envelope_2.append(dataset[i][j])
                    
                j=j+1 
            i=i+1            
    csvfile.close()
    print('数据总长度：'+str(len(dataset))) 
    
    return force, Raw_1,Envelope_1,Raw_2,Envelope_2 

########列表转置########
def List_Transpose(data):
    data = list(map(list, zip(*data)))    #转置
    return data
########列表转置########

'''
featurefilename：提取的特征值文件
返回值：
feature_value：按行排列的特征值,最后一行是标签
feature_name：特征值文件中第一行的特征名称
singleCh_feature_name：从每个传感器中提取的特征值名称
'''
def LoadFeatures(featurefilename):   ####################力信号、两路包络信号和两路原始信号  
    i=0
    j=0
    with open(featurefilename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        feature_name = dataset[0]
        feature_value = dataset[1:] #特征值数值
        feature_value = List_Transpose(feature_value)#特征值转置
        
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
    
    Ch = 2 #传感器通道数
    length = (len(feature_name) - 3)/Ch
    singleCh_feature_name = []
    i = 0
    while i<length:
        feature_name_str = feature_name[i]
        singleCh_feature_name.append(str(i) + '-' + feature_name_str[0:-1]) 
        
        i=i+1
    
    return feature_value, feature_name, singleCh_feature_name

#########将feature_value特征值归一化#########
'''
normalization_feature_value中的最后一行是标签label,此函数将力随时间的积分值也进行了归一化
'''
def FeatureNormalization(feature_value):
    length = len(feature_value)
    
    normalization_feature_value = []
    i = 0
    while i<length-1:
        max_feature = max(feature_value[i])
        min_feature = min(feature_value[i])
        normalization_feature_value.append(list((array(feature_value[i])-min_feature)/(max_feature-min_feature)))
        
        
        i=i+1
        
    normalization_feature_value.append(feature_value[-1])
    
    return normalization_feature_value #返回归一化特征值
#########将feature_value特征值归一化#########

#########将feature_value特征值归一化#########
'''
normalization_feature_value中的最后一行是类别标签，倒数第二行是力积分值,此函数只将特征进行了归一化，力随时间的积分值没有进行归一化
'''
def FeatureNormalizationExceptForceIntegral(feature_value):
    length = len(feature_value)
    
    normalization_feature_value = []
    i = 0
    while i<length-3:
        max_feature = max(feature_value[i])
        min_feature = min(feature_value[i])
        normalization_feature_value.append(list((array(feature_value[i])-min_feature)/(max_feature-min_feature)))
        
        
        i=i+1
        
    normalization_feature_value.append(list(array(feature_value[-2])))#力积分值
    normalization_feature_value.append(feature_value[-1])#类别标签
    
    return normalization_feature_value #返回归一化特征值
#########将feature_value特征值归一化#########

#########将回归得到的结果读取处理#########
'''
y_label:每一类的标签
force_integral_true:真实的力累加
force_integral_predict:预测的力累加
slope_true:真实的力斜率
slope_predict:预测的力斜率
'''
def LoadRegressionResultFile(RegressionResultFilename):
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
        dataset = dataset[1:]#从第二行开始读取
        
        x_line=size(dataset,0)  #####行
        y_line=size(dataset,1)  #####列
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
    print('数据总长度：'+str(len(dataset))) 
    
    return y_label, force_integral_true, force_integral_predict, slope_true, slope_predict
#########将回归得到的结果读取处理#########




