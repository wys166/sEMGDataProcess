# _*_ coding=utf-8 _*_
import numpy as np
from pylab import *
import csv
import random
import operator
import csv
import matplotlib
from scipy import signal
'''
Function
'''

########Function#########
'''
Load Reference file
'''
def LoadReferenceDataSetByChNum(filename, CH=2):   ####################Two channel
    Envelope_1=[]
    Raw_1=[]
    Envelope_2=[]
    Raw_2=[]   
        
    i=0
    j=0
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        x_line=size(dataset,0)  
        y_line=size(dataset,1)  
        while i<x_line:
            j=0
            while j<y_line:
                dataset[i][j]=np.int(dataset[i][j])
                if CH==2:
                    if j==0:
                        Raw_1.append(dataset[i][j])                                           
                    elif j==1:
                        Envelope_1.append(dataset[i][j])
                elif CH==4:    
                    if j==0:
                        Raw_1.append(dataset[i][j])                                           
                    elif j==1:
                        Envelope_1.append(dataset[i][j])
                    elif j==2:
                        Raw_2.append(dataset[i][j])                    
                    elif j==3:
                        Envelope_2.append(dataset[i][j])                    
                    
                j=j+1 
            i=i+1            
    csvfile.close()
    print('Data Length:'+str(len(dataset))) 
    if CH==2:
        return Raw_1,Envelope_1
    elif CH==4:
        return Raw_1,Envelope_1,Raw_2,Envelope_2
########Function#########

########Function#########
'''
Load DataSet
'''
def LoadDataSet(filename):   
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
        x_line=size(dataset,0)  
        y_line=size(dataset,1)  
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
    print('Data Length:'+str(len(dataset))) 
    
    return force, Raw_1,Envelope_1,Raw_2,Envelope_2 
########Function#########

########Function#########
'''
list Transpose
'''
def List_Transpose(data):
    data = list(map(list, zip(*data)))    
    return data
########Function#########

########Function#########
'''
Load Feature file
'''
def LoadFeatures(featurefilename):    
    i=0
    j=0
    with open(featurefilename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        feature_name = dataset[0]
        feature_value = dataset[1:] 
        feature_value = List_Transpose(feature_value)
        
        x_line=size(feature_value,0)  
        y_line=size(feature_value,1)  
        while i<x_line:
            j=0
            while j<y_line:
                feature_value[i][j]=np.float(feature_value[i][j])
               
                j=j+1 
            i=i+1            
    csvfile.close()
    print('Data Length:'+str(len(dataset))) 
    
    Ch = 2 
    length = (len(feature_name) - 3)/Ch
    singleCh_feature_name = []
    i = 0
    while i<length:
        feature_name_str = feature_name[i]
        singleCh_feature_name.append(str(i) + '-' + feature_name_str[0:-1]) 
        
        i=i+1
    
    return feature_value, feature_name, singleCh_feature_name
########Function#########

########Function#########
'''
All Features Normalization
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
    
    return normalization_feature_value 
########Function#########

########Function#########
'''
Feature Normalization Except the 'IFORCE' feature
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
        
    normalization_feature_value.append(list(array(feature_value[-2])))
    normalization_feature_value.append(feature_value[-1])
    
    return normalization_feature_value 
########Function#########






