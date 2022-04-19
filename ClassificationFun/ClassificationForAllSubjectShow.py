from pylab import *

from FileImport.ReadData import *
from ClassificationFun.FeatureProcess import *

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# 显示高度
def autolabel(rects):    
    for rect in rects:
        height = rect.get_height()
        if rect.get_x()==0.6 and height==91.89:
            plt.text(rect.get_x()+rect.get_width()/2.- 0.45, 1.1*height, '%s' % float(height), fontsize=8)
        else:
            plt.text(rect.get_x()+rect.get_width()/2.- 0.45, 1.03*height, '%s' % float(height), fontsize=8)

def ridgeautolabel(rects):    
    for rect in rects:
        print(rect)
        height = rect.get_height()
        if rect.get_x()==-0.4:
            plt.text(rect.get_x()+rect.get_width()/2.- 0.45, 1.1*height, '%s' % float(height), fontsize=8)
        elif rect.get_x()==0.6:
            plt.text(rect.get_x()+rect.get_width()/2.- 0.45, 1.06*height, '%s' % float(height), fontsize=8)
        elif rect.get_x()==1.6:
            plt.text(rect.get_x()+rect.get_width()/2.- 0.45, 1.14*height, '%s' % float(height), fontsize=8)
        elif rect.get_x()==2.6:
            plt.text(rect.get_x()+rect.get_width()/2.- 0.45, 1.135*height, '%s' % float(height), fontsize=8)
        elif rect.get_x()==3.6:
            plt.text(rect.get_x()+rect.get_width()/2.- 0.45, 1.1*height, '%s' % float(height), fontsize=8)
        elif rect.get_x()==4.6:
            plt.text(rect.get_x()+rect.get_width()/2.- 0.45, 1.08*height, '%s' % float(height), fontsize=8)


def allclasserautolabel(rects):    
    for rect in rects:
        height = rect.get_height()
        if rect.get_x()==0.6 and height==91.89:
            plt.text(rect.get_x()+rect.get_width()/2.- 0.35, 1.1*height, '%s' % float(height), fontsize=9)
        else:
            plt.text(rect.get_x()+rect.get_width()/2.- 0.35, 1.03*height, '%s' % float(height), fontsize=9)
                                     
'''
装载每个受试者的分类结果文件
'''
def Load_PerSubject_Classification_Result(filaname):
    with open(filaname, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        dataset = dataset[1:]#从第二行开始读取
        
        LDA_Classer_Accuracy = []
        Ridge_Classer_Accuracy = []
        Log_Classer_Accuracy = []
        Knn_Classer_Accuracy = []
        MLP_Classer_Accuracy = []
        SVM_Classer_Accuracy = []

        x_line=size(dataset,0)  #####行
        y_line=size(dataset,1)  #####列
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

'''
6种分类算法下得到所有受试者的平均分类结果
FilePath:存放数据集的根目录
'''
def Get_AllSubject_Classification_Result(FilePath):
    subject = ['Subject1', 'Subject2', 'Subject3', 'Subject4', 'Subject5', 'Subject6', 'Subject7', 'Subject8', 'Subject9', 'Subject10', 'Subject11', 'Subject12', 'Subject13', 'Subject14', 'Subject15', 'Subject16', 'Subject17', 'Subject18', 'Subject19', 'Subject20']
    length = len(subject)
    
    LDA_Accuracy = []
    Ridge_Accuracy = []
    Log_Accuracy = []
    Knn_Accuracy = []
    MLP_Accuracy = []
    SVM_Accuracy = []
    
    i=0
    while i<length:
        filaname = FilePath + r'\\' + subject[i] + r'\\ClassificationResult.csv'
        LDA_Classer_Accuracy, Ridge_Classer_Accuracy, Log_Classer_Accuracy, Knn_Classer_Accuracy, MLP_Classer_Accuracy, SVM_Classer_Accuracy = Load_PerSubject_Classification_Result(filaname)
        LDA_Accuracy = LDA_Accuracy + [LDA_Classer_Accuracy]
        Ridge_Accuracy = Ridge_Accuracy + [Ridge_Classer_Accuracy]
        Log_Accuracy = Log_Accuracy + [Log_Classer_Accuracy]
        Knn_Accuracy = Knn_Accuracy + [Knn_Classer_Accuracy]
        MLP_Accuracy = MLP_Accuracy + [MLP_Classer_Accuracy]
        SVM_Accuracy = SVM_Accuracy + [SVM_Classer_Accuracy]
        
        i=i+1

    LDA_Accuracy_min = list(np.min(LDA_Accuracy, axis = 0))
    Ridge_Accuracy_min = list(np.min(Ridge_Accuracy, axis = 0))
    Log_Accuracy_min = list(np.min(Log_Accuracy, axis = 0))
    Knn_Accuracy_min = list(np.min(Knn_Accuracy, axis = 0))
    MLP_Accuracy_min = list(np.min(MLP_Accuracy, axis = 0))
    SVM_Accuracy_min = list(np.min(SVM_Accuracy, axis = 0))
    
    LDA_Accuracy_max = list(np.max(LDA_Accuracy, axis = 0))
    Ridge_Accuracy_max = list(np.max(Ridge_Accuracy, axis = 0))
    Log_Accuracy_max = list(np.max(Log_Accuracy, axis = 0))
    Knn_Accuracy_max = list(np.max(Knn_Accuracy, axis = 0))
    MLP_Accuracy_max = list(np.max(MLP_Accuracy, axis = 0))
    SVM_Accuracy_max = list(np.max(SVM_Accuracy, axis = 0))
    
    LDA_Accuracy_mean = list(np.mean(LDA_Accuracy, axis = 0))
    Ridge_Accuracy_mean = list(np.mean(Ridge_Accuracy, axis = 0))
    Log_Accuracy_mean = list(np.mean(Log_Accuracy, axis = 0))
    Knn_Accuracy_mean = list(np.mean(Knn_Accuracy, axis = 0))
    MLP_Accuracy_mean = list(np.mean(MLP_Accuracy, axis = 0))
    SVM_Accuracy_mean = list(np.mean(SVM_Accuracy, axis = 0))
    
    i=0
    while i<len(LDA_Accuracy_mean):
        LDA_Accuracy_mean[i] = round(LDA_Accuracy_mean[i], 2)
        Ridge_Accuracy_mean[i] = round(Ridge_Accuracy_mean[i], 2)
        Log_Accuracy_mean[i] = round(Log_Accuracy_mean[i], 2)
        Knn_Accuracy_mean[i] = round(Knn_Accuracy_mean[i], 2)
        MLP_Accuracy_mean[i] = round(MLP_Accuracy_mean[i], 2)
        SVM_Accuracy_mean[i] = round(SVM_Accuracy_mean[i], 2)
        
        LDA_Accuracy_min[i] = round(LDA_Accuracy_min[i], 2)
        Ridge_Accuracy_min[i] = round(Ridge_Accuracy_min[i], 2)
        Log_Accuracy_min[i] = round(Log_Accuracy_min[i], 2)
        Knn_Accuracy_min[i] = round(Knn_Accuracy_min[i], 2)
        MLP_Accuracy_min[i] = round(MLP_Accuracy_min[i], 2)
        SVM_Accuracy_min[i] = round(SVM_Accuracy_min[i], 2)
        
        LDA_Accuracy_max[i] = round(LDA_Accuracy_max[i], 2)
        Ridge_Accuracy_max[i] = round(Ridge_Accuracy_max[i], 2)
        Log_Accuracy_max[i] = round(Log_Accuracy_max[i], 2)
        Knn_Accuracy_max[i] = round(Knn_Accuracy_max[i], 2)
        MLP_Accuracy_max[i] = round(MLP_Accuracy_max[i], 2)
        SVM_Accuracy_max[i] = round(SVM_Accuracy_max[i], 2)
        
        
        i=i+1
    
    return LDA_Accuracy_mean, Ridge_Accuracy_mean, Log_Accuracy_mean, Knn_Accuracy_mean, MLP_Accuracy_mean, SVM_Accuracy_mean, \
           LDA_Accuracy_min, Ridge_Accuracy_min, Log_Accuracy_min, Knn_Accuracy_min, MLP_Accuracy_min, SVM_Accuracy_min,\
           LDA_Accuracy_max, Ridge_Accuracy_max, Log_Accuracy_max, Knn_Accuracy_max, MLP_Accuracy_max, SVM_Accuracy_max

def Classification_Analysis_Show(FilePath):
    LDA_Accuracy_mean, Ridge_Accuracy_mean, Log_Accuracy_mean, Knn_Accuracy_mean, MLP_Accuracy_mean, SVM_Accuracy_mean,\
    LDA_Accuracy_min, Ridge_Accuracy_min, Log_Accuracy_min, Knn_Accuracy_min, MLP_Accuracy_min, SVM_Accuracy_min,\
    LDA_Accuracy_max, Ridge_Accuracy_max, Log_Accuracy_max, Knn_Accuracy_max, MLP_Accuracy_max, SVM_Accuracy_max = Get_AllSubject_Classification_Result(FilePath)
    
    name_list=['LDA', 'Ridge', 'Log', 'KNN', 'MLP', 'SVM']
    
    classer_result_mean = [LDA_Accuracy_mean[0], Ridge_Accuracy_mean[0], Log_Accuracy_mean[0], Knn_Accuracy_mean[0], MLP_Accuracy_mean[0], SVM_Accuracy_mean[0]]
    classer_result_min = [LDA_Accuracy_min[0], Ridge_Accuracy_min[0], Log_Accuracy_min[0], Knn_Accuracy_min[0], MLP_Accuracy_min[0], SVM_Accuracy_min[0]]
    classer_result_max = [LDA_Accuracy_max[0], Ridge_Accuracy_max[0], Log_Accuracy_max[0], Knn_Accuracy_max[0], MLP_Accuracy_max[0], SVM_Accuracy_max[0]]
    
    marker_size = 10
    marker_size_all = 15
    figure(num=1)
    plt.title('all classifiers')
    name_list=['LDA', 'Ridge', 'Log', 'KNN', 'MLP', 'SVM']
    allclasserautolabel(plt.bar(range(len(classer_result_mean)), classer_result_mean, color=['#45075B','#423D84','#2D6F8E', '#1E998A', '#77D052', '#DFE318'], alpha=0.8, tick_label=name_list))
    polt_index = 0
    plt.plot([polt_index, polt_index], [classer_result_min[polt_index], classer_result_max[polt_index]],  color='black', linestyle='dashed', marker='_', linewidth = 0.7, markersize=marker_size_all)
    polt_index = 1
    plt.plot([polt_index, polt_index], [classer_result_min[polt_index], classer_result_max[polt_index]],  color='black', linestyle='dashed', marker='_', linewidth = 0.7, markersize=marker_size_all)
    polt_index = 2
    plt.plot([polt_index, polt_index], [classer_result_min[polt_index], classer_result_max[polt_index]],  color='black', linestyle='dashed', marker='_', linewidth = 0.7, markersize=marker_size_all)
    polt_index = 3
    plt.plot([polt_index, polt_index], [classer_result_min[polt_index], classer_result_max[polt_index]],  color='black', linestyle='dashed', marker='_', linewidth = 0.7, markersize=marker_size_all)
    polt_index = 4
    plt.plot([polt_index, polt_index], [classer_result_min[polt_index], classer_result_max[polt_index]],  color='black', linestyle='dashed', marker='_', linewidth = 0.7, markersize=marker_size_all)
    polt_index = 5
    plt.plot([polt_index, polt_index], [classer_result_min[polt_index], classer_result_max[polt_index]],  color='black', linestyle='dashed', marker='_', linewidth = 0.7, markersize=marker_size_all)
    plt.ylim(0, 110)
        
    figure(num=2, figsize=(3, 2.5))
    plt.title('LDA')
    name_list=['Mean', 'T1', 'T2', 'T3', 'T4', 'T5']
    autolabel(plt.bar(range(len(LDA_Accuracy_mean)), LDA_Accuracy_mean, color=['#45075B','#423D84','#2D6F8E', '#1E998A', '#77D052', '#DFE318'], alpha=0.8, tick_label=name_list))
    polt_index = 0
    plt.plot([polt_index, polt_index], [LDA_Accuracy_min[polt_index], LDA_Accuracy_max[polt_index]],  color='black', linestyle='dashed', marker='_', linewidth = 0.7, markersize=marker_size)
    polt_index = 1
    plt.plot([polt_index, polt_index], [LDA_Accuracy_min[polt_index], LDA_Accuracy_max[polt_index]],  color='black', linestyle='dashed', marker='_', linewidth = 0.7, markersize=marker_size)
    polt_index = 2
    plt.plot([polt_index, polt_index], [LDA_Accuracy_min[polt_index], LDA_Accuracy_max[polt_index]],  color='black', linestyle='dashed', marker='_', linewidth = 0.7, markersize=marker_size)
    polt_index = 3
    plt.plot([polt_index, polt_index], [LDA_Accuracy_min[polt_index], LDA_Accuracy_max[polt_index]],  color='black', linestyle='dashed', marker='_', linewidth = 0.7, markersize=marker_size)
    polt_index = 4
    plt.plot([polt_index, polt_index], [LDA_Accuracy_min[polt_index], LDA_Accuracy_max[polt_index]],  color='black', linestyle='dashed', marker='_', linewidth = 0.7, markersize=marker_size)
    polt_index = 5
    plt.plot([polt_index, polt_index], [LDA_Accuracy_min[polt_index], LDA_Accuracy_max[polt_index]],  color='black', linestyle='dashed', marker='_', linewidth = 0.7, markersize=marker_size)  
    plt.ylim(0, 110)
    
    figure(num=3, figsize=(3, 2.5))
    plt.title('Ridge')
    name_list=['mean', '1', '2', '3', '4', '5']
    autolabel(plt.bar(range(len(Ridge_Accuracy_mean)), Ridge_Accuracy_mean, color=['#45075B','#423D84','#2D6F8E', '#1E998A', '#77D052', '#DFE318'], alpha=0.8, tick_label=name_list))
    polt_index = 0
    plt.plot([polt_index, polt_index], [Ridge_Accuracy_min[polt_index], Ridge_Accuracy_max[polt_index]],  color='black', linestyle='dashed', marker='_', linewidth = 0.7, markersize=marker_size)
    polt_index = 1
    plt.plot([polt_index, polt_index], [Ridge_Accuracy_min[polt_index], Ridge_Accuracy_max[polt_index]],  color='black', linestyle='dashed', marker='_', linewidth = 0.7, markersize=marker_size)
    polt_index = 2
    plt.plot([polt_index, polt_index], [Ridge_Accuracy_min[polt_index], Ridge_Accuracy_max[polt_index]],  color='black', linestyle='dashed', marker='_', linewidth = 0.7, markersize=marker_size)
    polt_index = 3
    plt.plot([polt_index, polt_index], [Ridge_Accuracy_min[polt_index], Ridge_Accuracy_max[polt_index]],  color='black', linestyle='dashed', marker='_', linewidth = 0.7, markersize=marker_size)
    polt_index = 4
    plt.plot([polt_index, polt_index], [Ridge_Accuracy_min[polt_index], Ridge_Accuracy_max[polt_index]],  color='black', linestyle='dashed', marker='_', linewidth = 0.7, markersize=marker_size)
    polt_index = 5
    plt.plot([polt_index, polt_index], [Ridge_Accuracy_min[polt_index], Ridge_Accuracy_max[polt_index]],  color='black', linestyle='dashed', marker='_', linewidth = 0.7, markersize=marker_size)  
    plt.ylim(0, 110)
    
    figure(num=4, figsize=(3, 2.5))
    plt.title('Log')
    name_list=['mean', '1', '2', '3', '4', '5']
    autolabel(plt.bar(range(len(Log_Accuracy_mean)), Log_Accuracy_mean, color=['#45075B','#423D84','#2D6F8E', '#1E998A', '#77D052', '#DFE318'], alpha=0.8, tick_label=name_list))
    polt_index = 0
    plt.plot([polt_index, polt_index], [Log_Accuracy_min[polt_index], Log_Accuracy_max[polt_index]],  color='black', linestyle='dashed', marker='_', linewidth = 0.7, markersize=marker_size)
    polt_index = 1
    plt.plot([polt_index, polt_index], [Log_Accuracy_min[polt_index], Log_Accuracy_max[polt_index]],  color='black', linestyle='dashed', marker='_', linewidth = 0.7, markersize=marker_size)
    polt_index = 2
    plt.plot([polt_index, polt_index], [Log_Accuracy_min[polt_index], Log_Accuracy_max[polt_index]],  color='black', linestyle='dashed', marker='_', linewidth = 0.7, markersize=marker_size)
    polt_index = 3
    plt.plot([polt_index, polt_index], [Log_Accuracy_min[polt_index], Log_Accuracy_max[polt_index]],  color='black', linestyle='dashed', marker='_', linewidth = 0.7, markersize=marker_size)
    polt_index = 4
    plt.plot([polt_index, polt_index], [Log_Accuracy_min[polt_index], Log_Accuracy_max[polt_index]],  color='black', linestyle='dashed', marker='_', linewidth = 0.7, markersize=marker_size)
    polt_index = 5
    plt.plot([polt_index, polt_index], [Log_Accuracy_min[polt_index], Log_Accuracy_max[polt_index]],  color='black', linestyle='dashed', marker='_', linewidth = 0.7, markersize=marker_size)  
    plt.ylim(0, 110)
    
    figure(num=5, figsize=(3, 2.5))
    plt.title('KNN')
    name_list=['mean', '1', '2', '3', '4', '5']
    autolabel(plt.bar(range(len(Knn_Accuracy_mean)), Knn_Accuracy_mean, color=['#45075B','#423D84','#2D6F8E', '#1E998A', '#77D052', '#DFE318'], alpha=0.8, tick_label=name_list))
    polt_index = 0
    plt.plot([polt_index, polt_index], [Knn_Accuracy_min[polt_index], Knn_Accuracy_max[polt_index]],  color='black', linestyle='dashed', marker='_', linewidth = 0.7, markersize=marker_size)
    polt_index = 1
    plt.plot([polt_index, polt_index], [Knn_Accuracy_min[polt_index], Knn_Accuracy_max[polt_index]],  color='black', linestyle='dashed', marker='_', linewidth = 0.7, markersize=marker_size)
    polt_index = 2
    plt.plot([polt_index, polt_index], [Knn_Accuracy_min[polt_index], Knn_Accuracy_max[polt_index]],  color='black', linestyle='dashed', marker='_', linewidth = 0.7, markersize=marker_size)
    polt_index = 3
    plt.plot([polt_index, polt_index], [Knn_Accuracy_min[polt_index], Knn_Accuracy_max[polt_index]],  color='black', linestyle='dashed', marker='_', linewidth = 0.7, markersize=marker_size)
    polt_index = 4
    plt.plot([polt_index, polt_index], [Knn_Accuracy_min[polt_index], Knn_Accuracy_max[polt_index]],  color='black', linestyle='dashed', marker='_', linewidth = 0.7, markersize=marker_size)
    polt_index = 5
    plt.plot([polt_index, polt_index], [Knn_Accuracy_min[polt_index], Knn_Accuracy_max[polt_index]],  color='black', linestyle='dashed', marker='_', linewidth = 0.7, markersize=marker_size)  
    plt.ylim(0, 110)
    
    figure(num=6, figsize=(3, 2.5))
    plt.title('MLP')
    name_list=['mean', '1', '2', '3', '4', '5']
    autolabel(plt.bar(range(len(MLP_Accuracy_mean)), MLP_Accuracy_mean, color=['#45075B','#423D84','#2D6F8E', '#1E998A', '#77D052', '#DFE318'], alpha=0.8, tick_label=name_list))
    polt_index = 0
    plt.plot([polt_index, polt_index], [MLP_Accuracy_min[polt_index], MLP_Accuracy_max[polt_index]],  color='black', linestyle='dashed', marker='_', linewidth = 0.7, markersize=marker_size)
    polt_index = 1
    plt.plot([polt_index, polt_index], [MLP_Accuracy_min[polt_index], MLP_Accuracy_max[polt_index]],  color='black', linestyle='dashed', marker='_', linewidth = 0.7, markersize=marker_size)
    polt_index = 2
    plt.plot([polt_index, polt_index], [MLP_Accuracy_min[polt_index], MLP_Accuracy_max[polt_index]],  color='black', linestyle='dashed', marker='_', linewidth = 0.7, markersize=marker_size)
    polt_index = 3
    plt.plot([polt_index, polt_index], [MLP_Accuracy_min[polt_index], MLP_Accuracy_max[polt_index]],  color='black', linestyle='dashed', marker='_', linewidth = 0.7, markersize=marker_size)
    polt_index = 4
    plt.plot([polt_index, polt_index], [MLP_Accuracy_min[polt_index], MLP_Accuracy_max[polt_index]],  color='black', linestyle='dashed', marker='_', linewidth = 0.7, markersize=marker_size)
    polt_index = 5
    plt.plot([polt_index, polt_index], [MLP_Accuracy_min[polt_index], MLP_Accuracy_max[polt_index]],  color='black', linestyle='dashed', marker='_', linewidth = 0.7, markersize=marker_size)  
    plt.ylim(0, 110)
    
    figure(num=7, figsize=(3, 2.5))
    plt.title('SVM')
    name_list=['mean', '1', '2', '3', '4', '5']
    autolabel(plt.bar(range(len(SVM_Accuracy_mean)), SVM_Accuracy_mean, color=['#45075B','#423D84','#2D6F8E', '#1E998A', '#77D052', '#DFE318'], alpha=0.8, tick_label=name_list))
    polt_index = 0
    plt.plot([polt_index, polt_index], [SVM_Accuracy_min[polt_index], SVM_Accuracy_max[polt_index]],  color='black', linestyle='dashed', marker='_', linewidth = 0.7, markersize=marker_size)
    polt_index = 1
    plt.plot([polt_index, polt_index], [SVM_Accuracy_min[polt_index], SVM_Accuracy_max[polt_index]],  color='black', linestyle='dashed', marker='_', linewidth = 0.7, markersize=marker_size)
    polt_index = 2
    plt.plot([polt_index, polt_index], [SVM_Accuracy_min[polt_index], SVM_Accuracy_max[polt_index]],  color='black', linestyle='dashed', marker='_', linewidth = 0.7, markersize=marker_size)
    polt_index = 3
    plt.plot([polt_index, polt_index], [SVM_Accuracy_min[polt_index], SVM_Accuracy_max[polt_index]],  color='black', linestyle='dashed', marker='_', linewidth = 0.7, markersize=marker_size)
    polt_index = 4
    plt.plot([polt_index, polt_index], [SVM_Accuracy_min[polt_index], SVM_Accuracy_max[polt_index]],  color='black', linestyle='dashed', marker='_', linewidth = 0.7, markersize=marker_size)
    polt_index = 5
    plt.plot([polt_index, polt_index], [SVM_Accuracy_min[polt_index], SVM_Accuracy_max[polt_index]],  color='black', linestyle='dashed', marker='_', linewidth = 0.7, markersize=marker_size)  
    plt.ylim(0, 110)
    
    plt.show()
    
def Classification_Analysis_Save(FilePath): 
    LDA_Accuracy_mean, Ridge_Accuracy_mean, Log_Accuracy_mean, Knn_Accuracy_mean, MLP_Accuracy_mean, SVM_Accuracy_mean,\
    LDA_Accuracy_min, Ridge_Accuracy_min, Log_Accuracy_min, Knn_Accuracy_min, MLP_Accuracy_min, SVM_Accuracy_min,\
    LDA_Accuracy_max, Ridge_Accuracy_max, Log_Accuracy_max, Knn_Accuracy_max, MLP_Accuracy_max, SVM_Accuracy_max = Get_AllSubject_Classification_Result(FilePath)
    
    
    lenght = len(LDA_Accuracy_mean)
    LDA_Accuracy_mean_ = np.array(LDA_Accuracy_mean) / 100
    Ridge_Accuracy_mean_ = np.array(Ridge_Accuracy_mean) / 100
    Log_Accuracy_mean_ = np.array(Log_Accuracy_mean) / 100
    Knn_Accuracy_mean_ = np.array(Knn_Accuracy_mean) / 100
    MLP_Accuracy_mean_ = np.array(MLP_Accuracy_mean) / 100
    SVM_Accuracy_mean_ = np.array(SVM_Accuracy_mean) / 100
    
    LDA_Accuracy_mean.append(round(np.std(LDA_Accuracy_mean_[1:]), 6))
    Ridge_Accuracy_mean.append(round(np.std(Ridge_Accuracy_mean_[1:]), 6))
    Log_Accuracy_mean.append(round(np.std(Log_Accuracy_mean_[1:]), 6))
    Knn_Accuracy_mean.append(round(np.std(Knn_Accuracy_mean_[1:]), 6))
    MLP_Accuracy_mean.append(round(np.std(MLP_Accuracy_mean_[1:]), 6))
    SVM_Accuracy_mean.append(round(np.std(SVM_Accuracy_mean_[1:]), 6))
    
    LDA_Accuracy_min_ = np.array(LDA_Accuracy_min) / 100
    Ridge_Accuracy_min_ = np.array(Ridge_Accuracy_min) / 100
    Log_Accuracy_min_ = np.array(Log_Accuracy_min) / 100
    Knn_Accuracy_min_ = np.array(Knn_Accuracy_min) / 100
    MLP_Accuracy_min_ = np.array(MLP_Accuracy_min) / 100
    SVM_Accuracy_min_ = np.array(SVM_Accuracy_min) / 100
    
    LDA_Accuracy_min.append(round(np.std(LDA_Accuracy_min_[1:]), 6))
    Ridge_Accuracy_min.append(round(np.std(Ridge_Accuracy_min_[1:]), 6))
    Log_Accuracy_min.append(round(np.std(Log_Accuracy_min_[1:]), 6))
    Knn_Accuracy_min.append(round(np.std(Knn_Accuracy_min_[1:]), 6))
    MLP_Accuracy_min.append(round(np.std(MLP_Accuracy_min_[1:]), 6))
    SVM_Accuracy_min.append(round(np.std(SVM_Accuracy_min_[1:]), 6))
    
    LDA_Accuracy_max_ = np.array(LDA_Accuracy_max) / 100
    Ridge_Accuracy_max_ = np.array(Ridge_Accuracy_max) / 100
    Log_Accuracy_max_ = np.array(Log_Accuracy_max) / 100
    Knn_Accuracy_max_ = np.array(Knn_Accuracy_max) / 100
    MLP_Accuracy_max_ = np.array(MLP_Accuracy_max) / 100
    SVM_Accuracy_max_ = np.array(SVM_Accuracy_max) / 100
    
    LDA_Accuracy_max.append(round(np.std(LDA_Accuracy_max_[1:]), 6))
    Ridge_Accuracy_max.append(round(np.std(Ridge_Accuracy_max_[1:]), 6))
    Log_Accuracy_max.append(round(np.std(Log_Accuracy_max_[1:]), 6))
    Knn_Accuracy_max.append(round(np.std(Knn_Accuracy_max_[1:]), 6))
    MLP_Accuracy_max.append(round(np.std(MLP_Accuracy_max_[1:]), 6))
    SVM_Accuracy_max.append(round(np.std(SVM_Accuracy_max_[1:]), 6))

    print(LDA_Accuracy_mean)
    print(LDA_Accuracy_min)
    print(LDA_Accuracy_max)
    i=0
    while i<lenght:
        LDA_Accuracy_mean_
        
        
        i=i+1
    
    #################将相关参数保存到CSV文件中
    Accuracy_mean = ['mean', 'T1', 'T2', 'T3', 'T4', 'T5', 'std']
    classer_type = ['mean-min-max', 'LDA mean', 'Ridge mean', 'Log mean', 'knn mean', 'MLP mean', 'SVM mean', \
                     'LDA min', 'Ridge min', 'Log min', 'knn min', 'MLP min', 'SVM min',\
                     'LDA max', 'Ridge max', 'Log max', 'knn max', 'MLP max', 'SVM max']
    data = [Accuracy_mean, LDA_Accuracy_mean, Ridge_Accuracy_mean, Log_Accuracy_mean, Knn_Accuracy_mean, MLP_Accuracy_mean, SVM_Accuracy_mean,\
            LDA_Accuracy_min, Ridge_Accuracy_min, Log_Accuracy_min, Knn_Accuracy_min, MLP_Accuracy_min, SVM_Accuracy_min,\
            LDA_Accuracy_max, Ridge_Accuracy_max, Log_Accuracy_max, Knn_Accuracy_max, MLP_Accuracy_max, SVM_Accuracy_max]
    data = List_Transpose(data)
    data.insert(0, classer_type)
    data = List_Transpose(data)
    
    with open(FilePath+r'\\AllClassificationResult.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerows(data)
        csvfile.close()
    #################将相关参数保存到CSV文件中






