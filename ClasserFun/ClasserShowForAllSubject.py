from pylab import *

from FileImport.ReadData import *
from ClasserFun.FeatureProcess import *

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
        plt.text(rect.get_x()+rect.get_width()/2.- 0.2, 1.03*height, '%s' % float(height))

'''
装载每个受试者的分类结果文件
'''
def Load_PerSubject_Classer_Result(filaname):
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
'''
def Get_AllSubject_Classer_Result(FilePath, subject):

    length = len(subject)
    
    LDA_Accuracy = []
    Ridge_Accuracy = []
    Log_Accuracy = []
    Knn_Accuracy = []
    MLP_Accuracy = []
    SVM_Accuracy = []
    
    i=0
    while i<length:
        filaname = FilePath + r'\\' + subject[i] + r'\\ClasserResult.csv'
        LDA_Classer_Accuracy, Ridge_Classer_Accuracy, Log_Classer_Accuracy, Knn_Classer_Accuracy, MLP_Classer_Accuracy, SVM_Classer_Accuracy = Load_PerSubject_Classer_Result(filaname)
        LDA_Accuracy = LDA_Accuracy + [LDA_Classer_Accuracy]
        Ridge_Accuracy = Ridge_Accuracy + [Ridge_Classer_Accuracy]
        Log_Accuracy = Log_Accuracy + [Log_Classer_Accuracy]
        Knn_Accuracy = Knn_Accuracy + [Knn_Classer_Accuracy]
        MLP_Accuracy = MLP_Accuracy + [MLP_Classer_Accuracy]
        SVM_Accuracy = SVM_Accuracy + [SVM_Classer_Accuracy]
        
        i=i+1
    
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
        
        i=i+1
    
    return LDA_Accuracy_mean, Ridge_Accuracy_mean, Log_Accuracy_mean, Knn_Accuracy_mean, MLP_Accuracy_mean, SVM_Accuracy_mean


def Classer_Analysis_Show(FilePath):
    LDA_Accuracy_mean, Ridge_Accuracy_mean, Log_Accuracy_mean, Knn_Accuracy_mean, MLP_Accuracy_mean, SVM_Accuracy_mean = Get_AllSubject_Classer_Result(FilePath)
    
    name_list=['LDA', 'Ridge', 'Log', 'KNN', 'MLP', 'SVM']
    
    classer_result_mean = [LDA_Accuracy_mean[0], Ridge_Accuracy_mean[0], Log_Accuracy_mean[0], Knn_Accuracy_mean[0], MLP_Accuracy_mean[0], SVM_Accuracy_mean[0]]
    
    figure(1)
    plt.title('all classers')
    name_list=['LDA', 'Ridge', 'Log', 'KNN', 'MLP', 'SVM']
    autolabel(plt.bar(range(len(classer_result_mean)), classer_result_mean, color='rgbkyc', tick_label=name_list))
    
    figure(2)
    plt.title('LDA')
    name_list=['mean', '1', '2', '3', '4', '5']
    autolabel(plt.bar(range(len(LDA_Accuracy_mean)), LDA_Accuracy_mean, color='rgbkyc', tick_label=name_list))
    
    figure(3)
    plt.title('Ridge')
    name_list=['mean', '1', '2', '3', '4', '5']
    autolabel(plt.bar(range(len(Ridge_Accuracy_mean)), Ridge_Accuracy_mean, color='rgbkyc', tick_label=name_list))
    
    figure(4)
    plt.title('Log')
    name_list=['mean', '1', '2', '3', '4', '5']
    autolabel(plt.bar(range(len(Log_Accuracy_mean)), Log_Accuracy_mean, color='rgbkyc', tick_label=name_list))
    
    figure(5)
    plt.title('KNN')
    name_list=['mean', '1', '2', '3', '4', '5']
    autolabel(plt.bar(range(len(Knn_Accuracy_mean)), Knn_Accuracy_mean, color='rgbkyc', tick_label=name_list))
    
    figure(6)
    plt.title('MLP')
    name_list=['mean', '1', '2', '3', '4', '5']
    autolabel(plt.bar(range(len(MLP_Accuracy_mean)), MLP_Accuracy_mean, color='rgbkyc', tick_label=name_list))
    
    figure(7)
    plt.title('SVM')
    name_list=['mean', '1', '2', '3', '4', '5']
    autolabel(plt.bar(range(len(SVM_Accuracy_mean)), SVM_Accuracy_mean, color='rgbkyc', tick_label=name_list))
    
    
    plt.show()
    
    
Classer_Analysis_Show(r'D:\\sEMGData') 


