from pylab import *

from Classification.ClassificationProcessFun import *


'''
Show overall classification result
filename=r'D:\\DataSet\\ClassificationResult.csv'
'''
def OverallClassificationResultShow(filename):
    accuracy_mean, accuracy_min, accuracy_max = LoadOverallClassificationResult(filename)
 
    LDA_Accuracy_mean = accuracy_mean[0]
    Ridge_Accuracy_mean = accuracy_mean[1]
    Log_Accuracy_mean = accuracy_mean[2]
    Knn_Accuracy_mean = accuracy_mean[3]
    MLP_Accuracy_mean = accuracy_mean[4]
    SVM_Accuracy_mean = accuracy_mean[5]
    
    LDA_Accuracy_min = accuracy_min[0]
    Ridge_Accuracy_min = accuracy_min[1]
    Log_Accuracy_min = accuracy_min[2]
    Knn_Accuracy_min = accuracy_min[3]
    MLP_Accuracy_min = accuracy_min[4]
    SVM_Accuracy_min = accuracy_min[5]
    
    LDA_Accuracy_max = accuracy_max[0]
    Ridge_Accuracy_max = accuracy_max[1]
    Log_Accuracy_max = accuracy_max[2]
    Knn_Accuracy_max = accuracy_max[3]
    MLP_Accuracy_max = accuracy_max[4]
    SVM_Accuracy_max = accuracy_max[5]
    
    name_list=['LDA', 'Ridge', 'Log', 'KNN', 'MLP', 'SVM']
    
    classer_result_mean = [LDA_Accuracy_mean[0], Ridge_Accuracy_mean[0], Log_Accuracy_mean[0], Knn_Accuracy_mean[0], MLP_Accuracy_mean[0], SVM_Accuracy_mean[0]]
    classer_result_min = [LDA_Accuracy_min[0], Ridge_Accuracy_min[0], Log_Accuracy_min[0], Knn_Accuracy_min[0], MLP_Accuracy_min[0], SVM_Accuracy_min[0]]
    classer_result_max = [LDA_Accuracy_max[0], Ridge_Accuracy_max[0], Log_Accuracy_max[0], Knn_Accuracy_max[0], MLP_Accuracy_max[0], SVM_Accuracy_max[0]]
    
    marker_size = 10
    marker_size_all = 15
    figure(num=1)
    plt.title('All classifiers')
    name_list=['LDA', 'Ridge', 'Log', 'KNN', 'MLP', 'SVM']
    allclassificationautolabel(plt.bar(range(len(classer_result_mean)), classer_result_mean, color=['#45075B','#423D84','#2D6F8E', '#1E998A', '#77D052', '#DFE318'], alpha=0.8, tick_label=name_list))
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
    
    
'''
input: r'D:\\DataSet\\ClassificationResult.csv'
'''
OverallClassificationResultShow(r'D:\\DataSet\\ClassificationResult.csv')


