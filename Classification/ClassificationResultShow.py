import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from Classification.ClassificationProcessFun import *

'''
Save a subject classification result
filepath: for example: filepath=r'D:\\DataSet\\Subject20', mode=1
mode:select train set and test set. When mode=1 or mode=2, the number of train set and test set is equal inevitably(40 for every subject)
Please note: 
(1) Different selection of train set and test set may make slight difference in classification results. 
(2) The classification results of the MLP algorithm may be slightly different after every time it is trained due to its attribute.  
'''
def SaveOneSubjectClassificationResult(filepath, mode=1):
    featurefilename = filepath + r'\\Feature.csv'
    trainSet_x, trainSet_y, testSet_x, testSet_y = TrainAndTestSetSelect(featurefilename, mode)
    
    ###############LDA###############
    LDA_clf = LinearDiscriminantAnalysis(solver='eigen', shrinkage=None,).fit(trainSet_x, trainSet_y)
    LDA_predict_y = LDA_clf.predict(testSet_x)
    LDA_accuracy = GetAllMeanAccuracy(LDA_predict_y, testSet_y)
    ###############LDA###############
    
    ###############RidgeClassifier###############
    Ridge_clf = RidgeClassifier(alpha=0.00001).fit(trainSet_x, trainSet_y)
    Ridge_predict_y = Ridge_clf.predict(testSet_x)
    Ridge_accuracy = GetAllMeanAccuracy(Ridge_predict_y, testSet_y)
    ###############RidgeClassifier###############
    
    ###############logRegression###############
    logRegression_clf = LogisticRegression(penalty='none').fit(trainSet_x, trainSet_y)
    log_predict_y = logRegression_clf.predict(testSet_x)
    log_accuracy = GetAllMeanAccuracy(log_predict_y, testSet_y)
    ###############logRegression###############
    
    ###############KNN###############
    KNN_clf = KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='auto').fit(trainSet_x, trainSet_y)
    KNN_predict_y = KNN_clf.predict(testSet_x)
    KNN_accuracy = GetAllMeanAccuracy(KNN_predict_y, testSet_y)
    ###############KNN###############
    
    ###############MLP###############
    '''
    Please note: The classification results of the MLP algorithm may be slightly different after every time it is trained due to its attribute.  
    '''
    MLP_clf = MLPClassifier(hidden_layer_sizes=(100), activation='tanh', solver='lbfgs', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001).fit(trainSet_x, trainSet_y)
    MLP_predict_y = MLP_clf.predict(testSet_x)
    MLP_accuracy = GetAllMeanAccuracy(MLP_predict_y, testSet_y)
    ###############MLP###############
    
    ###############KNN###############
    SVM_clf = SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200,decision_function_shape='ovr').fit(trainSet_x, trainSet_y)
    SVM_predict_y = SVM_clf.predict(testSet_x)
    SVM_accuracy = GetAllMeanAccuracy(SVM_predict_y, testSet_y)
    ###############KNN###############
    
    LDA_MultiAccuracy = GetAccuracyForMultiClass(LDA_predict_y, testSet_y)
    Ridge_MultiAccuracy = GetAccuracyForMultiClass(Ridge_predict_y, testSet_y)
    log_MultiAccuracy = GetAccuracyForMultiClass(log_predict_y, testSet_y)
    KNN_MultiAccuracy = GetAccuracyForMultiClass(KNN_predict_y, testSet_y)
    MLP_MultiAccuracy = GetAccuracyForMultiClass(MLP_predict_y, testSet_y)
    SVM_MultiAccuracy = GetAccuracyForMultiClass(SVM_predict_y, testSet_y)
    
    print('LDA accuracy: {}%'.format(LDA_MultiAccuracy[1][0]))
    print('Ridge accuracy: {}%'.format(Ridge_MultiAccuracy[1][0]))
    print('Log accuracy: {}%'.format(log_MultiAccuracy[1][0]))
    print('KNN accuracy: {}%'.format(KNN_MultiAccuracy[1][0]))
    print('MLP accuracy: {}%'.format(MLP_MultiAccuracy[1][0]))
    print('SVM accuracy: {}%'.format(SVM_MultiAccuracy[1][0]))
    fisrt_row = ['Classer/Label', 'mean', '1', '2', '3', '4', '5']
    LDA_row = LDA_MultiAccuracy[1]
    LDA_row.insert(0, 'LDA')
    Ridge_row = Ridge_MultiAccuracy[1]
    Ridge_row.insert(0, 'Ridge')
    log_row = log_MultiAccuracy[1]
    log_row.insert(0, 'Log')
    KNN_row = KNN_MultiAccuracy[1]
    KNN_row.insert(0, 'KNN')
    MLP_row = MLP_MultiAccuracy[1]
    MLP_row.insert(0, 'MLP')
    SVM_row = SVM_MultiAccuracy[1]
    SVM_row.insert(0, 'SVM')
    
    #################
    newdata = [fisrt_row, LDA_row, Ridge_row, log_row, KNN_row, MLP_row, SVM_row]
    with open(filepath + r'\\ClassificationResult.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerows(newdata)
        csvfile.close()
    #################

'''
input: FilePath=r'D:\\DataSet'
'''
def ReadAllSubjectClassificationResult(FilePath):
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
        filename = FilePath + r'\\' + subject[i] + r'\\ClassificationResult.csv'
        LDA_Classer_Accuracy, Ridge_Classer_Accuracy, Log_Classer_Accuracy, Knn_Classer_Accuracy, MLP_Classer_Accuracy, SVM_Classer_Accuracy = LoadSingleSubjectClassificationResult(filename)
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

'''
Save overall classification result
input: FilePath=r'D:\\DataSet'
'''
def SaveOverallClassificationReault(FilePath): 
    LDA_Accuracy_mean, Ridge_Accuracy_mean, Log_Accuracy_mean, Knn_Accuracy_mean, MLP_Accuracy_mean, SVM_Accuracy_mean,\
    LDA_Accuracy_min, Ridge_Accuracy_min, Log_Accuracy_min, Knn_Accuracy_min, MLP_Accuracy_min, SVM_Accuracy_min,\
    LDA_Accuracy_max, Ridge_Accuracy_max, Log_Accuracy_max, Knn_Accuracy_max, MLP_Accuracy_max, SVM_Accuracy_max = ReadAllSubjectClassificationResult(FilePath)
    
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
     
    #################
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
    
    with open(FilePath+r'\\ClassificationResult.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerows(data)
        csvfile.close()
    #################
    
    print('Finish!')

'''
6 classifiers results show for every subject
featurefilename: for example: featurefilename=r'D:\\DataSet\\Subject20\\Feature.csv'
mode:select train set and test set.
'''
def ClassificationAccuracyShow(featurefilename, mode=1):
    trainSet_x, trainSet_y, testSet_x, testSet_y = TrainAndTestSetSelect(featurefilename, mode)
    
    ###############LDA###############
    LDA_clf = LinearDiscriminantAnalysis(solver='eigen', shrinkage=None).fit(trainSet_x, trainSet_y)
    LDA_predict_y = LDA_clf.predict(testSet_x)
    LDA_accuracy = GetAllMeanAccuracy(LDA_predict_y, testSet_y)
    ###############LDA###############
    
    ###############RidgeClassifier###############
    Ridge_clf = RidgeClassifier(alpha=0.00001).fit(trainSet_x, trainSet_y)
    Ridge_predict_y = Ridge_clf.predict(testSet_x)
    Ridge_accuracy = GetAllMeanAccuracy(Ridge_predict_y, testSet_y)
    ###############RidgeClassifier###############
    
    ###############logRegression###############
    logRegression_clf = LogisticRegression(penalty='none').fit(trainSet_x, trainSet_y)
    log_predict_y = logRegression_clf.predict(testSet_x)
    log_accuracy = GetAllMeanAccuracy(log_predict_y, testSet_y)
    ###############logRegression###############
    
    ###############KNN###############
    KNN_clf = KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='auto').fit(trainSet_x, trainSet_y)
    KNN_predict_y = KNN_clf.predict(testSet_x)
    KNN_accuracy = GetAllMeanAccuracy(KNN_predict_y, testSet_y)
    ###############KNN###############
    
    ###############MLP###############
    '''
    Please note: The classification results of the MLP algorithm may be slightly different every time it is trained due to its attribute.  
    '''
    MLP_clf = MLPClassifier(hidden_layer_sizes=(100), activation='tanh', solver='lbfgs', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001).fit(trainSet_x, trainSet_y)
    MLP_predict_y = MLP_clf.predict(testSet_x)
    MLP_accuracy = GetAllMeanAccuracy(MLP_predict_y, testSet_y)
    ###############MLP###############
    
    ###############KNN###############
    SVM_clf = SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200,decision_function_shape='ovr').fit(trainSet_x, trainSet_y)
    SVM_predict_y = SVM_clf.predict(testSet_x)
    SVM_accuracy = GetAllMeanAccuracy(SVM_predict_y, testSet_y)
    ###############KNN###############
     
    print('LDA accuracy: {}%'.format(LDA_accuracy))
    print('Ridge accuracy: {}%'.format(Ridge_accuracy))
    print('Log accuracy: {}%'.format(log_accuracy))
    print('KNN accuracy: {}%'.format(KNN_accuracy))
    print('MLP accuracy: {}%'.format(MLP_accuracy))
    print('SVM accuracy: {}%'.format(SVM_accuracy))
    
    LDA_MultiAccuracy = GetAccuracyForMultiClass(LDA_predict_y, testSet_y)
    Ridge_MultiAccuracy = GetAccuracyForMultiClass(Ridge_predict_y, testSet_y)
    log_MultiAccuracy = GetAccuracyForMultiClass(log_predict_y, testSet_y)
    KNN_MultiAccuracy = GetAccuracyForMultiClass(KNN_predict_y, testSet_y)
    MLP_MultiAccuracy = GetAccuracyForMultiClass(MLP_predict_y, testSet_y)
    SVM_MultiAccuracy = GetAccuracyForMultiClass(SVM_predict_y, testSet_y)
    
    Accuracy = [LDA_accuracy, Ridge_accuracy, log_accuracy, KNN_accuracy, MLP_accuracy, SVM_accuracy]
    
    figure(1)
    plt.title('All classifiers')
    name_list=['LDA', 'Ridge', 'Log', 'KNN', 'MLP', 'SVM']
    autolabel(plt.bar(range(len(Accuracy)), Accuracy, color='rgbkyc', tick_label=name_list))
    
    name_list=['mean', 'T1', 'T2', 'T3', 'T4', 'T5']
    figure(2)  
    plt.title('LDA')  
    autolabel(plt.bar(range(len(LDA_MultiAccuracy[1])), LDA_MultiAccuracy[1], color='rgbkyc', tick_label=name_list))
    
    figure(3)
    plt.title('Ridge')    
    autolabel(plt.bar(range(len(Ridge_MultiAccuracy[1])), Ridge_MultiAccuracy[1], color='rgbkyc', tick_label=name_list))
    
    figure(4)  
    plt.title('log')  
    autolabel(plt.bar(range(len(log_MultiAccuracy[1])), log_MultiAccuracy[1], color='rgbkyc', tick_label=name_list))
    
    figure(5)  
    plt.title('KNN')  
    autolabel(plt.bar(range(len(KNN_MultiAccuracy[1])), KNN_MultiAccuracy[1], color='rgbkyc', tick_label=name_list))
    
    figure(6)   
    plt.title('MLP') 
    autolabel(plt.bar(range(len(MLP_MultiAccuracy[1])), MLP_MultiAccuracy[1], color='rgbkyc', tick_label=name_list))
    
    figure(7)   
    plt.title('SVM') 
    autolabel(plt.bar(range(len(SVM_MultiAccuracy[1])), SVM_MultiAccuracy[1], color='rgbkyc', tick_label=name_list))
    
    
    plt.show()

 

'''
Show classification result
'''
ClassificationAccuracyShow(r'D:\\DataSet\\Subject20\\Feature.csv', mode=1)



