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

def ClasserAccuracy(featurefilename):
    trainSet_x, trainSet_y, testSet_x, testSet_y = TrainAndTestSetSelect(featurefilename, mode=1)
    
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
    # clf = LogisticRegression(penalty='l2').fit(trainSet_x, trainSet_y)
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
    MLP_clf = MLPClassifier(hidden_layer_sizes=(100), activation='tanh', solver='lbfgs', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001).fit(trainSet_x, trainSet_y)
    MLP_predict_y = MLP_clf.predict(testSet_x)
    MLP_accuracy = GetAllMeanAccuracy(MLP_predict_y, testSet_y)
    ###############MLP###############
    
    ###############KNN###############
    SVM_clf = SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200,decision_function_shape='ovr').fit(trainSet_x, trainSet_y)
    SVM_predict_y = SVM_clf.predict(testSet_x)
    SVM_accuracy = GetAllMeanAccuracy(SVM_predict_y, testSet_y)
    ###############KNN###############
    
    
    print('LDA正确率：{}%'.format(LDA_accuracy))
    print('岭回归正确率：{}%'.format(Ridge_accuracy))
    print('逻辑回归正确率：{}%'.format(log_accuracy))
    print('KNN正确率：{}%'.format(KNN_accuracy))
    print('MLP正确率：{}%'.format(MLP_accuracy))
    print('SVM正确率：{}%'.format(SVM_accuracy))
    
    LDA_MultiAccuracy = GetAccuracyForMultiClass(LDA_predict_y, testSet_y)
    Ridge_MultiAccuracy = GetAccuracyForMultiClass(Ridge_predict_y, testSet_y)
    log_MultiAccuracy = GetAccuracyForMultiClass(log_predict_y, testSet_y)
    KNN_MultiAccuracy = GetAccuracyForMultiClass(KNN_predict_y, testSet_y)
    MLP_MultiAccuracy = GetAccuracyForMultiClass(MLP_predict_y, testSet_y)
    SVM_MultiAccuracy = GetAccuracyForMultiClass(SVM_predict_y, testSet_y)
    
    
    Accuracy = [LDA_accuracy, Ridge_accuracy, log_accuracy, KNN_accuracy, MLP_accuracy, SVM_accuracy]
    
    figure(1)
    plt.title('all classers')
    name_list=['LDA', 'Ridge', 'Log', 'KNN', 'MLP', 'SVM']
    autolabel(plt.bar(range(len(Accuracy)), Accuracy, color='rgbkyc', tick_label=name_list))
    
    name_list=LDA_MultiAccuracy[0]
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
    