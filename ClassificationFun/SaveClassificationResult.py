
from FileImport.ReadData import *
from ClassificationFun.FeatureProcess import *

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

'''
每个受试者的分类结果保存到以该受试者的文件夹下
'''
def Save_OneSubject_Classer_Result(filepath):
    featurefilename = filepath + r'\\Feature.csv'
    trainSet_x, trainSet_y, testSet_x, testSet_y = TrainAndTestSetSelect(featurefilename, mode=2)#由于随机选取的训练集与测试集，每次结果可能有差异
    
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
    
    print('LDA正确率：{}%'.format(LDA_MultiAccuracy[1][0]))
    print('岭回归正确率：{}%'.format(Ridge_MultiAccuracy[1][0]))
    print('逻辑回归正确率：{}%'.format(log_MultiAccuracy[1][0]))
    print('KNN正确率：{}%'.format(KNN_MultiAccuracy[1][0]))
    print('MLP正确率：{}%'.format(MLP_MultiAccuracy[1][0]))
    print('SVM正确率：{}%'.format(SVM_MultiAccuracy[1][0]))
    fisrt_row = ['Classifier/Label', 'mean', '1', '2', '3', '4', '5']
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
    
    #################将相关参数保存到CSV文件中
    newdata = [fisrt_row, LDA_row, Ridge_row, log_row, KNN_row, MLP_row, SVM_row]
    with open(filepath + r'\\ClassificationResult.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerows(newdata)
        csvfile.close()
    #################将相关参数保存到CSV文件中  










 
    