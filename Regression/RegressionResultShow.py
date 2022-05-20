from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR

from Regression.RegressionProcessFun import *

'''
input: for example: filepath=r'D:\\DataSet\\Subject20', model=5, mode=2
model=1:LinearRegression
model=2:RidgeRegression
model=3:KnnRegression
model=4:MLPRegression 
model=5:SVMRegression
mode:select train set and test set. When mode=1 or mode=2, the number of train set and test set is equal inevitably(40 for every subject)
Please note: 
(1) Different selection of train set and test set may make slight difference in regression results. 
(2) The regression results of these models may be slightly different after every time they are trained due to their attribute. 
'''
def SaveReggressionResultByModel(filepath, model=1, mode=1):  
    featurefilename = filepath + r'\\' + r'Feature.csv' 
    #Different selection of train set and test set may make slight difference in results.  
    trainSet_x, trainSet_y, trainSet_y_label, testSet_x, testSet_y, testSet_y_label = TrainAndTestSetSelect(featurefilename, mode)
    if model == 1:
        ##############LinearRegression##############
        Linear_reg = LinearRegression().fit(trainSet_x, trainSet_y)
        all_predict_y = Linear_reg.predict(testSet_x)
        predict_y, true_y = SingleClassSeparate(all_predict_y, testSet_y, testSet_y_label)
        Linear_error_value, Linear_std_value = GetStdForPredictAndTrue(all_predict_y, testSet_y)
        print('Std of LinearRegression:{}'.format(Linear_std_value))
        ##############LinearRegression############## 
    elif model == 2:    
        ##############RidgeRegression##############
        Ridge_reg = Ridge().fit(trainSet_x, trainSet_y)
        all_predict_y = Ridge_reg.predict(testSet_x)
        predict_y, true_y = SingleClassSeparate(all_predict_y, testSet_y, testSet_y_label)
        Ridge_error_value, Ridge_std_value = GetStdForPredictAndTrue(all_predict_y, testSet_y)#得到真实值与预测值的误差与标准差
        print('Std of RidgeRegression:{}'.format(Ridge_std_value))
        ##############RidgeRegression##############
    elif model == 3: 
        ##############KnnRegression##############
        Knn_reg = KNeighborsRegressor(n_neighbors=5).fit(trainSet_x, trainSet_y)
        all_predict_y = Knn_reg.predict(testSet_x)
        predict_y, true_y = SingleClassSeparate(all_predict_y, testSet_y, testSet_y_label)
        Knn_error_value, Knn_std_value = GetStdForPredictAndTrue(all_predict_y, testSet_y)
        print('Std of KnnRegression:{}'.format(Knn_std_value))
        ##############KnnRegression############## 
    elif model == 4:     
        ##############MLPRegression##############
        MLP_reg = MLPRegressor(hidden_layer_sizes=(500), activation='identity', solver='lbfgs', alpha=0.00001, batch_size='auto', learning_rate='constant').fit(trainSet_x, trainSet_y)
        all_predict_y = MLP_reg.predict(testSet_x)
        predict_y, true_y = SingleClassSeparate(all_predict_y, testSet_y, testSet_y_label)
        MLP_error_value, MLP_std_value = GetStdForPredictAndTrue(all_predict_y, testSet_y)
        print('Std of MLPRegression:{}'.format(MLP_std_value))
        ##############MLPRegression##############
    elif model == 5:
        ##############SVMRegression##############
        SVM_reg = LinearSVR(C=2, loss='squared_epsilon_insensitive').fit(trainSet_x, trainSet_y)
        all_predict_y = SVM_reg.predict(testSet_x)
        predict_y, true_y = SingleClassSeparate(all_predict_y, testSet_y, testSet_y_label)
        SVM_error_value, SVM_std_value = GetStdForPredictAndTrue(all_predict_y, testSet_y)
        print('Std of SVMRegression:{}'.format(SVM_std_value))
        ##############SVMRegression##############         
            
    file_list = ['T1.csv', 'T2.csv', 'T3.csv', 'T4.csv', 'T5.csv'] 

    K_predict= []
    K_true = []
    i=0
    while i<len(file_list): 
        forcefilename = filepath + r'\\' + file_list[i]    
        K_predict_, K_true_ = GetReggressionSlope(forcefilename, testSet_y, predict_y, true_y)  
        K_predict = K_predict + K_predict_
        K_true = K_true + K_true_
        
        i = i+1
        
    all_predict_y = np.rint(all_predict_y)
    testSet_y = list(np.rint(testSet_y))
    
    all_predict_y = list(all_predict_y)
    testSet_y_label.insert(0, 'label')  
    testSet_y.insert(0, 'true_force_integral') 
    all_predict_y.insert(0, 'predict_force_integral') 
    K_true.insert(0, 'ture_slope')
    K_predict.insert(0, 'predict_slope')
    
    if model == 1:
        Regressionmode = 'Linear'
    elif model == 2:
        Regressionmode = 'Ridge'
    elif model == 3:
        Regressionmode = 'Knn'
    elif model == 4:
        Regressionmode = 'MLP'
    elif model == 5:
        Regressionmode = 'SVM'
        
    #################
    newdata = [testSet_y_label, testSet_y, all_predict_y, K_true, K_predict]
    newdata_t = List_Transpose(newdata)
    with open(filepath + r'\\' + Regressionmode + r'RegressionResult.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerows(newdata_t)
        csvfile.close()
    ################# 
   
'''
input: for example: filepath=r'D:\\DataSet\\Subject20'
'''    
def SaveRegressionResultAnalysis(filepath):
    filename = [r'LinearRegressionResult.csv', r'RidgeRegressionResult.csv', r'KnnRegressionResult.csv', r'MLPRegressionResult.csv', r'SVMRegressionResult.csv']
    Force_Integral_Std = []
    Slope_Std_Value = []
    
    i=0
    while i<5:
        RegressionResultFilename = filepath + r'\\' + filename[i]
        y_label, force_integral_true, force_integral_predict, slope_true, slope_predict = LoadRegressionResult(RegressionResultFilename)
        force_integral_error_value, force_integral_std_value = GetStdForPredictAndTrue(force_integral_predict, force_integral_true)
        Force_Integral_Std.append(force_integral_std_value)
        
        slope_std_value = np.std(np.array(slope_predict) - np.array(slope_true))
        Slope_Std_Value.append(slope_std_value)        
        
        i=i+1
    
    force_integral_std_value_max = max(Force_Integral_Std)
    force_integral_std_value_min = min(Force_Integral_Std)
    Normalization_Force_Integral_Std = list((array(Force_Integral_Std)-force_integral_std_value_min)/(force_integral_std_value_max-force_integral_std_value_min))
    
    Normalization_Slope_Std = Slope_Std_Value
    #################
    regressionfucname = ['reggression', 'Linear', 'Ridge', 'Knn', 'MLP', 'SVM']
    Normalization_Force_Integral_Std.insert(0, 'Force_Integral_Std')
    Normalization_Slope_Std.insert(0, 'Slope_Std')
    newdata = [regressionfucname, Normalization_Force_Integral_Std, Normalization_Slope_Std]
    with open(filepath + r'\\' + r'RegressionResultAnalysis.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerows(newdata)
        csvfile.close()
    #################   

'''
input: for example: filepath=r'D:\\DataSet'
'''   
def SaveRegressionResultForAllSubject(filepath): 
    subject = ['Subject1', 'Subject2', 'Subject3', 'Subject4', 'Subject5', 'Subject6', 'Subject7', 'Subject8', 'Subject9', 'Subject10', 'Subject11', 'Subject12', 'Subject13', 'Subject14', 'Subject15', 'Subject16', 'Subject17', 'Subject18', 'Subject19', 'Subject20']
    length = len(subject)
    
    Force_Integral_Std = []
    Slope_Std = []
    
    i=0
    while i<length:
        filename = filepath + r'\\' + subject[i] + r'\\' + r'RegressionResultAnalysis.csv'
        force_integral_std, slope_std = LoadRegressionResultAnalysisFile(filename)
        Force_Integral_Std = Force_Integral_Std + [force_integral_std]
        Slope_Std = Slope_Std + [slope_std]
        
        i=i+1
        
    force_integral_mean_std = list(np.mean(Force_Integral_Std, axis=0))
    #################
    regressionfucname = ['Reggression', 'Linear', 'Ridge', 'Knn', 'MLP', 'SVM']
    force_integral_mean_std.insert(0, 'Force_Integral_Mean_Std')
    newdata = [regressionfucname, force_integral_mean_std]
    with open(filepath + r'\\' + r'RegressionResultMeanStd.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerows(newdata)
        csvfile.close()
    #################
    
    subject_insertname = subject
    subject_insertname.insert(0, 'Subject')    
    Force_Integral_Std_insertname = Force_Integral_Std
    Force_Integral_Std_insertname.insert(0, ['Linear', 'Ridge', 'Knn', 'MLP', 'SVM'])
    Force_Integral_Std_insertname_t = List_Transpose(Force_Integral_Std_insertname)
    Force_Integral_Std_insertname_t.insert(0, subject_insertname)
    #################
    newdata = List_Transpose(Force_Integral_Std_insertname_t)
    with open(filepath + r'\\' + r'AllSubjectRegressionResult.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerows(newdata)
        csvfile.close()
    #################
    
    Slope_Std_insertname = Slope_Std
    Slope_Std_insertname.insert(0, ['Linear', 'Ridge', 'Knn', 'MLP', 'SVM'])
    Slope_Std_insertname_t = List_Transpose(Slope_Std_insertname)
    Slope_Std_insertname_t.insert(0, subject_insertname)
    #################
    newdata = List_Transpose(Slope_Std_insertname_t)
    with open(filepath + r'\\' + r'AllSubjectSlopeResult.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerows(newdata)
        csvfile.close()
    #################
      
    print('Finish!')

'''
input: for example: featurefilename=r'D:\\DataSet\\Subject20\\Feature.csv', mode=1
mode:select train set and test set.
The regression results of these models may be slightly different after every time they are trained due to their attribute. 
'''
def RegressionAllPredictShow(featurefilename, mode=1):
    trainSet_x, trainSet_y, trainSet_y_label, testSet_x, testSet_y, testSet_y_label = TrainAndTestSetSelect(featurefilename, mode)
    
    ##############LinearRegression##############
    Linear_reg = LinearRegression().fit(trainSet_x, trainSet_y)
    Linear_all_predict_y = Linear_reg.predict(testSet_x)
    Linear_predict_y, true_y = SingleClassSeparate(Linear_all_predict_y, testSet_y, testSet_y_label)
    Linear_error_value, Linear_std_value = GetStdForPredictAndTrue(Linear_all_predict_y, testSet_y)
    print('Std of LinearRegression:{}'.format(Linear_std_value))
    ##############LinearRegression##############
    
    ##############RidgeRegression##############
    Ridge_reg = Ridge().fit(trainSet_x, trainSet_y)
    Ridge_all_predict_y = Ridge_reg.predict(testSet_x)
    Ridge_predict_y, true_y = SingleClassSeparate(Ridge_all_predict_y, testSet_y, testSet_y_label)
    Ridge_error_value, Ridge_std_value = GetStdForPredictAndTrue(Ridge_all_predict_y, testSet_y)
    print('Std of RidgeRegression:{}'.format(Ridge_std_value))
    ##############RidgeRegression##############
    
    ##############KnnRegression##############
    Knn_reg = KNeighborsRegressor(n_neighbors=5).fit(trainSet_x, trainSet_y)
    Knn_all_predict_y = Knn_reg.predict(testSet_x)
    Knn_predict_y, true_y = SingleClassSeparate(Knn_all_predict_y, testSet_y, testSet_y_label)
    Knn_error_value, Knn_std_value = GetStdForPredictAndTrue(Knn_all_predict_y, testSet_y)
    print('Std of KnnRegression:{}'.format(Knn_std_value))
    ##############KnnRegression##############
    
    ##############MLPRegression##############
    MLP_reg = MLPRegressor(hidden_layer_sizes=(500), activation='identity', solver='lbfgs', alpha=0.00001, batch_size='auto', learning_rate='constant').fit(trainSet_x, trainSet_y)
    MLP_all_predict_y = MLP_reg.predict(testSet_x)
    MLP_predict_y, true_y = SingleClassSeparate(MLP_all_predict_y, testSet_y, testSet_y_label)
    MLP_error_value, MLP_std_value = GetStdForPredictAndTrue(MLP_all_predict_y, testSet_y)
    print('Std of MLPRegression:{}'.format(MLP_std_value))
    ##############MLPRegression##############
    
    ##############SVMRegression##############
    SVM_reg = LinearSVR(C=2, loss='squared_epsilon_insensitive').fit(trainSet_x, trainSet_y)
    SVM_all_predict_y = SVM_reg.predict(testSet_x)
    SVM_predict_y, true_y = SingleClassSeparate(SVM_all_predict_y, testSet_y, testSet_y_label)
    SVM_error_value, SVM_std_value = GetStdForPredictAndTrue(SVM_all_predict_y, testSet_y)
    print('Std of SVMRegression:{}'.format(SVM_std_value))
    ##############SVMRegression##############
     
    
    figure(0)
    plt.title('error')
    plt.plot(range(len(Linear_error_value)),Linear_error_value,'k.-',label='Linear')
    plt.plot(range(len(Ridge_error_value)),Ridge_error_value,'r.-',label='Ridge')
    plt.plot(range(len(Knn_error_value)),Knn_error_value,'g.-',label='Knn')
    plt.plot(range(len(MLP_error_value)),MLP_error_value,'b.-',label='MLP')
    plt.plot(range(len(SVM_error_value)),SVM_error_value,'y.-',label='SVM')
    plt.legend()
    
    figure(1)
    plt.title('All')
    plt.plot(range(len(testSet_y)),testSet_y,'k.--',label='true')
    plt.plot(range(len(Linear_all_predict_y)),Linear_all_predict_y,'r.-',label='linear predict')
    plt.plot(range(len(Ridge_all_predict_y)),Ridge_all_predict_y,'g.-',label='ridge predict')
    plt.plot(range(len(Knn_all_predict_y)),Knn_all_predict_y,'b.-',label='Knn predict')
    plt.plot(range(len(MLP_all_predict_y)),MLP_all_predict_y,'y.-',label='MLP predict')
    plt.plot(range(len(SVM_all_predict_y)),SVM_all_predict_y,'m.-',label='SVM predict')
    plt.legend()
    
    figure(2)
    plt.title('Linear')
    plt.plot(range(len(Linear_predict_y[0])),Linear_predict_y[0],'k.-',label='predict')
    plt.plot(range(len(true_y[0])),true_y[0],'k.--',label='true')
    plt.plot(range(len(Linear_predict_y[1])),Linear_predict_y[1],'r.-',label='predict')
    plt.plot(range(len(true_y[1])),true_y[1],'r.--',label='true')
    plt.plot(range(len(Linear_predict_y[2])),Linear_predict_y[2],'g.-',label='predict')
    plt.plot(range(len(true_y[2])),true_y[2],'g.--',label='true')
    plt.plot(range(len(Linear_predict_y[3])),Linear_predict_y[3],'b.-',label='predict')
    plt.plot(range(len(true_y[3])),true_y[3],'b.--',label='true')
    plt.plot(range(len(Linear_predict_y[4])),Linear_predict_y[4],'y.-',label='predict')
    plt.plot(range(len(true_y[4])),true_y[4],'y.--',label='true')
    plt.legend()
    
    figure(3)
    plt.title('Ridge')
    plt.plot(range(len(Ridge_predict_y[0])),Ridge_predict_y[0],'k.-',label='predict')
    plt.plot(range(len(true_y[0])),true_y[0],'k.--',label='true')
    plt.plot(range(len(Ridge_predict_y[1])),Ridge_predict_y[1],'r.-',label='predict')
    plt.plot(range(len(true_y[1])),true_y[1],'r.--',label='true')
    plt.plot(range(len(Ridge_predict_y[2])),Ridge_predict_y[2],'g.-',label='predict')
    plt.plot(range(len(true_y[2])),true_y[2],'g.--',label='true')
    plt.plot(range(len(Ridge_predict_y[3])),Ridge_predict_y[3],'b.-',label='predict')
    plt.plot(range(len(true_y[3])),true_y[3],'b.--',label='true')
    plt.plot(range(len(Ridge_predict_y[4])),Ridge_predict_y[4],'y.-',label='predict')
    plt.plot(range(len(true_y[4])),true_y[4],'y.--',label='true')
    plt.legend()
    
    figure(4)
    plt.title('Knn')
    plt.plot(range(len(Knn_predict_y[0])),Knn_predict_y[0],'k.-',label='predict')
    plt.plot(range(len(true_y[0])),true_y[0],'k.--',label='true')
    plt.plot(range(len(Knn_predict_y[1])),Knn_predict_y[1],'r.-',label='predict')
    plt.plot(range(len(true_y[1])),true_y[1],'r.--',label='true')
    plt.plot(range(len(Knn_predict_y[2])),Knn_predict_y[2],'g.-',label='predict')
    plt.plot(range(len(true_y[2])),true_y[2],'g.--',label='true')
    plt.plot(range(len(Knn_predict_y[3])),Knn_predict_y[3],'b.-',label='predict')
    plt.plot(range(len(true_y[3])),true_y[3],'b.--',label='true')
    plt.plot(range(len(Knn_predict_y[4])),Knn_predict_y[4],'y.-',label='predict')
    plt.plot(range(len(true_y[4])),true_y[4],'y.--',label='true')
    plt.legend()
    
    figure(5)
    plt.title('MLP')
    plt.plot(range(len(MLP_predict_y[0])),MLP_predict_y[0],'k.-',label='predict')
    plt.plot(range(len(true_y[0])),true_y[0],'k.--',label='true')
    plt.plot(range(len(MLP_predict_y[1])),MLP_predict_y[1],'r.-',label='predict')
    plt.plot(range(len(true_y[1])),true_y[1],'r.--',label='true')
    plt.plot(range(len(MLP_predict_y[2])),MLP_predict_y[2],'g.-',label='predict')
    plt.plot(range(len(true_y[2])),true_y[2],'g.--',label='true')
    plt.plot(range(len(MLP_predict_y[3])),MLP_predict_y[3],'b.-',label='predict')
    plt.plot(range(len(true_y[3])),true_y[3],'b.--',label='true')
    plt.plot(range(len(MLP_predict_y[4])),MLP_predict_y[4],'y.-',label='predict')
    plt.plot(range(len(true_y[4])),true_y[4],'y.--',label='true')
    plt.legend()
    
    figure(6)
    plt.title('SVM')
    plt.plot(range(len(SVM_predict_y[0])),SVM_predict_y[0],'k.-',label='predict')
    plt.plot(range(len(true_y[0])),true_y[0],'k.--',label='true')
    plt.plot(range(len(SVM_predict_y[1])),SVM_predict_y[1],'r.-',label='predict')
    plt.plot(range(len(true_y[1])),true_y[1],'r.--',label='true')
    plt.plot(range(len(SVM_predict_y[2])),SVM_predict_y[2],'g.-',label='predict')
    plt.plot(range(len(true_y[2])),true_y[2],'g.--',label='true')
    plt.plot(range(len(SVM_predict_y[3])),SVM_predict_y[3],'b.-',label='predict')
    plt.plot(range(len(true_y[3])),true_y[3],'b.--',label='true')
    plt.plot(range(len(SVM_predict_y[4])),SVM_predict_y[4],'y.-',label='predict')
    plt.plot(range(len(true_y[4])),true_y[4],'y.--',label='true')
    plt.legend()
    
    name_list=['Linear', 'Ridge', 'Knn', 'MLP', 'SVM', ]
    Std_Regresion = [Linear_std_value, Ridge_std_value, Knn_std_value, MLP_std_value, SVM_std_value]
    figure(7)  
    plt.title('Std ')  
    autolabel(plt.bar(name_list, Std_Regresion, width = 0.8, color='krgby', label='true'))
    
    plt.show()


'''
Show all Regression Model results
'''
RegressionAllPredictShow(featurefilename=r'D:\\DataSet\\Subject20\\Feature.csv', mode=1)


