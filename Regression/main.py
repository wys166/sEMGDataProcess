from Regression.RegressionProcessFun import *

######################Function######################
def autolabel1(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.- 0.3, 1.03*height, '%s' % float(height), fontsize = 10)
######################Function######################

######################Function######################        
def autolabel2(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.- 0.2, 1.01*height, '%s' % int(height), fontsize=7)
######################Function######################
        
'''
input: filepath=r'D:\\DataSet'
'''
def RegressionPredictionResultShow(filepath):
    filename1 = filepath + r'\\' + r'RegressionResultMeanStd.csv'
    force_integral_mean_std = LoadRegressionResultMeanStd(filename1)
    filename2 = filepath + r'\\' + r'RegressionDataShow\\MLPRegressionResult.csv'
    Regression_Result = LoadMLPRegressionResult(filename2)
    true_force_integral = Regression_Result[1]
    predict_force_integral = Regression_Result[2]
    ture_slope = Regression_Result[3]
    predict_slope = Regression_Result[4]  
      
    True_IFORCE, Predict_IFORCE, True_Slope, Predict_Slope = SingleClassSeparateForRegressionResult(true_force_integral, predict_force_integral, ture_slope, predict_slope)     
    
    Mean_True_IFORCE_1 = np.mean(True_IFORCE[0])
    Mean_True_IFORCE_2 = np.mean(True_IFORCE[1])
    Mean_True_IFORCE_3 = np.mean(True_IFORCE[2])
    Mean_True_IFORCE_4 = np.mean(True_IFORCE[3])
    Mean_True_IFORCE_5 = np.mean(True_IFORCE[4])
    
    Mean_Predict_IFORCE_1 = np.mean(Predict_IFORCE[0])
    Mean_Predict_IFORCE_2 = np.mean(Predict_IFORCE[1])
    Mean_Predict_IFORCE_3 = np.mean(Predict_IFORCE[2])
    Mean_Predict_IFORCE_4 = np.mean(Predict_IFORCE[3])
    Mean_Predict_IFORCE_5 = np.mean(Predict_IFORCE[4])
    
    name_list = ['Linear', 'Ridge', 'KNN', 'MLP', 'SVM']
    figure(1)  
    plt.title('Mean Std')  
    autolabel1(plt.bar(name_list, force_integral_mean_std, width = 0.6, color='k', label='Linear'))
    plt.bar(name_list, force_integral_mean_std, width = 0.6, color='r',  label='Ridge')
    plt.bar(name_list, force_integral_mean_std, width = 0.6, color='g',  label='KNN')
    plt.bar(name_list, force_integral_mean_std, width = 0.6, color='b',  label='MLP')
    plt.bar(name_list, force_integral_mean_std, width = 0.6, color='y',  label='SVM')
    plt.bar(name_list, force_integral_mean_std, width = 0.6, color='krgby')
    plt.legend()
    plt.ylim(0, 1)
    
    figure(2)
    plt.plot(range(len(true_force_integral)),true_force_integral,'k.-',label='true')#类1
    plt.plot(range(len(predict_force_integral)),predict_force_integral,'r.--',label='predict')#类1
    plt.legend() 
    
    figure(3)
    sample_num = np.arange(1, 41, 1)
    plt.plot(sample_num,True_IFORCE[0],'k.-',label='T1-true')
    plt.plot(sample_num,Predict_IFORCE[0],'k.--',label='T1-predict')
    plt.plot(sample_num,True_IFORCE[1],'r.-',label='T2-true')
    plt.plot(sample_num,Predict_IFORCE[1],'r.--',label='T2-predict')
    plt.plot(sample_num,True_IFORCE[2],'g.-',label='T3-true')
    plt.plot(sample_num,Predict_IFORCE[2],'g.--',label='T3-predict')
    plt.plot(sample_num,True_IFORCE[3],'b.-',label='T4-true')
    plt.plot(sample_num,Predict_IFORCE[3],'b.--',label='T4-predict')
    plt.plot(sample_num,True_IFORCE[4],'y.-',label='T5-true')
    plt.plot(sample_num,Predict_IFORCE[4],'y.--',label='T5-predict')
    plt.legend()  
        
    X_name=np.array([1, 2, 3, 4, 5])
    name_list = ['T1', 'T2', 'T3', 'T4', 'T5']
    Mean_True_IFORCE = [Mean_True_IFORCE_1, Mean_True_IFORCE_2, Mean_True_IFORCE_3, Mean_True_IFORCE_4, Mean_True_IFORCE_5]
    Mean_Predict_IFORCE = [Mean_Predict_IFORCE_1, Mean_Predict_IFORCE_2, Mean_Predict_IFORCE_3, Mean_Predict_IFORCE_4, Mean_Predict_IFORCE_5]
    figure(4)    
    autolabel2(plt.bar(X_name-0.2, Mean_True_IFORCE, width = 0.4, color='#423D84', label='true'))
    autolabel2(plt.bar(X_name+0.2, Mean_Predict_IFORCE, width = 0.4, color='#1E998A', label='predict'))
    plt.bar(X_name, [0, 0, 0, 0, 0], width = 0, color='w', tick_label=name_list)
    plt.legend()
    
    plt.show()
    
'''
Show regression result
input: r'D:\\DataSet'
'''    
RegressionPredictionResultShow(r'D:\\DataSet')

