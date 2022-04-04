from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR

from RegressionFun.RegressionFeatureProcess import *


# 显示高度
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.- 0.2, 1.03*height, '%s' % int(height))
        
'''
所有类别放一块进行训练和测试
'''
def RegressionAllPredictShow(featurefilename):
    trainSet_x, trainSet_y, trainSet_y_label, testSet_x, testSet_y, testSet_y_label = TrainAndTestSetSelectForAllClass(featurefilename, mode=1)
    
    ##############线性回归##############
    Linear_reg = LinearRegression().fit(trainSet_x, trainSet_y)
    Linear_all_predict_y = Linear_reg.predict(testSet_x)
    Linear_predict_y, true_y = SingleClassSeparate(Linear_all_predict_y, testSet_y, testSet_y_label)
    Linear_error_value, Linear_std_value = GetStdForPredictAndTrue(Linear_all_predict_y, testSet_y)
    print('线性回归预测值与真实值之间的标准差：{}'.format(Linear_std_value))
    ##############线性回归##############
    
    ##############岭回归##############
    Ridge_reg = Ridge().fit(trainSet_x, trainSet_y)
    Ridge_all_predict_y = Ridge_reg.predict(testSet_x)
    Ridge_predict_y, true_y = SingleClassSeparate(Ridge_all_predict_y, testSet_y, testSet_y_label)
    Ridge_error_value, Ridge_std_value = GetStdForPredictAndTrue(Ridge_all_predict_y, testSet_y)
    print('岭回归预测值与真实值之间的标准差：{}'.format(Ridge_std_value))
    ##############岭回归##############
    
    ##############k近邻回归##############
    Knn_reg = KNeighborsRegressor(n_neighbors=5).fit(trainSet_x, trainSet_y)
    Knn_all_predict_y = Knn_reg.predict(testSet_x)
    Knn_predict_y, true_y = SingleClassSeparate(Knn_all_predict_y, testSet_y, testSet_y_label)
    Knn_error_value, Knn_std_value = GetStdForPredictAndTrue(Knn_all_predict_y, testSet_y)
    print('K近邻回归预测值与真实值之间的标准差：{}'.format(Knn_std_value))
    ##############k近邻回归##############
    
    ##############MLP回归##############
    MLP_reg = MLPRegressor(hidden_layer_sizes=(500), activation='identity', solver='lbfgs', alpha=0.00001, batch_size='auto', learning_rate='constant').fit(trainSet_x, trainSet_y)
    MLP_all_predict_y = MLP_reg.predict(testSet_x)
    MLP_predict_y, true_y = SingleClassSeparate(MLP_all_predict_y, testSet_y, testSet_y_label)
    MLP_error_value, MLP_std_value = GetStdForPredictAndTrue(MLP_all_predict_y, testSet_y)
    print('MLP回归预测值与真实值之间的标准差：{}'.format(MLP_std_value))
    ##############MLP回归##############
    
    ##############SVM回归##############
    SVM_reg = LinearSVR(C=2, loss='squared_epsilon_insensitive').fit(trainSet_x, trainSet_y)
    SVM_all_predict_y = SVM_reg.predict(testSet_x)
    SVM_predict_y, true_y = SingleClassSeparate(SVM_all_predict_y, testSet_y, testSet_y_label)
    SVM_error_value, SVM_std_value = GetStdForPredictAndTrue(SVM_all_predict_y, testSet_y)
    print('SVM回归预测值与真实值之间的标准差：{}'.format(SVM_std_value))
    ##############SVM回归##############
        
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
    plt.plot(range(len(Linear_predict_y[0])),Linear_predict_y[0],'k.-',label='predict')#类1
    plt.plot(range(len(true_y[0])),true_y[0],'k.--',label='true')#类1
    plt.plot(range(len(Linear_predict_y[1])),Linear_predict_y[1],'r.-',label='predict')#类2
    plt.plot(range(len(true_y[1])),true_y[1],'r.--',label='true')#类2
    plt.plot(range(len(Linear_predict_y[2])),Linear_predict_y[2],'g.-',label='predict')#类3
    plt.plot(range(len(true_y[2])),true_y[2],'g.--',label='true')#类3
    plt.plot(range(len(Linear_predict_y[3])),Linear_predict_y[3],'b.-',label='predict')#类4
    plt.plot(range(len(true_y[3])),true_y[3],'b.--',label='true')#类4
    plt.plot(range(len(Linear_predict_y[4])),Linear_predict_y[4],'y.-',label='predict')#类5
    plt.plot(range(len(true_y[4])),true_y[4],'y.--',label='true')#类5
    plt.legend()
    
    figure(3)
    plt.title('Ridge')
    plt.plot(range(len(Ridge_predict_y[0])),Ridge_predict_y[0],'k.-',label='predict')#类1
    plt.plot(range(len(true_y[0])),true_y[0],'k.--',label='true')#类1
    plt.plot(range(len(Ridge_predict_y[1])),Ridge_predict_y[1],'r.-',label='predict')#类2
    plt.plot(range(len(true_y[1])),true_y[1],'r.--',label='true')#类2
    plt.plot(range(len(Ridge_predict_y[2])),Ridge_predict_y[2],'g.-',label='predict')#类3
    plt.plot(range(len(true_y[2])),true_y[2],'g.--',label='true')#类3
    plt.plot(range(len(Ridge_predict_y[3])),Ridge_predict_y[3],'b.-',label='predict')#类4
    plt.plot(range(len(true_y[3])),true_y[3],'b.--',label='true')#类4
    plt.plot(range(len(Ridge_predict_y[4])),Ridge_predict_y[4],'y.-',label='predict')#类5
    plt.plot(range(len(true_y[4])),true_y[4],'y.--',label='true')#类5
    plt.legend()
    
    figure(4)
    plt.title('Knn')
    plt.plot(range(len(Knn_predict_y[0])),Knn_predict_y[0],'k.-',label='predict')#类1
    plt.plot(range(len(true_y[0])),true_y[0],'k.--',label='true')#类1
    plt.plot(range(len(Knn_predict_y[1])),Knn_predict_y[1],'r.-',label='predict')#类2
    plt.plot(range(len(true_y[1])),true_y[1],'r.--',label='true')#类2
    plt.plot(range(len(Knn_predict_y[2])),Knn_predict_y[2],'g.-',label='predict')#类3
    plt.plot(range(len(true_y[2])),true_y[2],'g.--',label='true')#类3
    plt.plot(range(len(Knn_predict_y[3])),Knn_predict_y[3],'b.-',label='predict')#类4
    plt.plot(range(len(true_y[3])),true_y[3],'b.--',label='true')#类4
    plt.plot(range(len(Knn_predict_y[4])),Knn_predict_y[4],'y.-',label='predict')#类5
    plt.plot(range(len(true_y[4])),true_y[4],'y.--',label='true')#类5
    plt.legend()
    
    figure(5)
    plt.title('MLP')
    plt.plot(range(len(MLP_predict_y[0])),MLP_predict_y[0],'k.-',label='predict')#类1
    plt.plot(range(len(true_y[0])),true_y[0],'k.--',label='true')#类1
    plt.plot(range(len(MLP_predict_y[1])),MLP_predict_y[1],'r.-',label='predict')#类2
    plt.plot(range(len(true_y[1])),true_y[1],'r.--',label='true')#类2
    plt.plot(range(len(MLP_predict_y[2])),MLP_predict_y[2],'g.-',label='predict')#类3
    plt.plot(range(len(true_y[2])),true_y[2],'g.--',label='true')#类3
    plt.plot(range(len(MLP_predict_y[3])),MLP_predict_y[3],'b.-',label='predict')#类4
    plt.plot(range(len(true_y[3])),true_y[3],'b.--',label='true')#类4
    plt.plot(range(len(MLP_predict_y[4])),MLP_predict_y[4],'y.-',label='predict')#类5
    plt.plot(range(len(true_y[4])),true_y[4],'y.--',label='true')#类5
    plt.legend()
    
    figure(6)
    plt.title('SVM')
    plt.plot(range(len(SVM_predict_y[0])),SVM_predict_y[0],'k.-',label='predict')#类1
    plt.plot(range(len(true_y[0])),true_y[0],'k.--',label='true')#类1
    plt.plot(range(len(SVM_predict_y[1])),SVM_predict_y[1],'r.-',label='predict')#类2
    plt.plot(range(len(true_y[1])),true_y[1],'r.--',label='true')#类2
    plt.plot(range(len(SVM_predict_y[2])),SVM_predict_y[2],'g.-',label='predict')#类3
    plt.plot(range(len(true_y[2])),true_y[2],'g.--',label='true')#类3
    plt.plot(range(len(SVM_predict_y[3])),SVM_predict_y[3],'b.-',label='predict')#类4
    plt.plot(range(len(true_y[3])),true_y[3],'b.--',label='true')#类4
    plt.plot(range(len(SVM_predict_y[4])),SVM_predict_y[4],'y.-',label='predict')#类5
    plt.plot(range(len(true_y[4])),true_y[4],'y.--',label='true')#类5
    plt.legend()
    
    name_list=['Linear', 'Ridge', 'Knn', 'MLP', 'SVM', ]
    Std_Regresion = [Linear_std_value, Ridge_std_value, Knn_std_value, MLP_std_value, SVM_std_value]
    figure(7)  
    plt.title('Std ')  
    autolabel(plt.bar(name_list, Std_Regresion, width = 0.8, color='krgby', label='true'))
    
    plt.show()

'''
单个类别进行训练和测试
'''
def RegressionSingleClassPredictShow(featurefilename):   
#     class_label =1
    trainSet_x_1, trainSet_y_1, trainSet_y_label_1, testSet_x_1, testSet_y_1, testSet_y_label_1 = TrainAndTestSetSelectForSingleClass(featurefilename, 1, mode=1)
    trainSet_x_2, trainSet_y_2, trainSet_y_label_2, testSet_x_2, testSet_y_2, testSet_y_label_2 = TrainAndTestSetSelectForSingleClass(featurefilename, 2, mode=1)
    trainSet_x_3, trainSet_y_3, trainSet_y_label_3, testSet_x_3, testSet_y_3, testSet_y_label_3 = TrainAndTestSetSelectForSingleClass(featurefilename, 3, mode=1)
    trainSet_x_4, trainSet_y_4, trainSet_y_label_4, testSet_x_4, testSet_y_4, testSet_y_label_4 = TrainAndTestSetSelectForSingleClass(featurefilename, 4, mode=1)
    trainSet_x_5, trainSet_y_5, trainSet_y_label_5, testSet_x_5, testSet_y_5, testSet_y_label_5 = TrainAndTestSetSelectForSingleClass(featurefilename, 5, mode=1)
    
    ##############线性回归##############
    ####类别1
    Linear_reg_1 = LinearRegression().fit(trainSet_x_1, trainSet_y_1)
    Linear_predict_y_1 = Linear_reg_1.predict(testSet_x_1)
    #类别2
    Linear_reg_2 = LinearRegression().fit(trainSet_x_2, trainSet_y_2)
    Linear_predict_y_2 = Linear_reg_2.predict(testSet_x_2)
    #类别3
    Linear_reg_3 = LinearRegression().fit(trainSet_x_3, trainSet_y_3)
    Linear_predict_y_3 = Linear_reg_3.predict(testSet_x_3)
    #类别4
    Linear_reg_4 = LinearRegression().fit(trainSet_x_4, trainSet_y_4)
    Linear_predict_y_4 = Linear_reg_4.predict(testSet_x_4)
    #类别5
    Linear_reg_5 = LinearRegression().fit(trainSet_x_5, trainSet_y_5)
    Linear_predict_y_5 = Linear_reg_5.predict(testSet_x_5)
    ##############线性回归##############    
    
    figure(1)
    plt.title('all class')
    plt.plot(range(len(testSet_y_1)),testSet_y_1,'k.--',label='true')
    plt.plot(range(len(Linear_predict_y_1)),Linear_predict_y_1,'k.-',label='predict')
    plt.plot(range(len(testSet_y_2)),testSet_y_2,'r.--',label='true')
    plt.plot(range(len(Linear_predict_y_2)),Linear_predict_y_2,'r.-',label='predict')
    plt.plot(range(len(testSet_y_3)),testSet_y_3,'g.--',label='true')
    plt.plot(range(len(Linear_predict_y_3)),Linear_predict_y_3,'g.-',label='predict')
    plt.plot(range(len(testSet_y_4)),testSet_y_4,'b.--',label='true')
    plt.plot(range(len(Linear_predict_y_4)),Linear_predict_y_4,'b.-',label='predict')
    plt.plot(range(len(testSet_y_5)),testSet_y_5,'y.--',label='true')
    plt.plot(range(len(Linear_predict_y_5)),Linear_predict_y_5,'y.-',label='predict')
    plt.legend()    
        
    plt.show()
    
   







