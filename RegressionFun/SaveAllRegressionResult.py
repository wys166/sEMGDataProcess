from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR

from RegressionFun.RegressionFeatureProcess import *
from ActivityDetect.GetActivity import *
from Denoise.ForceDenoiseProcess import *
from ActivityDetect.ForceActivity import *

########函数:3:找到data中与specialvalue接近的值的索引位置#########
def Get_Special_Index(data, specialvalue):
    length = len(data)
    
    value = []
    i=0
    while i<length:
        value.append(abs(data[i]-specialvalue))
                
        i=i+1
    value_min = min(value)
    
    return value.index(value_min)
########函数3:找到data中与specialvalue接近的值的索引位置#########

'''
构建力三角形，求斜率
'''
def Structure_Triangle(force, start_index, end_index, Linear_predict_y, Linear_true_y, predict_force_index, force_speed):
    ###########采集的活动段内的力曲线构造三角形###########
    activity_force = list(force[start_index[2*predict_force_index+1]:end_index[2*predict_force_index+1]])#训练集中偶数作为训练，奇数作为测试，活动段内的力值大小
    activity_force_axis = range(len(activity_force))#活动段内的力对应的X轴坐标
    
    activity_force_max_value = np.max(activity_force)#最大力
    activity_force_max_value_index = activity_force.index(activity_force_max_value)+30#最大力对应的索引坐标后移动30个点将力的凸起点中心作为力三角形的顶点
    
    activity_force_front = activity_force[:activity_force_max_value_index]#实际力三角形左边的边对应的y坐标
    activity_force_front_axis = activity_force_axis[:activity_force_max_value_index]#实际力三角形左边的边对应的x坐标

    p_front = np.polyfit(activity_force_front_axis, activity_force_front, 1)#线性拟合三角形左边的边
    activity_force_front_fit = activity_force_front_axis*p_front[0] + p_front[1]#线性拟合三角形左边的边对应的y坐标
    
    activity_force_back = [activity_force_front_fit[-1], activity_force[-1]]#实际力三角形右边的边对应的y坐标
    activity_force_back_axis = [activity_force_front_axis[-1], activity_force_axis[-1]]#实际力三角形右边的边对应的x坐标
    
    activity_force_front_5_index = Get_Special_Index(activity_force_front_fit, force_th)#力三角形左边的边从力阈值5开始绘制边长
    activity_force_front_fit_part = activity_force_front_fit[activity_force_front_5_index:]
    activity_force_front_axis_part = activity_force_front_axis[activity_force_front_5_index:]
    
    Heigh_true_triangle = activity_force_front_fit[-1] - force_th #构造出来的三角形的高
    Bottom_true_triangle = activity_force_back_axis[-1] - activity_force_front_axis_part[0]#构造出来的三角形的低
    S_true_triangle = Heigh_true_triangle * Bottom_true_triangle #构造出来的三角形的面积的二倍
    ###########采集的活动段内的力曲线构造三角形###########
     
    ###########预测的力积分值构造三角形########### 
    predict_force_integral_class = Linear_predict_y[int(force_speed)-1]#根据文件名自动识别是哪种速度的力
    predict_force_integral = predict_force_integral_class[predict_force_index]
    true_force_integral_class = Linear_true_y[int(force_speed)-1]
    true_force_integral = true_force_integral_class[predict_force_index]
    predict_force_integral = predict_force_integral - force_th*len(activity_force)#力的阈值为5，要减去此部分正方形的面积
    true_force_integral = true_force_integral - force_th*len(activity_force)#力的阈值为5，要减去此部分正方形的面积
    
    S_predict_triangle = S_true_triangle * (predict_force_integral/true_force_integral)#换算成在坐标轴中的面积的2倍
    Heigh_predict_triangle = S_predict_triangle/Bottom_true_triangle#预测的三角形的高
    Heigh_predict_triangle_y = Heigh_predict_triangle +force_th #在坐标轴中显示时需要加上力阈值5
    triangle_predict_front_y_axis = [activity_force_front_fit_part[0], Heigh_predict_triangle_y] #预测的三角形左边的边的坐标y轴
    triangle_predict_front_x_axis = [activity_force_front_axis_part[0], activity_force_front_axis_part[-1]]#预测的三角形左边的边对应的X轴坐标
    triangle_predict_back_y_axis = [Heigh_predict_triangle_y, force_th] #预测的三角形右边的边的坐标y轴
    triangle_predict_back_x_axis = [activity_force_front_axis_part[-1], activity_force_back_axis[-1]]#预测的三角形右边的边的坐标x轴
    
    p_predict_front = np.polyfit(triangle_predict_front_x_axis, triangle_predict_front_y_axis, 1)#线性拟合三角形左边的边
    
    k_true = round(p_front[0], 8) #真实的力斜率
    k_predict = round(p_predict_front[0], 8) #预测的力斜率
    ###########预测的力积分值构造三角形########### 
    
    return k_predict, k_true

'''
testSet_y:所有类的y值为一行
true_y:每一类的y值为一行
'''
def Get_Reggression_Slope(forcefilename, testSet_y, predict_y, true_y):
    sampling_rate=1000
    force,Raw_1,Envelope_1,Raw_2,Envelope_2=LoadDataSetAfterProcess(forcefilename)
    force = ButterLowFilter(force, 5, sampling_rate)#力去噪
    force = np.rint(force) #四舍五入取整
    ################使用力阈值的活动段检测算法################
    force_speed = int(forcefilename[-5])#根据文件名称中的倒数第五位确定是哪种速度下的力
    start_index,end_index,force_start_y,force_end_y,\
    raw1_start_y,raw1_end_y,raw2_start_y,raw2_end_y,\
    envelope1_start_y,envelope1_end_y,envelope2_start_y,envelope2_end_y=GetStartAndEndByForceUsingInterp1d(force, Raw_1,Envelope_1,Raw_2,Envelope_2, force_speed)
    ################使用力阈值的活动段检测算法################
    
    per_class_num = 40 #每一类有40个测试样本
    K_predict = []
    K_true = []
    i = 0
    while i<per_class_num:
        k_predict_, k_true_ = Structure_Triangle(force, start_index, end_index, predict_y, true_y, i, force_speed)
        K_predict.append(k_predict_)
        K_true.append(k_true_)
        i=i+1
        
    return K_predict, K_true

'''
将回归结果保存下来
'''
def Save_Reggression_Result(filepath, RegressionFun = 1):  
    featurefilename = filepath + r'\\' + r'Feature.csv'   
    trainSet_x, trainSet_y, trainSet_y_label, testSet_x, testSet_y, testSet_y_label = TrainAndTestSetSelectForAllClass(featurefilename, mode=1)#偶数用于训练，奇数用于测试
    
    if RegressionFun == 1:
        ##############线性回归##############
        Linear_reg = LinearRegression().fit(trainSet_x, trainSet_y)
        all_predict_y = Linear_reg.predict(testSet_x)
        predict_y, true_y = SingleClassSeparate(all_predict_y, testSet_y, testSet_y_label)#将每一类的真实值与预测值分开
        Linear_error_value, Linear_std_value = GetStdForPredictAndTrue(all_predict_y, testSet_y)#得到真实值与预测值的误差与标准差
        print('线性回归预测值与真实值之间的标准差：{}'.format(Linear_std_value))
        ##############线性回归############## 
    elif RegressionFun == 2:    
        ##############岭回归##############
        Ridge_reg = Ridge().fit(trainSet_x, trainSet_y)
        all_predict_y = Ridge_reg.predict(testSet_x)
        predict_y, true_y = SingleClassSeparate(all_predict_y, testSet_y, testSet_y_label)
        Ridge_error_value, Ridge_std_value = GetStdForPredictAndTrue(all_predict_y, testSet_y)#得到真实值与预测值的误差与标准差
        print('岭回归预测值与真实值之间的标准差：{}'.format(Ridge_std_value))
        ##############岭回归##############
    elif RegressionFun == 3: 
        ##############k近邻回归##############
        Knn_reg = KNeighborsRegressor(n_neighbors=5).fit(trainSet_x, trainSet_y)
        all_predict_y = Knn_reg.predict(testSet_x)
        predict_y, true_y = SingleClassSeparate(all_predict_y, testSet_y, testSet_y_label)
        Knn_error_value, Knn_std_value = GetStdForPredictAndTrue(all_predict_y, testSet_y)
        print('K近邻回归预测值与真实值之间的标准差：{}'.format(Knn_std_value))
        ##############k近邻回归############## 
    elif RegressionFun == 4:     
        ##############MLP回归##############
        MLP_reg = MLPRegressor(hidden_layer_sizes=(500), activation='identity', solver='lbfgs', alpha=0.00001, batch_size='auto', learning_rate='constant').fit(trainSet_x, trainSet_y)
        all_predict_y = MLP_reg.predict(testSet_x)
        predict_y, true_y = SingleClassSeparate(all_predict_y, testSet_y, testSet_y_label)
        MLP_error_value, MLP_std_value = GetStdForPredictAndTrue(all_predict_y, testSet_y)
        print('MLP回归预测值与真实值之间的标准差：{}'.format(MLP_std_value))
        ##############MLP回归##############
    elif RegressionFun == 5:
        ##############SVM回归##############
        SVM_reg = LinearSVR(C=2, loss='squared_epsilon_insensitive').fit(trainSet_x, trainSet_y)
        all_predict_y = SVM_reg.predict(testSet_x)
        predict_y, true_y = SingleClassSeparate(all_predict_y, testSet_y, testSet_y_label)
        SVM_error_value, SVM_std_value = GetStdForPredictAndTrue(all_predict_y, testSet_y)
        print('SVM回归预测值与真实值之间的标准差：{}'.format(SVM_std_value))
        ##############SVM回归##############         
            
    file_list = ['new_1.csv', 'new_2.csv', 'new_3.csv', 'new_4.csv', 'new_5.csv'] #插值后的力文件    
    
    K_predict= []
    K_true = []
    i=0
    while i<len(file_list): 
        forcefilename = filepath + r'\\' + file_list[i]    
        K_predict_, K_true_ = Get_Reggression_Slope(forcefilename, testSet_y, predict_y, true_y)  
        K_predict = K_predict + K_predict_
        K_true = K_true + K_true_
        
        i = i+1
        print('已处理{}个文件'.format(i))
    
    all_predict_y = np.rint(all_predict_y)
    testSet_y = list(np.rint(testSet_y))
    
    all_predict_y = list(all_predict_y)
    testSet_y_label.insert(0, 'label')  
    testSet_y.insert(0, 'true_force_integral') 
    all_predict_y.insert(0, 'predict_force_integral') 
    K_true.insert(0, 'ture_slope')
    K_predict.insert(0, 'predict_slope')
    
    if RegressionFun == 1:
        Regressionmode = 'Linear'
    elif RegressionFun == 2:
        Regressionmode = 'Ridge'
    elif RegressionFun == 3:
        Regressionmode = 'Knn'
    elif RegressionFun == 4:
        Regressionmode = 'MLP'
    elif RegressionFun == 5:
        Regressionmode = 'SVM'
        
    #################将相关参数保存到CSV文件中
    newdata = [testSet_y_label, testSet_y, all_predict_y, K_true, K_predict]
    newdata_t = List_Transpose(newdata)
    with open(filepath + r'\\' + Regressionmode + r'RegressionResult.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerows(newdata_t)
        csvfile.close()
    #################将相关参数保存到CSV文件中   
 
def main(filepath):
    
    i = 1
    while i<6:
        Save_Reggression_Result(filepath, RegressionFun = i)  
        
        i=i+1
 
    