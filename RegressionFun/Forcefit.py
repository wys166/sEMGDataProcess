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
对力活动段内的信号进行预测并构建三角形
predict_force_index:第几个力预测的样本，小于40，0~39
'''
def DrawActivityData(forcefilename, featurefilename, predict_force_index):
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
    
    trainSet_x, trainSet_y, trainSet_y_label, testSet_x, testSet_y, testSet_y_label = TrainAndTestSetSelectForAllClass(featurefilename, mode=1)
        
    ##############MLP回归##############
    MLP_reg = MLPRegressor(hidden_layer_sizes=(500), activation='identity', solver='lbfgs', alpha=0.00001, batch_size='auto', learning_rate='constant').fit(trainSet_x, trainSet_y)
    MLP_all_predict_y = MLP_reg.predict(testSet_x)
    MLP_predict_y, MLP_true_y = SingleClassSeparate(MLP_all_predict_y, testSet_y, testSet_y_label)
    MLP_error_value, MLP_std_value = GetStdForPredictAndTrue(MLP_all_predict_y, testSet_y)
    print('MLP回归预测值与真实值之间的标准差：{}'.format(MLP_std_value))
    ##############MLP回归##############
    
    regression_predict_y = MLP_predict_y
    regression_true_y = MLP_true_y
    
    
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
    
    true_triangle_x = list(activity_force_front_axis_part) + list(activity_force_back_axis) #实际的三角形X轴
    true_triangle_y = list(activity_force_front_fit_part) + list(activity_force_back) #实际的三角形Y轴
    
    Heigh_true_triangle = activity_force_front_fit[-1] - force_th #构造出来的三角形的高
    Bottom_true_triangle = activity_force_back_axis[-1] - activity_force_front_axis_part[0]#构造出来的三角形的低
    S_true_triangle = Heigh_true_triangle * Bottom_true_triangle #构造出来的三角形的面积的二倍
    ###########采集的活动段内的力曲线构造三角形###########
     
    
    ###########预测的力积分值构造三角形########### 
    predict_force_integral_class = regression_predict_y[int(force_speed)-1]#根据文件名自动识别是哪种速度的力
    predict_force_integral = predict_force_integral_class[predict_force_index]
    true_force_integral_class = regression_true_y[int(force_speed)-1]
    true_force_integral = true_force_integral_class[predict_force_index]
    print('实际的力累加值:{},预测的力累加值:{}'.format(true_force_integral, predict_force_integral))
    predict_force_integral = predict_force_integral - force_th*len(activity_force)#力的阈值为5，要减去此部分正方形的面积
    true_force_integral = true_force_integral - force_th*len(activity_force)#力的阈值为5，要减去此部分正方形的面积
    
    S_predict_triangle = S_true_triangle * (predict_force_integral/true_force_integral)#换算成在坐标轴中的面积的2倍
    Heigh_predict_triangle = S_predict_triangle/Bottom_true_triangle#预测的三角形的高
    print('实际的三角形的高:{},预测的三角形的高:{}'.format(Heigh_true_triangle, Heigh_predict_triangle))
    Heigh_predict_triangle_y = Heigh_predict_triangle +force_th #在坐标轴中显示时需要加上力阈值5
    triangle_predict_front_y_axis = [activity_force_front_fit_part[0], Heigh_predict_triangle_y] #预测的三角形左边的边的坐标y轴
    triangle_predict_front_x_axis = [activity_force_front_axis_part[0], activity_force_front_axis_part[-1]]#预测的三角形左边的边对应的X轴坐标
    triangle_predict_back_y_axis = [Heigh_predict_triangle_y, force_th] #预测的三角形右边的边的坐标y轴
    triangle_predict_back_x_axis = [activity_force_front_axis_part[-1], activity_force_back_axis[-1]]#预测的三角形右边的边的坐标x轴
    
    predict_triangle_x = list(triangle_predict_front_x_axis) + list(triangle_predict_back_x_axis) #预测的三角形X轴
    predict_triangle_y = list(triangle_predict_front_y_axis) + list(triangle_predict_back_y_axis) #预测的三角形Y轴
    
    p_predict_front = np.polyfit(triangle_predict_front_x_axis, triangle_predict_front_y_axis, 1)#线性拟合三角形左边的边
    ###########预测的力积分值构造三角形########### 
     
    print('实际的力从小到大的变化斜率：{}'.format(round(p_front[0], 6)))
    print('预测的力从小到大的变化斜率：{}'.format(round(p_predict_front[0], 6))) 
        
    figure(1)
    plt.title(str(force_speed))
    plt.plot(activity_force_axis,activity_force,'k-',label='force')
#     plt.plot(activity_force_front_axis,activity_force_front_fit,'r-',label='force_front')
    plt.plot(true_triangle_x, true_triangle_y,'b-',label='ture force')
    plt.plot(predict_triangle_x, predict_triangle_y,'g--',label='predict force')

    plt.legend()
    
    plt.show()
       


