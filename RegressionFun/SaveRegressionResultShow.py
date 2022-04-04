from FileImport.ReadData import *
from RegressionFun.RegressionFeatureProcess import *

# 显示高度
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.- 0.2, 1.03*height, '%s' % int(height))

'''
将回归结果显示处理
'''
def RegressionResultShow(RegressionResultFilename):
    y_label, force_integral_true, force_integral_predict, slope_true, slope_predict = LoadRegressionResultFile(RegressionResultFilename)
    
    single_force_integral_predict, single_force_integral_true = SingleClassSeparate(force_integral_predict, force_integral_true, y_label)#每一行代表一类
    single_slope_predict, single_slope_true = SingleClassSeparate(slope_predict, slope_true, y_label)#每一行代表一类
    
    error_force_integral = np.array(force_integral_true) - np.array(force_integral_predict) #所有类的预测力积分值与真实力积分值误差
    print('力积分值预测标准差为:{}'.format(round(np.std(error_force_integral), 5)))#所有类的误差值的标准差
    
    error_slope = np.array(slope_true) - np.array(slope_predict)#所有类的斜率的误差
    print('斜率的标准差为:{}'.format(round(np.std(error_slope), 5)))#所有类的误差值的标准差
    
    mean_force_integral_predict_1 = int(np.mean(single_force_integral_predict[0])) #力1积分预测均值
    mean_force_integral_predict_2 = int(np.mean(single_force_integral_predict[1])) #力2积分预测均值
    mean_force_integral_predict_3 = int(np.mean(single_force_integral_predict[2])) #力3积分预测均值
    mean_force_integral_predict_4 = int(np.mean(single_force_integral_predict[3])) #力4积分预测均值
    mean_force_integral_predict_5 = int(np.mean(single_force_integral_predict[4])) #力5积分预测均值
    mean_force_integral_predict = np.rint(np.array([mean_force_integral_predict_1, mean_force_integral_predict_2, mean_force_integral_predict_3, mean_force_integral_predict_4, mean_force_integral_predict_5]))
    
    mean_force_integral_true_1 = int(np.mean(single_force_integral_true[0])) #力1积分预测均值
    mean_force_integral_true_2 = int(np.mean(single_force_integral_true[1])) #力2积分预测均值
    mean_force_integral_true_3 = int(np.mean(single_force_integral_true[2])) #力3积分预测均值
    mean_force_integral_true_4 = int(np.mean(single_force_integral_true[3])) #力4积分预测均值
    mean_force_integral_true_5 = int(np.mean(single_force_integral_true[4])) #力5积分预测均值
    mean_force_integral_true = np.rint(np.array([mean_force_integral_true_1, mean_force_integral_true_2, mean_force_integral_true_3, mean_force_integral_true_4, mean_force_integral_true_5]))
    
    mean_slope_predict_1 = round(np.mean(single_slope_predict[0]), 8) #力1预测斜率均值
    mean_slope_predict_2 = round(np.mean(single_slope_predict[1]), 8) #力2预测斜率均值
    mean_slope_predict_3 = round(np.mean(single_slope_predict[2]), 8) #力3预测斜率均值
    mean_slope_predict_4 = round(np.mean(single_slope_predict[3]), 8) #力4预测斜率均值
    mean_slope_predict_5 = round(np.mean(single_slope_predict[4]), 8) #力5预测斜率均值
    mean_slope_predict = [mean_slope_predict_1, mean_slope_predict_2, mean_slope_predict_3, mean_slope_predict_4, mean_slope_predict_5]
    
    mean_slope_true_1 = round(np.mean(single_slope_true[0]), 8) #力1实际斜率均值
    mean_slope_true_2 = round(np.mean(single_slope_true[1]), 8) #力2实际斜率均值
    mean_slope_true_3 = round(np.mean(single_slope_true[2]), 8) #力3实际斜率均值
    mean_slope_true_4 = round(np.mean(single_slope_true[3]), 8) #力4实际斜率均值
    mean_slope_true_5 = round(np.mean(single_slope_true[4]), 8) #力5实际斜率均值
    mean_slope_true = [mean_slope_true_1, mean_slope_true_2, mean_slope_true_3, mean_slope_true_4, mean_slope_true_5]
    
    figure(1)
    plt.title('Linear slope')
    plt.plot(range(len(slope_true)),slope_true,'k.-',label='true')#类1
    plt.plot(range(len(slope_predict)),slope_predict,'r.--',label='predict')#类1
    plt.legend()
    
    name_list=np.array([1, 2, 3, 4, 5])
    
    figure(2)
    plt.title('Linear force integral')
    plt.plot(name_list,mean_force_integral_true,'k.-',label='true')#类1
    plt.plot(name_list,mean_force_integral_predict,'r.--',label='predict')#类1
    plt.legend()
    
    figure(3)
    plt.title('Linear force slope')
    plt.plot(name_list,mean_slope_true,'k.-',label='true')#类1
    plt.plot(name_list,mean_slope_predict,'r.--',label='predict')#类1
    plt.legend()
    
    name_list2=np.array([2, 3, 4, 5, 6])
    figure(4)  
    plt.title('linear ')  
    autolabel(plt.bar(name_list-0.2, mean_force_integral_true, width = 0.4, color='r', label='true'))
    autolabel(plt.bar(name_list+0.2, mean_force_integral_predict, width = 0.4, color='b', label='predict'))
    plt.legend()
    
    plt.show()
    