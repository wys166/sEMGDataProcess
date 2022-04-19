from RegressionFun.RegressionFeatureProcess import *

'''
装载回归结果
返回值：第一行：五类标签，第二行：真实的力积分值，第三行：预测的力积分值，第四行：真实的力斜率，第五行：预测的力斜率
'''
def LoadRegressionResult(RegressionResultFilename):    
    with open(RegressionResultFilename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        data = dataset[1:]#去掉第一行标签        
        
        x_line=size(data,0)  #####行，有多少种特征值
        y_line=size(data,1)  #####列，一种特征值提取了多少个
        i=0
        j=0
        while i<x_line:
            j=0
            while j<y_line:
                data[i][j]=np.float(data[i][j])
               
                j=j+1 
            i=i+1            
    csvfile.close()
    
    data = List_Transpose(data) #特征值转置
    
    Regression_Result = data
    return Regression_Result

def SingleClassSeparateForRegressionResult(true_force_integral, predict_force_integral, ture_slope, predict_slope):
    sample_width = 40    
    True_IFORCE = [true_force_integral[0:1*sample_width], true_force_integral[1*sample_width:2*sample_width], true_force_integral[2*sample_width:3*sample_width], true_force_integral[3*sample_width:4*sample_width], true_force_integral[4*sample_width:5*sample_width]]
    Predict_IFORCE = [predict_force_integral[0:1*sample_width], predict_force_integral[1*sample_width:2*sample_width], predict_force_integral[2*sample_width:3*sample_width], predict_force_integral[3*sample_width:4*sample_width], predict_force_integral[4*sample_width:5*sample_width]]    
    True_Slope = [ture_slope[0:1*sample_width], ture_slope[1*sample_width:2*sample_width], ture_slope[2*sample_width:3*sample_width], ture_slope[3*sample_width:4*sample_width], ture_slope[4*sample_width:5*sample_width]]
    Predict_Slope = [predict_slope[0:1*sample_width], predict_slope[1*sample_width:2*sample_width], predict_slope[2*sample_width:3*sample_width], predict_slope[3*sample_width:4*sample_width], predict_slope[4*sample_width:5*sample_width]]    
    
    return True_IFORCE, Predict_IFORCE, True_Slope, Predict_Slope

# 显示高度
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.- 0.2, 1.01*height, '%s' % int(height), fontsize=7)

def ResultShow(RegressionResultFilename): 
    Regression_Result = LoadRegressionResult(RegressionResultFilename)
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
    
    Mean_True_Slope_1 = np.mean(True_Slope[0])
    Mean_True_Slope_2 = np.mean(True_Slope[1])
    Mean_True_Slope_3 = np.mean(True_Slope[2])
    Mean_True_Slope_4 = np.mean(True_Slope[3])
    Mean_True_Slope_5 = np.mean(True_Slope[4])
    
    Mean_Predict_Slope_1 = np.mean(Predict_Slope[0])
    Mean_Predict_Slope_2 = np.mean(Predict_Slope[1])
    Mean_Predict_Slope_3 = np.mean(Predict_Slope[2])
    Mean_Predict_Slope_4 = np.mean(Predict_Slope[3])
    Mean_Predict_Slope_5 = np.mean(Predict_Slope[4])  
    
    figure(1)
    plt.plot(range(len(true_force_integral)),true_force_integral,'k.-',label='true')#类1
    plt.plot(range(len(predict_force_integral)),predict_force_integral,'r.--',label='predict')#类1
    plt.legend() 
    
    sample_num = np.arange(1, 41, 1)
    print(sample_num)
    figure(2)
    plt.plot(sample_num,True_IFORCE[0],'k.-',label='T1-true')#类1
    plt.plot(sample_num,Predict_IFORCE[0],'k.--',label='T1-predict')#类1
    plt.plot(sample_num,True_IFORCE[1],'r.-',label='T2-true')#类2
    plt.plot(sample_num,Predict_IFORCE[1],'r.--',label='T2-predict')#类2
    plt.plot(sample_num,True_IFORCE[2],'g.-',label='T3-true')#类3
    plt.plot(sample_num,Predict_IFORCE[2],'g.--',label='T3-predict')#类3
    plt.plot(sample_num,True_IFORCE[3],'b.-',label='T4-true')#类4
    plt.plot(sample_num,Predict_IFORCE[3],'b.--',label='T4-predict')#类4
    plt.plot(sample_num,True_IFORCE[4],'y.-',label='T5-true')#类5
    plt.plot(sample_num,Predict_IFORCE[4],'y.--',label='T5-predict')#类5
    plt.legend()  
        
    X_name=np.array([1, 2, 3, 4, 5])
    name_list = ['T1', 'T2', 'T3', 'T4', 'T5']
    Mean_True_IFORCE = [Mean_True_IFORCE_1, Mean_True_IFORCE_2, Mean_True_IFORCE_3, Mean_True_IFORCE_4, Mean_True_IFORCE_5]
    Mean_Predict_IFORCE = [Mean_Predict_IFORCE_1, Mean_Predict_IFORCE_2, Mean_Predict_IFORCE_3, Mean_Predict_IFORCE_4, Mean_Predict_IFORCE_5]
    figure(3)   
    autolabel(plt.bar(X_name-0.2, Mean_True_IFORCE, width = 0.4, color='#423D84', label='true'))
    autolabel(plt.bar(X_name+0.2, Mean_Predict_IFORCE, width = 0.4, color='#1E998A', label='predict'))
    plt.bar(X_name, [0, 0, 0, 0, 0], width = 0, color='w', tick_label=name_list)
    plt.legend()
    
    figure(4)
    plt.plot(X_name,Mean_True_IFORCE,'k.-',label='true')
    plt.plot(X_name,Mean_Predict_IFORCE,'r.--',label='predict')
    plt.xticks(X_name, name_list)
    plt.legend()
    
    Mean_True_Slope = [Mean_True_Slope_1, Mean_True_Slope_2, Mean_True_Slope_3, Mean_True_Slope_4, Mean_True_Slope_5]
    Mean_Predict_Slope = [Mean_Predict_Slope_1, Mean_Predict_Slope_2, Mean_Predict_Slope_3, Mean_Predict_Slope_4, Mean_Predict_Slope_5]
    figure(5)
    plt.plot(X_name,Mean_True_Slope,'b.-',label='standard')
    plt.plot(X_name,Mean_Predict_Slope,'r.--',label='simulative')
    plt.xticks(X_name, name_list)
    plt.legend()
    
    plt.show()


