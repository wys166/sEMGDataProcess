from FileImport.ReadData import *
from RegressionFun.RegressionFeatureProcess import *

def GetPerSubjectStdForPredictAndTrue(filepath):
   
    filename = [r'LinearRegressionResult.csv', r'RidgeRegressionResult.csv', r'KnnRegressionResult.csv', r'MLPRegressionResult.csv', r'SVMRegressionResult.csv']
    Force_Integral_Std = []
    Slope_Std_Value = []
    
    i=0
    while i<5:
        RegressionResultFilename = filepath + r'\\' + filename[i]
        y_label, force_integral_true, force_integral_predict, slope_true, slope_predict = LoadRegressionResultFile(RegressionResultFilename)
        force_integral_error_value, force_integral_std_value = GetStdForPredictAndTrue(force_integral_predict, force_integral_true)
        Force_Integral_Std.append(force_integral_std_value)
        
        slope_std_value = np.std(np.array(slope_predict) - np.array(slope_true))
        Slope_Std_Value.append(slope_std_value)        
        
        i=i+1
    
    force_integral_std_value_max = max(Force_Integral_Std)
    force_integral_std_value_min = min(Force_Integral_Std)
    Normalization_Force_Integral_Std = list((array(Force_Integral_Std)-force_integral_std_value_min)/(force_integral_std_value_max-force_integral_std_value_min))
    
    # slope_std_value_max = max(Slope_Std_Value)
    # slope_std_value_min = min(Slope_Std_Value)
    # Normalization_Slope_Std = list((array(Slope_Std_Value)-slope_std_value_min)/(slope_std_value_max-slope_std_value_min))
    Normalization_Slope_Std = Slope_Std_Value
    #################将相关参数保存到CSV文件中
    regressionfucname = ['reggression', 'Linear', 'Ridge', 'Knn', 'MLP', 'SVM']
    Normalization_Force_Integral_Std.insert(0, 'Force_Integral_Std')
    Normalization_Slope_Std.insert(0, 'Slope_Std')
    newdata = [regressionfucname, Normalization_Force_Integral_Std, Normalization_Slope_Std]
    with open(filepath + r'\\' + r'RegressionResultAnalysis.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerows(newdata)
        csvfile.close()
    #################将相关参数保存到CSV文件中   
    
'''
将每个人下得到的力积分值标准差与斜率大小文件读取出来
'''
def LoadRegressionResultAnalysisFile(RegressionResultAnalysisfile):
    with open(RegressionResultAnalysisfile, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        dataset = dataset[1:]#从第二行开始读取
        
        force_integral_std = dataset[0]
        force_integral_std = force_integral_std[1:]
        
        slope_std = dataset[1]
        slope_std = slope_std[1:]
        
        i = 0
        while i<5:
            force_integral_std[i] = float(force_integral_std[i])
            slope_std[i] = float(slope_std[i])
            i=i+1            
    csvfile.close()
    
    return force_integral_std, slope_std
    

########列表转置########
def List_Transpose(data):
    data = list(map(list, zip(*data)))    #转置
    return data
########列表转置########
 
    
    