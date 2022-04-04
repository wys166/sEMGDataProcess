from FileImport.ReadData import *
from RegressionFun.RegressionFeatureProcess import *

# 显示高度
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.- 0.2, 1.03*height, '%s' % float(height))
        
        
def LoadRegressionResultAnalysisFile(RegressionResultMeanStdfile):
    with open(RegressionResultMeanStdfile, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        dataset = dataset[1:]#从第二行开始读取
        
        force_integral_mean_std = dataset[0]
        force_integral_mean_std = force_integral_mean_std[1:]
        
        i = 0
        while i<5:
            force_integral_mean_std[i] = float(force_integral_mean_std[i])
            force_integral_mean_std[i] = round(force_integral_mean_std[i], 5)
            i=i+1            
    csvfile.close()
    
    return force_integral_mean_std

def RegressionResultMeanStdShow(filename):
    force_integral_mean_std = LoadRegressionResultAnalysisFile(filename)
    
    name_list = ['Linear', 'Ridge', 'Knn', 'MLP', 'SVM']
    
    
    figure(1)  
    plt.title('Mean Std')  
    autolabel(plt.bar(name_list, force_integral_mean_std, width = 0.8, color='krgby', label='mean std'))
    plt.legend()
    
    plt.show()
    
    
RegressionResultMeanStdShow(r'D:\\sEMGData\\RegressionResultMeanStd.csv')   
    
    
    
    