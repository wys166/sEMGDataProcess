from sklearn import decomposition
from FileImport.ReadData import *


#####################Dimensionality reduction show by PCA#####################
'''
In order to observe how discrete these features are
'''
def FeatureReductionByPCA(featurefilename):
    feature_value, feature_name, singleCh_feature_name = LoadFeatures(featurefilename)
    normalization_feature_value = FeatureNormalization(feature_value)
    Y = normalization_feature_value[-2]
    window_length = 80
    Force = [Y[0:window_length*1], Y[window_length*1:window_length*2], Y[window_length*2:window_length*3], Y[window_length*3:window_length*4], Y[window_length*4:window_length*5]]

    normalization_feature_value_ = normalization_feature_value[:-3]
    normalization_feature_value_T = List_Transpose(normalization_feature_value_)
    pca = decomposition.PCA(n_components=2)
    pca.fit(normalization_feature_value_T)
    X = pca.transform(normalization_feature_value_T)
    length = len(X)
    Feature_list_1 = []
    Feature_list_2 = []
    i = 0
    while i<length:
        Feature_list_1.append(X[i][0])
        Feature_list_2.append(X[i][1])
        
        i=i+1
    
    Feature_X =[Feature_list_1[0:window_length*1], Feature_list_1[window_length*1:window_length*2], Feature_list_1[window_length*2:window_length*3], Feature_list_1[window_length*3:window_length*4], Feature_list_1[window_length*4:window_length*5]]
    Feature_Y =[Feature_list_2[0:window_length*1], Feature_list_2[window_length*1:window_length*2], Feature_list_2[window_length*2:window_length*3], Feature_list_2[window_length*3:window_length*4], Feature_list_2[window_length*4:window_length*5]]

    figure(1)
    plt.title('PCA Dimensionality Reduction')
    plt.plot(Feature_X[0], Feature_Y[0], 'k.', label='T1')
    plt.plot(Feature_X[1], Feature_Y[1], 'r.', label='T2')
    plt.plot(Feature_X[2], Feature_Y[2], 'g.', label='T3')
    plt.plot(Feature_X[3], Feature_Y[3], 'b.', label='T4')
    plt.plot(Feature_X[4], Feature_Y[4], 'y.', label='T5')
     
    plt.show()
#####################Dimensionality reduction show by PCA#####################  
  
   
'''
Show sample dispersion by PCA
input: featurefilename=r'D:\\DataSet\\FeatureDataShow\\Feature.csv'
''' 
FeatureReductionByPCA(r'D:\\DataSet\\FeatureDataShow\\Feature.csv')

