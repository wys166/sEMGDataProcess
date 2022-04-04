from FileImport.ReadData import *
from ActivityDetect.GetActivity import *
from Denoise.LMSandNLMSDenoiseProcess import *
from FeatureExtract.GetFeature import *

'''
样本熵特征显示
'''
def SampEnShow(filename):
    sampling_rate = 1000
    force,Raw_1,Envelope_1,Raw_2,Envelope_2=LoadDataSetAfterProcess(filename)
    
    ################信号去噪################
    force = ButterLowFilter(force, 5, sampling_rate)#力去噪
    yn, error, weight=NLMSFiltering(xn, Raw_1, 9, 0.0052) #表面肌电原始信号去噪
    yn, error, weight=NLMSFiltering(xn, Raw_2, 9, 0.0052) #表面肌电原始信号去噪
    Envelope_1 = SlidingMeanFiltering(Envelope_1, 51)
    Envelope_2 = SlidingMeanFiltering(Envelope_2, 51)
    ################信号去噪################
    
    ################活动段检测################
    short_energy=GetActivity(force)
    start_index,end_index,force_start_y,force_end_y,\
    raw1_start_y,raw1_end_y,raw2_start_y,raw2_end_y,\
    envelope1_start_y,envelope1_end_y,envelope2_start_y,envelope2_end_y=GetStartAndEndByShortEnergyUsingInterp1d(short_energy, force, Raw_1,Envelope_1,Raw_2,Envelope_2)
    ################活动段检测################
    
    m = 1
    SampEnFeature_raw1 = []
    SampEnFeature_raw2 = []
    SampEnFeature_envelope1 = []
    SampEnFeature_envelope2 = []
    
    length = len(end_index)
    i = 0
    while i < length:
        data = Raw_1[start_index[i]:end_index[i]]        
        SampEnFeature_raw1.append(SampEnMatrix(data, m))
        
        data = Raw_2[start_index[i]:end_index[i]]        
        SampEnFeature_raw2.append(SampEnMatrix(data, m))
        
        data = Envelope_1[start_index[i]:end_index[i]]        
        SampEnFeature_envelope1.append(SampEnMatrix(data, m))
        
        data = Envelope_2[start_index[i]:end_index[i]]        
        SampEnFeature_envelope2.append(SampEnMatrix(data, m))
        print('第{}个特征'.format(i))
        
        i=i+1
    
    figure(1)
    plt.title('SampEn')
    plt.plot(range(len(SampEnFeature_raw1)),SampEnFeature_raw1,'k.-',label='SampEnFeature_raw1')
    plt.plot(range(len(SampEnFeature_raw2)),SampEnFeature_raw2,'g.-',label='SampEnFeature_raw2')
    plt.plot(range(len(SampEnFeature_envelope1)),SampEnFeature_envelope1,'b.-',label='SampEnFeature_envelope1')
    plt.plot(range(len(SampEnFeature_envelope2)),SampEnFeature_envelope2,'r.-',label='SampEnFeature_envelope2')
    plt.legend()

    plt.show()

'''
从标准化后的特征中挑选要观察的特征
'''
def select_feature(normalization_feature_value, feature_index, CH):
    select_feature_allCH = []
    i=0
    while i<CH:
        select_feature_allCH.append(normalization_feature_value[feature_index + i*feature_index])
        
        
        i=i+1
        
    return select_feature_allCH  
'''
从所有文件中提取的特征在同一个图中绘制处理
'''        
def All_Feature_Show(featurefilename):
    feature_value, feature_name, singleCh_feature_name = LoadFeatures(featurefilename)
    print(feature_name)
    print(singleCh_feature_name)
    normalization_feature_value = FeatureNormalization(feature_value)
    MAV = select_feature(normalization_feature_value, 0, 2)
    RMS = select_feature(normalization_feature_value, 1, 2)
    Var = select_feature(normalization_feature_value, 2, 2)
    IEMG = select_feature(normalization_feature_value, 3, 2)
    WL = select_feature(normalization_feature_value, 4, 2)
    WAMP = select_feature(normalization_feature_value, 5, 2)
    SSC = select_feature(normalization_feature_value, 6, 2)
    ZC = select_feature(normalization_feature_value, 7, 2)
    SampEnValue_raw = select_feature(normalization_feature_value, 8, 2)
    PKF = select_feature(normalization_feature_value, 9, 2)
    MNF = select_feature(normalization_feature_value, 10, 2)
    MDF = select_feature(normalization_feature_value, 11, 2)
    integral_envelope = select_feature(normalization_feature_value, 12, 2)
    mean_envelope = select_feature(normalization_feature_value, 13, 2)
    SampEnValue_envelope = select_feature(normalization_feature_value, 14, 2)
    force_mean = normalization_feature_value[-3] #力均值
    force_integral = normalization_feature_value[-2] #力积分值
    
    figure(0)
    plt.title('force')
    plt.plot(range(len(force_mean)),force_mean,'k.-',label='force_mean')
    plt.plot(range(len(force_integral)),force_integral,'r.-',label='force_integral')
    plt.ylim(0, 1)
    plt.legend()
    
    
    figure(1)
    plt.title('ch1')
    plt.plot(range(len(MAV[0])),MAV[0],'k.-',label='MAV')
    plt.plot(range(len(RMS[0])),RMS[0],'r.-',label='RMS')
    plt.plot(range(len(Var[0])),Var[0],'g.-',label='Var')
    plt.ylim(0, 1)
    plt.legend()
    
    figure(2)
    plt.title('ch2')
    plt.plot(range(len(MAV[1])),MAV[1],'k.-',label='MAV')
    plt.plot(range(len(RMS[1])),RMS[1],'r.-',label='RMS')
    plt.plot(range(len(Var[1])),Var[1],'g.-',label='Var')
    plt.ylim(0, 1)
    plt.legend()

    figure(3)
    plt.title('ch1')
    plt.plot(range(len(IEMG[0])),IEMG[0],'k.-',label='IEMG')
    plt.plot(range(len(WL[0])),WL[0],'r.-',label='WL')
    plt.plot(range(len(WAMP[0])),WAMP[0],'g.-',label='WAMP')
    plt.ylim(0, 1)
    plt.legend()
    
    figure(4)
    plt.title('ch2')
    plt.plot(range(len(IEMG[1])),IEMG[1],'k.-',label='IEMG')
    plt.plot(range(len(WL[1])),WL[1],'r.-',label='WL')
    plt.plot(range(len(WAMP[1])),WAMP[1],'g.-',label='WAMP')
    plt.ylim(0, 1)
    plt.legend()
    
    figure(5)
    plt.title('ch1')
    plt.plot(range(len(SSC[0])),SSC[0],'k.-',label='SSC')
    plt.plot(range(len(ZC[0])),ZC[0],'r.-',label='ZC')
    plt.plot(range(len(SampEnValue_raw[0])),SampEnValue_raw[0],'g.-',label='SampEnValue_raw')
    plt.ylim(0, 1)
    plt.legend()
    
    figure(6)
    plt.title('ch2')
    plt.plot(range(len(SSC[1])),SSC[1],'k.-',label='SSC')
    plt.plot(range(len(ZC[1])),ZC[1],'r.-',label='ZC')
    plt.plot(range(len(SampEnValue_raw[1])),SampEnValue_raw[1],'g.-',label='SampEnValue_raw')
    plt.ylim(0, 1)
    plt.legend()
    
    figure(7)
    plt.title('ch1')
    plt.plot(range(len(PKF[0])),PKF[0],'k.-',label='PKF')
    plt.plot(range(len(MNF[0])),MNF[0],'r.-',label='MNF')
    plt.plot(range(len(MDF[0])),MDF[0],'g.-',label='MDF')
    plt.ylim(0, 1)
    plt.legend()
    
    figure(8)
    plt.title('ch2')
    plt.plot(range(len(PKF[1])),PKF[1],'k.-',label='PKF')
    plt.plot(range(len(MNF[1])),MNF[1],'r.-',label='MNF')
    plt.plot(range(len(MDF[1])),MDF[1],'g.-',label='MDF')
    plt.ylim(0, 1)
    plt.legend()
    
    figure(9)
    plt.title('ch1')
    plt.plot(range(len(integral_envelope[0])),integral_envelope[0],'k.-',label='integral_envelope')
    plt.plot(range(len(mean_envelope[0])),mean_envelope[0],'r.-',label='mean_envelope')
    plt.plot(range(len(SampEnValue_envelope[0])),SampEnValue_envelope[0],'g.-',label='SampEnValue_envelope')
    plt.ylim(0, 1)
    plt.legend()
    
    figure(10)
    plt.title('ch2')
    plt.plot(range(len(integral_envelope[1])),integral_envelope[1],'k.-',label='integral_envelope')
    plt.plot(range(len(mean_envelope[1])),mean_envelope[1],'r.-',label='mean_envelope')
    plt.plot(range(len(SampEnValue_envelope[1])),SampEnValue_envelope[1],'g.-',label='SampEnValue_envelope')
    plt.ylim(0, 1)
    plt.legend()
    
    plt.show()
   
'''
feature:各个标签对应的所有特征，一行
label_class：标签的种类
'''    
def SameFeatrueSeparateShow(feature, label_class):
    feature_ch1 = feature[0]#通道1
    feature_ch2 = feature[1]#通道2
    feature_total = len(feature_ch1)
    feature_total_row = round(feature_total / label_class) 
    feature_single_ch1 = np.zeros((label_class, feature_total_row))
    feature_single_ch2 = np.zeros((label_class, feature_total_row))
    
    i=0
    while i<label_class:
        feature_single_ch1[i, :] = feature_ch1[i*feature_total_row : (i+1)*feature_total_row]
        feature_single_ch2[i, :] = feature_ch2[i*feature_total_row : (i+1)*feature_total_row]
        
        i=i+1
    
    return feature_single_ch1, feature_single_ch2
def SingleFeatureShow(featurefilename):
    feature_value, feature_name, singleCh_feature_name = LoadFeatures(featurefilename)
    print(feature_name)
    print(singleCh_feature_name)
    normalization_feature_value = FeatureNormalization(feature_value)
    
    MAV = select_feature(normalization_feature_value, 0, 2)
    RMS = select_feature(normalization_feature_value, 1, 2)
    Var = select_feature(normalization_feature_value, 2, 2)
    IEMG = select_feature(normalization_feature_value, 3, 2)
    WL = select_feature(normalization_feature_value, 4, 2)
    WAMP = select_feature(normalization_feature_value, 5, 2)
    SSC = select_feature(normalization_feature_value, 6, 2)
    ZC = select_feature(normalization_feature_value, 7, 2)
    SampEnValue_raw = select_feature(normalization_feature_value, 8, 2)
    PKF = select_feature(normalization_feature_value, 9, 2)
    MNF = select_feature(normalization_feature_value, 10, 2)
    MDF = select_feature(normalization_feature_value, 11, 2)
    integral_envelope = select_feature(normalization_feature_value, 12, 2)
    mean_envelope = select_feature(normalization_feature_value, 13, 2)
    SampEnValue_envelope = select_feature(normalization_feature_value, 14, 2)
    force_mean = normalization_feature_value[-3] #力均值
    force_integral = normalization_feature_value[-2] #力积分值
    
    feature_single_ch1, feature_single_ch2 = SameFeatrueSeparateShow(SampEnValue_envelope, 5)#5个类别
    
    
    figure(0)
    plt.title('force_mean')
    plt.plot(range(len(force_mean[0:100])),force_mean[0:100],'k.-',label='1')
    plt.plot(range(len(force_mean[100:200])),force_mean[100:200],'r.-',label='2')
    plt.plot(range(len(force_mean[200:300])),force_mean[200:300],'g.-',label='3')
    plt.plot(range(len(force_mean[300:400])),force_mean[300:400],'b.-',label='4')
    plt.ylim(0, 1)
    plt.legend()
    
    feature_num = 80
    figure(1)
    plt.title('force_integral')
    plt.plot(range(len(force_integral[0:feature_num])),force_integral[0:feature_num],'k.-',label='1')
    plt.plot(range(len(force_integral[feature_num:2*feature_num])),force_integral[feature_num:2*feature_num],'r.-',label='2')
    plt.plot(range(len(force_integral[2*feature_num:3*feature_num])),force_integral[2*feature_num:3*feature_num],'g.-',label='3')
    plt.plot(range(len(force_integral[3*feature_num:4*feature_num])),force_integral[3*feature_num:4*feature_num],'b.-',label='4')
    plt.plot(range(len(force_integral[4*feature_num:5*feature_num])),force_integral[4*feature_num:5*feature_num],'y.-',label='5')
    plt.ylim(0, 1)
    plt.legend()
    
    figure(2)
    plt.title('ch1')
    plt.plot(range(len(feature_single_ch1[0])),feature_single_ch1[0],'k.-',label='1')
    plt.plot(range(len(feature_single_ch1[1])),feature_single_ch1[1],'r.-',label='2')
    plt.plot(range(len(feature_single_ch1[2])),feature_single_ch1[2],'g.-',label='3')
    plt.plot(range(len(feature_single_ch1[3])),feature_single_ch1[3],'b.-',label='4')
    plt.plot(range(len(feature_single_ch1[4])),feature_single_ch1[4],'y.-',label='5')
    plt.ylim(0, 1)
    plt.legend()
    
    figure(3)
    plt.title('ch2')
    plt.plot(range(len(feature_single_ch2[0])),feature_single_ch2[0],'k.-',label='1')
    plt.plot(range(len(feature_single_ch2[1])),feature_single_ch2[1],'r.-',label='2')
    plt.plot(range(len(feature_single_ch2[2])),feature_single_ch2[2],'g.-',label='3')
    plt.plot(range(len(feature_single_ch2[3])),feature_single_ch2[3],'b.-',label='4')
    plt.plot(range(len(feature_single_ch2[4])),feature_single_ch2[4],'y.-',label='5')
    plt.ylim(0, 1)
    plt.legend()
    
    plt.show()

'''
本函数用于5种速度下的握力随时间积分值的大小
'''    
def Integral_forceShow(featurefilename): 
    feature_value, feature_name, singleCh_feature_name = LoadFeatures(featurefilename)
    print(feature_name)
    print(singleCh_feature_name)
    normalization_feature_value = FeatureNormalization(feature_value)  
    force_mean = normalization_feature_value[-3] #力均值
    force_integral = normalization_feature_value[-2] #力积分值 
    label_y = normalization_feature_value[-1] #标签
    
    IEMG = select_feature(normalization_feature_value, 3, 2)
    feature_single_ch1, feature_single_ch2 = SameFeatrueSeparateShow(IEMG, 5)#5个类别
    
    feature_num = 80
    label_1 = [1] * feature_num
    label_2 = [2] * feature_num
    label_3 = [3] * feature_num
    label_4 = [4] * feature_num
    label_5 = [5] * feature_num
    
    force_integral_1 = []
    force_integral_2 = []
    force_integral_3 = []
    force_integral_4 = []
    force_integral_5 = []
    
    Force_Integral = []
    Force_Label = [1, 2, 3, 4, 5]
    
    i=0
    while i<feature_num:
        force_integral_1.append(force_integral[i])
        force_integral_2.append(force_integral[i+feature_num])
        force_integral_3.append(force_integral[i+2*feature_num])
        force_integral_4.append(force_integral[i+3*feature_num])
        force_integral_5.append(force_integral[i+4*feature_num])
        Force_Integral.append([force_integral_1[i], force_integral_2[i], force_integral_3[i], force_integral_4[i], force_integral_5[i]])
        
        i=i+1
    
    figure(1)
    plt.title('force_integral')
    plt.plot(range(len(force_integral_1)),force_integral_1,'k.',label='1')
    plt.plot(range(len(force_integral_2)),force_integral_2,'r.',label='2')
    plt.plot(range(len(force_integral_3)),force_integral_3,'g.',label='3')
    plt.plot(range(len(force_integral_4)),force_integral_4,'b.',label='4')
    plt.plot(range(len(force_integral_5)),force_integral_5,'y.',label='5')
    plt.ylim(0, 1)
    plt.legend()
    
    figure(2)
    plt.title('force_integral_scater')
    plt.scatter(force_integral_1, label_1, c='k', marker='.', label='1')
    plt.scatter(force_integral_2, label_2, c='r', marker='.', label='2')
    plt.scatter(force_integral_3, label_3, c='g', marker='.', label='3')
    plt.scatter(force_integral_4, label_4, c='b', marker='.', label='4')
    plt.scatter(force_integral_5, label_5, c='y', marker='.', label='5')
    plt.legend()
    
    figure(3)
    plt.title('force_integral_scater')
    plt.scatter(feature_single_ch1[0], force_integral_1, c='k', marker='.', label='1')
    plt.scatter(feature_single_ch1[1], force_integral_2, c='r', marker='.', label='2')
    plt.scatter(feature_single_ch1[2], force_integral_3, c='g', marker='.', label='3')
    plt.scatter(feature_single_ch1[3], force_integral_4, c='b', marker='.', label='4')
    plt.scatter(feature_single_ch1[4], force_integral_5, c='y', marker='.', label='5')
    plt.legend()
    
    figure(4)
    plt.title('force_integral_scater')
    plt.scatter(feature_single_ch2[0], force_integral_1, c='k', marker='.', label='1')
    plt.scatter(feature_single_ch2[1], force_integral_2, c='r', marker='.', label='2')
    plt.scatter(feature_single_ch2[2], force_integral_3, c='g', marker='.', label='3')
    plt.scatter(feature_single_ch2[3], force_integral_4, c='b', marker='.', label='4')
    plt.scatter(feature_single_ch2[4], force_integral_5, c='y', marker='.', label='5')
    plt.legend()
    
    figure(5)
    plt.title('force_integral_plot')
    plt.plot(Force_Integral[0], Force_Label, c='k', marker='.', label='1')
    plt.plot(Force_Integral[1], Force_Label, c='r', marker='.', label='2')
    plt.plot(Force_Integral[2], Force_Label, c='g', marker='.', label='3')
    plt.plot(Force_Integral[3], Force_Label, c='b', marker='.', label='4')
    plt.plot(Force_Integral[4], Force_Label, c='y', marker='.', label='5')
    plt.legend()
    
    
    plt.show()
        


