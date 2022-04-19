from FileImport import *
from ActivityDetect.GetActivity import *
from FeatureExtract.GetFeature import *
from Denoise.ForceDenoiseProcess import *
from Denoise.LMSandNLMSDenoiseProcess import *
from Denoise.EnvelopeDenoiseProcess import *
from ActivityDetect.ForceActivity import *


######################从时域、频域、提取表面肌电原始信号和包洛信号特征值#####################
def ExtractFeatures(filename, Raw1_noise, Raw2_noise, force_speed):
    sampling_rate = 1000
    force,Raw_1,Envelope_1,Raw_2,Envelope_2=LoadDataSetAfterProcess(filename)
        
    ################信号去噪################
    force = ButterLowFilter(force, 5, sampling_rate)#力去噪
    force = np.rint(force)#四舍五入取整
    yn, Raw_1, weight=NLMSFiltering(Raw1_noise, Raw_1, 9, 0.0052) #表面肌电原始信号去噪
    yn, Raw_2, weight=NLMSFiltering(Raw2_noise, Raw_2, 9, 0.0052) #表面肌电原始信号去噪
    Envelope_1 = SlidingMeanFiltering(Envelope_1, 51)
    Envelope_2 = SlidingMeanFiltering(Envelope_2, 51)
    ################信号去噪################
    
    # ################通过短时能量发进行活动段检测################
    # short_energy=GetActivity(force)
    # start_index,end_index,force_start_y,force_end_y,\
    # raw1_start_y,raw1_end_y,raw2_start_y,raw2_end_y,\
    # envelope1_start_y,envelope1_end_y,envelope2_start_y,envelope2_end_y=GetStartAndEndByShortEnergyUsingInterp1d(short_energy, force, Raw_1,Envelope_1,Raw_2,Envelope_2)
    # ################通过短时能量发进行活动段检测################
    
    ################通过力阈值进行活动段检测################
    start_index,end_index,force_start_y,force_end_y,\
    raw1_start_y,raw1_end_y,raw2_start_y,raw2_end_y,\
    envelope1_start_y,envelope1_end_y,envelope2_start_y,envelope2_end_y=GetStartAndEndByForceUsingInterp1d(force, Raw_1,Envelope_1,Raw_2,Envelope_2, force_speed)
    ################通过力阈值进行活动段检测################
    
    #############力特征值
    force_integral = []#力积分值
    force_mean =[]#力均值
    #############力特征值
    
    #######################################################表面肌电原始信号
    #############表面肌电原始信号时域特征值
    MAV_raw1 = []
    RMS_raw1 = []
    Var_raw1 = []
    IEMG_raw1 = []
    WL_raw1 = []
    WAMP_raw1 = []
    SSC_raw1 = []
    ZC_raw1 = []
    SampEnValue_raw1 = []
    
    MAV_raw2 = []
    RMS_raw2 = []
    Var_raw2 = []
    IEMG_raw2 = []
    WL_raw2 = []
    WAMP_raw2 = []
    SSC_raw2 = []
    ZC_raw2 = []
    SampEnValue_raw2 = []
    #############表面肌电原始信号时域特征值
    
    #############表面肌电原始信号频域特征值
    PKF_raw1 = []
    MNF_raw1 = []
    MDF_raw1 = []
    
    PKF_raw2 = []
    MNF_raw2 = []
    MDF_raw2 = []
    #############表面肌电原始信号频域特征值
    #######################################################表面肌电原始信号
    
    #######################################################表面肌电包洛信号
    #############表面肌电包洛信号特征值
    integral_envelope1 = []
    mean_envelope1 = []
    SampEnValue_envelope1 = []
    
    integral_envelope2 = []
    mean_envelope2 = []  
    SampEnValue_envelope2 = []  
    #############表面肌电包洛信号特征值
    #######################################################表面肌电包洛信号
    
    Num_Feature = 80
    length = len(end_index)
    if length<Num_Feature:
        print('识别的特征数量少于需要提取的特征数量！')
        return
        
    i = 0
    while i < Num_Feature:
        ##########力的积分和均值##########
        force_data = force[start_index[i]:end_index[i]]
        force_integral_, force_mean_ = FeatureExtractForForce(force_data)
        force_integral.append(force_integral_)
        force_mean.append(force_mean_)
        ##########力的积分和均值##########
        
        ##########表面肌电原始信号时域特征##########
        Raw = Raw_1[start_index[i]:end_index[i]]
        MAV_, RMS_, Var_, IEMG_, WL_, WAMP_, SSC_, ZC_, SampEnValue_ = TDFeatureExtractForRaw(Raw)#MAV, RMS, Var, IEMG, WL, WAMP, SSC, ZC
        MAV_raw1.append(MAV_)
        RMS_raw1.append(RMS_)
        Var_raw1.append(Var_)
        IEMG_raw1.append(IEMG_)
        WL_raw1.append(WL_)
        WAMP_raw1.append(WAMP_)
        SSC_raw1.append(SSC_)
        ZC_raw1.append(ZC_)
        SampEnValue_raw1.append(SampEnValue_)
        
        Raw = Raw_2[start_index[i]:end_index[i]]
        MAV_, RMS_, Var_, IEMG_, WL_, WAMP_, SSC_, ZC_, SampEnValue_ = TDFeatureExtractForRaw(Raw)#MAV, RMS, Var, IEMG, WL, WAMP, SSC, ZC
        MAV_raw2.append(MAV_)
        RMS_raw2.append(RMS_)
        Var_raw2.append(Var_)
        IEMG_raw2.append(IEMG_)
        WL_raw2.append(WL_)
        WAMP_raw2.append(WAMP_)
        SSC_raw2.append(SSC_)
        ZC_raw2.append(ZC_)
        SampEnValue_raw2.append(SampEnValue_)
        ##########表面肌电原始信号时域特征##########
        
        ##########表面肌电原始信号频域特征##########
        Raw = Raw_1[start_index[i]:end_index[i]]
        PKF_, MNF_, MDF_ = FDFeatureExtractForRaw(Raw, sampling_rate)
        PKF_raw1.append(PKF_)
        MNF_raw1.append(MNF_)
        MDF_raw1.append(MDF_)
        
        Raw = Raw_2[start_index[i]:end_index[i]]
        PKF_, MNF_, MDF_ = FDFeatureExtractForRaw(Raw, sampling_rate)
        PKF_raw2.append(PKF_)
        MNF_raw2.append(MNF_)
        MDF_raw2.append(MDF_)        
        ##########表面肌电原始信号频域特征##########
        
        ##########表面肌电包洛信号时域特征##########
        Envelope = Envelope_1[start_index[i]:end_index[i]]
        envelope_integral_, envelope_mean_, SampEnValue_ = TDFeatureExtractForEnvelope(Envelope)
        integral_envelope1.append(envelope_integral_)
        mean_envelope1.append(envelope_mean_)
        SampEnValue_envelope1.append(SampEnValue_)
        
        Envelope = Envelope_2[start_index[i]:end_index[i]]
        envelope_integral_, envelope_mean_, SampEnValue_ = TDFeatureExtractForEnvelope(Envelope)
        integral_envelope2.append(envelope_integral_)
        mean_envelope2.append(envelope_mean_)
        SampEnValue_envelope2.append(SampEnValue_)
        ##########表面肌电包洛信号时域特征##########
        
        print('共{}个特征，已提取{}个特征'.format(Num_Feature, i+1))
        i = i + 1 
 
    return [MAV_raw1, RMS_raw1, Var_raw1, IEMG_raw1, WL_raw1, WAMP_raw1, SSC_raw1, ZC_raw1, SampEnValue_raw1,\
            PKF_raw1, MNF_raw1, MDF_raw1,\
            integral_envelope1, mean_envelope1, SampEnValue_envelope1,\
            MAV_raw2, RMS_raw2, Var_raw2, IEMG_raw2, WL_raw2, WAMP_raw2, SSC_raw2, ZC_raw2, SampEnValue_raw2,\
            PKF_raw2, MNF_raw2, MDF_raw2,\
            integral_envelope2, mean_envelope2, SampEnValue_envelope2,\
            force_mean, force_integral]
######################从时域、频域、提取表面肌电原始信号和包洛信号特征值#####################

######################将提取的特征值保存下来#####################
feature_list = [['MAV_raw1', 'RMS_raw1', 'Var_raw1', 'IEMG_raw1', 'WL_raw1', 'WAMP_raw1', 'SSC_raw1', 'ZC_raw1', 'SampEnValue_raw1',\
                 'PKF_raw1', 'MNF_raw1', 'MDF_raw1',\
                 'integral_envelope1', 'mean_envelope1', 'SampEnValue_envelope1',\
                 'MAV_raw2', 'RMS_raw2', 'Var_raw2', 'IEMG_raw2', 'WL_raw2', 'WAMP_raw2', 'SSC_raw2', 'ZC_raw2', 'SampEnValue_raw2',\
                 'PKF_raw2', 'MNF_raw2', 'MDF_raw2',\
                 'integral_envelope2', 'mean_envelope2', 'SampEnValue_envelope2',\
                 'force_mean', 'force_integral',\
                 'label']]
def SaveFeatures(filepath, file_list):
    noisefilename = filepath + r'\\' + 'Reference.csv' #参考噪声
    Raw1_noise,Envelope1_noise,Raw2_noise,Envelope2_noise = LoadRawDataSetByChNum(noisefilename, CH=4)
    
    length = len(file_list)
    
    all_features = feature_list
    i = 0
    while i<length:
        filename = filepath + r'\\' + file_list[i]
        force_speed = int(filename[-5])#倒数第五位表示力速度1/2/3/4/5
        features = ExtractFeatures(filename, Raw1_noise, Raw2_noise, force_speed)
        features.append([i+1] * len(features[0]))
        features = List_Transpose(features)
        all_features = all_features + features
        
        print('已处理{}个文件'.format(i+1))    
        i=i+1

    #################将相关参数保存到CSV文件中
    with open(filepath + r'\\Feature.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerows(all_features)
        csvfile.close()
    #################将相关参数保存到CSV文件中  
######################将提取的特征值保存下来#####################        
 
'''
filepath:存放数据集的路径
'''       
def ExtractAndSaveFeatureForPerSubject(filepath):
    file_list = ['T1.csv', 'T2.csv', 'T3.csv', 'T4.csv', 'T5.csv']
    SaveFeatures(filepath, file_list)
    


