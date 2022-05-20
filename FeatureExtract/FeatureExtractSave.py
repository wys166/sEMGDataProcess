from FileImport.ReadData import *
from ActivityDetection.DetectionFun import *
from FeatureExtract.FeatureExtractFun import *
from Denoise.EnvelopeDenoiseFun import *
from Denoise.ForceDenoiseFun import *
from Denoise.RawDenoiseFun import *

######################Extract all features#####################
'''
Please note: force MEAN feature is extracted, but it isn't used in the study
'''
def ExtractFeatures(filename, Raw1_noise, Raw2_noise, force_speed):
    sampling_rate = 1000
    force,Raw_1,Envelope_1,Raw_2,Envelope_2=LoadDataSet(filename)
        
    ################denoise################
    force = ButterLowFilter(force, 5, sampling_rate)
    force = np.rint(force)
    yn, Raw_1, weight=NLMSFiltering(Raw1_noise, Raw_1, 9, 0.0052) 
    yn, Raw_2, weight=NLMSFiltering(Raw2_noise, Raw_2, 9, 0.0052) 
    Envelope_1 = SlidingMeanFiltering(Envelope_1, 51)
    Envelope_2 = SlidingMeanFiltering(Envelope_2, 51)
    ################denoise################
    
    ################muscle contraction status detection################
    subject_index = GetSubjectIndex(filename)
    start_index,end_index,force_start_y,force_end_y,\
    raw1_start_y,raw1_end_y,raw2_start_y,raw2_end_y,\
    envelope1_start_y,envelope1_end_y,envelope2_start_y,envelope2_end_y=GetStartAndEndByForce(force, Raw_1,Envelope_1,Raw_2,Envelope_2, force_speed, subject_index)
    ################muscle contraction status detection################
    
    #############feature of force
    force_integral = []#IFORCE
    force_mean =[]#force MEAN, it isn't used in the study
    #############feature of force
    
    #######################################################raw
    #############Time domain feature
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
    #############Time domain feature
    
    #############Frequency domain feature
    PKF_raw1 = []
    MNF_raw1 = []
    MDF_raw1 = []
    
    PKF_raw2 = []
    MNF_raw2 = []
    MDF_raw2 = []
    #############Frequency domain feature
    #######################################################raw
    
    #######################################################envelope
    integral_envelope1 = []
    mean_envelope1 = []
    SampEnValue_envelope1 = []
    
    integral_envelope2 = []
    mean_envelope2 = []  
    SampEnValue_envelope2 = []  
    #######################################################envelope
    
    Num_Feature = 80 #retain 80 samples
    length = len(end_index)
    if length<Num_Feature:
        print('The number of sample is insufficient and less than 80!')
        return
        
    i = 0
    while i < Num_Feature:
        ##########extract force feature##########
        force_data = force[start_index[i]:end_index[i]]
        force_integral_, force_mean_ = FeatureExtractForForce(force_data)
        force_integral.append(force_integral_)
        force_mean.append(force_mean_)
        ##########extract force feature##########
        
        ##########extract time domain feature for raw##########
        Raw = Raw_1[start_index[i]:end_index[i]]
        MAV_, RMS_, Var_, IEMG_, WL_, WAMP_, SSC_, ZC_, SampEnValue_ = TDFeatureExtractForRaw(Raw)#MAV, RMS, Var, IEMG, WL, WAMP, SSC, ZC, SampEn
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
        MAV_, RMS_, Var_, IEMG_, WL_, WAMP_, SSC_, ZC_, SampEnValue_ = TDFeatureExtractForRaw(Raw)#MAV, RMS, Var, IEMG, WL, WAMP, SSC, ZC, SampEn
        MAV_raw2.append(MAV_)
        RMS_raw2.append(RMS_)
        Var_raw2.append(Var_)
        IEMG_raw2.append(IEMG_)
        WL_raw2.append(WL_)
        WAMP_raw2.append(WAMP_)
        SSC_raw2.append(SSC_)
        ZC_raw2.append(ZC_)
        SampEnValue_raw2.append(SampEnValue_)
        ##########extract time domain feature for raw##########
        
        ##########extract frequency domain feature for raw##########
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
        ##########extract frequency domain feature for raw##########
        
        ##########extract feature for envelope##########
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
        ##########extract feature for envelope##########
        
        print('Number of total sample :{}, Extracted sample ID:{}'.format(Num_Feature, i+1))
        i = i + 1
 
    return [MAV_raw1, RMS_raw1, Var_raw1, IEMG_raw1, WL_raw1, WAMP_raw1, SSC_raw1, ZC_raw1, SampEnValue_raw1,\
            PKF_raw1, MNF_raw1, MDF_raw1,\
            integral_envelope1, mean_envelope1, SampEnValue_envelope1,\
            MAV_raw2, RMS_raw2, Var_raw2, IEMG_raw2, WL_raw2, WAMP_raw2, SSC_raw2, ZC_raw2, SampEnValue_raw2,\
            PKF_raw2, MNF_raw2, MDF_raw2,\
            integral_envelope2, mean_envelope2, SampEnValue_envelope2,\
            force_mean, force_integral]
######################Extract all features#####################

######################Save feature to file#####################
'''
Please note: force MEAN feature(force_mean) is extracted, but it isn't used in the study
'''
feature_list = [['MAV_raw1', 'RMS_raw1', 'Var_raw1', 'IEMG_raw1', 'WL_raw1', 'WAMP_raw1', 'SSC_raw1', 'ZC_raw1', 'SampEnValue_raw1',\
                 'PKF_raw1', 'MNF_raw1', 'MDF_raw1',\
                 'integral_envelope1', 'mean_envelope1', 'SampEnValue_envelope1',\
                 'MAV_raw2', 'RMS_raw2', 'Var_raw2', 'IEMG_raw2', 'WL_raw2', 'WAMP_raw2', 'SSC_raw2', 'ZC_raw2', 'SampEnValue_raw2',\
                 'PKF_raw2', 'MNF_raw2', 'MDF_raw2',\
                 'integral_envelope2', 'mean_envelope2', 'SampEnValue_envelope2',\
                 'force_mean', 'force_integral',\
                 'label']]
def SaveFeatures(filepath, file_list):
    noisefilename = filepath + r'\\' + 'Reference.csv'
    Raw1_noise,Envelope1_noise,Raw2_noise,Envelope2_noise = LoadReferenceDataSetByChNum(noisefilename, CH=4)
    
    length = len(file_list)
    
    all_features = feature_list
    i = 0
    while i<length:
        filename = filepath + r'\\' + file_list[i]
        force_speed = int(filename[-5])#grip pattern
        features = ExtractFeatures(filename, Raw1_noise, Raw2_noise, force_speed)
        features.append([i+1] * len(features[0]))
        features = List_Transpose(features)
        all_features = all_features + features
        
        print('Processed file number: {}'.format(i+1))    
        i=i+1

    #################Save feature to file
    with open(filepath + r'\\Feature.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerows(all_features)
        csvfile.close()
    #################Save feature to file 
######################Save feature to file#####################        
 
######################Save to file##################### 
'''
filepath: file path obeys specific format, for example: filepath=r'D:\\DataSet\\Subject20'
Please note: This function takes a lot of time to run
'''     
def SaveToFile(filepath):
    file_list = ['T1.csv', 'T2.csv', 'T3.csv', 'T4.csv', 'T5.csv']
    SaveFeatures(filepath, file_list)
######################Save to file#####################      
    
    
'''
Save feature to file
input: obey specific format, for example: r'D:\\DataSet\\Subject20'
Please note: This function takes a lot of time to run
'''        
SaveToFile(r'D:\\DataSet\\Subject20')
