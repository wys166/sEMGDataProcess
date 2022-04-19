from FileImport.ReadData import *
from Denoise.LMSandNLMSDenoiseProcess import *
import math
import scipy

########时域下计算信号平均功率#########
'''
时域下计算离散信号平均功率时，先求信号能量之和（平方和），再取均值
'''
def GerPowerTime(data):#计算功率
    length = len(data)
    
    Sum = 0
    i=0
    while i < length:
        d = data[i]/1
        Sum = Sum + d**2
      
        i = i+1
    P = Sum/length

    return P
########时域下计算信号平均功率#########

########频域下计算信号平均功率#########
'''
频域下计算离散信号平均功率时，先求信号的功率谱，再将功率谱求和
'''
def GerPowerFre(data, sampling_rate):#计算功率
    freqs, power_spectrum = signal.periodogram(data, fs=sampling_rate, window='boxcar', nfft=None,  scaling='spectrum')
    
    return sum(power_spectrum)
    
########频域下计算信号平均功率#########

########时域下信噪比展示#########  
def SNRShow_Time(filename1, filename2):#时域下
    sampling_rate = 1000
    force, Raw_1,Envelope_1,Raw_2,Envelope_2 = LoadDataSetAfterProcess(filename1)
    raw_1,envelope_1,raw_2,envelope_2 = LoadRawDataSetByChNum(filename2, CH=4)
    
    start_index = 0
    end_index = 59400     
    sig = Raw_1[start_index:end_index]
    noise = raw_1[:end_index-start_index]
    
    P_sig = GerPowerTime(sig)
    P_noise = GerPowerTime(noise)
    SNR_org = round(10* math.log((P_sig-P_noise)/P_noise, 10), 4)
    
    order = range(9, 13, 1) #滤波器阶数
    n = 0.0058#NLMS算法步长修正因子
    
    SNR_filter = []
    SNR_dif = []
    i=0
    while i<len(order):
        yn, sig_denoise, weight=NLMSFiltering(noise, sig, order[i], n)  
        P_sig_denoise = GerPowerTime(sig_denoise)
        P_yn = GerPowerTime(yn)
        P_denoise = P_noise - P_yn
        SNR_filter.append(round(10 * math.log((P_sig_denoise-P_denoise)/P_denoise, 10), 4))
        SNR_dif.append(SNR_filter[i] - SNR_org)
        
        print('第：{}次'.format(i + 1))
        i=i+1 
    
    print("原来信噪比：{}" .format(SNR_org))
    print("现在信噪比：{}" .format(SNR_filter))
    print("信噪比之差：{}" .format(SNR_dif))
    
    name=str(n)
    figure(1)
    plt.title(name)
    plt.plot(range(len(SNR_filter)),SNR_filter,'r.-',label='SNR_filter')
    plt.legend()
    
    figure(2)
    plt.title(name)
    plt.plot(range(len(SNR_dif)),SNR_dif,'r.-',label='SNR_dif')
    plt.legend()
    
    plt.show()     
########时域下信噪比展示######### 

########频域下信噪比展示######### 
'''
步长因子恒定，改变滤波器阶数
'''
def SNRShow_Fre(filename1, filename2):  #频域下
    sampling_rate = 1000
    force, Raw_1,Envelope_1,Raw_2,Envelope_2 = LoadDataSetAfterProcess(filename1)
    raw_1,envelope_1,raw_2,envelope_2 = LoadRawDataSetByChNum(filename2, CH=4)
        
    sig = Raw_1[:100000]
    noise = raw_1[:100000]
    
    P_sig = GerPowerFre(sig, sampling_rate)
    P_noise = GerPowerFre(noise, sampling_rate)
    SNR_org = round(10* math.log((P_sig-P_noise)/P_noise, 10), 4)
    
    order = arange(6, 20, 1) #滤波器阶数
    n = 0.003#NLMS算法步长修正因子
    
    SNR_filter = []
    SNR_dif = []
    i=0
    while i<len(order):
        yn, sig_denoise, weight=NLMSFiltering(noise, sig, order[i], n)    
        P_sig_denoise = GerPowerFre(sig_denoise, sampling_rate)
        P_yn = GerPowerFre(yn, sampling_rate)
        P_remain_noise = P_noise - P_yn
        try:
            SNR_filter.append(round(10 * math.log((P_sig_denoise-P_remain_noise)/P_remain_noise, 10), 4))
        except:
            print("滤波器的最大阶数为：{}".format(i-1))
            print("剩余噪声：{}".format(P_remain_noise))
            print("有用信号噪声-剩余噪声：{}".format(P_sig_denoise-P_remain_noise))
            break
        SNR_dif.append(SNR_filter[i] - SNR_org)
        
        print('第{}次'.format(i + 1))
        i=i+1 
    
    print("原来信噪比：{}" .format(SNR_org))
    print("现在信噪比：{}" .format(SNR_filter))
    print("信噪比之差：{}" .format(SNR_dif))
    
    name=str(n)
    figure(1)
    plt.title(name)
    plt.plot(range(len(SNR_filter)),SNR_filter,'r.-',label='SNR_filter')
    plt.legend()
    
    figure(2)
    plt.title(name)
    plt.plot(range(len(SNR_dif)),SNR_dif,'r.-',label='SNR_dif')
    plt.legend()
    
    plt.show()
########频域下信噪比展示######### 
 

########滤波器最佳阶数下计算信噪比######### 
'''
滤波器最佳阶数下计算信噪比
'''
def SNRShow_Fre_Opt_Order(filename1, filename2, order):
    sampling_rate = 1000
    force, Raw_1,Envelope_1,Raw_2,Envelope_2 = LoadDataSetAfterProcess(filename1)
    raw_1,envelope_1,raw_2,envelope_2 = LoadRawDataSetByChNum(filename2, CH=4)
        
    sig = Raw_1[:100000]
    noise = raw_1[:100000]
    
    P_sig = GerPowerFre(sig, sampling_rate)
    P_noise = GerPowerFre(noise, sampling_rate)
    SNR_org = round(10* math.log((P_sig-P_noise)/P_noise, 10), 4)
    
    n = 0.00000464#LMS算法步长修正因子
      
    yn, sig_denoise, weight = LMSFiltering(noise, sig, order, n)  
    P_sig_denoise = GerPowerFre(sig_denoise, sampling_rate)
    P_yn = GerPowerFre(yn, sampling_rate)
    P_remain_noise = P_noise - P_yn
    print("剩余噪声：{}".format(P_remain_noise))
    print("有用信号噪声-剩余噪声：{}".format(P_sig_denoise-P_remain_noise))
    
    SNR_filter = round(10 * math.log((P_sig_denoise-P_remain_noise)/P_remain_noise, 10), 4)
    SNR_dif = SNR_filter - SNR_org   

    
    print("原来信噪比：{}" .format(SNR_org))
    print("现在信噪比：{}" .format(SNR_filter))
    print("信噪比之差：{}" .format(SNR_dif))
    
    R, p_value = scipy.stats.pearsonr(noise, yn)
    print("相关性：{}" .format(R))
    
########滤波器最佳阶数下计算信噪比######### 

########滤波器最佳阶数不变的情况下，改变步长n计算信噪比######### 
'''
滤波器阶数固定时，改变步长因子，信噪比展示
'''
def SNRShow_Fre_Change_n(filename1, filename2):
    sampling_rate = 1000
    force, Raw_1,Envelope_1,Raw_2,Envelope_2 = LoadDataSetAfterProcess(filename1)
    raw_1,envelope_1,raw_2,envelope_2 = LoadRawDataSetByChNum(filename2, CH=4)
        
    start_index = 0
    end_index = 100000 
    sig=Raw_1[start_index:end_index]
    noise=raw_1[0:end_index-start_index]
    
    P_sig = GerPowerFre(sig, sampling_rate)
    P_noise = GerPowerFre(noise, sampling_rate)
    SNR_org = round(10* math.log((P_sig-P_noise)/P_noise, 10), 4) #原信号信噪比为28.9798 
    
    ####################NLMS算法####################
    # order = 11 #滤波器阶数  
    # n = arange(0.007, 0.008, 0.00002)#NLMS算法步长修正因子
    ####################NLMS算法####################
    
    ####################LMS算法####################
    order = 9 #滤波器阶数  
    n = arange(0.0000045, 0.0000059, 0.00000001)#LMS算法步长修正因子
    ####################LMS算法####################
    
    print("所需的运算次数：{}".format(len(n)))

    SNR_filter = []
    SNR_dif = []
    i=0
    while i<len(n):
        # yn, sig_denoise, weight=NLMSFiltering(noise, sig, order, n[i])  
        yn, sig_denoise, weight = LMSFiltering(noise, sig, order, n[i])  
        P_sig_denoise = GerPowerFre(sig_denoise, sampling_rate)
        P_yn = GerPowerFre(yn, sampling_rate)
        P_remain_noise = P_noise - P_yn
        try:
            SNR_filter.append(round(10 * math.log((P_sig_denoise-P_remain_noise)/P_remain_noise, 10), 4))            
        except:            
            print('最大步长修正系数n:{}'.format(n[i-1]))
            print("剩余噪声：{}".format(P_remain_noise))
            print("有用信号噪声-剩余噪声：{}".format(P_sig_denoise-P_remain_noise))
            break
        SNR_dif.append(SNR_filter[i] - SNR_org)
        print("信噪比之差:{}".format(SNR_dif[i]))
        print("第{}次".format(i))
        
        i=i+1 
    
    print("原来信噪比：{}" .format(SNR_org))
    print("现在信噪比：{}" .format(SNR_filter))
    print("信噪比之差：{}" .format(SNR_dif))
    
    name=str(order)
    figure(1)
    plt.title(name)
    plt.plot(range(len(SNR_filter)),SNR_filter,'r.-',label='SNR_filter')
    plt.legend()
    
    figure(2)
    plt.title(name)
    plt.plot(range(len(SNR_dif)),SNR_dif,'r.-',label='SNR_dif')
    plt.legend()
    
    plt.show()     
########滤波器最佳阶数不变的情况下，改变步长n计算信噪比######### 

########查看剩余噪声的平均功率######### 
def Get_Power_yn(Xn, weight, order, weight_idex, sampling_rate):
    itr = len(Xn)
    Yn=array(np.zeros(itr))
    i=order
    while i<itr:
        x=Xn[i-order:i]
        x=np.matrix(x[::-1])#order阶滤波器抽头输入
        y=x*weight[:,weight_idex]
        Yn[i]=y[0, 0]
        
        i=i+1
    
    P_yn = GerPowerFre(Yn, sampling_rate)
#     P_yn = GerPowerTime(Yn)
    return P_yn

###########阶数一样时查看两种算法的收敛速度，也即剩余噪声的多少###########
'''
将data_array内的所有数据归一化
'''
def Normalization(data_array):
    length = len(data_array)
    
    data_array_nor = []
    data_max = max(data_array)
    data_min = min(data_array)
    
    i = 0
    while i<length:
        data_array_nor.append(round((data_array[i]-data_min)/(data_max-data_min), 5))
        
        i = i + 1
    
    return data_array_nor    
 
'''
剩余噪声的变化曲线，查看收敛速度
'''   
def Remain_Noise_Power_Show(filename1, filename2):
    sampling_rate = 1000
    force, Raw_1,Envelope_1,Raw_2,Envelope_2 = LoadDataSetAfterProcess(filename1)
    raw_1,envelope_1,raw_2,envelope_2 = LoadRawDataSetByChNum(filename2, CH=4)
        
    start_index = 0
    end_index = 59400 
    sig=Raw_1[start_index:end_index]
    noise=raw_1[0:end_index-start_index]
    
    P_noise = GerPowerFre(noise, sampling_rate)
#     P_noise = GerPowerTime(noise)
    
    order = 9
    yn_NLMS, sig_denoise_NLMS, weight_NLMS = NLMSFiltering(noise, sig, order, 0.0052)
    yn_LMS, sig_denoise_LMS, weight_LMS = LMSFiltering(noise, sig, order, 0.0000053)  
#     yn_NLMS, sig_denoise_NLMS, weight_NLMS = NLMSFiltering(noise, sig, order, 0.0001)
#     yn_LMS, sig_denoise_LMS, weight_LMS = LMSFiltering(noise, sig, order, 0.0001)  
    
    P_Rest_LMS = []
    P_Rest_NLMS = []   
    
    itr = len(noise)
    i = itr - 3000
    i_step = 5
    
    while i < itr:
        P_yn_LMS = Get_Power_yn(sig, weight_LMS, order, i, sampling_rate)
        P_remain_LMS = (P_noise - P_yn_LMS)**2  #剩余噪声的平方
        print("LMS剩余噪声：{}".format(P_remain_LMS))
        P_Rest_LMS.append(P_remain_LMS)
        
        P_yn_NLMS = Get_Power_yn(sig, weight_NLMS, order, i, sampling_rate)
        P_remain_NLMS = (P_noise - P_yn_NLMS)**2  #剩余噪声的平方
        print("NLMS剩余噪声：{}".format(P_remain_NLMS))
        P_Rest_NLMS.append(P_remain_NLMS)
        
        print("第{}次".format(i))
        i = i + i_step*order
        
    P_Rest_LMS_nor = Normalization(P_Rest_LMS) #误差平方归一化
    P_Rest_NLMS_nor = Normalization(P_Rest_NLMS) #误差平方归一化
    
    print("LMS算法标准差:{}".format(np.std(P_Rest_LMS_nor)))
    print("NLMS算法标准差:{}".format(np.std(P_Rest_NLMS_nor)))
    x_range = array(arange(0, 1000, round(1000/len(P_Rest_LMS_nor))))
    figure(1)
#     plt.plot(range(len(P_Rest_LMS_nor)),array(P_Rest_LMS_nor),'r-',label='P_Rest_LMS')
#     plt.plot(range(len(P_Rest_NLMS_nor)),array(P_Rest_NLMS_nor),'b-',label='P_Rest_NLMS')

    plt.plot(x_range,array(P_Rest_LMS_nor),'r-',label='P_Rest_LMS')
    plt.plot(x_range,array(P_Rest_NLMS_nor),'b-',label='P_Rest_NLMS')
    plt.xticks([0, 200, 400, 600, 800, 1000])
    plt.legend()
    
    plt.show()   
########查看剩余噪声的平均功率######### 

########查看权值的变化######### 
'''
查看随着迭代次数的增加权值的变化
'''   
def Weight_Change_Show(filename1, filename2):
    sampling_rate = 1000
    force, Raw_1,Envelope_1,Raw_2,Envelope_2 = LoadDataSetAfterProcess(filename1)
    raw_1,envelope_1,raw_2,envelope_2 = LoadRawDataSetByChNum(filename2, CH=4)
        
    start_index = 0
    end_index = 14300 
    sig=Raw_1[start_index:end_index]
    noise=raw_1[0:end_index-start_index]
    
    order = 9
    yn_NLMS, sig_denoise_NLMS, weight_NLMS = NLMSFiltering(noise, sig, order, 0.0052)
    yn_LMS, sig_denoise_LMS, weight_LMS = LMSFiltering(noise, sig, order, 0.0000053)  
    
    weight_LMS = array(weight_LMS)
    weight_NLMS = array(weight_NLMS)
    
    n = 2
    W_LMS = array(weight_LMS[n])
    W_NLMS = array(weight_NLMS[n])
    
    W_LMS = Normalization(W_LMS)
    W_NLMS = Normalization(W_NLMS)
                 
    figure(1)
    plt.plot(range(len(W_LMS)), W_LMS,'r-',label='W_LMS')
    plt.plot(range(len(W_NLMS)), W_NLMS,'b-',label='W_NLMS')
    plt.legend()
    
    plt.show()   
#######查看权值的变化###########

########查看预测噪声与源噪声的相关性######### 
'''
滤波器阶数不变时，预测噪声与原噪声功率谱之间的相关性分析
'''
def Get_Pearsonr_Fre(Xn, weight, order, weight_idex, sampling_rate):
    itr = len(Xn)
    Yn=array(np.zeros(itr))
    i=order
    while i<itr:
        x=Xn[i-order:i]
        x=np.matrix(x[::-1])#order阶滤波器抽头输入
        y=x*weight[:,weight_idex]
        Yn[i]=y[0, 0]
        
        i=i+1
    
    freqs, Xn_power_spectrum = signal.periodogram(Xn, fs=sampling_rate, window='boxcar', nfft=None,  scaling='spectrum')
    freqs, Yn_power_spectrum = signal.periodogram(Yn, fs=sampling_rate, window='boxcar', nfft=None,  scaling='spectrum')
    
    R, p_value = scipy.stats.pearsonr(Xn_power_spectrum, Yn_power_spectrum)#计算原噪声与估计噪声之间的相关性

    return abs(R)

def Get_CorrelationShow(filename1, filename2):
    sampling_rate = 1000
    force, Raw_1,Envelope_1,Raw_2,Envelope_2 = LoadDataSetAfterProcess(filename1)
    raw_1,envelope_1,raw_2,envelope_2 = LoadRawDataSetByChNum(filename2, CH=4)
        
    start_index = 0
    end_index = 59400 
    sig=Raw_1[start_index:end_index]
    noise=raw_1[0:end_index-start_index]
    
    order = 9
    yn_NLMS, sig_denoise_NLMS, weight_NLMS = NLMSFiltering(noise, sig, order, 0.0052)
    yn_LMS, sig_denoise_LMS, weight_LMS = LMSFiltering(noise, sig, order, 0.0000053)  
#     yn_NLMS, sig_denoise_NLMS, weight_NLMS = NLMSFiltering(noise, sig, order, 0.0001)
#     yn_LMS, sig_denoise_LMS, weight_LMS = LMSFiltering(noise, sig, order, 0.0001)  
    
    r_LMS = []
    r_NLMS = []   
    
    itr = len(noise)
    i = itr - 3000
    i_step = 5
    
    while i < itr:
        R_LMS = Get_Pearsonr_Fre(sig, weight_LMS, order, i, sampling_rate)
        print("LMS相关系数：{}".format(R_LMS))
        r_LMS.append(R_LMS)
        
        R_NLMS = Get_Pearsonr_Fre(sig, weight_NLMS, order, i, sampling_rate)
        print("NLMS相关系数：{}".format(R_NLMS))
        r_NLMS.append(R_NLMS)
        
        print("第{}次".format(i))
        i = i + i_step*order
        
    figure(1)
    plt.plot(range(len(r_LMS)),array(r_LMS),'r.-',label='r_LMS')
    plt.plot(range(len(r_NLMS)),array(r_NLMS),'b.-',label='r_NLMS')
    plt.legend()
    
    plt.show()    
########查看预测噪声与源噪声的相关性#########


