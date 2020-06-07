"""
Quantization module for generating the calibration tables will be used by 
quantized (INT8) models from FP32 models.with bucket split,[k, k, cin, cout]
cut into "cout" buckets.
This tool is based on Caffe Framework.
"""
from __future__ import division
from __future__ import print_function
import argparse
import numpy as np
import math, copy
import matplotlib.pyplot as plt
import sys,os
import caffe
import caffe.proto.caffe_pb2 as caffe_pb2
import time
import datetime
from google.protobuf import text_format
from scipy import stats
from PIL import Image


np.set_printoptions(threshold='nan')
np.set_printoptions(suppress=True)

def parse_args():
    parser = argparse.ArgumentParser(
        description='find the pretrained caffe models int8 quantize scale value')
    parser.add_argument('--proto', dest='proto',
                        help="path to deploy prototxt.", type=str)
    parser.add_argument('--model', dest='model',
                        help='path to pretrained weights', type=str)
    parser.add_argument('--mean', dest='mean',
                        help='value of mean', type=float, nargs=3)
    parser.add_argument('--norm', dest='norm',
                        help='value of normalize', type=float, nargs=1, default=1.0)                            
    parser.add_argument('--images', dest='images',
                        help='path to calibration images', type=str)
    parser.add_argument('--output', dest='output',
                        help='path to output calibration table file', type=str, default='calibration-dev.table')
    parser.add_argument('--group', dest='group',
                        help='enable the group scale', type=int, default=1)        
    parser.add_argument('--gpu', dest='gpu',
                        help='use gpu to forward', type=int, default=0)

    args = parser.parse_args()
    return args, parser


global args, parser
args, parser = parse_args()


# global params
QUANTIZE_NUM = 127 # 2^7=128，对应8位量化
QUANTIZE_WINOGRAND_NUM = 31 # 2^5=32，对应6位量化
STATISTIC = 1
INTERVAL_NUM = 2048 # FP32分bins

# 保存QuantizeLayer实例，要做量化的层
quantize_layer_lists = []


# 在推理前对网络进行预处理 
def network_prepare(net, mean, norm):
    
    print("Network initial……（Function：network_prepare）")

    # caffe.io读进的数据格式是RGB，[H,W,C]，0-1之间的浮点数
    # 需要转换到caffe内部数据格式：[C,H,W]，变换到0-255，减均值，再转换为BGR
    
    # 设定图片的shape格式为网络data层格式
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    # print(net.blobs['data'].data.shape)
    # [H,W,C] -> [C,H,W] 
    transformer.set_transpose('data', (2,0,1))
    # 减去均值
    transformer.set_mean('data', np.array(mean))
    # 将图片数据从[0,1]缩放到[0,255]
    transformer.set_raw_scale('data', 255)   
    # RGB -> BGR
    transformer.set_channel_swap('data', (2,1,0))   
    # normalize
    transformer.set_input_scale('data', norm)

    return transformer  


# 获取图片目录下的所有图片路径
def file_name(file_dir):
    """
    Find the all file path with the directory
    Args:
        file_dir: The source file directory
    Returns:
        files_path: all the file path into a list
    """
    files_path = []

    for root, dir, files in os.walk(file_dir):
        for name in files:
            file_path = root + "/" + name
            print(file_path)
            files_path.append(file_path)

    return files_path


class QuantizeLayer:
    def __init__(self, name, blob_name, group_num):
        '''
        name: layer.name
        blob_name: layer.bottom
        group_num: layer.convolution_param.group
        '''
        self.name = name
        self.blob_name = blob_name
        self.group_num = group_num
        self.weight_scale = np.zeros(group_num) # 1个group的weight对应1个weight_scale
        self.blob_max = 0.0 # 每层activation的最大值
        self.blob_distubution_interval = 0.0 # 分布间隔，即1个bin的宽度，右激活值-左激活值
        self.blob_distubution = np.zeros(INTERVAL_NUM) # 直方分布，FP32分2048bins，保存每个bins上激活值的频次
        self.blob_threshold = 0 # 最优映射边界对应的FP32 bins number
        self.blob_scale = 1.0 # 该层输出激活值对应的缩放因子
        self.group_zero = np.zeros(group_num) # 表明某一group的weight_scale是否为0，group_zero=1则weight_scale=0

    def quantize_weight(self, weight_data, flag):
        '''
        最大值映射
        weight_data: 单层权重数组
        flag: 标准conv-False，conv3*3s1-True
        '''
        # spilt the weight data by cout num
        # 根据group_num将每层的weight值分组
        blob_group_data = np.array_split(weight_data, self.group_num)

        for i, group_data in enumerate(blob_group_data):
            max_val = np.max(group_data)
            min_val = np.min(group_data)
            threshold = max(abs(max_val), abs(min_val))
            if threshold < 0.0001:
                self.weight_scale[i] = 0
                self.group_zero[i] = 1
            else: # FP32_A = scale_A * Q_A, 1/scale=127/|T|
                if(flag == True):
                    self.weight_scale[i] = QUANTIZE_WINOGRAND_NUM / threshold # 31/threshold
                else:
                    self.weight_scale[i] = QUANTIZE_NUM / threshold # 127/threshold
            print("层：%-20s number : %-5d T : %-10f scale : %-10f" % (self.name + "_param0", i, threshold, self.weight_scale[i]))

    def initial_blob_max(self, blob_data):
        '''
        获取层中blob(activation)最大值
        '''
        max_val = np.max(blob_data)
        min_val = np.min(blob_data)
        self.blob_max = max(self.blob_max, max(abs(max_val), abs(min_val)))

    def initial_blob_distubution_interval(self):
        '''
        计算bins宽度
        '''
        # 最大值 * 1/2048
        self.blob_distubution_interval = STATISTIC * self.blob_max / INTERVAL_NUM
        print("层：%-20s 最大值: %-10.8f bin_width : %-10.8f" % (self.name, self.blob_max, self.blob_distubution_interval))

    def initial_histograms(self, blob_data):
        # 计算直方图
        th = self.blob_max
        # hist_edge：直方图的横坐标，这里因为计算完了th，所以hist_edge是不变的
        # hist：直方图纵坐标，表示[bin_startValue,bin_endValue]之间有几个元素，频数
        hist, hist_edge = np.histogram(blob_data, bins=INTERVAL_NUM, range=(0, th))
        # 每进行一次前向推理，就累计一次，以此得到直方分布
        self.blob_distubution += hist

    def quantize_blob(self):
        # 获取某层激活值的直方分布  
        distribution = np.array(self.blob_distubution)
        # normalize_blob_distribution(distribution)
        # print('distribution_norm:')
        # for i in distribution:
        #     print(i)
        # 计算使KL散度值最小的FP32 bins number
        threshold_bin = threshold_distribution(distribution) 
        self.blob_threshold = threshold_bin
        # 得到阀值T
        threshold = (threshold_bin + 0.5) * self.blob_distubution_interval
        # 得到该层激活值的缩放因子
        self.blob_scale = QUANTIZE_NUM / threshold # 1/scale = 127/|T|
        print("层：%-20s bin_number : %-8d T : %-10f bin_width : %-10f scale : %-10f" % (self.name, threshold_bin, threshold, self.blob_distubution_interval, self.blob_scale))


def normalize_blob_distribution(distribution):
        distribution_len = distribution.size
        distribution_sum = sum(distribution)
        distribution /= distribution_sum    
    
def threshold_distribution(distribution, target_bin=128):
    """
    distribution: 最初将FP32分为2048bins的直方分布
    target_bin: INT8对应的直方分布的bins number
    返回: 使KL散度值最小的FP32 bins number
    """   
    # 某层激活值的直方分布，一个保存频次的数组
    distribution = distribution[1:]
    # 原直方分布分了多少个bins
    length = distribution.size
    # 将截断外区域(映射红色部分)值的频次全部求和
    threshold_sum = sum(distribution[target_bin:])
    # 保存不同分bins下产生分布的kl_divergence值，一共有(length - target_bin)个
    kl_divergence = np.zeros(length - target_bin)

    for threshold in range(target_bin, length): # threshold 128:2048，表示FP32要分成多少个bins
        # 生成截断区域
        sliced_nd_hist = copy.deepcopy(distribution[:threshold]) #复制0～threshold-1

        # 生成样本P(FP32)的概率分布
        p = sliced_nd_hist.copy()
        p[threshold-1] += threshold_sum # 截断区外值的频次加到截断样本P的最后一个值之上
        threshold_sum = threshold_sum - distribution[threshold] # 更新threshold_sum用于下次循环

        # is_nonzeros[k] indicates whether hist[k] is nonzero
        is_nonzeros = (p != 0).astype(np.int64)
        # FP32 bins压入INT8 bins后得到的新的分布，即样本Q(INT8)，长度为128
        quantized_bins = np.zeros(target_bin, dtype=np.int64)
        # 计算1个INT8 bin对应几个FP32 bin
        num_merged_bins = sliced_nd_hist.size // target_bin # 得到结果的整数部分 258//128=2……2(余数)
        
        # 将FP32 bins压入INT8 bins后得到样本Q(INT8)的分布
        for j in range(target_bin):
            start = j * num_merged_bins
            stop = start + num_merged_bins
            quantized_bins[j] = sliced_nd_hist[start:stop].sum()
        quantized_bins[-1] += sliced_nd_hist[target_bin * num_merged_bins:].sum() # 余数对应的bins(FP32)压入最后一个bin(INT8)
        
        # 将Q样本长度拓展到i bins，使得和原样本P具有相同长度
        q = np.zeros(sliced_nd_hist.size, dtype=np.float64)
        for j in range(target_bin):
            start = j * num_merged_bins
            if j == target_bin - 1:
                stop = -1
            else:
                stop = start + num_merged_bins
            norm = is_nonzeros[start:stop].sum()
            if norm != 0:
                q[start:stop] = float(quantized_bins[j]) / float(norm)
        q[p == 0] = 0
        # p = _smooth_distribution(p) # with some bugs, need to fix
        # q = _smooth_distribution(q)
        p[p == 0] = 0.0001
        q[q == 0] = 0.0001
        
        # 求P、Q的KL散度值
        kl_divergence[threshold - target_bin] = stats.entropy(p, q)

    min_kl_divergence = np.argmin(kl_divergence) # 返回最小值的索引
    threshold_value = min_kl_divergence + target_bin # 得到使KL散度值最小的FP32 bins number

    return threshold_value


def net_forward(net, image_path, transformer):
    """
    network inference and statistics the cost time
    Args:
        net: the instance of Caffe inference
        image_path: a image need to be inference
        transformer:
    Returns:
        none
    """ 
    # load image
    image = caffe.io.load_image(image_path)
    # image.resize(112, 112, 3)
    # transformer.preprocess the image
    net.blobs['data'].data[...] = transformer.preprocess('data',image)
    # net forward
    output = net.forward()


def weight_quantize(net, net_file, group_on):
    """
    量化权值
    net: the instance of Caffe inference
    net_file: deploy caffe prototxt
    """
    print("\nQuantize the kernel weight:")

    # parse the net param from deploy prototxt
    params = caffe_pb2.NetParameter()
    with open(net_file) as f:
        text_format.Merge(f.read(), params)

    # 对每个卷积层进行量化
    # 根据group_on参数初始化QuantizeLayer
    # 量化卷积层：对conv3x3s1进行6-bit卷积，对其他conv进行8-bit卷积
    # 保存量化后的层到量化列表
    for i, layer in enumerate(params.layer):
        if(layer.type == "Convolution" or layer.type == "ConvolutionDepthwise"):
            weight_blob = net.params[layer.name][0].data
            
            # 初始化
            if (group_on == 1): # 每层的每个group内的卷积核使用同一个量化参数，不同组内使用不同参数
                quanitze_layer = QuantizeLayer(layer.name, layer.bottom[0], layer.convolution_param.num_output) # ！！
            else:
                quanitze_layer = QuantizeLayer(layer.name, layer.bottom[0], 1)
            
            # 量化
            if(layer.type == "Convolution" and layer.convolution_param.kernel_size[0] == 3 and ((len(layer.convolution_param.stride) == 0) or layer.convolution_param.stride[0] == 1)):
                if(layer.convolution_param.group != layer.convolution_param.num_output):
                    quanitze_layer.quantize_weight(weight_blob, True) # !!不对conv3*3作6-bit量化
                else:
                    quanitze_layer.quantize_weight(weight_blob, False)
            else:
                quanitze_layer.quantize_weight(weight_blob, False)
            
            quantize_layer_lists.append(quanitze_layer)

    return None                


def activation_quantize(net, transformer, images_files):
    
    print("\n搜索激活最大值:")
    # 在校准数据集上进行FP32推理，获得activation范围，最后得到的是所有校准图片生成的activation范围(|最大值|)
    for i , image in enumerate(images_files):
        # FP32推理
        net_forward(net, image, transformer)
        # 找到每一层的最大activation值
        for layer in quantize_layer_lists:
            # 获取该层输出
            blob = net.blobs[layer.blob_name].data[0].flatten() # 把每层输出的激活特征图打平
            layer.initial_blob_max(blob)
        if i % 1000 == 0:
            print("%d/%d" % (i, len(images_files)))
    
    # 计算激活值分布间隔，即bins宽度
    for layer in quantize_layer_lists:
        layer.initial_blob_distubution_interval()   

    # 对于每一层，计算直方分布
    print("\n收集激活直方图:")
    for i, image in enumerate(images_files):
        net_forward(net, image, transformer)
        for layer in quantize_layer_lists:
            blob = net.blobs[layer.blob_name].data[0].flatten() # 把每层输出的激活特征图打平
            layer.initial_histograms(blob)
        if i % 1000 == 0:
            print("%d/%d" % (i, len(images_files)))          

    # 通过KL divergence计算激活值映射阈值、bins number、缩放因子
    for layer in quantize_layer_lists:
        layer.quantize_blob()  

    return None


def save_calibration_file(calibration_path):
    calibration_file = open(calibration_path, 'w') 
    # save temp
    save_temp = []
    # save weight scale
    for layer in quantize_layer_lists:
        save_string = layer.name + "_param_0"
        for i in range(layer.group_num):
            save_string = save_string + " " + str(layer.weight_scale[i])
        save_temp.append(save_string)

    # save bottom blob scales
    for layer in quantize_layer_lists:
        save_string = layer.name + " " + str(layer.blob_scale)
        save_temp.append(save_string)

    # save into txt file
    for data in save_temp:
        calibration_file.write(data + "\n")

    calibration_file.close()

    # save calibration logs
#    save_temp_log = []
#    calibration_file_log = open(calibration_path + ".log", 'w')
#    for layer in quantize_layer_lists:
#        save_string = layer.name + ": value range 0 - " + str(layer.blob_max) \
#                                 + ", interval " + str(layer.blob_distubution_interval) \
#                                 + ", interval num " + str(INTERVAL_NUM) \
#                                 + ", threshold num " + str(layer.blob_threshold) + "\n" \
#                                 + str(layer.blob_distubution.astype(dtype=np.int64))
#        save_temp_log.append(save_string)

#     save into txt file
#    for data in save_temp_log:
#        calibration_file_log.write(data + "\n")


def usage_info():
    """
    usage info
    """
    print("Input params is illegal...")
    print("try it again:\n python caffe-int8-scale-tools-dev.py -h")


def main():
    """
    main function
    """
    # 计算生成量化表时间
    time_start = datetime.datetime.now()

    print(args)
    if args.proto == None or args.model == None or args.mean == None or args.images == None:
        usage_info()
        return None
    
    # 读取参数
    net_file = args.proto # .prototxt
    caffe_model = args.model # .caffemodel
    mean = args.mean
    norm = 1.0
    if args.norm != 1.0:
        norm = args.norm[0]
    images_path = args.images # 校准数据集
    calibration_path = args.output # 量化表保存地址
    group_on = args.group
    
    # 默认使用CPU
    if args.gpu != 0:
        caffe.set_device(0)
        caffe.set_mode_gpu()
    
    # 初始化caffe模型
    net = caffe.Net(net_file,caffe_model,caffe.TEST)
    # 准备CNN网络
    transformer = network_prepare(net, mean, norm)
    # 获取校准数据集，获取目录下的所有图片路径
    images_files = file_name(images_path)
    # 量化卷积核的权值
    weight_quantize(net, net_file, group_on)
    # 量化激活值
    activation_quantize(net, transformer, images_files)
    # 保存量化表
    save_calibration_file(calibration_path)

    time_end = datetime.datetime.now()
    print("\n量化表计算时间： %s" % (time_end - time_start))

if __name__ == "__main__":
    main()
