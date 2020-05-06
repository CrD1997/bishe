#include "mobilefacenet.h"
#if NCNN_VULKAN
#include "gpu.h"
#endif // NCNN_VULKAN

//加载模型
Recognize::Recognize(const std::string &model_path) {

//	std::string param_files = model_path + "/insightface_mfnet_mxnet-nobn-int8.param";
//	std::string bin_files = model_path + "/insightface_mfnet_mxnet-nobn-int8.bin";
    std::string param_files = model_path + "/mobilefacenet_h-int8.param";
    std::string bin_files = model_path + "/mobilefacenet_h-int8.bin";
    std::cout<< "Model : " << param_files << std::endl;

    Recognet.opt.num_threads=1;
//    Recognet.opt.use_winograd_convolution=false;
//	Recognet.opt.use_int8_inference=false;
//	Recognet.opt.use_sgemm_convolution=false;

	Recognet.load_param(param_files.c_str());
	Recognet.load_model(bin_files.c_str());

#if NCNN_VULKAN
    Recognet.opt.use_vulkan_compute = true;
#endif // NCNN_VULKAN
}

Recognize::~Recognize() {

	Recognet.clear();
}

//执行Extractor，得到128维的特征
void Recognize::RecogNet(ncnn::Mat& img_) {

//    img_ = (img_.data - 128)/128;

	ncnn::Extractor ex = Recognet.create_extractor();
	//设置单线程，多线程反而慢
//	ex.set_num_threads(1);
	ex.set_light_mode(true);

    ex.input("data", img_);

    ncnn::Mat temp_out1;
    ncnn::Mat temp_out2;

//    //计算fp32模型ConvDW+BN时间
//    ex.extract("res_3_block0_conv_sep_relu", temp_out1);
//    clock_t start_time1 = clock();
//    ex.input("res_3_block0_conv_sep_relu", temp_out1);
//    ex.extract("res_3_block0_conv_dw_batchnorm", temp_out2);
//    clock_t finish_time1 = clock();
//    double total_time1 = (double)(finish_time1 - start_time1);
//    std::cout << "ConvDW time : " << total_time1 / 1000<< "ms" << std::endl;

    //计算int8模型ConvDW+BN时间
//    ex.extract("res_3_block0_conv_sep_relu", temp_out1);
//    clock_t start_time1 = clock();
//    ex.input("res_3_block0_conv_sep_relu", temp_out1);
//    ex.extract("res_3_block0_conv_dw_batchnorm", temp_out2);
//    clock_t finish_time1 = clock();
//    double total_time1 = (double)(finish_time1 - start_time1);
//    std::cout << "ConvDW time : " << total_time1 / 1000<< "ms" << std::endl;

	ncnn::Mat out;
    clock_t start_time = clock();
	ex.extract("fc1", out);
    clock_t finish_time = clock();

    //计算人脸识别时间
    double total_time = (double)(finish_time - start_time);
    std::cout << "Inference time : " << total_time / 1000<< "ms" << std::endl;

	feature_out.resize(128);
	for (int j = 0; j < 128; j++){
		feature_out[j] = out[j];
	}

//    normalize(feature_out);
}

void Recognize::normalize(std::vector<float> &feature)
{
    float sum = 0.f;
    for(auto it = feature.begin(); it != feature.end(); ++it)
        sum += (*it) * (*it);

    sum = sqrt(sum);
    for(auto it = feature.begin(); it != feature.end(); ++it)
        *it /= sum;
}

//计算图片人脸特征值
void Recognize::start(const cv::Mat& img, std::vector<float>&feature) {

	ncnn::Mat ncnn_img = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR2RGB, img.cols, img.rows, 112, 112);
	//(每个通道-127.5)/127.5，提高准确率大大地！
    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {0.007843f, 0.007843f, 0.007843f};
    ncnn_img.substract_mean_normalize(mean_vals, norm_vals);

	RecogNet(ncnn_img);
	feature = feature_out;
}

double calculSimilar(std::vector<float>& v1, std::vector<float>& v2){

	assert(v1.size() == v2.size());
//	//计算余弦距离，这里可以优化一下
//	double ret = 0.0, mod1 = 0.0, mod2 = 0.0;
//	for (std::vector<double>::size_type i = 0; i != v1.size(); ++i){
//		ret += v1[i] * v2[i];
//		mod1 += v1[i] * v1[i];
//		mod2 += v2[i] * v2[i];
//	}
//	return (ret / sqrt(mod1) / sqrt(mod2) ) ;

    //  计算欧式距离
    double dist = 0.0;
    for(int i=0; i<v1.size(); i++){
        dist += (v1[i]-v2[i])*(v1[i]-v2[i]);
    }
    return sqrt(dist);
}

float CalculateSimilarity(const std::vector<float>&feat1, const std::vector<float>& feat2) {
    if (feat1.size() != feat2.size()) {
        std::cout << "feature size not match." << std::endl;
        return 10003;
    }
    float inner_product = 0.0f;
    float feat_norm1 = 0.0f;
    float feat_norm2 = 0.0f;

    for(int i = 0; i < feat1.size(); ++i) {
        inner_product += feat1[i] * feat2[i];
        feat_norm1 += feat1[i] * feat1[i];
        feat_norm2 += feat2[i] * feat2[i];
    }
    return inner_product / (sqrt(feat_norm1) * sqrt(feat_norm2));
}
