#include "mobilefacenet.h"


//加载模型
Recognize::Recognize(const std::string &model_path) {

    std::string model_file = "mobilefacenet.caffemodel"; //模型结构文件
    std::string model_text = "mobilefacenet.prototxt";  //模型数据
    Recognet = readNetFromCaffe(model_text, model_file);
}

//执行Extractor，得到128维的特征
void Recognize::RecogNet(cv::Mat& img_) {

    //设置单线程
    cv::setNumThreads(1);
    Recognet.setPreferableTarget(DNN_TARGET_CPU);
    Recognet.setInput(img_, "data");

	feature_out.resize(128);
    clock_t start_time = clock();
    feature_out = Recognet.forward("fc1_scale");
    clock_t finish_time = clock();

    //计算人脸识别时间
    double total_time = (double)(finish_time - start_time) / CLOCKS_PER_SEC;
    std::cout << "Inference time : " << total_time*1000 << "ms" << std::endl;

    normalize(feature_out);
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

    cv::Mat inputBlob = blobFromImage(img, 1.0, cv::Size(112, 112), cv::Scalar(), false);

	RecogNet(inputBlob);
	feature = feature_out;
}

double calculSimilar(std::vector<float>& v1, std::vector<float>& v2){

	assert(v1.size() == v2.size());
	//计算余弦距离，这里可以优化一下
//	double ret = 0.0, mod1 = 0.0, mod2 = 0.0;
//	for (std::vector<double>::size_type i = 0; i != v1.size(); ++i){
//		ret += v1[i] * v2[i];
//		mod1 += v1[i] * v1[i];
//		mod2 += v2[i] * v2[i];
//	}
//	return (ret / sqrt(mod1) / sqrt(mod2) ) ;

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

    for(int i = 0; i < 128; ++i) {
        inner_product += feat1[i] * feat2[i];
        feat_norm1 += feat1[i] * feat1[i];
        feat_norm2 += feat2[i] * feat2[i];
    }
    return inner_product / sqrt(feat_norm1) / sqrt(feat_norm2);
}
