#pragma once
#ifndef MOBILEFACENET_H_
#define MOBILEFACENET_H_
#include <string>
#include "opencv2/opencv.hpp"

#include <opencv2/dnn.hpp>
using namespace cv::dnn;


class Recognize {

public:
	Recognize(const std::string &model_path);
	void start(const cv::Mat& img, std::vector<float>&feature);

private:
	void RecogNet(cv::Mat& img_); ////
	Net Recognet; ////
	std::vector<float> feature_out;

    void normalize(std::vector<float> &feature);
};

double calculSimilar(std::vector<float>& v1, std::vector<float>& v2);
float CalculateSimilarity(const std::vector<float>&feat1, const std::vector<float>& feat2);


#endif // !MOBILEFACENET_H_