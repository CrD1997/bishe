#pragma once
#ifndef MOBILEFACENET_H_
#define MOBILEFACENET_H_
#include <string>
#include "net.h"
#include "opencv2/opencv.hpp"


class Recognize {

public:
	Recognize(const std::string &model_path);
	~Recognize();
	void start(const cv::Mat& img, std::vector<float>&feature);

private:
	void RecogNet(ncnn::Mat& img_);
	ncnn::Net Recognet;
	std::vector<float> feature_out;

    void normalize(std::vector<float> &feature);
};

double calculSimilar(std::vector<float>& v1, std::vector<float>& v2);
float CalculateSimilarity(const std::vector<float>&feat1, const std::vector<float>& feat2);

#endif // !MOBILEFACENET_H_