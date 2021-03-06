// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <stdio.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>

#include "platform.h"
#include "net.h"

#if NCNN_VULKAN
#include "gpu.h"
#endif // NCNN_VULKAN

struct FaceObject
{
    cv::Rect_<float> rect; //人脸矩形框，左上(x,y)，宽、高
    cv::Point2f landmark[5]; //五个基准点
    float prob; //是人脸的可能性
};

//理想状态下，landmark在112*112图片中的位置
float points_dst[5][2] = {
        { 30.2946f + 8.0f, 51.6963f },
        { 65.5318f + 8.0f, 51.5014f },
        { 48.0252f + 8.0f, 71.7366f },
        { 33.5493f + 8.0f, 92.3655f },
        { 62.7299f + 8.0f, 92.2041f }
};

cv::Mat MeanAxis0(const cv::Mat & src) {
    int num = src.rows;
    int dim = src.cols;

    // x1 y1
    // x2 y2

    cv::Mat output(1, dim, CV_32FC1);
    for (int i = 0; i < dim; i++) {
        float sum = 0;
        for (int j = 0; j < num; j++) {
            sum += src.at<float>(j, i);
        }
        output.at<float>(0, i) = sum / num;
    }

    return output;
}

cv::Mat ElementwiseMinus(const cv::Mat & A, const cv::Mat & B) {
    cv::Mat output(A.rows, A.cols, A.type());
    assert(B.cols == A.cols);
    if (B.cols == A.cols) {
        for (int i = 0; i < A.rows; i++) {
            for (int j = 0; j < B.cols; j++) {
                output.at<float>(i, j) = A.at<float>(i, j) - B.at<float>(0, j);
            }
        }
    }

    return output;
}

cv::Mat VarAxis0(const cv::Mat & src) {
    cv::Mat temp_ = ElementwiseMinus(src, MeanAxis0(src));
    cv::multiply(temp_, temp_, temp_);
    return MeanAxis0(temp_);
}

int MatrixRank(cv::Mat M) {
    cv::Mat w, u, vt;
    cv::SVD::compute(M, w, u, vt);
    cv::Mat1b nonZeroSingularValues = w > 0.0001;
    int rank = countNonZero(nonZeroSingularValues);
    return rank;
}

cv::Mat SimilarTransform(const cv::Mat & src, const cv::Mat & dst) {
    int num = src.rows;
    int dim = src.cols;
    cv::Mat src_mean = MeanAxis0(src);
    cv::Mat dst_mean = MeanAxis0(dst);
    cv::Mat src_demean = ElementwiseMinus(src, src_mean);
    cv::Mat dst_demean = ElementwiseMinus(dst, dst_mean);
    cv::Mat A = (dst_demean.t() * src_demean) / static_cast<float>(num);
    cv::Mat d(dim, 1, CV_32F);
    d.setTo(1.0f);
    if (cv::determinant(A) < 0) {
        d.at<float>(dim - 1, 0) = -1;

    }
    cv::Mat T = cv::Mat::eye(dim + 1, dim + 1, CV_32F);
    cv::Mat U, S, V;
    cv::SVD::compute(A, S, U, V);

    // the SVD function in opencv differ from scipy .

    int rank = MatrixRank(A);
    if (rank == 0) {
        assert(rank == 0);

    }
    else if (rank == dim - 1) {
        if (cv::determinant(U) * cv::determinant(V) > 0) {
            T.rowRange(0, dim).colRange(0, dim) = U * V;
        }
        else {
            int s = d.at<float>(dim - 1, 0) = -1;
            d.at<float>(dim - 1, 0) = -1;

            T.rowRange(0, dim).colRange(0, dim) = U * V;
            cv::Mat diag_ = cv::Mat::diag(d);
            cv::Mat twp = diag_ * V; //np.dot(np.diag(d), V.T)
            cv::Mat B = cv::Mat::zeros(3, 3, CV_8UC1);
            cv::Mat C = B.diag(0);
            T.rowRange(0, dim).colRange(0, dim) = U * twp;
            d.at<float>(dim - 1, 0) = s;
        }
    }
    else {
        cv::Mat diag_ = cv::Mat::diag(d);
        cv::Mat twp = diag_ * V.t(); //np.dot(np.diag(d), V.T)
        cv::Mat res = U * twp; // U
        T.rowRange(0, dim).colRange(0, dim) = -U.t()* twp;
    }
    cv::Mat var_ = VarAxis0(src_demean);
    float val = cv::sum(var_).val[0];
    cv::Mat res;
    cv::multiply(d, S, res);
    float scale = 1.0 / val * cv::sum(res).val[0];
    T.rowRange(0, dim).colRange(0, dim) = -T.rowRange(0, dim).colRange(0, dim).t();
    cv::Mat  temp1 = T.rowRange(0, dim).colRange(0, dim); // T[:dim, :dim]
    cv::Mat  temp2 = src_mean.t();
    cv::Mat  temp3 = temp1 * temp2;
    cv::Mat temp4 = scale * temp3;
    T.rowRange(0, dim).colRange(dim, dim + 1) = -(temp4 - dst_mean.t());
    T.rowRange(0, dim).colRange(0, dim) *= scale;
    return T;
}

int align_face(const cv::Mat& bgr, const FaceObject& faceobject, cv::Mat * face_aligned) {

    std::cout << "start align face..." << std::endl;

    float points_src[5][2] = {
            {faceobject.landmark[0].x, faceobject.landmark[0].y},
            {faceobject.landmark[1].x, faceobject.landmark[1].y},
            {faceobject.landmark[2].x, faceobject.landmark[2].y},
            {faceobject.landmark[3].x, faceobject.landmark[3].y},
            {faceobject.landmark[4].x, faceobject.landmark[4].y}
    };

    cv::Mat src_mat(5, 2, CV_32FC1, points_src);
    cv::Mat dst_mat(5, 2, CV_32FC1, points_dst);

    cv::Mat transform = SimilarTransform(src_mat, dst_mat);

    face_aligned->create(112, 112, CV_32FC3);

    cv::Mat transfer_mat = transform(cv::Rect(0, 0, 3, 2));
    cv::warpAffine(bgr.clone(), *face_aligned, transfer_mat, cv::Size(112, 112), 1, 0, 0);

    std::cout << "end align face." << std::endl;
    return 0;
}

void get_alignedfaces(const cv::Mat& bgr, const std::vector<FaceObject>& faceobjects){

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < faceobjects.size(); i++) {
        const FaceObject &obj = faceobjects[i];

        cv::Mat face_aligned;
        align_face(image, obj, &face_aligned);
        char filename[30];
        sprintf(filename,"aligned/fp32/face_%d.jpg",i);
        cv::imwrite(filename, face_aligned);
    }

//    cv::waitKey(0);
}

void print_mat(const ncnn::Mat& m)
{
    for (int q=0; q<m.c; q++)
    {
        const float* ptr = m.channel(q);
        for (int y=0; y<m.h; y++)
        {
            for (int x=0; x<m.w; x++)
            {
                printf("%f ", ptr[x]);
            }
            ptr += m.w;
            printf("\n");
        }
        printf("------------------------\n");
    }
    printf("|||||||||||||||||||||||||\n");
}

static inline float intersection_area(const FaceObject& a, const FaceObject& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<FaceObject>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

//将faceObjects按score大小从高到低排序
static void qsort_descent_inplace(std::vector<FaceObject>& faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<FaceObject>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const FaceObject& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const FaceObject& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
//             float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

// copy from src/layer/proposal.cpp
//base_size：原区域大小，ratios：变换比例，scales：对宽和高放大的倍数
static ncnn::Mat generate_anchors(int base_size, const ncnn::Mat& ratios, const ncnn::Mat& scales)
{
    int num_ratio = ratios.w;
    int num_scale = scales.w;

    ncnn::Mat anchors;
    anchors.create(4, num_ratio * num_scale);

    const float cx = base_size * 0.5f;
    const float cy = base_size * 0.5f;

    for (int i = 0; i < num_ratio; i++)
    {
        float ar = ratios[i];

        int r_w = round(base_size / sqrt(ar));
        int r_h = round(r_w * ar);//round(base_size * sqrt(ar));

        for (int j = 0; j < num_scale; j++)
        {
            float scale = scales[j];

            float rs_w = r_w * scale;
            float rs_h = r_h * scale;

            float* anchor = anchors.row(i * num_scale + j);

            anchor[0] = cx - rs_w * 0.5f;
            anchor[1] = cy - rs_h * 0.5f;
            anchor[2] = cx + rs_w * 0.5f;
            anchor[3] = cy + rs_h * 0.5f;
        }
    }

    return anchors;
}

static void generate_proposals(const ncnn::Mat& anchors, int feat_stride, const ncnn::Mat& score_blob, const ncnn::Mat& bbox_blob, const ncnn::Mat& landmark_blob, float prob_threshold, std::vector<FaceObject>& faceobjects)
{
    int w = score_blob.w;
    int h = score_blob.h;

    // generate face proposal from bbox deltas and shifted anchors
    const int num_anchors = anchors.h;

    for (int q=0; q<num_anchors; q++)
    {
        const float* anchor = anchors.row(q);

        const ncnn::Mat score = score_blob.channel(q + num_anchors);
        const ncnn::Mat bbox = bbox_blob.channel_range(q * 4, 4);
        const ncnn::Mat landmark = landmark_blob.channel_range(q * 10, 10);

        // shifted anchor
        float anchor_y = anchor[1];

        float anchor_w = anchor[2] - anchor[0];
        float anchor_h = anchor[3] - anchor[1];

        for (int i=0; i<h; i++)
        {
            float anchor_x = anchor[0];

            for (int j=0; j<w; j++)
            {
                int index = i * w + j;

                float prob = score[index];

                if (prob >= prob_threshold)
                {
                    // apply center size
                    float dx = bbox.channel(0)[index];
                    float dy = bbox.channel(1)[index];
                    float dw = bbox.channel(2)[index];
                    float dh = bbox.channel(3)[index];

                    //获取anchor中心点坐标
                    float cx = anchor_x + anchor_w * 0.5f;
                    float cy = anchor_y + anchor_h * 0.5f;

                    float pb_cx = cx + anchor_w * dx;
                    float pb_cy = cy + anchor_h * dy;

                    float pb_w = anchor_w * exp(dw);
                    float pb_h = anchor_h * exp(dh);

                    float x0 = pb_cx - pb_w * 0.5f;
                    float y0 = pb_cy - pb_h * 0.5f;
                    float x1 = pb_cx + pb_w * 0.5f;
                    float y1 = pb_cy + pb_h * 0.5f;

                    FaceObject obj;
                    obj.rect.x = x0;
                    obj.rect.y = y0;
                    obj.rect.width = x1 - x0 + 1;
                    obj.rect.height = y1 - y0 + 1;
                    obj.landmark[0].x = cx + (anchor_w + 1) * landmark.channel(0)[index];
                    obj.landmark[0].y = cy + (anchor_h + 1) * landmark.channel(1)[index];
                    obj.landmark[1].x = cx + (anchor_w + 1) * landmark.channel(2)[index];
                    obj.landmark[1].y = cy + (anchor_h + 1) * landmark.channel(3)[index];
                    obj.landmark[2].x = cx + (anchor_w + 1) * landmark.channel(4)[index];
                    obj.landmark[2].y = cy + (anchor_h + 1) * landmark.channel(5)[index];
                    obj.landmark[3].x = cx + (anchor_w + 1) * landmark.channel(6)[index];
                    obj.landmark[3].y = cy + (anchor_h + 1) * landmark.channel(7)[index];
                    obj.landmark[4].x = cx + (anchor_w + 1) * landmark.channel(8)[index];
                    obj.landmark[4].y = cy + (anchor_h + 1) * landmark.channel(9)[index];
                    obj.prob = prob;

                    faceobjects.push_back(obj);
                }

                anchor_x += feat_stride;
            }

            anchor_y += feat_stride;
        }
    }

}

//参数m：图片矩阵，faceobjects：最终得到的人脸对象
static int detect_retinaface(ncnn::Net &retinaface, const cv::Mat& bgr, std::vector<FaceObject>& faceobjects)
{

    const float prob_threshold = 0.6f; //0.8，分类概率的阈值，超过这个阈值的检测被判定为正例
    // NMS: non maximum suppression，非极大值抑制，作用：去掉detection任务重复的检测框
    //非极大值抑制中的IOU阈值，即在nms中与正例的IOU超过这个阈值的检测将被舍弃
    const float nms_threshold = 0.4f; //0.4

    //实例化Mat，第一个参数是data，后面是w、h
    int img_w = bgr.cols;
    int img_h = bgr.rows;
    ncnn::Mat in = ncnn::Mat::from_pixels(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h);

    //实例化Extractor
    ncnn::Extractor ex = retinaface.create_extractor();
    ex.set_num_threads(4);
    //设置输入，这里“data”是deploy中的数据层名字
    clock_t start_time = clock();
    ex.input("data", in);

    std::vector<FaceObject> faceproposals;
    // stride 32
    {
        ncnn::Mat score_blob, bbox_blob, landmark_blob;
        //ex.extract(设置层索引，得到该层输出)
        ex.extract("face_rpn_cls_prob_reshape_stride32", score_blob);
        ex.extract("face_rpn_bbox_pred_stride32", bbox_blob);
        ex.extract("face_rpn_landmark_pred_stride32", landmark_blob);

        const int base_size = 16;
        const int feat_stride = 32;//感受野大小，越来越小
        ncnn::Mat ratios(1);
        ratios[0] = 1.f;
        ncnn::Mat scales(2);
        scales[0] = 32.f;
        scales[1] = 16.f;
        //获得2个anchors的坐标，[x_left_up, y_left_up, x_right_down, y_right_down]
        ncnn::Mat anchors = generate_anchors(base_size, ratios, scales);

        std::vector<FaceObject> faceobjects32;
        generate_proposals(anchors, feat_stride, score_blob, bbox_blob, landmark_blob, prob_threshold, faceobjects32);

        faceproposals.insert(faceproposals.end(), faceobjects32.begin(), faceobjects32.end());
    }

    // stride 16
    {
        ncnn::Mat score_blob, bbox_blob, landmark_blob;
        ex.extract("face_rpn_cls_prob_reshape_stride16", score_blob);
        ex.extract("face_rpn_bbox_pred_stride16", bbox_blob);
        ex.extract("face_rpn_landmark_pred_stride16", landmark_blob);

        const int base_size = 16;
        const int feat_stride = 16;
        ncnn::Mat ratios(1);
        ratios[0] = 1.f;
        ncnn::Mat scales(2);
        scales[0] = 8.f;
        scales[1] = 4.f;
        ncnn::Mat anchors = generate_anchors(base_size, ratios, scales);

        std::vector<FaceObject> faceobjects16;
        generate_proposals(anchors, feat_stride, score_blob, bbox_blob, landmark_blob, prob_threshold, faceobjects16);

        faceproposals.insert(faceproposals.end(), faceobjects16.begin(), faceobjects16.end());
    }

    // stride 8
    {
        ncnn::Mat score_blob, bbox_blob, landmark_blob;
        ex.extract("face_rpn_cls_prob_reshape_stride8", score_blob);
        ex.extract("face_rpn_bbox_pred_stride8", bbox_blob);
        ex.extract("face_rpn_landmark_pred_stride8", landmark_blob);

        const int base_size = 16;
        const int feat_stride = 8;
        ncnn::Mat ratios(1);
        ratios[0] = 1.f;
        ncnn::Mat scales(2);
        scales[0] = 2.f;
        scales[1] = 1.f;
        ncnn::Mat anchors = generate_anchors(base_size, ratios, scales);

        std::vector<FaceObject> faceobjects8;
        generate_proposals(anchors, feat_stride, score_blob, bbox_blob, landmark_blob, prob_threshold, faceobjects8);

        faceproposals.insert(faceproposals.end(), faceobjects8.begin(), faceobjects8.end());
    }
    clock_t finish_time = clock();

    //计算人脸识别时间
    double total_time = (double)(finish_time - start_time) / CLOCKS_PER_SEC;
    std::cout << "Total time : " << total_time*1000 << "ms" << std::endl;

    // sort all proposals by score from highest to lowest
    // 将faceObjects按score大小从高到低排序
    qsort_descent_inplace(faceproposals);

    // apply nms with nms_threshold
    // 去掉多余的选择框
    std::vector<int> picked;
    nms_sorted_bboxes(faceproposals, picked, nms_threshold);
    int face_count = picked.size();

    // faceobjects：最终得到的人脸对象
    faceobjects.resize(face_count);
    for (int i = 0; i < face_count; i++)
    {
        faceobjects[i] = faceproposals[ picked[i] ];

        // clip to image size
        float x0 = faceobjects[i].rect.x;
        float y0 = faceobjects[i].rect.y;
        float x1 = x0 + faceobjects[i].rect.width;
        float y1 = y0 + faceobjects[i].rect.height;

        x0 = std::max(std::min(x0, (float)img_w - 1), 0.f);
        y0 = std::max(std::min(y0, (float)img_h - 1), 0.f);
        x1 = std::max(std::min(x1, (float)img_w - 1), 0.f);
        y1 = std::max(std::min(y1, (float)img_h - 1), 0.f);

        faceobjects[i].rect.x = x0;
        faceobjects[i].rect.y = y0;
        faceobjects[i].rect.width = x1 - x0;
        faceobjects[i].rect.height = y1 - y0;
    }

    return 0;
}

static void draw_faceobjects(const cv::Mat& bgr, const std::vector<FaceObject>& faceobjects)
{
    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < faceobjects.size(); i++)
    {
        const FaceObject& obj = faceobjects[i];

        fprintf(stderr, "%.5f at %.2f %.2f %.2f x %.2f\n",
                obj.prob, obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, cv::Scalar(0, 255, 0));

        cv::circle(image, obj.landmark[0], 2, cv::Scalar(0, 255, 255), -1);
        cv::circle(image, obj.landmark[1], 2, cv::Scalar(0, 255, 255), -1);
        cv::circle(image, obj.landmark[2], 2, cv::Scalar(0, 255, 255), -1);
        cv::circle(image, obj.landmark[3], 2, cv::Scalar(0, 255, 255), -1);
        cv::circle(image, obj.landmark[4], 2, cv::Scalar(0, 255, 255), -1);

        char text[256];
        sprintf(text, "%.1f%%", obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y),
                                      cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    cv::imshow("image", image);
    cv::imwrite("test1.jpg", image);
    cv::waitKey(0);
}

int detect_on_fddb(){

    std::string fddbpath = "./FDDB-folds/";
    std::string imagespath = "./originalPics/";
    std::string file_path;
    std::string img_path;

    std::string out_fold_path = "./out-folds/";
    std::string out_file;

    ncnn::Net retinaface;
    retinaface.load_param("models/retinaface-mnet0.25-fp32-int8-n1.0.param");
    retinaface.load_model("models/retinaface-mnet0.25-int8-n1.0.bin");

    for(int i=1; i<=10; i++){
        //读取FDDB中filepath文件
        std::stringstream ss;
        ss << i;
        std::string str = ss.str();

        std::cout << i << std::endl;
        if(i!=10){
            file_path = fddbpath+"FDDB-fold-0"+str+".txt";
            out_file = out_fold_path + "fold-0" + str +"-out.txt";
        }
        else {
            file_path = fddbpath+"FDDB-fold-"+str+".txt";
            out_file = out_fold_path + "fold-" + str +"-out.txt";
        }

        std::ifstream filein(file_path, std::ios::in);
//        if(filein.fail()){
//            std::cout << "fddb-fold file path 文件打开失败" << std::endl;
//            exit(0);
//        }

        std::cout << out_file << std::endl;
        std::ofstream fileout(out_file);

        while(getline(filein, img_path)){
            //读取图片
            cv::Mat m = cv::imread(imagespath + img_path + ".jpg", 1);

            //识别
            std::vector<FaceObject> faceobjects;
            detect_retinaface(retinaface, m, faceobjects);

            fileout.open(out_file, std::ios::out|std::ios::app);
            if(!fileout.is_open()){
                std::cout << "out file 文件打开失败" << std::endl;
                exit(0);
            }

            fileout << img_path << std::endl;
            fileout << faceobjects.size() << std::endl;
            for (size_t i = 0; i < faceobjects.size(); i++) {
                const FaceObject &obj = faceobjects[i];
                fileout << obj.rect.x << " " << obj.rect.y << " " << obj.rect.width << " " << obj.rect.height << " " << obj.prob << std::endl;
            }
            fileout.close();
        }
        filein.close();
    }

    return 0;
}

int test_detect(){

    const char* imagepath = "images/img_1194.jpg";
    std::string model = "int8";

    //实例化ncnn::Net
    ncnn::Net retinaface;
    //加载二进制文件.param和.bin
    if(model == "int8"){
        std::cout << "Using INT8 model......" << std::endl;
        retinaface.load_param("models/retinaface-mnet0.25-int8-n1.0.param");
        retinaface.load_model("models/retinaface-mnet0.25-int8-n1.0.bin");
    }

    if(model == "fp32"){
        std::cout << "Using FP32 model......" << std::endl;
        retinaface.load_param("models/retinaface-mnet0.25.param");
        retinaface.load_model("models/retinaface-mnet0.25.bin");
    }

    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty()){
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    std::vector<FaceObject> faceobjects;

    for(int i=0; i<1; i++){
        detect_retinaface(retinaface, m, faceobjects);
    }

    get_alignedfaces(m, faceobjects);
    draw_faceobjects(m, faceobjects);

    return 0;
}

int main(int argc, char** argv){

//    if (argc != 3){
//        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
//        fprintf(stderr, "Usage: %s [model type]\n", argv[0]);
//        return -1;
//    }
//    const char* imagepath = argv[1];
//    std::string model = argv[2];

    test_detect();
//    detect_on_fddb();

    return 0;
}
