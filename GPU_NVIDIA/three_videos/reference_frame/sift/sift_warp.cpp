#include<bits/stdc++.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/stitching/detail/warpers.hpp>
#include <time.h>
#include<opencv4/opencv2/opencv.hpp>
#include<opencv4/opencv2/features2d.hpp>

// global declaration
cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

// genrating keypoints
std::vector<cv::KeyPoint> genrateKeyPoints(cv::Mat frame, std::vector<cv::KeyPoint> keypoints){
    sift->detect(frame, keypoints);
    return keypoints;
}

// genrating descriptors
cv::Mat genrateDescriptors(cv::Mat frame, std::vector<cv::KeyPoint> keypoints){
    cv::Mat descriptor;
    sift->compute(frame, keypoints, descriptor);
    return descriptor;
}

// shifting the image by x , y

cv::cuda::GpuMat shiftFrame(cv::cuda::GpuMat image, int x, int y) {
  cv::Mat shift = (cv::Mat_<double>(3, 3) << 1, 0, x, 0, 1, y, 0, 0, 1);
  cv::cuda::warpPerspective(image, image, shift, image.size() * 2);
  return image;
}

// perfroming lowes ratio test and cleaning bad keypoints
std::vector<cv::DMatch> LowesRatioClean(std::vector<std::vector<cv::DMatch>> rawmatches){
    std::vector<cv::DMatch> goodMatches;

    
    double ratio = 0.8;
    for(auto match : rawmatches){
        // std::cout << match[0].distance << " " << match[1].distance << "\n";
        if(match[0].distance < ratio * match[1].distance) {
            goodMatches.push_back(match[0]);
        }
    }
    return goodMatches;
}

int main(){
    cv::VideoCapture video1("./markerVid/left.mp4");
    cv::VideoCapture video2("./markerVid/center.mp4");
    cv::VideoCapture video3("./markerVid/right.mp4");

    cv::Mat Frame1;
    cv::Mat Frame2;
    cv::Mat Frame3;

    cv::cuda::GpuMat Frame1_gpu;
    cv::cuda::GpuMat Frame2_gpu;
    cv::cuda::GpuMat Frame3_gpu;

    cv::Mat Tranformation21;
    cv::Mat Tranformation23;

    bool isHomographyComputed21 = false;
    bool isHomographyComputed23 = false;
    bool isShifted = false;

    clock_t start, end;
    double time_s;
    int frame_count = 0;

    clock_t start, end;
    double time_s;
    int frame_count = 0;

    // start clock

    // start = clock();
    while(video1.isOpened() && video2.isOpened() && video3.isOpened()){

        bool isFrame1Active = video1.read(Frame1);
        bool isFrame2Active = video2.read(Frame2);
        bool isFrame3Active = video3.read(Frame3);
        
        // upload frames to GPU
        Frame1_gpu.upload(Frame1);
        Frame2_gpu.upload(Frame2);
        Frame3_gpu.upload(Frame3);

        // converting frame to grayscale
        cv::cvtColor(Frame1, Frame1, cv::COLOR_BGR2GRAY);
        cv::cvtColor(Frame2, Frame2, cv::COLOR_BGR2GRAY);
        cv::cvtColor(Frame3, Frame3, cv::COLOR_BGR2GRAY);

        // shifing the reference image to the center

        Frame2_gpu = shiftFrame(Frame2_gpu, 300, 300);
        cv::imwrite("output_images/tranformations/tansformedframe2.png", Frame2);

        // keypoints and descriptors
        std::vector<cv::KeyPoint> kp_vid1 , kp_vid2, kp_vid3;
        cv::Mat des_vid1, des_vid2, des_vid3;

        kp_vid1 = genrateKeyPoints(Frame1, kp_vid1);
        kp_vid2 = genrateKeyPoints(Frame2, kp_vid2);
        kp_vid3 = genrateKeyPoints(Frame3, kp_vid3);

        cv::Mat drawkp_vid1;
        cv::Mat drawkp_vid2;
        cv::Mat drawkp_vid3;

        // cv::drawKeypoints(Frame1, kp_vid1, drawkp_vid1);
        // cv::drawKeypoints(Frame2, kp_vid2, drawkp_vid2);
        // cv::drawKeypoints(Frame3, kp_vid3, drawkp_vid3);

        // cv::imwrite("output_images/keypoints/vid1_kp.png", drawkp_vid1);
        // cv::imwrite("output_images/keypoints/vid2_kp.png", drawkp_vid2);
        // cv::imwrite("output_images/keypoints/vid3_kp.png", drawkp_vid3);

        des_vid1 = genrateDescriptors(Frame1, kp_vid1);
        des_vid2 = genrateDescriptors(Frame2, kp_vid2);
        des_vid3 = genrateDescriptors(Frame3, kp_vid3);

        // converting it to opencv descriptor format sift
        des_vid1.convertTo(des_vid1, CV_32F);
        des_vid2.convertTo(des_vid2, CV_32F);
        des_vid3.convertTo(des_vid3, CV_32F);

        cv::cuda::GpuMat des1_gpu;
        cv::cuda::GpuMat des2_gpu;
        cv::cuda::GpuMat des3_gpu;

        des1_gpu.upload(des_vid1);
        des2_gpu.upload(des_vid2);
        des3_gpu.upload(des_vid3);

        // performing matching
        cv::BFMatcher matcher(cv::NORM_L2);
        std::vector<std::vector<cv::DMatch>> rawMatches21;
        std::vector<std::vector<cv::DMatch>> rawMatches23;

        matcher.knnMatch(des_vid2, des_vid1, rawMatches21, 2);
        matcher.knnMatch(des_vid2, des_vid3, rawMatches23, 2);

        cv::Mat rawMatches21_display;
        cv::Mat rawMatches23_display;

        cv::drawMatches(Frame1, kp_vid1, Frame2, kp_vid2, rawMatches21,rawMatches21_display);
        cv::drawMatches(Frame3, kp_vid3, Frame2, kp_vid2, rawMatches23, rawMatches23_display);

        cv::imwrite("output_images/matching/rawMatch21.png", rawMatches21_display);
        cv::imwrite("output_images/matching/rawMatch23.png", rawMatches23_display);

    }
 
}