#include <bits/stdc++.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/stitching/detail/warpers.hpp>
#include <time.h>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/features2d.hpp>
#include "../headers/homographer.h"


cv::Mat shiftFrame(cv::Mat image, int x, int y, cv::Size_<int> maskSize) {
  cv::Mat shift = (cv::Mat_<double>(3, 3) << 1, 0, x, 0, 1, y, 0, 0, 1);
//   cv::Size_<int> mask_size(1280 * 2, 720);
  cv::warpPerspective(image, image, shift, maskSize);
  return image;
}

int main(){
    cv::VideoCapture video1("../fingvideos/left.mp4");
    cv::VideoCapture video2("../fingvideos/center.mp4");
    cv::VideoCapture video3("../fingvideos/right.mp4");
     
    cv::Mat Frame1;
    cv::Mat Frame2;
    cv::Mat Frame3; 
    

    cv::cuda::GpuMat Frame1_gpu;
    cv::cuda::GpuMat Frame2_gpu;
    cv::cuda::GpuMat Frame3_gpu;
    
    bool isFrame1Active = video1.read(Frame1);
    bool isFrame2Active = video2.read(Frame2);
    bool ifFrame3Active = video3.read(Frame3);

    cv::Size_<int> maskSize = (Frame1.size() + Frame2.size() + Frame3.size());
    Frame2 = shiftFrame(Frame2, Frame2.size().width, 0, cv::Size_<int>(maskSize.width, Frame1.size().height));

    Homography hestimator;
    std::cout << "caliberation and estimating homography ... \n";
    cv::Mat h1 = hestimator.inputFrame(Frame1, Frame2);
    cv::Mat h2 = hestimator.inputFrame(Frame3, Frame2);
    // h1.convertTo(h1, CV_32F);
    // cv::Mat homoMatrixRInvNorm = descaleMatrix * h1 * scale1Matrix;

    // start = clock();
    bool running = true;

    int frameCount = 0, windowSize = 1000;
    std::vector<float> frameTimeList;
    bool rendering = true;
    float fps_wma = 0.0f, frameTime = 0.0f, totalWeight = 0.0f;

    try{
        std::ofstream myfile;
        myfile.open("./FPS_CPU_WMA.csv");
        myfile << "Frame," << "WMA" << std::endl;

        while(running){
            frameTime = 0.0f;
            totalWeight = 0.0f;

            auto t1 = std::chrono::high_resolution_clock::now();
            try{ 
              video1 >> Frame1;
              video2 >> Frame2;
              video3 >> Frame3;
              if (frameCount <= 5000) {
                if (Frame1.empty() || Frame2.empty() || Frame3.empty()) {
                    std::cout << "Loop back\n";
                    video1.set(cv::CAP_PROP_POS_FRAMES, 0);
                    video2.set(cv::CAP_PROP_POS_FRAMES, 0);
                    video3.set(cv::CAP_PROP_POS_FRAMES, 0);
                    continue;
                }
              }
              std::cout << frameCount << "\n";
            }catch (const std::exception &e){
              std::cout << "res\n";
            }

            cv::Mat transformedFrameLeft;
            cv::Mat transformedFrameRight;

            Frame2 = shiftFrame(Frame2, Frame2.size().width, 0, cv::Size_<int>(maskSize.width, Frame1.size().height));
            cv::warpPerspective(Frame1, transformedFrameLeft, h1, Frame2.size());
            cv::warpPerspective(Frame3, transformedFrameRight, h2, Frame2.size());
            // image addion 
            cv::Mat LeftFrame;
            cv::Mat RightFrame;
            cv::Mat ReferenceFrame = Frame2;

            cv::subtract(transformedFrameLeft, ReferenceFrame, LeftFrame);
            cv::subtract(transformedFrameRight, ReferenceFrame, RightFrame);
            cv::Mat result;
            cv::add(LeftFrame, ReferenceFrame, result);
            cv::add(result, RightFrame, result);
            auto t2 = std::chrono::high_resolution_clock::now();

            cv::imshow("result", result);
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / (float)1000; double fps = 1000 / (double)duration; // Optional
            frameCount++; //Optional
            frameTimeList.push_back((float)duration);
            if(frameTimeList.size() > windowSize){frameTimeList.erase(frameTimeList.begin());} // Can also use a better method.
            
            for(int k = 0;k < frameTimeList.size();k++)
            {
              float w = (k + 1) / (float)frameTimeList.size();
              totalWeight += w;
              frameTime += frameTimeList[k] * w; // There are better ways of doing this but for simplicity I am going with this.
            }

            fps_wma = 1000 / (float)(frameTime / totalWeight);
            myfile << frameCount << "," << fps_wma << std::endl;
            std::cout << "WMA: " << fps_wma << std::endl;
            
            // removing excess mask
            int key = cv::waitKey(10);
            if (key == 'q' || key == 27) { // 'q' key or Esc key (27) to exit
                running = false;
            }
        }
    }catch (const std::exception &e){
    }

    video1.release();
    video2.release();
    cv::destroyAllWindows();
}
