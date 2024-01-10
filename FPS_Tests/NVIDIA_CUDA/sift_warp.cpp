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
#include "./headers/homographer.h"

cv::cuda::GpuMat shiftFrame(cv::cuda::GpuMat image, int x, int y, cv::Size_<int> maskSize) {
  cv::Mat shift = (cv::Mat_<double>(3, 3) << 1, 0, x, 0, 1, y, 0, 0, 1);
  cv::cuda::warpPerspective(image, image, shift, maskSize);
  return image;
}

int main(){
    cv::VideoCapture video1("./fingvideos/left.mp4");
    cv::VideoCapture video2("./fingvideos/center.mp4");
    cv::VideoCapture video3("./fingvideos/right.mp4");

    cv::Mat Frame1;
    cv::Mat Frame2;
    cv::Mat Frame3; 

    cv::cuda::GpuMat Frame1_gpu;
    cv::cuda::GpuMat Frame2_gpu;
    cv::cuda::GpuMat Frame3_gpu;

    // cv::cuda::GpuMat Tranfromed_frame1; 
    // cv::Mat Tranfromed_frame1_cpu;
    // cv::resize(Frame1, Frame1, cv::Size_<int> (640, 360));
    // cv::resize(Frame2, Frame2, cv::Size_<int> (640, 360));

    bool isFrame1Active = video1.read(Frame1);
    bool isFrame2Active = video2.read(Frame2);
    bool ifFrame3Active = video3.read(Frame3);

    // cv::imwrite("./output_images/inputs/Frame1.png", Frame1);
    // cv::imwrite("./output_images/inputs/Frame2.png", Frame2);
    // cv::imwrite("./output_images/inputs/Frame3.png", Frame3);

    int ROI_y, ROI_x; 
    
    cv::Size_<int> maskSize = (Frame1.size() + Frame2.size() + Frame3.size());
    ROI_y = Frame2.size().height;
    Frame2_gpu.upload(Frame2);
    Frame2_gpu = shiftFrame(Frame2_gpu, Frame2.size().width, 0, cv::Size_<int>(maskSize.width, Frame1.size().height));
    Frame2_gpu.download(Frame2);

    Homography hestimator;
    std::cout << "caliberation and estimating homography ... \n";
    cv::Mat h1 = hestimator.inputFrame(Frame1, Frame2);
    cv::Mat h2 = hestimator.inputFrame(Frame3, Frame2);
    ROI_x = Frame2.size().width;

	int frameCount = 0, windowSize = 1000;
	std::vector<float> frameTimeList;
    try{

        float fps_sma = 0.0f, frameTime_sma = 0.0f;
        float fps_wma = 0.0f, frameTime_wma = 0.0f, totalWeight = 0.0f;
        std::ofstream myfile;
        myfile.open("./FPS_CUDA_FINAL.csv");
        myfile << "Frame," << "SMA,"  << "WMA" << std::endl;
        bool running = true;
        while(running){
            // online phase start
            frameTime_sma = 0.0f;
            frameTime_wma = 0.0f;
            totalWeight = 0.0f;
            auto t1 = std::chrono::high_resolution_clock::now(); // start time

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

            Frame1_gpu.upload(Frame1);
            Frame2_gpu.upload(Frame2);
            Frame3_gpu.upload(Frame3);
            
            cv::cuda::GpuMat transformedFrameLeft;
            cv::cuda::GpuMat tranformedFrameRight;

            Frame2_gpu = shiftFrame(Frame2_gpu, Frame2.size().width, 0, cv::Size_<int>(maskSize.width, Frame1.size().height));

            cv::cuda::warpPerspective(Frame1_gpu, transformedFrameLeft, h1, Frame2_gpu.size());
            cv::cuda::warpPerspective(Frame3_gpu, tranformedFrameRight, h2, Frame2_gpu.size());
            cv::Mat transformedFrameLeft_cpu;
            cv::Mat tranformedFrameRight_cpu;

            Frame2_gpu.download(Frame2);
            transformedFrameLeft.download(transformedFrameLeft_cpu);
            tranformedFrameRight.download(tranformedFrameRight_cpu);
            // cv::imwrite("TransformedLeft.png", transformedFrameLeft_cpu);
            // image addion 
            cv::Mat LeftFrame;
            cv::Mat RightFrame;
            cv::Mat ReferenceFrame = Frame2;

            cv::subtract(transformedFrameLeft_cpu, ReferenceFrame, LeftFrame);
            cv::subtract(tranformedFrameRight_cpu, ReferenceFrame, RightFrame);
            cv::Mat result;
            cv::add(LeftFrame, ReferenceFrame, result);
            cv::add(result, RightFrame, result);

            auto t2 = std::chrono::high_resolution_clock::now(); // end time
            cv::imshow("result", result);


            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / (float)1000;
            double fps = 1000 / (double)duration; // Optional
            frameCount++; //Optional
            frameTimeList.push_back((float)duration);
            if(frameTimeList.size() > windowSize){frameTimeList.erase(frameTimeList.begin());} 
            
            for(int k = 0;k < frameTimeList.size();k++)
            {
              frameTime_sma += frameTimeList[k]; // There are better ways of doing this but for simplicity I am going with this.
              float w = (k + 1) / (float)frameTimeList.size();
              totalWeight += w;
              frameTime_wma += frameTimeList[k] * w; // There are better ways of doing this but for simplicity I am going with this.
            }
            
            // for(int k = 0;k < frameTimeList.size();k++)
            // {
            //   float w = (k + 1) / (float)frameTimeList.size();
            //   totalWeight += w;
            //   frameTime_wma += frameTimeList[k] * w; // There are better ways of doing this but for simplicity I am going with this.
            // }

            fps_wma = 1000 / (float)(frameTime_wma / totalWeight);
            fps_sma = frameTimeList.size() / (frameTime_sma / (float)1000);
            std::cout << "FrameCount: " << frameCount << "\n";
            std::cout << "SMA: " << fps_sma << std::endl; 
            std::cout << "WMA: " << fps_wma << std::endl;
            myfile << frameCount << "," << fps_sma << "," << fps_wma << std::endl;
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
