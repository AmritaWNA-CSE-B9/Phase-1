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

cv::cuda::GpuMat shiftFrame(cv::cuda::GpuMat image, int x, int y) {
  cv::Mat shift = (cv::Mat_<double>(3, 3) << 1, 0, x, 0, 1, y, 0, 0, 1);
  cv::Size_<int> mask_size(1280 * 2, 720);
  cv::cuda::warpPerspective(image, image, shift, mask_size * 2);
  return image;
}

int main(){
    cv::VideoCapture video2("./fingvideos/center.mp4");
    cv::VideoCapture video1("./fingvideos/right.mp4");
    
    cv::Mat Frame1;
    cv::Mat Frame2;
    
    cv::cuda::GpuMat Frame1_gpu;
    cv::cuda::GpuMat Frame2_gpu;

    // cv::cuda::GpuMat Tranfromed_frame1; 
    // cv::Mat Tranfromed_frame1_cpu;
    // cv::resize(Frame1, Frame1, cv::Size_<int> (640, 360));
    // cv::resize(Frame2, Frame2, cv::Size_<int> (640, 360));
    Homography h1estimator;

    bool isFrame1Active = video1.read(Frame1);
    bool isFrame2Active = video2.read(Frame2);
    int ROI_y, ROI_x; 
    
    ROI_y = Frame2.size().height;
    Frame2_gpu.upload(Frame2);
    Frame2_gpu = shiftFrame(Frame2_gpu, 0, 100);
    Frame2_gpu.download(Frame2);


    std::cout << "caliberation and estimating homography ... \n";
    cv::Mat h1 = h1estimator.inputFrame(Frame1, Frame2);
    // h1.convertTo(h1, CV_32F);
    ROI_x = Frame2.size().width;
    // cv::Mat homoMatrixRInvNorm = descaleMatrix * h1 * scale1Matrix;

    // start = clock();
    bool running = true;
    int frame_count = 0;
    try{

        while(running){
            // online phase start
            const auto renderTimeStart = std::chrono::high_resolution_clock::now();
            video1 >> Frame1;
            video2 >> Frame2;

            Frame1_gpu.upload(Frame1);
            Frame2_gpu.upload(Frame2);

            cv::cuda::GpuMat transformedFrameLeft;
            Frame2_gpu = shiftFrame(Frame2_gpu, 0 , 100);
            cv::cuda::warpPerspective(Frame1_gpu, transformedFrameLeft, h1, Frame2_gpu.size());
            cv::Mat transformedFrameLeft_cpu;
            Frame2_gpu.download(Frame2);
            transformedFrameLeft.download(transformedFrameLeft_cpu);

            // image addion 
            cv::Mat LeftFrame;
            cv::Mat RightFrame = Frame2;
            cv::subtract(transformedFrameLeft_cpu, Frame2, LeftFrame);
            cv::Mat result;
            cv::add(LeftFrame, RightFrame, result);
            const auto renderTimeEnd = std::chrono::high_resolution_clock::now();
            /* cv::add(transformedFrameLeft_cpu, Frame2, result); */

            // removing excess mask
            cv::Rect ROI(0, 100, Frame2_gpu.size().width / 2 + 400, Frame1_gpu.size().height);
            cv::imwrite("TransformedFrameLeft.png", transformedFrameLeft_cpu(ROI));
            result = result(ROI);
            try{
                cv::imshow("result", result);
                cv::imwrite("result.png", result);
                const auto renderTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(renderTimeEnd-renderTimeStart).count();

                std::cout << "Total Time taken for rendering (s): " << (float)(renderTimeMs / 1000.0) << "\n" ;
                std::cout << "Frames Per Second (FPS): " << (float) (1000.0 / (renderTimeMs)) << "\n" ;
            }catch(const std::exception &e){
                // cv::imshow("result", result);
                std::cout << "fds\n";
                break;
            }
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
