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
  /* cv::Size_<int> mask_size(1280 * 2, 720); */
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
    // std::cout << "hello\n";
    // h1.convertTo(h1, CV_32F);
    ROI_x = Frame2.size().width;
    // cv::Mat homoMatrixRInvNorm = descaleMatrix * h1 * scale1Matrix;

    // start = clock();
    bool running = true;
    int frame_count = 0;
    float frame_net = 0;
    float exec_net = 0;
    float warp_net = 0;

    try{

        while(running){
            // online phase start
            const auto renderTimeStart = std::chrono::high_resolution_clock::now();
            video1 >> Frame1;
            video2 >> Frame2;
            video3 >> Frame3;

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
            const auto renderTimeEnd = std::chrono::high_resolution_clock::now();

            try{
                cv::imshow("result", result);
                std::cout << result.size() << "\n";
                const auto renderTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(renderTimeEnd-renderTimeStart).count();
                std::cout << "Total Time taken for rendering (s): " << (float)(renderTimeMs / 1000.0) << "\n" ;
                std::cout << "Frames Per Second (FPS): " << (float) (1000.0 / (renderTimeMs)) << "\n" ;
                frame_net += (float) (1000.0 / (renderTimeMs));
                exec_net += (float)(renderTimeMs / 1000.0);
                frame_count += 1;
            }catch(const std::exception &e){
                break;
            }
            std::cout << "Average FPS : " << (float) frame_net / frame_count << "\n";
            std::cout << "Average Exec : " << (float) exec_net / frame_count << "\n";
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
