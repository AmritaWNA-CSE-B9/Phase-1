#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/features2d.hpp>
#include <bits/stdc++.h>

// transformation imports
#include <glm/ext/matrix_transform.hpp>
#include <glm/fwd.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>
#include <string>

class Homography{
    public:
        void print(cv::Mat img){
            int l = img.rows;
            int b = img.cols;

            for(int i = 0; i < l; i++){
                for(int j = 0; j < l; j++){
                    std::cout << img.at<double>(i, j) << " ";
                }
                std::cout << "\n";
            }
        }

        std::vector<cv::DMatch> LowesRatioClean(std::vector<std::vector<cv::DMatch>> rawmatches){
            std::vector<cv::DMatch> goodMatches;

            double ratio = 0.8;
            for(auto match : rawmatches){
                if(match[0].distance < ratio * match[1].distance) {
                    goodMatches.push_back(match[0]);
                }
            }
            return goodMatches;
        }

        cv::Mat inputFrame(cv::Mat img1, cv::Mat img2){        

            cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
            cv::Mat des1, des2;
            std::vector<cv::KeyPoint> kp1, kp2;

            sift->detect(img1, kp1);
            sift->compute(img1,kp1, des1);
            sift->detect(img2, kp2);
            sift->compute(img2,kp2, des2);

            cv::Mat keypoints1;
            cv::Mat keypoints2;

            cv::drawKeypoints(img1, kp1, keypoints1);
            cv::drawKeypoints(img2, kp2, keypoints2);

            // cv::imwrite("./output_images/keypoints/keypoint1.png", keypoints1);
            // cv::imwrite("./output_images/keypoints/keypoint2.png", keypoints2);

            cv::BFMatcher matcher(cv::NORM_L2);
            std::vector<std::vector<cv::DMatch>> knnmatches;

            int k = 2;
            matcher.knnMatch(des1, des2, knnmatches, k);

            cv::Mat rawmatches_img; 
            cv::drawMatches(img1, kp1, img2, kp2, knnmatches, rawmatches_img);
            

            double ratio = 0.7;
            std::vector<cv::DMatch> goodmatches = LowesRatioClean(knnmatches);
            std::vector<cv::Point2f> goodkp1, goodkp2;

            for(auto match : goodmatches){
                goodkp1.push_back(kp1[match.queryIdx].pt);
                goodkp2.push_back(kp2[match.trainIdx].pt);
            }
            
            cv::Mat Homography = cv::findHomography(goodkp1, goodkp2, cv::RANSAC);

            print(Homography);
            std::cout << "Homography done\n";
            std::cout << Homography << "\n";

            return Homography;
        }
};
