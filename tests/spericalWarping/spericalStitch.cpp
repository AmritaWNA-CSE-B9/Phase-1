#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/stitching/detail/warpers.hpp>
#include <opencv2/stitching/warpers.hpp>
#include <opencv2/stitching.hpp>
void sphericalWarp(const cv::Mat& input, cv::Mat& output, double fov) {
    int width = input.cols;
    int height = input.rows;

    double radius = width / (2 * M_PI); // Radius of the sphere

    // Center of the input image
    double cx = width / 2.0;
    double cy = height / 2.0;

    // Create the output image
    output = cv::Mat::zeros(height, width, input.type());

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // Convert pixel coordinates to spherical coordinates
            double theta = (x - cx) / radius;
            double phi = (y - cy) / radius;

            // Convert spherical coordinates to Cartesian coordinates
            double x_sphere = radius * sin(theta);
            double y_sphere = radius * phi;
            double z_sphere = radius * cos(theta);

            // Map Cartesian coordinates to input image
            int x_input = static_cast<int>(x_sphere + cx);
            int y_input = static_cast<int>(y_sphere + cy);

            // Check if the mapped coordinates are within the input image bounds
            if (x_input >= 0 && x_input < width && y_input >= 0 && y_input < height) {
                // Perform bilinear interpolation to get pixel value
                output.at<cv::Vec3b>(y, x) = input.at<cv::Vec3b>(y_input, x_input);
            }
        }
    }
}
int main() {
    // Read the input images
    cv::Mat img1 = cv::imread("./images/q11.jpg");
    cv::Mat img2 = cv::imread("./images/q22.jpg");

    // Apply spherical warping
    cv::Mat img1_spherical, img2_spherical;
    sphericalWarp(img1, img1_spherical, 78.0);
    sphericalWarp(img2, img2_spherical, 78.0);

    // Perform feature detection and matching
    cv::Ptr<cv::Feature2D> detector = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;

    detector->detectAndCompute(img1_spherical, cv::noArray(), keypoints1, descriptors1);
    detector->detectAndCompute(img2_spherical, cv::noArray(), keypoints2, descriptors2);

    cv::BFMatcher matcher;
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    // Filter matches based on a threshold
    double max_dist = 0;
    double min_dist = 100;

    for (int i = 0; i < descriptors1.rows; i++) {
        double dist = matches[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    std::vector<cv::DMatch> good_matches;
    for (int i = 0; i < descriptors1.rows; i++) {
        if (matches[i].distance < 2 * min_dist) {
            good_matches.push_back(matches[i]);
        }
    }

    // Prepare images and keypoints for stitching
    std::vector<cv::Mat> images = {img1_spherical, img2_spherical};
    std::vector<std::vector<cv::KeyPoint>> keypoints = {keypoints1, keypoints2};
    std::vector<cv::Mat> descriptors = {descriptors1, descriptors2};

    // Create a Stitcher object
    cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create();

    // Configure stitcher parameters
    stitcher->setWarper(new cv::detail::SphericalWarper(1.0));
    stitcher->setFeaturesMatcher(new cv::detail::BestOf2NearestMatcher(false, 0.3));
    stitcher->setBundleAdjuster(new cv::detail::BundleAdjusterRay());
    
    // Create a Mat to hold the stitched result
    cv::Mat result;

    // Use the Stitcher to stitch the images
    cv::Stitcher::Status status = stitcher->stitch(images, result);

    // Check if stitching was successful
    if (status == cv::Stitcher::Status::OK) {
        // Display the stitched result
        cv::imshow("Stitched Image", result);
        cv::waitKey(0);
    } else {
        std::cerr << "Error: Stitching failed.\n";
        return -1;
    }

    return 0;
}