#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/features2d.hpp>
#include <bits/stdc++.h>
class ParallelMandelbrot : public cv::ParallelLoopBody
{
public:
	ParallelMandelbrot(cv::Mat& TransformedImage, cv::Mat& InitialImage, std::vector<int> ti_x, std::vector<int> ti_y, std::vector<int> ii_tl_x, std::vector<int> ii_tl_y, std::vector<float> weight_tl, std::vector<float> weight_tr, std::vector<float> weight_bl, std::vector<float> weight_br)
		: TransformedImage(TransformedImage), InitialImage(InitialImage), ti_x(ti_x), ti_y(ti_y), ii_tl_x(ii_tl_x), ii_tl_y(ii_tl_y), weight_tl(weight_tl), weight_tr(weight_tr), weight_bl(weight_bl), weight_br(weight_br)
	{
	}
	virtual void operator ()(const cv::Range& range) const CV_OVERRIDE
	{
		for (int i = range.start; i < range.end; i++)
		{
			// https://stackoverflow.com/questions/7899108/opencv-get-pixel-channel-value-from-mat-image

			cv::Vec3b& TransformedImage_intensity = TransformedImage.at<cv::Vec3b>(ti_y[i], ti_x[i]);
			cv::Vec3b& InitialImage_intensity_tl = InitialImage.at<cv::Vec3b>(ii_tl_y[i], ii_tl_x[i]);
			cv::Vec3b& InitialImage_intensity_tr = InitialImage.at<cv::Vec3b>(ii_tl_y[i], ii_tl_x[i] + 1);
			cv::Vec3b& InitialImage_intensity_bl = InitialImage.at<cv::Vec3b>(ii_tl_y[i] + 1, ii_tl_x[i]);
			cv::Vec3b& InitialImage_intensity_br = InitialImage.at<cv::Vec3b>(ii_tl_y[i] + 1, ii_tl_x[i] + 1);

			for (int k = 0; k < InitialImage.channels(); k++)
			{
				TransformedImage_intensity.val[k] = ( weight_tl[i] * InitialImage_intensity_tl.val[k] ) +
													( weight_tr[i] * InitialImage_intensity_tr.val[k] ) +
													( weight_bl[i] * InitialImage_intensity_bl.val[k] ) +
													( weight_br[i] * InitialImage_intensity_br.val[k] );
			}
		}
	}
	ParallelMandelbrot& operator=(const ParallelMandelbrot&) {
		return *this;
	};
private:
	cv::Mat& TransformedImage;
	cv::Mat& InitialImage;
	std::vector<int> ti_x;
	std::vector<int> ti_y;
	std::vector<int> ii_tl_x;
	std::vector<int> ii_tl_y;
	std::vector<float> weight_tl;
	std::vector<float> weight_tr;
	std::vector<float> weight_bl;
	std::vector<float> weight_br;
};
class CylendricalWarper{
public:
	void Convert_xy(std::vector<int> ti_x, std::vector<int> ti_y, std::vector<float>& xt, std::vector<float>& yt, int center_x, int center_y, int f)
	{
		for (int i = 0; i < ti_y.size(); i++)
		{
			xt.push_back((f * tan((float)(ti_x[i] - center_x) / f)) + center_x);
			yt.push_back(((float)(ti_y[i] - center_y) / cos((float)(ti_x[i] - center_x) / f)) + center_y);
		}
	}

	void ProjectOntoCylinder(cv::Mat InitialImage, cv::Mat& TransformedImage, std::vector<int>& mask_x, std::vector<int>& mask_y)
	{
		int h = InitialImage.rows, w = InitialImage.cols;
		int center_x = w / 2, center_y = h / 2;
		int f = 1100;			// 1100 field; 1000 Sun; 1500 Rainier; 1050 Helens

		// Creating a blank transformed image.
		TransformedImage = cv::Mat::zeros(cv::Size(InitialImage.cols, InitialImage.rows), InitialImage.type());

		// Storing all coordinates of the transformed image in 2 arrays (x and y coordinates)
		std::vector<int> ti_x, ti_y;
		for (int j = 0; j < h; j++)
		{
			for (int i = 0; i < w; i++)
			{
				ti_x.push_back(i);
				ti_y.push_back(j);
			}
		}

		// Finding corresponding coordinates of the transformed image in the initial image
		std::vector<float> ii_x, ii_y;
		Convert_xy(ti_x, ti_y, ii_x, ii_y, center_x, center_y, f);

		// Rounding off the coordinate values to get exact pixel values (top-left corner).
		std::vector<int> ii_tl_x, ii_tl_y;
		for (int i = 0; i < ii_x.size(); i++)
		{
			ii_tl_x.push_back((int)ii_x[i]);
			ii_tl_y.push_back((int)ii_y[i]);
		}

		// Finding transformed image points whose corresponding
		// initial image points lies inside the initial image
		std::vector<bool> GoodIndices;
		for (int i = 0; i < ii_tl_x.size(); i++)
			GoodIndices.push_back((ii_tl_x[i] >= 0) && (ii_tl_x[i] <= (w - 2)) && (ii_tl_y[i] >= 0) && (ii_tl_y[i] <= (h - 2)));
		
		// Removing all the outside points from everywhere
		ti_x.erase(std::remove_if(ti_x.begin(), ti_x.end(), [&GoodIndices, &ti_x](auto const& i) { return !GoodIndices.at(&i - ti_x.data()); }), ti_x.end());
		ti_y.erase(std::remove_if(ti_y.begin(), ti_y.end(), [&GoodIndices, &ti_y](auto const& i) { return !GoodIndices.at(&i - ti_y.data()); }), ti_y.end());
		
		ii_x.erase(std::remove_if(ii_x.begin(), ii_x.end(), [&GoodIndices, &ii_x](auto const& i) { return !GoodIndices.at(&i - ii_x.data()); }), ii_x.end());
		ii_y.erase(std::remove_if(ii_y.begin(), ii_y.end(), [&GoodIndices, &ii_y](auto const& i) { return !GoodIndices.at(&i - ii_y.data()); }), ii_y.end());
		
		ii_tl_x.erase(std::remove_if(ii_tl_x.begin(), ii_tl_x.end(), [&GoodIndices, &ii_tl_x](auto const& i) { return !GoodIndices.at(&i - ii_tl_x.data()); }), ii_tl_x.end());
		ii_tl_y.erase(std::remove_if(ii_tl_y.begin(), ii_tl_y.end(), [&GoodIndices, &ii_tl_y](auto const& i) { return !GoodIndices.at(&i - ii_tl_y.data()); }), ii_tl_y.end());

		// Bilinear interpolation
		std::vector<float> dx(ii_x.size()), dy(ii_y.size());
		std::transform(ii_x.begin(), ii_x.end(), ii_tl_x.begin(), dx.begin(), std::minus<float>());
		std::transform(ii_y.begin(), ii_y.end(), ii_tl_y.begin(), dy.begin(), std::minus<float>());

		std::vector<float> weight_tl, weight_tr, weight_bl, weight_br;
		for (int i = 0; i < dx.size(); i++)
		{
			weight_tl.push_back((1.0 - dx[i]) * (1.0 - dy[i]));
			weight_tr.push_back((dx[i]) * (1.0 - dy[i]));
			weight_bl.push_back((1.0 - dx[i]) * (dy[i]));
			weight_br.push_back((dx[i]) * (dy[i]));
		}

		// Used this website for code
		// https://docs.opencv.org/master/d7/dff/tutorial_how_to_use_OpenCV_parallel_for_.html
		ParallelMandelbrot parallelMandelbrot(TransformedImage, InitialImage, ti_x, ti_y, ii_tl_x, ii_tl_y, weight_tl, weight_tr, weight_bl, weight_br);
		cv::parallel_for_(cv::Range(0, weight_tl.size()), parallelMandelbrot);

		// Getting x coorinate to remove black region from rightand left in the transformed image
		int min_x = *min_element(ti_x.begin(), ti_x.end());

		// Cropping out the black region from both sides(using symmetricity)
		TransformedImage(cv::Rect(min_x, 0, TransformedImage.cols - min_x*2, TransformedImage.rows)).copyTo(TransformedImage);

		// Setting return values
		// mask_x = ti_x - min_x
		std::vector<int> min_x_v(ti_x.size(), min_x);
		std::transform(ti_x.begin(), ti_x.end(), min_x_v.begin(), std::back_inserter(mask_x),
			[](int a, int b) { return (a - b); });
		//mask_y = ti_y
		mask_y = ti_y;
	}
};