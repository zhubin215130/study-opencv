#include <iostream>
#include <opencv2\opencv.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include"opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <opencv2/legacy/legacy.hpp>

using namespace cv;
using namespace std;

/*
// used by 2
void createAlphaMat(Mat &mat)
{
	for (int i = 0; i < mat.rows; ++i) {
		for (int j = 0; j < mat.cols; ++j) {
			Vec4b& rgba = mat.at<Vec4b>(i, j);
			rgba[0] = saturate_cast<uchar>((float(mat.cols - j)) / ((float)mat.cols) *UCHAR_MAX); //R
			rgba[1] = saturate_cast<uchar>((float(mat.rows - i)) / ((float)mat.rows) *UCHAR_MAX); //G
			rgba[2] = saturate_cast<uchar>(0.5 * (rgba[0] + rgba[1])); //B
			rgba[3] = UCHAR_MAX; //A
		}
	}
}

// used by 5
Mat dst;
Mat trackbar_img;
int threshval;
static void on_trackbar(int, void*)
{
	Mat bw = threshval < 128 ? (trackbar_img < threshval) : (trackbar_img > threshval);
	//Mat bw = trackbar_img.clone();
	//定义点和向量
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	//查找轮廓
	findContours(bw, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	//初始化dst
	dst = Mat::zeros(trackbar_img.size(), trackbar_img.type());
	//开始处理
	if (!contours.empty() && !hierarchy.empty())
	{
		//遍历所有顶层轮廓，随机生成颜色值绘制给各连接组成部分
		int idx = 0;
		for (; idx >= 0; idx = hierarchy[idx][0])
		{
			Scalar color((rand() & 255), (rand() & 255), (rand() & 255));
			//绘制填充轮廓
			drawContours(dst, contours, idx, color, CV_FILLED, 8, hierarchy);
		}
	}
	//cout << "dst:" << dst.size().width << "," << dst.size().height << endl;
	//显示窗口
	imshow("contours", dst);
}

// used by 6
int g_nContrastValue; //对比度值
int g_nBrightValue;  //亮度值
Mat g_srcImage, g_dstImage;
static void ContrastAndBright(int, void *)
{

	//创建窗口
	namedWindow("origin", 1);

	//三个for循环，执行运算 g_dstImage(i,j) =a*g_srcImage(i,j) + b
	for (int y = 0; y < g_srcImage.rows; y++)
	{
		for (int x = 0; x < g_srcImage.cols; x++)
		{
			for (int c = 0; c < 3; c++)
			{
				g_dstImage.at<Vec3b>(y, x)[c] = saturate_cast<uchar>((g_nContrastValue*0.01)*(g_srcImage.at<Vec3b>(y, x)[c]) + g_nBrightValue);
			}
		}
	}

	//显示图像
	imshow("origin", g_srcImage);
	imshow("effect", g_dstImage);
}


// used by 8
int g_nMedianBlurValue = 3;  //中值滤波参数值
int g_nBilateralFilterValue = 10;  //双边滤波参数值
Mat g_dstImage4, g_dstImage5;
//-----------------------------【on_MedianBlur( )函数】------------------------------------
//            描述：中值滤波操作的回调函数
//-----------------------------------------------------------------------------------------------
static void on_MedianBlur(int, void *)
{
	int64 app_start_time = getTickCount();
	medianBlur(g_srcImage, g_dstImage4, g_nMedianBlurValue * 2 + 1);
	imshow("medianBlur", g_dstImage4);
	cout<<"MedianBlur Finished, total time: " << ((getTickCount() - app_start_time) / getTickFrequency()) << " sec"<<endl;
}


//-----------------------------【on_BilateralFilter( )函数】------------------------------------
//            描述：双边滤波操作的回调函数
//-----------------------------------------------------------------------------------------------
static void on_BilateralFilter(int, void *)
{
	//if(!g_srcImage.data){
	int64 app_start_time = getTickCount();
	bilateralFilter(g_srcImage, g_dstImage5, g_nBilateralFilterValue, g_nBilateralFilterValue * 2, g_nBilateralFilterValue / 2);
	imshow("bilateralFilter", g_dstImage5);
	cout << "bilateralFilter Finished, total time: " << ((getTickCount() - app_start_time) / getTickFrequency()) << " sec"<<endl;
	//}
}


// used by 13
Mat g_srcImage, g_srcImage1, g_grayImage;
int thresh = 30; //当前阈值
int max_thresh = 175; //最大阈值
void on_CornerHarris(int, void*)
{

	Mat dstImage;//目标图
	Mat normImage;//归一化后的图
	Mat scaledImage;//线性变换后的八位无符号整型的图

	dstImage = Mat::zeros(g_srcImage.size(), CV_32FC1);
	g_srcImage1 = g_srcImage.clone();
	
	int64 app_start_time = getTickCount();
	cornerHarris(g_grayImage, dstImage, 2, 3, 0.04, BORDER_DEFAULT);	
	cout << "cornerHarris Finished, total time: " << ((getTickCount() - app_start_time) / getTickFrequency()) << " sec" << endl;
	// 10ms on haswell
	normalize(dstImage, normImage, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(normImage, scaledImage);//将归一化后的图线性变换成8位无符号整型 
		
	for (int j = 0; j < normImage.rows; j++)
	{
		for (int i = 0; i < normImage.cols; i++)
		{
			if ((int)normImage.at<float>(j, i) > thresh+70)
			{
				circle(g_srcImage1, Point(i, j), 5, Scalar(10, 10, 255), 2, 8, 0);
				circle(scaledImage, Point(i, j), 5, Scalar(0, 10, 255), 2, 8, 0);
			}
		}
	}
	
	imshow("origin", g_srcImage1);
	imshow("result", scaledImage);

}
*/

int main() {

	string wnd = "picture";
	//Mat img = imread("person.jpg", 1);//0 is grayscale, 1/default is color
	Mat img = imread("pic.jpg", 1);//0 is grayscale, 1/default is color
	imshow("origin image:", img);

	/*

	// 1. Show an image
	namedWindow(wnd);
	imshow(wnd, img);
	waitKey();

	//2. 创建带alpha通道的Mat
	Mat mat(480, 640, CV_8UC4);
	createAlphaMat(mat);
	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9);

	try {
		imwrite("alpha.png", mat, compression_params);
		imshow(wnd+"1", mat);
	}
	catch (runtime_error& ex) {
		fprintf(stderr, "图像转换成PNG格式发生错误：%s\n", ex.what());
		return 1;
	}
	waitKey();


	// 3. Create, merge and display ROI
	Mat roiImg=img.clone();
	Mat imgROI = roiImg(Rect(0, 0, 400, 400));//x,y,width, height
	Mat imgROI2 = roiImg(Rect(400, 100, 400, 400));
	addWeighted(imgROI, 0.7, imgROI2, 0.2, 0., imgROI2); // 2 src imgs must have same cols/rows!
	imshow(wnd + "3", roiImg);
	waitKey();

	


	// 4. split multi-channel image and merge blue channel
	vector<Mat> channels;
	Mat spl = img.clone();
	split(spl, channels);
	Mat blue = channels.at(0);
	Mat green = channels.at(1);
	Mat red = channels.at(2);
	Mat gray = imread("pic.jpg", 0);
	addWeighted(blue(Rect(0, 0, 200, 400)), 1, gray(Rect(0, 0, 200, 400)), 0.5, 0., blue(Rect(0, 0, 200, 400)));
	addWeighted(green(Rect(200, 0, 200, 400)), 1, gray(Rect(200, 0, 200, 400)), 0.5, 0., green(Rect(200, 0, 200, 400)));
	addWeighted(red(Rect(400, 0, 200, 400)), 1, gray(Rect(400, 0, 200, 400)), 0.5, 0., red(Rect(400, 0, 200, 400)));
	merge(channels, spl);
	imshow(wnd + "5", spl);
	waitKey();

	

	
	// 5. find contours	(it will crash due to unknown issue, ignore it)
	//创建轨迹条	
	threshval = 168;
	namedWindow("contours",1);
	trackbar_img = imread("cartoon.jpg", 0);
	Mat newd = trackbar_img < 120;
	imshow("contours", newd);
	waitKey();
	createTrackbar("Threshold:", "contours", &threshval, 255, on_trackbar);
	on_trackbar(threshval, 0);//轨迹条回调函数


	
	// 6. change contrast and lumni
	//读入用户提供的图像
	g_srcImage = imread("pic.jpg");
	if (!g_srcImage.data) { printf("Oh，no，读取g_srcImage图片错误~！\n"); return false; }
	g_dstImage = Mat::zeros(g_srcImage.size(), g_srcImage.type());

	//设定对比度和亮度的初值
	g_nContrastValue = 80;
	g_nBrightValue = 80;

	namedWindow("effect", 1);
    createTrackbar("contrast:", "effect", &g_nContrastValue, 300, ContrastAndBright);
	createTrackbar("lumi:", "effect", &g_nBrightValue, 200, ContrastAndBright);
	
	ContrastAndBright(g_nContrastValue, 0);
	ContrastAndBright(g_nBrightValue, 0);
		
	//按下“q”键时，程序退出
	while (char(waitKey(1)) != 'q') {}
	

	// 8. non-linear filter

	g_srcImage = imread("person.jpg");
	//=================【<4>中值滤波】===========================
	//创建窗口
	namedWindow("medianBlur", 1);
	//创建轨迹条
	createTrackbar("param:", "medianBlur", &g_nMedianBlurValue, 50, on_MedianBlur);
	on_MedianBlur(g_nMedianBlurValue, 0);
	//=======================================================


	//=================【<5>双边滤波】===========================
	//创建窗口
	namedWindow("bilateralFilter", 1);
	//创建轨迹条
	createTrackbar("param:", "bilateralFilter", &g_nBilateralFilterValue, 50, on_BilateralFilter);
	on_BilateralFilter(g_nBilateralFilterValue, 0);
	//=======================================================

	//按下“q”键时，程序退出
	while (char(waitKey(1)) != 'q') {}

	
	

	// 9. 实时对摄像头图像做美白磨皮
	VideoCapture vc1(1);
	Mat src, tmp, out;
	while (1) {
		vc1 >> src;
		if (!tmp.data) {
			tmp = Mat::zeros(src.size(), src.type());
		}		
		imshow("source", src);
		// 美白
		for (int y = 0; y < src.rows; y++)
		{
			for (int x = 0; x < src.cols; x++)
			{
				for (int c = 0; c < 3; c++)
				{
					//g(i, j) = contrast*f(i, j) + brightness.
					// brightness= (1-contrast)*125
					tmp.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(1*(src.at<Vec3b>(y, x)[c]) + 40);
				}
			}
		}
		//磨皮
		bilateralFilter(tmp, out, 8, 16, 4); // 双边滤波 边缘留存效果好 8, 8*2, 8/2
		//medianBlur(src, out, 5);
		//GaussianBlur(tmp, out, Size(7, 7), 0, 0);
		imshow("out", out);
		if (char(waitKey(1)) == 'q') {
			break;
		}
	}
	

	// 10. 边缘检测
	VideoCapture vc1(1);
	Mat src;
	Mat dst, edge, gray;

	while (1) {
		vc1 >> src;
		if (!dst.data) {
			dst = Mat::zeros(src.size(), src.type());
		}
		imshow("source", src);

		// Canny 
		cvtColor(src, gray, CV_BGR2GRAY);
		blur(gray, edge, Size(3, 3));
		Canny(edge, edge, 20, 60, 3);
		dst = Scalar::all(0);
		src.copyTo(dst, edge);
		imshow("Canny output", dst);

		// Sobel
		Mat grad_x, grad_y;
		Mat abs_grad_x, abs_grad_y, dst;
		Sobel(src, grad_x, CV_16S, 1, 0, 3, 1, 1, BORDER_DEFAULT);
		convertScaleAbs(grad_x, abs_grad_x);
		Sobel(src, grad_y, CV_16S, 0, 1, 3, 1, 1, BORDER_DEFAULT);
		convertScaleAbs(grad_y, abs_grad_y);
		addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst);
		imshow("Sobel output", dst);

		// Laplacian
		Mat abs_dst;
		GaussianBlur(src, src, Size(3, 3), 0, 0, BORDER_DEFAULT);
		Laplacian(gray, dst, CV_16S, 3, 1, 0, BORDER_DEFAULT);
		convertScaleAbs(dst, abs_dst);
		imshow("Laplace output", abs_dst);

		// Scharr
		Scharr(src, grad_x, CV_16S, 1, 0, 1, 0, BORDER_DEFAULT);
		convertScaleAbs(grad_x, abs_grad_x);
		Scharr(src, grad_y, CV_16S, 0, 1, 1, 0, BORDER_DEFAULT);
		convertScaleAbs(grad_y, abs_grad_y);
		addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst);
		imshow("Scharr output", dst);

		if (char(waitKey(1)) == 'q') {
			break;
		}
	}	
	

	// 11. 图像金字塔
	Mat dst1, dst2, tmp;
	resize(img, tmp, Size(img.cols / 2, img.rows / 2), 0.0, 0.0, CV_INTER_AREA);
	resize(tmp, dst1, Size(tmp.cols *2, tmp.rows * 2), 0.0, 0.0, CV_INTER_AREA);
	resize(img, tmp, Size(img.cols / 2, img.rows / 2), 0.0, 0.0, CV_INTER_LINEAR);
	resize(tmp, dst2, Size(tmp.cols * 2, tmp.rows * 2), 0.0, 0.0, CV_INTER_LINEAR);
	imshow("resize up&down area interpolation:", dst1);
	imshow("resize up&down linear interpolation:", dst2);

	Mat dst3;
	pyrUp(img, tmp, Size(img.cols * 2, img.rows * 2));
	pyrDown(tmp, dst3, Size(tmp.cols / 2, tmp.rows / 2));
	imshow("pyramid up&down:", dst3);
	waitKey();
	
	

	// 12. 霍夫变化
	Mat cartoon = imread("cartoon.jpg", 1);//0 is grayscale, 1/default is color
	Mat mid, dst;
	Canny(cartoon, mid, 50, 200, 3);

	cvtColor(mid, dst, CV_GRAY2BGR);
	vector<Vec2f> lines;

	int64 app_start_time = getTickCount();
	HoughLines(mid, lines, 1, CV_PI / 180, 150, 0, 0);
	cout << "HoughLines Finished, total time: " << ((getTickCount() - app_start_time) / getTickFrequency()) << " sec" << endl;
	// 48ms on haswell
	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(dst, pt1, pt2, Scalar(55, 100, 195), 1, CV_AA);
	}
	imshow("Hough line: ", dst);

	Mat dst2;
	cvtColor(mid, dst2, CV_GRAY2BGR);
	vector<Vec4i> lines2;//定义一个矢量结构lines用于存放得到的线段矢量集合  

	app_start_time = getTickCount();
	HoughLinesP(mid, lines2, 1, CV_PI / 180, 80, 50, 10); 
	cout << "HoughLinesP Finished, total time: " << ((getTickCount() - app_start_time) / getTickFrequency()) << " sec" << endl;
	// 64ms on haswell
	for (size_t j = 0; j < lines2.size(); j++)
	{
		Vec4i l = lines2[j];
		line(dst2, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(186, 88, 255), 1, CV_AA);
	}
	imshow("Hough line P: ", dst2);


	Mat buble = imread("buble.jpg", 1);//0 is grayscale, 1/default is color
	cvtColor(buble, mid, CV_BGR2GRAY);//转化边缘检测后的图为灰度图  
	//GaussianBlur(mid, mid, Size(9, 9), 2, 2);

	//进行霍夫圆变换  
	app_start_time = getTickCount();
	vector<Vec3f> circles;
	HoughCircles(mid, circles, CV_HOUGH_GRADIENT, 1.5, 10, 200, 100, 0, 0); 
	cout << "HoughCircles Finished, total time: " << ((getTickCount() - app_start_time) / getTickFrequency()) << " sec" << endl;
	// 880ms on haswell without GaussianBlur

	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		circle(buble, center, 3, Scalar(0, 255, 0), -1, 8, 0);
		circle(buble, center, radius, Scalar(155, 50, 255), 3, 8, 0);
	}
	imshow("Hough Circle: ", buble);

	waitKey();


	// 13 角点检测
	g_srcImage = imread("greatwall.jpg");
	g_srcImage1 = g_srcImage.clone();
	cvtColor(g_srcImage1, g_grayImage, CV_BGR2GRAY);
	namedWindow("origin", CV_WINDOW_AUTOSIZE);
	createTrackbar("threshold: ", "origin", &thresh, max_thresh, on_CornerHarris);
	on_CornerHarris(0, 0);


	
	
	// 17. SURF特征点
	int minHessian = 400;//定义SURF中的hessian阈值特征点检测算子
	SurfFeatureDetector detector(minHessian);//定义一个SurfFeatureDetector（SURF） 特征检测类对象
	std::vector<KeyPoint> keypoints_1;//vector模板类是能够存放任意类型的动态数组，能够增加和压缩数据

	//【3】调用detect函数检测出SURF特征关键点，保存在vector容器中
	detector.detect(img, keypoints_1);

	//【4】绘制特征关键点
	Mat img_keypoints_1;
	drawKeypoints(img, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

	//【5】显示效果图
	imshow("特征点检测效果图1", img_keypoints_1);
	*/

	// 18. surf matching
	Mat srcImage1 = imread("surf1.jpg", 1);
	Mat srcImage2 = imread("surf2.jpg", 1);

	int minHessian = 700;//SURF算法中的hessian阈值
	SurfFeatureDetector detector(minHessian);//定义一个SurfFeatureDetector（SURF） 特征检测类对象  
	std::vector<KeyPoint> keyPoint1, keyPoints2;//vector模板类，存放任意类型的动态数组

	detector.detect(srcImage1, keyPoint1);
	detector.detect(srcImage2, keyPoints2);

	SurfDescriptorExtractor extractor;
	Mat descriptors1, descriptors2;
	extractor.compute(srcImage1, keyPoint1, descriptors1);
	extractor.compute(srcImage2, keyPoints2, descriptors2);

	FlannBasedMatcher matcher;
	std::vector< DMatch > matches;
	//匹配两幅图中的描述子（descriptors）
	matcher.match(descriptors1, descriptors2, matches);
	Mat imgMatches;
	drawMatches(srcImage1, keyPoint1, srcImage2, keyPoints2, matches, imgMatches);//进行绘制
	imshow("匹配图", imgMatches);


	waitKey();

	return 0;
}