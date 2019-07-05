/*******************************************************************************************************************
@Copyright 2018 Inspur Co., Ltd

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
documentation files(the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and / or sell copies of the Software, and 
to permit persons to whom the Software is furnished to do so, subject to the following conditions : 

The above copyright notice and this permission notice shall be included in all copies or substantial portions of
the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.

@Filename:  ImageConsProd.cpp
@Author:    Michael.Chen
@Version:   5.0
@Date:      31st/Jul/2018
*******************************************************************************************************************/

#include "DetectProcess.hpp"
#include "opencv2/opencv.hpp"
#include <opencv2/highgui.hpp>
#include <time.h>
#include<stdio.h> 

using namespace cv;
//Window config
#define VIDEO_WIDTH  640
#define VIDEO_HEIGHT 480
#define WINDOW_NAME "Param Adjust"

//Image buffer size
#define BUFFER_SIZE 1

//RGB color threshold 
int	red_threshold = 200;
int	green_threshold = 200;
int	blue_threshold = 200;

//save lane point of last detection
vector<Point> old_l;
vector<Point> old_r;

//y=ax^2+bx+c,this is coefficient b for left line and right line
float coefficientLB=0;
float coefficientRB=0;

volatile unsigned int prdIdx = 0; //image reading index
volatile unsigned int csmIdx = 0; //image processing index

//darknet variable definition
char *datacfg ;		//data path config
char *name_list ;	//class list
char **names  ;		//list name
image im;			//net input image
char *cfgfile ;		//net config file
char *weightfile;	//weight file
float thresh , hier_thresh;	//output hreshold
network *net;	//darknet network
image **alphabet;

//UI arrow
Mat leftarrow,rightarrow,sarrow,arrow;

//image processing speed output
double time_process;
char process_time[30];
char timenow[20];

//#define USE_CAMERA

struct ImageData {
	Mat img;             //camare data
	unsigned int frame;  //frame count
};

ImageData capturedata[BUFFER_SIZE];   //buffer of capture

/************************************************* 
    Function:       ImageProducer 
    Description:    Image read
    Input:          video file or camare image 
    Output:         one frame of reading iamge
    Return:         void
    Others:         none
    *************************************************/
void DetectProcess::ImageReader()
{
	Settings & setting = *_settings;
	string video_name = "../video/";

	#ifndef USE_CAMERA
	//read video file
	video_name+=setting.video_name;
	VideoCapture cap(video_name);

	#else
	//open camare
	VideoCapture cap(0);
	#endif

	if (!cap.isOpened())
    {
            std::cout<<"can't open video or cam"<<std::endl;
            return;
	}
		
	while(true)
    {
        //wait for next image
       	while(prdIdx - csmIdx >= BUFFER_SIZE);
        cap >> capturedata[prdIdx % BUFFER_SIZE].img;
        capturedata[prdIdx % BUFFER_SIZE].frame++; 	//frame is the index of picture
        ++prdIdx;
    }
}

/************************************************* 
    Function:       ImageConsumer 
    Description:    Image process
    Input:          one frame of reading iamge
    Output:         frame after processing display
    Return:         void
    Others:         none
    *************************************************/
void DetectProcess::ImageProcesser() {

	Settings & setting = *_settings;
	datacfg = "../coco.data";		//read data file, which contains（.names）file path,two (.txt)file
	name_list = option_find_str(read_data_cfg(datacfg), "names", "names.list");	//find the value of names in data file
	names = get_labels(name_list);		//get labels
	//egg 4 data code
	cfgfile = "../yolov3-tiny/yolov3-tiny.cfg";
	weightfile = "../yolov3-tiny/yolov3-tiny.weights";
	thresh = .5; hier_thresh = .5;
	net = load_network(cfgfile, weightfile, 0);		//load network
	set_batch_network(net, 1);		//set the batch of each layer 1
	alphabet = load_alphabet();		//load ASCII 32-127 in data/labels for lable displaying

	Mat frame,detect,road;		//frame - input image
								//detect - after road detect
								//road - ROI need to detet car
	//read UI image
	leftarrow = imread("../param/left.png");
	rightarrow = imread("../param/right.png");
	sarrow = imread("../param/s.png");
	resize(leftarrow,leftarrow,Size(128,128));
	resize(rightarrow,rightarrow,Size(128,128));
	resize(sarrow,sarrow,Size(128,128));

	while(true){
		//get time to caculate processing time
		time_process = what_time_is_it_now();
		//get current time
		time_t tt;
		time(&tt);
		tt = tt + 8 * 3600; // transform the time zone
		tm *t = gmtime(&tt);
		//云台视频储存
		sprintf(timenow, "%d-%02d-%02d-%02d:%02d:%02d",
				t->tm_year + 1900,
				t->tm_mon + 1,
				t->tm_mday,
				t->tm_hour,
				t->tm_min,
				t->tm_sec);

		while (prdIdx - csmIdx == 0);
		capturedata[csmIdx % BUFFER_SIZE].img.copyTo(frame);
		++csmIdx;
		
		resize(frame,frame,Size(1280,720));
		imshow("frame", frame);

		namedWindow(WINDOW_NAME);
		//RGB threshold Trackbar
		createTrackbar("Blue threshold",WINDOW_NAME,&blue_threshold,255);
		createTrackbar("Green threshold",WINDOW_NAME,&green_threshold,255);
		createTrackbar("Red threshold",WINDOW_NAME,&red_threshold,255);

		if(LaneDetecter(frame,road))	//lane detect
		{
			detect = road(Rect(0,320,road.cols,road.rows-320));
			CarDetecter(detect);		//car detect

			//FPS caculate
			float fps = 1/(what_time_is_it_now() - time_process);
			sprintf(process_time,"FPS: %.2f ",fps);
			putText(road,process_time,Point(15,20),CV_FONT_HERSHEY_SIMPLEX , 0.8, Scalar(0, 0, 0), 1);
			putText(road,timenow,Point(1000,20),CV_FONT_HERSHEY_SCRIPT_COMPLEX  , 0.5, Scalar(0, 255, 128), 1);

			imshow("detect", road);
			free_image(im);
		}	
		
		if(setting.debug_mode<1)	//wait for keyboard press
		{
			char key = waitKey(5);
			if(key == 27)
				exit(0);
			if(key == 13)
			{
				char filename[40];
				sprintf(filename,"../output/%s.jpg",timenow);
				cout<<"Note: "<<filename<<" write to file sucess!";
				imwrite(filename,road);
			}
		}
		else	//cotinue processing
		{
			char key = waitKey(0);
			if(key == 27)
				exit(0);
			if(key == 13)
			{
				char filename[40];
				sprintf(filename,"../output/%s.jpg",timenow);
				cout<<"Note: "<<filename<<" write to file sucess!";
				imwrite(filename,road);
			}
			
		}
	}
}

/************************************************* 
    Function:       polynomial_curve_fit 
    Description:   	Using least squares to fit line
	Input:          
					vector<Point>& key_point - observation value
					int n - power
					Mat& A - output matrix
    Output:         frame after processing display
    Return:         bool
    Others:         none
    *************************************************/
bool polynomial_curve_fit(vector<Point>& key_point, int n, Mat& A)	//最小二乘法
{
	//Number of key points
	int N = key_point.size();
 
	//structure matrix X
	cv::Mat X = cv::Mat::zeros(n + 1, n + 1, CV_64FC1);
	for (int i = 0; i < n + 1; i++)
	{
		for (int j = 0; j < n + 1; j++)
		{
			for (int k = 0; k < N; k++)
			{
				X.at<double>(i, j) = X.at<double>(i, j) +
					std::pow(key_point[k].x, i + j);
			}
		}
	}
 
	//structure matrix Y
	cv::Mat Y = cv::Mat::zeros(n + 1, 1, CV_64FC1);
	for (int i = 0; i < n + 1; i++)
	{
		for (int k = 0; k < N; k++)
		{
			Y.at<double>(i, 0) = Y.at<double>(i, 0) +
				std::pow(key_point[k].x, i) * key_point[k].y;
		}
	}
 
	A = cv::Mat::zeros(n + 1, 1, CV_64FC1);
	//solve A
	cv::solve(X, Y, A, cv::DECOMP_LU);
	return true;
}

/************************************************* 
    Function:       LaneDetect 
    Description:   	lane detect
	Input:          
					Mat src - input image
					Mat &detect_area - output image after processing
    Output:         output image after processing
    Return:         bool
    Others:         none
    *************************************************/
bool DetectProcess::LaneDetecter(Mat src,Mat &detect_area)
{
	Mat color;
	src.copyTo(color);
	//channel split
	vector<Mat> channels;
	split(color, channels);
	//Perspective Transform point
	vector<Point2f> corners(4); 
	corners[0]=(Point2f(203,720));
	corners[1]=(Point2f(585,450));
	corners[2]=(Point2f(695,450));
	corners[3]=(Point2f(1127,720));
	//Reverse perspective Transform point
	vector<Point2f> corners_trans(4); 
	corners_trans[0]=(Point2f(320,720));
	corners_trans[1]=(Point2f(320,0));
	corners_trans[2]=(Point2f(960,0));
	corners_trans[3]=(Point2f(960,720));

	Mat transform_img = getPerspectiveTransform(corners,corners_trans);		//caculate Perspective Transform
	Mat vtransform_img = getPerspectiveTransform(corners_trans,corners);	//caculate Reverse perspective Transform

	//data of each channel
	Mat Red = channels.at(2);
	Mat Green = channels.at(1);
	Mat Blue = channels.at(0);
	
	//choose white and yellow
	for (int x = 0; x < color.rows; x++)
	{
		for (int y = 0; y < color.cols; y++)
		{
			if ((Red.at<uchar>(x, y) > red_threshold && Green.at<uchar>(x, y) > green_threshold && Blue.at<uchar>(x, y) > blue_threshold) || (Red.at<uchar>(x, y) > red_threshold && Green.at<uchar>(x, y) && Blue.at<uchar>(x, y) < 150))
			{
				Red.at<uchar>(x, y) = 255;
				Green.at<uchar>(x, y) = 255;
				Blue.at<uchar>(x, y) = 255;
			}
			else
			{
				Red.at<uchar>(x, y) = 0;
				Green.at<uchar>(x, y) = 0;
				Blue.at<uchar>(x, y) = 0;
			}
		}
	}

	//channel mix
	vector<Mat> mix;
	mix.push_back(Blue);
	mix.push_back(Green);
	mix.push_back(Red);
	Mat result;
	merge(mix, result);

	imshow("binery", result);

	//Perspective Transform
	Mat warped;
	warpPerspective(result, warped, transform_img, warped.size(), INTER_LINEAR);
	morphologyEx(warped, warped, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(30, 30)));
	cvtColor(warped, warped, COLOR_BGR2GRAY);
	threshold(warped, warped, 128, 255, THRESH_BINARY);
	imshow("warper", warped);

	//get white frequency histogram
	vector<Point> left_points;
	vector<Point> right_points;
	Mat Hog = Mat::zeros(720, 1280, CV_8UC3);

	int leftx_current = 0, rightx_current = 0; //the x value of highest wihte
	int whitemax = 0;						   //the value of highest wihte 
	
	//left part
	for (int x = 0; x < warped.cols / 2; x++)
	{
		int white = 0;
		for (int y = 0; y < warped.rows; y++)
		{
			if (warped.at<uchar>(y, x) > 0)	//if the pixel contians white, count it and save point
			{
				white++;
				left_points.push_back(Point(x, y));
			}
		}
		if (white > whitemax)		//update x value of highest wihte
		{
			whitemax = white;
			leftx_current = x;
		}
		rectangle(Hog, Rect(Point(x, warped.rows - white), Size(1, white)), Scalar(255, 255, 255), 1, 8);	//draw histogram
	}

	whitemax = 0;
	//right part
	for (int x = warped.cols / 2; x < warped.cols; x++)
	{
		int white = 0;
		for (int y = 0; y < warped.rows; y++)
		{
			if (warped.at<uchar>(y, x) > 0)		//if the pixel contians white, count it and save point
			{
				white++;
				right_points.push_back(Point(x, y));
			}
		}
		if (white > whitemax)		//update x value of highest wihte
		{
			whitemax = white;
			rightx_current = x;
		}
		rectangle(Hog, Rect(Point(x, warped.rows - white), Size(1, white)), Scalar(255, 255, 255), 1, 8);	//draw histogram
	}

	imshow("Hog",Hog);

	//slide windows
	int nwindows = 9;			//totol number of windows
	int window_height = int(warped.rows/9);
	int margin = 100, minpix=50; 	// Set the width of the windows +/- margin,Set minimum number of pixels found to recenter window
	//index of lane
	vector<int> left_lane_inds;
	vector<int> right_lane_inds;
	Mat curimg = Mat::zeros(720, 1280, CV_8UC3);

	int minwindowleft=1280,maxwindowleft=0;				//store the region windows have slided
	int minwindowright=1280,maxwindowright=0;			//store the region windows have slided

	Point left_max;			//the most left and low point
	Point right_max;		//the most right and low point
	Point left_min;			//the most left and high point
	Point right_min;		//the most right and high point
	
	for(int window=0;window<nwindows;window++)
	{
		vector<int> good_left_inds;		//nonzero index
		vector<int> good_right_inds;	//nonzero index
		vector<int> nonzerox_left;		//nonzero x
		vector<int> nonzerox_right;		//nonzero x

		//window vertical region
		int win_y_low = warped.rows - (window + 1) *window_height;
		int win_y_high = warped.rows - window *window_height;

		//window horizontal region
		int win_xleft_low = leftx_current - margin;
		int win_xleft_high = leftx_current + margin;
		int win_xright_low = rightx_current - margin;
		int win_xright_high = rightx_current + margin;

		//update the region  windows have slided
		if(minwindowleft > win_xleft_low) minwindowleft=win_xleft_low;
		if(maxwindowleft < win_xleft_high) maxwindowleft= win_xleft_high;
		if(minwindowright > win_xright_low) minwindowright=win_xright_low;
		if(maxwindowright < win_xright_high) maxwindowright=win_xright_high;

		for(int i=0;i<left_points.size();i++)
		{
			if(left_points[i].x>=win_xleft_low && left_points[i].x <win_xleft_high 
			&& left_points[i].y >= win_y_low && left_points[i].y <win_y_high)
			{
				good_left_inds.push_back(i);	//save the index of left_points in the region
				nonzerox_left.push_back(left_points[i].x);	//save the nonzero x of left_points in the region
			}
		}
		for(int n=0;n<good_left_inds.size();n++)
		{
			left_lane_inds.push_back(good_left_inds[n]);
		}
		if(good_left_inds.size()>minpix)
		{	//update highest whiet x for next window
			double left_sum = accumulate(begin(nonzerox_left), end(nonzerox_left), 0.0);
			leftx_current = left_sum/nonzerox_left.size();
		}

		for(int i=0;i<right_points.size();i++)
		{
			if(right_points[i].x>=win_xright_low && right_points[i].x <win_xright_high 
			&& right_points[i].y >= win_y_low && right_points[i].y <win_y_high)
			{
				good_right_inds.push_back(i);	//save the index of right_points in the region
				nonzerox_right.push_back(right_points[i].x);	//save the nonzero x of right_points in the region
			}
		}
		for(int n=0;n<good_right_inds.size();n++)
		{
			right_lane_inds.push_back(good_right_inds[n]);
		}
		if(good_right_inds.size()>minpix)
		{	//update highest whiet x for next window
			double right_sum = accumulate(begin(nonzerox_right), end(nonzerox_right), 0.0);
			rightx_current = right_sum/nonzerox_right.size();
		}
		//draw slide windows
		Rect left_rect(Point(win_xleft_low,win_y_low),Size(200,window_height));
		Rect right_rect(Point(win_xright_low,win_y_low),Size(200,window_height));
		rectangle(curimg,left_rect,Scalar(128,255,0));
		rectangle(curimg,right_rect,Scalar(128,0,255));
		//save the most left/right and low/high point
		if(window == 8){
			Point left_max = Point(leftx_current,win_y_high);
			Point right_max = Point(rightx_current,win_y_high);
		}	
		if(window == 0){
			Point left_min = Point(leftx_current,win_y_low);
			Point right_min = Point(rightx_current,win_y_low);
		}	
	}

	vector<Point> left_line;	//key_point of left part

	for(int a=0;a<left_lane_inds.size();a++)
	{
		left_line.push_back(left_points[left_lane_inds[a]]);
	}



	vector<Point> right_line;	//key_point of left part
	for(int a=0;a<right_lane_inds.size();a++)
	{
		right_line.push_back(right_points[right_lane_inds[a]]);
	}
	//draw key_point
	for (int i = 0; i < left_line.size(); i++)
	{
		circle(curimg, left_line[i], 1, cv::Scalar(0, 0, 255), 1, 8, 0);
	}
	for (int i = 0; i < right_line.size(); i++)
	{
		//cout<< right_points[i];
		circle(curimg, right_line[i], 1, cv::Scalar(255, 0, 0), 1, 8, 0);
	}
	imshow("curimg",curimg);

	Mat coefficientLeft;	//coefficient of left
	//manual add point
	for (int x = left_max.x - 5; x < left_max.x + 5; x++)
	{
		for(int y = left_max.y-5;y<left_max.y+5;y++)
		{
			left_line.push_back(Point(x, y));
		}
	}
	for (int x = left_min.x - 5; x < left_min.x + 5; x++)
	{
		for(int y = left_min.y-5;y<left_min.y+5;y++)
		{
			left_line.push_back(Point(x, y));
		}
	}
	//fit a line
	vector<Point> points_fittedleft;
	if (polynomial_curve_fit(left_line, 2, coefficientLeft)&&left_line.size()!=0)
	{
		float polar_y = (4*coefficientLeft.at<double>(2, 0)*coefficientLeft.at<double>(0, 0)-pow(coefficientLeft.at<double>(1, 0),2))/4/coefficientLeft.at<double>(2, 0);
		if (abs(coefficientLeft.at<double>(2, 0)) > 0.015||polar_y<10||polar_y>700||old_l.size()==0)
		{
			coefficientLB = coefficientLeft.at<double>(1, 0);
			old_l.clear();
			for (int x = 0; x < 1280; x++)
			{
				double y = coefficientLeft.at<double>(0, 0) + coefficientLeft.at<double>(1, 0) * x +
						   coefficientLeft.at<double>(2, 0) * pow(x, 2); // + A.at<double>(3, 0) * std::pow(x, 3);
				if (x > minwindowleft - 10 && x < maxwindowleft + 10)
				{
					points_fittedleft.push_back(cv::Point(x, y));
					old_l.push_back(cv::Point(x, y));
				}	
			}
		}
		else
		{	
			cout<<"!!!";
			for(int i=0;i<old_l.size();i++)
			{
				points_fittedleft.push_back(old_l[i]);
			}
		}
		//draw fit line	
		polylines(curimg, points_fittedleft, false, cv::Scalar(0, 255, 255), 2, 8, 0);
		printf("y_l = %.3f*x^2 + %.3f*x+%.3f\n", coefficientLeft.at<double>(2, 0),coefficientLeft.at<double>(1, 0),coefficientLeft.at<double>(0, 0));
	}
	else return false;
	//else return false;
	Mat coefficientRight;
	//right_line.insert(right_line.begin(),add_point_right.begin(),add_point_right.end());
	//manual add point
	for (int x = right_max.x - 5; x < right_max.x + 5; x++)
	{
		for(int y = right_max.y-5;y<right_max.y+5;y++)
		{
			right_line.push_back(Point(x,y));
		}
	}
	for (int x = right_min.x - 5; x < right_min.x + 5; x++)
	{
		for(int y = right_min.y-5;y<right_min.y+5;y++)
		{
			right_line.push_back(Point(x, y));
		}
	}
	//fit a line
	vector<Point> points_fittedright;
	if (polynomial_curve_fit(right_line, 2, coefficientRight)&&right_line.size()!=0)
	{
		coefficientRB = coefficientRight.at<double>(1, 0);
		float polar_y = (4*coefficientRight.at<double>(2, 0)*coefficientRight.at<double>(0, 0)-pow(coefficientRight.at<double>(1, 0),2))/4/coefficientRight.at<double>(2, 0);
		if (abs(coefficientRight.at<double>(2, 0)) > 0.015||polar_y<10||polar_y>700||old_l.size()==0)
		{
			old_r.clear();
			for (int x = 0; x < 1280; x++)
			{
				double y = coefficientRight.at<double>(0, 0) + coefficientRight.at<double>(1, 0) * x +
						   coefficientRight.at<double>(2, 0) * pow(x, 2); // + A.at<double>(3, 0) * std::pow(x, 3);
				if (x > minwindowright - 10 && x < maxwindowright + 10)
				{
					points_fittedright.push_back(cv::Point(x, y));
					old_r.push_back(cv::Point(x, y));
				}	
			}
		}
		else
		{	
			cout<<"???";
			for(int i=0;i<old_r.size();i++)
			{
				points_fittedright.push_back(old_r[i]);
			}
		}
		printf("yr = %.3f*x^2 + %.3f*x+%.3f\n", coefficientRight.at<double>(2, 0),coefficientRight.at<double>(1, 0),coefficientRight.at<double>(0, 0));
		polylines(curimg, points_fittedright, false, cv::Scalar(0, 255, 255), 2, 8, 0);
	}
	else return false;

	imshow("fit", curimg);

	//image mask
	Mat mask = Mat::zeros(src.size(),CV_8UC3);
	vector<vector<Point>> contour;
	vector<Point> pts;

	if(points_fittedleft[0].y< points_fittedleft[points_fittedleft.size()-1].y)
	{
		for(int i = 0;i < points_fittedleft.size(); i++)	pts.push_back(points_fittedleft[i]);
	}
	else{
		for(int i = points_fittedleft.size()-1;i >-1; i--)	pts.push_back(points_fittedleft[i]);
	}
	if(points_fittedright[0].y > points_fittedright[points_fittedright.size()-1].y)
	{
		for(int i = 0;i < points_fittedright.size(); i++)	pts.push_back(points_fittedright[i]);
	}
	else{
		for(int i = points_fittedright.size()-1;i >-1; i--)	pts.push_back(points_fittedright[i]);
	}

	contour.push_back(pts);
	drawContours(mask,contour,0,Scalar(0,255,0),-1);
	imshow("mask",mask);

	Mat road_mask ;
	warpPerspective(mask, road_mask,vtransform_img,warped.size(),INTER_LINEAR) ;
	imshow("road_mask",road_mask);
	
	Mat road;
	double alpha = 1, beta = 0.2;
	addWeighted(src, alpha, road_mask, beta, 0.0, road);
	
	arrow = road(Rect(574,5,128,128));
	
	if(abs(coefficientLB)>6||abs(coefficientRB)>6)
	{//stright
		addWeighted(arrow, alpha, sarrow, alpha*0.8, 0.0, arrow);
	}
	else if((coefficientLB>-6&&coefficientLB<-1)||(coefficientRB>-6&&coefficientRB<-1))
	{//left
		addWeighted(arrow, alpha, leftarrow, alpha*0.8, 0.0, arrow);
	}
	else if((coefficientLB<6&&coefficientLB>1)||(coefficientRB<6&&coefficientRB>1))
	{//right
		addWeighted(arrow, alpha,rightarrow, alpha*0.8, 0.0, arrow);
	}
	//imshow("road",road);
	road.copyTo(detect_area);
	
	return true;
}

/************************************************* 
    Function:       Mat2Image 
    Description:   	change format: mat->image
	Input:          
					Mat RefImg - mat image need to reformat
					image *im - the output image format image
    Output:         image *im - the output image format image
    Return:         void
    Others:         none
    *************************************************/
void DetectProcess::Mat2Image(Mat RefImg,image *im)
{
	CV_Assert(RefImg.depth() == CV_8U);		//judge if  RefImag is CV_8U
	int h = RefImg.rows;
	int w = RefImg.cols;
	int channels = RefImg.channels();
	*im = make_image(w, h, 3);		//create 3 channels image
	int count = 0;
	switch (channels)
	{
	case 1:
	{
		MatIterator_<unsigned char> it, end;
		for (it = RefImg.begin<unsigned char>(), end = RefImg.end<unsigned char>(); it != end; ++it)
		{
			im->data[count] = im->data[w * h + count] = im->data[w * h * 2 + count] = (float)(*it) / 255.0;

			++count;
		}
		break;
	}

	case 3:
	{
		MatIterator_<Vec3b> it, end;
		for (it = RefImg.begin<Vec3b>(), end = RefImg.end<Vec3b>(); it != end; ++it)
		{
			im->data[count] = (float)(*it)[2] / 255.0;
			im->data[w * h + count] = (float)(*it)[1] / 255.0;
			im->data[w * h * 2 + count] = (float)(*it)[0] / 255.0;

			++count;
		}
		break;
	}

	default:
		printf("Channel number not supported.\n");
		break;
	}
}

/************************************************* 
    Function:       get_pixel 
    Description:   	change format: image->mat
	Input:          
					image m	- image need to get pixel
					int x - width
					int y - height
					int c - channels
    Output:         pixel of input image
    Return:         float
    Others:         none
    *************************************************/
float get_pixel(image m, int x, int y, int c)
{
    assert(x < m.w && y < m.h && c < m.c);
    return m.data[c*m.h*m.w + y*m.w + x];
}

/************************************************* 
    Function:       image2mat 
    Description:   	change format: image->mat
	Input:          
					image p	- image image need to reformat
					Mat *Img -	the output mat format image
    Output:         Mat *Img - the output immatage format image
    Return:         void
    Others:         none
    *************************************************/
void DetectProcess::Image2Mat(image p,Mat &Img)
{
	IplImage *disp = cvCreateImage(cvSize(p.w,p.h), IPL_DEPTH_8U, p.c);
    image copy = copy_image(p);
    constrain_image(copy);

	int x, y, k;
	if (p.c == 3)
		rgbgr_image(p);

	int step = disp->widthStep;
	for (y = 0; y < p.h; ++y)
	{
		for (x = 0; x < p.w; ++x)
		{
			for (k = 0; k < p.c; ++k)
			{
				disp->imageData[y * step + x * p.c + k] = (unsigned char)(get_pixel(p, x, y, k) * 255);
			}
		}
	}
	if (0)
	{
		int w = 448;
		int h = w * p.h / p.w;
		if (h > 1000)
		{
			h = 1000;
			w = h * p.w / p.h;
		}
		IplImage *buffer = disp;
		disp = cvCreateImage(cvSize(w, h), buffer->depth, buffer->nChannels);
		cvResize(buffer, disp, CV_INTER_LINEAR);
		cvReleaseImage(&buffer);
	}
	
	Img=cvarrToMat(disp);
	free_image(copy);
   	cvReleaseImage(&disp);
}

/************************************************* 
    Function:       detecter 
    Description:   	darknet detect car
	Input:          
					Mat &src - image need to detect
    Output:         Mat &src - image after drawing detect target
    Return:         void
    Others:         none
    *************************************************/
void DetectProcess::CarDetecter(Mat &src)
{  
	//format change
	Mat2Image(src,&im);
    float nms=.45;
   	
    layer l = net->layers[(net->n)-1];    
    image sized = letterbox_image(im, net->w, net->h);
    float *X = sized.data;
	network_predict(net, X);

	int nboxes = 0;
	detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);
	if (nms)
	{
		do_nms_sort(dets, nboxes, 80, nms);
	}
	int rect_scalar[nboxes][4];	//rect_scalar[i][0] left rect_scalar[i][1] right rect_scalar[i][2] top rect_scalar[i][3] bottom
	//draw_detections(im, dets, nboxes, thresh, names, alphabet,80);
	get_detections(im, dets, nboxes, thresh, names, alphabet,80,rect_scalar);
	vector<Rect> detectBox;
	for(int i=0;i<nboxes;i++)
	{
		detectBox.push_back(Rect(rect_scalar[i][0],rect_scalar[i][2],rect_scalar[i][1]-rect_scalar[i][0],rect_scalar[i][3]-rect_scalar[i][2]));
	}
	Mat result;
	Image2Mat(im,result);
	for(int i=0;i<detectBox.size();i++)
	{
		char position[10];
		sprintf(position,"(%d,%d)",detectBox[i].x+detectBox[i].width/2,detectBox[i].y+detectBox[i].height/2);
		putText(result,position,Point(detectBox[i].tl().x,detectBox[i].tl().y+30),CV_FONT_HERSHEY_PLAIN, 1, Scalar(128, 0, 255), 1);
	}
	result.copyTo(src);
    free_detections(dets, nboxes);
	free_image(sized);
}
