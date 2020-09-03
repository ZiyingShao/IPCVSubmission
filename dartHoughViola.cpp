/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - face.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
#include <stdio.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cxcore.h>
#include <opencv2/opencv.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>

using namespace std;
using namespace cv;

void builtInSobel(Mat grayFrameBlurred, Mat dfdx, Mat dfdy, Mat magnitude, Mat magnitudeThresholded, Mat direction, Mat normalisedDirection);
void myHoughCircle(Mat frame, Mat magnitudeThresholded, Mat direction, Mat houghSpace, Mat normalisedHoughSpace, Mat thresholdedHoughSpace);
void detectAndDisplay(Mat frame, Mat grayFrame, char* name, Mat thresholdedHoughSpace);

String cascade_name = "dartcascade500/cascade.xml";
CascadeClassifier cascade;

struct rect{
	int x1;
	int y1;
	int x2;
	int y2;
	int marked;
};
typedef struct rect myrect;

int main(int argc, char** argv) {
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	Mat grayFrame(frame.rows, frame.cols, CV_32F, Scalar(0));
	Mat grayFrameBlurred(frame.rows, frame.cols, CV_32F, Scalar(0));
	cvtColor(frame, grayFrame, CV_BGR2GRAY);
	GaussianBlur(grayFrame, grayFrameBlurred, Size(3, 3), 0, 0, BORDER_DEFAULT);
	imwrite("frame.png", frame);
	imwrite("grayFrame.png", grayFrame);
	imwrite("grayFrameBlurred.png", grayFrameBlurred);

	Mat dfdx(frame.rows, frame.cols, CV_32F, Scalar(0));
	Mat dfdy(frame.rows, frame.cols, CV_32F, Scalar(0));
	Mat magnitude(frame.rows, frame.cols, CV_32F, Scalar(0));
	Mat magnitudeThresholded(frame.rows, frame.cols, CV_32F, Scalar(0));
	Mat direction(frame.rows, frame.cols, CV_32F, Scalar(0));
	Mat normalisedDirection(frame.rows, frame.cols, CV_32F, Scalar(0));

	builtInSobel(grayFrameBlurred, dfdx, dfdy, magnitude, magnitudeThresholded, direction, normalisedDirection);
	imwrite("sobelGradientX.png", dfdx);
	imwrite("sobelGradientY.png", dfdy);
	imwrite("sobelGradientMagnitude.png", magnitude);
	imwrite("sobelGradientMagnitudeThresholded.png", magnitudeThresholded);
	imwrite("sobelGradientDirection.png", direction);
	imwrite("sobelGradientDirectionNormalised.png", normalisedDirection);

	Mat houghSpace(frame.rows, frame.cols, CV_32F, Scalar(0));
	Mat normalisedHoughSpace(frame.rows, frame.cols, CV_32F, Scalar(0));
	Mat thresholdedHoughSpace(frame.rows, frame.cols, CV_32F, Scalar(0));
	myHoughCircle(frame, magnitudeThresholded, direction, houghSpace, normalisedHoughSpace, thresholdedHoughSpace);
	imwrite("houghSpace.png", houghSpace);
	imwrite("houghSpaceNormalised.png", normalisedHoughSpace);
	imwrite("houghSpaceThresholded.png", thresholdedHoughSpace);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if(!cascade.load(cascade_name) ){printf("--(!)Error loading\n"); return -1;};

	// 3. Detect Faces and Display Result
	char* name = argv[1];
	detectAndDisplay(frame, grayFrame, name, thresholdedHoughSpace);

	// 4. Save Result Image
	string imageOutName = "subtask3/detectedDartboard-";
	imageOutName.append(name);
	imwrite(imageOutName, frame);
	return 0;
}

void builtInSobel(Mat grayFrameBlurred, Mat dfdx, Mat dfdy, Mat magnitude, Mat magnitudeThresholded, Mat direction, Mat normalisedDirection) {
		double thresholdValue = 145;
		cv::Sobel(grayFrameBlurred, dfdx, CV_32F, 1, 0);
	  cv::Sobel(grayFrameBlurred, dfdy, CV_32F, 0, 1);
	  cv::magnitude(dfdx, dfdy, magnitude);
		cv::threshold(magnitude, magnitudeThresholded, thresholdValue, 255, THRESH_BINARY);
	  cv::phase(dfdx, dfdy, direction, false);
	  cv::normalize(direction, normalisedDirection, 0, 255, NORM_MINMAX, CV_32F);
}

void myHoughCircle(Mat frame, Mat magnitudeThresholded, Mat direction, Mat houghSpace, Mat normalisedHoughSpace, Mat thresholdedHoughSpace) {
	//40-300 is range of dart radii ground truths, search range must also look for inner circles
	int radiusRangeLowerBound = 20;
	int radiusRangeUpperBound = 350;
	float maxVote = 0;
	double thetaRadians;
	int thetaDegrees;
	float ry, rx, a, b;

	for(int y = 0; y < frame.rows; y++) {
		for(int x = 0; x < frame.cols; x++) {
			if (magnitudeThresholded.at<float>(y,x) == 255) {
				thetaRadians = direction.at<float>(y,x); //fixes theta to limit search space and speed up loop
				//for (thetaDegrees = 0; thetaDegrees<360; thetaDegrees++) {
					//thetaRadians = (thetaDegrees * M_PI) / 180;
					for (int r = radiusRangeLowerBound; r < radiusRangeUpperBound; r++) { //for each radius r in search range
						ry = r * sin(thetaRadians);
						rx = r * cos(thetaRadians);
						a = y - ry;
						b = x - rx;
						if(a>=0 && a<frame.rows && b>=0 && b<frame.cols) {
							houghSpace.at<float>(a,b) += 1;
							if (houghSpace.at<float>(y,x) > maxVote) {
								maxVote = houghSpace.at<float>(y,x);
							}
						}
						a = y + ry;
						b = x + rx;
						if(a>=0 && a<frame.rows && b>=0 && b<frame.cols) {
							houghSpace.at<float>(a,b) += 1;
							if (houghSpace.at<float>(y,x) > maxVote) {
								maxVote = houghSpace.at<float>(y,x);
							}
						}
					}
				//}
			}
    }
	}
	//normalise values to display hough space
	cv::normalize(houghSpace, normalisedHoughSpace, 0, 255, NORM_MINMAX, CV_32F);
	cv::GaussianBlur(houghSpace, thresholdedHoughSpace, Size(25, 25), 3, 0, BORDER_DEFAULT);
	imwrite("houghSpaceBlurred.png", thresholdedHoughSpace);
	cv::threshold(thresholdedHoughSpace, thresholdedHoughSpace, maxVote, 255, THRESH_BINARY);

	// for(int y = 0; y < frame.rows; y++) {
	// 	for (int x = 0; x < frame.cols; x++) {
	// 		if(thresholdedHoughSpace.at<float>(y,x) >= maxVote) {
	// 			circle(frame, Point(x,y), 1, Scalar(255,0,255), 1); //draws centre
	// 		}
	// 	}
	// }

	// cout << "Max vote: " << maxVote << endl;
}

/** @function detectAndDisplay */
void detectAndDisplay(Mat frame, Mat grayFrame, char* name, Mat thresholdedHoughSpace) {
	std::vector<Rect> faces;
	std::vector<Rect> facesReduced;

	// 1. Prepare Image by normalising lighting
	equalizeHist(grayFrame, grayFrame);

	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale(grayFrame, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500));

	int maxHoughInBox;
	// 3. Print number of Faces found
	//cout << faces.size() << endl;
	//Remove detections that don't match hough space
	for( int k = 0; k < faces.size(); k++ ) {
		maxHoughInBox = 0;
		//Search bounding box for high hough space values
		for (int i = faces[k].y; i <= faces[k].y + faces[k].height; i++) {
			for (int j = faces[k].x; j<= faces[k].x + faces[k].width; j++) {
				if (thresholdedHoughSpace.at<float>(i,j) > maxHoughInBox) {
					maxHoughInBox = thresholdedHoughSpace.at<float>(i,j);
				}
			}
		}
		//cout << maxHoughInBox << endl;
		if (maxHoughInBox > 0) {
			facesReduced.push_back(faces[k]);
		}
	}


	int x1, y1, x2, y2;
	std::vector<myrect> rects;
	int numrect;
	std::string groundTruth = "subtask2/"; //or subtask2/
	groundTruth.append(name);
	groundTruth.append(".darts.csv"); //or .darts.csv
	const char *cstr = groundTruth.c_str();
	FILE*ifp=fopen(cstr,"r");
	fscanf(ifp,"%d",&numrect);
	//Reads from file into rects
	for (int i=0; i<numrect; i++) {
		fscanf(ifp,"%d, %d, %d, %d",&x1,&y1,&x2,&y2);
		myrect rect;
		rect.x1 = x1;
		rect.y1 = y1;
		rect.x2 = x2;
		rect.y2 = y2;
		rect.marked = 0;
		rects.push_back(rect);
		rectangle(frame, Point(rect.x1, rect.y1), Point(rect.x2, rect.y2), Scalar( 0, 255, 0 ), 2);
	}

	int truePositives = 0;
  //4. Draw box around faces found
	for( int i = 0; i < facesReduced.size(); i++ )
	{
		rectangle(frame, Point(facesReduced[i].x, facesReduced[i].y), Point(facesReduced[i].x + facesReduced[i].width, facesReduced[i].y + facesReduced[i].height), Scalar( 0, 0, 255 ), 2);
		//Prints from rects
		for (int j=0; j<numrect; j++) {
			myrect rect = rects[j];
			int width1 = rect.x2-rect.x1;
			int height1 = rect.y2-rect.y1;

			Rect ground(rect.x1, rect.y1, width1, height1);
			Rect detected(facesReduced[i].x, facesReduced[i].y, facesReduced[i].width, facesReduced[i].height);
			Rect intersection = ground & detected;
			double intersectionArea = intersection.area();
			if (intersectionArea > 0) {
				Rect unionOfBoxes = ground | detected;
				double unionArea = unionOfBoxes.area();
				double IoU = intersectionArea/unionArea;
				//printf("IoU: %f\n", IoU);
				if (IoU >= 0.5) {
						if (rect.marked ==0) {
							truePositives +=1;
						}
						rect.marked = 1;
				}
			}
		}
	}
	double truePositiveRate = (double) truePositives / (double) numrect;
	double recall = truePositiveRate;
	double numberOfPositives = facesReduced.size();
	double precision = truePositives/numberOfPositives;
	double f1 = 0;
	if ((precision+recall)!=0){
	 	f1 = 2*((precision*recall)/(precision+recall));
	}
	cout << "True Positive Rate: " << truePositiveRate << endl;
	// cout << "Recall: " << recall << endl; //1 is best, 0 is worst
	// cout << "Precision: " << precision << endl; //1 is best, 0 is worst
	cout << "F1: " << f1 << endl; //1 is best, 0 is worst
}
