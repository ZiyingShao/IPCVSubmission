/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - face.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
#include <stdio.h>
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

/** Function Headers */
void detectAndDisplay( Mat frame, char*name);

/** Global variables */
String cascade_name = "frontalface.xml";
CascadeClassifier cascade;

struct rect{
	int x1;
	int y1;
	int x2;
	int y2;
	int used;
};
typedef struct rect myrect;

/** @function main */
int main( int argc, char** argv )
{
  // 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	char*name = argv[1];
	// 3. Detect Faces and Display Result
	detectAndDisplay( frame, name );

	// 4. Save Result Image
	std::string imageOutName = "subtask1/detectedFace-";
	imageOutName.append(name);
	imwrite( imageOutName, frame );

	return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame, char*name)
{
	std::vector<Rect> faces;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

  // 3. Print number of Faces found
	//std::cout << faces.size() << std::endl;

	int x1, y1, x2, y2;
	std::vector<myrect> rects;
	int numrect;
	std::string groundTruth = "subtask1/";
	groundTruth.append(name);
	groundTruth.append(".faces.csv");
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
		rect.used = 0;
		rects.push_back(rect);
		rectangle(frame, Point(rect.x1, rect.y1), Point(rect.x2, rect.y2), Scalar( 0, 255, 0 ), 2);
	}

	int truePositives = 0;
  //4. Draw box around faces found
	for( int i = 0; i < faces.size(); i++ )
	{
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 0, 255 ), 2);
		//Prints from rects
		for (int j=0; j<numrect; j++) {
			myrect rect = rects[j];
			int width1 = rect.x2-rect.x1;
			int height1 = rect.y2-rect.y1;
			//printf("\nGround: %d, %d, %d, %d\n",rect.x1, rect.y1, width1, height1);
			//printf("Detected: %d, %d, %d, %d\n",faces[i].x, faces[i].y, faces[i].width, faces[i].height);
			Rect ground(rect.x1, rect.y1, width1, height1);
			Rect detected(faces[i].x, faces[i].y, faces[i].width, faces[i].height);
			Rect intersection = ground & detected;
			double intersectionArea = intersection.area();
			if (intersectionArea > 0) {
				//printf("Ground area: %d\n", ground.area());
				//printf("Detected area: %d\n", detected.area());

				//printf("Intersection area: %f\n", intersectionArea);
				Rect unionOfBoxes = ground | detected;
				double unionArea = unionOfBoxes.area();

				//printf("Union area: %f\n", unionArea);
				double IoU = intersectionArea/unionArea;
				//printf("IoU: %f\n", IoU);
				if (IoU >= 0.5) {
						if (rect.used == 0) {
							truePositives +=1;
						}
						rect.used = 1;
				}
			}
		}
	}

	// cout << "Number of faces (ground truth): " << numrect << endl;
	// cout << "Number of faces detected correctly (true positives): " << truePositives << endl;
	// cout << "Number of faces detected correctly (true positives): " << truePositives << endl;

	double truePositiveRate = (double) truePositives / (double) numrect;
	double recall = truePositiveRate;
	double numberOfPositives = faces.size();
	double precision = truePositives/numberOfPositives;
	double f1 = 2*((precision*recall)/(precision+recall));

	// cout << "Number of faces detected (positves): " << numberOfPositives << endl;
	cout << "True Positive Rate: " << truePositiveRate << endl;
	// cout << "Recall: " << recall << endl; //1 is best, 0 is worst
	// cout << "Precision: " << precision << endl; //1 is best, 0 is worst
	cout << "F1: " << f1 << endl; //1 is best, 0 is worst
}
