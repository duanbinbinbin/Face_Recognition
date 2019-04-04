#pragma once
#include "public.h"


// 创建一个具有标准大小、对比度和亮度的灰度人脸图像。
Mat getProcessedFace(Mat& srcImg, int desiredFaceWidth, CascadeClassifier& faceCascade, \
	CascadeClassifier& eyeCascade1, CascadeClassifier &eyeCascade2, bool doLeftAndRightSeparately, \
	Rect* storeFaceRect = NULL, Point* storeLeftEye = NULL, Point* storeRightEye = NULL, Rect* searchedLeftEye = NULL, \
	Rect* searchedRightEye = NULL);

void detectBothEyes(const Mat &face, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2, \
	Point &leftEye, Point &rightEye, Rect *searchedLeftEye = NULL, Rect *searchedRightEye = NULL);

// 两只眼睛分别直方图均衡化。
void equalizeLeftAndRightHalves(Mat& faceImg);



