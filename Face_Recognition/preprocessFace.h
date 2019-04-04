#pragma once
#include "public.h"


// ����һ�����б�׼��С���ԱȶȺ����ȵĻҶ�����ͼ��
Mat getProcessedFace(Mat& srcImg, int desiredFaceWidth, CascadeClassifier& faceCascade, \
	CascadeClassifier& eyeCascade1, CascadeClassifier &eyeCascade2, bool doLeftAndRightSeparately, \
	Rect* storeFaceRect = NULL, Point* storeLeftEye = NULL, Point* storeRightEye = NULL, Rect* searchedLeftEye = NULL, \
	Rect* searchedRightEye = NULL);

void detectBothEyes(const Mat &face, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2, \
	Point &leftEye, Point &rightEye, Rect *searchedLeftEye = NULL, Rect *searchedRightEye = NULL);

// ��ֻ�۾��ֱ�ֱ��ͼ���⻯��
void equalizeLeftAndRightHalves(Mat& faceImg);



