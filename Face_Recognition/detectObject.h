#pragma once

#include "public.h"



// ����ͼ���еĵ������󣬱�����������������洢����largestObject���С�
void detectLargestObject(const Mat& img, CascadeClassifier& cascade, Rect& largestObject, int scaledWidth = 320);

// �ø����Ĳ�������һ��ͼƬ���󣬴浽objects��
void detectObjectsCustom(const Mat& img, CascadeClassifier& cascade, vector<Rect>& objects, \
	int scaleWidth, int flags, Size minFeatureSize, float searchScaleFactor, int minNeighbors);


