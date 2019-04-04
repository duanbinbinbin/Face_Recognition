#pragma once

#include "public.h"



// 搜索图像中的单个对象，比如最大的脸，将结果存储到“largestObject”中。
void detectLargestObject(const Mat& img, CascadeClassifier& cascade, Rect& largestObject, int scaledWidth = 320);

// 用给出的参数检测出一个图片对象，存到objects中
void detectObjectsCustom(const Mat& img, CascadeClassifier& cascade, vector<Rect>& objects, \
	int scaleWidth, int flags, Size minFeatureSize, float searchScaleFactor, int minNeighbors);


