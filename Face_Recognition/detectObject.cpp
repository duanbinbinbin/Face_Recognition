#include "detectObject.h"

// 搜索图像中的单个对象，比如最大的脸，将结果存储到“largestObject”中。
/*
1.定义级联人脸检测的参数
2.具体级联检测函数 detectObjectsCustom（）
3.检测成功返回正确图像矩形，失败则返回一个异常矩形
*/
void detectLargestObject(const Mat& img, CascadeClassifier& cascade, Rect& largestObject, int scaledWidth)
{
	// 只检测最大的物体
	int flags = CASCADE_FIND_BIGGEST_OBJECT;
	// 最小窗口大小
	Size minFeatureSize = Size(20, 20);
	// 每次扩大的比率
	float searchScalarFactor = 1.1f;
	// 最小检测到的数量阀值
	int minNeighbors = 4;

	//if (myDebug) printf("[myDebug][%s][%d] img.cols = %d\n", __FILE__, __LINE__, img.cols);//640

	// 检测到的最大头像的方框
	vector<Rect> objects;
	detectObjectsCustom(img, cascade, objects, scaledWidth, flags, minFeatureSize, searchScalarFactor, minNeighbors);
	
	//if (myDebug) printf("[mydebug][%s][%d] 级联检测的头像方框个数为: %d\n", __FILE__, __LINE__, objects.size()); // 0

	if (objects.size() > 0)
		largestObject = objects.at(0);
	else
		largestObject = Rect(-1,-1,-1,-1);
}

// 用给出的参数检测出一个图片对象，存到objects中
/*
1.转灰度图像
2.缩小图像，加快速度
3.直方图均衡化
4.级联人脸检测
5.将图像放大还原
6.防止object的坐标在图片之外，做容错处理

得到人脸图像矩形
*/
void detectObjectsCustom(const Mat& img, CascadeClassifier& cascade, vector<Rect>& objects, \
	int scaleWidth, int flags, Size minFeatureSize, float searchScaleFactor, int minNeighbors)
{
	// 将输入的图像转为灰度图
	Mat gray;
	if (img.channels() == 3)
		cvtColor(img, gray, COLOR_BGR2GRAY);
	else if (img.channels() == 4)
		cvtColor(img, gray, COLOR_BGRA2GRAY);
	else
		gray = img;

	//if (myDebug) printf("[myDebug][%s][%d] img.cols = %d, gray.cols = %d\n", __FILE__, __LINE__, img.cols, gray.cols);//640

	// 缩小图像加快运行速度
	Mat inputImg;
	float scale = img.cols / (float)scaleWidth;
	if (img.cols > scaleWidth)
	{
		// 在保持相同长宽比的情况下缩小图像
		int scaledHeight = cvRound(img.rows / scale);
		resize(gray, inputImg, Size(scaleWidth, scaledHeight));
	}
	else
	{
		inputImg = gray;
		//resize(gray, inputImg, Size(200, 200));
		//if (myDebug) printf("[myDebug][%s][%d] inputImg.cols = %d\n", __FILE__, __LINE__, inputImg.cols);
	}
	// 容错处理
	if (gray.cols < 20)
	{
		printf("\n[DEBUG][%s][%d] 输入图像有误\n", __FILE__, __LINE__);
		imshow("inputImg", inputImg); waitKey(0);
	}

	// 直方图均衡化，增强对比度
	Mat equalizedImg;
	equalizeHist(inputImg, equalizedImg);

	//if (myDebug) printf("[myDebug][%s][%d] equalizedImg.cols = %d\n", __FILE__, __LINE__, equalizedImg.cols);

	// 级联检测器检测
	cascade.detectMultiScale(equalizedImg, objects, searchScaleFactor, minNeighbors, flags, minFeatureSize);

	//if (myDebug) printf("[myDebug][%s][%d] objects.size() = %d\n", __FILE__, __LINE__, objects.size());

	// 如果在检测结果是缩小了，则放大结果
	if (img.cols > scaleWidth)
	{
		for (int i = 0; i < objects.size(); i++)
		{
			objects[i].x = cvRound(objects[i].x * scale);
			objects[i].y = cvRound(objects[i].y * scale);
			objects[i].width = cvRound(objects[i].width * scale);
			objects[i].height = cvRound(objects[i].height * scale);
		}
	}

	// 确保对象完全在图像中，以防它位于边框以外。
	for (int i = 0; i < objects.size(); i++)
	{
		if (objects[i].x < 0)
			objects[i].x = 0;
		if (objects[i].y < 0)
			objects[i].y = 0;
		if (objects[i].x + objects[i].width > img.cols)
			objects[i].x = img.cols - objects[i].width;
		if (objects[i].y + objects[i].height > img.rows)
			objects[i].y = img.rows - objects[i].height;
	}
}
