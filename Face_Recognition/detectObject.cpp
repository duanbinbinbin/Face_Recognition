#include "detectObject.h"

// ����ͼ���еĵ������󣬱�����������������洢����largestObject���С�
/*
1.���弶���������Ĳ���
2.���弶����⺯�� detectObjectsCustom����
3.���ɹ�������ȷͼ����Σ�ʧ���򷵻�һ���쳣����
*/
void detectLargestObject(const Mat& img, CascadeClassifier& cascade, Rect& largestObject, int scaledWidth)
{
	// ֻ�����������
	int flags = CASCADE_FIND_BIGGEST_OBJECT;
	// ��С���ڴ�С
	Size minFeatureSize = Size(20, 20);
	// ÿ������ı���
	float searchScalarFactor = 1.1f;
	// ��С��⵽��������ֵ
	int minNeighbors = 4;

	//if (myDebug) printf("[myDebug][%s][%d] img.cols = %d\n", __FILE__, __LINE__, img.cols);//640

	// ��⵽�����ͷ��ķ���
	vector<Rect> objects;
	detectObjectsCustom(img, cascade, objects, scaledWidth, flags, minFeatureSize, searchScalarFactor, minNeighbors);
	
	//if (myDebug) printf("[mydebug][%s][%d] ��������ͷ�񷽿����Ϊ: %d\n", __FILE__, __LINE__, objects.size()); // 0

	if (objects.size() > 0)
		largestObject = objects.at(0);
	else
		largestObject = Rect(-1,-1,-1,-1);
}

// �ø����Ĳ�������һ��ͼƬ���󣬴浽objects��
/*
1.ת�Ҷ�ͼ��
2.��Сͼ�񣬼ӿ��ٶ�
3.ֱ��ͼ���⻯
4.�����������
5.��ͼ��Ŵ�ԭ
6.��ֹobject��������ͼƬ֮�⣬���ݴ���

�õ�����ͼ�����
*/
void detectObjectsCustom(const Mat& img, CascadeClassifier& cascade, vector<Rect>& objects, \
	int scaleWidth, int flags, Size minFeatureSize, float searchScaleFactor, int minNeighbors)
{
	// �������ͼ��תΪ�Ҷ�ͼ
	Mat gray;
	if (img.channels() == 3)
		cvtColor(img, gray, COLOR_BGR2GRAY);
	else if (img.channels() == 4)
		cvtColor(img, gray, COLOR_BGRA2GRAY);
	else
		gray = img;

	//if (myDebug) printf("[myDebug][%s][%d] img.cols = %d, gray.cols = %d\n", __FILE__, __LINE__, img.cols, gray.cols);//640

	// ��Сͼ��ӿ������ٶ�
	Mat inputImg;
	float scale = img.cols / (float)scaleWidth;
	if (img.cols > scaleWidth)
	{
		// �ڱ�����ͬ����ȵ��������Сͼ��
		int scaledHeight = cvRound(img.rows / scale);
		resize(gray, inputImg, Size(scaleWidth, scaledHeight));
	}
	else
	{
		inputImg = gray;
		//resize(gray, inputImg, Size(200, 200));
		//if (myDebug) printf("[myDebug][%s][%d] inputImg.cols = %d\n", __FILE__, __LINE__, inputImg.cols);
	}
	// �ݴ���
	if (gray.cols < 20)
	{
		printf("\n[DEBUG][%s][%d] ����ͼ������\n", __FILE__, __LINE__);
		imshow("inputImg", inputImg); waitKey(0);
	}

	// ֱ��ͼ���⻯����ǿ�Աȶ�
	Mat equalizedImg;
	equalizeHist(inputImg, equalizedImg);

	//if (myDebug) printf("[myDebug][%s][%d] equalizedImg.cols = %d\n", __FILE__, __LINE__, equalizedImg.cols);

	// ������������
	cascade.detectMultiScale(equalizedImg, objects, searchScaleFactor, minNeighbors, flags, minFeatureSize);

	//if (myDebug) printf("[myDebug][%s][%d] objects.size() = %d\n", __FILE__, __LINE__, objects.size());

	// ����ڼ��������С�ˣ���Ŵ���
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

	// ȷ��������ȫ��ͼ���У��Է���λ�ڱ߿����⡣
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
