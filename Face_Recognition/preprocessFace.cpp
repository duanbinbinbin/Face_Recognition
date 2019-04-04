#include "preprocessFace.h"
#include "detectObject.h"

bool myDebug = true;
const double DESIRED_LEFT_EYE_X = 0.16; // ���۾���߿�
const double DESIRED_LEFT_EYE_Y = 0.14; 
const double FACE_ELLIPSE_CY = 0.40;
const double FACE_ELLIPSE_W = 0.50;
const double FACE_ELLIPSE_H = 0.80;

// ����һ�����б�׼��С���ԱȶȺ����ȵĻҶ�����ͼ��
Mat getProcessedFace(Mat& srcImg, int desiredFaceWidth, CascadeClassifier& faceCascade, \
	CascadeClassifier& eyeCascade1, CascadeClassifier &eyeCascade2, bool doLeftAndRightSeparately, \
	Rect* storeFaceRect, Point* storeLeftEye, Point* storeRightEye, Rect* searchedLeftEye, Rect* searchedRightEye)
{
	// ���ĵĸ߶ȣ���� = 70��
	int desiredFaceHeight = desiredFaceWidth;

	// ��ʼ��������⵽������������۾�����������Ϊ��Ч���Է�����û�б���⵽��
	if (storeFaceRect)
		storeFaceRect->width = -1;
	if (storeLeftEye)
		storeLeftEye->x = -1;
	if (storeRightEye)
		storeRightEye->x = -1;
	if (searchedLeftEye)
		searchedLeftEye->width = -1;
	if (searchedRightEye)
		searchedRightEye->width = -1;

	Rect faceRect;
	// ���������������õ��������η���
	detectLargestObject(srcImg, faceCascade, faceRect);
	//if (myDebug) printf("[myDebug] faceRect.width = %d\n", faceRect.width); // 0

	// ����⵽���������۾�
	if (faceRect.width > 0)
	{
		if (storeFaceRect)
			*storeFaceRect = faceRect;

		// ��ȡ����ROI
		Mat faceImg = srcImg(faceRect);

		// ������ROIת��Ϊ�Ҷ�ͼ��
		Mat gray;
		if (faceImg.channels() == 3)
			cvtColor(faceImg, gray, COLOR_BGR2GRAY);
		else if (faceImg.channels() == 4)
			cvtColor(faceImg, gray, COLOR_BGRA2GRAY);
		else
			gray = faceImg;

		// �ۼ��������
		Point leftEye, rightEye; // �۾���������
		detectBothEyes(gray, eyeCascade1, eyeCascade2, leftEye, rightEye, searchedLeftEye, searchedRightEye);
		if (storeLeftEye)
			*storeLeftEye = leftEye;
		if (storeRightEye)
		{
			*storeRightEye = rightEye;
			if (myDebug)
				printf("[DEBUG] �۾������Լ����������ȡ�ɹ�\n");
		}

		// �����ֻ�۾�������⵽
		if (leftEye.x > 0 && rightEye.x > 0)
		{
			// �õ����۵�����
			Point2f eyesCenter = Point2f((leftEye.x + rightEye.x) * 0.5f, (leftEye.y + rightEye.y) * 0.5f);
			// �õ����۵ĽǶ�
			double dy = rightEye.y - leftEye.y;
			double dx = rightEye.x - leftEye.x;
			double len = sqrt(dx * dx + dy * dy);
			// ����ת��Ϊ�Ƕ�
			double angle = atan2(dy, dx) * 180 / CV_PI;
			// DESIRED_LEFT_EYE_X ����ѧ���۾���߽���0.16
			const double DESIRED_RIGHT_EYE_X = (1.0f - DESIRED_LEFT_EYE_X);
			// ��ȡ�۾��ĳ���
			double desiredLen = (DESIRED_RIGHT_EYE_X - DESIRED_LEFT_EYE_X) * desiredFaceWidth;
			double scale = desiredLen / len; // ����ϵ��
			// ��ȡת�������Խ�����ת�����ŵ�����ĽǶȺʹ�С��
			Mat rot_mat = getRotationMatrix2D(eyesCenter, angle, scale);
			// ���۾��������ƶ����۾�֮����Ҫ������
			// x�����ƽ�ƾ���
			rot_mat.at<double>(0, 2) += desiredFaceWidth * 0.5f - eyesCenter.x;
			// y�����ƽ�ƾ���
			rot_mat.at<double>(1, 2) += desiredFaceHeight * DESIRED_LEFT_EYE_Y - eyesCenter.y;
		
			Mat warped = Mat(desiredFaceHeight, desiredFaceWidth, CV_8UC1, Scalar(128));
			warpAffine(gray, warped, rot_mat, warped.size());
			if (doLeftAndRightSeparately)
				equalizeLeftAndRightHalves(warped); // �ֿ���⣬��ֹ�����߹���ǿ�Ȳ�һ����ɵ���������
			else
				equalizeHist(warped, warped); // ֱ�Ӿ��⻯
			
			// ʹ��˫���˾�ͨ��ƽ��ͼ��������������㣬�����ֱ�Ե����
			Mat filtered = Mat(warped.size(), CV_8UC1);
			bilateralFilter(warped, filtered, 0, 20.0, 2.0);

			// ���˵������Ľ��䣬��Ϊ������Ҫֻ�����м䲿�֡�
			Mat mask = Mat(warped.size(), CV_8UC1, Scalar(0));
			Point faceCenter = Point(desiredFaceWidth / 2, cvRound(desiredFaceHeight * FACE_ELLIPSE_CY));
			Size size = Size(cvRound(desiredFaceWidth * FACE_ELLIPSE_W), cvRound(desiredFaceHeight * FACE_ELLIPSE_H));
			ellipse(mask, faceCenter, size, 0, 0, 360, Scalar(255), CV_FILLED);
			// if(myDebug) imshow("mask", mask);
		
			// ʹ�����룬ɾ���ⲿͼ��
			Mat dstImg = Mat(warped.size(), CV_8UC1, Scalar(128));
			filtered.copyTo(dstImg, mask);
			return dstImg;
		}
	}
	return Mat();
}

// ͨ��������һ�����һ���и�ǿ�Ĺ⡣ ����������£������ֻ�����������Ͻ���ֱ��ͼ���⣬��ô����ʹһ�밵��һ������ ��ˣ����ǽ���ÿ�����һ��ֱ����ֱ��ͼ���⣬������ǿ�����ƽ�����ơ� 
// ������������м��γ�һ������ı�Ե����Ϊ��벿�ֺ��Ұ벿�ֻ�ͻȻ��ͬ�� ��������Ҳֱ��ͼ����������ͼ�����м䲿�֣����ǽ�3��ͼ������һ����ʵ��ƽ�������ȹ��ɡ�
void equalizeLeftAndRightHalves(Mat &faceImg)
{
	int w = faceImg.cols;
	int h = faceImg.rows;

	// 1. ���⻯����ͼ��
	Mat wholeFace;
	equalizeHist(faceImg, wholeFace);

	// 2. �ֱ���⻯��ֻ�۾�
	int midX = w / 2;
	Mat leftSide = faceImg(Rect(0, 0, midX, h));
	Mat rightSide = faceImg(Rect(midX, 0, w - midX, h));
	equalizeHist(leftSide, leftSide);
	equalizeHist(rightSide, rightSide);

	// 3. ����벿�ֺ��Ұ벿���Լ������沿�����һ��ʹ�����ƽ������
	for (int y = 0; y < h; y++)
	{
		for (int x = 0; x < w; x++)
		{
			int v;
			if (x < w / 4) // ���25%��ֻ��������
				v = leftSide.at<uchar>(y, x);
			else if (x < w * 2 / 4) // ����25%:�����������������
			{
				int lv = leftSide.at<uchar>(y, x);
				int wv = wholeFace.at<uchar>(y, x);
				float f = (x - w * 0.25f) / (float)(w * 0.25f);
				v = cvRound((1.0f - f) * lv + f * wv);
			}
			else if (x < w * 3 / 4) // ����25%:�����������������
			{
				int rv = rightSide.at<uchar>(y, x - midX);
				int wv = wholeFace.at<uchar>(y, x);
				float f = (x - w * 0.5f) / (float)(w * 0.25f);
				v = cvRound((1.0f - f) * wv + f * rv);
			}
			else // �ұ�25%��ֻʹ������
				v = rightSide.at<uchar>(y, x - midX);

			faceImg.at<uchar>(y, x) = v;
		}
	}
}

void detectBothEyes(const Mat &face, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2, \
	Point &leftEye, Point &rightEye, Rect *searchedLeftEye , Rect *searchedRightEye)
{
	if (myDebug) printf("[myDEBUG][%s][%d], ��ʼ����۾�...\n", __FILE__, __LINE__);

	const float EYE_SX = 0.16f;
	const float EYE_SY = 0.26f;
	const float EYE_SW = 0.30f;
	const float EYE_SH = 0.28f;

	int leftX = cvRound(face.cols * EYE_SX);
	int topY = cvRound(face.cols * EYE_SY);
	int widthX = cvRound(face.cols * EYE_SW);
	int heightY = cvRound(face.rows * EYE_SH);
	int rightX = cvRound(face.cols * (1.0 - EYE_SX - EYE_SW));

	// ����ROI
	Mat topleftOfFace = face(Rect(leftX, topY, widthX, heightY));
	// ����ROI
	Mat toprightOfFace = face(Rect(rightX, topY, widthX, heightY));

	if (searchedLeftEye)
		*searchedLeftEye = Rect(leftX, topY, widthX, heightY);
	if (searchedRightEye)
		*searchedRightEye = Rect(rightX, topY, widthX, heightY);

	// �ֱ�������ۺ����۾��ο�
	Rect leftEyeRect, rightEyeRect;
	detectLargestObject(topleftOfFace, eyeCascade1, leftEyeRect, topleftOfFace.cols);
	detectLargestObject(toprightOfFace, eyeCascade1, rightEyeRect, toprightOfFace.cols);

	// ���û�м�������һ��������ļ�
	if (leftEyeRect.width <= 0 && !eyeCascade2.empty())
		detectLargestObject(topleftOfFace, eyeCascade2, leftEyeRect, topleftOfFace.cols);
	if(rightEyeRect.width <= 0 && !eyeCascade2.empty())
		detectLargestObject(toprightOfFace, eyeCascade2, rightEyeRect, toprightOfFace.cols);


	// �����⵽���۾�����õ��۾��ķ����Լ��۾�����������
	if (leftEyeRect.width > 0)
	{
		leftEyeRect.x += leftX;
		leftEyeRect.y += topY;
		// ������������
		leftEye = Point(leftEyeRect.x + leftEyeRect.width / 2, leftEyeRect.y + leftEyeRect.height / 2);
	}
	else
		leftEye = Point(-1,-1);
	if (rightEyeRect.width > 0)
	{
		rightEyeRect.x += rightX;
		rightEyeRect.y += topY;
		// ������������
		rightEye = Point(rightEyeRect.x + rightEyeRect.width / 2, rightEyeRect.y + rightEyeRect.height / 2);
	}
	else
		rightEye = Point(-1,-1);

	if (myDebug) printf("[myDEBUG][%s][%d], ����۾�����...\n", __FILE__, __LINE__);
}