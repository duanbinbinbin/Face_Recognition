#include "preprocessFace.h"
#include "detectObject.h"

bool myDebug = true;
const double DESIRED_LEFT_EYE_X = 0.16; // 左眼距离边框
const double DESIRED_LEFT_EYE_Y = 0.14; 
const double FACE_ELLIPSE_CY = 0.40;
const double FACE_ELLIPSE_W = 0.50;
const double FACE_ELLIPSE_H = 0.80;

// 创建一个具有标准大小、对比度和亮度的灰度人脸图像。
Mat getProcessedFace(Mat& srcImg, int desiredFaceWidth, CascadeClassifier& faceCascade, \
	CascadeClassifier& eyeCascade1, CascadeClassifier &eyeCascade2, bool doLeftAndRightSeparately, \
	Rect* storeFaceRect, Point* storeLeftEye, Point* storeRightEye, Rect* searchedLeftEye, Rect* searchedRightEye)
{
	// 脸的的高度，宽度 = 70；
	int desiredFaceHeight = desiredFaceWidth;

	// 初始化，将检测到的人脸区域和眼睛搜索区域标记为无效，以防它们没有被检测到。
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
	// 级联检测出人脸，得到人脸矩形方框
	detectLargestObject(srcImg, faceCascade, faceRect);
	//if (myDebug) printf("[myDebug] faceRect.width = %d\n", faceRect.width); // 0

	// 若检测到人脸则检测眼睛
	if (faceRect.width > 0)
	{
		if (storeFaceRect)
			*storeFaceRect = faceRect;

		// 获取人脸ROI
		Mat faceImg = srcImg(faceRect);

		// 将人脸ROI转化为灰度图像
		Mat gray;
		if (faceImg.channels() == 3)
			cvtColor(faceImg, gray, COLOR_BGR2GRAY);
		else if (faceImg.channels() == 4)
			cvtColor(faceImg, gray, COLOR_BGRA2GRAY);
		else
			gray = faceImg;

		// 眼见级联检测
		Point leftEye, rightEye; // 眼睛坐标中心
		detectBothEyes(gray, eyeCascade1, eyeCascade2, leftEye, rightEye, searchedLeftEye, searchedRightEye);
		if (storeLeftEye)
			*storeLeftEye = leftEye;
		if (storeRightEye)
		{
			*storeRightEye = rightEye;
			if (myDebug)
				printf("[DEBUG] 眼睛矩形以及中心坐标获取成功\n");
		}

		// 如果两只眼睛都被检测到
		if (leftEye.x > 0 && rightEye.x > 0)
		{
			// 得到两眼的中心
			Point2f eyesCenter = Point2f((leftEye.x + rightEye.x) * 0.5f, (leftEye.y + rightEye.y) * 0.5f);
			// 得到两眼的角度
			double dy = rightEye.y - leftEye.y;
			double dx = rightEye.x - leftEye.x;
			double len = sqrt(dx * dx + dy * dy);
			// 弧度转化为角度
			double angle = atan2(dy, dx) * 180 / CV_PI;
			// DESIRED_LEFT_EYE_X 生物学左眼距离边界在0.16
			const double DESIRED_RIGHT_EYE_X = (1.0f - DESIRED_LEFT_EYE_X);
			// 获取眼睛的长度
			double desiredLen = (DESIRED_RIGHT_EYE_X - DESIRED_LEFT_EYE_X) * desiredFaceWidth;
			double scale = desiredLen / len; // 缩放系数
			// 获取转换矩阵，以将面旋转并缩放到所需的角度和大小。
			Mat rot_mat = getRotationMatrix2D(eyesCenter, angle, scale);
			// 将眼睛的中心移动到眼睛之间想要的中心
			// x方向的平移距离
			rot_mat.at<double>(0, 2) += desiredFaceWidth * 0.5f - eyesCenter.x;
			// y方向的平移距离
			rot_mat.at<double>(1, 2) += desiredFaceHeight * DESIRED_LEFT_EYE_Y - eyesCenter.y;
		
			Mat warped = Mat(desiredFaceHeight, desiredFaceWidth, CV_8UC1, Scalar(128));
			warpAffine(gray, warped, rot_mat, warped.size());
			if (doLeftAndRightSeparately)
				equalizeLeftAndRightHalves(warped); // 分开检测，防止脸两边光照强度不一样造成的明暗差异
			else
				equalizeHist(warped, warped); // 直接均衡化
			
			// 使用双边滤镜通过平滑图像来减少像素噪点，但保持边缘锐利
			Mat filtered = Mat(warped.size(), CV_8UC1);
			bilateralFilter(warped, filtered, 0, 20.0, 2.0);

			// 过滤掉脸部的角落，因为我们主要只关心中间部分。
			Mat mask = Mat(warped.size(), CV_8UC1, Scalar(0));
			Point faceCenter = Point(desiredFaceWidth / 2, cvRound(desiredFaceHeight * FACE_ELLIPSE_CY));
			Size size = Size(cvRound(desiredFaceWidth * FACE_ELLIPSE_W), cvRound(desiredFaceHeight * FACE_ELLIPSE_H));
			ellipse(mask, faceCenter, size, 0, 0, 360, Scalar(255), CV_FILLED);
			// if(myDebug) imshow("mask", mask);
		
			// 使用掩码，删除外部图像
			Mat dstImg = Mat(warped.size(), CV_8UC1, Scalar(128));
			filtered.copyTo(dstImg, mask);
			return dstImg;
		}
	}
	return Mat();
}

// 通常，脸的一半比另一半有更强的光。 在这种情况下，如果你只是在整个脸上进行直方图均衡，那么它将使一半暗和一半亮。 因此，我们将在每个面的一半分别进行直方图均衡，因此它们看起来平均相似。 
// 但这会在脸部中间形成一个尖锐的边缘，因为左半部分和右半部分会突然不同。 所以我们也直方图均衡了整体图像，在中间部分，我们将3个图像混合在一起，以实现平滑的亮度过渡。
void equalizeLeftAndRightHalves(Mat &faceImg)
{
	int w = faceImg.cols;
	int h = faceImg.rows;

	// 1. 均衡化整个图像
	Mat wholeFace;
	equalizeHist(faceImg, wholeFace);

	// 2. 分别均衡化两只眼睛
	int midX = w / 2;
	Mat leftSide = faceImg(Rect(0, 0, midX, h));
	Mat rightSide = faceImg(Rect(midX, 0, w - midX, h));
	equalizeHist(leftSide, leftSide);
	equalizeHist(rightSide, rightSide);

	// 3. 将左半部分和右半部分以及整个面部组合在一起，使其具有平滑过渡
	for (int y = 0; y < h; y++)
	{
		for (int x = 0; x < w; x++)
		{
			int v;
			if (x < w / 4) // 左边25%，只是用左眼
				v = leftSide.at<uchar>(y, x);
			else if (x < w * 2 / 4) // 中左25%:混合左脸和整个脸。
			{
				int lv = leftSide.at<uchar>(y, x);
				int wv = wholeFace.at<uchar>(y, x);
				float f = (x - w * 0.25f) / (float)(w * 0.25f);
				v = cvRound((1.0f - f) * lv + f * wv);
			}
			else if (x < w * 3 / 4) // 中右25%:混合右脸和整个脸。
			{
				int rv = rightSide.at<uchar>(y, x - midX);
				int wv = wholeFace.at<uchar>(y, x);
				float f = (x - w * 0.5f) / (float)(w * 0.25f);
				v = cvRound((1.0f - f) * wv + f * rv);
			}
			else // 右边25%，只使用右眼
				v = rightSide.at<uchar>(y, x - midX);

			faceImg.at<uchar>(y, x) = v;
		}
	}
}

void detectBothEyes(const Mat &face, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2, \
	Point &leftEye, Point &rightEye, Rect *searchedLeftEye , Rect *searchedRightEye)
{
	if (myDebug) printf("[myDEBUG][%s][%d], 开始检测眼睛...\n", __FILE__, __LINE__);

	const float EYE_SX = 0.16f;
	const float EYE_SY = 0.26f;
	const float EYE_SW = 0.30f;
	const float EYE_SH = 0.28f;

	int leftX = cvRound(face.cols * EYE_SX);
	int topY = cvRound(face.cols * EYE_SY);
	int widthX = cvRound(face.cols * EYE_SW);
	int heightY = cvRound(face.rows * EYE_SH);
	int rightX = cvRound(face.cols * (1.0 - EYE_SX - EYE_SW));

	// 左眼ROI
	Mat topleftOfFace = face(Rect(leftX, topY, widthX, heightY));
	// 右眼ROI
	Mat toprightOfFace = face(Rect(rightX, topY, widthX, heightY));

	if (searchedLeftEye)
		*searchedLeftEye = Rect(leftX, topY, widthX, heightY);
	if (searchedRightEye)
		*searchedRightEye = Rect(rightX, topY, widthX, heightY);

	// 分别检测出左眼和右眼矩形框
	Rect leftEyeRect, rightEyeRect;
	detectLargestObject(topleftOfFace, eyeCascade1, leftEyeRect, topleftOfFace.cols);
	detectLargestObject(toprightOfFace, eyeCascade1, rightEyeRect, toprightOfFace.cols);

	// 如果没有检测出，则换一个检测器文件
	if (leftEyeRect.width <= 0 && !eyeCascade2.empty())
		detectLargestObject(topleftOfFace, eyeCascade2, leftEyeRect, topleftOfFace.cols);
	if(rightEyeRect.width <= 0 && !eyeCascade2.empty())
		detectLargestObject(toprightOfFace, eyeCascade2, rightEyeRect, toprightOfFace.cols);


	// 如果检测到了眼睛，则得到眼睛的方框，以及眼睛的中心坐标
	if (leftEyeRect.width > 0)
	{
		leftEyeRect.x += leftX;
		leftEyeRect.y += topY;
		// 左眼中心坐标
		leftEye = Point(leftEyeRect.x + leftEyeRect.width / 2, leftEyeRect.y + leftEyeRect.height / 2);
	}
	else
		leftEye = Point(-1,-1);
	if (rightEyeRect.width > 0)
	{
		rightEyeRect.x += rightX;
		rightEyeRect.y += topY;
		// 右眼中心坐标
		rightEye = Point(rightEyeRect.x + rightEyeRect.width / 2, rightEyeRect.y + rightEyeRect.height / 2);
	}
	else
		rightEye = Point(-1,-1);

	if (myDebug) printf("[myDEBUG][%s][%d], 检测眼睛结束...\n", __FILE__, __LINE__);
}