#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
using namespace std;
using namespace cv;
using namespace cv::face;

const char* faceCascadeFilename = "lbpcascade_frontalface.xml";
const char* eyeCascadeFilename1 = "haarcascade_eye.xml";
const char* eyeCascadeFilename2 = "haarcascade_eye_tree_eyeglasses.xml";

const char* windowName = "WebcamFaceRec";

// 尝试设置相机分辨率。 请注意，这仅适用于某些计算机上的某些相机，仅适用于某些驱动程序，所以不要依赖它来工作！
const int DESIRED_CAMERA_WIDTH = 640;
const int DESIRED_CAMERA_HEIGHT = 480;

// 设置面部尺寸
const int faceWidth = 70;
const int faceHeight = faceWidth;

// 在窗口的位置
Rect m_rcBtnAdd;
Rect m_rcBtnDel;
Rect m_rcBtnDebug;
int m_gui_faces_left = -1;
int m_gui_faces_top = -1;

// 图像的下标
int m_selectedPerson = -1;
// 人脸的数量
int m_numPersons = 0;
// 最新的图像，但不是镜像
vector<int> m_latestFaces;

enum MODES {
	MODE_STARTUP = 0,
	MODE_DETECTION,
	MODE_COLLECT_FACES,
	MODE_TRAINING,
	MODE_RECOGNITION,
	MODE_DELETE_ALL,
	MODE_END
};
MODES m_mode = MODE_STARTUP;
const char* MODE_NAMES[] = {"Startup", "Detection", "Collect Faces", "Training", "Recognition", "Delete All", "ERROR!"};

// 如果要查看创建的许多窗口，则设置为true，显示各种调试信息。 否则设为false
bool m_debug = false;

// 加载脸部和两只眼睛的级联检测器的XML
void initDetectors(CascadeClassifier& faceCascade, CascadeClassifier& eyeCascade1, CascadeClassifier& eyeCascade2)
{
	faceCascade.load(faceCascadeFilename);
	if (faceCascade.empty())
	{
		printf("ERROR: Could not load Face Detection cascade classifier %s.\n", faceCascadeFilename);
		exit(1);
	}

	faceCascade.load(eyeCascadeFilename1);
	if (faceCascade.empty())
	{
		printf("ERROR: Could not load Face Detection cascade classifier %s.\n", eyeCascadeFilename1);
		exit(1);
	}

	faceCascade.load(eyeCascadeFilename2);
	if (faceCascade.empty())
	{
		printf("ERROR: Could not load Face Detection cascade classifier %s.\n", eyeCascadeFilename2);
		exit(1);
	}
}

// 初始化摄像头
void initWebcam(VideoCapture& videoCapture, int cameraNumber)
{
	videoCapture.open(cameraNumber);
	if (!videoCapture.isOpened())
	{
		printf("ERROR: Could not access the camera!\n");
		exit(1);
	}
	printf("Loaded camera %d.\n", cameraNumber);
}

bool isPointInRect(const Point pt, const Rect rc)
{
	if (pt.x >= rc.x && pt.x <= (rc.x + rc.width - 1))
		if (pt.y >= rc.y && pt.y <= (rc.y + rc.height - 1))
			return true;

	return false;
}

void onMouse(int event, int x, int y, int, void*)
{
	if (event != EVENT_LBUTTONDOWN)
		return;

	Point pt = Point(x, y);
	if (isPointInRect(pt, m_rcBtnAdd))
	{
		printf("user clicked [Add Persion] button when numPersons was %d.\n", m_numPersons);
		if ((m_numPersons == 0) && m_latestFaces[m_numPersons - 1] == 0)
		{
			m_numPersons++;
			m_latestFaces.push_back(-1);// 分配空间
			printf("Num Persons: %d\n", m_numPersons);
		}
		m_selectedPerson = m_numPersons - 1; // 当前人的下标
		m_mode = MODE_COLLECT_FACES;
	}
	else if (isPointInRect(pt, m_rcBtnDel))
	{
		printf("User clicked [Delete ALl] button.\n");
		m_mode = MODE_DELETE_ALL;
	}
	else if (isPointInRect(pt, m_rcBtnDebug))
	{
		printf("User clicked [Debug] button.\n");
		m_debug = !m_debug;
	}
	else
	{
		printf("User clicked on the image.\n");
		// 检查用户是否单击了面部列表
		int clickedPerson = -1;
		for (int i = 0; i < m_numPersons; i++)
		{
			if (m_gui_faces_top >= 0)
			{
				Rect rcFace = Rect(m_gui_faces_left, m_gui_faces_top + i * faceHeight, faceWidth, faceHeight);
				if (isPointInRect(pt, rcFace))
				{
					clickedPerson = i;
					break;
				}
			}
		}
		// 如果单击了头像
		if (clickedPerson >= 0)
		{
			m_selectedPerson = clickedPerson;
			m_mode = MODE_COLLECT_FACES;
		}
		else
		{	// 如果是收集面部，则更改为训练模式。
			if (m_mode == MODE_COLLECT_FACES)
			{
				printf("User wants to begin training.\n");
				m_mode = MODE_TRAINING;
			}
		}
	}
}

void recognizeAndTrainUsingWebcam(VideoCapture& videoCapture, CascadeClassifier& faceCascade, CascadeClassifier& eyeCascade1, CascadeClassifier& eyeCascade2)
{
	Ptr<FaceRecognizer> model;
	vector<Mat> preprocessedFaces;
	vector<int> faceLabels;
	Mat old_prepreprocessedFace;
	double old_time = 0;

	m_mode = MODE_DETECTION;

	while (true)
	{
		// 扑捉下一个相机框架。注意，不能修改相机帧。
		Mat cameraFrame;
		videoCapture >> cameraFrame;
		if (cameraFrame.empty())
		{
			printf("ERROR: Couldn't grab the next camera frame.\n");
			exit(1);
		}

		// 获取我们可以绘制的相机框架的副本
		Mat displayedFrame;
		cameraFrame.copyTo(displayedFrame);

		// 对相机图像运行人脸识别系统。它会在给定的图像上绘制一些东西，所以要确保它不是只读内存
		int identity = -1;

		// 找一张脸，对它进行预处理，使其具有标准的尺寸、对比度和亮度。
		Rect faceRect; // 检测到脸的位置。
		Rect searchedLeftEye, searchedRightEye; // 脸部的左上和右上区域，也就是眼睛被搜索的区域
		Point leftEye, rightEye; // 检测到的眼睛的位置
	}
}

int main(int argc, char** argv)
{
	CascadeClassifier faceCascade; // face
	CascadeClassifier eyeCascade1;// eye
	CascadeClassifier eyeCascade2;// glass eye
	VideoCapture videoCapture; // 摄像头

	printf("Compiled with OpenCV version %s.\n", CV_VERSION);

	// 加载脸部和两种眼睛的级联检测器的XML
	initDetectors(faceCascade, eyeCascade1, eyeCascade2);

	printf("Hit 'Escape' in the GUI window to quit.\n");

	// 相机设备编号
	int cameraNumber = 0;

	// 初始化摄像头
	initWebcam(videoCapture, cameraNumber);
	videoCapture.set(CAP_PROP_FRAME_WIDTH, DESIRED_CAMERA_WIDTH);
	videoCapture.set(CAP_PROP_FRAME_HEIGHT, DESIRED_CAMERA_HEIGHT);

	namedWindow(windowName, WINDOW_AUTOSIZE);
	setMouseCallback(windowName, onMouse, 0);

	// 人脸识别，直到退出才停止运行该函数。
	recognizeAndTrainUsingWebcam(videoCapture, faceCascade, eyeCascade1, eyeCascade2);

	system("pause");
	waitKey(0);
	return 0;
}