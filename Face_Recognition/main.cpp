#include "public.h"
#include "preprocessFace.h" 

const char* faceCascadeFilename = "lbpcascade_frontalface.xml";
const char* eyeCascadeFilename1 = "haarcascade_eye.xml";
const char* eyeCascadeFilename2 = "haarcascade_eye_tree_eyeglasses.xml";

const char* windowName = "WebcamFaceRec";
// 分别对脸部左右两侧进行预处理，以防一侧光线较强
const bool preprocessLeftAndRightSeparately = true;

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

	eyeCascade1.load(eyeCascadeFilename1);
	if (eyeCascade1.empty())
	{
		printf("ERROR: Could not load eyeCascade1 Detection cascade classifier %s.\n", eyeCascadeFilename1);
		exit(1);
	}

	eyeCascade2.load(eyeCascadeFilename2);
	if (eyeCascade2.empty())
	{
		printf("ERROR: Could not load eyeCascade2 Detection cascade classifier %s.\n", eyeCascadeFilename2);
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
		//videoCapture.read( cameraFrame );
		if (cameraFrame.empty())
		{
			printf("ERROR: Couldn't grab the next camera frame.\n");
			exit(1);
		}
		
		// 获取我们可以绘制的相机框架的副本
		Mat displayedFrame;
		cameraFrame.copyTo(displayedFrame);
		//if (myDebug) printf("[myDebug][%s][%d] displayedFrame图像的宽度为： %d\n", __FILE__, __LINE__, displayedFrame.cols);;

		// 对相机图像运行人脸识别系统。它会在给定的图像上绘制一些东西，所以要确保它不是只读内存
		int identity = -1;

		// 找一张脸，对它进行预处理，使其具有标准的尺寸、对比度和亮度。
		Rect faceRect; // 检测到脸的位置。
		Rect searchedLeftEye, searchedRightEye; // 脸部的左上和右上区域，也就是眼睛被搜索的区域
		Point leftEye, rightEye; // 检测到的眼睛的位置
		
		// 得到均衡化之后的椭圆图像
		Mat preprocessedFace = getProcessedFace(displayedFrame, faceWidth, faceCascade, eyeCascade1, eyeCascade2, \
			preprocessLeftAndRightSeparately, &faceRect, &leftEye, &rightEye, &searchedLeftEye, &searchedRightEye);
	
		bool gotFaceAndEyes = false;
		if (preprocessedFace.data)
			gotFaceAndEyes = true;

		// 在检测到的面部周围绘制抗锯齿矩形 并且画出眼睛的圆
		if (faceRect.width > 0)
		{
			rectangle(displayedFrame, faceRect, Scalar(255, 255, 0), 2, CV_AA);
			if (leftEye.x >= 0)
				circle(displayedFrame, Point(faceRect.x + leftEye.x, faceRect.y + leftEye.y), 6, Scalar(0, 255, 255), 1, CV_AA);
			if(rightEye.x >= 0)
				circle(displayedFrame, Point(faceRect.x + rightEye.x, faceRect.y + rightEye.y), 6, Scalar(0, 255, 255), 1, CV_AA);
		}
		// duan
		
		

		imshow(windowName, displayedFrame);

		if ( myDebug ) {
			Mat face;
			if (faceRect.width > 0) {
				printf("[myDebug][%s][%d]*****************************\n", __FILE__, __LINE__);
				face = cameraFrame(faceRect);
				if (searchedLeftEye.width > 0 && searchedRightEye.width > 0) {
					Mat topLeftOfFace = face(searchedLeftEye);
					Mat topRightOfFace = face(searchedRightEye);
					imshow("Left", topLeftOfFace);
					imshow("Right", topRightOfFace);
				}
			}
			else
				printf("[myDebug][%s][%d] ***********没有检测到人脸***********\n", __FILE__, __LINE__);
		}

		if (waitKey(20) == 27)
			break;
	}
}

int main(int argc, char** argv)
{
	CascadeClassifier faceCascade; // face
	CascadeClassifier eyeCascade1;// eye
	CascadeClassifier eyeCascade2;// glass eye

	// 相机设备编号
	int cameraNumber = 0;
	VideoCapture videoCapture(cameraNumber); // 摄像头

	printf("Compiled with OpenCV version %s.\n", CV_VERSION);

	// 加载脸部和两种眼睛的级联检测器的XML
	initDetectors(faceCascade, eyeCascade1, eyeCascade2);

	printf("Hit 'Escape' in the GUI window to quit.\n");

	// 初始化摄像头
	initWebcam(videoCapture, cameraNumber);
	videoCapture.set(CAP_PROP_FRAME_WIDTH, DESIRED_CAMERA_WIDTH);
	videoCapture.set(CAP_PROP_FRAME_HEIGHT, DESIRED_CAMERA_HEIGHT);

	namedWindow(windowName, WINDOW_AUTOSIZE);
	setMouseCallback(windowName, onMouse, 0);

	if (myDebug) printf("[myDebug][%s][%d] 开始人脸识别主程序...\n", __FILE__, __LINE__);

	// 人脸识别，直到退出才停止运行该函数。
	recognizeAndTrainUsingWebcam(videoCapture, faceCascade, eyeCascade1, eyeCascade2);

	if (myDebug) printf("[myDebug][%s][%d] 人脸识别主程序退出...\n", __FILE__, __LINE__);

	system("pause");
	return 0;
}