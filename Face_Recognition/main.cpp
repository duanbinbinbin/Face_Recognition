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

// ������������ֱ��ʡ� ��ע�⣬���������ĳЩ������ϵ�ĳЩ�������������ĳЩ�����������Բ�Ҫ��������������
const int DESIRED_CAMERA_WIDTH = 640;
const int DESIRED_CAMERA_HEIGHT = 480;

// �����沿�ߴ�
const int faceWidth = 70;
const int faceHeight = faceWidth;

// �ڴ��ڵ�λ��
Rect m_rcBtnAdd;
Rect m_rcBtnDel;
Rect m_rcBtnDebug;
int m_gui_faces_left = -1;
int m_gui_faces_top = -1;

// ͼ����±�
int m_selectedPerson = -1;
// ����������
int m_numPersons = 0;
// ���µ�ͼ�񣬵����Ǿ���
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

// ���Ҫ�鿴��������ര�ڣ�������Ϊtrue����ʾ���ֵ�����Ϣ�� ������Ϊfalse
bool m_debug = false;

// ������������ֻ�۾��ļ����������XML
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

// ��ʼ������ͷ
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
			m_latestFaces.push_back(-1);// ����ռ�
			printf("Num Persons: %d\n", m_numPersons);
		}
		m_selectedPerson = m_numPersons - 1; // ��ǰ�˵��±�
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
		// ����û��Ƿ񵥻����沿�б�
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
		// ���������ͷ��
		if (clickedPerson >= 0)
		{
			m_selectedPerson = clickedPerson;
			m_mode = MODE_COLLECT_FACES;
		}
		else
		{	// ������ռ��沿�������Ϊѵ��ģʽ��
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
		// ��׽��һ�������ܡ�ע�⣬�����޸����֡��
		Mat cameraFrame;
		videoCapture >> cameraFrame;
		if (cameraFrame.empty())
		{
			printf("ERROR: Couldn't grab the next camera frame.\n");
			exit(1);
		}

		// ��ȡ���ǿ��Ի��Ƶ������ܵĸ���
		Mat displayedFrame;
		cameraFrame.copyTo(displayedFrame);

		// �����ͼ����������ʶ��ϵͳ�������ڸ�����ͼ���ϻ���һЩ����������Ҫȷ��������ֻ���ڴ�
		int identity = -1;

		// ��һ��������������Ԥ����ʹ����б�׼�ĳߴ硢�ԱȶȺ����ȡ�
		Rect faceRect; // ��⵽����λ�á�
		Rect searchedLeftEye, searchedRightEye; // ���������Ϻ���������Ҳ�����۾�������������
		Point leftEye, rightEye; // ��⵽���۾���λ��
	}
}

int main(int argc, char** argv)
{
	CascadeClassifier faceCascade; // face
	CascadeClassifier eyeCascade1;// eye
	CascadeClassifier eyeCascade2;// glass eye
	VideoCapture videoCapture; // ����ͷ

	printf("Compiled with OpenCV version %s.\n", CV_VERSION);

	// ���������������۾��ļ����������XML
	initDetectors(faceCascade, eyeCascade1, eyeCascade2);

	printf("Hit 'Escape' in the GUI window to quit.\n");

	// ����豸���
	int cameraNumber = 0;

	// ��ʼ������ͷ
	initWebcam(videoCapture, cameraNumber);
	videoCapture.set(CAP_PROP_FRAME_WIDTH, DESIRED_CAMERA_WIDTH);
	videoCapture.set(CAP_PROP_FRAME_HEIGHT, DESIRED_CAMERA_HEIGHT);

	namedWindow(windowName, WINDOW_AUTOSIZE);
	setMouseCallback(windowName, onMouse, 0);

	// ����ʶ��ֱ���˳���ֹͣ���иú�����
	recognizeAndTrainUsingWebcam(videoCapture, faceCascade, eyeCascade1, eyeCascade2);

	system("pause");
	waitKey(0);
	return 0;
}