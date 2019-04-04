#include "public.h"
#include "preprocessFace.h" 

const char* faceCascadeFilename = "lbpcascade_frontalface.xml";
const char* eyeCascadeFilename1 = "haarcascade_eye.xml";
const char* eyeCascadeFilename2 = "haarcascade_eye_tree_eyeglasses.xml";

const char* windowName = "WebcamFaceRec";
// �ֱ�����������������Ԥ�����Է�һ����߽�ǿ
const bool preprocessLeftAndRightSeparately = true;

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
		//videoCapture.read( cameraFrame );
		if (cameraFrame.empty())
		{
			printf("ERROR: Couldn't grab the next camera frame.\n");
			exit(1);
		}
		
		// ��ȡ���ǿ��Ի��Ƶ������ܵĸ���
		Mat displayedFrame;
		cameraFrame.copyTo(displayedFrame);
		//if (myDebug) printf("[myDebug][%s][%d] displayedFrameͼ��Ŀ��Ϊ�� %d\n", __FILE__, __LINE__, displayedFrame.cols);;

		// �����ͼ����������ʶ��ϵͳ�������ڸ�����ͼ���ϻ���һЩ����������Ҫȷ��������ֻ���ڴ�
		int identity = -1;

		// ��һ��������������Ԥ����ʹ����б�׼�ĳߴ硢�ԱȶȺ����ȡ�
		Rect faceRect; // ��⵽����λ�á�
		Rect searchedLeftEye, searchedRightEye; // ���������Ϻ���������Ҳ�����۾�������������
		Point leftEye, rightEye; // ��⵽���۾���λ��
		
		// �õ����⻯֮�����Բͼ��
		Mat preprocessedFace = getProcessedFace(displayedFrame, faceWidth, faceCascade, eyeCascade1, eyeCascade2, \
			preprocessLeftAndRightSeparately, &faceRect, &leftEye, &rightEye, &searchedLeftEye, &searchedRightEye);
	
		bool gotFaceAndEyes = false;
		if (preprocessedFace.data)
			gotFaceAndEyes = true;

		// �ڼ�⵽���沿��Χ���ƿ���ݾ��� ���һ����۾���Բ
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
				printf("[myDebug][%s][%d] ***********û�м�⵽����***********\n", __FILE__, __LINE__);
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

	// ����豸���
	int cameraNumber = 0;
	VideoCapture videoCapture(cameraNumber); // ����ͷ

	printf("Compiled with OpenCV version %s.\n", CV_VERSION);

	// ���������������۾��ļ����������XML
	initDetectors(faceCascade, eyeCascade1, eyeCascade2);

	printf("Hit 'Escape' in the GUI window to quit.\n");

	// ��ʼ������ͷ
	initWebcam(videoCapture, cameraNumber);
	videoCapture.set(CAP_PROP_FRAME_WIDTH, DESIRED_CAMERA_WIDTH);
	videoCapture.set(CAP_PROP_FRAME_HEIGHT, DESIRED_CAMERA_HEIGHT);

	namedWindow(windowName, WINDOW_AUTOSIZE);
	setMouseCallback(windowName, onMouse, 0);

	if (myDebug) printf("[myDebug][%s][%d] ��ʼ����ʶ��������...\n", __FILE__, __LINE__);

	// ����ʶ��ֱ���˳���ֹͣ���иú�����
	recognizeAndTrainUsingWebcam(videoCapture, faceCascade, eyeCascade1, eyeCascade2);

	if (myDebug) printf("[myDebug][%s][%d] ����ʶ���������˳�...\n", __FILE__, __LINE__);

	system("pause");
	return 0;
}