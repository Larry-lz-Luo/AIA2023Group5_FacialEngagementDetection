// AIA2023Group5.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <windows.h>
HWND hwnd;

#include "wtypes.h"
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core.hpp>

#include <utils/images_capture.h>

#define CVUI_IMPLEMENTATION
#include "cvui.h"

#include <fstream>
#include <sstream>
#include <chrono>
#include <ctime> 
#include <time.h>
using namespace cv;

#include "GazeUtils.h"
GazeUtils *gazeUtils;

#include "FaceRecognizerUtils.h"
FaceRecognizerUtils* faceRecognizerUtils;

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

// Get the horizontal and vertical screen sizes in pixel
void GetDesktopResolution(int& horizontal, int& vertical)
{
    RECT desktop;
    // Get a handle to the desktop window
    const HWND hDesktop = GetDesktopWindow();
    // Get the size of screen to the variable desktop
    GetWindowRect(hDesktop, &desktop);
    // The top left corner will have coordinates (0,0)
    // and the bottom right corner will have coordinates
    // (horizontal, vertical)
    horizontal = desktop.right;
    vertical = desktop.bottom;
}

std::string windowName = "AIA2023 Group5 Demo";
std::string sizeString = "1280x720";
cv::Size frameSize = stringToSize(sizeString);

cv::Size downSize = cv::Size(640 / 3, 360 / 3);
cv::Size downSizeVideo = cv::Size(1280 - (640 / 3) - 10, 720 - (360 / 3));
cv::Size reSize = cv::Size(640*1.5 , 360*1.5);

cv::Mat status = cv::Mat(cv::Size(1000, 70), CV_8UC3);
cv::Mat status2 = cv::Mat(cv::Size(1000, 50), CV_8UC3);
std::unique_ptr<ImagesCapture> cap;

std::mutex mu;

int sceneStatus = 0;
cv::Mat cameraFrame;
bool isRunning = false;

void Init() {

    faceRecognizerUtils = new FaceRecognizerUtils();
    gazeUtils = new GazeUtils();

    cap = openImagesCapture("0", false, read_type::efficient, 0, std::numeric_limits<size_t>::max(), frameSize);

    //Alert UI Window
    {
        WNDCLASS wc = {};
        wc.lpfnWndProc = WindowProc;
        wc.hInstance = GetModuleHandle(NULL);
        wc.lpszClassName = L"MyClass";
        RegisterClass(&wc);
        hwnd = CreateWindow(L"MyClass", L"User Name", WS_OVERLAPPEDWINDOW & ~WS_SYSMENU,
            CW_USEDEFAULT, CW_USEDEFAULT, 330, 150,
            NULL, NULL, GetModuleHandle(NULL), NULL);

        HWND hTextBox = CreateWindowEx(WS_EX_CLIENTEDGE, L"EDIT", NULL,
            WS_CHILD | WS_VISIBLE | ES_MULTILINE | ES_AUTOVSCROLL | ES_AUTOHSCROLL,
            10, 20, 200, 30, hwnd, NULL, GetModuleHandle(NULL), NULL);

        HWND hButton = CreateWindow(L"BUTTON", L"Register",
            WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,
            210, 10, 80, 30, hwnd, (HMENU)1, GetModuleHandle(NULL), NULL);

        HWND hButton2 = CreateWindow(L"BUTTON", L"Cancel", WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,
            210, 50, 80, 30, hwnd, (HMENU)2, GetModuleHandle(NULL), NULL);

        std::thread([&]() {
            MSG msg = {};
        while (GetMessage(&msg, NULL, 0, 0))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);

        }
            }).detach();
    }
    
}

void Release() {

    if (gazeUtils)delete gazeUtils;
    if (faceRecognizerUtils)delete faceRecognizerUtils;

}

void updateStatusThread() {

    std::thread([&]() {
        while (sceneStatus == 3 && isRunning)
        {
            std::unique_lock<std::mutex> locker(mu);
            if (!gazeUtils->getResultWithXGBooster()) {
                cv::putText(status, "Not concentrated", cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255), 2); // 在圖像上添加警報文字
            }
            else {
                status.setTo(cv::Scalar(0, 0, 0));
            }

            if (!gazeUtils->getResultWithAngles()) {
                cv::putText(status2, "Not concentrated", cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255, 0, 0), 2); // 在圖像上添加警報文字
            }
            else {
                status2.setTo(cv::Scalar(0, 0, 0));
            }
            locker.unlock();
            Sleep(1000);
        }
        }).detach();
}

void RunScene3() {
    cameraFrame = cap->read();
    sceneStatus = 4;
    ShowWindow(hwnd, SW_SHOWDEFAULT);
    UpdateWindow(hwnd);
}

cv::Mat RunScene1(cv::Mat canvas) {

    cv::Mat frame = cap->read();
    cv::resize(frame, frame, reSize, INTER_LINEAR);

    frame=faceRecognizerUtils->recongnizer(frame, gazeUtils->faceDetector);

    int x = 50;
    int y = (canvas.rows / 2) - (reSize.height / 2);
    cvui::image(canvas, x, y, frame);

    if (cvui::button(canvas, reSize.width+x,y, "LOGIN")) {

        if (faceRecognizerUtils->isMember) {
            status.setTo(cv::Scalar(0, 0, 0));
            cv::putText(status, "Welcome!! " + faceRecognizerUtils->getCurrentMemberName() +" please wait.....", cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 255, 0), 2); // 在圖像上添加警報文字
            sceneStatus = 2;
        }
        else {
            status.setTo(cv::Scalar(0, 0, 0));
            cv::putText(status, "Login Fail" , cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255), 2); // 在圖像上添加警報文字
        }
        
    }

    if (cvui::button(canvas, reSize.width + x, y+30, "SIGN UP")) {

        RunScene3();
    }

    return frame;

}

cv::Mat RunScene2(cv::Mat canvas) {

    cv::Mat frame = cap->read();
    std::unique_lock<std::mutex> locker(mu);
    frame = gazeUtils->checkConcentrated(frame);
    locker.unlock();

    cv::resize(frame, frame, downSize, INTER_LINEAR);
    int x = canvas.cols - downSize.width - 10;
    int y = 10;
    cvui::window(canvas,x , y, downSize.width, downSize.height, "Participant");
    cvui::image(canvas, x, y+20, frame);
    if (cvui::button(canvas, x, y+ downSize.height+30, "LEAVE")) {
        sceneStatus = 0;
    }

    return frame;

}

int main()
{
    std::cout << "AIA2023 Group5 Demo\n";
    Init();

    int horizontal = 0, vertical = 0;
    GetDesktopResolution(horizontal, vertical);
    cvui::init(windowName);
    cv::resizeWindow(windowName, horizontal ,vertical);
    cv::moveWindow(windowName, 0, 0);
    // Create a frame
    cv::Mat canvas = cv::Mat(cv::Size(horizontal, vertical), CV_8UC3);

    isRunning = true;
   
    while (isRunning)
    {
        if (getWindowProperty(windowName, WND_PROP_VISIBLE) < 1)
            isRunning = false;
        
        if (sceneStatus == 0) {
            status.setTo(cv::Scalar(0, 0, 0));
            canvas.setTo(cv::Scalar(0, 0, 0));
            faceRecognizerUtils->loadDB(reSize, gazeUtils->faceDetector);
            sceneStatus = 1;
        }
        else if (sceneStatus==1) {
            
             RunScene1(canvas);
            cvui::image(canvas, 0, 0, status);
        }
        else if (sceneStatus == 2) {
            VideoCapture vid_capture("https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_20mb.mp4");
            // Print error message if the stream is invalid
            if (!vid_capture.isOpened())
            {
                std::cout << "Error opening video stream or file\n";
            }
            else
            {
                status.setTo(cv::Scalar(0, 0, 0));
                status2.setTo(cv::Scalar(0, 0, 0));
                canvas.setTo(cv::Scalar(0, 0, 0));
                std::thread([&](VideoCapture vid_capture) {
                        while (sceneStatus >= 2 && isRunning)
                        {
                            Mat frame;
                            // Initialize a boolean to check if frames are there or not
                            bool isSuccess = vid_capture.read(frame);

                            // If frames are present, show it
                            if (isSuccess == true)
                            {
                                resize(frame, frame, downSizeVideo, INTER_LINEAR);
                                cvui::image(canvas, 0, 0, frame);
                            }
                            else {
                        
                                if (frame.empty()) {
                                    vid_capture.set(cv::CAP_PROP_POS_FRAMES, 0);
                                    continue;
                                }
                            }
                            Sleep(33);
                        }
                        vid_capture.release();

                        canvas.setTo(cv::Scalar(0, 0, 0));
                    }, vid_capture).detach();
                sceneStatus = 3;
                updateStatusThread();
            }
        }
        else if (sceneStatus == 3) {
            cameraFrame = RunScene2(canvas);
            cvui::image(canvas, 0, vertical-200, status);
            cvui::image(canvas, 0, vertical-120, status2);
        }
        else if (sceneStatus ==4) {

            cvui::image(canvas, 0, 0, cameraFrame);
        }

        cvui::update();
        cvui::imshow(windowName,canvas);
        int key=waitKey(1);
        if ( key== 27) {
            isRunning = false;
            Sleep(1000);
            break; 
        }
        else if (key=='s') {
            gazeUtils->switchShowResultMaker();
        }
        else if(key==50){
            break;
        }
        
    }

    Release();
}

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    switch (uMsg)
    {
    case WM_COMMAND:
        if (LOWORD(wParam) == 1)
        {
            TCHAR buffer[1024];
            GetWindowText(GetDlgItem(hwnd, 0), buffer, sizeof(buffer) / sizeof(buffer[0]));

            OutputDebugString(buffer);
            OutputDebugString(L"\n");
            std::wstring ws(buffer);
            std::string name(ws.begin(), ws.end());

            if (name.size() > 0) {

                bool result = faceRecognizerUtils->saveToDB( name, cameraFrame);

                if (!result)
                {
                    std::cerr << "Failed to save image!" << std::endl;
                }

                ShowWindow(hwnd, SW_HIDE);
                UpdateWindow(hwnd);

                sceneStatus = 0;
            }
        }
        else if (LOWORD(wParam) == 2)
        {
            OutputDebugString(L"cancel");
            OutputDebugString(L"\n");
            ShowWindow(hwnd, SW_HIDE);
            UpdateWindow(hwnd);
            sceneStatus = 0;
        }
        break;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;

    default:
        return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }
}
