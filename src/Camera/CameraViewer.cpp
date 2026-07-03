//
// Created by moinshaikh on 7/2/26.
//

#include "Camera/CameraViewer.hpp"

#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace Camera {

    CameraViewer::CameraViewer(std::string_view windowName, int device)
        : windowName_(windowName), device_(device)
    {
        capture_.open(device_);
        if (!capture_.isOpened())
        {
            throw std::runtime_error(
                "CameraViewer: Failed to open camera device " + std::to_string(device_));
        }
        // Give the camera time to stabilize (especially V4L2 devices)
        cv::waitKey(500);
    }

    CameraViewer::~CameraViewer()
    {
        if (capture_.isOpened())
        {
            capture_.release();
        }
        cv::destroyWindow(windowName_);
    }

    void CameraViewer::setRecognitionCallback(IRecognitionCallback* callback)
    {
        recognitionCallback_ = callback;
    }

    void CameraViewer::setOverlayDrawer(std::function<void(cv::Mat&)> overlayFunc)
    {
        overlayDrawer_ = std::move(overlayFunc);
    }

    void CameraViewer::run()
    {
        if (!capture_.isOpened())
        {
            throw std::runtime_error("CameraViewer::run(): Camera is not opened");
        }

        cv::namedWindow(windowName_, cv::WINDOW_AUTOSIZE);

        std::cout << "CameraViewer: Press SPACE to capture frame for recognition, "
                  << "'q' or ESC to quit." << std::endl;

        cv::Mat frame;

        while (true)
        {
            if (!capture_.read(frame))
            {
                std::cerr << "CameraViewer: Failed to read frame from camera." << std::endl;
                break;
            }

            if (frame.empty())
            {
                std::cerr << "CameraViewer: Received empty frame." << std::endl;
                continue;
            }

            // Draw overlays if set (e.g. face bounding boxes, labels)
            if (overlayDrawer_)
            {
                overlayDrawer_(frame);
            }

            // Display the frame
            cv::imshow(windowName_, frame);

            // Handle key press
            int key = cv::waitKey(30);

            if (key == 'q' || key == 'Q' || key == 27) // 27 = ESC
            {
                std::cout << "CameraViewer: Exiting." << std::endl;
                break;
            }

            if (key == ' ') // SPACE bar
            {
                std::cout << "CameraViewer: Capturing frame for recognition." << std::endl;

                // Clone the frame so the callback gets an independent copy
                cv::Mat capturedFrame = frame.clone();

                if (recognitionCallback_)
                {
                    recognitionCallback_->onFrameCaptured(capturedFrame);
                }
                else
                {
                    std::cout << "CameraViewer: No recognition callback set. "
                              << "Use setRecognitionCallback() to register one." << std::endl;
                }
            }
        }

        cv::destroyWindow(windowName_);
    }

} // namespace Camera
