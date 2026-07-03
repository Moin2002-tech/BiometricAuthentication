//
// Created by moinshaikh on 6/30/26.
//

#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<doctest.hpp>

TEST_CASE("CameraTest")
{
    // 1. Open the default built-in camera (index 0)
    cv::VideoCapture cap(0);

    // Check if the camera opened successfully
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open the camera feed." << std::endl;

    }

    // Create a container window for the video feed
    const std::string windowName = "Live Camera Feed";
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

    // 2. Initialize a Matrix to store individual frames
    cv::Mat frame;

    std::cout << "Press 'q' or 'ESC' to exit the stream." << std::endl;

    // 3. Continuously capture and display frames
    while (true) {
        // Read a new frame from the camera
        cap >> frame; // Alternative syntax: cap.read(frame);

        // Verify if the frame contains valid image data
        if (frame.empty()) {
            std::cerr << "Error: Captured an empty frame." << std::endl;
            break;
        }

        // 4. Display the frame in the created window
        cv::imshow(windowName, frame);

        // 5. Wait for 30 milliseconds and monitor for exit keys
        char key = (char)cv::waitKey(30);
        if (key == 'q' || key == 27) { // 'q' or ESC key
            break;
        }
    }

    // 6. Release resources and close windows
    cap.release();
    cv::destroyAllWindows();
}