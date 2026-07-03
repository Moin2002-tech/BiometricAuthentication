//
// Created by moinshaikh on 7/1/26.
//

#pragma once

#ifndef BIOMETRICAUTHENTICATION_CAMERAVIEWER_HPP
#define BIOMETRICAUTHENTICATION_CAMERAVIEWER_HPP

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <string_view>
#include <string>
#include <functional>

namespace Camera {

    /**
     * @brief Abstract interface for recognition processing.
     *
     * Users derive from this class and override onFrameCaptured()
     * to provide custom recognition logic.
     */
    class IRecognitionCallback
    {
    public:
        virtual ~IRecognitionCallback() = default;

        /**
         * @brief Called when a frame is captured for recognition.
         * @param frame  The captured frame (cloned copy, safe to keep).
         */
        virtual void onFrameCaptured(const cv::Mat& frame) = 0;
    };

    /**
     * @brief CameraViewer provides a self-contained interface to display a camera feed
     *        and capture frames for recognition processing.
     *
     * Manages cv::VideoCapture internally. The run() method opens a window
     * showing the live feed and supports:
     *   - 'q' or ESC: quit
     *   - SPACE: capture the current frame and pass it to the recognition callback
     *
     * Usage:
     *   CameraViewer viewer("Face Recognition", 0);
     *   MyRecognizer recognizer;
     *   viewer.setRecognitionCallback(&recognizer);
     *   viewer.run();
     */
    class CameraViewer
    {
    public:
        /**
         * @brief Construct a CameraViewer and open the camera device.
         * @param windowName  Name of the display window.
         * @param device      Camera device index (default: 0).
         * @throws std::runtime_error if the camera cannot be opened.
         */
        explicit CameraViewer(std::string_view windowName = "Camera", int device = 0);

        /// Default destructor releases the capture and closes windows.
        ~CameraViewer();

        // Non-copyable, non-movable
        CameraViewer(const CameraViewer&) = delete;
        CameraViewer& operator=(const CameraViewer&) = delete;

        /**
         * @brief Set the callback invoked when a frame is captured for recognition.
         * @param callback  Pointer to an IRecognitionCallback instance. Pass nullptr to clear.
         *
         * The callback object must remain valid while CameraViewer is using it.
         */
        void setRecognitionCallback(IRecognitionCallback* callback);

        /**
         * @brief Run the main display loop.
         *
         * Reads frames from the camera, shows them in a named window, and
         * responds to key presses for capture and exit.
         */
        void run();

        /**
         * @brief Check if the camera is currently opened.
         */
        bool isOpened() const { return capture_.isOpened(); }

        /**
         * @brief Get the current frame without displaying.
         * @return true if a frame was successfully read.
         */
        bool readFrame(cv::Mat& frame) { return capture_.read(frame); }

        /**
         * @brief Set a function to draw overlays on each frame before display.
         * @param overlayFunc  Function that takes (frame&) and draws on it.
         */
        void setOverlayDrawer(std::function<void(cv::Mat&)> overlayFunc);

    private:
        std::string windowName_;
        cv::VideoCapture capture_;
        int device_ = 0;
        IRecognitionCallback* recognitionCallback_ = nullptr;
        std::function<void(cv::Mat&)> overlayDrawer_ = nullptr;
    };

} // namespace Camera

#endif //BIOMETRICAUTHENTICATION_CAMERAVIEWER_HPP