//
// Created by moinshaikh on 7/2/26.
//

#pragma once

#ifndef BIOMETRICAUTHENTICATION_FACERECOGNITIONAPP_HPP
#define BIOMETRICAUTHENTICATION_FACERECOGNITIONAPP_HPP

#include "Camera/CameraViewer.hpp"
#include "Recognition/modelsHandler.hpp"
#include "Database/FaceDatabase.hpp"

#include <memory>
#include <string>
#include <opencv2/core.hpp>

/**
 * @brief FaceRecognitionApp integrates camera, face detection, embedding extraction,
 *        and database into a complete face recognition application.
 *
 * Features:
 *   - Real-time face detection with bounding box overlay
 *   - Automatic recognition showing person's name on screen
 *   - Press 'a' to add current face to database with a name (GUI popup)
 *   - Press SPACE to manually trigger recognition on grabbed frame
 *   - Press 'q' to quit
 */
class FaceRecognitionApp : public Camera::IRecognitionCallback {
public:
    FaceRecognitionApp();
    ~FaceRecognitionApp() override = default;

    /**
     * @brief Run the main application loop.
     */
    void run();

    /**
     * @brief IRecognitionCallback: called when SPACE is pressed.
     */
    void onFrameCaptured(const cv::Mat& frame) override;

private:
    std::unique_ptr<Camera::CameraViewer> viewer_;
    std::unique_ptr<Recognition::ModelsHandler> models_;
    std::unique_ptr<Database::FaceDatabase> database_;

    cv::Mat currentFrame_;

    // Recognized person name to display as overlay
    std::string recognizedName_;

    // GUI input state for adding person
    bool addingPerson_ = false;
    bool inNameInputMode_ = false;
    std::string nameInput_;
    cv::Mat addPersonFaceCrop_;
    std::vector<float> addPersonEmbedding_;
    cv::Rect addPersonFaceRect_;

    // Reset confirmation state
    bool resetConfirmPending_ = false;

    /**
     * @brief Run real-time detection and recognition on a frame.
     */
    void recognizeFrame(const cv::Mat& frame);

    /**
     * @brief Start the "add person" flow.
     * First detects the face, then enters name input mode.
     */
    void startAddPerson(const cv::Mat& frame);

    /**
     * @brief Confirm adding the person with the typed name.
     */
    void confirmAddPerson();

    /**
     * @brief Reset the entire face database after user confirmation.
     *
     * Displays an on-screen confirmation prompt. Press 'y' to confirm
     * reset or any other key to cancel.
     */
    void resetDatabase();
};

#endif //BIOMETRICAUTHENTICATION_FACERECOGNITIONAPP_HPP