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
#include <vector>
#include <opencv2/core.hpp>

/**
 * @brief FaceRecognitionApp integrates camera, face detection, embedding extraction,
 *        and database into a complete face recognition application.
 *
 * Features:
 *   - Real-time multi-face detection with per-person bounding box / name overlay
 *   - Recognition of ALL detected faces, showing each person's name
 *   - Face selection for "add person": cycle with arrow keys or click
 *   - Manual ID entry for better authorization control
 *   - Press 'a' to add a NEW person to database (choose face + enter ID + name)
 *   - Press 'b' to add an ADDITIONAL face angle to an EXISTING person (type their ID)
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

    // --- Multi-face recognition ---
    std::vector<cv::Rect> detectedFaceRects_;      // All detected face bounding boxes
    std::vector<std::string> detectedFaceNames_;    // Recognized name per face (empty if unknown)
    std::vector<std::vector<float>> detectedFaceEmbeddings_; // Embedding per face

    // Selected face index for "add person" flow (when multiple faces)
    int selectedFaceIndex_ = -1;
    bool faceSelectionMode_ = false;

    // GUI input state for adding person (with manual ID)
    bool addingPerson_ = false;
    bool inNameInputMode_ = false;
    bool inputtingId_ = true;  // true = entering ID, false = entering name
    std::string idInput_;
    std::string nameInput_;
    cv::Rect addPersonFaceRect_;
    cv::Mat addPersonFaceCrop_;
    std::vector<float> addPersonEmbedding_;

    // Reset confirmation state
    bool resetConfirmPending_ = false;

    /**
     * @brief Run multi-face detection and recognition on a frame.
     * Fills detectedFaceRects_, detectedFaceNames_, detectedFaceEmbeddings_.
     */
    void recognizeFrame(const cv::Mat& frame);

    /**
     * @brief Get the name for a single embedding by matching against the database.
     * @return Recognized name, "Unknown" if no match, or empty if no embedding.
     */
    std::string getNameForEmbedding(const std::vector<float>& embedding);

    /**
     * @brief Start the "add person" flow with face selection.
     * If multiple faces are present, enters face selection mode first.
     */
    void startAddPerson(const cv::Mat& frame);

    /**
     * @brief Confirm adding the person with the typed ID and name.
     */
    void confirmAddPerson();

    /**
     * @brief Draw face bounding boxes with per-face names.
     */
    void drawFaceOverlays(cv::Mat& frame, int frameCount);

    /**
     * @brief Draw the name/ID input overlay.
     */
    void drawInputOverlay(cv::Mat& frame, int frameCount);

    /**
     * @brief Draw face selection overlay when multiple faces are present.
     */
    void drawSelectionOverlay(cv::Mat& frame, int frameCount);

    /**
     * @brief Draw the reset confirmation overlay.
     */
    void drawResetOverlay(cv::Mat& frame);

    /**
     * @brief Reset the entire face database after user confirmation.
     */
    void resetDatabase();

    /**
     * @brief Cycle to the next/previous face selection.
     */
    void cycleSelection(int direction);
};

#endif //BIOMETRICAUTHENTICATION_FACERECOGNITIONAPP_HPP
