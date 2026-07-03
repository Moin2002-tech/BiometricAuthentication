//
// Created by moinshaikh on 7/2/26.
//

#include "Recognition/FaceRecognitionApp.hpp"

#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// ASCII key codes
static constexpr int KEY_BACKSPACE = 8;
static constexpr int KEY_ENTER = 13;

FaceRecognitionApp::FaceRecognitionApp()
    : viewer_(nullptr)
    , models_(nullptr)
    , database_(nullptr)
{
    std::cout << "[FaceRecognitionApp] Initializing..." << std::endl;
    models_ = std::make_unique<Recognition::ModelsHandler>();
    database_ = std::make_unique<Database::FaceDatabase>();
    viewer_ = std::make_unique<Camera::CameraViewer>("Face Recognition", 0);
    viewer_->setRecognitionCallback(this);

    std::cout << "[FaceRecognitionApp] Ready. Fully GUI-based." << std::endl;
    std::cout << "  Controls:" << std::endl;
    std::cout << "    'a'    - Add current face to database (type name on screen)" << std::endl;
    std::cout << "    SPACE  - Manual recognize" << std::endl;
    std::cout << "    'r'    - Reset database (with confirmation)" << std::endl;
    std::cout << "    'q'    - Quit" << std::endl;
}

void FaceRecognitionApp::run()
{
    if (!viewer_->isOpened())
    {
        throw std::runtime_error("Camera is not opened");
    }

    const std::string& winName = "Face Recognition";
    cv::namedWindow(winName, cv::WINDOW_AUTOSIZE);

    std::cout << "[FaceRecognitionApp] Running..." << std::endl;

    cv::Mat frame;
    int frameCount = 0;

    // Cached data for overlay (update every N frames)
    std::vector<cv::Rect> lastFaces;
    std::string lastName;

    // Warm up the camera: give it a few retries to stabilize
    int warmupRetries = 30;
    bool cameraReady = false;
    for (int i = 0; i < warmupRetries; ++i)
    {
        if (viewer_->readFrame(frame) && !frame.empty())
        {
            cameraReady = true;
            break;
        }
        std::cout << "[FaceRecognitionApp] Warming up camera... attempt " << (i+1) << "/" << warmupRetries << std::endl;
        cv::waitKey(100);
    }

    if (!cameraReady)
    {
        std::cerr << "[FaceRecognitionApp] Camera failed to stabilize after " << warmupRetries << " attempts." << std::endl;
        cv::destroyWindow(winName);
        return;
    }

    while (true)
    {
        // Retry reading with a few attempts on failure
        bool gotFrame = false;
        for (int retry = 0; retry < 5; ++retry)
        {
            if (viewer_->readFrame(frame) && !frame.empty())
            {
                gotFrame = true;
                break;
            }
            cv::waitKey(50);
        }

        if (!gotFrame)
        {
            std::cerr << "[FaceRecognitionApp] Failed to read frame after retries." << std::endl;
            break;
        }

        currentFrame_ = frame.clone();

        // Run detection + recognition every 10 frames
        if (frameCount % 10 == 0)
        {
            lastName = recognizedName_;
            if (!inNameInputMode_)  // Don't run recognition while typing name
            {
                recognizeFrame(currentFrame_);
                lastFaces = models_->detectFaces(currentFrame_);
            }
        }

        // --- Draw overlays ---

        // 1. Face bounding boxes + names (cached)
        for (const auto& face : lastFaces)
        {
            cv::rectangle(frame, face, cv::Scalar(0, 255, 0), 2);

            std::string displayName = recognizedName_.empty() ? lastName : recognizedName_;
            if (!displayName.empty())
            {
                int baseline = 0;
                cv::Size textSize = cv::getTextSize(displayName, cv::FONT_HERSHEY_SIMPLEX,
                                                     0.6, 2, &baseline);
                cv::rectangle(frame,
                              cv::Point(face.x, face.y - textSize.height - 10),
                              cv::Point(face.x + textSize.width + 5, face.y),
                              cv::Scalar(0, 0, 0, 128), cv::FILLED);
                cv::putText(frame, displayName,
                            cv::Point(face.x + 3, face.y - 5),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6,
                            cv::Scalar(0, 255, 0), 2);
            }
        }

        // 4. Reset confirmation overlay (when reset is pending)
        if (resetConfirmPending_)
        {
            cv::Mat overlay = frame.clone();

            int boxX = frame.cols / 2 - 200, boxY = frame.rows / 2 - 60;
            int boxW = 400, boxH = 120;
            cv::rectangle(overlay,
                          cv::Point(boxX, boxY),
                          cv::Point(boxX + boxW, boxY + boxH),
                          cv::Scalar(0, 0, 128), cv::FILLED);

            cv::putText(overlay, "RESET DATABASE?",
                        cv::Point(boxX + 100, boxY + 35),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8,
                        cv::Scalar(0, 0, 255), 2);

            cv::putText(overlay, "Press 'y' to confirm, any other key to cancel",
                        cv::Point(boxX + 30, boxY + 80),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(255, 255, 255), 1);

            cv::addWeighted(overlay, 0.8, frame, 0.2, 0, frame);
        }

        // 2. Static info bar at bottom
        cv::putText(frame, "'a'=Add Person | SPACE=Recognize | 'r'=Reset DB | 'q'=Quit",
                    cv::Point(10, frame.rows - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(255, 255, 255), 1);

        std::string dbInfo = "DB: " + std::to_string(database_->size()) + " persons";
        cv::putText(frame, dbInfo, cv::Point(10, 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(200, 200, 255), 1);

        // 3. Name input GUI overlay (when in add-person mode)
        if (inNameInputMode_)
        {
            // Draw a semi-transparent overlay on the left side
            cv::Mat overlay = frame.clone();

            // Input box background
            int boxX = 50, boxY = frame.rows / 2 - 80;
            int boxW = frame.cols - 100, boxH = 120;
            cv::rectangle(overlay,
                          cv::Point(boxX, boxY),
                          cv::Point(boxX + boxW, boxY + boxH),
                          cv::Scalar(0, 0, 0), cv::FILLED);

            // Show the face crop thumbnail
            if (!addPersonFaceCrop_.empty())
            {
                cv::Mat thumb;
                cv::resize(addPersonFaceCrop_, thumb, cv::Size(80, 80));
                int thumbX = boxX + 15;
                int thumbY = boxY + 20;
                // Draw border around thumb
                cv::rectangle(overlay,
                              cv::Point(thumbX - 2, thumbY - 2),
                              cv::Point(thumbX + 82, thumbY + 82),
                              cv::Scalar(0, 255, 0), 2);
                thumb.copyTo(overlay(cv::Rect(thumbX, thumbY, 80, 80)));
            }

            // "Enter name:" label
            cv::putText(overlay, "Enter person name:",
                        cv::Point(boxX + 110, boxY + 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(200, 200, 255), 1);

            // The name being typed (with blinking cursor effect)
            std::string displayName = nameInput_;
            if (frameCount % 20 < 10)  // blink cursor
                displayName += "_";
            cv::putText(overlay, displayName,
                        cv::Point(boxX + 110, boxY + 65),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7,
                        cv::Scalar(0, 255, 255), 2);

            // Instructions
            cv::putText(overlay, "ENTER to confirm | BACKSPACE to delete | ESC to cancel",
                        cv::Point(boxX + 15, boxY + 100),
                        cv::FONT_HERSHEY_SIMPLEX, 0.45,
                        cv::Scalar(150, 150, 150), 1);

            // Blend overlay (60% opacity)
            cv::addWeighted(overlay, 0.75, frame, 0.25, 0, frame);
        }
        else if (addingPerson_)
        {
            // Show detection status while waiting for face
            cv::putText(frame, "Detecting face... press 'a' again when face is visible",
                        cv::Point(frame.cols / 2 - 200, 50),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(0, 0, 255), 2);
        }

        // Show the frame
        cv::imshow(winName, frame);

        // --- Handle key press ---
        int key = cv::waitKey(30);

        if (inNameInputMode_)
        {
            // Handle name input keys
            if (key == KEY_ENTER)
            {
                confirmAddPerson();
            }
            else if (key == KEY_BACKSPACE || key == 127)
            {
                if (!nameInput_.empty())
                    nameInput_.pop_back();
            }
            else if (key == 27) // ESC - cancel
            {
                std::cout << "[FaceRecognitionApp] Name input cancelled." << std::endl;
                inNameInputMode_ = false;
                addingPerson_ = false;
                nameInput_.clear();
            }
            else if (key >= 32 && key <= 126) // printable ASCII
            {
                if (nameInput_.length() < 30) // limit name length
                    nameInput_ += static_cast<char>(key);
            }
            continue; // skip other key processing while typing
        }

        // Handle reset confirmation mode keys
        if (resetConfirmPending_)
        {
            if (key == 'y' || key == 'Y')
            {
                resetDatabase();
                // Immediately clear cached recognition data to prevent stale
                // overlays from appearing before the next recognition cycle.
                lastName.clear();
                lastFaces.clear();
                resetConfirmPending_ = false;
            }
            else if (key != -1)  // Any key press other than -1 (no key) cancels
            {
                std::cout << "[FaceRecognitionApp] Reset cancelled." << std::endl;
                resetConfirmPending_ = false;
            }
            continue;  // Stay in confirmation mode until a key is pressed
        }

        // Normal mode keys
        if (key == 'q' || key == 'Q' || key == 27)
        {
            std::cout << "[FaceRecognitionApp] Exiting." << std::endl;
            break;
        }

        if (key == 'a' || key == 'A')
        {
            startAddPerson(currentFrame_);
        }

        if (key == ' ')
        {
            std::cout << "[FaceRecognitionApp] Manual capture triggered." << std::endl;
            recognizeFrame(currentFrame_);
        }

        if (key == 'r' || key == 'R')
        {
            std::cout << "[FaceRecognitionApp] Reset requested. Confirm on screen." << std::endl;
            resetConfirmPending_ = true;
        }

        ++frameCount;
    }

    cv::destroyWindow(winName);
}

void FaceRecognitionApp::onFrameCaptured(const cv::Mat& frame)
{
    recognizeFrame(frame);
}

void FaceRecognitionApp::recognizeFrame(const cv::Mat& frame)
{
    cv::Rect faceRect;
    std::vector<float> embedding = models_->detectAndEmbed(frame, faceRect);

    if (embedding.empty())
    {
        recognizedName_ = "";
        return;
    }

    // If the database is empty, there's nothing to match against.
    if (database_->size() == 0)
    {
        recognizedName_ = "";
        return;
    }

    int matchIndex = database_->findMatch(embedding, 0.5f);

    if (matchIndex >= 0)
    {
        recognizedName_ = database_->getName(matchIndex);
        std::cout << "[FaceRecognitionApp] Recognized: " << recognizedName_ << std::endl;
    }
    else
    {
        recognizedName_ = "Unknown";
    }
    addingPerson_ = false;
}

void FaceRecognitionApp::startAddPerson(const cv::Mat& frame)
{
    addingPerson_ = true;

    cv::Rect faceRect;
    std::vector<float> embedding = models_->detectAndEmbed(frame, faceRect);

    if (embedding.empty())
    {
        std::cout << "[FaceRecognitionApp] No face detected. Cannot add person." << std::endl;
        addingPerson_ = false;
        return;
    }

    // Store face data for when user confirms
    addPersonFaceCrop_ = frame(faceRect).clone();
    addPersonEmbedding_ = embedding;
    addPersonFaceRect_ = faceRect;
    nameInput_.clear();
    inNameInputMode_ = true;

    std::cout << "[FaceRecognitionApp] Face detected! Type name on screen and press ENTER." << std::endl;
}

void FaceRecognitionApp::confirmAddPerson()
{
    if (nameInput_.empty())
    {
        std::cout << "[FaceRecognitionApp] Empty name. Cancelled." << std::endl;
        inNameInputMode_ = false;
        addingPerson_ = false;
        return;
    }

    int id = database_->addPerson(nameInput_, addPersonEmbedding_, addPersonFaceCrop_);
    recognizedName_ = nameInput_;

    std::cout << "[FaceRecognitionApp] Person '" << nameInput_ << "' added with ID " << id << std::endl;

    inNameInputMode_ = false;
    addingPerson_ = false;
    nameInput_.clear();
}

void FaceRecognitionApp::resetDatabase()
{
    std::cout << "[FaceRecognitionApp] Resetting database..." << std::endl;
    database_->reset();
    recognizedName_ = "";
    std::cout << "[FaceRecognitionApp] Database has been reset to empty." << std::endl;
}