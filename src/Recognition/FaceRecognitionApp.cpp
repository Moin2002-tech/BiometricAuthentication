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
static constexpr int KEY_TAB = 9;

// Color constants for overlays
static const cv::Scalar COLOR_KNOWN(0, 255, 0);       // Green - recognized person
static const cv::Scalar COLOR_UNKNOWN(0, 0, 255);     // Red - unknown person
static const cv::Scalar COLOR_SELECTED(255, 255, 0);  // Cyan - selected face
static const cv::Scalar COLOR_NORMAL(255, 255, 255);  // White - normal text

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
    std::cout << "    'a'    - Add NEW person to database (ID + name + face)" << std::endl;
    std::cout << "    'b'    - Add ADDITIONAL face angle to EXISTING person" << std::endl;
    std::cout << "    SPACE  - Manual recognize" << std::endl;
    std::cout << "    'r'    - Reset database (with confirmation)" << std::endl;
    std::cout << "    'q'    - Quit" << std::endl;
    std::cout << "  Multi-face:" << std::endl;
    std::cout << "    LEFT/RIGHT arrow keys to cycle through faces when adding" << std::endl;
    std::cout << "    TAB to switch between ID and Name fields during input" << std::endl;
    std::cout << "  Tips:" << std::endl;
    std::cout << "    Register the same person from multiple angles for better recognition" << std::endl;
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

    // Tracks whether we're adding a new person ('a') or adding angle to existing ('b')
    int addMode = 0; // 0 = new person, 1 = additional face

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

        // Run multi-face detection + recognition every 10 frames
        if (frameCount % 10 == 0)
        {
            if (!inNameInputMode_ && !faceSelectionMode_)  // Don't run recognition while in input modes
            {
                recognizeFrame(currentFrame_);
            }
        }

        // --- Draw overlays ---
        drawFaceOverlays(frame, frameCount);

        // Draw reset confirmation overlay
        if (resetConfirmPending_)
        {
            drawResetOverlay(frame);
        }

        // Draw face selection overlay
        if (faceSelectionMode_)
        {
            drawSelectionOverlay(frame, frameCount);
        }

        // Draw name/ID input overlay
        if (inNameInputMode_)
        {
            drawInputOverlay(frame, frameCount);
        }

        // Static info bar at bottom
        std::string controls = "'a'=Add Person | 'b'=Add Angle | SPACE=Recognize | 'r'=Reset | 'q'=Quit";
        cv::putText(frame, controls,
                    cv::Point(10, frame.rows - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45,
                    cv::Scalar(255, 255, 255), 1);

        int totalFaces = database_->totalFaces();
        std::string dbInfo = "DB: " + std::to_string(database_->size()) + " persons ("
                             + std::to_string(totalFaces) + " angles)"
                             + " | Cam: " + std::to_string(detectedFaceRects_.size()) + " faces";
        cv::putText(frame, dbInfo, cv::Point(10, 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(200, 200, 255), 1);

        // Show the frame
        cv::imshow(winName, frame);

        // --- Handle key press ---
        int key = cv::waitKey(30);

        // Handle name/ID input mode keys
        if (inNameInputMode_)
        {
            if (key == KEY_ENTER)
            {
                confirmAddPerson();
            }
            else if (key == KEY_TAB)
            {
                inputtingId_ = !inputtingId_;
                std::cout << "[FaceRecognitionApp] Switched to "
                          << (inputtingId_ ? "ID" : "Name") << " input." << std::endl;
            }
            else if (key == KEY_BACKSPACE || key == 127)
            {
                if (inputtingId_ && !idInput_.empty())
                    idInput_.pop_back();
                else if (!inputtingId_ && !nameInput_.empty())
                    nameInput_.pop_back();
            }
            else if (key == 27) // ESC - cancel
            {
                std::cout << "[FaceRecognitionApp] Input cancelled." << std::endl;
                inNameInputMode_ = false;
                addingPerson_ = false;
                idInput_.clear();
                nameInput_.clear();
            }
            else if (key >= 32 && key <= 126) // printable ASCII
            {
                if (inputtingId_)
                {
                    // Only allow digits for ID input
                    if (key >= '0' && key <= '9' && idInput_.length() < 6)
                        idInput_ += static_cast<char>(key);
                }
                else
                {
                    if (nameInput_.length() < 30)
                        nameInput_ += static_cast<char>(key);
                }
            }
            continue; // skip other key processing while typing
        }

        // Handle face selection mode keys
        if (faceSelectionMode_)
        {
            if (key == KEY_ENTER || key == ' ')
            {
                // Confirm selection - proceed to input mode
                if (selectedFaceIndex_ >= 0 &&
                    selectedFaceIndex_ < static_cast<int>(detectedFaceRects_.size()))
                {
                    addPersonFaceRect_ = detectedFaceRects_[selectedFaceIndex_];
                    addPersonFaceCrop_ = currentFrame_(addPersonFaceRect_).clone();
                    addPersonEmbedding_ = detectedFaceEmbeddings_[selectedFaceIndex_];
                    if (addPersonEmbedding_.empty())
                    {
                        // Compute embedding for the selected face if not already done
                        addPersonEmbedding_ = models_->getEmbedding(addPersonFaceCrop_);
                    }
                    idInput_.clear();
                    nameInput_.clear();
                    inputtingId_ = true;
                    inNameInputMode_ = true;
                    faceSelectionMode_ = false;

                    if (addMode == 1)
                        std::cout << "[FaceRecognitionApp] Face selected. Enter existing person ID (name optional)." << std::endl;
                    else
                        std::cout << "[FaceRecognitionApp] Face selected. Enter ID and name." << std::endl;
                }
                else
                {
                    std::cout << "[FaceRecognitionApp] No face selected." << std::endl;
                    faceSelectionMode_ = false;
                    addingPerson_ = false;
                }
            }
            else if (key == 27) // ESC - cancel
            {
                std::cout << "[FaceRecognitionApp] Face selection cancelled." << std::endl;
                faceSelectionMode_ = false;
                addingPerson_ = false;
            }
            else if (key == 81 || key == 83 || key == 65361 || key == 65363) // LEFT=81/65361, RIGHT=83/65363
            {
                cycleSelection(key == 81 || key == 65361 ? -1 : 1);
            }
            continue;
        }

        // Handle reset confirmation mode keys
        if (resetConfirmPending_)
        {
            if (key == 'y' || key == 'Y')
            {
                resetDatabase();
                detectedFaceRects_.clear();
                detectedFaceNames_.clear();
                detectedFaceEmbeddings_.clear();
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
            addMode = 0;  // New person mode
            startAddPerson(currentFrame_);
        }

        if (key == 'b' || key == 'B')
        {
            addMode = 1;  // Additional face angle mode
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

        // Arrow keys to cycle through faces in normal mode too (for selection preview)
        if ((key == 81 || key == 65361) && !detectedFaceRects_.empty())
        {
            int newIdx = (selectedFaceIndex_ - 1 + static_cast<int>(detectedFaceRects_.size()))
                         % static_cast<int>(detectedFaceRects_.size());
            selectedFaceIndex_ = newIdx;
            std::cout << "[FaceRecognitionApp] Selected face " << selectedFaceIndex_ << std::endl;
        }
        if ((key == 83 || key == 65363) && !detectedFaceRects_.empty())
        {
            int newIdx = (selectedFaceIndex_ + 1) % static_cast<int>(detectedFaceRects_.size());
            selectedFaceIndex_ = newIdx;
            std::cout << "[FaceRecognitionApp] Selected face " << selectedFaceIndex_ << std::endl;
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
    // Detect all faces and get embeddings
    models_->detectAndEmbedAll(frame, detectedFaceRects_, detectedFaceEmbeddings_);

    detectedFaceNames_.clear();
    detectedFaceNames_.reserve(detectedFaceRects_.size());

    for (size_t i = 0; i < detectedFaceRects_.size(); ++i)
    {
        if (i < detectedFaceEmbeddings_.size() && !detectedFaceEmbeddings_[i].empty())
        {
            std::string name = getNameForEmbedding(detectedFaceEmbeddings_[i]);
            detectedFaceNames_.push_back(name);
        }
        else
        {
            detectedFaceNames_.push_back("");
        }
    }

    if (detectedFaceRects_.empty())
    {
        std::cout << "[FaceRecognitionApp] No faces detected." << std::endl;
    }
    else
    {
        std::cout << "[FaceRecognitionApp] Detected " << detectedFaceRects_.size()
                  << " faces." << std::endl;
    }
}

std::string FaceRecognitionApp::getNameForEmbedding(const std::vector<float>& embedding)
{
    if (embedding.empty() || database_->size() == 0)
    {
        return "";
    }

    // Lower threshold (0.35 vs 0.5) to better handle side/angled faces
    int matchIndex = database_->findMatch(embedding, 0.35f);

    if (matchIndex >= 0)
    {
        return database_->getName(matchIndex);
    }

    return "Unknown";
}

void FaceRecognitionApp::startAddPerson(const cv::Mat& frame)
{
    // Run detection + recognition on the current frame first
    recognizeFrame(frame);

    if (detectedFaceRects_.empty())
    {
        std::cout << "[FaceRecognitionApp] No face detected. Cannot start." << std::endl;
        addingPerson_ = false;
        return;
    }

    addingPerson_ = true;

    // If only one face, skip selection mode
    if (detectedFaceRects_.size() == 1)
    {
        selectedFaceIndex_ = 0;
        addPersonFaceRect_ = detectedFaceRects_[0];
        addPersonFaceCrop_ = frame(addPersonFaceRect_).clone();
        addPersonEmbedding_ = detectedFaceEmbeddings_[0];
        if (addPersonEmbedding_.empty())
        {
            addPersonEmbedding_ = models_->getEmbedding(addPersonFaceCrop_);
        }
        idInput_.clear();
        nameInput_.clear();
        inputtingId_ = true;
        inNameInputMode_ = true;
        std::cout << "[FaceRecognitionApp] Single face detected. Enter ID and name." << std::endl;
    }
    else
    {
        // Multiple faces - enter selection mode
        faceSelectionMode_ = true;
        selectedFaceIndex_ = 0;  // Default to first face
        std::cout << "[FaceRecognitionApp] Multiple faces (" << detectedFaceRects_.size()
                  << ") detected. Select one:" << std::endl;
        std::cout << "  - LEFT/RIGHT arrows to cycle" << std::endl;
        std::cout << "  - ENTER/SPACE to confirm" << std::endl;
        std::cout << "  - ESC to cancel" << std::endl;
    }
}

void FaceRecognitionApp::confirmAddPerson()
{
    if (idInput_.empty())
    {
        std::cout << "[FaceRecognitionApp] Empty ID. Cancelled." << std::endl;
        inNameInputMode_ = false;
        addingPerson_ = false;
        idInput_.clear();
        nameInput_.clear();
        return;
    }

    int manualId = std::stoi(idInput_);

    // Check if this is an "add angle to existing person" operation
    if (database_->hasId(manualId))
    {
        // Add additional face angle to existing person
        bool success = database_->addFaceToPerson(manualId, addPersonEmbedding_, addPersonFaceCrop_);
        if (success)
        {
            std::string personName = database_->getName(
                database_->findMatch(addPersonEmbedding_, 0.0f));
            std::cout << "[FaceRecognitionApp] Added face angle to '" << personName
                      << "' (ID=" << manualId << ")" << std::endl;
        }
        else
        {
            std::cout << "[FaceRecognitionApp] Failed to add face to ID " << manualId << std::endl;
            inNameInputMode_ = false;
            addingPerson_ = false;
            idInput_.clear();
            nameInput_.clear();
            return;
        }
    }
    else
    {
        // New person
        if (nameInput_.empty())
        {
            std::cout << "[FaceRecognitionApp] New person requires a name. Cancelled." << std::endl;
            inNameInputMode_ = false;
            addingPerson_ = false;
            idInput_.clear();
            nameInput_.clear();
            return;
        }

        std::string personName = nameInput_;

        // Try to add with manual ID
        int result = database_->addPerson(manualId, personName, addPersonEmbedding_, addPersonFaceCrop_);

        if (result < 0)
        {
            std::cout << "[FaceRecognitionApp] Failed to add person. ID " << manualId
                      << " might already exist." << std::endl;
            // Stay in input mode so user can change ID
            return;
        }

        std::cout << "[FaceRecognitionApp] Person '" << personName << "' added with ID "
                  << result << std::endl;
    }

    inNameInputMode_ = false;
    addingPerson_ = false;
    idInput_.clear();
    nameInput_.clear();
    selectedFaceIndex_ = -1;

    // Re-run recognition to show the new/updated person on screen
    recognizeFrame(currentFrame_);
}

void FaceRecognitionApp::drawFaceOverlays(cv::Mat& frame, int frameCount)
{
    // Draw bounding boxes and names for each detected face
    for (size_t i = 0; i < detectedFaceRects_.size(); ++i)
    {
        const auto& face = detectedFaceRects_[i];
        const std::string& name = (i < detectedFaceNames_.size()) ? detectedFaceNames_[i] : "";

        // Determine color based on recognition status and selection
        cv::Scalar boxColor;
        if (static_cast<int>(i) == selectedFaceIndex_)
        {
            boxColor = COLOR_SELECTED;  // Cyan for selected
        }
        else if (name == "Unknown")
        {
            boxColor = COLOR_UNKNOWN;   // Red for unknown
        }
        else if (!name.empty())
        {
            boxColor = COLOR_KNOWN;     // Green for recognized
        }
        else
        {
            boxColor = cv::Scalar(200, 200, 200); // Gray for undetermined
        }

        cv::rectangle(frame, face, boxColor, 2);

        // Draw label background and name
        if (!name.empty())
        {
            int baseline = 0;
            cv::Size textSize = cv::getTextSize(name, cv::FONT_HERSHEY_SIMPLEX,
                                                 0.6, 2, &baseline);
            // Draw filled background rectangle for text
            cv::rectangle(frame,
                          cv::Point(face.x, face.y - textSize.height - 10),
                          cv::Point(face.x + textSize.width + 5, face.y),
                          cv::Scalar(0, 0, 0), cv::FILLED);
            cv::putText(frame, name,
                        cv::Point(face.x + 3, face.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        boxColor, 2);
        }

        // Draw face index number
        std::string idxStr = "#" + std::to_string(i + 1);
        cv::putText(frame, idxStr,
                    cv::Point(face.x + face.width - 25, face.y + 15),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4,
                    cv::Scalar(200, 200, 200), 1);
    }
}

void FaceRecognitionApp::drawInputOverlay(cv::Mat& frame, int frameCount)
{
    cv::Mat overlay = frame.clone();

    // Input box background
    int boxX = 50, boxY = frame.rows / 2 - 100;
    int boxW = frame.cols - 100, boxH = 200;
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
        cv::rectangle(overlay,
                      cv::Point(thumbX - 2, thumbY - 2),
                      cv::Point(thumbX + 82, thumbY + 82),
                      cv::Scalar(0, 255, 0), 2);
        thumb.copyTo(overlay(cv::Rect(thumbX, thumbY, 80, 80)));
    }

    // "Enter ID:" label (highlighted if currently entering ID)
    cv::Scalar idLabelColor = inputtingId_ ? cv::Scalar(0, 255, 255) : cv::Scalar(150, 150, 150);
    cv::putText(overlay, "Enter ID:",
                cv::Point(boxX + 110, boxY + 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.6,
                idLabelColor, inputtingId_ ? 2 : 1);

    // ID input field
    std::string displayId = idInput_;
    if (inputtingId_ && frameCount % 20 < 10)
        displayId += "_";
    cv::putText(overlay, displayId,
                cv::Point(boxX + 110, boxY + 65),
                cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0, 255, 255), 2);

    // "Enter name:" label (highlighted if currently entering name)
    cv::Scalar nameLabelColor = !inputtingId_ ? cv::Scalar(0, 255, 255) : cv::Scalar(150, 150, 150);
    cv::putText(overlay, "Enter Name:",
                cv::Point(boxX + 110, boxY + 95),
                cv::FONT_HERSHEY_SIMPLEX, 0.6,
                nameLabelColor, !inputtingId_ ? 2 : 1);

    // Name input field
    std::string displayName = nameInput_;
    if (!inputtingId_ && frameCount % 20 < 10)
        displayName += "_";
    cv::putText(overlay, displayName,
                cv::Point(boxX + 110, boxY + 130),
                cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0, 255, 255), 2);

    // Instructions
    cv::putText(overlay, "TAB to switch fields | ENTER to confirm | ESC to cancel",
                cv::Point(boxX + 15, boxY + 175),
                cv::FONT_HERSHEY_SIMPLEX, 0.45,
                cv::Scalar(150, 150, 150), 1);

    // Blend overlay
    cv::addWeighted(overlay, 0.75, frame, 0.25, 0, frame);
}

void FaceRecognitionApp::drawSelectionOverlay(cv::Mat& frame, int frameCount)
{
    // Draw a semi-transparent overlay with selection instructions
    cv::Mat overlay = frame.clone();

    // Instruction box at top center
    std::string instrText = "SELECT A FACE: LEFT/RIGHT arrows or click | ENTER to confirm | ESC to cancel";
    int baseline = 0;
    cv::Size textSize = cv::getTextSize(instrText, cv::FONT_HERSHEY_SIMPLEX, 0.55, 2, &baseline);
    int boxX = frame.cols / 2 - textSize.width / 2 - 10;
    int boxY = 40;
    int boxW = textSize.width + 20;
    int boxH = textSize.height + 20;

    cv::rectangle(overlay,
                  cv::Point(boxX, boxY),
                  cv::Point(boxX + boxW, boxY + boxH),
                  cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(overlay, instrText,
                cv::Point(boxX + 10, boxY + textSize.height + 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.55,
                cv::Scalar(0, 255, 255), 2);

    // Alpha blend
    cv::addWeighted(overlay, 0.7, frame, 0.3, 0, frame);

    // Draw selection indicator on the selected face (blinking thick border)
    if (selectedFaceIndex_ >= 0 && selectedFaceIndex_ < static_cast<int>(detectedFaceRects_.size()))
    {
        const auto& selRect = detectedFaceRects_[selectedFaceIndex_];
        // Draw a thick blinking border
        if (frameCount % 10 < 5)
        {
            cv::rectangle(frame, selRect, COLOR_SELECTED, 4);
            // Also show "SELECTED" label
            int labelBaseline = 0;
            cv::Size lblSize = cv::getTextSize("SELECTED", cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &labelBaseline);
            cv::rectangle(frame,
                          cv::Point(selRect.x, selRect.y - lblSize.height - 12),
                          cv::Point(selRect.x + lblSize.width + 6, selRect.y),
                          cv::Scalar(0, 0, 0), cv::FILLED);
            cv::putText(frame, "SELECTED",
                        cv::Point(selRect.x + 3, selRect.y - 7),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        COLOR_SELECTED, 2);
        }
    }
}

void FaceRecognitionApp::drawResetOverlay(cv::Mat& frame)
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

void FaceRecognitionApp::resetDatabase()
{
    std::cout << "[FaceRecognitionApp] Resetting database..." << std::endl;
    database_->reset();
    std::cout << "[FaceRecognitionApp] Database has been reset to empty." << std::endl;
}

void FaceRecognitionApp::cycleSelection(int direction)
{
    if (detectedFaceRects_.empty())
    {
        selectedFaceIndex_ = -1;
        return;
    }

    int numFaces = static_cast<int>(detectedFaceRects_.size());

    if (selectedFaceIndex_ < 0)
    {
        selectedFaceIndex_ = (direction > 0) ? 0 : numFaces - 1;
    }
    else
    {
        selectedFaceIndex_ = (selectedFaceIndex_ + direction + numFaces) % numFaces;
    }

    std::cout << "[FaceRecognitionApp] Selected face " << selectedFaceIndex_
              << " / " << (numFaces - 1) << std::endl;
}