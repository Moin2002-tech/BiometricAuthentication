//
// Created by moinshaikh on 7/2/26.
//

#include "Recognition/modelsHandler.hpp"
#include<torch/torch.h>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

namespace Recognition {

    bool ModelsHandler::isCudaAvailable() {
        return torch::cuda::is_available();
    }

    ModelsHandler::ModelsHandler()
        : useGpu_(torch::cuda::is_available())
        , device_(useGpu_ ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU))
    {
        // 1. Load the face detection model (OpenCV DNN / Caffe)
        std::cout << "[ModelsHandler] Loading face detection model..." << std::endl;
        faceDetector_ = cv::dnn::readNetFromCaffe(PROTOTXT, CAFFEMODEL);
        if (faceDetector_.empty())
        {
            throw std::runtime_error("Failed to load face detection model: " +
                                     std::string(CAFFEMODEL));
        }
        // Use CPU for detection (it's fast enough)
        faceDetector_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        faceDetector_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        // 2. Load the face recognition model (TorchScript / ArcFace)
        std::cout << "[ModelsHandler] Loading face recognition model..." << std::endl;
        
        if (useGpu_) {
            std::cout << "[ModelsHandler] CUDA is available. Using GPU for recognition." << std::endl;
        } else {
            std::cout << "[ModelsHandler] CUDA is NOT available. Using CPU for recognition." << std::endl;
        }

        try
        {
            recognizer_ = torch::jit::load(TORCHSCRIPT_PATH);
            recognizer_.to(device_);  // Move model to GPU/CPU
            recognizer_.eval();
        }
        catch (const c10::Error& e)
        {
            throw std::runtime_error("Failed to load TorchScript model: " +
                                     std::string(TORCHSCRIPT_PATH) + " - " + e.what());
        }

        // 3. Load eye cascade for face alignment
        std::cout << "[ModelsHandler] Loading eye cascade for face alignment..." << std::endl;
        if (!eyeCascade_.load(EYE_CASCADE_PATH))
        {
            std::cerr << "[ModelsHandler] Warning: Could not load eye cascade from "
                      << EYE_CASCADE_PATH << ". Face alignment disabled." << std::endl;
        }

        std::cout << "[ModelsHandler] All models loaded successfully." << std::endl;
        std::cout << "[ModelsHandler] Recognition device: " << (useGpu_ ? "GPU (CUDA)" : "CPU") << std::endl;
    }

    cv::Mat ModelsHandler::preprocessForDetection(const cv::Mat& frame) const
    {
        // SSD expects 300x300 BGR image
        cv::Mat blob = cv::dnn::blobFromImage(
            frame,
            DETECT_SCALE_FACTOR,
            cv::Size(DETECT_INPUT_WIDTH, DETECT_INPUT_HEIGHT),
            cv::Scalar(104.0, 177.0, 123.0),  // mean subtraction (from Caffe SSD)
            false,   // swapRB
            false    // crop
        );
        return blob;
    }

    torch::Tensor ModelsHandler::preprocessForRecognition(const cv::Mat& faceImg) const
    {
        // Resize to 112x112
        cv::Mat resized;
        cv::resize(faceImg, resized, cv::Size(RECOG_INPUT_SIZE, RECOG_INPUT_SIZE));

        // Convert BGR to RGB
        cv::Mat rgb;
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

        // Convert to float tensor: HWC -> CHW, normalize to [0,1]
        torch::Tensor tensor = torch::from_blob(
            rgb.data,
            {RECOG_INPUT_SIZE, RECOG_INPUT_SIZE, 3},
            torch::kByte
        ).to(torch::kFloat32) / 255.0f;

        // Permute HWC -> CHW and add batch dimension
        tensor = tensor.permute({2, 0, 1}).unsqueeze(0);

        // Normalize with mean=0.5, std=0.5 (maps [0,1] -> [-1, 1])
        tensor = (tensor - 0.5f) / 0.5f;

        // Move tensor to the appropriate device (GPU or CPU)
        tensor = tensor.to(device_);

        return tensor;
    }

    std::vector<cv::Rect> ModelsHandler::detectFaces(const cv::Mat& frame,
                                                       float minConfidence)
    {
        std::vector<cv::Rect> faces;

        cv::Mat blob = preprocessForDetection(frame);
        faceDetector_.setInput(blob, "data");
        cv::Mat detection = faceDetector_.forward("detection_out");

        cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

        float heightScale = static_cast<float>(frame.rows);
        float widthScale = static_cast<float>(frame.cols);

        for (int i = 0; i < detectionMat.rows; ++i)
        {
            float confidence = detectionMat.at<float>(i, 2);
            if (confidence > minConfidence)
            {
                int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * widthScale);
                int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * heightScale);
                int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * widthScale);
                int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * heightScale);

                // Clamp to frame boundaries
                x1 = std::max(0, std::min(x1, frame.cols - 1));
                y1 = std::max(0, std::min(y1, frame.rows - 1));
                x2 = std::max(0, std::min(x2, frame.cols - 1));
                y2 = std::max(0, std::min(y2, frame.rows - 1));

                cv::Rect faceRect(x1, y1, x2 - x1 + 1, y2 - y1 + 1);
                if (faceRect.width > 0 && faceRect.height > 0)
                {
                    faces.push_back(faceRect);
                }
            }
        }

        return faces;
    }

    cv::Mat ModelsHandler::alignFace(const cv::Mat& faceImg)
    {
        // If eye cascade not loaded, return original
        if (eyeCascade_.empty())
            return faceImg;

        // Convert to grayscale for eye detection
        cv::Mat gray;
        cv::cvtColor(faceImg, gray, cv::COLOR_BGR2GRAY);

        // Detect eyes in the face crop
        std::vector<cv::Rect> eyes;
        eyeCascade_.detectMultiScale(gray, eyes, 1.1, 3, 0, cv::Size(10, 10));

        // Need at least 2 eyes for alignment
        if (eyes.size() < 2)
            return faceImg;

        // Find the two leftmost eyes (we want left and right eye centers)
        // Sort by x coordinate
        std::sort(eyes.begin(), eyes.end(),
                  [](const cv::Rect& a, const cv::Rect& b) { return a.x < b.x; });

        cv::Point leftEye(eyes[0].x + eyes[0].width / 2, eyes[0].y + eyes[0].height / 2);
        cv::Point rightEye(eyes[1].x + eyes[1].width / 2, eyes[1].y + eyes[1].height / 2);

        // Calculate the angle between the eyes
        double dx = static_cast<double>(rightEye.x - leftEye.x);
        double dy = static_cast<double>(rightEye.y - leftEye.y);
        double angle = std::atan2(dy, dx) * 180.0 / CV_PI;

        // Desired: eyes should be perfectly horizontal (angle = 0)
        // If angle is already near 0, skip alignment
        if (std::abs(angle) < 2.0)
            return faceImg;

        // Compute the center point between the eyes
        cv::Point center((leftEye.x + rightEye.x) / 2, (leftEye.y + rightEye.y) / 2);

        // Get the rotation matrix to align eyes horizontally
        cv::Mat rotMatrix = cv::getRotationMatrix2D(center, angle, 1.0);

        // Warp the face image to align eyes horizontally
        cv::Mat aligned;
        cv::warpAffine(faceImg, aligned, rotMatrix, faceImg.size(),
                       cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

        return aligned;
    }

    std::vector<float> ModelsHandler::getEmbedding(const cv::Mat& faceImg)
    {
        // First align the face to normalize head rotation
        cv::Mat alignedFace = alignFace(faceImg);
        torch::Tensor tensor = preprocessForRecognition(alignedFace);
        torch::Tensor output = recognizer_.forward({tensor}).toTensor();

        // Move output to CPU and extract as vector
        output = output.squeeze().cpu();  // [512]
        std::vector<float> embedding(output.data_ptr<float>(),
                                     output.data_ptr<float>() + output.numel());
        return embedding;
    }

    std::vector<float> ModelsHandler::detectAndEmbed(const cv::Mat& frame,
                                                       cv::Rect& faceRect)
    {
        std::vector<cv::Rect> faces = detectFaces(frame);

        if (faces.empty())
        {
            faceRect = cv::Rect();
            return {};
        }

        // Use the largest face (by area)
        int largestIdx = 0;
        int largestArea = faces[0].area();
        for (size_t i = 1; i < faces.size(); ++i)
        {
            int area = faces[i].area();
            if (area > largestArea)
            {
                largestArea = area;
                largestIdx = static_cast<int>(i);
            }
        }

        faceRect = faces[largestIdx];
        cv::Mat faceCrop = frame(faceRect).clone();
        return getEmbedding(faceCrop);
    }

    void ModelsHandler::detectAndEmbedAll(const cv::Mat& frame,
                                           std::vector<cv::Rect>& faceRects,
                                           std::vector<std::vector<float>>& embeddings,
                                           float minConfidence)
    {
        faceRects = detectFaces(frame, minConfidence);
        embeddings.clear();
        embeddings.reserve(faceRects.size());

        for (const auto& rect : faceRects)
        {
            cv::Mat faceCrop = frame(rect).clone();
            std::vector<float> emb = getEmbedding(faceCrop);
            embeddings.push_back(emb);
        }

        std::cout << "[ModelsHandler] Detected " << faceRects.size()
                  << " faces and computed embeddings." << std::endl;
    }

} // namespace Recognition