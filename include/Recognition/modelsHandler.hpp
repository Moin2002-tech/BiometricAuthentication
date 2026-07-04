//
// Created by moinshaikh on 7/2/26.
//

#pragma once

#ifndef BIOMETRICAUTHENTICATION_MODELSHANDLER_HPP
#define BIOMETRICAUTHENTICATION_MODELSHANDLER_HPP

#include <torch/script.h>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/objdetect.hpp>
#include <vector>
#include <string>

namespace Recognition {

    /**
     * @brief ModelsHandler loads and runs face detection + recognition models.
     *
     * Uses:
     *   1. OpenCV DNN + Caffe model (SSD) for face detection
     *   2. LibTorch TorchScript model (ArcFace backbone) for embedding extraction
     *
     * The output embedding is a 512-dimensional feature vector that can be
     * used for face recognition via cosine similarity.
     */
    class ModelsHandler {
    public:
        /**
         * @brief Construct the handler and load both models.
         * @throws std::runtime_error if models fail to load.
         */
        ModelsHandler();

        /**
         * @brief Check if CUDA/GPU is available.
         * @return true if CUDA is available, false otherwise.
         */
        static bool isCudaAvailable();

        /**
         * @brief Check if GPU mode is enabled.
         * @return true if GPU mode is enabled.
         */
        bool isGpuEnabled() const { return useGpu_; }

        /**
         * @brief Detect faces in an image using the SSD detector.
         * @param frame  Input BGR frame.
         * @param minConfidence  Minimum detection confidence (default: 0.5).
         * @return Vector of detected face bounding boxes (x, y, w, h).
         */
        std::vector<cv::Rect> detectFaces(const cv::Mat& frame,
                                           float minConfidence = 0.5f);

        /**
         * @brief Extract a 512-dim face embedding from a face image.
         * @param faceImg  Input face crop (BGR, any size, will be resized to 112x112).
         * @return 512-dimensional embedding vector.
         */
        std::vector<float> getEmbedding(const cv::Mat& faceImg);

        /**
         * @brief Detect the largest face and return its embedding.
         * @param frame  Full BGR frame.
         * @param faceRect [out] Bounding box of the detected face.
         * @return 512-dim embedding, or empty vector if no face detected.
         */
        std::vector<float> detectAndEmbed(const cv::Mat& frame,
                                           cv::Rect& faceRect);

        /**
         * @brief Detect ALL faces in the frame and return embeddings for each.
         * @param frame  Full BGR frame.
         * @param faceRects [out] Bounding boxes of ALL detected faces.
         * @param embeddings [out] 512-dim embeddings for each detected face.
         * @param minConfidence  Minimum detection confidence.
         */
        void detectAndEmbedAll(const cv::Mat& frame,
                               std::vector<cv::Rect>& faceRects,
                               std::vector<std::vector<float>>& embeddings,
                               float minConfidence = 0.5f);

    private:
        // Face detection model (OpenCV DNN / Caffe)
        cv::dnn::Net faceDetector_;
        static constexpr const char* PROTOTXT = MODELS_DIR "/deploy.prototxt";
        static constexpr const char* CAFFEMODEL = MODELS_DIR "/res10_300x300_ssd_iter_140000.caffemodel";
        static constexpr int DETECT_INPUT_WIDTH = 300;
        static constexpr int DETECT_INPUT_HEIGHT = 300;
        static constexpr float DETECT_SCALE_FACTOR = 1.0f;

        // Face recognition model (TorchScript / ArcFace)
        torch::jit::Module recognizer_;
        static constexpr const char* TORCHSCRIPT_PATH = MODELS_DIR "/model.torchscript.pt";
        static constexpr int RECOG_INPUT_SIZE = 112;

        // Device configuration
        torch::Device device_;
        bool useGpu_;

        // Face alignment: eye cascade for rotation normalization
        cv::CascadeClassifier eyeCascade_;
        static constexpr const char* EYE_CASCADE_PATH = "/usr/share/opencv4/haarcascades/haarcascade_eye.xml";

        // Preprocessing helpers
        cv::Mat preprocessForDetection(const cv::Mat& frame) const;

        /**
         * @brief Align a face crop to a canonical frontal orientation using eye detection.
         * @param faceImg  Input face crop (BGR).
         * @return Aligned face crop, or original if alignment fails.
         */
        cv::Mat alignFace(const cv::Mat& faceImg);

        torch::Tensor preprocessForRecognition(const cv::Mat& faceImg) const;
    };

} // namespace Recognition

#endif //BIOMETRICAUTHENTICATION_MODELSHANDLER_HPP