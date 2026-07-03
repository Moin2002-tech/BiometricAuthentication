//
// Created by moinshaikh on 6/30/26.
//

#include <doctest.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <cstring>

// Test suite for the OpenCV DNN Caffe face detection model (SSD)
TEST_SUITE("Caffe SSD Face Detection Tests") {

    // Model paths (set via -DMODELS_DIR compile definition in CMakeLists.txt)
    const std::string PROTOTXT_PATH = std::string(MODELS_DIR) + "/deploy.prototxt";
    const std::string CAFFEMODEL_PATH = std::string(MODELS_DIR) + "/res10_300x300_ssd_iter_140000.caffemodel";

    // Helper to load the model
    cv::dnn::Net load_model() {
        cv::dnn::Net net = cv::dnn::readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH);
        REQUIRE_FALSE(net.empty());
        return net;
    }

    TEST_CASE("Model loads successfully from Caffe files") {
        // Verify model files exist first
        std::ifstream proto_file(PROTOTXT_PATH);
        REQUIRE(proto_file.good());
        proto_file.close();

        std::ifstream model_file(CAFFEMODEL_PATH, std::ios::binary);
        REQUIRE(model_file.good());
        model_file.close();

        // Load model
        cv::dnn::Net net = cv::dnn::readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH);
        CHECK_FALSE(net.empty());
    }

    TEST_CASE("Model forward pass produces correct output shape") {
        cv::dnn::Net net = load_model();

        // Create a synthetic 300x300 BGR image
        cv::Mat image(300, 300, CV_8UC3, cv::Scalar(120, 120, 120));
        cv::randu(image, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

        // Convert to blob: 1x3x300x300
        cv::Mat blob = cv::dnn::blobFromImage(image, 1.0, cv::Size(300, 300),
                                               cv::Scalar(104.0, 177.0, 123.0), false, false);

        net.setInput(blob, "data");
        cv::Mat output = net.forward();

        // SSD output format: 1x1xNx7 where N = number of detections
        // Each detection: [batch_id, class_id, confidence, x1, y1, x2, y2]
        CHECK_EQ(output.dims, 4);
        CHECK_EQ(output.size[0], 1);  // batch
        CHECK_EQ(output.size[1], 1);  // class count dimension
        CHECK_GT(output.size[2], 0);  // at least some detections
        CHECK_EQ(output.size[3], 7);  // 7 values per detection
    }

    TEST_CASE("Synthetic face-like input produces confident detection") {
        cv::dnn::Net net = load_model();

        // Create a 300x300 image with a face-like oval in the center
        cv::Mat image(300, 300, CV_8UC3, cv::Scalar(100, 100, 100));
        // Draw a rough face-like shape (ellipse) in the center
        cv::ellipse(image, cv::Point(150, 150), cv::Size(60, 80), 0, 0, 360,
                    cv::Scalar(200, 180, 160), -1);
        // Add some noise for realism
        cv::Mat noise = cv::Mat::zeros(300, 300, CV_8UC3);
        cv::randu(noise, cv::Scalar(0, 0, 0), cv::Scalar(30, 30, 30));
        image += noise;

        // Preprocess
        cv::Mat blob = cv::dnn::blobFromImage(image, 1.0, cv::Size(300, 300),
                                               cv::Scalar(104.0, 177.0, 123.0), false, false);

        net.setInput(blob, "data");
        cv::Mat output = net.forward();

        // Parse detections
        float* data = (float*)output.data;
        int num_detections = output.size[2];
        float best_conf = 0.0f;

        for (int i = 0; i < num_detections; i++) {
            float* detection = data + i * 7;
            float confidence = detection[2];

            if (confidence > best_conf) {
                best_conf = confidence;
            }
        }

        // The synthetic face-like input should trigger at least some response
        CHECK_GT(best_conf, 0.01f);
    }

    TEST_CASE("Detection coordinates are within valid range") {
        cv::dnn::Net net = load_model();

        cv::Mat image(300, 300, CV_8UC3, cv::Scalar(120, 120, 120));
        cv::randu(image, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

        cv::Mat blob = cv::dnn::blobFromImage(image, 1.0, cv::Size(300, 300),
                                               cv::Scalar(104.0, 177.0, 123.0), false, false);

        net.setInput(blob, "data");
        cv::Mat output = net.forward();

        float* data = (float*)output.data;
        int num_detections = output.size[2];

        for (int i = 0; i < num_detections; i++) {
            float* detection = data + i * 7;

            // Coordinates are normalized to [0, 1]
            float x1 = detection[3];
            float y1 = detection[4];
            float x2 = detection[5];
            float y2 = detection[6];

            CHECK_GE(x1, 0.0f);
            CHECK_GE(y1, 0.0f);
            CHECK_LE(x2, 1.0f);
            CHECK_LE(y2, 1.0f);
            CHECK_GT(x2, x1);
            CHECK_GT(y2, y1);
        }
    }

    TEST_CASE("Deterministic: same input gives same output") {
        cv::dnn::Net net = load_model();

        cv::Mat image(300, 300, CV_8UC3, cv::Scalar(100, 120, 140));

        cv::Mat blob = cv::dnn::blobFromImage(image, 1.0, cv::Size(300, 300),
                                               cv::Scalar(104.0, 177.0, 123.0), false, false);

        // Run twice
        net.setInput(blob, "data");
        cv::Mat output1 = net.forward();

        // Re-set the same input
        net.setInput(blob, "data");
        cv::Mat output2 = net.forward();

        // Compare total data size
        size_t total_size1 = output1.total() * output1.elemSize();
        size_t total_size2 = output2.total() * output2.elemSize();
        CHECK_EQ(total_size1, total_size2);

        // Compare raw data
        bool identical = std::memcmp(output1.data, output2.data, total_size1) == 0;
        CHECK(identical);
    }

    TEST_CASE("Model metadata checks") {
        // Verify the deploy prototxt has expected structure
        std::ifstream proto_file(PROTOTXT_PATH);
        REQUIRE(proto_file.is_open());

        std::string content((std::istreambuf_iterator<char>(proto_file)),
                            std::istreambuf_iterator<char>());
        proto_file.close();

        // Should contain key architecture elements
        CHECK_NE(content.find("input: \"data\""), std::string::npos);
        CHECK_NE(content.find("dim: 300"), std::string::npos);
        CHECK_NE(content.find("Convolution"), std::string::npos);
        CHECK_NE(content.find("Pooling"), std::string::npos);

        // Verify the caffemodel file has reasonable size (should be ~10MB for this model)
        std::ifstream model_file(CAFFEMODEL_PATH, std::ios::binary | std::ios::ate);
        REQUIRE(model_file.is_open());

        std::streamsize size = model_file.tellg();
        model_file.close();

        CHECK_GT(size, 5 * 1024 * 1024L);   // > 5 MB
        CHECK_LT(size, 20 * 1024 * 1024L);  // < 20 MB
    }

    TEST_CASE("Full pipeline: image to detection") {
        cv::dnn::Net net = load_model();

        // Simulate full pipeline: read (synthetic), preprocess, detect
        cv::Mat img(300, 300, CV_8UC3, cv::Scalar(50, 50, 50));
        // Draw a bright centered rectangle simulating a face region
        cv::rectangle(img, cv::Rect(80, 60, 140, 180), cv::Scalar(200, 180, 160), -1);

        // Step 1: Create blob from image (preprocessing)
        cv::Mat blob = cv::dnn::blobFromImage(img, 1.0, cv::Size(300, 300),
                                               cv::Scalar(104.0, 177.0, 123.0),
                                               false,   // swapRB = false (BGR input)
                                               false);  // crop = false

        // Step 2: Set input and run forward pass
        net.setInput(blob, "data");
        cv::Mat detections = net.forward();

        // Step 3: Verify output structure
        CHECK_EQ(detections.size[0], 1);
        CHECK_EQ(detections.size[1], 1);
        CHECK_EQ(detections.size[3], 7);

        // Step 4: Parse and verify top detection
        float* data = (float*)detections.data;
        int num_detections = detections.size[2];

        float best_conf = 0.0f;
        float best_x1 = 0, best_y1 = 0, best_x2 = 0, best_y2 = 0;
        int best_class = -1;

        for (int i = 0; i < num_detections; i++) {
            float* detection = data + i * 7;
            float confidence = detection[2];
            if (confidence > best_conf) {
                best_conf = confidence;
                best_class = static_cast<int>(detection[1]);
                best_x1 = detection[3];
                best_y1 = detection[4];
                best_x2 = detection[5];
                best_y2 = detection[6];
            }
        }

        // The model should produce at least some detections with reasonable confidence
        CHECK_GT(best_conf, 0.0f);
        CHECK_GE(best_x2, best_x1);
        CHECK_GE(best_y2, best_y1);
    }

    TEST_CASE("Output layer name is accessible") {
        cv::dnn::Net net = load_model();

        // Get output layer names
        std::vector<cv::String> out_names = net.getUnconnectedOutLayersNames();
        CHECK_GT(out_names.size(), 0);

    }

    TEST_CASE("Model can be set to preferred backend") {
        cv::dnn::Net net = load_model();

        // Try setting CPU backend (always available)
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        cv::Mat image(300, 300, CV_8UC3, cv::Scalar(100, 100, 100));
        cv::Mat blob = cv::dnn::blobFromImage(image, 1.0, cv::Size(300, 300),
                                               cv::Scalar(104.0, 177.0, 123.0), false, false);

        net.setInput(blob, "data");
        cv::Mat output = net.forward();

        CHECK_FALSE(output.empty());
        CHECK_EQ(output.size[0], 1);
    }
}