//
// Created by moinshaikh on 6/30/26.
//

#include <doctest.hpp>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstring>

// Test suite for the MS1MV3 ArcFace backbone model
// Tests both ONNX (via OpenCV DNN) and TorchScript (via LibTorch) inference paths
TEST_SUITE("ArcFace MS1MV3 Backbone Tests") {

    const std::string TORCHSCRIPT_PATH = std::string(MODELS_DIR) + "/model.torchscript.pt";
    const std::string ONNX_PATH = std::string(MODELS_DIR) + "/model.onnx";



    torch::jit::Module load_torchscript_model() {
        torch::jit::Module model = torch::jit::load(TORCHSCRIPT_PATH);
        model.eval();
        return model;
    }



    cv::dnn::Net load_onnx_model() {
        cv::dnn::Net net = cv::dnn::readNetFromONNX(ONNX_PATH);
        REQUIRE_FALSE(net.empty());
        return net;
    }

    // Convert an OpenCV image (HWC, BGR, uint8) to Torch tensor (NCHW, float32, normalized to [-1, 1])
    torch::Tensor preprocess_cv_to_torch(const cv::Mat &img) {
        cv::Mat rgb;
        cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);

        // Convert to float tensor: HWC -> CHW, normalize to [0,1]
        torch::Tensor tensor = torch::from_blob(
                rgb.data,
                {112, 112, 3},
                torch::kByte
        ).to(torch::kFloat32) / 255.0f;

        // Permute HWC -> CHW and add batch dimension
        tensor = tensor.permute({2, 0, 1}).unsqueeze(0);

        // Normalize with mean=0.5, std=0.5 (maps [0,1] -> [-1, 1])
        tensor = (tensor - 0.5f) / 0.5f;

        return tensor;
    }

    // Create blob for OpenCV DNN ONNX (same preprocessing: scale 1/255, mean=0.5, std=0.5, RGB)
    cv::Mat preprocess_cv_to_blob(const cv::Mat &img) {
        // OpenCV's blobFromImage does: (img * scale) - mean
        // We want: (img / 255.0 - 0.5) / 0.5  =  img / 127.5 - 1.0
        // blobFromImage with scale=1/127.5, mean=1.0, swapRB=true gives: (img * (1/127.5)) - 1.0 = img/127.5 - 1.0
        // But our normalization is: ((img/255.0) - 0.5) / 0.5 = img/127.5 - 1.0
        // So scale=1/127.5, mean=(1.0, 1.0, 1.0), swapRB=true gives same result
        cv::Mat blob = cv::dnn::blobFromImage(img, 1.0 / 127.5, cv::Size(112, 112),
                                              cv::Scalar(1.0, 1.0, 1.0), true, false);
        return blob;
    }

    // ------------- Tests -------------

    TEST_CASE("TorchScript model loads successfully") {
        CHECK_NOTHROW(load_torchscript_model());
    }

    TEST_CASE("OpenCV ONNX model loads successfully") {
        // Verify ONNX file exists
        std::ifstream onnx_file(ONNX_PATH, std::ios::binary);
        REQUIRE(onnx_file.good());
        onnx_file.close();

        CHECK_NOTHROW(load_onnx_model());
    }

    TEST_CASE("TorchScript produces correct output shape [1, 512]") {
        auto model = load_torchscript_model();

        torch::Tensor input = torch::randn({1, 3, 112, 112});
        std::vector<torch::jit::IValue> inputs = {input};

        torch::Tensor output = model.forward(inputs).toTensor();

        CHECK_EQ(output.sizes().size(), 2);
        CHECK_EQ(output.size(0), 1);
        CHECK_EQ(output.size(1), 512);
    }

    TEST_CASE("OpenCV ONNX produces correct output shape [1, 512]") {
        auto net = load_onnx_model();

        cv::Mat image(112, 112, CV_8UC3, cv::Scalar(128, 128, 128));
        cv::randu(image, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

        cv::Mat blob = preprocess_cv_to_blob(image);
        net.setInput(blob, "input.1");
        cv::Mat output = net.forward();

        // Output should be 2D: [1, 512]
        CHECK_EQ(output.dims, 2);
        CHECK_EQ(output.size[0], 1);
        CHECK_EQ(output.size[1], 512);
        CHECK_EQ(output.type(), CV_32F);
    }

    TEST_CASE("TorchScript output has expected magnitude (norm > 1.0 for unnormalized backbone)") {
        auto model = load_torchscript_model();

        for (int i = 0; i < 5; i++) {
            torch::Tensor input = torch::randn({1, 3, 112, 112});
            torch::Tensor output = model.forward({input}).toTensor();
            float norm = output.norm().item<float>();

            // This model outputs unnormalized embeddings (norm ~8-12 range)
            CHECK_GT(norm, 1.0f);
            CHECK_LT(norm, 20.0f);
        }
    }

    TEST_CASE("Batch inference works correctly with TorchScript") {
        auto model = load_torchscript_model();

        torch::Tensor input = torch::randn({4, 3, 112, 112});
        torch::Tensor output = model.forward({input}).toTensor();

        CHECK_EQ(output.size(0), 4);
        CHECK_EQ(output.size(1), 512);

        // Each sample in batch should have a valid embedding (not NaN, finite)
        for (int i = 0; i < 4; i++) {
            float norm = output[i].norm().item<float>();
            CHECK_GT(norm, 1.0f);
            CHECK_LT(norm, 20.0f);
            CHECK_FALSE(std::isnan(norm));
        }
    }

    TEST_CASE("Deterministic: same input gives same output (TorchScript)") {
        auto model = load_torchscript_model();

        torch::Tensor input = torch::randn({1, 3, 112, 112});

        torch::Tensor output1 = model.forward({input}).toTensor();
        torch::Tensor output2 = model.forward({input}).toTensor();

        DOCTEST_CHECK(torch::allclose(output1, output2));
    }

    TEST_CASE("Deterministic: same input gives same output (OpenCV ONNX)") {
        auto net = load_onnx_model();

        cv::Mat image(112, 112, CV_8UC3, cv::Scalar(100, 120, 140));

        cv::Mat blob = preprocess_cv_to_blob(image);

        // Run twice
        net.setInput(blob, "input.1");
        cv::Mat output1 = net.forward();

        net.setInput(blob, "input.1");
        cv::Mat output2 = net.forward();

        size_t total_size = output1.total() * output1.elemSize();
        bool identical = std::memcmp(output1.data, output2.data, total_size) == 0;
        DOCTEST_CHECK(identical);
    }

    TEST_CASE("Different inputs produce different embeddings") {
        auto model = load_torchscript_model();

        torch::Tensor input1 = torch::randn({1, 3, 112, 112});
        torch::Tensor input2 = torch::randn({1, 3, 112, 112});

        torch::Tensor emb1 = model.forward({input1}).toTensor();
        torch::Tensor emb2 = model.forward({input2}).toTensor();

        float max_diff = (emb1 - emb2).abs().max().item<float>();
        CHECK_GT(max_diff, 1e-6f);
    }

    TEST_CASE("Full OpenCV to TorchTensor preprocessing pipeline") {
        auto model = load_torchscript_model();

        // Create a synthetic face-like image (112x112 BGR)
        cv::Mat img(112, 112, CV_8UC3, cv::Scalar(100, 120, 140));
        cv::randu(img, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

        // Run through preprocessing and inference
        torch::Tensor tensor = preprocess_cv_to_torch(img);
        torch::Tensor embedding = model.forward({tensor}).toTensor();

        // Verify output shape and valid embedding
        CHECK_EQ(embedding.size(0), 1);
        CHECK_EQ(embedding.size(1), 512);
        float norm = embedding.norm().item<float>();
        CHECK_GT(norm, 1.0f);
        CHECK_LT(norm, 20.0f);
        CHECK_FALSE(std::isnan(norm));
    }

    TEST_CASE("Model file metadata checks") {
        // Verify the TorchScript model file exists and has reasonable size
        std::ifstream ts_file(TORCHSCRIPT_PATH, std::ios::binary | std::ios::ate);
        REQUIRE(ts_file.is_open());

        std::streamsize ts_size = ts_file.tellg();
        ts_file.close();

        // Should be between 200MB and 300MB (ArcFace backbone with IR-SE50)
        CHECK_GT(ts_size, 200 * 1024 * 1024L);  // > 200 MB
        CHECK_LT(ts_size, 300 * 1024 * 1024L);  // < 300 MB

        // Verify the ONNX model file exists and has reasonable size
        std::ifstream onnx_file(ONNX_PATH, std::ios::binary | std::ios::ate);
        REQUIRE(onnx_file.is_open());

        std::streamsize onnx_size = onnx_file.tellg();
        onnx_file.close();

        // Should be > 200MB like the TorchScript version
        CHECK_GT(onnx_size, 200 * 1024 * 1024L);  // > 200 MB
        CHECK_LT(onnx_size, 300 * 1024 * 1024L);  // < 300 MB
    }

    TEST_CASE("TorchScript model runs on CPU") {
        auto model = load_torchscript_model();
        model.to(torch::kCPU);

        torch::Tensor input = torch::randn({1, 3, 112, 112});
        torch::Tensor output = model.forward({input}).toTensor();

        CHECK_EQ(output.device().type(), torch::kCPU);
        CHECK_EQ(output.size(1), 512);
    }

    TEST_CASE("OpenCV ONNX produces same output across multiple runs on same image") {
        auto net = load_onnx_model();

        cv::Mat image(112, 112, CV_8UC3, cv::Scalar(100, 120, 140));
        cv::randu(image, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

        cv::Mat blob = preprocess_cv_to_blob(image);

        // Run multiple times and collect results
        std::vector<cv::Mat> outputs;
        for (int i = 0; i < 3; i++) {
            net.setInput(blob.clone(), "input.1");
            cv::Mat out = net.forward().clone();
            outputs.push_back(out);
        }

        // All outputs should be identical
        for (size_t i = 1; i < outputs.size(); i++) {
            CHECK_EQ(outputs[0].total(), outputs[i].total());
            bool match = std::memcmp(outputs[0].data, outputs[i].data,
                                     outputs[0].total() * outputs[0].elemSize()) == 0;
            DOCTEST_CHECK(match);
        }
    }

}