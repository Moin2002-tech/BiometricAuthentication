//
// Created by moinshaikh on 6/30/26.
//

#include <doctest.hpp>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

// Test suite for the ArcFace IR-SE50 backbone model exported as TorchScript
TEST_SUITE("ResNet50 Backbone Tests") {

    // Load the model once for all tests
    torch::jit::Module load_model() {
        std::string model_path = std::string(MODELS_DIR) + "/backbone.torchscript.pt";
        torch::jit::Module model = torch::jit::load(model_path);
        model.eval();
        return model;
    }

    TEST_CASE("Model loads successfully") {
        CHECK_NOTHROW(load_model());
    }

    TEST_CASE("Model produces correct output shape [1, 512]") {
        auto model = load_model();

        // Create random input matching expected size: 1x3x112x112
        torch::Tensor input = torch::randn({1, 3, 112, 112});
        std::vector<torch::jit::IValue> inputs = {input};

        torch::Tensor output = model.forward(inputs).toTensor();

        // Check output shape
        CHECK_EQ(output.sizes().size(), 2);              // 2D tensor
        CHECK_EQ(output.size(0), 1);                      // batch size = 1
        CHECK_EQ(output.size(1), 512);                    // embedding dimension = 512
    }

    TEST_CASE("Output is L2-normalized (norm = 1.0)") {
        auto model = load_model();

        // Test with multiple random inputs
        for (int i = 0; i < 5; i++) {
            torch::Tensor input = torch::randn({1, 3, 112, 112});
            std::vector<torch::jit::IValue> inputs = {input};

            torch::Tensor output = model.forward(inputs).toTensor();
            float norm = output.norm().item<float>();

            // L2 norm should be approximately 1.0 (within float tolerance)
            CHECK_EQ(doctest::Approx(norm).epsilon(0.001), 1.0f);
        }
    }

    TEST_CASE("Batch inference works correctly") {
        auto model = load_model();

        // Test with batch size 4
        torch::Tensor input = torch::randn({4, 3, 112, 112});
        std::vector<torch::jit::IValue> inputs = {input};

        torch::Tensor output = model.forward(inputs).toTensor();

        CHECK_EQ(output.size(0), 4);    // 4 embeddings
        CHECK_EQ(output.size(1), 512);  // each 512-d

        // Each sample in batch should be L2-normalized
        for (int i = 0; i < 4; i++) {
            float norm = output[i].norm().item<float>();
            CHECK_EQ(doctest::Approx(norm).epsilon(0.001), 1.0f);
        }
    }

    TEST_CASE("Deterministic: same input gives same output") {
        auto model = load_model();

        torch::Tensor input = torch::randn({1, 3, 112, 112});
        std::vector<torch::jit::IValue> inputs = {input};

        // Run twice
        torch::Tensor output1 = model.forward(inputs).toTensor();
        torch::Tensor output2 = model.forward(inputs).toTensor();

        // Should be identical
        CHECK(torch::allclose(output1, output2));
    }

    TEST_CASE("Different inputs produce different embeddings") {
        auto model = load_model();

        // Two different random inputs
        torch::Tensor input1 = torch::randn({1, 3, 112, 112});
        torch::Tensor input2 = torch::randn({1, 3, 112, 112});

        torch::Tensor emb1 = model.forward({input1}).toTensor();
        torch::Tensor emb2 = model.forward({input2}).toTensor();

        // Even though cosine similarity may be high for non-face inputs,
        // embeddings should NOT be identical
        float max_diff = (emb1 - emb2).abs().max().item<float>();
        CHECK_GT(max_diff, 1e-6f);
    }

    TEST_CASE("Synthetic face-like input has meaningful variance") {
        auto model = load_model();

        // Generate two different synthetic face-like inputs (not pure noise)
        cv::Mat img1(112, 112, CV_8UC3, cv::Scalar(100, 100, 100));
        cv::Mat img2(112, 112, CV_8UC3, cv::Scalar(140, 140, 140));
        cv::randu(img1, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
        cv::randu(img2, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

        auto to_tensor = [](cv::Mat &img) {
            cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
            torch::Tensor t = torch::from_blob(img.data, {112, 112, 3}, torch::kByte)
                .to(torch::kFloat32) / 255.0f;
            t = t.permute({2, 0, 1}).unsqueeze(0);
            return (t - 0.5f) / 0.5f;
        };

        torch::Tensor emb1 = model.forward({to_tensor(img1)}).toTensor();
        torch::Tensor emb2 = model.forward({to_tensor(img2)}).toTensor();

        // Verify different inputs produce different embeddings
        float max_diff = (emb1 - emb2).abs().max().item<float>();
        CHECK_GT(max_diff, 1e-6f);
    }

    TEST_CASE("Full OpenCV to TorchTensor preprocessing pipeline") {
        auto model = load_model();

        // Create a synthetic face-like image (112x112 RGB)
        cv::Mat img(112, 112, CV_8UC3, cv::Scalar(100, 120, 140));
        // Add some random noise to make it more realistic
        cv::randu(img, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

        // Convert BGR (OpenCV default) to RGB
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

        // Convert to float tensor: HWC -> CHW, normalize to [0,1]
        torch::Tensor tensor = torch::from_blob(
            img.data,
            {112, 112, 3},
            torch::kByte
        ).to(torch::kFloat32) / 255.0f;

        // Permute HWC -> CHW and add batch dimension
        tensor = tensor.permute({2, 0, 1}).unsqueeze(0);

        // Normalize with mean=0.5, std=0.5 (as used during export)
        tensor = (tensor - 0.5f) / 0.5f;

        // Run inference
        torch::Tensor embedding = model.forward({tensor}).toTensor();

        // Verify
        CHECK_EQ(embedding.size(0), 1);
        CHECK_EQ(embedding.size(1), 512);
        float norm = embedding.norm().item<float>();
        CHECK_EQ(doctest::Approx(norm).epsilon(0.001), 1.0f);
    }

    TEST_CASE("Model runs on CPU") {
        auto model = load_model();
        model.to(torch::kCPU);

        torch::Tensor input = torch::randn({1, 3, 112, 112});
        torch::Tensor output = model.forward({input}).toTensor();

        CHECK_EQ(output.device().type(), torch::kCPU);
        CHECK_EQ(output.size(1), 512);
    }

    TEST_CASE("Partial model metadata") {
        // Verify the model file exists and has reasonable size
        std::string model_path = std::string(MODELS_DIR) + "/backbone.torchscript.pt";

        std::ifstream file(model_path, std::ios::binary | std::ios::ate);
        REQUIRE(file.is_open());

        std::streamsize size = file.tellg();
        file.close();

        // Should be between 100MB and 300MB (for IR-SE50)
        CHECK_GT(size, 100 * 1024 * 1024L);  // > 100 MB
        CHECK_LT(size, 300 * 1024 * 1024L);  // < 300 MB
    }
}