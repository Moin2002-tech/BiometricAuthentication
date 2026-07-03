cmake_minimum_required(VERSION 3.20)
project(BiometricAuthentication)

set(TORCH_CUDA_ARCH_LIST "7.5")
set(CAFFE2_USE_CUDNN 9)
set(OpenCV_ROOT "${VCPKG_INSTALLED_DIR}/x64-linux/share/opencv4")

find_package(OpenCV CONFIG REQUIRED)

set(CMAKE_CXX_STANDARD 20)

add_executable(BiometricAuthentication
        external/third_party/doctest.cpp
        external/third_party/doctest.hpp
        Tests/main.cpp
        Tests/resnet50Test.cpp
        Tests/CaffeModelTests.cpp
        Tests/resnet50MS1MV3Tests.cpp
        Tests/cameraTests.cpp
        Tests/faceRecognitionTests.cpp
        include/Camera/CameraViewer.hpp
        src/Camera/CameraViewer.cpp
        include/Recognition/modelsHandler.hpp
        src/Recognition/modelsHandler.cpp
        include/Recognition/FaceRecognitionApp.hpp
        src/Recognition/FaceRecognitionApp.cpp
        include/Database/FaceDatabase.hpp
        src/Database/FaceDatabase.cpp
)

target_include_directories(BiometricAuthentication PRIVATE
        "${CMAKE_SOURCE_DIR}/external/libtorch/include"
        "${CMAKE_SOURCE_DIR}/external/libtorch/include/torch/csrc/api/include"
        "${OpenCV_INCLUDE_DIRS}"
        ${PROJECT_SOURCE_DIR}/external/third_party
        ${PROJECT_SOURCE_DIR}/include
)

add_definitions(-DMODELS_DIR="${CMAKE_SOURCE_DIR}/models")


target_link_libraries(BiometricAuthentication PRIVATE

        -Wl,--no-as-needed
        -Wl,--whole-archive
        "${CMAKE_SOURCE_DIR}/external/libtorch/lib/libtorch_cuda.so"
        "${CMAKE_SOURCE_DIR}/external/libtorch/lib/libtorch_cpu.so"
        "${CMAKE_SOURCE_DIR}/external/libtorch/lib/libtorch.so"
        -Wl,--no-whole-archive

        "${CMAKE_SOURCE_DIR}/external/libtorch/lib/libc10_cuda.so"
        "${CMAKE_SOURCE_DIR}/external/libtorch/lib/libc10.so"

        # 1. CUDA Runtime (from your toolkit)
        "/usr/local/cuda/lib64/libcudart.so"

        # 2. CUDA Driver (The one usually missing - found in system libs)
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1"

        # 3. NVRTC (Compiler library)
        "/usr/local/cuda/lib64/libnvrtc.so"

        ${OpenCV_LIBS}

        pthread
        dl
        m
)

# Standalone application (no test framework - runs the face recognition app directly)
add_executable(biometric-auth-app
        main.cpp
        include/Camera/CameraViewer.hpp
        src/Camera/CameraViewer.cpp
        include/Recognition/modelsHandler.hpp
        src/Recognition/modelsHandler.cpp
        include/Recognition/FaceRecognitionApp.hpp
        src/Recognition/FaceRecognitionApp.cpp
        include/Database/FaceDatabase.hpp
        src/Database/FaceDatabase.cpp
)

target_include_directories(biometric-auth-app PRIVATE
        "${CMAKE_SOURCE_DIR}/external/libtorch/include"
        "${CMAKE_SOURCE_DIR}/external/libtorch/include/torch/csrc/api/include"
        "${OpenCV_INCLUDE_DIRS}"
        ${PROJECT_SOURCE_DIR}/include
)

add_definitions(-DMODELS_DIR="${CMAKE_SOURCE_DIR}/models")

target_link_libraries(biometric-auth-app PRIVATE
        -Wl,--no-as-needed
        -Wl,--whole-archive
        "${CMAKE_SOURCE_DIR}/external/libtorch/lib/libtorch_cuda.so"
        "${CMAKE_SOURCE_DIR}/external/libtorch/lib/libtorch_cpu.so"
        "${CMAKE_SOURCE_DIR}/external/libtorch/lib/libtorch.so"
        -Wl,--no-whole-archive
        "${CMAKE_SOURCE_DIR}/external/libtorch/lib/libc10_cuda.so"
        "${CMAKE_SOURCE_DIR}/external/libtorch/lib/libc10.so"
        "/usr/local/cuda/lib64/libcudart.so"
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1"
        "/usr/local/cuda/lib64/libnvrtc.so"
        ${OpenCV_LIBS}
        pthread dl m
)

target_compile_options(biometric-auth-app PRIVATE -O3 -Wall)
set_target_properties(biometric-auth-app PROPERTIES
        INSTALL_RPATH "${CMAKE_SOURCE_DIR}/external/libtorch/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
)

target_compile_options(BiometricAuthentication PRIVATE -O3 -Wall)
set_target_properties(BiometricAuthentication PROPERTIES
        INSTALL_RPATH "${CMAKE_SOURCE_DIR}/external/libtorch/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
)
