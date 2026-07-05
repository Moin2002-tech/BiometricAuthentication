cmake_minimum_required(VERSION 3.20)
project(BiometricAuthentication)



set(CMAKE_CXX_STANDARD 17)

set(CMAKE_TOOLCHAIN_FILE "C:/dev/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain")


if(MSVC)
    add_compile_options(/utf-8 /Zc:__cplusplus)
endif()



find_package(OpenCV CONFIG REQUIRED)

set(OpenCV_ROOT "${VCPKG_INSTALLED_DIR}/x64-windows/share/opencv4")
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

target_compile_definitions(BiometricAuthentication PRIVATE
    MODELS_DIR="${CMAKE_SOURCE_DIR}/models"
)

target_link_libraries(BiometricAuthentication PRIVATE



        "${CMAKE_SOURCE_DIR}/external/libtorch/lib/torch.lib"
        "${CMAKE_SOURCE_DIR}/external/libtorch/lib/torch_cpu.lib"
        "${CMAKE_SOURCE_DIR}/external/libtorch/lib/c10.lib"
        "${CMAKE_SOURCE_DIR}/external/libtorch/lib/fmt.lib"


        ${OpenCV_LIBS}
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

target_compile_definitions(biometric-auth-app PRIVATE
    MODELS_DIR="${CMAKE_SOURCE_DIR}/models"
)

target_link_libraries(biometric-auth-app PRIVATE


        "${CMAKE_SOURCE_DIR}/external/libtorch/lib/torch.lib"
        "${CMAKE_SOURCE_DIR}/external/libtorch/lib/torch_cpu.lib"
        "${CMAKE_SOURCE_DIR}/external/libtorch/lib/c10.lib"
        "${CMAKE_SOURCE_DIR}/external/libtorch/lib/fmt.lib"


        ${OpenCV_LIBS}
)

#target_compile_options(biometric-auth-app PRIVATE -O3 -Wall)
#set_target_properties(biometric-auth-app PROPERTIES
  #      INSTALL_RPATH "${CMAKE_SOURCE_DIR}/external/libtorch/lib"
 #       BUILD_WITH_INSTALL_RPATH TRUE
#)

#target_compile_options(BiometricAuthentication PRIVATE -O3 -Wall)
#set_target_properties(BiometricAuthentication PROPERTIES
#        INSTALL_RPATH "${CMAKE_SOURCE_DIR}/external/libtorch/lib"
#        BUILD_WITH_INSTALL_RPATH TRUE
#)
