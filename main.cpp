 //
// Created by moinshaikh on 7/2/26.
//

/**
 * @brief Standalone entry point for the Face Recognition Application.
 *
 * This is the main application - NOT a test. Run this directly to use
 * the face recognition system with GUI.
 *
 * Controls (all on the OpenCV camera window):
 *   'a'    - Add current face (type name on screen, ENTER to confirm)
 *   SPACE  - Manual recognition trigger
 *   'q'    - Quit
 *
 * Build & Run:
 *   cd cmake-build-debug && ./BiometricAuthentication-app
 */

#include "include/Recognition/FaceRecognitionApp.hpp"
#include <iostream>

int main() {
    try {
        FaceRecognitionApp app;
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}