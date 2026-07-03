               //
// Created by moinshaikh on 7/2/26.
//

/**
 * @brief Entry point for the Face Recognition Application with Database.
 *
 * This test showcases:
 *   1. Loading face detection + recognition models
 *   2. Real-time camera feed with face bounding box overlay
 *   3. Automatic face recognition against the database
 *   4. Adding new persons to the database with ID, name, photo, and embedding
 *   5. Persistent storage of face data in databases/ folder
 *
 * Controls:
 *   SPACE - Capture & recognize current frame
 *   'a'   - Add current face to database with a name
 *   'q'   - Quit
 */

#include <doctest.hpp>
#include "Recognition/FaceRecognitionApp.hpp"

TEST_CASE("FaceRecognitionApplicationwithDatabase") {
    FaceRecognitionApp app;
    app.run();
}