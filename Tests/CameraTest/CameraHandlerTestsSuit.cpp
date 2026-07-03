//
// Created by moinshaikh on 7/1/26.
//

#include <doctest.hpp>
#include <opencv2/core/core.hpp>
#include <cmath>
#include "Camera/CameraHandler.hpp"

using namespace Camera;

TEST_SUITE("CameraHandler")
{

    // =====================================================
    // FrustumHandler Tests
    // =====================================================

    TEST_CASE("FrustumHandler Default Constructor")
    {
        FrustumHandler f;
        CHECK(f.left_ == doctest::Approx(-1.0f));
        CHECK(f.right_ == doctest::Approx(1.0f));
        CHECK(f.bottom_ == doctest::Approx(-1.0f));
        CHECK(f.top_ == doctest::Approx(1.0f));
        CHECK(f.nearPlane_ == doctest::Approx(0.1f));
        CHECK(f.farPlane_ == doctest::Approx(100.0f));
    }

    TEST_CASE("FrustumHandler Parameterized Constructor")
    {
        FrustumHandler f(-2.0f, 2.0f, -1.5f, 1.5f, 0.5f, 200.0f);
        CHECK(f.left_ == doctest::Approx(-2.0f));
        CHECK(f.right_ == doctest::Approx(2.0f));
        CHECK(f.bottom_ == doctest::Approx(-1.5f));
        CHECK(f.top_ == doctest::Approx(1.5f));
        CHECK(f.nearPlane_ == doctest::Approx(0.5f));
        CHECK(f.farPlane_ == doctest::Approx(200.0f));
    }

    TEST_CASE("FrustumHandler Extreme Values")
    {
        FrustumHandler f(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
        CHECK(f.left_ == doctest::Approx(0.0f));
        CHECK(f.right_ == doctest::Approx(0.0f));
        CHECK(f.nearPlane_ == doctest::Approx(0.0f));
        CHECK(f.farPlane_ == doctest::Approx(0.0f));
    }

    // =====================================================
    // Rotation Matrix Helper Tests
    // =====================================================

    TEST_CASE("createRotationMatrixX Zero Angle")
    {
        cv::Mat rot = createRotationMatrixX(0.0f);
        CHECK(rot.rows == 4);
        CHECK(rot.cols == 4);
        // Should be identity matrix
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                CHECK(rot.at<float>(i, j) == doctest::Approx((i == j) ? 1.0f : 0.0f));
    }

    TEST_CASE("createRotationMatrixX PI_2")
    {
        float angle = static_cast<float>(M_PI_2);
        cv::Mat rot = createRotationMatrixX(angle);
        // cos(pi/2) = 0, sin(pi/2) = 1
        CHECK(rot.at<float>(1, 1) == doctest::Approx(0.0f));
        CHECK(rot.at<float>(1, 2) == doctest::Approx(-1.0f));
        CHECK(rot.at<float>(2, 1) == doctest::Approx(1.0f));
        CHECK(rot.at<float>(2, 2) == doctest::Approx(0.0f));
        // Top row should be (1,0,0,0)
        CHECK(rot.at<float>(0, 0) == doctest::Approx(1.0f));
        CHECK(rot.at<float>(0, 1) == doctest::Approx(0.0f));
        CHECK(rot.at<float>(0, 2) == doctest::Approx(0.0f));
        CHECK(rot.at<float>(0, 3) == doctest::Approx(0.0f));
        // Bottom row should be (0,0,0,1)
        CHECK(rot.at<float>(3, 0) == doctest::Approx(0.0f));
        CHECK(rot.at<float>(3, 1) == doctest::Approx(0.0f));
        CHECK(rot.at<float>(3, 2) == doctest::Approx(0.0f));
        CHECK(rot.at<float>(3, 3) == doctest::Approx(1.0f));
    }

    TEST_CASE("createRotationMatrixX Negative Angle")
    {
        float angle = -static_cast<float>(M_PI_4);
        cv::Mat rot = createRotationMatrixX(angle);
        // Rotation around X should preserve volume (determinant = 1)
        double det = cv::determinant(rot);
        CHECK(det == doctest::Approx(1.0).epsilon(1e-5));
    }

    TEST_CASE("createRotationMatrixY Zero Angle")
    {
        cv::Mat rot = createRotationMatrixY(0.0f);
        CHECK(rot.rows == 4);
        CHECK(rot.cols == 4);
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                CHECK(rot.at<float>(i, j) == doctest::Approx((i == j) ? 1.0f : 0.0f));
    }

    TEST_CASE("createRotationMatrixY PI_2")
    {
        float angle = static_cast<float>(M_PI_2);
        cv::Mat rot = createRotationMatrixY(angle);
        // cos(pi/2) = 0, sin(pi/2) = 1
        // Row0: cos, 0, sin, 0 => 0, 0, 1, 0
        CHECK(rot.at<float>(0, 0) == doctest::Approx(0.0f));
        CHECK(rot.at<float>(0, 2) == doctest::Approx(1.0f));
        // Row1: 0, 1, 0, 0
        CHECK(rot.at<float>(1, 0) == doctest::Approx(0.0f));
        CHECK(rot.at<float>(1, 1) == doctest::Approx(1.0f));
        CHECK(rot.at<float>(1, 2) == doctest::Approx(0.0f));
        // Row2: -sin, 0, cos, 0 => -1, 0, 0, 0
        CHECK(rot.at<float>(2, 0) == doctest::Approx(-1.0f));
        CHECK(rot.at<float>(2, 1) == doctest::Approx(0.0f));
        CHECK(rot.at<float>(2, 2) == doctest::Approx(0.0f));
        // Bottom row: 0, 0, 0, 1
        CHECK(rot.at<float>(3, 3) == doctest::Approx(1.0f));
    }

    TEST_CASE("Rotation Matrix Determinant Is One")
    {
        float angleX = static_cast<float>(M_PI / 3.0);
        float angleY = static_cast<float>(M_PI / 6.0);
        cv::Mat rotX = createRotationMatrixX(angleX);
        cv::Mat rotY = createRotationMatrixY(angleY);
        CHECK(cv::determinant(rotX) == doctest::Approx(1.0).epsilon(1e-5));
        CHECK(cv::determinant(rotY) == doctest::Approx(1.0).epsilon(1e-5));
    }

    // =====================================================
    // CameraHandler Constructor Tests
    // =====================================================

    TEST_CASE("CameraHandler Default Constructor")
    {
        CameraHandler cam;
        // Default eye should be (0,0,0)
        CHECK(cam.getEye()[0] == doctest::Approx(0.0f));
        CHECK(cam.getEye()[1] == doctest::Approx(0.0f));
        CHECK(cam.getEye()[2] == doctest::Approx(0.0f));
        // Default gaze should be (0,0,-1)
        CHECK(cam.getAt()[0] == doctest::Approx(0.0f));
        CHECK(cam.getAt()[1] == doctest::Approx(0.0f));
        CHECK(cam.getAt()[2] == doctest::Approx(-1.0f));
        // Default up should be (0,1,0)
        CHECK(cam.getUp()[0] == doctest::Approx(0.0f));
        CHECK(cam.getUp()[1] == doctest::Approx(1.0f));
        CHECK(cam.getUp()[2] == doctest::Approx(0.0f));
        // Horizontal and vertical angles should be 0
        CHECK(cam.getHorizontalAngle() == doctest::Approx(0.0f));
        CHECK(cam.getVerticalAngle() == doctest::Approx(0.0f));
        // Vector lengths should be 1 (unit vectors)
        CHECK(cv::norm(cam.getForwardVector(), cv::NORM_L2) == doctest::Approx(1.0f).epsilon(1e-5));
        CHECK(cv::norm(cam.getRightVector(), cv::NORM_L2) == doctest::Approx(1.0f).epsilon(1e-5));
        CHECK(cv::norm(cam.getUpVector(), cv::NORM_L2) == doctest::Approx(1.0f).epsilon(1e-5));
    }

    TEST_CASE("CameraHandler Constructor With FrustumHandler")
    {
        FrustumHandler frustum(-5.0f, 5.0f, -4.0f, 4.0f, 1.0f, 50.0f);
        CameraHandler cam(frustum);
        // Should have default eye/gaze/up like default constructor
        CHECK(cam.getEye()[0] == doctest::Approx(0.0f));
        CHECK(cam.getEye()[1] == doctest::Approx(0.0f));
        CHECK(cam.getEye()[2] == doctest::Approx(0.0f));
        CHECK(cam.getAt()[0] == doctest::Approx(0.0f));
        CHECK(cam.getAt()[1] == doctest::Approx(0.0f));
        CHECK(cam.getAt()[2] == doctest::Approx(-1.0f));
        CHECK(cam.getUp()[1] == doctest::Approx(1.0f));
        CHECK(cam.getHorizontalAngle() == doctest::Approx(0.0f));
        CHECK(cam.getVerticalAngle() == doctest::Approx(0.0f));
    }

    TEST_CASE("CameraHandler Constructor With EyePosition GazePosition Frustum")
    {
        FrustumHandler frustum;
        cv::Vec3f eye(1.0f, 2.0f, 3.0f);
        cv::Vec3f gaze(0.0f, 0.0f, -5.0f);
        CameraHandler cam(eye, gaze, frustum);

        CHECK(cam.getEye()[0] == doctest::Approx(1.0f));
        CHECK(cam.getEye()[1] == doctest::Approx(2.0f));
        CHECK(cam.getEye()[2] == doctest::Approx(3.0f));
        CHECK(cam.getAt()[0] == doctest::Approx(0.0f));
        CHECK(cam.getAt()[1] == doctest::Approx(0.0f));
        CHECK(cam.getAt()[2] == doctest::Approx(-5.0f));
        CHECK(cam.getUp()[1] == doctest::Approx(1.0f));
        CHECK(cam.getHorizontalAngle() == doctest::Approx(0.0f));
        CHECK(cam.getVerticalAngle() == doctest::Approx(0.0f));

        // Forward should be normalized opposite of gaze: (0,0,5)/5 = (0,0,1)
        CHECK(cam.getForwardVector()[0] == doctest::Approx(0.0f));
        CHECK(cam.getForwardVector()[1] == doctest::Approx(0.0f));
        CHECK(cam.getForwardVector()[2] == doctest::Approx(1.0f));
    }

    TEST_CASE("CameraHandler Constructor With Eye And Angles Frustum")
    {
        FrustumHandler frustum;
        cv::Vec3f eye(5.0f, 0.0f, 0.0f);
        float hAngle = static_cast<float>(M_PI_4);  // 45 degrees
        float vAngle = 0.0f;
        CameraHandler cam(eye, hAngle, vAngle, frustum);

        CHECK(cam.getEye()[0] == doctest::Approx(5.0f));
        CHECK(cam.getEye()[1] == doctest::Approx(0.0f));
        CHECK(cam.getEye()[2] == doctest::Approx(0.0f));
        CHECK(cam.getHorizontalAngle() == doctest::Approx(hAngle));
        CHECK(cam.getVerticalAngle() == doctest::Approx(vAngle));

        // With 45 degree horizontal rotation from initial (0,0,-1):
        // forward = Ry(45) * Rx(0) * (0,0,-1)
        // Ry(45) = [cos45, 0, sin45; 0,1,0; -sin45,0,cos45]
        // Ry(45)*(0,0,-1): x = sin45*(-1) = -0.707, y = 0, z = cos45*(-1) = -0.707
        float s = static_cast<float>(M_SQRT2 / 2.0);
        CHECK(cam.getForwardVector()[0] == doctest::Approx(-s).epsilon(1e-5f));
        CHECK(cam.getForwardVector()[1] == doctest::Approx(0.0f).epsilon(1e-5f));
        CHECK(cam.getForwardVector()[2] == doctest::Approx(-s).epsilon(1e-5f));

        // Gaze = -forward: (s, 0, s)
        CHECK(cam.getAt()[0] == doctest::Approx(s).epsilon(1e-5f));
        CHECK(cam.getAt()[1] == doctest::Approx(0.0f).epsilon(1e-5f));
        CHECK(cam.getAt()[2] == doctest::Approx(s).epsilon(1e-5f));
    }

    // =====================================================
    // CameraHandler updateParameters Tests
    // =====================================================

    TEST_CASE("updateParameters Basic")
    {
        CameraHandler cam;
        cv::Vec3f newEye(10.0f, 20.0f, 30.0f);
        cv::Vec3f newGaze(1.0f, 0.0f, 0.0f);  // Looking along +X axis
        cam.updateParameters(newEye, newGaze);

        CHECK(cam.getEye()[0] == doctest::Approx(10.0f));
        CHECK(cam.getEye()[1] == doctest::Approx(20.0f));
        CHECK(cam.getEye()[2] == doctest::Approx(30.0f));
        CHECK(cam.getAt()[0] == doctest::Approx(1.0f));
        CHECK(cam.getAt()[1] == doctest::Approx(0.0f));
        CHECK(cam.getAt()[2] == doctest::Approx(0.0f));

        // Gaze is (1,0,0), so forward should be (-1,0,0) normalized
        CHECK(cam.getForwardVector()[0] == doctest::Approx(-1.0f));
        CHECK(cam.getForwardVector()[1] == doctest::Approx(0.0f));
        CHECK(cam.getForwardVector()[2] == doctest::Approx(0.0f));
        // Norm of forward should be 1
        CHECK(cv::norm(cam.getForwardVector(), cv::NORM_L2) == doctest::Approx(1.0f));
    }

    TEST_CASE("updateParameters With Custom Up Vector")
    {
        CameraHandler cam;
        cv::Vec3f newEye(0.0f, 0.0f, 0.0f);
        cv::Vec3f newGaze(0.0f, -1.0f, 0.0f);  // Looking down along -Y
        cv::Vec3f customUp(0.0f, 0.0f, 1.0f);  // Z is up

        cam.updateParameters(newEye, newGaze, customUp);

        CHECK(cam.getUp()[0] == doctest::Approx(0.0f));
        CHECK(cam.getUp()[1] == doctest::Approx(0.0f));
        CHECK(cam.getUp()[2] == doctest::Approx(1.0f));

        // Gaze = (0,-1,0), so forward = (0,1,0) normalized
        CHECK(cam.getForwardVector()[0] == doctest::Approx(0.0f));
        CHECK(cam.getForwardVector()[1] == doctest::Approx(1.0f));
        CHECK(cam.getForwardVector()[2] == doctest::Approx(0.0f));

        // Right = up x forward = (0,0,1) x (0,1,0) = (-1,0,0)
        CHECK(cam.getRightVector()[0] == doctest::Approx(-1.0f));
        CHECK(cam.getRightVector()[1] == doctest::Approx(0.0f));
        CHECK(cam.getRightVector()[2] == doctest::Approx(0.0f));

        // UpVector = forward x right = (0,1,0) x (-1,0,0) = (0,0,1) = customUp
        CHECK(cam.getUpVector()[0] == doctest::Approx(0.0f));
        CHECK(cam.getUpVector()[1] == doctest::Approx(0.0f));
        CHECK(cam.getUpVector()[2] == doctest::Approx(1.0f));
    }

    TEST_CASE("updateParameters Orthogonal Basis Check")
    {
        CameraHandler cam;
        cv::Vec3f eye(0.0f, 0.0f, 0.0f);
        cv::Vec3f gaze(1.0f, 2.0f, 3.0f);
        cam.updateParameters(eye, gaze);

        // forward, right, up should form an orthonormal basis
        const Vec3f& f = cam.getForwardVector();
        const Vec3f& r = cam.getRightVector();
        const Vec3f& u = cam.getUpVector();

        // Each vector should have unit length
        CHECK(cv::norm(f, cv::NORM_L2) == doctest::Approx(1.0f).epsilon(1e-5f));
        CHECK(cv::norm(r, cv::NORM_L2) == doctest::Approx(1.0f).epsilon(1e-5f));
        CHECK(cv::norm(u, cv::NORM_L2) == doctest::Approx(1.0f).epsilon(1e-5f));

        // Dot products should be ~0 (orthogonal)
        CHECK(std::abs(f.dot(r)) == doctest::Approx(0.0f).epsilon(1e-5f));
        CHECK(std::abs(f.dot(u)) == doctest::Approx(0.0f).epsilon(1e-5f));
        CHECK(std::abs(r.dot(u)) == doctest::Approx(0.0f).epsilon(1e-5f));

        // Cross product check: f x r should equal u
        Vec3f fCrossR = f.cross(r);
        CHECK(fCrossR[0] == doctest::Approx(u[0]).epsilon(1e-5f));
        CHECK(fCrossR[1] == doctest::Approx(u[1]).epsilon(1e-5f));
        CHECK(fCrossR[2] == doctest::Approx(u[2]).epsilon(1e-5f));
    }

    TEST_CASE("updateParameters Gaze Along Up Direction")
    {
        CameraHandler cam;
        cv::Vec3f eye(0.0f, 0.0f, 0.0f);
        cv::Vec3f gaze(0.0f, 1.0f, 0.0f);  // Looking up along Y (same as default up)
        // This is a degenerate case where up and gaze are parallel
        // The right vector would be zero, which could cause division by zero
        // We just check that the function doesn't crash and produces a forward = -gaze
        cam.updateParameters(eye, gaze);
        // After update, forward = (0,-1,0) (normalized -gaze)
        CHECK(cam.getForwardVector()[0] == doctest::Approx(0.0f));
        CHECK(cam.getForwardVector()[1] == doctest::Approx(-1.0f));
        CHECK(cam.getForwardVector()[2] == doctest::Approx(0.0f));
    }

    // =====================================================
    // CameraHandler updateForwards Tests
    // =====================================================

    TEST_CASE("updateForwards Zero Angles")
    {
        CameraHandler cam;
        cv::Vec3f eye(0.0f, 0.0f, 5.0f);
        cam.updateForwards(eye);

        // With zero angles: forward should be (0,0,-1)
        CHECK(cam.getForwardVector()[0] == doctest::Approx(0.0f));
        CHECK(cam.getForwardVector()[1] == doctest::Approx(0.0f));
        CHECK(cam.getForwardVector()[2] == doctest::Approx(-1.0f));

        // Gaze = -forward: (0,0,1)
        CHECK(cam.getAt()[0] == doctest::Approx(0.0f));
        CHECK(cam.getAt()[1] == doctest::Approx(0.0f));
        CHECK(cam.getAt()[2] == doctest::Approx(1.0f));

        // Right should be (1,0,0) from Ry(0)*Rx(0)*(1,0,0)
        CHECK(cam.getRightVector()[0] == doctest::Approx(1.0f));
        CHECK(cam.getRightVector()[1] == doctest::Approx(0.0f));
        CHECK(cam.getRightVector()[2] == doctest::Approx(0.0f));

        // UpVector = forward x right = (0,0,-1) x (1,0,0) = (0,-1,0)
        CHECK(cam.getUpVector()[0] == doctest::Approx(0.0f));
        CHECK(cam.getUpVector()[1] == doctest::Approx(-1.0f));
        CHECK(cam.getUpVector()[2] == doctest::Approx(0.0f));

        CHECK(cam.getEye()[0] == doctest::Approx(0.0f));
        CHECK(cam.getEye()[1] == doctest::Approx(0.0f));
        CHECK(cam.getEye()[2] == doctest::Approx(5.0f));
    }

    TEST_CASE("updateForwards Horizontal Rotation 90 Degrees")
    {
        CameraHandler cam(
            cv::Vec3f(0.0f, 0.0f, 0.0f),
            static_cast<float>(M_PI_2),   // 90 degrees horizontal
            0.0f,
            FrustumHandler()
        );

        const Vec3f& f = cam.getForwardVector();
        const Vec3f& r = cam.getRightVector();
        const Vec3f& u = cam.getUpVector();

        // Ry(90)*(0,0,-1):
        // Ry = [cos, 0, sin; 0,1,0; -sin,0,cos]
        // x' = cos90*0 + 0*0 + sin90*(-1) = 1*(-1) = -1
        // y' = 0
        // z' = -sin90*0 + 0*0 + cos90*(-1) = 0
        // forward = (-1,0,0)
        CHECK(f[0] == doctest::Approx(-1.0f).epsilon(1e-5f));
        CHECK(f[1] == doctest::Approx(0.0f).epsilon(1e-5f));
        CHECK(f[2] == doctest::Approx(0.0f).epsilon(1e-5f));

        // Ry(90)*(1,0,0):
        // x' = 0*1 + 0*0 + 1*0 = 0
        // y' = 0
        // z' = -1*1 + 0*0 + 0*0 = -1
        // right = (0,0,-1)
        CHECK(r[0] == doctest::Approx(0.0f).epsilon(1e-5f));
        CHECK(r[1] == doctest::Approx(0.0f).epsilon(1e-5f));
        CHECK(r[2] == doctest::Approx(-1.0f).epsilon(1e-5f));

        // upVector = forward x right = (-1,0,0) x (0,0,-1) = (0, -1, 0)
        CHECK(u[0] == doctest::Approx(0.0f).epsilon(1e-5f));
        CHECK(u[1] == doctest::Approx(-1.0f).epsilon(1e-5f));
        CHECK(u[2] == doctest::Approx(0.0f).epsilon(1e-5f));

        // Orthonormal check
        CHECK(cv::norm(f, cv::NORM_L2) == doctest::Approx(1.0f).epsilon(1e-5f));
        CHECK(cv::norm(r, cv::NORM_L2) == doctest::Approx(1.0f).epsilon(1e-5f));
        CHECK(cv::norm(u, cv::NORM_L2) == doctest::Approx(1.0f).epsilon(1e-5f));
        CHECK(std::abs(f.dot(r)) == doctest::Approx(0.0f).epsilon(1e-5f));
        CHECK(std::abs(f.dot(u)) == doctest::Approx(0.0f).epsilon(1e-5f));
    }

    TEST_CASE("updateForwards Vertical Rotation 90 Degrees")
    {
        CameraHandler cam(
            cv::Vec3f(0.0f, 0.0f, 0.0f),
            0.0f,
            static_cast<float>(M_PI_2),   // 90 degrees vertical (pitch down)
            FrustumHandler()
        );

        const Vec3f& f = cam.getForwardVector();

        // Rx(90) * (0,0,-1):
        // Rx = [1,0,0; 0,cos,-sin; 0,sin,cos]
        // x' = 0
        // y' = cos90*0 + (-sin90)*(-1) = (-1)*(-1) = 1
        // z' = sin90*0 + cos90*(-1) = 0
        // So forward = (0,1,0)
        CHECK(f[0] == doctest::Approx(0.0f).epsilon(1e-5f));
        CHECK(f[1] == doctest::Approx(1.0f).epsilon(1e-5f));
        CHECK(f[2] == doctest::Approx(0.0f).epsilon(1e-5f));

        // Gaze = -forward = (0,-1,0)
        CHECK(cam.getAt()[0] == doctest::Approx(0.0f).epsilon(1e-5f));
        CHECK(cam.getAt()[1] == doctest::Approx(-1.0f).epsilon(1e-5f));
        CHECK(cam.getAt()[2] == doctest::Approx(0.0f).epsilon(1e-5f));
    }

    TEST_CASE("updateForwards Combined Rotation")
    {
        // Set angles via the constructor
        CameraHandler cam2(
            cv::Vec3f(2.0f, 3.0f, 4.0f),
            static_cast<float>(M_PI / 4.0f),   // 45 deg horizontal
            static_cast<float>(M_PI / 6.0f),   // 30 deg vertical
            FrustumHandler()
        );

        const Vec3f& f = cam2.getForwardVector();
        const Vec3f& r = cam2.getRightVector();
        const Vec3f& u = cam2.getUpVector();

        // Orthonormal check
        CHECK(cv::norm(f, cv::NORM_L2) == doctest::Approx(1.0f).epsilon(1e-5f));
        CHECK(cv::norm(r, cv::NORM_L2) == doctest::Approx(1.0f).epsilon(1e-5f));
        CHECK(cv::norm(u, cv::NORM_L2) == doctest::Approx(1.0f).epsilon(1e-5f));
        CHECK(std::abs(f.dot(r)) == doctest::Approx(0.0f).epsilon(1e-5f));
        CHECK(std::abs(f.dot(u)) == doctest::Approx(0.0f).epsilon(1e-5f));
        CHECK(std::abs(r.dot(u)) == doctest::Approx(0.0f).epsilon(1e-5f));

        // Eye position should be preserved
        CHECK(cam2.getEye()[0] == doctest::Approx(2.0f));
        CHECK(cam2.getEye()[1] == doctest::Approx(3.0f));
        CHECK(cam2.getEye()[2] == doctest::Approx(4.0f));
    }

    TEST_CASE("updateForwards With Custom Up Vector")
    {
        CameraHandler cam;
        cv::Vec3f eye(0.0f, 0.0f, 0.0f);
        cv::Vec3f customUp(0.0f, 0.0f, 1.0f);  // Z is up

        cam.updateForwards(eye, customUp);

        // With zero angles:
        // forward = (0,0,-1), right = (1,0,0)
        // upVector = forward x right = (0,0,-1) x (1,0,0) = (0,-1,0), not matching customUp!
        // This is because updateForwards computes upVector from forward x right, ignoring customUp
        // But getUp() should return customUp
        CHECK(cam.getUp()[0] == doctest::Approx(0.0f));
        CHECK(cam.getUp()[1] == doctest::Approx(0.0f));
        CHECK(cam.getUp()[2] == doctest::Approx(1.0f));
    }

    // =====================================================
    // CameraHandler Chain of Updates Tests
    // =====================================================

    TEST_CASE("Multiple updateParameters Calls")
    {
        CameraHandler cam;
        cv::Vec3f eye1(1.0f, 0.0f, 0.0f);
        cv::Vec3f gaze1(0.0f, 0.0f, -1.0f);
        cam.updateParameters(eye1, gaze1);

        CHECK(cam.getEye()[0] == doctest::Approx(1.0f));

        // Second update should overwrite
        cv::Vec3f eye2(10.0f, 20.0f, 30.0f);
        cv::Vec3f gaze2(0.0f, 1.0f, 0.0f);
        cam.updateParameters(eye2, gaze2);

        CHECK(cam.getEye()[0] == doctest::Approx(10.0f));
        CHECK(cam.getEye()[1] == doctest::Approx(20.0f));
        CHECK(cam.getEye()[2] == doctest::Approx(30.0f));
        CHECK(cam.getAt()[0] == doctest::Approx(0.0f));
        CHECK(cam.getAt()[1] == doctest::Approx(1.0f));
        CHECK(cam.getAt()[2] == doctest::Approx(0.0f));
    }

    TEST_CASE("updateParameters Then updateForwards")
    {
        CameraHandler cam(
            cv::Vec3f(0.0f, 0.0f, 0.0f),
            static_cast<float>(M_PI_4),  // 45 deg horizontal
            0.0f,
            FrustumHandler()
        );

        // Call updateParameters - this resets the forward based on gaze parameter
        cam.updateParameters(cv::Vec3f(5.0f, 5.0f, 5.0f), cv::Vec3f(0.0f, 0.0f, -1.0f));

        // Forward should now be (0,0,1) (opposite of gaze) regardless of angles
        CHECK(cam.getForwardVector()[2] == doctest::Approx(1.0f).epsilon(1e-5f));
        CHECK(cam.getEye()[0] == doctest::Approx(5.0f));
    }

    // =====================================================
    // Edge Cases and Boundary Tests
    // =====================================================

    TEST_CASE("CameraHandler Large Eye Position Values")
    {
        FrustumHandler frustum;
        cv::Vec3f eye(1e6f, 2e6f, 3e6f);
        cv::Vec3f gaze(0.0f, 0.0f, -1.0f);
        CameraHandler cam(eye, gaze, frustum);

        CHECK(cam.getEye()[0] == doctest::Approx(1e6f));
        CHECK(cam.getEye()[1] == doctest::Approx(2e6f));
        CHECK(cam.getEye()[2] == doctest::Approx(3e6f));
        CHECK(cv::norm(cam.getForwardVector(), cv::NORM_L2) == doctest::Approx(1.0f).epsilon(1e-3f));
    }

    TEST_CASE("CameraHandler Negative Eye Position")
    {
        FrustumHandler frustum;
        cv::Vec3f eye(-5.0f, -10.0f, -15.0f);
        cv::Vec3f gaze(1.0f, 0.0f, 0.0f);
        CameraHandler cam(eye, gaze, frustum);

        CHECK(cam.getEye()[0] == doctest::Approx(-5.0f));
        CHECK(cam.getEye()[1] == doctest::Approx(-10.0f));
        CHECK(cam.getEye()[2] == doctest::Approx(-15.0f));
        CHECK(cam.getAt()[0] == doctest::Approx(1.0f));
    }

    TEST_CASE("CameraHandler Fractional Angles")
    {
        float tinyAngle = 0.001f;  // ~0.057 degrees
        CameraHandler cam(
            cv::Vec3f(0.0f, 0.0f, 0.0f),
            tinyAngle,
            0.0f,
            FrustumHandler()
        );

        // For very small angles, forward should be close to (0,0,-1)
        // Ry(tiny)*(0,0,-1): x = sin(tiny)*(-1) ≈ -tiny, y = 0, z = cos(tiny)*(-1) ≈ -1
        const Vec3f& f = cam.getForwardVector();
        CHECK(f[0] == doctest::Approx(-tinyAngle).epsilon(1e-5f));  // sin(tiny) ≈ tiny
        CHECK(f[1] == doctest::Approx(0.0f).epsilon(1e-5f));
        CHECK(f[2] == doctest::Approx(-1.0f).epsilon(1e-5f));
    }

    // =====================================================
    // VideoCapture Constructor Tests (require camera hardware)
    // =====================================================

    TEST_CASE("CameraHandler VideoCapture Constructor Opens Capture")
    {
        cv::Mat dummyFrame;
        // Try to create a CameraHandler with the video source constructor
        // If no camera is available, it should throw
        bool cameraAvailable = true;
        try {
            cv::VideoCapture testCap(0);
            if (!testCap.isOpened()) {
                cameraAvailable = false;
            }
            testCap.release();
        } catch (...) {
            cameraAvailable = false;
        }

        if (cameraAvailable) {
            SUBCASE("Constructor should succeed with camera")
            {
                cv::Mat frame;
                CHECK_NOTHROW(CameraHandler("TestWindow", frame, 0));
            }
        } else {
            SUBCASE("Constructor should throw without camera")
            {
                cv::Mat frame;
                CHECK_THROWS_AS(CameraHandler("TestWindow", frame, 0), std::invalid_argument);
            }
        }
    }

    // =====================================================
    // CameraHandler Getter Consistency Tests
    // =====================================================

    TEST_CASE("Getter Consistency After Construction")
    {
        FrustumHandler frustum;
        cv::Vec3f eye(3.0f, 4.0f, 5.0f);
        cv::Vec3f gaze(1.0f, 1.0f, 0.0f);
        CameraHandler cam(eye, gaze, frustum);

        // Check forward vector is opposite of gaze direction (normalized)
        cv::Vec3f expectedForward = -gaze;
        expectedForward /= cv::norm(expectedForward, cv::NORM_L2);

        CHECK(cam.getForwardVector()[0] == doctest::Approx(expectedForward[0]).epsilon(1e-5f));
        CHECK(cam.getForwardVector()[1] == doctest::Approx(expectedForward[1]).epsilon(1e-5f));
        CHECK(cam.getForwardVector()[2] == doctest::Approx(expectedForward[2]).epsilon(1e-5f));

        // Gaze should equal at
        CHECK(cam.getAt()[0] == doctest::Approx(gaze[0]));
        CHECK(cam.getAt()[1] == doctest::Approx(gaze[1]));
        CHECK(cam.getAt()[2] == doctest::Approx(gaze[2]));
    }

    // =====================================================
    // Multiple Camera Instances Tests
    // =====================================================

    TEST_CASE("Multiple Independent CameraInstances")
    {
        FrustumHandler frustum1, frustum2;
        cv::Vec3f eye1(1.0f, 0.0f, 0.0f);
        cv::Vec3f eye2(-1.0f, 0.0f, 0.0f);
        cv::Vec3f gaze(0.0f, 0.0f, -1.0f);

        CameraHandler cam1(eye1, gaze, frustum1);
        CameraHandler cam2(eye2, gaze, frustum2);

        // Should have different eye positions
        CHECK(cam1.getEye()[0] == doctest::Approx(1.0f));
        CHECK(cam2.getEye()[0] == doctest::Approx(-1.0f));

        // But same gaze
        CHECK(cam1.getAt()[2] == doctest::Approx(cam2.getAt()[2]));

        // Modifying one should not affect the other
        cam1.updateParameters(cv::Vec3f(10.0f, 0.0f, 0.0f), cv::Vec3f(0.0f, 0.0f, -1.0f));
        CHECK(cam1.getEye()[0] == doctest::Approx(10.0f));
        CHECK(cam2.getEye()[0] == doctest::Approx(-1.0f));
    }

    // =====================================================
    // Check FrustumHandler via Camera Access (if accessible)
    // =====================================================

    TEST_CASE("CameraHandler Uses Supplied FrustumHandler")
    {
        FrustumHandler frustum(-10.0f, 10.0f, -5.0f, 5.0f, 2.0f, 200.0f);
        CameraHandler cam(
            cv::Vec3f(0.0f, 0.0f, 5.0f),
            cv::Vec3f(0.0f, 0.0f, -1.0f),
            frustum
        );

        // The frustum_ is private, so we can't access it directly
        // but we can verify the camera was constructed correctly
        CHECK(cam.getEye()[2] == doctest::Approx(5.0f));
        CHECK(cam.getAt()[2] == doctest::Approx(-1.0f));
    }

    // =====================================================
    // Reset Method Tests
    // =====================================================

    TEST_CASE("CameraHandler Reset Method")
    {
        // Create camera with default constructor (no video capture involved)
        CameraHandler cam;

        // reset() tries to reopen capture device - may throw if device unavailable
        bool cameraAvailable = true;
        try {
            cv::VideoCapture testCap(0);
            if (!testCap.isOpened()) {
                cameraAvailable = false;
            }
            testCap.release();
        } catch (...) {
            cameraAvailable = false;
        }

        if (!cameraAvailable) {
            // Without camera, reset should throw runtime_error
            CHECK_THROWS_AS(cam.reset(), std::runtime_error);
        }
    }

} // TEST_SUITE("CameraHandler")