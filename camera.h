#ifndef CAMERA
#define CAMERA

#include "world.h"
#include <vector>
#include <Eigen/Dense>

using namespace Eigen;

class Camera
{
public:
    // Render resolution; 1080p widescreen by default.
    int image_width { 1920 };
    int image_height { 1080 };
    // Store double versions for the sake of calculation.
    double double_width;
    double double_height;
    // Horizontal FoV in degrees.
    // TODO: Make the program assume that the FoV spans across the wider of either the width or height of the camera view.
    double fov_width;
    double fov_width_rad;
    // Cartesian spacetime location of the camera.
    Vector4d camera_location;
    // Quaternion that determines the orientation of the camera, defined as a rotation from pointing along +x with zero angle.
    Vector4d camera_orientation { 1., 0., 0., 0. };
    // Pixel array of the camera's view. This is stored as a 1D vector in row-major order
    // from top-left to bottom-right. Every 3 bytes (3 entries) stores the RBG entries of a pixel.
    std::vector<unsigned char> camera_view;

    void setWidthHeight(int pixels_width, int pixels_height);
    void setFov(double fov);
    // Note that sets the location in 4D spacetime.
    void setCameraLocation(Vector4d location);
    void setCameraOrientation(Vector4d orientation);

    // Calculates the starting direction of the photon mapped to pixel x and pixel y.
    // x=0 and y=0 is the bottom left corner.
    Vector3d calculateStartDirection(int x, int y);
    // Renders the actual image.
    void traceImage(World &simulation);
    // Output the pixel array of the camera's view to an image file.
    void writeCameraImage(char* image_path);

private:
    const double pi = 3.141592653589793;
    const double two_pi = 2.*pi;

    unsigned char* RGBToHSV(unsigned char colour[]);
};

// Returns the Hamilton product of u with v.
Vector4d hamiltonProduct(Vector4d u, Vector4d v);

// Returns a 3D cartesian vector rotated by the given quaternion.
Vector3d quaternionRotate(Vector3d vec, Vector4d rotation_quat);

#endif
