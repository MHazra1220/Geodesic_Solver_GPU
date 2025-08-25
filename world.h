#ifndef WORLD
#define WORLD

#include <vector>
#include <Eigen/Dense>

using namespace Eigen;

/*
 *  Everything in the simulation works in coordinates of (ct, x, y, z)
 *  with the assumption that c = 1 is set, so the coordinates are
 *  in terms of just (t, x, y, z).
 */

// World contains information and routines about the spacetime and background image.
class World
{
public:
    // Image parameters of the sky map.
    int sky_width { 0 };
    int sky_height { 0 };
    // Interval of phi and theta between each pixel.
    double phi_interval;
    double theta_interval;
    // Note this is the number of bytes used to store each pixel, not bits!
    int byte_depth { 0 };
    // Use a vector in practice to let C++ manage its memory properly.
    std::vector<unsigned char> sky_map;
    const double sky_map_distance { 150. };
    const double sky_map_distance_squared { sky_map_distance*sky_map_distance };

    // Calculates the metric tensor at x.
    Matrix4d getMetricTensor(Vector4d x);
    // Calculates the Christoffel symbols at x using central difference numerical derivatives.
    // Overwrites the array of 4 Eigen::Matrix4d objects, one for each coordinate of the upper index.
    void getChristoffelSymbols(Vector4d x, Matrix4d &metric, Matrix4d christoffel_symbols[4]);
    // Reads in a bitmap as a spherical sky map in equirectangular projection.
    // Assumed for now to be a bitmap with 2:1 aspect ratio.
    void importSkyMap(char* image_path);
    // Returns a pointer to the RGB pixel from the sky map at pixel (x, y), where (0, 0) is the top-left pixel.
    unsigned char* readPixelFromSkyMap(int x, int y);

private:
    // Pointer to the pixel array of the sky map.
    unsigned char* sky_map_pointer { nullptr };
    const double pi = 3.141592653589793;
};

#endif
