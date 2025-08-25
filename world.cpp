#include "world.h"
#include <iostream>
#include <cmath>
#include <array>
#include <vector>
#include <Eigen/Dense>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

using namespace Eigen;

// Currently returns the Schwarzschild metric with a Schwarzschild radius of 1 (normalised units).
// Keep in mind that, for a Schwarzschild metric, there is no need to further advance a photon that crosses inside
// the photon sphere; assume there is nothing within the photon sphere emitting light and treat
// the pixel as black.
// TODO: Store a set of common, analytical metrics that the user can select.
Matrix4d World::getMetricTensor(Vector4d x)
{
    const double r_s = 1.;
    double r_squared = x(seq(1, 3)).dot(x(seq(1, 3)));
    double r = sqrt(r_squared);
    double mult_factor = r_s / (r_squared * (r - r_s));
    Matrix4d metric;
    metric.setZero();
    for (int j { 1 }; j < 4; j++)
    {
        for (int i { j }; i < 4; i++)
        {
            metric(i, j) = x(i)*x(j);
            metric(j, i) = metric(i, j);
        }
    }
    metric *= mult_factor;
    metric(0, 0) = -1. + r_s/r;
    metric(1, 1) += 1.;
    metric(2, 2) += 1.;
    metric(3, 3) += 1.;

    return metric;
}

// Calculates the Christoffel symbols at x using central difference numerical derivatives.
// Overwrites the array of 4 Eigen::Matrix4d objects, one for each coordinate of the upper index.
void World::getChristoffelSymbols(Vector4d x, Matrix4d &metric, Matrix4d christoffel_symbols[4])
{
    // Assumed default step in each coordinate.
    // TODO: How do you define this adaptively to not break near areas of extreme distortion?
    // For now, just set it to a small number.
    double step { 1e-4 };

    // Second-order accurate central-difference derivatives of the metric along each component.
    std::array<Matrix4d, 4> metric_derivs;
    Matrix4d metric_forward;
    Matrix4d metric_backward;
    Vector4d intermediate_x { x };
    for (int mu { 0 }; mu < 4; mu++)
    {
        // WARNING: Euler can be significantly faster than central difference; just use a very small step and it'll be okay.
        // The limiting factor in accuracy is the overarching parameter step in the geodesic equation, not the metric derivatives.
        intermediate_x(mu) += step;
        metric_forward = getMetricTensor(intermediate_x);
        metric_derivs[mu] = (metric_forward - metric) / step;
        intermediate_x(mu) = x(mu);
    }

    Matrix4d metricInverse { metric.inverse() };

    // Go through each upper index of the Christoffel symbols and only calculate the 40 independent components.
    for (int alpha { 0 }; alpha < 4; alpha++)
    {
        Matrix4d christoffel_component;
        for (int nu { 0 }; nu < 4; nu++)
        {
            for (int mu = nu; mu < 4; mu++)
            {
                Vector4d metric_deriv_components;
                for (int gamma { 0 }; gamma < 4; gamma++)
                {
                    metric_deriv_components(gamma) = metric_derivs[nu](gamma, mu)
                    + metric_derivs[mu](gamma, nu)
                    - metric_derivs[gamma](mu, nu);
                }
                christoffel_component(mu, nu) = 0.5*metricInverse.col(alpha).dot(metric_deriv_components);
                // This is redundant when mu = nu, but probably faster than an if statement.
                christoffel_component(nu, mu) = christoffel_component(mu, nu);
            }
        }
        christoffel_symbols[alpha] = christoffel_component;
    }
}

// Image path should lead to a 2:1 aspect ratio, equirectangular-projected, panoramic image.
// TODO: currently no check that the image is indeed 2:1 or anything to crop the image when it isn't.
void World::importSkyMap(char* image_path)
{
    // image_path should be a pointer to a C-style array of char[].
    // This is often too large for stack allocation, so stbi_load() returns a pointer to the array on the heap.
    // Force to load as RGB (3 bytes per pixel).
    sky_map_pointer = stbi_load(image_path, &sky_width, &sky_height, &byte_depth, 3);
    if (sky_map_pointer != NULL)
    {
        // Store in a std::vector to let C++ handle memory management.
        sky_map = std::vector<unsigned char>(sky_map_pointer, sky_map_pointer + sky_width*sky_height*byte_depth);
        phi_interval = (2.*pi) / sky_width;
        theta_interval = pi / sky_height;
    }
    else
    {
        std::cout << "Image failed to load." << "\n";
        exit(-1);
    }
    stbi_image_free(sky_map_pointer);
}

// Gets the RGB pixel from the sky map at pixel (x, y), where (0, 0) is the top-left pixel.
unsigned char* World::readPixelFromSkyMap(int x, int y)
{
    if (x >= sky_width || y >= sky_height)
    {
        std::cout << "Warning: requested pixel: (" << x << ", " << y << ") is out of range." << "\n";
        exit(-1);
    }

    // Gets the address of this particular pixel; use pixel[i] with i=0, 1, 2 for R, G, B.
    // Note that this is not returned as int! Use static_cast<int>() if necessary.
    unsigned char* pixel { &sky_map[(y*sky_width + x)*byte_depth] };
    return pixel;
}
