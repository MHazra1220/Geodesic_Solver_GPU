#include "camera.h"
#include "world.h"
#include "particle.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <Eigen/Dense>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <omp.h>

using namespace Eigen;

void Camera::setWidthHeight(int pixels_width, int pixels_height)
{
    image_width = pixels_width;
    image_height = pixels_height;
    double_width = image_width;
    double_height = image_height;
    // Set the size of the array of the camera view pixel array.
    // 3 bytes per pixel for RGB.
    camera_view.resize(image_width*image_height*3);
}

void Camera::setFov(double fov)
{
    fov_width = fov;
    // Convert to radians.
    fov_width_rad = fov_width * (pi/180.);
}

void Camera::setCameraLocation(Vector4d location)
{
    camera_location = location;
}

// Used to rotate the starting 4-velocities of photons.
void Camera::setCameraOrientation(Vector4d orientation)
{
    camera_orientation = orientation / orientation.norm();
}

// Calculates the starting direction of the photon mapped to pixel x and pixel y.
// x=0 and y=0 are the top-left corner of the camera.
Vector3d Camera::calculateStartDirection(int x, int y)
{
    double conversion_factor { fov_width_rad/double_width };
    double phi { -((x-0.5*double_width)*conversion_factor) };
    double theta { (y-0.5*double_height)*conversion_factor + 0.5*pi };

    // Convert to a Cartesian unit vector in (x, y, z).
    Vector3d start_direction;
    start_direction(0) = sin(theta)*cos(phi);
    start_direction(1) = sin(theta)*sin(phi);
    start_direction(2) = cos(theta);
    // Rotate to align with the camera orientation.
    return quaternionRotate(start_direction, camera_orientation);
    // From here, the Particle/Photon object this is sent to needs
    // to normalise the direction to a null 4-velocity.
}

// Renders the actual image.
void Camera::traceImage(World &simulation)
{
    double conversion_factor { fov_width_rad/double_width };
    // Go through each ray.
    #pragma omp parallel for
    for (int y = 0; y < image_height; y++)
    {
        for (int x = 0; x < image_width; x++)
        {
            int pixel_index { 3*(y*image_width + x) };
            double phi;
            double theta;
            Vector3d start_direction { calculateStartDirection(x, y) };
            Particle photon;
            photon.setX(camera_location);
            photon.updateMetric(simulation.getMetricTensor(photon.x));
            Vector4d initial_v { 0., 0., 0., 0. };
            initial_v(seq(1, 3)) = start_direction;
            photon.setV(initial_v);
            photon.makeVNull();

            double squared_radius { photon.x(seq(1, 3)).squaredNorm() };
            bool consumed { false };
            while (squared_radius < simulation.sky_map_distance_squared)
            {
                if (squared_radius < 2.25)
                {
                    // Entered the photon sphere; photon is guaranteed to cross the event horizon.
                    consumed = true;
                    break;
                }
                photon.advance(simulation);
                squared_radius = photon.x(seq(1, 3)).squaredNorm();
            }

            if (consumed == true)
            {
                // Photon entered the photon sphere if true (set pixel to black).
                camera_view[pixel_index] = 0;
                camera_view[pixel_index+1] = 0;
                camera_view[pixel_index+2] = 0;
            }
            else
            {
                // Photon escaped; get a pixel from the skybox.
                phi = atan2(photon.x(2), photon.x(1));
                if (phi < 0)
                {
                    // Get into the range 0 to 2pi.
                    phi += two_pi;
                }
                theta = acos(photon.x(3) / photon.getEuclideanDistance());

                // Convert to pixel locations on the sky map; floor the number.
                // Because phi goes anticlockwise, 2.*pi - phi is needed here to
                // stop images being reversed along phi.
                int sky_x { (int)floor((two_pi-phi)/simulation.phi_interval) };
                int sky_y { (int)floor(theta/simulation.theta_interval) };
                // These if statements stop errors when the ray is exactly along the +x or +z axis.
                if (sky_x >= simulation.sky_width)
                {
                    sky_x = simulation.sky_width - 1;
                }
                if (sky_y >= simulation.sky_height)
                {
                    sky_y = simulation.sky_height - 1;
                }
                // Get pixel colour.
                photon.pixelColour = simulation.readPixelFromSkyMap(sky_x, sky_y);
                camera_view[pixel_index] = photon.pixelColour[0];
                camera_view[pixel_index+1] = photon.pixelColour[1];
                camera_view[pixel_index+2] = photon.pixelColour[2];
            }
        }
    }
}

// Output the pixel array of the camera's view to an image file.
void Camera::writeCameraImage(char* image_path)
{
    // stbi_write needs a pointer to the first element of the pixel array.
    unsigned char* data { &camera_view[0] };
    stbi_write_jpg(image_path, image_width, image_height, 3, data, 100);
}

// Returns the Hamilton (quaternionic) product of u with v.
Vector4d hamiltonProduct(Vector4d u, Vector4d v)
{
    Vector4d result;
    Vector3d u_vec { u(seq(1, 3)) };
    Vector3d v_vec { v(seq(1, 3)) };
    result(0) = u(0)*v(0) - u_vec.dot(v_vec);
    result(seq(1, 3)) = u(0)*v_vec + v(0)*u_vec + u_vec.cross(v_vec);

    return result;
}

// Returns a 3D cartesian vector rotated by the given quaternion.
Vector3d quaternionRotate(Vector3d vec, Vector4d rotation_quat)
{
    // Assume that rotation_quat is normalised.
    Vector4d rotation_quat_inverse { rotation_quat };
    rotation_quat_inverse(seq(1, 3)) *= -1.;
    Vector4d vec_as_quat;
    vec_as_quat(0) = 0.;
    vec_as_quat(seq(1, 3)) = vec;

    return hamiltonProduct(rotation_quat, hamiltonProduct(vec_as_quat, rotation_quat_inverse))(seq(1, 3));
}
