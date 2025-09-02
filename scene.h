#ifndef SCENE_H
#define SCENE_H

// pi is needed by the host when importing sky maps and used by the device when getting pixels.
const float pi_host = 3.141592653589793;
__device__ __constant__ float pi_device = 3.141592653589793;

/*
 *  Metrics are currently defined in coordinates of (ct, x, y, z)
 *  with the assumption that c = 1 is set, so the coordinates are
 *  in just (t, x, y, z). However, the framework
 *  can handle general metrics in arbitrary coordinates.
 */

// Scene contains information and routines about the scene, its objects and the sky map.
// This would be probably be nicer if it were modularised into smaller pieces, but that
// makes GPU programming more annoying. Deal with some sort of refactorisation later.
class Scene
{
    public:
        // Image parameters of the sky map (determines the number of photons to trace).
        int sky_pixels_w;
        int sky_pixels_h;
        // Float versions are useful for calculation when sampling pixels from the sky map.
        float sky_pixels_w_f;
        float sky_pixels_h_f;
        // Interval of phi and theta in radians between each pixel of the sky map.
        float phi_interval;
        float theta_interval;
        // Number of bytes used to store each pixel, not bits! Should be 3 for RGB.
        int byte_depth;
        // Distance to the sky map from the spatial coordinate centre.
        // Once a photon reaches this distance, use its spatial velocity to extend it to infinity,
        // find its polar and azimuthal angles and get the appropriate pixel sky map.
        // TODO: a check is needed to make sure the camera never goes beyond this radius.
        const float sky_map_distance { 75. };
        const float sky_map_distance_squared { sky_map_distance*sky_map_distance };
        // Camera FoV width in degrees and radians.
        float fov_width;
        float fov_width_rad;
        float camera_coords[4];
        // Quaternion representing the camera orientation. (1., 0., 0., 0.) represents a camera pointing along +x with zero rotation.
        float camera_quat[4];
        // Camera pixel resolution.
        int pixels_w;
        int pixels_h;

        // ------------- Function forward declarations.
        // Initialise scene parameters with no sky map and default camera parameters.
        void initialiseDefault(char sky_map[]);
        void setSkyMapDistance(float sky_distance);
        void setCameraRes(int width, int height);
        void setCameraFoV(float new_fov_width);
        void setCameraCoordinates(float x[4]);
        void setCameraQuaternion(float quaternion[4]);
        // Sky map image should be a 2:1 aspect ratio, 360-degree panoramic image, but there is no restriction on this.
        void importSkyMap(char image_path[]);
        void runTraceKernel();
        void writeCameraImage(char image_path[]);
        void freeHostPixelArrays();
        void freeDevicePixelArrays();

    private:
        const float default_camera_coords[4] { 0., -10., 0., 0. };
        // This default orientation corresponds to pointing along +x with no rotation.
        const float default_camera_quat[4] { 1., 0., 0., 0, };
        // Default horizontal FoV in degrees.
        const float default_fov { 90. };
        // Default resolution.
        const int default_width { 1920 };
        const int default_height { 1080 };
        // Pointer to the pixel array of the sky map and camera image on the host.
        unsigned char *host_sky_map { nullptr };
        unsigned char *host_camera_pixel_array { nullptr };
};

// Some quaternion arithmetic functions. Used when setting the starting velocities of photons.
__device__ void hamiltonProduct(float u[4], float v[4], float result[4]);
__device__ void rotateVecByQuat(float vec[4], float rotation_quat[4], float result[4]);

// CUDA kernels.
__global__ void traceImage(unsigned char *device_sky_map, unsigned char *device_camera_pixel_array);

#endif
