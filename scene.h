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
        // Parameter step will never exceed this when evolving photons.
        const float max_parameter_step { 5. };
        // Image parameters of the sky map (determines the number of photons to trace).
        int sky_pixels_w;
        int sky_pixels_h;
        // Float versions are useful for calculation when sampling pixels from the sky map.
        float sky_pixels_w_float;
        float sky_pixels_h_float;
        int num_photons;
        // Interval of phi and theta in radians between each pixel.
        float phi_interval;
        float theta_interval;
        // Number of bytes used to store each pixel, not bits! Should be 3 for RGB.
        int byte_depth;
        // Distance to the sky map from the spatial coordinate centre.
        // Once a photon reaches this distance, use its spatial velocity to extend it to infinity,
        // find its polar and azimuthal angles and get the appropriate pixel sky map.
        // TODO: a check is needed to make sure the camera never goes beyond this radius.
        const float sky_map_distance { 50. };
        const float sky_map_distance_squared { sky_map_distance*sky_map_distance };
        // Camera FoV width in degrees and radians.
        float fov_width { 75. };
        float fov_width_rad;
        // Quaternion representing the camera orientation. (1., 0., 0., 0.) represents a camera pointing along +x with zero rotation.
        float camera_quat[4];
        float *host_camera_quat { &camera_quat[0] };

        // ------------- Function forward declarations.
        // Initialise scene parameters with no sky map and default camera parameters.
        void initialiseDefault();
        void setCameraFoV(float new_fov_width);
        // Sky map image should be a 2:1 aspect ratio, 360-degree panoramic image, but there is no restriction on this.
        void importSkyMap(char image_path[]);
        void freeSkyMapHost();
        void freeSkyMapDevice();

    private:
        // This default orientation corresponds to pointing along +x with no rotation.
        const float default_quat[4] { 1., 0., 0., 0, };
        // Pointer to the pixel array of the sky map.
        unsigned char* host_sky_map { nullptr };

        // ------------- Function forward declarations.
        void setCameraQuaternion(float quaternion[4]);
};

#endif
