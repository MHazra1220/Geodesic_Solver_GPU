#include "scene.h"
#include "quaternion.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <stdexcept>

// These objects and functions are used by the GPU when tracing photons. They are defined outside Scene
// in a separate namespace so that the GPU doesn't need a copy of the entire Scene object in device memory when calculating paths.
namespace DeviceTraceTools
{
    // Device variables and functions.
    // This pointer is left in host memory and assigned with cudaMemcpy because otherwise it cannot be deallocated by the host
    // with cudaFree().
    unsigned char* device_sky_map { nullptr };
    // Default camera coordinates are along -x and the camera faces along +x.
    __device__ float device_camera_coords[4] { 0., -10., 0., 0. };
    __device__ float device_camera_quat[4] { 1., 0., 0., 0. };
    __device__ float device_fov_conversion_factor;
    __device__ float device_pixels_w;
    __device__ float device_pixels_h;

    __device__ void calculateStartVelocity(float pixel_x, float pixel_y, float photon_v[4]);
    __device__ void getMetricTensor(float x_func[4], float metric_func[4][4]);
    __device__ void getChristoffelSymbols(float x_func[4], float metric_func[4][4], float c_symbols_func[4][4][4]);
    __device__ void makeVNull(float v_func[4], float metric_func[4][4]);
    __device__ void invertMetric(float metric_func[4][4], float metric_inverse[4][4]);
    __device__ void readPixelFromSkyMap(unsigned char *pixel, unsigned char *device_sky_map, int &x, int &y, int &sky_pixels_w, int &byte_depth);

    // CUDA kernels.
    __global__ void traceImage();
};

// Initialise Scene object with no sky map and default camera parameters.
void Scene::initialiseDefault(char sky_map[])
{
    // Note sky_map will already have decayed to a char* pointer here; no need to convert.
    importSkyMap(sky_map);
    // Set camera quaternion to default position and orientation and copy to device.
    setCameraCoordinates((float*)&default_camera_coords);
    setCameraQuaternion((float*)&default_camera_quat);
    // Default horizontal FoV is 75 degrees.
    setCameraFoV(75.);
}

// Sky map image should be a 2:1 aspect ratio, 360-degree panoramic image, but there is no restriction on this.
void Scene::importSkyMap(char image_path[])
{
    // image_path should be a pointer to a C-style array of char[].
    // This is usually too large for stack allocation, so stbi_load() returns a pointer to the array on the heap.
    // Force to load as RGB (3 bytes per pixel).
    host_sky_map = stbi_load(image_path, &sky_pixels_w, &sky_pixels_h, &byte_depth, 3);
    if (host_sky_map != NULL)
    {
        num_photons = sky_pixels_w*sky_pixels_h;
        sky_pixels_w_float = static_cast<float>(sky_pixels_w);
        sky_pixels_h_float = static_cast<float>(sky_pixels_h);
        phi_interval = (2.*pi_host) / sky_pixels_w;
        theta_interval = pi_host / sky_pixels_h;

        // Reset existing device map (if it exists), then copy the new map and related information.
        // This cannot be allocated in initialiseDefault() because its size is not known at compile-time.
        freeSkyMapDevice();
        cudaError_t err { cudaSuccess };
        size_t map_size { sizeof(unsigned char)*num_photons*byte_depth };
        err = cudaMalloc((void **)&DeviceTraceTools::device_sky_map, map_size);
        if (err != cudaSuccess)
        {
            throw std::runtime_error("Error: failed to allocate sky map memory on device.");
        }
        err = cudaMemcpy(DeviceTraceTools::device_sky_map, host_sky_map, map_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            throw std::runtime_error("Error: failed to copy sky map to device.");
        }

        err = cudaMemcpyToSymbol(DeviceTraceTools::device_pixels_w, &sky_pixels_w_float , sizeof(float));
        if (err != cudaSuccess)
        {
            throw std::runtime_error("Error: failed to copy sky pixel width to device.");
        }
        err = cudaMemcpyToSymbol(DeviceTraceTools::device_pixels_h, &sky_pixels_h_float, sizeof(float));
        if (err != cudaSuccess)
        {
            throw std::runtime_error("Error: failed to copy sky pixel height to device.");
        }
    }
    else
    {
        throw std::runtime_error("Error: failed to load sky map image.");
    }
}

void Scene::freeSkyMapHost()
{
    stbi_image_free(host_sky_map);
}

void Scene::freeSkyMapDevice()
{
    cudaFree(DeviceTraceTools::device_sky_map);
}

// Set a new FoV in degrees and transfer the corresponding conversion factor to the device.
void Scene::setCameraFoV(float new_fov_width)
{
    fov_width = new_fov_width;
    fov_width_rad = fov_width * (pi_host/180.);
    float conversion_factor = fov_width_rad / sky_pixels_w_float;
    cudaError_t err;
    err = cudaMemcpyToSymbol(DeviceTraceTools::device_fov_conversion_factor, &conversion_factor, sizeof(float));
    if (err != cudaSuccess)
    {
        throw std::runtime_error("Error: failed to copy FoV conversion factor to device.");
    }
}

// When called, this will both set the host variable and copy it to DeviceTraceTools::camera_coords
// to let the GPU access its own copy in device memory. Same thing for setCameraQuaternion.
void Scene::setCameraCoordinates(float x[4])
{
    for (int i { 0 }; i < 4; i++)
    {
        camera_coords[i] = x[i];
    }
    cudaError_t err { cudaMemcpyToSymbol(DeviceTraceTools::device_camera_coords, camera_coords, sizeof(float)*4) };
    if (err != cudaSuccess)
    {
        throw std::runtime_error("Error: failed to copy camera coordinates to device.");
    }
}

void Scene::setCameraQuaternion(float quaternion[4])
{
    for (int i { 0 }; i < 4; i++)
    {
        camera_quat[i] = quaternion[i];
    }
    cudaError_t err { cudaMemcpyToSymbol(DeviceTraceTools::device_camera_quat, camera_quat, sizeof(float)*4) };
    if (err != cudaSuccess)
    {
        throw std::runtime_error("Error: failed to copy camera quaternion to device.");
    }
}

// Calculates the start velocity of a photon at pixel (x, y), where (0, 0) is the top-left corner of the image.
__device__ void DeviceTraceTools::calculateStartVelocity(float pixel_x, float pixel_y, float photon_v[4])
{
    float phi { (pixel_x - 0.5*DeviceTraceTools::device_pixels_w) * DeviceTraceTools::device_fov_conversion_factor };
    float theta { (pixel_y - 0.5*DeviceTraceTools::device_pixels_h) * DeviceTraceTools::device_fov_conversion_factor + 0.5*pi_device };
    photon_v[0] = phi;
    photon_v[1] = theta;
}

// Currently defined to return the Schwarzschild metric with a Schwarzschild radius of 1.
// Gets the metric at x_func and overwrites it into metric_func.
__device__ void DeviceTraceTools::getMetricTensor(float x_func[4], float metric_func[4][4])
{
    const float r_s { 1. };
    float r { norm3df(x_func[1], x_func[2], x_func[3]) };
    float r_squared { r*r };
    float mult_factor { r_s / (r_squared*(r-r_s)) };
    for (int mu { 1 }; mu < 4; mu++)
    {
        metric_func[0][mu] = 0.;
        metric_func[mu][0] = 0.;
        for (int nu { mu }; nu < 4; nu++)
        {
            metric_func[mu][nu] = mult_factor*x_func[mu]*x_func[nu];
            metric_func[nu][mu] = metric_func[mu][nu];
        }
    }
    metric_func[0][0] = -1. + r_s/r;
    metric_func[1][1] += 1.;
    metric_func[2][2] += 1.;
    metric_func[3][3] += 1.;
}

__device__ void DeviceTraceTools::getChristoffelSymbols(float x_func[4], float metric_func[4][4], float c_symbols_func[4][4][4])
{
    // Assumed default step in each coordinate.
    // TODO: How do you define this adaptively to not break near areas of extreme distortion?
    // For now, just set it to a small number.
    const float step { 1e-4 };

    // Simple Euler forward-difference derivatives of the metric along each component.
    float metric_derivs[4][4][4];
    float metric_temp[4][4];
    for (int alpha { 0 }; alpha < 4; alpha++)
    {
        // WARNING: Euler can be significantly faster than central difference; just use a very small step and it'll probably be okay.
        // The limiting factor in accuracy is probably the overarching parameter step in the geodesic equation, not the metric derivatives.
        // TODO: Automatic differentiation with dual numbers?
        x_func[alpha] += step;
        getMetricTensor(x_func, metric_temp);
        for (int mu { 0 }; mu < 4; mu++)
        {
            for (int nu { mu }; nu < 4; nu++)
            {
                metric_derivs[alpha][mu][nu] = metric_func[mu][nu] - metric_temp[mu][nu];
                metric_derivs[alpha][nu][mu] = metric_derivs[alpha][mu][nu];
            }
        }
        x_func[alpha] -= step;
    }

    // Calculate the inverse of metric_func and overwrite it into metric_temp.
    // Set metric_temp to the identity matrix first.
    for (int mu { 0 }; mu < 4; mu++)
    {
        for (int nu { mu }; nu < 4; nu++)
        {
            if (mu == nu)
            {
                metric_temp[mu][nu] = 1.;
            }
            else
            {
                metric_temp[mu][nu] = 0.;
                metric_temp[nu][mu] = 0.;
            }
        }
    }
    // Store the inverse metric into metric_temp. metric_func is now useless
    // until it is assigned again in getMetricTensor() (it gets overwritten by invertMetric()).
    invertMetric(metric_func, metric_temp);

    // Calculate the 40 independent Christoffel symbols.
    for (int alpha { 0 }; alpha < 4; alpha++)
    {
        for (int mu { 0 }; mu < 4; mu++)
        {
            for (int nu { mu }; nu < 4; nu++)
            {
                float component[4];
                for (int gamma { 0 }; gamma < 4; gamma++)
                {
                    component[gamma] = metric_derivs[nu][mu][gamma] + metric_derivs[mu][nu][gamma] - metric_derivs[gamma][mu][nu];
                }
                // Remember that metric_temp is the inverse metric here.
                c_symbols_func[alpha][mu][nu] = 0.5*(
                    metric_temp[alpha][0]*component[0] + metric_temp[alpha][1]*component[1]
                    + metric_temp[alpha][2]*component[2] + metric_temp[alpha][3]*component[3]
                );
                c_symbols_func[alpha][nu][mu] = c_symbols_func[alpha][mu][nu];
            }
        }
    }
}

/*
 * Modifies the t-component of the 4-velocity to make the vector null.
 * This requires solving a quadratic equation for the t-component; assume
 * that you should take the positive root because a=g_00 is
 * probably negative. Note that the more negative solution is needed
 * because the raytracer evolves photons "backwards".
 */
__device__ void DeviceTraceTools::makeVNull(float v_func[4], float metric_func[4][4])
{
    float a { metric_func[0][0] };
    float b { 0. };
    // c is the scalar product of the spatial metric with the spatial velocity components.
    float c { 0. };
    float contraction;
    for (int i { 1 }; i < 4; i++)
    {
        b += metric_func[0][i] * v_func[i];
        contraction = 0.;
        for (int j { 1 }; j < 4; j++)
        {
            contraction += metric_func[i][j] * v_func[j];
        }
        c += v_func[i]*contraction;
    }
    b *= 2.;

    // Take the positive root solution (note a=g_00 is usually negative, so this normally gives the negative solution).
    v_func[0] = (-b + sqrt(b*b - 4.*a*c)) / (2.*a);
}

// TODO: This doesn't get the correct result for asymmetric matrices! Not technically important here, but it's
// indicative that something is wrong underneath.
__device__ void DeviceTraceTools::invertMetric(float metric_func[4][4], float metric_inverse[4][4])
{
    // Assume that that there are no zeros on the diagonal of metric_func and that metric_inverse is currently the identity matrix.
    // Invert with forward and backward-propagation (i.e. LU-decomposition). metric_func and metric_temp are both overwritten to avoid memory allocation.
    // WARNING: For now, assume that there are no zeros on the diagonal of the metric (very unlikely in t, x, y, z coordinates).

    float multiplier;

    // Forward-propagation pass.
    for (int i { 0 }; i < 3; i++)
    {
        for (int j { i+1 }; j < 4; j++)
        {
            multiplier = metric_func[j][i] / metric_func[i][i];
            // Use the accumulating zeros in the lower-triangular half to reduce the number of calculations.
            for (int k { i }; k < 4; k++)
            {
                metric_func[j][k] -= multiplier*metric_func[i][k];
            }
            // Use the fact that metric_inverse is currently an identity matrix to reduce the number of calculations.
            for (int k { 0 }; k < j; k++)
            {
                metric_inverse[j][k] -= multiplier*metric_inverse[i][k];
            }
        }
    }

    // Backward-propagation pass.
    for (int i { 3 }; i > 0; i--)
    {
        for (int j { i-1 }; j > -1; j--)
        {
            multiplier = metric_func[j][i] / metric_func[i][i];
            // Use the zeros in the lower-triangular half of metric_func to reduce the number of calculations.
            for (int k { i }; k < 4; k++)
            {
                metric_func[j][k] -= multiplier*metric_func[i][k];
            }
            for (int k { 0 }; k < 4; k++)
            {
                metric_inverse[j][k] -= multiplier*metric_inverse[i][k];
            }
        }
    }

    // Last task is to normalise the rows of metric_inverse by whatever is left in the diagonal of metric_func.
    for (int i { 0 }; i < 4; i++)
    {
        multiplier = 1./metric_func[i][i];
        metric_func[i][i] = 1.;
        for (int k { 0 }; k < 4; k++)
        {
            metric_inverse[i][k] *= multiplier;
        }
    }
}

// Gets a pointer to the RGB pixel from the sky map at pixel (x, y), where (0, 0) is the top-left pixel.
__device__ void DeviceTraceTools::readPixelFromSkyMap(unsigned char *pixel, unsigned char *device_sky_map, int &x, int &y, int &sky_pixels_w, int &byte_depth)
{
    pixel = &device_sky_map[(y*sky_pixels_w + x)*byte_depth];
}
