#include "scene.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <stdexcept>

// These objects and functions are used by the GPU when tracing photons. They are defined outside Scene
// in a separate namespace so that the GPU doesn't need a copy of the entire Scene object in device memory when calculating paths.
namespace DeviceTraceTools
{
    // Device variables and functions.
    // This pointer is left in host memory and assigned with cudaMemcpy because otherwise it cannot be deallocated by the host
    // with cudaFree(). This points to the sky map in device memory.
    unsigned char *device_sky_map { nullptr };
    // This points to the pixel array for the camera in device memory.
    unsigned char *device_camera_pixel_array { nullptr };
    // Default camera coordinates are along -x and the camera faces along +x.
    __device__ float device_camera_coords[4];
    __device__ float device_camera_quat[4];
    __device__ int device_pixels_w;
    __device__ int device_pixels_h;
    __device__ float device_fov_conversion_factor;
    __device__ int device_sky_pixels_w;
    __device__ int device_sky_pixels_h;
    __device__ const float device_sky_map_distance_squared{ 50.*50. };

    __device__ void calculateStartVelocity(float pixel_x, float pixel_y, float photon_v[4], float metric[4][4]);
    __device__ void getMetricTensor(float x_func[4], float metric_func[4][4]);
    __device__ void getChristoffelSymbols(float x_func[4], float metric_func[4][4], float c_symbols_func[4][4][4], float metric_derivs[4][4][4]);
    __device__ void makeVNull(float v_func[4], float metric_func[4][4]);
    __device__ void normaliseV(float v_func[4]);
    __device__ void invertMetric(float metric_func[4][4], float metric_inverse[4][4]);
    __device__ float calculateParameterStep(float metric[4][4]);
    __device__ void advance(float x[4], float v[4], float metric[4][4], float c_symbols[4][4][4], float metric_derivs[4][4][4]);
    __device__ void readPixelFromSkyMap(unsigned char *pixel, unsigned char *device_sky_map, int &x, int &y, int &sky_pixels_w, int &byte_depth);
};

// Initialise Scene object with no sky map and default camera parameters.
void Scene::initialiseDefault(char sky_map[])
{
    // Note sky_map will already have decayed to a char* pointer here; no need to convert.
    importSkyMap(sky_map);
    // // Set camera quaternion to default position and orientation and copy to device.
    // setCameraCoordinates((float*)&default_camera_coords);
    // setCameraQuaternion((float*)&default_camera_quat);
    // // Default resolution of 1920x1080.
    // setCameraRes(default_width, default_height);
    // // Default horizontal FoV is 75 degrees.
    // setCameraFoV(default_fov);
}

// Sky map image should be a 2:1 aspect ratio, 360-degree panoramic image, but there is no restriction on this.
void Scene::importSkyMap(char image_path[])
{
    // image_path should be a pointer to a C-style array of char[].
    // This is usually too large for stack allocation, so host_sky_map becomes a pointer to a pixel array on the heap.
    // Force to load as RGB (3 bytes per pixel).
    host_sky_map = stbi_load(image_path, &sky_pixels_w, &sky_pixels_h, &byte_depth, 3);
    if (host_sky_map != NULL)
    {
        sky_pixels_w_f = static_cast<float>(sky_pixels_w);
        sky_pixels_h_f = static_cast<float>(sky_pixels_h);
        phi_interval = (2.*pi_host) / sky_pixels_w;
        theta_interval = pi_host / sky_pixels_h;

        // Reset existing device map (if it exists), then copy the new map and related information.
        // This cannot be allocated in initialiseDefault() because its size is only known at run time.
        cudaFree(DeviceTraceTools::device_sky_map);
        cudaError_t err { cudaSuccess };
        size_t map_size { sizeof(unsigned char)*sky_pixels_w*sky_pixels_h*byte_depth };
        err = cudaMalloc((void **)&DeviceTraceTools::device_sky_map, map_size);
        if (err != cudaSuccess)
        {
            throw std::runtime_error("Error: failed to allocate memory for sky map on device.");
        }
        err = cudaMemcpy(DeviceTraceTools::device_sky_map, host_sky_map, map_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            throw std::runtime_error("Error: failed to copy sky map to device.");
        }

        err = cudaMemcpyToSymbol(DeviceTraceTools::device_sky_pixels_w, &sky_pixels_w, sizeof(int));
        if (err != cudaSuccess)
        {
            throw std::runtime_error("Error: failed to copy sky pixel width to device.");
        }
        err = cudaMemcpyToSymbol(DeviceTraceTools::device_sky_pixels_h, &sky_pixels_h, sizeof(int));
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

void Scene::runTraceKernel()
{
    // Use 32 threads per block for now. This is mostly limited by the available shared memory to store the Christoffel symbols and metric derivatives.
    // Use smaller blocks also allows the scheduler to naturally load-balance against the fact that pixels looking into the black hole probably require
    // more computation.
    dim3 threadsPerBlock(8, 4);
    int num_blocks_x { pixels_w / 8 };
    int num_blocks_y { pixels_h / 4 };
    if (pixels_w % 8 > 0)
    {
        num_blocks_x += 1;
    }
    if (pixels_h % 4 > 0)
    {
        num_blocks_y += 1;
    }
    dim3 numBlocks(num_blocks_x, num_blocks_y);
    for (int i { 0 }; i < 50; i++)
    {
        traceImage<<<numBlocks, threadsPerBlock>>>(DeviceTraceTools::device_sky_map, DeviceTraceTools::device_camera_pixel_array);
    }
    // Copy image back to host.
    cudaError_t err;
    err = cudaMemcpy(host_camera_pixel_array, DeviceTraceTools::device_camera_pixel_array, 3*sizeof(unsigned char)*pixels_w*pixels_h, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        throw std::runtime_error("Error: failed to copy camera pixel array from device to host.");
    }
}

void Scene::writeCameraImage(char image_path[])
{
    unsigned char *data { &host_camera_pixel_array[0] };
    stbi_write_jpg(image_path, pixels_w, pixels_h, 3, data, 100);
}

void Scene::freeHostPixelArrays()
{
    stbi_image_free(host_sky_map);
    free(host_camera_pixel_array);
}

void Scene::freeDevicePixelArrays()
{
    cudaFree(DeviceTraceTools::device_sky_map);
    cudaFree(DeviceTraceTools::device_camera_pixel_array);
}

// Sets the width and height resolution of the camera and copies it to the device.
void Scene::setCameraRes(int width, int height)
{
    pixels_w = width;
    pixels_h = height;
    cudaError_t err;
    err = cudaMemcpyToSymbol(DeviceTraceTools::device_pixels_w, &pixels_w, sizeof(int));
    if (err != cudaSuccess)
    {
        throw std::runtime_error("Error: failed to copy camera pixel width to device.");
    }
    err = cudaMemcpyToSymbol(DeviceTraceTools::device_pixels_h, &pixels_h, sizeof(int));
    if (err != cudaSuccess)
    {
        throw std::runtime_error("Error: failed to copy camera pixel height to device.");
    }
    // Allocate memory for the camera pixel array.
    free(host_camera_pixel_array);
    cudaFree(DeviceTraceTools::device_camera_pixel_array);
    host_camera_pixel_array = (unsigned char*)malloc(3*sizeof(unsigned char)*pixels_w*pixels_h);
    if (host_camera_pixel_array == nullptr)
    {
        throw std::runtime_error("Error: failed to allocate memory for camera pixel array on host.");
    }
    err = cudaMalloc((void **)&DeviceTraceTools::device_camera_pixel_array, 3*sizeof(unsigned char)*pixels_w*pixels_h);
    if (err != cudaSuccess)
    {
        throw std::runtime_error("Error: failed to allocate memory for camera pixel array on device.");
    }
}

// Set a new FoV in degrees and transfer the corresponding conversion factor to the device.
void Scene::setCameraFoV(float new_fov_width)
{
    fov_width = new_fov_width;
    fov_width_rad = fov_width * (pi_host/180.f);
    float conversion_factor = fov_width_rad / pixels_w;
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
__device__ void DeviceTraceTools::calculateStartVelocity(float pixel_x, float pixel_y, float photon_v[4], float metric[4][4])
{
    float phi { (pixel_x - 0.5f*device_pixels_w) * device_fov_conversion_factor };
    float theta { (pixel_y - 0.5f*device_pixels_h) * device_fov_conversion_factor + 0.5f*pi_device };
    // Convert to Cartesian coordinates.
    float unrotated_v[4];
    unrotated_v[0] = 0.;
    unrotated_v[1] = sin(theta)*cos(phi);
    unrotated_v[2] = sin(theta)*sin(phi);
    unrotated_v[3] = cos(theta);
    // Rotate to align with the camera orientation.
    rotateVecByQuat(unrotated_v, device_camera_quat, photon_v);
    // Set the t-component to make the velocity null.
    makeVNull(photon_v, metric);
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

__device__ void DeviceTraceTools::getChristoffelSymbols(float x_func[4], float metric_func[4][4], float c_symbols_func[4][4][4], float metric_derivs[4][4][4])
{
    // Assumed default step in each coordinate.
    // TODO: How do you define this adaptively to not break near areas of extreme distortion?
    // For now, just set it to a small number.
    const float step { 1e-4 };
    const float inverse_step { 1./step };

    // Simple Euler forward-difference derivatives of the metric along each component.
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
                metric_derivs[alpha][mu][nu] = (metric_temp[mu][nu] - metric_func[mu][nu])*inverse_step;
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
                #pragma unroll
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

// Makes the L2 norm of the velocity 1 for the sake of maintaining a roughly consistent affine parameter.
// This does turn it into a "unit" velocity!
__device__ void DeviceTraceTools::normaliseV(float v_func[4])
{
    float inv_norm { rnorm4df(v_func[0], v_func[1], v_func[2], v_func[3]) };
    v_func[0] *= inv_norm;
    v_func[1] *= inv_norm;
    v_func[2] *= inv_norm;
    v_func[3] *= inv_norm;
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
            #pragma unroll
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
        #pragma unroll
        for (int k { 0 }; k < 4; k++)
        {
            metric_inverse[i][k] *= multiplier;
        }
    }
}

// Crude way of testing how distorted the metric is from the Minkowski metric without resorting to the Riemann tensor.
// Used for adaptive step size. This only works in (t, x, y, z) coordinates.
__device__ float DeviceTraceTools::calculateParameterStep(float metric[4][4])
{
    // "Normalise" against things that scale the whole metric but introduce no curvature.
    float scale_factor { 0. };
    #pragma unroll
    for (int i { 0 }; i < 4; i++)
    {
        scale_factor += fabs(metric[i][i]);
    }
    scale_factor *= 0.25;
    scale_factor = 1./scale_factor;

    // Subtract the Minkowski metric from the scaled metric and add up all the absolute components.
    float deviation { 0. };
    deviation += fabs(metric[0][0]*scale_factor + 1.);
    #pragma unroll
    for (int i { 1 }; i < 4; i++)
    {
        // Diagonal components.
        deviation += fabs(metric[i][i]*scale_factor - 1.);
    }
    for (int i { 0 }; i < 3; i++)
    {
        for (int j { i+1 }; j < 4; j++)
        {
            // Off-diagonal components.
            deviation += 2.*fabs(metric[i][j]*scale_factor);
        }
    }

    if (deviation == 0)
    {
        // Metric is flat; set to the maximum step size.
        return 5.;
    }
    else
    {
        float dl;
        // This is designed to give reasonable stability for 1 or 2 orbits on the photon sphere of a Schwarzschild black hole of radius 1.
        dl = 1e-1 * (8./(deviation*deviation));
        if (dl > 5.)
        {
            // Too large; set to max parameter step.
            return 5.;
        }
        else
        {
            return dl;
        }
    }
}

// Advances a photon/pixel with an adaptive timestep using RK4.
__device__ void DeviceTraceTools::advance(float x[4], float v[4], float metric[4][4], float c_symbols[4][4][4], float metric_derivs[4][4][4])
{
    // Calculate parameter step.
    float dl { calculateParameterStep(metric) };
    // float dl { 0.1 };
    float mult_factor { dl/6.f };

    // Currently advances with RK4.
    // WARNING: Potential register spilling here; this requires a lot of memory.
    float x_step[4];
    float v_step[4];
    float x_temp[4];
    float v_temp[4];
    float k_n_minus_1_x[4];
    float k_n_x[4];
    float k_n_minus_1_v[4];
    float k_n_v[4];

    // Calculate k_1.
    getChristoffelSymbols(x, metric, c_symbols, metric_derivs);
    for (int i { 0 }; i < 4; i++)
    {
        k_n_x[i] = v[i];
        k_n_v[i] = 0;
        for (int j { 0 }; j < 4; j++)
        {
            #pragma unroll
            for (int k { 0 }; k < 4; k++)
            {
                k_n_v[i] -= c_symbols[i][j][k]*v[j]*v[k];
            }
        }
        x_step[i] = k_n_x[i];
        v_step[i] = k_n_v[i];
    }

    // Calculate k_2 and k_3.
    for (int u { 0 }; u < 2; u++)
    {
        #pragma unroll
        for (int i { 0 }; i < 4; i++)
        {
            k_n_minus_1_x[i] = k_n_x[i];
            k_n_minus_1_v[i] = k_n_v[i];
            x_temp[i] = x[i] + 0.5*dl*k_n_minus_1_x[i];
            v_temp[i] = v[i] + 0.5*dl*k_n_minus_1_v[i];
        }
        // Overwrite metric to avoid allocating another 16 floats.
        getMetricTensor(x_temp, metric);
        getChristoffelSymbols(x_temp, metric, c_symbols, metric_derivs);
        for (int i { 0 }; i < 4; i++)
        {
            k_n_x[i] = v[i] + 0.5*dl*k_n_minus_1_v[i];
            k_n_v[i] = 0;
            for (int j { 0 }; j < 4; j++)
            {
                #pragma unroll
                for (int k { 0 }; k < 4; k++)
                {
                    k_n_v[i] -= c_symbols[i][j][k]*v_temp[j]*v_temp[k];
                }
            }
            x_step[i] += 2.*k_n_x[i];
            v_step[i] += 2.*k_n_v[i];
        }
    }

    // Calculate k_4.
    #pragma unroll
    for (int i { 0 }; i < 4; i++)
    {
        k_n_minus_1_x[i] = k_n_x[i];
        k_n_minus_1_v[i] = k_n_v[i];
        x_temp[i] = x[i] + dl*k_n_minus_1_x[i];
        v_temp[i] = v[i] + dl*k_n_minus_1_v[i];
    }
    getMetricTensor(x_temp, metric);
    getChristoffelSymbols(x_temp, metric, c_symbols, metric_derivs);
    for (int i { 0 }; i < 4; i++)
    {
        k_n_x[i] = v[i] + dl*k_n_minus_1_v[i];
        k_n_v[i] = 0.;
        for (int j { 0 }; j < 4; j++)
        {
            #pragma unroll
            for (int k { 0 }; k < 4; k++)
            {
                k_n_v[i] -= c_symbols[i][j][k]*v_temp[j]*v_temp[k];
            }
        }
        x_step[i] += k_n_x[i];
        v_step[i] += k_n_v[i];

        // Advance x and v.
        x[i] += mult_factor*x_step[i];
        v[i] += mult_factor*v_step[i];
    }
    // Update metric to get it ready for the next step.
    getMetricTensor(x, metric);
}

// Gets a pointer to the RGB pixel from the sky map at pixel (x, y), where (0, 0) is the top-left pixel.
__device__ void DeviceTraceTools::readPixelFromSkyMap(unsigned char *pixel, unsigned char *device_sky_map, int &x, int &y, int &sky_pixels_w, int &byte_depth)
{
    pixel = &device_sky_map[(y*sky_pixels_w + x)*byte_depth];
}

// Run the actual raytracing loop. All the appropriate variables need to be assigned and defined before this can work (without undefined behaviour).
// TODO: Currently only defined for a camera outside the photon sphere of the Schwarzschild metric. Make this work for general metrics
// (i.e. some sort of event horizon-detector to terminate a photon?).
__global__ void traceImage(unsigned char *device_sky_map, unsigned char *device_camera_pixel_array)
{
    // This is currently intended for 8x4 thread blocks.
    // TODO: For now, the Christoffel symbols are in shared memory and will use 8 KiB. Test later
    // whether they can be moved to registers without spilling for a significant speed boost.
    // Same thing for the metric derivative components.
    __shared__ float c_symbols[8][4][4][4][4];
    __shared__ float metric_derivs[8][4][4][4][4];

    int pixel_x = blockIdx.x*blockDim.x + threadIdx.x;
    int pixel_y = blockIdx.y*blockDim.y + threadIdx.y;
    int pixel_index = 3*(pixel_y*DeviceTraceTools::device_pixels_w + pixel_x);
    bool consumed = false;

    if (pixel_x < DeviceTraceTools::device_pixels_w && pixel_y < DeviceTraceTools::device_pixels_h)
    {
        float x[4];
        float v[4];
        float metric[4][4];
        #pragma unroll
        for (int i { 0 }; i < 4; i++)
        {
            x[i] = DeviceTraceTools::device_camera_coords[i];
        }

        DeviceTraceTools::getMetricTensor(x, metric);
        DeviceTraceTools::calculateStartVelocity(pixel_x, pixel_y, v, metric);
        DeviceTraceTools::normaliseV(v);

        // Set to true if the photon enters the photon sphere.
        float sky_dist_squared = DeviceTraceTools::device_sky_map_distance_squared;
        float dist_squared = x[1]*x[1] + x[2]*x[2] + x[3]*x[3];
        while (dist_squared < sky_dist_squared)
        {
            if (dist_squared < 2.25)
            {
                // Entered the photon radius if true; terminate the photon.
                consumed = true;
                break;
            }
            // Otherwise, advance the simulation with RK4.
            DeviceTraceTools::advance(x, v, metric, &c_symbols[threadIdx.x][threadIdx.y][0], &metric_derivs[threadIdx.x][threadIdx.y][0]);
            dist_squared = x[1]*x[1] + x[2]*x[2] + x[3]*x[3];
        }
    }

    if (consumed == true)
    {
        // Entered the photon sphere; set to black.
        device_camera_pixel_array[pixel_index] = 0;
        device_camera_pixel_array[pixel_index+1] = 0;
        device_camera_pixel_array[pixel_index+2] = 0;
    }
    else
    {
        device_camera_pixel_array[pixel_index] = 150;
        device_camera_pixel_array[pixel_index+1] = 150;
        device_camera_pixel_array[pixel_index+2] = 150;
    }
}

// Calculates the Hamilton (quaternionic) product of u with v.
__device__ void hamiltonProduct(float u[4], float v[4], float result[4])
{
    result[0] = u[0]*v[0] - (u[1]*v[1] + u[2]*v[2] + u[3]*v[3]);
    // Cross product of the vector components of u and v is needed.
    float cross[3];
    cross[0] = u[2]*v[3] - u[3]*v[2];
    cross[1] = u[3]*v[1] - u[1]*v[3];
    cross[2] = u[1]*v[2] - u[2]*v[1];
    for (int i { 1 }; i < 4; i++)
    {
        result[i] = u[0]*v[i] + v[0]*u[i] + cross[i-1];
    }
}

// Rotates a 3D cartesian vector, vec (given as a quaternion with no real part), by the given quaternion, rotation_quat.
// result will be the rotated vector represented as a quaternion with no real part.
__device__ void rotateVecByQuat(float vec[4], float rotation_quat[4], float result[4])
{
    // Assume that rotation_quat is normalised.
    float rotation_quat_inverse[4];
    rotation_quat_inverse[0] = rotation_quat[0];
    rotation_quat_inverse[1] = -rotation_quat[1];
    rotation_quat_inverse[2] = -rotation_quat[2];
    rotation_quat_inverse[3] = -rotation_quat[3];
    float intermediate_result[4];
    hamiltonProduct(vec, rotation_quat_inverse, intermediate_result);
    hamiltonProduct(rotation_quat, intermediate_result, result);
}
