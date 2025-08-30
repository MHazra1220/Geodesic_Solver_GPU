#include "scene.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <stdexcept>

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
        phi_interval = (2.*pi_host) / sky_pixels_w;
        theta_interval = pi_host / sky_pixels_h;

        // Copy pixel map to device memory.
        cudaError_t err { cudaSuccess };
        size_t map_size { sizeof(unsigned char)*num_photons*byte_depth };
        err = cudaMalloc((void **)&device_sky_map, map_size);
        if (err != cudaSuccess)
        {
            throw std::runtime_error("Error: failed to allocate sky map memory on device.");
        }
        err = cudaMemcpy(device_sky_map, host_sky_map, map_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            throw std::runtime_error("Error: failed to copy sky map to device.");
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
    cudaFree(device_sky_map);
}

// Gets a pointer to the RGB pixel from the sky map at pixel (x, y), where (0, 0) is the top-left pixel.
__device__ void Scene::readPixelFromSkyMap(unsigned char *pixel, int &x, int &y)
{
    pixel = &device_sky_map[(y*sky_pixels_w + x)*byte_depth];
}

// Currently defined to return the Schwarzschild metric with a Schwarzschild radius of 1.
// Gets the metric at x_func and overwrites it into metric_func.
__device__ void Scene::getMetricTensor(float x_func[4], float metric_func[4][4])
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

__device__ void Scene::getChristoffelSymbols(float x_func[4], float metric_func[4][4], float c_symbols_func[4][4][4])
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
__device__ void Scene::makeVNull(float v_func[4], float metric_func[4][4])
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
__device__ void invertMetric(float metric_func[4][4], float metric_inverse[4][4])
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

    // Last task is to normalise the rows of metric_inverse by whatever is left in the diagonals of metric_func.
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
