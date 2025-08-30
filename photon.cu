#include "photon.h"

__device__ void Photon::setX(float new_x[4])
{
    for (int i { 0 }; i < 4; i++)
    {
        x[i] = new_x[i];
    }
}

__device__ void Photon::setV(float new_v[4])
{
    for (int i { 0 }; i < 4; i++)
    {
        v[i] = new_v[i];
    }
}

/*
 * Modifies the t-component of the 4-velocity to make the vector null.
 * This requires solving a quadratic equation for the t-component; assume
 * that you should take the more positive component because g_00 is
 * probably negative. Note that the more negative solution is needed
 * because the raytracer evolves photons "backwards".
 */
__device__ void Photon::makeVNull()
{
    float a { metric[0][0] };
    float b { 0. };
    // c is the scalar product of the spatial metric with the spatial velocity components.
    float c { 0. };
    float contraction;
    for (int i { 1 }; i < 4; i++)
    {
        b += metric[0][i] * v[i];
        contraction = 0.;
        for (int j { 1 }; j < 4; j++)
        {
            contraction += metric[i][j] * v[j];
        }
        c += v[i]*contraction;
    }
    b *= 2.;

    // Take the more negative solution (note a is usually negative, so this means the positive root).
    v[0] = (-b + sqrt(b*b - 4.*a*c)) / (2.*a);

    normaliseV();
}

// Makes the L2-norm of the 4-velocity 1 for the sake
// of maintaining a roughly consistent affine parameterisation.
// WARNING: this is not a "unit" velocity!
__device__ void Photon::normaliseV()
{
    float norm { norm4df(v[0], v[1], v[2], v[3]) };
    for (int i { 0 }; i < 4; i++)
    {
        v[i] /= norm;
    }
}

// Currently defined to return the Schwarzschild metric with a Schwarzschild radius of 1.
// Gets the metric at x_temp and writes it into metric_temp.
__device__ void Photon::getMetricTensor(float x_temp[4], float metric_temp[4][4])
{
    const float r_s { 1. };
    float r_squared { norm3df(x_temp[1], x_temp[2], x_temp[3]) };
    float r { sqrt(r_squared) };
    float mult_factor { r_s / (r_squared*(r-r_s)) };
    for (int mu { 1 }; mu < 4; mu++)
    {
        metric_temp[0][mu] = 0.;
        metric_temp[mu][0] = 0.;
        for (int nu { mu }; nu < 4; nu++)
        {
            metric_temp[mu][nu] = mult_factor*x_temp[mu]*x_temp[nu];
            metric_temp[nu][mu] = metric_temp[mu][nu];
        }
    }
    metric_temp[0][0] = -1. + r_s/r;
    metric_temp[1][1] += 1.;
    metric_temp[2][2] += 1.;
    metric_temp[3][3] += 1.;
}

__global__ void normalisePhotonVelocities(Photon photons[], int width, int height)
{
    int photon_num = blockDim.x * blockIdx.x + threadIdx.x;
    if (photon_num < width*height)
    {
        Photon &photon = photons[photon_num];
        photon.x[0] = 0.;
        photon.x[1] = -10.;
        photon.x[2] = 0.;
        photon.x[3] = 0.;
        photon.v[1] = 1.;
        photon.v[2] = 1.;
        photon.v[3] = 1.;
        // This will update photon.metric due to array decay to a pointer.
        photon.getMetricTensor(photon.x, photon.metric);
        photon.makeVNull();
    }
}
