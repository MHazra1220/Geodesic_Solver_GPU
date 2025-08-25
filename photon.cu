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
    for (int i { 1 }; i < 4; i++)
    {
        b += metric[0][i] * v[i];
        float contraction { 0. };
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

// Makes the L2 (Euclidean) norm of the 4-velocity 1 for the sake
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

__device__ void Photon::setMetric()
{
    metric[0][0] = -1.;
    for (int i { 1 }; i < 4; i++)
    {
        for (int j { i }; j < 4; j++)
        {
            if (i == j)
            {
                metric[i][j] = 1.;
            }
            else
            {
                metric[i][j] = 0.;
                metric[j][i] = 0.;
            }
        }
    }
}
