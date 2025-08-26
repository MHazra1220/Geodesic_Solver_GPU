#ifndef PARTICLE
#define PARTICLE

// This should be allocated on host and moved to device as a large array of photons (one per pixel).
// Calculations should be done on the device.
class Photon
{
    public:
        // Maximum parameter step that the solver will never exceed.
        const float max_parameter_step { 5. };

        // Spacetime coordinates and 4-velocity.
        float x[4];
        float v[4];
        // Metric tensor at the coordinates (assumed to be symmetric/torsion-free).
        float metric[4][4];
        // Christoffel symbols at the coordinates (the first index is the upper index of the symbols).
        float christoffel_symbols[4][4][4];

        __device__ void setX(float new_x[4]);
        __device__ void setV(float new_v[4]);
        __device__ void makeVNull();
        __device__ void normaliseV();
        __device__ void setMetric();
};

#endif
