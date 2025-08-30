#include "quaternion.h"

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
