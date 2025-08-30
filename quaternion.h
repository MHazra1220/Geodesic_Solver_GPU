#ifndef QUATERNION
#define QUATERNION

// Some quaternion arithmetic functions. Used when setting the starting velocities of photons.

__device__ void hamiltonProduct(float u[4], float v[4], float result[4]);
__device__ void quaternionRotate(float vec[4], float rotation_quat[4], float result[4]);

#endif
