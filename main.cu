#include "world.h"
#include "particle.h"
#include "camera.h"
#include <iostream>
#include <Eigen/Dense>
#include "cuda.h"
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

using namespace Eigen;

// First ever GPU kernel test!
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements) {
        C[i] = A[i] + B[i] + 0.0f;
    }
}

int main()
{
    // World world_test;
    // char file[] { "/media/mh2001/SSD2/Programming/General Relativity/Geodesic_Solver/sky_box_samples/full_milky_way.jpg" };
    // char* ref { file };
    // world_test.importSkyMap(ref);
    // Camera camera_test;
    // camera_test.setWidthHeight(2560, 1440);
    // camera_test.setFov(90.);
    // camera_test.setCameraLocation(Vector4d { 0., 20., 0., 0. });
    // // Point along +x (No change in orientation required).
    // camera_test.setCameraOrientation(Vector4d { 0., 0., 0., 1. });
    // camera_test.traceImage(world_test);
    // char output_image[] { "/media/mh2001/SSD2/Programming/General Relativity/Geodesic_Solver/output_images/test_pointer.jpg" };
    // char* output_image_ref { output_image };
    // camera_test.writeCameraImage(output_image_ref);


    // CUDA TEST HERE!

    // For checking return values of CUDA calls.
    cudaError_t err { cudaSuccess };
    int num_elements { 200000 };
    // Amount of memory to allocate.
    size_t size {num_elements * sizeof(float)};


    // Allocate memory on host.
    float host_A[num_elements];
    float host_B[num_elements];
    float host_C[num_elements];

    // Verify success.
    if (host_A == NULL || host_B == NULL || host_C == NULL){
        std::cout << "Failed to allocate host vectors." << "\n";
        exit(EXIT_FAILURE);
    }

    // Initialise host input vectors.
    for (int i = 0; i < num_elements; i++)
    {
        float element { static_cast<float>(i) };
        host_A[i] = element;
        host_B[i] = element;
    }

    // Allocate device memory.
    float *device_A { nullptr };
    float *device_B { nullptr };
    float *device_C { nullptr };
    err = cudaMalloc((void **)&device_A, size);
    err = cudaMalloc((void **)&device_B, size);
    err = cudaMalloc((void **)&device_C, size);

    // Copy host vectors to device.
    // Note that host_A undergoes array decay to a pointer here; there is no need for malloc() nonsense!
    err = cudaMemcpy(device_A, host_A, size, cudaMemcpyHostToDevice);
    err = cudaMemcpy(device_B, host_B, size, cudaMemcpyHostToDevice);

    // Now for the magic: launch the kernel!
    int threadsPerBlock { 512 };
    int blocksPerGrid { (num_elements + threadsPerBlock - 1) / threadsPerBlock };
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(device_A, device_B, device_C, num_elements);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        std::cout << "Failed to launch vectorAdd() kernel." << "\n";
    }

    // Copy back to host.
    err = cudaMemcpy(host_C, device_C, size, cudaMemcpyDeviceToHost);

    // Free device memory.
    err = cudaFree(device_A);
    err = cudaFree(device_B);
    err = cudaFree(device_C);

    // Verify result.
    for (int i { 0 }; i < num_elements; i++)
    {
        std::cout << host_C[i] << "\n";
    }

    return 0;
}
